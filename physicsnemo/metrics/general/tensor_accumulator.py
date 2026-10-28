# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable detached tensor accumulation with user-supplied statistical rules."""

from typing import Callable, Protocol

import torch
from torch.distributed.tensor import Replicate

TensorState = dict[str, torch.Tensor | int]


class StatisticRule(Protocol):
    """Statistical math supplied to :class:`TensorAccumulator`.

    Implement three pure operations: summarize a batch, merge independent states,
    and compute a result. State is a nonempty dictionary of floating-point tensors
    and Python integers. Tensor fields may have different shapes, but each field's
    shape, dtype, device and distributed layout must remain stable across updates.

    Hooks must not mutate their arguments or retain input references. Use the
    supplied ``reduce_sum`` for reductions before nonlinear arithmetic: it completes
    pending distributed sums while preserving unreduced shards. ``dim=None`` means
    one observation; an integer names the observation axis. Empty batches must
    produce a correctly shaped state without dividing by zero.

    Rules must compare equal only when their statistical configuration matches.
    A frozen dataclass is sufficient for stateless rules or scalar configuration.
    Keep that configuration unchanged throughout an accumulator's lifetime.
    """

    def summarize(
        self, values: torch.Tensor, *, dim: int | None, reduce_sum: Callable, **inputs
    ) -> TensorState:
        """Summarize detached observations and optional detached tensor inputs."""
        ...

    def merge(self, left: TensorState, right: TensorState) -> TensorState:
        """Return a state for disjoint observations without modifying either input."""
        ...

    def result(
        self, state: TensorState, *, reduce_sum: Callable, **options
    ) -> torch.Tensor:
        """Read a result without modifying statistical state."""
        ...


def _complete(value):
    """Finish pending reductions without gathering unreduced spatial shards."""
    placements = getattr(value, "placements", ())
    if any(p.is_partial() for p in placements):
        return value.redistribute(
            placements=[Replicate() if p.is_partial() else p for p in placements]
        )
    return value


def _check_compatible(left, right):
    if (
        type(left) is not type(right)
        or left.shape != right.shape
        or left.device != right.device
        or left.dtype != right.dtype
    ):
        raise ValueError(
            "Statistics require matching tensor type, output shape, device, and dtype"
        )
    if getattr(left, "placements", ()) != getattr(right, "placements", ()):
        raise ValueError("Statistics require aligned distributed placements")
    if getattr(left, "device_mesh", None) != getattr(right, "device_mesh", None):
        raise ValueError("Statistics require the same device mesh")
    # ShardTensor supports non-chunked partitions. Equal global shapes and
    # placements alone do not establish correspondence of output elements.
    left_shapes = getattr(getattr(left, "_spec", None), "sharding_shapes", None)
    right_shapes = getattr(getattr(right, "_spec", None), "sharding_shapes", None)
    if left_shapes is not None and left_shapes() != right_shapes():
        raise ValueError("Statistics require aligned shard boundaries")


def _shares_storage(left, right):
    """Check local backing storage, including offset views and distributed tensors."""
    if getattr(left, "device_mesh", None) is not None:
        left = left.to_local()
    if getattr(right, "device_mesh", None) is not None:
        right = right.to_local()
    if left.device != right.device:
        return False
    try:
        a, b = left.untyped_storage(), right.untyped_storage()
        # Storage identity also detects aliases of zero-sized allocations.
        if a == b:
            return True
        # Address ranges also catch distinct storage objects wrapping the same
        # external buffer. Tensor.data_ptr() alone misses offset views.
        return (
            a.nbytes() > 0
            and b.nbytes() > 0
            and a.data_ptr() < b.data_ptr() + b.nbytes()
            and b.data_ptr() < a.data_ptr() + a.nbytes()
        )
    except (NotImplementedError, RuntimeError):
        # Preserve ownership for tensor layouts without inspectable storage.
        return True


class TensorAccumulator:
    """Manage tensor state while a rule defines the statistic's mathematics.

    Parameters
    ----------
    rule : StatisticRule
        Rule implementing ``summarize``, ``merge`` and ``result``. See the rule
        protocol for purity, state and equality requirements.
    dtype : torch.dtype, optional
        Explicit accumulation precision: float32 or float64. Otherwise infer at
        least float32 from the first successful update, including auxiliary inputs.

    Notes
    -----
    Ordinary tensors remain local. Sharded tensor operations follow their mesh
    placements; all participating ranks must follow the same update schedule.
    ``combine`` joins disjoint observations of the same output elements, not
    spatial partitions or replicated copies of the same observations.
    This class does not infer a distributed algorithm from arbitrary rule code.
    """

    def __init__(self, rule: StatisticRule, *, dtype: torch.dtype | None = None):
        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError("Accumulation dtype must be float32 or float64")
        self.rule = rule
        self.dtype = dtype
        self._explicit_dtype = dtype is not None
        self.count = 0
        self._state = None
        self._template = None

    @staticmethod
    def _sum(values, dim, *, keepdim=False):
        """Complete a reduction before a rule performs nonlinear arithmetic."""
        if dim is None:
            return values.clone()
        reduced = _complete(values.sum(dim=dim, keepdim=True))
        # Singleton sum preserves uneven ShardTensor metadata; squeeze does not.
        return reduced if keepdim else reduced.sum(dim=dim)

    def _prepare(self, values, inputs):
        tensors = [values, *inputs.values()]
        if any(
            not isinstance(value, torch.Tensor) or not value.is_floating_point()
            for value in tensors
        ):
            raise TypeError("Expected a real floating-point tensor for every input")
        inferred = (
            torch.float64
            if any(x.dtype == torch.float64 for x in tensors)
            else torch.float32
        )
        if (
            not self._explicit_dtype
            and self.dtype == torch.float32
            and inferred == torch.float64
        ):
            raise ValueError(
                "Accumulator dtype is torch.float32; use dtype=torch.float64 to retain float64 inputs"
            )
        dtype = self.dtype or inferred
        prepared = [_complete(value.detach().to(dtype=dtype)) for value in tensors]
        for auxiliary in prepared[1:]:
            _check_compatible(prepared[0], auxiliary)
        return prepared[0], dict(zip(inputs, prepared[1:])), dtype

    @staticmethod
    def _copy(state, *, zero=False):
        return {
            key: (torch.zeros_like(value) if zero else value.clone())
            if isinstance(value, torch.Tensor)
            else (0 if zero else value)
            for key, value in state.items()
        }

    @staticmethod
    def _validate(state, reference, dtype):
        if (
            not isinstance(state, dict)
            or not state
            or not all(isinstance(k, str) for k in state)
        ):
            raise TypeError("Rule state must be a nonempty dictionary with string keys")
        if reference is not None and state.keys() != reference.keys():
            raise ValueError("Rule state fields must remain unchanged")
        has_tensor = False
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                has_tensor = True
                if value.dtype != dtype or value.requires_grad:
                    raise ValueError(
                        "Rule state tensors must be detached and use the accumulation dtype"
                    )
                if any(p.is_partial() for p in getattr(value, "placements", ())):
                    raise ValueError(
                        "Rule must complete pending reductions with reduce_sum"
                    )
                if reference is not None:
                    if not isinstance(reference[key], torch.Tensor):
                        raise TypeError("Rule state field types must remain unchanged")
                    _check_compatible(reference[key], value)
            elif type(value) is not int:
                raise TypeError("Rule state fields must be tensors or Python integers")
            elif reference is not None and type(reference[key]) is not int:
                raise TypeError("Rule state field types must remain unchanged")
        if not has_tensor:
            raise TypeError("Rule state must contain at least one tensor")

    def update(self, values: torch.Tensor, dim: int | None = 0, **inputs) -> None:
        """Accumulate observations without retaining their computation graphs.

        Parameters
        ----------
        values : torch.Tensor
            Real floating-point observations. Ordinary tensors remain local;
            distributed tensors follow their mesh placements.
        dim : int or None, optional
            Observation axis. ``None`` treats the entire tensor as one observation.
        **inputs : torch.Tensor
            Auxiliary inputs interpreted by the rule, such as weights. They must
            match ``values`` in shape, device and distributed layout. Broadcast
            them explicitly before updating. All inputs are detached and share
            the accumulator's precision policy.

        Returns
        -------
        None
            State is updated in place. Read the result with ``finalize``.

        Raises
        ------
        TypeError
            Inputs are not real floating-point tensors, ``dim`` has an invalid
            type, or the rule returns an invalid state structure.
        IndexError
            The observation axis is out of range.
        ValueError
            Precision, input layouts or state fields are incompatible, or the
            rule rejects the observations.

        Notes
        -----
        With a pure rule, failed validation leaves state and inferred dtype
        unchanged. Participating mesh ranks must follow the same update schedule;
        empty local shards still participate in distributed reductions.
        """
        values, inputs, dtype = self._prepare(values, inputs)
        if dim is not None:
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise TypeError("dim must be an integer observation axis or None")
            if not -values.ndim <= dim < values.ndim:
                raise IndexError("Observation axis is out of range")
            dim %= values.ndim
        n = 1 if dim is None else values.shape[dim]
        batch = self.rule.summarize(values, dim=dim, reduce_sum=self._sum, **inputs)
        reference = self._state if self._state is not None else self._template
        self._validate(batch, reference, dtype)
        if n:
            candidate = (
                self.rule.merge(self._state, batch) if self.count else self._copy(batch)
            )
            self._validate(candidate, batch, dtype)
            self._state = candidate
            self._template = None
            self.count += n
        elif reference is None:
            self._template = self._copy(batch, zero=True)
        self.dtype = dtype

    def combine(self, other: "TensorAccumulator") -> "TensorAccumulator":
        """Merge independent observations with matching rules, dtype and layouts.

        Parameters
        ----------
        other : TensorAccumulator
            Accumulator containing disjoint observations of the same output
            elements. Its rule type, statistical configuration, settled dtype
            and state layouts must match. The other accumulator is unchanged.

        Returns
        -------
        TensorAccumulator
            This accumulator with the observations incorporated in place.

        Raises
        ------
        ValueError
            ``other`` is this accumulator, or rules, precision or layouts differ.
        TypeError
            The rule returns an invalid state structure.

        Notes
        -----
        Rule equality must include configuration affecting sufficient statistics.
        Callers must ensure the observations are independent: this method does
        not detect duplicated observations in distinct accumulators. It does not
        assemble spatial partitions or combine replicated copies of observations.
        """
        if other is self:
            raise ValueError("An accumulator cannot be combined with itself")
        if (
            not isinstance(other, TensorAccumulator)
            or type(self.rule) is not type(other.rule)
            or self.rule != other.rule
            or (
                self.dtype is not None
                and other.dtype is not None
                and self.dtype != other.dtype
            )
        ):
            raise ValueError(
                "combine requires matching rules and settled accumulation dtype"
            )
        incoming = other._state if other._state is not None else other._template
        if incoming is None:
            self.dtype = self.dtype or other.dtype
            return self
        reference = self._state if self._state is not None else self._template
        dtype = self.dtype or other.dtype
        self._validate(incoming, reference, dtype)
        if other.count:
            candidate = (
                self.rule.merge(self._state, incoming)
                if self.count
                else self._copy(incoming)
            )
            self._validate(candidate, incoming, dtype)
            self._state = candidate
            self._template = None
            self.count += other.count
        elif reference is None:
            self._template = self._copy(incoming, zero=True)
        self.dtype = dtype
        return self

    def finalize(self, **options) -> torch.Tensor:
        """Read a result without exposing mutable accumulator state.

        Parameters
        ----------
        **options : object
            Keyword arguments passed to the rule's ``result`` method, such as
            ``correction`` or ``statistic`` for ``VarianceRule``.

        Returns
        -------
        torch.Tensor
            Result independent of stored state. A result sharing backing storage
            with any state tensor is cloned; a fresh result is returned directly.
            Distributed results retain their layout.

        Raises
        ------
        ValueError
            The accumulator is empty or the rule rejects the requested result.
        TypeError
            The rule returns a non-tensor result or receives unsupported options.

        Notes
        -----
        Aliasing checks inspect local storage without gathering shards or adding
        collectives. Layouts without inspectable storage are conservatively
        cloned. A rule may itself require collective participation during reads.
        """
        if not self.count:
            raise ValueError("Cannot read an empty accumulator")
        result = self.rule.result(self._state, reduce_sum=self._sum, **options)
        if not isinstance(result, torch.Tensor):
            raise TypeError("Rule result must be a tensor")
        if any(
            _shares_storage(result, value)
            for value in self._state.values()
            if isinstance(value, torch.Tensor)
        ):
            return result.clone()
        return result

    def reset(self) -> "TensorAccumulator":
        """Discard observations while retaining precision and each field's layout.

        Returns
        -------
        TensorAccumulator
            This empty accumulator. Its count is zero; its rule, dtype policy
            and settled state layouts are retained for subsequent updates.

        Notes
        -----
        This method takes no parameters. Reading after reset raises an error
        until new observations have been accumulated.
        """
        if self._state is not None:
            self._template = self._copy(self._state, zero=True)
        self._state = None
        self.count = 0
        return self
