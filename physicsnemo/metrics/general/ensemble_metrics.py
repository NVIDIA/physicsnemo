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

import warnings
from abc import ABC
from typing import List, Literal, Self, Tuple, Union

import torch
import torch.distributed as dist

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.metrics.general._moments import _check_correction, _StreamingMoments

Tensor = torch.Tensor


class StreamingStatistic(ABC):
    """Base interface for statistics accumulated across successive inputs.

    Subclasses maintain statistical state: ``__call__`` initializes or replaces
    it, ``update`` incorporates additional observations, and ``finalize`` computes
    a result from the accumulated state. Observations may represent ensemble
    members, time steps, or other samples, as defined by the concrete statistic.

    This base provides shape, device, and dtype configuration. Subclasses define
    their reduction axes, state representation, and distributed behavior;
    inheriting this interface does not implement accumulation or communication.

    Parameters
    ----------
    input_shape : Union[Tuple[int,...], List]
        Shape of input tensors without batched dimension.
    device : torch.device, optional
        Pytorch device model is on, by default 'cpu'
    dtype : torch.dtype, optional
        Standard dtype to initialize any tensor with
    """

    def __init__(
        self,
        input_shape: Union[Tuple[int, ...], List[int]],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_shape = list(input_shape)
        self.device = torch.device(device)
        self.dtype = dtype

    def _check_shape(self, inputs: Tensor) -> None:
        """
        Check input shapes for non-batched dimension.
        """
        if not all([i == s for (i, s) in zip(inputs.shape[1:], self.input_shape)]):
            raise ValueError(
                "Expected new input to have compatible shape with existing shapes but got"
                + str(inputs.shape)
                + "and"
                + str(self.input_shape)
                + "."
            )

    def __call__(self, *args):
        """
        Initialize statistical state from inputs, replacing any previous state.
        """
        raise NotImplementedError("Class member must implement a __call__ method.")

    def update(self, *args):
        """
        Incorporate additional observations into the statistical state.
        """
        raise NotImplementedError("Class member must implement an update method.")

    def finalize(self, *args):
        """
        Compute a result from the accumulated state.

        The concrete statistic defines the return value and any state changes.
        """
        raise NotImplementedError("Class member must implement a finalize method.")


class _TensorModeMixin:
    """Opt-in operations shared only by Mean and Variance."""

    # Pickles written before tensor mode have no instance-level _moments.
    _moments = None

    @property
    def count(self) -> int:
        """Number of observations in tensor mode, rather than updates.

        Returns
        -------
        int
            Accumulated observation count.

        Raises
        ------
        AttributeError
            In legacy mode, where callers use ``n`` instead.
        """
        if self._moments is None:
            raise AttributeError(
                "count is available in mode='tensor'; legacy callers use n"
            )
        return self._moments.count

    def combine(self, other: Self) -> Self:
        """Append an independent state's observations; leave the other state unchanged.

        Parameters
        ----------
        other : Self
            Statistic containing independent observations, left unchanged.

        Returns
        -------
        Self
            This accumulator, updated in place.

        Raises
        ------
        ValueError
            If modes, statistic types, accumulation dtypes or layouts differ.

        Notes
        -----
        Tensor mode only. Outputs must have identical shape and distributed layout.
        States must represent disjoint observations; replicated copies must not be
        combined again. This method does not gather spatial partitions.
        """
        if (
            type(self) is not type(other)
            or self._moments is None
            or other._moments is None
        ):
            raise ValueError("combine requires matching tensor-mode statistics")
        self._moments.combine(other._moments)
        self.dtype = self._moments.dtype
        return self

    def reset(self) -> Self:
        """Discard observations, retaining the settled dtype and output layout.

        Returns
        -------
        Self
            This empty accumulator.

        Raises
        ------
        ValueError
            If called in legacy mode.
        """
        if self._moments is None:
            raise ValueError("reset is available in mode='tensor'")
        self._moments.reset()
        return self

    def _init_tensor_mode(
        self, *, variance: bool, like: Tensor | None, kwargs: dict
    ) -> None:
        unknown = kwargs.keys() - {"dtype"}
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected tensor-mode keyword(s): {', '.join(sorted(unknown))}"
            )
        dtype: torch.dtype | None = kwargs.get("dtype")
        self._moments = _StreamingMoments(variance=variance, like=like, dtype=dtype)
        self.input_shape = None
        self.device = None
        self.dtype = self._moments.dtype


def _update_mean(
    old_sum: Tensor,
    old_n: Union[int, Tensor],
    inputs: Tensor,
    batch_dim: Union[int, None] = 0,
) -> Tuple[Tensor, Union[int, Tensor]]:
    """Updated mean sufficient statistics given new data

    This method updates a running sum and number of samples with new (possibly batched)
    inputs

    Parameters
    ----------
    old_sum : Tensor
        Current, or old, running sum
    old_n : Union[int, Tensor]
        Current, or old, number of samples
    input : Tensor
        New input to add to current/old sum. May be batched, in which case the batched
        dimension must be flagged by passing an int to batch_dim.
    batch_dim : Union[int, None], optional
        Whether the new inputs contain a batch of new inputs and what dimension they are
        stored along. Will reduce all elements if None, by default 0.

    Returns
    -------
    Tuple[Tensor, Union[int, Tensor]]
        Updated (rolling sum, number of samples)
    """
    if batch_dim is None:
        inputs = torch.unsqueeze(inputs, 0)
        batch_dim = 0

    new_sum = old_sum + torch.sum(inputs, dim=batch_dim)
    new_n = old_n + inputs.shape[batch_dim]

    return new_sum, new_n


class Mean(_TensorModeMixin, StreamingStatistic):
    """Accumulate a mean across successive batches of observations.

    Call the statistic with an initial batch, pass additional observations to
    ``update``, and read the mean with ``finalize``. Calling the statistic again
    replaces its accumulated observations. Tensor mode also accepts ``update``
    as the first operation.

    Legacy mode uses batches shaped ``(B, *input_shape)`` and combines incoming
    sums and counts across the default process group when PhysicsNeMo's
    distributed manager and that group are initialized. Tensor mode reduces
    selected observation dimensions and follows the input tensor's distributed
    layout, retaining the unreduced dimensions in the result.

    Parameters
    ----------
    input_shape : Union[Tuple, List]
        Shape of each observation in legacy mode, excluding the leading batch
        dimension. Omit in tensor mode.
    mode : {"legacy", "tensor"}, optional
        ``"legacy"`` preserves the existing constructor and implicit world-group
        reductions. ``"tensor"`` enables detached streaming accumulation: ordinary
        tensors are local and distributed tensor layouts determine communication.
        Tensor mode infers output shape/device from the first update or ``like``;
        omit ``input_shape`` and ``device`` in this mode.
    like : torch.Tensor, optional
        Tensor-mode output template, including any spatial layout. It does not
        add an observation. Not needed when the first update supplies the layout.
    dtype : torch.dtype, optional
        Tensor-mode accumulation dtype (float32 or float64). By default, float64
        inputs remain float64 and lower-precision inputs accumulate in float32.
        The dtype settles from ``like`` or the first update. Inferred float32
        rejects later float64 inputs; specify ``dtype`` to allow explicit casts.

    Notes
    -----
    In tensor mode, ``update(value, dim=0)`` appends a batch of observations;
    ``dim=None`` appends one complete field or observable. Each observation has
    equal weight. State shape and output layout must remain aligned. Empty
    updates contribute no observations. ``combine(other)`` appends a disjoint
    state's observations, and ``reset()`` discards accumulated observations.
    Reading/finalizing does not mutate the accumulated state. In tensor mode,
    ``input_shape`` and ``device`` are None; results carry shape, device and
    layout, and ``dtype`` records the settled accumulation precision.
    ``update`` returns None; use ``finalize`` to read the result.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...] | list[int] | None = None,
        *,
        mode: Literal["legacy", "tensor"] = "legacy",
        like: Tensor | None = None,
        **kwargs,
    ):
        self._moments = None
        if mode == "tensor":
            if input_shape is not None or "device" in kwargs:
                raise ValueError("Tensor mode infers shape/device from inputs or like")
            self._init_tensor_mode(variance=False, like=like, kwargs=kwargs)
            return
        if mode != "legacy" or like is not None:
            raise ValueError("Use mode='tensor' for tensor-aware streaming statistics")
        super().__init__(input_shape, **kwargs)
        self.sum = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)
        self.n = torch.zeros([1], dtype=torch.int32, device=self.device)

    def __call__(self, inputs: Tensor, dim: int | None = 0) -> Tensor | None:
        """Calculate an initial mean

        Parameters
        ----------
        inputs : Tensor
            Input data
        dim : int or None, optional
            Observation axis; None appends one entire field in tensor mode.

        Returns
        -------
        Tensor or None
            Mean value, or None for an empty tensor-mode batch.
        """
        if self._moments is not None:
            self.reset()
            self.update(inputs, dim=dim)
            return self.finalize() if self.count else None
        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )
        self.sum = torch.sum(inputs, dim=dim)
        self.n = torch.as_tensor([inputs.shape[dim]], device=self.device)
        # TODO(Dallas) Move distributed calls into finalize.

        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.n, op=dist.ReduceOp.SUM)

        return self.sum / self.n

    def update(self, inputs: Tensor, dim: int | None = 0) -> Tensor | None:
        """Update current mean and essential statistics with new data

        Parameters
        ----------
        inputs : Tensor
            Inputs tensor
        dim : int or None
            Observation axis. In tensor mode, None appends one entire field.

        Returns
        -------
        Tensor or None
            Current mean in legacy mode; None in tensor mode.

        Raises
        ------
        TypeError
            Tensor-mode input is not a real floating-point tensor.
        ValueError
            Tensor-mode input is incompatible with the settled dtype or layout.
        IndexError
            Tensor-mode observation axis is out of range.
        """
        if self._moments is not None:
            self._moments.update(inputs, dim)
            self.dtype = self._moments.dtype
            return None
        self._check_shape(inputs)
        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )

        # TODO(Dallas) Move distributed calls into finalize.
        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            # Collect local sums, n
            sums = torch.sum(inputs, dim=dim)
            n = torch.as_tensor([inputs.shape[dim]], device=self.device)

            # Reduce
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)

            # Update
            self.sum += sums
            self.n += n
        else:
            self.sum, self.n = _update_mean(self.sum, self.n, inputs, batch_dim=dim)
        return self.sum / self.n

    def finalize(
        self,
    ) -> Tensor:
        """Read the mean (also stored as ``mean`` in legacy mode)

        Returns
        -------
        Tensor
            Mean without changing tensor-mode state. Legacy mode also stores it.

        Raises
        ------
        ValueError
            Tensor-mode accumulator is empty.
        """
        if self._moments is not None:
            return self._moments.read_mean()
        self.mean = self.sum / self.n

        return self.mean


def _update_var(
    old_sum: Tensor,
    old_sum2: Tensor,
    old_n: Union[int, Tensor],
    inputs: Tensor,
    batch_dim: Union[int, None] = 0,
) -> Tuple[Tensor, Tensor, Union[int, Tensor]]:
    """Updated variance sufficient statistics given new data

    This method updates a running running sum, squared sum, and number of samples with
    new (possibly batched) inputs

    Parameters
    ----------
    old_sum : Tensor
        Current, or old, running sum
    old_sum2 : Tensor
        Current, or old, running squared sum
    old_n : Union[int, Tensor]
        Current, or old, number of samples
    inputs : Tensor
        New input to add to current/old sum. May be batched, in which case the batched
        dimension must be flagged by passing an int to batch_dim.
    batch_dim : Union[int, None], optional
        Whether the new inputs contain a batch of new inputs and what dimension they are
        stored along. Will reduce all elements if None, by default 0.

    Returns
    -------
    Tuple[Tensor, Tensor, Union[int, Tensor]]
        Updated (rolling sum, rolling squared sum, number of samples)

    Note
    ----
    See "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
    by Chan et al.
    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    for details.
    """

    if batch_dim is None:
        inputs = torch.unsqueeze(inputs, 0)
        batch_dim = 0

    temp_n = inputs.shape[batch_dim]
    temp_sum = torch.sum(inputs, dim=batch_dim)
    temp_sum2 = torch.sum((inputs - temp_sum / temp_n) ** 2, dim=batch_dim)

    delta = old_sum * temp_n / old_n - temp_sum

    new_sum = old_sum + temp_sum
    new_sum2 = old_sum2 + temp_sum2
    new_sum2 += old_n / temp_n / (old_n + temp_n) * delta**2
    new_n = old_n + temp_n

    return new_sum, new_sum2, new_n


class Variance(_TensorModeMixin, StreamingStatistic):
    """Accumulate a variance across successive batches of observations.

    Call the statistic with an initial batch, then append observations with
    ``update``. ``finalize`` returns the variance, or its square root with
    ``std=True``. Calling the statistic again replaces its accumulated
    observations. Tensor mode also accepts ``update`` as the first operation.

    Legacy mode uses batches shaped ``(B, *input_shape)`` and combines incoming
    statistics across the default process group when PhysicsNeMo's distributed
    manager and that group are initialized. It computes the unbiased variance.
    Tensor mode reduces selected observation dimensions, follows the input
    tensor's distributed layout, and supports a configurable ``correction``.

    Parameters
    ----------
    input_shape : Union[Tuple, List]
        Shape of each observation in legacy mode, excluding the leading batch
        dimension. Omit in tensor mode.
    correction : int, optional
        Tensor-mode degrees-of-freedom correction, default 1. Use 0 for population
        variance across ensemble members or GP component means. Finalization
        requires more observations than the correction. Updates always return
        None; ``__call__`` returns None until that condition holds.
    mode : {"legacy", "tensor"}, optional
        ``"legacy"`` preserves the existing constructor and implicit world-group
        reductions. ``"tensor"`` enables detached streaming accumulation: ordinary
        tensors are local and distributed tensor layouts determine communication.
        Tensor mode infers output shape/device from the first update or ``like``;
        omit ``input_shape`` and ``device`` in this mode.
    like : torch.Tensor, optional
        Tensor-mode output template, including any spatial layout. It does not
        add an observation. Not needed when the first update supplies the layout.
    dtype : torch.dtype, optional
        Tensor-mode accumulation dtype (float32 or float64). By default, float64
        inputs remain float64 and lower-precision inputs accumulate in float32.
        The dtype settles from ``like`` or the first update. Inferred float32
        rejects later float64 inputs; specify ``dtype`` to allow explicit casts.

    Notes
    -----
    In tensor mode, ``update(value, dim=0)`` appends a batch of observations;
    ``dim=None`` appends one complete field or observable. Each observation has
    equal weight. State shape and output layout must remain aligned. Empty
    updates contribute no observations. ``combine(other)`` appends a disjoint
    state's observations, and ``reset()`` discards accumulated observations.
    Reading/finalizing does not mutate the accumulated state. In tensor mode,
    ``input_shape`` and ``device`` are None; results carry shape, device and
    layout, and ``dtype`` records the settled accumulation precision.
    ``update`` returns None; use ``finalize`` to read the result.

    Note
    ----
    See "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
    by Chan et al.
    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    for details.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...] | list[int] | None = None,
        *,
        mode: Literal["legacy", "tensor"] = "legacy",
        like: Tensor | None = None,
        correction: int = 1,
        **kwargs,
    ):
        self._moments = None
        _check_correction(correction)
        self.correction = correction
        if mode == "tensor":
            if input_shape is not None or "device" in kwargs:
                raise ValueError("Tensor mode infers shape/device from inputs or like")
            self._init_tensor_mode(variance=True, like=like, kwargs=kwargs)
            return
        if mode != "legacy" or like is not None or correction != 1:
            raise ValueError("Use mode='tensor' for tensor-aware streaming statistics")
        super().__init__(input_shape, **kwargs)
        self.n = torch.zeros([1], dtype=torch.int32, device=self.device)
        self.sum = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)
        self.sum2 = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)

    def __call__(self, inputs: Tensor, dim: int | None = 0) -> Tensor | None:
        """Calculate an initial variance

        Parameters
        ----------
        inputs : Tensor
            Input data
        dim : int or None, optional
            Observation axis; None appends one entire field in tensor mode.

        Returns
        -------
        Tensor or None
            Variance with the configured correction in tensor mode, or None
            when there are insufficient observations. Legacy behavior is unchanged.
        """

        if self._moments is not None:
            self.reset()
            self.update(inputs, dim=dim)
            return self.finalize() if self.count > self.correction else None
        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )
        self.sum = torch.sum(inputs, dim=dim)
        self.n = torch.as_tensor([inputs.shape[0]], device=self.device)

        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            # Compute mean and send around.
            dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.n, op=dist.ReduceOp.SUM)

            self.sum2 = torch.sum((inputs - self.sum / self.n) ** 2, dim=dim)
            dist.all_reduce(self.sum2, op=dist.ReduceOp.SUM)
        else:
            self.sum2 = torch.sum((inputs - self.sum / self.n) ** 2, dim=dim)

        if self.n < 2.0:
            return self.sum2
        else:
            return self.sum2 / (self.n - 1.0)

    def update(self, inputs: Tensor, dim: int | None = 0) -> Tensor | None:
        """Update current variance value and essential statistics with new data

        Parameters
        ----------
        inputs : Tensor
            Input data
        dim : int or None, optional
            Observation axis; None appends one entire field in tensor mode.
            Legacy mode supports only 0.

        Returns
        -------
        Tensor or None
            Unbiased variance in legacy mode; None in tensor mode.

        Raises
        ------
        TypeError
            Tensor-mode input is not a real floating-point tensor.
        ValueError
            Tensor-mode input is incompatible with the settled dtype or layout,
            or a nonzero observation axis is used in legacy mode.
        IndexError
            Tensor-mode observation axis is out of range.
        """

        if self._moments is not None:
            self._moments.update(inputs, dim)
            self.dtype = self._moments.dtype
            return None
        if dim != 0:
            raise ValueError("Legacy Variance.update only supports dim=0")
        self._check_shape(inputs)
        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )

        new_n = torch.as_tensor([inputs.shape[0]], device=self.device)
        new_sum = torch.sum(inputs, dim=0)
        # TODO(Dallas) Move distributed calls into finalize.
        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            dist.all_reduce(new_n, op=dist.ReduceOp.SUM)
            dist.all_reduce(new_sum, op=dist.ReduceOp.SUM)
            new_sum2 = torch.sum((inputs - new_sum / new_n) ** 2, dim=0)
            dist.all_reduce(new_sum2, op=dist.ReduceOp.SUM)

        else:
            # Calculate new statistics
            new_sum2 = torch.sum((inputs - new_sum / new_n) ** 2, dim=0)

        delta = self.sum * new_n / self.n - new_sum
        # Update
        self.sum += new_sum
        self.sum2 += new_sum2
        self.sum2 += self.n / new_n / (self.n + new_n) * (delta) ** 2
        self.n += new_n
        if self.n < 2.0:
            return self.sum2
        else:
            return self.sum2 / (self.n - 1.0)

    @property
    def mean(self) -> Tensor:
        """Mean value"""
        if self._moments is not None:
            return self._moments.read_mean()
        return self.sum / self.n

    def finalize(self, std: bool = False, *, correction: int | None = None) -> Tensor:
        """Read variance or standard deviation with the requested correction

        Parameters
        ----------
        std : bool, optional
            Compute standard deviation, by default False.
        correction : int, optional
            Override the tensor-mode correction for this read. Does not alter
            the configured correction or accumulated state.

        Returns
        -------
        Tensor
            Final variance or standard deviation. The mean is available through
            the ``mean`` property. Tensor mode does not mutate state; legacy
            mode also stores ``var`` and, when requested, ``std``.

        Raises
        ------
        ValueError
            Correction is invalid or there are insufficient observations.
        """
        if self._moments is not None:
            result = self._moments.read_variance(
                self.correction if correction is None else correction
            )
            return result.sqrt() if std else result
        if correction is not None and correction != 1:
            raise ValueError("Legacy Variance uses correction=1")
        if not (self.n > 1.0):
            raise ValueError(
                "Error! In order to finalize, there needs to be at least 2 samples."
            )
        self.var = self.sum2 / (self.n - 1.0)
        if std:
            self.std = torch.sqrt(self.var)
            return self.std
        else:
            return self.var


def __getattr__(name):
    """Keep the original base-class import available without eager warnings."""
    if name == "EnsembleMetrics":
        warnings.warn(
            "EnsembleMetrics is deprecated; use StreamingStatistic instead.",
            LegacyFeatureWarning,
            stacklevel=2,
        )
        return StreamingStatistic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Preserve historical wildcard exports, including imported public names, while
# resolving the deprecated base lazily so ordinary module imports stay silent.
__all__ = [name for name in globals() if not name.startswith("_")]
__all__.append("EnsembleMetrics")
