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

r"""Fused all-reduce for logging and metrics.

:func:`fused_all_reduce` packs many tensors - or a
:class:`~tensordict.TensorDict` (possibly nested) - into a single buffer, issues
one ``all_reduce``, and unpacks the result back into the caller's structure. It
is a metrics/logging reducer (not autograd-aware); see the function docstring
for the full contract.
"""

import functools
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDictBase


def _reduce_keyed(
    keys: list,
    values: list[torch.Tensor],
    reduce_fn: Callable[[list[torch.Tensor]], list[torch.Tensor]],
) -> list[torch.Tensor]:
    """Reduce keyed tensors in a rank-deterministic order.

    Packs ``values`` in sorted-key order (so every rank lays out the fused
    buffer identically) and scatters the results back into the original
    ``keys`` order.

    Parameters
    ----------
    keys : list
        The keys associated with ``values`` (flat or nested-tuple keys).
    values : list[torch.Tensor]
        The tensors to reduce, aligned with ``keys``.
    reduce_fn : Callable[[list[torch.Tensor]], list[torch.Tensor]]
        The order-preserving list reducer (see :func:`_fused_reduce_tensors`).

    Returns
    -------
    list[torch.Tensor]
        The reduced tensors in the original ``keys`` order.
    """
    # Pack in a rank-deterministic order; normalize flat (str) and nested
    # (tuple, e.g. ("sub", "x")) keys to tuples so they share one total order.
    packing_order = sorted(
        range(len(keys)),
        key=lambda i: keys[i] if isinstance(keys[i], tuple) else (keys[i],),
    )
    reduced_packed = reduce_fn([values[i] for i in packing_order])
    reduced: list[torch.Tensor | None] = [None] * len(keys)
    for slot, original_index in enumerate(packing_order):
        reduced[original_index] = reduced_packed[slot]
    return reduced  # type: ignore[return-value]


@torch.no_grad()
def _fused_reduce_tensors(
    tensors: list[torch.Tensor],
    *,
    op: dist.ReduceOp,
    group: dist.ProcessGroup | None,
    buffer_dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> list[torch.Tensor]:
    """All-reduce a list of arbitrarily-shaped tensors in ONE collective.

    This is the shared core of :func:`fused_all_reduce`: it flattens and
    concatenates every tensor into a single buffer, issues one ``all_reduce``,
    then splits the result back, restoring each tensor's original shape, dtype,
    and device. Outputs are always detached and independent of the inputs.

    Parameters
    ----------
    tensors : list[torch.Tensor]
        The tensors to reduce, in the (deterministic) order they should be
        packed onto the wire.
    op : torch.distributed.ReduceOp
        The reduction op applied to the fused buffer.
    group : torch.distributed.ProcessGroup | None
        The process group to reduce over (``None`` is the default group).
    buffer_dtype : torch.dtype | None
        Explicit fused-buffer dtype (opt-in to any cast), or ``None`` to infer
        it. The inferred dtype is the promotion of all leaf dtypes (so e.g.
        ``float32`` + ``float64`` accumulates in ``float64``), floored at
        ``float32`` for 16-bit float results (``float16`` / ``bfloat16``) whose
        sums would lose precision (mirroring
        :func:`~physicsnemo.distributed.utils._reduce`). Integer/bool leaves
        mixed with float leaves raise ``ValueError`` instead of being cast into
        a floating buffer.
    device : torch.device | str | None
        Device for the fused buffer / collective, or ``None`` to use the first
        tensor's device.

    Returns
    -------
    list[torch.Tensor]
        The reduced tensors, in the same order as the input.
    """
    if not tensors:
        return []
    detached = [t.detach() for t in tensors]

    # Resolve the fused-buffer dtype, refusing a silent lossy int/bool -> float
    # cast: promoting an integer leaf into a float buffer would round-trip it
    # through a mantissa and corrupt large / index-like values. Validate BEFORE
    # the no-op return so the contract fails loud single-process too, not only
    # under world_size > 1. The accumulation dtype is the promotion of all leaf
    # dtypes, floored at float32 for 16-bit floats (mirroring
    # :func:`~physicsnemo.distributed.utils._reduce`) whose half/bfloat16 sums
    # would lose too much precision.
    if buffer_dtype is not None:
        work_dtype = buffer_dtype  # explicit opt-in: the caller owns any casting
    else:
        work_dtype = functools.reduce(torch.promote_types, (t.dtype for t in detached))
        if work_dtype.is_floating_point:
            if any(not t.dtype.is_floating_point for t in detached):
                raise ValueError(
                    "fused_all_reduce would cast integer/bool leaves into a "
                    f"{work_dtype} buffer, silently corrupting large or "
                    "index-like values. Reduce integer leaves on their own (a "
                    "homogeneous integer bundle is exact), or pass buffer_dtype= "
                    "to opt in to the cast."
                )
            if work_dtype.itemsize < 4:
                work_dtype = torch.float32

    # Single-process / uninitialized: no collective, exact detached clones.
    # (Single-GPU logs stay byte-identical and never touch the network.)
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=group) > 1
    ):
        return [t.clone() for t in detached]

    buffer_device = torch.device(device) if device is not None else detached[0].device

    # Pack: flatten + cast every tensor and concatenate into ONE buffer.  ``cat``
    # of flattened tensors (not ``stack``) tolerates heterogeneous leaf shapes,
    # and always allocates, so the buffer never aliases the inputs.
    flats = [t.reshape(-1).to(device=buffer_device, dtype=work_dtype) for t in detached]
    numels = [f.numel() for f in flats]
    buffer = torch.cat(flats)

    # The one collective.
    dist.all_reduce(buffer, op=op, group=group)

    # Unpack: split back, restoring each tensor's shape, dtype, and device.
    # ``copy=True`` keeps every output independent of the shared buffer.
    reduced: list[torch.Tensor] = []
    offset = 0
    for source, n in zip(detached, numels):
        chunk = buffer[offset : offset + n].reshape(source.shape)
        reduced.append(chunk.to(device=source.device, dtype=source.dtype, copy=True))
        offset += n
    return reduced


@torch.no_grad()
def fused_all_reduce(
    tensors: TensorDictBase | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    buffer_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> TensorDictBase | dict[str, torch.Tensor] | list[torch.Tensor]:
    r"""All-reduce many tensors in a single collective, preserving structure.

    Reductions for logging and metrics frequently need to combine *many*
    independent scalars or small tensors across ranks (per-key losses, metric
    sums, sample counts, ...). Issuing one ``all_reduce`` per value is
    latency-bound; this helper instead flattens every value into a single buffer
    and performs **one** collective, then unpacks the result back into the same
    container type the caller passed in - a :class:`~tensordict.TensorDict`
    (possibly nested), a :class:`~collections.abc.Mapping`, or a
    :class:`~collections.abc.Sequence`.

    ``op`` defaults to :attr:`~torch.distributed.ReduceOp.SUM`, matching
    :func:`torch.distributed.all_reduce`. Summing fused sums and counts and
    dividing ``sum / count`` afterwards (see Examples) is the building block for
    a sample-weighted mean that stays correct across uneven shards.

    Parameters
    ----------
    tensors : TensorDictBase | Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
        The tensors to reduce. The return type mirrors the input type. Leaves
        may have heterogeneous shapes and dtypes; each output is returned on its
        leaf's original dtype and device.
    op : torch.distributed.ReduceOp, optional
        The reduction applied to every leaf, by default
        :attr:`~torch.distributed.ReduceOp.SUM`.
    group : torch.distributed.ProcessGroup | None, optional
        The process group to reduce over, by default ``None`` (the default,
        world-wide group).
    buffer_dtype : torch.dtype | None, optional
        Explicit dtype for the fused buffer. By default (``None``) the dtype is
        the promotion of all leaf dtypes (so e.g. ``float32`` + ``float64``
        accumulates in ``float64``), floored at ``float32`` for 16-bit floats
        (mirroring :func:`~physicsnemo.distributed.utils._reduce`); an
        all-integer bundle stays integer and reduces exactly, while mixing
        integer/bool with floating leaves raises ``ValueError`` instead of
        silently casting the integers through floating point (see Notes). Pass
        an explicit dtype to opt in to that cast - e.g. ``torch.float64`` to sum
        large-magnitude integer counts that would overflow ``float32``'s 24-bit
        mantissa.
    device : torch.device | str | None, optional
        Device for the fused buffer and collective, by default ``None`` (the
        first leaf's device). Outputs are always returned on their original
        per-leaf device regardless of this.

    Returns
    -------
    TensorDictBase | dict[str, torch.Tensor] | list[torch.Tensor]
        The reduced tensors in the same structure as ``tensors``: a
        ``TensorDict`` (same, possibly nested, keys) for a ``TensorDict`` input,
        a ``dict`` (same keys, original order) for a ``Mapping`` input, or a
        ``list`` (same order) for a ``Sequence`` input. Every leaf retains its
        input shape, dtype, and device, and is detached and independent of the
        input.

    Raises
    ------
    TypeError
        If ``tensors`` is not a ``TensorDict``, ``Mapping``, or ``Sequence``.
    ValueError
        If the inferred buffer dtype is floating point but some leaf is integer
        or boolean (which the cast would corrupt); pass ``buffer_dtype`` to opt
        in to the cast.

    Notes
    -----
    - **One collective.** All leaves are packed into a single contiguous buffer,
      so exactly one ``all_reduce`` is issued regardless of leaf count.
    - **Deterministic wire order.** ``Mapping`` / ``TensorDict`` keys are sorted
      before packing so every rank lays out the buffer identically; the result
      is returned in the caller's original key order.
    - **No-op fast path.** If ``torch.distributed`` is unavailable/uninitialized
      or ``world_size == 1``, detached clones are returned without any
      collective, so single-process runs are byte-identical.
    - **Not autograd-aware.** Inputs are detached; this is a metrics/logging
      reducer, not a tensor-parallel gradient primitive (see
      :func:`~physicsnemo.distributed.utils._reduce`).
    - **Integer-safe.** Integer/bool bundles reduce exactly in their own dtype;
      mixing them with floating leaves is refused (see Raises) rather than
      silently cast. An explicit ``buffer_dtype`` overrides this.

    Examples
    --------
    Sample-weighted mean via fused sums and counts (the canonical use). On a
    single, uninitialized process this is a no-op reduction, so ``sum / count``
    simply recovers the local mean:

    >>> import torch
    >>> from physicsnemo.distributed import fused_all_reduce
    >>> reduced = fused_all_reduce(
    ...     {"loss_sum": torch.tensor(3.0), "count": torch.tensor(2.0)}
    ... )
    >>> float(reduced["loss_sum"] / reduced["count"])
    1.5

    A sequence of heterogeneously-shaped tensors round-trips to a list:

    >>> out = fused_all_reduce([torch.ones(2, 2), torch.tensor(5.0)])
    >>> [tuple(t.shape) for t in out]
    [(2, 2), ()]

    Integer bundles (counts, indices) reduce exactly in their own dtype:

    >>> reduced = fused_all_reduce({"count": torch.tensor([5, 3])})
    >>> reduced["count"].dtype, reduced["count"].tolist()
    (torch.int64, [5, 3])
    """
    reduce_leaves = functools.partial(
        _fused_reduce_tensors,
        op=op,
        group=group,
        buffer_dtype=buffer_dtype,
        device=device,
    )

    # A ``TensorDict`` is *also* a ``Mapping``, so dispatch on it FIRST: this
    # round-trips a TensorDict to a TensorDict of the same (possibly nested)
    # structure instead of silently degrading it to a plain dict.
    if isinstance(tensors, TensorDictBase):
        leaves = list(tensors.items(include_nested=True, leaves_only=True))
        keys = [key for key, _ in leaves]
        reduced = _reduce_keyed(keys, [value for _, value in leaves], reduce_leaves)
        # Clone the input as a structure/dtype/device template, then write each
        # reduced leaf back by its (possibly nested) key.
        out = tensors.detach().clone()
        for key, value in zip(keys, reduced):
            out[key] = value
        return out

    if isinstance(tensors, Mapping):
        keys = list(tensors.keys())
        reduced = _reduce_keyed(keys, [tensors[key] for key in keys], reduce_leaves)
        return dict(zip(keys, reduced))

    if isinstance(tensors, Sequence):
        return reduce_leaves(list(tensors))

    raise TypeError(
        "fused_all_reduce expects a TensorDict, Mapping, or Sequence of tensors, "
        f"got {type(tensors)=!r}."
    )
