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

r"""All-reduce one or many tensors (or a TensorDict) in a single collective.

:func:`fused_all_reduce` packs one or many tensors - or a
:class:`~tensordict.TensorDict` (possibly nested) - into a single buffer, issues
one ``all_reduce``, and unpacks the result back into the caller's structure.
Inputs are detached, so it reduces values rather than gradients (it is not
autograd-aware); see the function docstring for the full contract.
"""

import functools
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDictBase

# The types fused_all_reduce returns (mirrors the input structure).
_ReducedContainer = (
    torch.Tensor | TensorDictBase | dict[str, torch.Tensor] | list[torch.Tensor]
)


class _FusedAllReduceWork:
    """Async handle returned by :func:`fused_all_reduce` when ``async_op=True``.

    The fused reduction packs every leaf into one buffer, reduces it, then
    unpacks it (split the buffer + cast each slice back to its leaf's
    dtype/device). With ``async_op`` the collective is launched non-blocking, so
    the unpack is *deferred* into :meth:`wait`: until then the output leaves hold
    this rank's own (un-reduced) values, exactly like
    :func:`torch.distributed.all_reduce`'s in-place async contract - a premature
    read is stale, never uninitialized garbage. :meth:`wait` blocks on the
    collective, writes the reduced values back into the outputs, and is
    idempotent. When there was no collective (empty input or single process) the
    handle starts already-complete and :meth:`wait` is a no-op.
    """

    def __init__(
        self,
        work: dist.Work | None = None,
        buffer: torch.Tensor | None = None,
        outputs: Sequence[torch.Tensor] = (),
        numels: Sequence[int] = (),
    ) -> None:
        self._work = work
        self._buffer = buffer
        self._outputs = outputs
        self._numels = numels
        self._completed = work is None  # no collective => already done

    def wait(self) -> bool:
        """Block on the collective, then fill the outputs in place.

        Returns
        -------
        bool
            ``True`` once the reduction is complete (mirroring
            :meth:`torch.distributed.Work.wait`). Safe to call repeatedly; only
            the first call does any work.
        """
        if self._completed:
            return True
        self._work.wait()
        # Deferred unpack: split the reduced buffer and cast each slice back to
        # its leaf's dtype/device (``copy_`` handles both). The buffer tiles the
        # leaves exactly by construction (it is ``cat`` of the same flats whose
        # ``numels`` we recorded); guard that invariant in eager as cheap
        # insurance against a future pack/unpack bookkeeping bug.
        if (
            not torch.compiler.is_compiling()
            and sum(self._numels) != self._buffer.numel()
        ):
            raise RuntimeError(
                "fused_all_reduce: reduced buffer does not tile its leaves "
                f"({sum(self._numels)=} != {self._buffer.numel()=})."
            )
        with torch.no_grad():
            offset = 0
            for out, n in zip(self._outputs, self._numels):
                out.copy_(self._buffer[offset : offset + n].reshape(out.shape))
                offset += n
        self._completed = True
        return True

    def is_completed(self) -> bool:
        """Whether the reduction has completed (``wait`` ran, or it was a no-op)."""
        return self._completed


def _reduce_keyed(
    keys: list,
    values: list[torch.Tensor],
    reduce_fn: Callable[
        [list[torch.Tensor]], tuple[list[torch.Tensor], _FusedAllReduceWork | None]
    ],
) -> tuple[list[torch.Tensor], _FusedAllReduceWork | None]:
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
    reduce_fn : Callable
        The order-preserving list reducer (see :func:`_fused_reduce_tensors`),
        returning ``(reduced_list, work_or_None)``.

    Returns
    -------
    tuple[list[torch.Tensor], _FusedAllReduceWork | None]
        The reduced tensors in the original ``keys`` order, and the reducer's
        work handle (``None`` unless reducing asynchronously).
    """
    # Pack in a rank-deterministic order; normalize flat (str) and nested
    # (tuple, e.g. ("sub", "x")) keys to tuples so they share one total order.
    packing_order = sorted(
        range(len(keys)),
        key=lambda i: keys[i] if isinstance(keys[i], tuple) else (keys[i],),
    )
    reduced_packed, work = reduce_fn([values[i] for i in packing_order])
    reduced: list[torch.Tensor | None] = [None] * len(keys)
    for slot, original_index in enumerate(packing_order):
        reduced[original_index] = reduced_packed[slot]
    return reduced, work  # type: ignore[return-value]


@torch.no_grad()
def _fused_reduce_tensors(
    tensors: list[torch.Tensor],
    *,
    op: dist.ReduceOp,
    group: dist.ProcessGroup | None,
    buffer_dtype: torch.dtype | None,
    device: torch.device | str | None,
    async_op: bool = False,
) -> tuple[list[torch.Tensor], _FusedAllReduceWork | None]:
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
    async_op : bool
        If ``True``, launch the collective non-blocking and defer the unpack
        into the returned handle's ``wait`` (the outputs start as local-value
        clones). If ``False`` (default), block and unpack eagerly, returning a
        ``None`` handle.

    Returns
    -------
    tuple[list[torch.Tensor], _FusedAllReduceWork | None]
        The reduced tensors (same order as the input) and a work handle. The
        handle is ``None`` when ``async_op`` is ``False``; otherwise it is a
        :class:`_FusedAllReduceWork` whose ``wait`` completes the collective and
        fills the outputs (which hold the local, un-reduced values until then).
    """
    if not tensors:
        return [], (_FusedAllReduceWork() if async_op else None)
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
        return [t.clone() for t in detached], (
            _FusedAllReduceWork() if async_op else None
        )

    buffer_device = torch.device(device) if device is not None else detached[0].device

    # Pack: flatten + cast every tensor and concatenate into ONE buffer.  ``cat``
    # of flattened tensors (not ``stack``) tolerates heterogeneous leaf shapes,
    # and always allocates, so the buffer never aliases the inputs.
    flats = [t.reshape(-1).to(device=buffer_device, dtype=work_dtype) for t in detached]
    numels = [f.numel() for f in flats]
    buffer = torch.cat(flats)

    if async_op:
        # Launch non-blocking and defer the unpack to the handle's wait().
        # Outputs start as clones of the local (un-reduced) leaves, so a
        # premature read is stale-but-valid (matching dist.all_reduce's in-place
        # semantics), never uninitialized garbage; wait() overwrites them.
        outputs = [s.clone() for s in detached]
        work = dist.all_reduce(buffer, op=op, group=group, async_op=True)
        return outputs, _FusedAllReduceWork(work, buffer, outputs, numels)

    else:
        # The one (blocking) collective.
        dist.all_reduce(buffer, op=op, group=group)

        # Unpack: split back, restoring each tensor's shape, dtype, and device.
        # ``copy=True`` keeps every output independent of the shared buffer.
        reduced: list[torch.Tensor] = []
        offset = 0
        for source, n in zip(detached, numels):
            chunk = buffer[offset : offset + n].reshape(source.shape)
            reduced.append(
                chunk.to(device=source.device, dtype=source.dtype, copy=True)
            )
            offset += n
        return reduced, None


@torch.no_grad()
def fused_all_reduce(
    tensors: torch.Tensor
    | TensorDictBase
    | Mapping[str, torch.Tensor]
    | Sequence[torch.Tensor],
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    buffer_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    async_op: bool = False,
) -> _ReducedContainer | tuple[_ReducedContainer, _FusedAllReduceWork]:
    r"""All-reduce one or many tensors in a single collective, preserving structure.

    Combining *many* independent scalars or small tensors across ranks with one
    ``all_reduce`` per value is latency-bound. This helper instead flattens
    every value into a single buffer and performs **one** collective, then
    unpacks the result back into the same container type the caller passed in -
    a :class:`~tensordict.TensorDict` (possibly nested), a
    :class:`~collections.abc.Mapping`, or a :class:`~collections.abc.Sequence`.

    ``op`` defaults to :attr:`~torch.distributed.ReduceOp.SUM`, matching
    :func:`torch.distributed.all_reduce`. Summing fused sums and counts and
    dividing ``sum / count`` afterwards (see Examples) is the building block for
    a sample-weighted mean that stays correct across uneven shards.

    Parameters
    ----------
    tensors : torch.Tensor | TensorDictBase | Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
        The tensors to reduce. The return type mirrors the input type: a single
        ``Tensor`` reduces to a single ``Tensor`` (the degenerate one-leaf case,
        mirroring :func:`torch.distributed.all_reduce`). Leaves may have
        heterogeneous shapes and dtypes; each output is returned on its leaf's
        original dtype and device.
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
    async_op : bool, optional
        If ``True``, issue the fused collective asynchronously and return
        ``(result, work)``; the result's leaves hold each rank's local values
        until ``work.wait()`` completes the reduction and fills them in place
        (see :class:`_FusedAllReduceWork`). By default ``False`` (blocking),
        returning just ``result``.

    Returns
    -------
    torch.Tensor | TensorDictBase | dict[str, torch.Tensor] | list[torch.Tensor]
        The reduced tensors in the same structure as ``tensors``: a single
        ``Tensor`` for a ``Tensor`` input, a ``TensorDict`` (same, possibly
        nested, keys) for a ``TensorDict`` input, a ``dict`` (same keys, original
        order) for a ``Mapping`` input, or a ``list`` (same order) for a
        ``Sequence`` input. Every leaf retains its input shape, dtype, and
        device, and is detached and independent of the input. When
        ``async_op=True`` the return is instead a ``(result, work)`` tuple whose
        ``result`` is valid only after ``work.wait()`` (see ``async_op``).

    Raises
    ------
    TypeError
        If ``tensors`` is not a ``Tensor``, ``TensorDict``, ``Mapping``, or
        ``Sequence``.
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
    - **Not autograd-aware.** Inputs are detached, so gradients do not flow
      through the reduction (unlike the tensor-parallel primitive
      :func:`~physicsnemo.distributed.utils._reduce`).
    - **Integer-safe.** Integer/bool bundles reduce exactly in their own dtype;
      mixing them with floating leaves is refused (see Raises) rather than
      silently cast. An explicit ``buffer_dtype`` overrides this.
    - **Async.** With ``async_op=True`` the collective is non-blocking and the
      caller gets a ``(result, work)`` pair; call ``work.wait()`` before reading
      ``result`` to let the reduction land (mirrors
      :func:`torch.distributed.all_reduce`'s ``async_op``, and the in-repo
      ``distributed_transpose``).

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

    When the per-rank shards are equal-weight, ``op=ReduceOp.AVG`` averages
    across ranks directly, with no separate count leaf. On a single process the
    reduction is a no-op, so the local value is returned unchanged:

    >>> import torch.distributed as dist
    >>> reduced = fused_all_reduce({"loss": torch.tensor(1.5)}, op=dist.ReduceOp.AVG)
    >>> float(reduced["loss"])
    1.5

    A single tensor reduces to a single tensor - the degenerate one-leaf case,
    mirroring :func:`torch.distributed.all_reduce`. On a single process this is
    a no-op:

    >>> reduced = fused_all_reduce(torch.tensor(5.0))
    >>> float(reduced)
    5.0

    A sequence of heterogeneously-shaped tensors round-trips to a list:

    >>> out = fused_all_reduce([torch.ones(2, 2), torch.tensor(5.0)])
    >>> [tuple(t.shape) for t in out]
    [(2, 2), ()]

    Integer bundles (counts, indices) reduce exactly in their own dtype:

    >>> reduced = fused_all_reduce({"count": torch.tensor([5, 3])})
    >>> reduced["count"].dtype, reduced["count"].tolist()
    (torch.int64, [5, 3])

    Asynchronous reduction returns a ``(result, work)`` pair; call ``wait``
    before reading ``result``. On a single process this is a no-op handle:

    >>> result, work = fused_all_reduce({"x": torch.tensor(1.0)}, async_op=True)
    >>> work.wait()
    True
    >>> float(result["x"])
    1.0
    """
    reduce_leaves = functools.partial(
        _fused_reduce_tensors,
        op=op,
        group=group,
        buffer_dtype=buffer_dtype,
        device=device,
        async_op=async_op,
    )

    # A single tensor is the degenerate one-leaf case (mirrors
    # ``torch.distributed.all_reduce``): route it through the same packing /
    # no-op / dtype-guard / async machinery and return the lone reduced leaf.
    if isinstance(tensors, torch.Tensor):
        [result], work = reduce_leaves([tensors])
    # A ``TensorDict`` is *also* a ``Mapping``, so dispatch on it FIRST: this
    # round-trips a TensorDict to a TensorDict of the same (possibly nested)
    # structure instead of silently degrading it to a plain dict.
    elif isinstance(tensors, TensorDictBase):
        leaves = list(tensors.items(include_nested=True, leaves_only=True))
        keys = [key for key, _ in leaves]
        reduced, work = _reduce_keyed(
            keys, [value for _, value in leaves], reduce_leaves
        )
        # Clone the input as a structure/dtype/device template, then write each
        # reduced leaf back by its (possibly nested) key.
        result = tensors.detach().clone()
        for key, value in zip(keys, reduced):
            result[key] = value
    elif isinstance(tensors, Mapping):
        keys = list(tensors.keys())
        reduced, work = _reduce_keyed(
            keys, [tensors[key] for key in keys], reduce_leaves
        )
        result = dict(zip(keys, reduced))
    elif isinstance(tensors, Sequence) and not isinstance(tensors, (str, bytes)):
        result, work = reduce_leaves(list(tensors))
    else:
        raise TypeError(
            "fused_all_reduce expects a Tensor, TensorDict, Mapping, or Sequence "
            f"of tensors, got {type(tensors)=!r}."
        )

    return (result, work) if async_op else result
