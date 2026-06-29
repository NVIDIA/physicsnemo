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

# TODO this also needs more docstrings
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase

from .manager import DistributedManager


def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    # treat trivial case first
    if num_chunks == 1:
        return [size]

    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


def get_memory_format(tensor):
    """Gets format for tensor"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format


def pad_helper(tensor, dim, new_size, mode="zero"):
    """Util for padding tensors"""
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode="constant", value=0.0)

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad[lhs_slice] = torch.flip(
            torch.conj(tensor_pad[rhs_slice]), dims=[dim]
        )

    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    """Util for truncating"""
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[output_slice].contiguous(memory_format=input_format)

    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    if dim >= tensor.dim():
        raise ValueError(
            f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
        )
    if tensor.shape[dim] < num_chunks:
        raise ValueError(
            "Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
        {num_chunks} chunks. Empty slices are currently not supported."
        )

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)

    return tensor_list


@torch.no_grad()
def reduce_loss(loss: float, dst_rank: int = 0, mean: bool = True):  # pragma: no cover
    """Reduces loss from all processes to destination rank for logging.

    Parameters
    ----------
    loss : float
        loss value
    dst_rank : int, Optional
        destination rank to redce to, by default 0.
    mean : bool, Optional
        Calculate the mean of the losses gathered, by default True.

    Raises
    ------
    Exception
        If DistributedManager has yet to be initialized
    """
    if not DistributedManager.is_initialized():
        raise Exception(
            "Distributed manager should be initialized when using reduce_loss"
        )

    distmng = DistributedManager()
    loss = torch.Tensor([loss]).to(distmng.device)

    # For serial runs, just return the current loss!
    if distmng.world_size == 1:
        return float(loss)

    op = torch.distributed.ReduceOp.SUM if not mean else torch.distributed.ReduceOp.AVG
    torch.distributed.reduce(loss, dst_rank, op, group=None)

    # Return loss if dst_rank, None otherwise
    if distmng.rank == dst_rank:
        return float(loss.cpu())
    else:
        return None


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
        sums would lose precision (mirroring :func:`_reduce`). Integer/bool
        leaves mixed with float leaves raise ``ValueError`` instead of being
        cast into a floating buffer.
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
    # dtypes, floored at float32 for 16-bit floats (mirroring :func:`_reduce`)
    # whose half/bfloat16 sums would lose too much precision.
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
        (mirroring :func:`_reduce`); an all-integer bundle stays integer and
        reduces exactly, while mixing integer/bool with floating leaves raises
        ``ValueError`` instead of silently casting the integers through floating
        point (see Notes). Pass an explicit dtype to opt in to that cast - e.g.
        ``torch.float64`` to sum large-magnitude integer counts that would
        overflow ``float32``'s 24-bit mantissa.
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
      reducer, not a tensor-parallel gradient primitive (see :func:`_reduce`).
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


# distributed primitives
def distributed_transpose(tensor, dim0, dim1, group=None, async_op=False):
    """Perform distributed transpose of tensor to switch sharding dimension"""
    # get input format
    input_format = get_memory_format(tensor)

    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [
        y.contiguous(memory_format=input_format)
        for y in torch.split(tensor, split_size, dim=dim0)
    ]
    x_recv = [torch.empty_like(x_send[0]) for _ in range(comm_size)]

    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)

    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):  # pragma: no cover
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce, use_fp32 only relevant for lower precisions
    # if input is already in double precision, nothing changes
    if use_fp32 and (input_.dtype.itemsize < 4) and input_.dtype.is_floating_point:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_


def _split(input_, dim_, group=None):  # pragma: no cover
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def all_gather_v_wrapper(
    tensor: torch.Tensor,
    sizes: Optional[List[int]] = None,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed AllGatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int], optional
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank, by default None.  Can be single integer
        per rank (assuming all other dimensions except `dim` below are equal)
        or can be full
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """

    comm_size = dist.get_world_size(group=group)

    if (sizes is not None) and (len(sizes) != comm_size):
        raise ValueError(f"Mismatch in sizes {len(sizes)} and comm_size {comm_size}")
    if dim >= tensor.dim():
        raise ValueError()

    if comm_size == 1:
        return tensor

    # This is valid if the the shape is a list of ints, but not if full tensor
    # shapes are passed on each rank.  Check if each element of sizes itself is iterable:

    tensor_format = get_memory_format(tensor)

    if sizes is not None:
        full_shapes = False
        try:
            iterator = iter(sizes[0])  # noqa: F841
        except TypeError:
            # Not iterable, use base tensor shape:
            tensor_shape = list(tensor.shape)
        else:
            # it is iterable, use shapes directly
            full_shapes = True
            tensor_shape = None  # Catch and replace below

        tensor_list = [None] * comm_size

        for src in range(comm_size):
            if full_shapes:
                tensor_shape = sizes[src]
            else:
                tensor_shape[dim] = sizes[src]
            tensor_list[src] = torch.empty(
                tensor_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
    else:
        # assume equal shape on all ranks
        tensor_list = [torch.empty_like(tensor) for _ in range(comm_size)]

    dist.all_gather(tensor_list, tensor, group=group)

    output = torch.cat(tensor_list, dim=dim).contiguous(memory_format=tensor_format)

    return output


def all_gather_v_bwd_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    use_fp32: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed AllReduceV primitive. It is based
    on the idea of a single global tensor which which can be distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes different global tensors of the same shape on each
    rank. It then re-distributes chunks of all these tensors such that each rank
    receives all corresponding parts of a global tensor. Each rank then sums up
    the chunks after receiving it. By design, this primitive thus implements the
    backward pass of the "all_gather_v" primitive. In this case, the result would
    be a single global gradient tensor distributed onto different ranks.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor on each rank (different one on each rank)
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local tensor, i.e. result of reduction of all corresponding chunks
        from all global tensors for each rank separately
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = sizes[rank]
    tmp = [
        torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(comm_size)
    ]
    scatter_list = list(torch.split(tensor, sizes, dim=dim))
    scatter_list = [t.contiguous() for t in scatter_list]

    dist.all_to_all(tmp, scatter_list, group=group)
    stack_dim = tensor.dim()
    tmp = torch.stack(tmp, dim=stack_dim)

    if use_fp32 and (tmp.dtype.itemsize < 4) and tmp.dtype.is_floating_point:
        # cast to float before sum and return float, then cast back
        output = tmp.sum(dim=stack_dim, dtype=torch.float32)
        output = output.to(dtype=tensor.dtype)
    else:
        # else: just do sum in native dtype
        output = tmp.sum(dim=stack_dim)

    return output


def gather_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    dst: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed GatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    dst : int, optional
        destination rank which contains the full global tensor after the
        operation, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on destination rank
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if not (0 <= dst < comm_size):
        raise ValueError()
    if tensor.size(dim) != sizes[rank]:
        raise ValueError()

    if comm_size == 1:
        return tensor

    tensor_shape = list(tensor.shape)
    x_recv = [None] * comm_size
    x_send = [None] * comm_size

    for r in range(comm_size):
        if rank == dst:
            tensor_shape[dim] = sizes[r]
        else:
            tensor_shape[dim] = 0

        x_recv[r] = torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        if r == dst:
            x_send[r] = tensor
        else:
            tensor_shape[dim] = 0
            x_send[r] = torch.empty(
                tensor_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )

    dist.all_to_all(x_recv, x_send, group=group)

    # TODO: clean gather/scatter and some examples up
    # main question is around whether e.g. gather returns
    # None for rank != dst or an empty dummy or an dummy
    # containing meta-information like dtype/etc..
    if rank != dst:
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r]
            x_recv[r] = torch.empty(
                tensor_shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )

    output = torch.cat(x_recv, dim=dim)

    return output


def scatter_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed ScatterV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor, valid on source rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    src : int, optional
        source rank of primitive, i.e. rank of original full global tensor, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        corresponding local part of the global tensor on each rank
    """
    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dist.get_rank(group=group) == 0 and dim >= tensor.dim():
        raise ValueError()
    if not (0 <= src < comm_size):
        raise ValueError()

    # all_to_all is already all_to_all_v, use empty tensors to "mask"-out irrelevant parts
    tensor_shape = list(tensor.shape)
    x_send = [None] * comm_size
    x_recv = [None] * comm_size
    if rank == src:
        scatter_list = torch.split(tensor, sizes, dim=dim)
        scatter_list = [t.contiguous() for t in scatter_list]
        x_send = scatter_list
    else:
        for r in range(comm_size):
            tensor_shape[dim] = 0
            x_send[r] = torch.empty(
                tensor_shape, device=tensor.device, dtype=tensor.dtype
            )

    for r in range(comm_size):
        if r == src:
            tensor_shape[dim] = sizes[rank]
        else:
            tensor_shape[dim] = 0
        x_recv[r] = torch.empty(tensor_shape, device=tensor.device, dtype=tensor.dtype)

    dist.all_to_all(x_recv, x_send, group=group)

    return x_recv[src]


def indexed_all_to_all_v_wrapper(
    tensor: torch.Tensor,
    indices: List[torch.Tensor],
    sizes: List[List[int]],
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements an indexed version of a distributed AllToAllV
    primitive. It is based on the idea of a single global tensor which
    is distributed along a specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local part of global tensor on each rank
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    x_send = [tensor[idx] for idx in indices]
    x_recv = [None] * comm_size
    tensor_shape = list(tensor.shape)
    for r in range(comm_size):
        tensor_shape[dim] = sizes[r][rank]
        x_recv[r] = torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    dist.all_to_all(x_recv, x_send, group=group)

    tensor_to_recv = torch.cat(x_recv, dim=dim)

    return tensor_to_recv


def indexed_all_to_all_v_wrapper_bwd(
    tensor: torch.Tensor,
    indices: List[torch.Tensor],
    sizes: List[List[int]],
    tensor_size_along_dim: int,
    use_fp32: bool = True,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements the backward pass to the indexed version of a distributed
    AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor, i.e. gradient on resulting tensor from forward pass
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    tensor_size_along_dim : int
        size of original local tensor along specified dimension,
        i.e. from the corresponding forward pass
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    dim : int, optional
        dimension along with global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    tensor_shape = list(tensor.shape)

    # scatter gradients, roles reversed compared to forward pass
    # recv_sizes in forward pass
    recv_sizes = [sizes[i][rank] for i in range(comm_size)]
    # send_sizes in forward pass
    send_sizes = [sizes[rank][i] for i in range(comm_size)]

    x_send = torch.split(tensor, recv_sizes, dim=dim)
    x_send = [t.contiguous() for t in x_send]
    x_recv = [None] * comm_size
    for r in range(comm_size):
        tensor_shape[dim] = send_sizes[r]
        x_recv[r] = torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    dist.all_to_all(x_recv, x_send, group=group)

    tensor_to_recv = torch.cat(x_recv, dim=dim)

    # sum up gathered gradients and taking
    # care of precision handling as specified
    # by boolean flag
    indices = torch.cat(indices, dim=0)
    tensor_shape[dim] = tensor_size_along_dim
    if use_fp32 and (tensor.dtype.itemsize < 4) and tensor.dtype.is_floating_point:
        out = torch.zeros(
            tensor_shape,
            dtype=torch.float32,
            device=tensor.device,
        )
        tensor_to_recv = tensor_to_recv.to(dtype=torch.float32)
    else:
        out = torch.zeros(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    out.index_add_(source=tensor_to_recv, index=indices, dim=dim)

    if out.dtype != tensor.dtype:
        out = out.to(tensor.dtype)

    return out


def mark_module_as_shared(
    module: nn.Module,
    process_group: Optional[str],
    recurse: bool = True,
    use_fp32_reduction: bool = True,
) -> nn.Module:
    """
    Helper function to mark parameters of a module as being shared
    across ranks by attaching gradient hooks to the corresponding tensors.

    Parameters
    ----------
    module : nn.Module
        PyTorch module which is to be marked as having shared parameters.
    process_group : str | None
        str indicating process_group which contains ranks across which
        the module's parameters are shared. If passed as None, will default
        to the world group.
    recurse : bool, default=True
        Flag indicating whether the module's parameters are traversed in
        a recursive fashion, i.e. whether sub-modules are also considered
        as having shared parameters.
    use_fp32_reduction : bool, default=True
        Flag indicating whether the reduction for accumulating gradients
        will be done in at least FP32 or the native datatype.
    """

    group = DistributedManager().group(process_group)
    handle_key = "_shared_weight_dist_hook"

    def hook(grad: torch.Tensor) -> torch.Tensor:
        # the documentation states that
        # "The hook should not modify its argument, but it can optionally return a new gradient
        #  which will be used in place of grad."
        # as all_reduce is an in-place operation, need to copy gradient
        grad = _reduce(grad.clone(), group=group, use_fp32=use_fp32_reduction)
        return grad

    def hook_post_accum(param: torch.Tensor) -> None:
        # the documentation states that
        # "Note that, unlike other autograd hooks, this hook operates on the tensor that requires grad
        #  and not the grad itself. The hook can in-place modify and access its Tensor argument,
        # including its .grad field."
        param.grad = _reduce(param.grad, group=group, use_fp32=use_fp32_reduction)

    for name, param in module.named_parameters(recurse=recurse):
        error_msg = f"Parameter {name} already marked as having shared weights, can't mark it again!"
        if hasattr(param, handle_key):
            raise RuntimeError(error_msg)
        if torch.__version__ < (2, 1):
            handle = param.register_hook(hook)
        else:
            handle = param.register_post_accumulate_grad_hook(hook_post_accum)
        setattr(param, handle_key, handle)

    return module


def unmark_module_as_shared(
    module: nn.Module,
    recurse: bool = True,
) -> nn.Module:
    """
    Helper function to unmark parameters of a module as being shared
    across ranks by removing attached gradient hooks.

    Parameters
    ----------
    module : nn.Module
        PyTorch module which is to be unmarked as having shared parameters.
    recurse : bool, default=True
        Flag indicating whether the module's parameters are traversed in
        a recursive fashion, i.e. whether sub-modules are also considered
        as having shared parameters.
    """
    handle_key = "_shared_weight_dist_hook"
    for name, param in module.named_parameters(recurse=recurse):
        error_msg = (
            f"Parameter {name} NOT marked as having shared weights, can't unmark it!"
        )
        if not hasattr(param, handle_key):
            raise RuntimeError(error_msg)
        handle = getattr(param, handle_key)
        handle.remove()
        delattr(param, handle_key)

    return module
