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

from __future__ import annotations

import itertools
import math
from typing import Any, Callable

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    Replicate,
    Shard,
)

from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    TensorMeta,
    _stride_from_contiguous_shape_C_style,
)
from physicsnemo.domain_parallel.shard_tensor import (
    _is_tracing,
    _torch_function_fallback_via_dtensor,
)
from physicsnemo.domain_parallel.shard_utils.exchange import (
    _accumulator_dtype,
    funcol_all_to_all_v_rows,
    resolve_group_name,
)
from physicsnemo.domain_parallel.shard_utils.patch_core import (
    MissingShardPatch,
)

aten = torch.ops.aten


# ---------------------------------------------------------------------------
# Routed gather / scatter-add along the sharded dim
# ---------------------------------------------------------------------------
#
# ``values`` is ``Shard(d)`` with known per-rank sizes; ``index`` holds GLOBAL
# positions along ``d`` that may live on any rank. Everything about the exchange
# is known from the specs before any data moves: the shard boundaries of
# ``values`` (``offsets``) and how many requests every rank holds
# (``request_sizes``, from the index's sharding shapes). So the exchange uses
# STATIC split sizes and no host sync:
#
#   1. Every rank sends its full request list to every owner (sizes:
#      ``request_sizes``).
#   2. Each owner gathers, for each requester, a full-length row buffer in
#      request order -- rows it does not own are filler -- and sends it back.
#   3. The requester holds ``W`` buffers of its own length and picks, per
#      request, the row from the owner it computed locally (``bucketize``).
#
# Bandwidth is ``W`` x the requested rows instead of 1 x plus a count exchange;
# for the small ``W`` of a domain group that is cheaper than stalling the CPU
# on a device-to-host copy twice per gather. The backward is the mirror image:
# the requester sends its gradient buffer to every owner, each owner
# ``index_add``s only the rows it owns, which yields a complete ``Shard(d)``
# gradient with no pending reduction. The exchanged requests are saved from
# the forward, so the backward is a single collective.
#
# Transport (row all-to-all with given split sizes), group-name resolution and
# the accumulation policy come from ``shard_utils.exchange`` (shared with
# ``halo_scatter``); only the routing is specific to this module.
#
# The routed core is a ``custom_op`` (opaque to fake mode), so the op is
# ``torch.compile`` safe: every input size is a spec constant and every output
# shape is static (index shape + trailing dims, or the local values shape for
# the backward).

_INDEX_DTYPES = (torch.int32, torch.int64)


def _wrap_negative(index_flat: torch.Tensor, n_global: int) -> torch.Tensor:
    r"""Wrap negative global positions like eager indexing does.

    Parameters
    ----------
    index_flat : torch.Tensor
        1-D global positions; negatives count from the end.
    n_global : int
        Global extent along the indexed dim. Positions ``>= n_global`` are not
        checked (they select filler rows on the owner side).

    Returns
    -------
    torch.Tensor
        Non-negative positions.
    """
    return torch.where(index_flat < 0, index_flat + n_global, index_flat)


def _owner_of(index_flat: torch.Tensor, offsets: list[int]) -> torch.Tensor:
    r"""Owning rank of every global position.

    Parameters
    ----------
    index_flat : torch.Tensor
        1-D non-negative global positions.
    offsets : list[int]
        The ``W + 1`` shard boundaries along the indexed dim.

    Returns
    -------
    torch.Tensor
        Rank index in ``[0, W)`` per position.
    """
    bounds = torch.tensor(offsets[1:-1], dtype=torch.int64, device=index_flat.device)
    return torch.bucketize(index_flat, bounds, right=True)


def _rank_request_slice(n_requests: int, rank: int, world_size: int) -> tuple[int, int]:
    r"""This rank's contiguous share of ``n_requests`` replicated requests.

    Used in the backward of a gather with a *replicated* index: every rank
    holds the full output gradient, so each is responsible for its share of
    the requests or the owners would accumulate ``world_size`` copies.

    Parameters
    ----------
    n_requests : int
        Total number of (flattened) requests.
    rank : int
        This rank's position on the mesh dim.
    world_size : int
        Mesh dim size.

    Returns
    -------
    tuple[int, int]
        ``(lo, hi)`` into the flattened requests; shares differ by at most one.
    """
    base, extra = divmod(n_requests, world_size)
    lo = rank * base + min(rank, extra)
    return lo, lo + base + (1 if rank < extra else 0)


@torch.library.custom_op("physicsnemo::exchange_requests", mutates_args=())
def _exchange_requests_op(
    index_flat: torch.Tensor, request_sizes: list[int], group_name: str
) -> torch.Tensor:
    r"""Send this rank's requests to every rank; receive every rank's requests.

    Runs once per gather in the forward; the result is saved for the backward,
    which therefore needs only the gradient exchange.

    Parameters
    ----------
    index_flat : torch.Tensor
        This rank's flattened, non-negative global positions.
    request_sizes : list[int]
        Number of requests held by each rank (spec constants).
    group_name : str
        c10d group name of the 1-D mesh.

    Returns
    -------
    torch.Tensor
        All requests, rank-ordered: ``sum(request_sizes)`` positions.
    """
    world_size = len(request_sizes)
    n_mine = index_flat.numel()
    return funcol_all_to_all_v_rows(
        index_flat.repeat(world_size), [n_mine] * world_size, request_sizes, group_name
    )


@_exchange_requests_op.register_fake
def _exchange_requests_fake(
    index_flat: torch.Tensor, request_sizes: list[int], group_name: str
) -> torch.Tensor:
    r"""Fake (meta) implementation: ``sum(request_sizes)`` positions."""
    return index_flat.new_empty((sum(request_sizes),))


def _owned_rows(
    values: torch.Tensor,
    all_requests: torch.Tensor,
    offsets: list[int],
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Local row ids of the requests this rank owns, plus the ownership mask.

    Parameters
    ----------
    values : torch.Tensor
        This rank's shard of rows.
    all_requests : torch.Tensor
        Every rank's requests (global positions), rank-ordered.
    offsets : list[int]
        The ``W + 1`` shard boundaries.
    rank : int
        This rank.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(local_rows, owned)``: local row ids clamped into range (filler for
        rows this rank does not own) and the boolean mask of owned requests.
    """
    lo, hi = offsets[rank], offsets[rank + 1]
    owned = (all_requests >= lo) & (all_requests < hi)
    n_local = values.shape[0]
    local_rows = (all_requests - lo).clamp_(0, max(n_local - 1, 0))
    return local_rows, owned


@torch.library.custom_op("physicsnemo::routed_gather", mutates_args=())
def _routed_gather_op(
    values: torch.Tensor,
    index: torch.Tensor,
    all_requests: torch.Tensor,
    offsets: list[int],
    request_sizes: list[int],
    rank: int,
    group_name: str,
    index_replicated: bool,
) -> torch.Tensor:
    r"""``full_values[index]`` along dim 0 without materializing ``full_values``.

    Parameters
    ----------
    values : torch.Tensor
        This rank's shard of rows ``[offsets[rank], offsets[rank + 1])``.
    index : torch.Tensor
        Integer tensor of any shape holding global row ids (already wrapped
        to non-negative positions).
    all_requests : torch.Tensor
        Every rank's requests, rank-ordered (from
        :func:`_exchange_requests_op`).
    offsets : list[int]
        The ``W + 1`` shard boundaries along dim 0.
    request_sizes : list[int]
        ``index.numel()`` on every rank, in rank order.
    rank : int
        This rank's position on the mesh dim.
    group_name : str
        c10d group name of the 1-D mesh.
    index_replicated : bool
        Whether every rank holds the same requests; unused in the forward, it
        rides along so the backward knows to split responsibility.

    Returns
    -------
    torch.Tensor
        Gathered rows, shape ``(*index.shape, *values.shape[1:])``.
    """
    world_size = len(request_sizes)
    n_mine = index.numel()
    trailing = values.shape[1:]
    index_flat = index.reshape(-1)

    # Rows for the requests this rank owns (filler for the rest), then one
    # full-length buffer back to every requester.
    local_rows, _ = _owned_rows(values, all_requests, offsets, rank)
    if values.shape[0] == 0:
        rows = values.new_zeros((all_requests.numel(), *trailing))
    else:
        rows = values.index_select(0, local_rows)
    received = funcol_all_to_all_v_rows(
        rows, request_sizes, [n_mine] * world_size, group_name
    )

    # ``received`` holds W buffers of my n_mine requests; take each row from
    # its owner.
    received = received.reshape(world_size, n_mine, *trailing)
    owner = _owner_of(index_flat, offsets)
    out = received[owner, torch.arange(n_mine, device=index.device)]
    return out.reshape((*index.shape, *trailing))


@_routed_gather_op.register_fake
def _routed_gather_fake(
    values: torch.Tensor,
    index: torch.Tensor,
    all_requests: torch.Tensor,
    offsets: list[int],
    request_sizes: list[int],
    rank: int,
    group_name: str,
    index_replicated: bool,
) -> torch.Tensor:
    r"""Fake (meta) implementation: the output shape is static, no data needed."""
    return values.new_empty((*index.shape, *values.shape[1:]))


@torch.library.custom_op("physicsnemo::routed_scatter_add", mutates_args=())
def _routed_scatter_add_op(
    grad: torch.Tensor,
    index: torch.Tensor,
    all_requests: torch.Tensor,
    offsets: list[int],
    request_sizes: list[int],
    request_slices: list[int],
    rank: int,
    n_local_rows: int,
    group_name: str,
) -> torch.Tensor:
    r"""Adjoint of :func:`_routed_gather_op`.

    Every requester sends its full gradient buffer to every owner; each owner
    accumulates the rows it owns into a zero tensor of its local rows.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient of the gathered rows, shape ``(*index.shape, *trailing)``.
    index : torch.Tensor
        The forward index (non-negative global row ids).
    all_requests : torch.Tensor
        Every rank's requests, saved from the forward.
    offsets : list[int]
        The ``W + 1`` shard boundaries along dim 0.
    request_sizes : list[int]
        ``index.numel()`` on every rank, in rank order.
    request_slices : list[int]
        ``2 * W`` ints, ``(lo, hi)`` per requester: the share of that
        requester's flattened requests whose gradient counts. The full range
        for a rank-local (sharded) index; disjoint shares for a replicated
        index, where every rank holds the same full gradient.
    rank : int
        This rank's position on the mesh dim.
    n_local_rows : int
        Number of rows in this rank's shard of ``values``.
    group_name : str
        c10d group name of the 1-D mesh.

    Returns
    -------
    torch.Tensor
        Gradient of this rank's shard of ``values``, shape
        ``(n_local_rows, *trailing)``, in ``grad.dtype``.
    """
    world_size = len(request_sizes)
    n_mine = index.numel()
    trailing = grad.shape[index.ndim :]

    grad_rows = grad.reshape(n_mine, *trailing)
    incoming = funcol_all_to_all_v_rows(
        grad_rows.repeat(world_size, *([1] * len(trailing))),
        [n_mine] * world_size,
        request_sizes,
        group_name,
    )

    # Keep a request iff this rank owns its row and it falls in the sender's
    # responsible share.
    local_rows, owned = _owned_rows(grad, all_requests, offsets, rank)
    position = torch.cat(
        [torch.arange(n, device=grad.device, dtype=torch.int64) for n in request_sizes]
    )
    lo = torch.tensor(
        [request_slices[2 * r] for r in range(world_size)],
        device=grad.device,
        dtype=torch.int64,
    ).repeat_interleave(torch.tensor(request_sizes, device=grad.device))
    hi = torch.tensor(
        [request_slices[2 * r + 1] for r in range(world_size)],
        device=grad.device,
        dtype=torch.int64,
    ).repeat_interleave(torch.tensor(request_sizes, device=grad.device))
    keep = owned & (position >= lo) & (position < hi)

    acc_dtype = _accumulator_dtype(grad.dtype)
    out = torch.zeros((n_local_rows, *trailing), dtype=acc_dtype, device=grad.device)
    weight = keep.to(acc_dtype).reshape(-1, *([1] * len(trailing)))
    out.index_add_(0, local_rows, incoming.to(acc_dtype) * weight)
    return out.to(grad.dtype)


@_routed_scatter_add_op.register_fake
def _routed_scatter_add_fake(
    grad: torch.Tensor,
    index: torch.Tensor,
    all_requests: torch.Tensor,
    offsets: list[int],
    request_sizes: list[int],
    request_slices: list[int],
    rank: int,
    n_local_rows: int,
    group_name: str,
) -> torch.Tensor:
    r"""Fake (meta) implementation: the local gradient shape is static."""
    return grad.new_empty((n_local_rows, *grad.shape[index.ndim :]))


def _routed_gather_setup_context(ctx, inputs, output) -> None:
    r"""Save what the backward needs: index, exchanged requests, spec constants."""
    (
        values,
        index,
        all_requests,
        offsets,
        request_sizes,
        rank,
        group_name,
        index_replicated,
    ) = inputs
    ctx.save_for_backward(index, all_requests)
    ctx.index_replicated = index_replicated
    ctx.offsets = list(offsets)
    ctx.request_sizes = list(request_sizes)
    ctx.rank = rank
    ctx.n_local_rows = values.shape[0]
    ctx.group_name = group_name


def _routed_gather_backward(ctx, grad):
    r"""Backward of the routed gather: send ``grad`` to the owners, accumulate.

    The request exchange is not repeated: ``all_requests`` was saved from the
    forward, so this is a single collective.
    """
    index, all_requests = ctx.saved_tensors
    world_size = len(ctx.request_sizes)
    if ctx.index_replicated:
        slices = [
            b
            for r in range(world_size)
            for b in _rank_request_slice(index.numel(), r, world_size)
        ]
    else:
        slices = [b for n in ctx.request_sizes for b in (0, n)]
    grad_values = _routed_scatter_add_op(
        grad.contiguous(),
        index,
        all_requests,
        ctx.offsets,
        ctx.request_sizes,
        slices,
        ctx.rank,
        ctx.n_local_rows,
        ctx.group_name,
    )
    return grad_values, None, None, None, None, None, None, None


_routed_gather_op.register_autograd(
    _routed_gather_backward, setup_context=_routed_gather_setup_context
)


def _local_index(index: Any) -> torch.Tensor:
    r"""Local rows of an index operand.

    Always via ``to_local()``: dynamo traces that as an op, whereas a raw
    ``_local_tensor`` read on a traced subclass resolves to the *real*
    tensor and fails fakeification when that tensor is a view.

    Parameters
    ----------
    index : ShardTensor or DTensor or torch.Tensor
        Index operand.

    Returns
    -------
    torch.Tensor
        The local tensor of a distributed index, or the plain tensor itself.
    """
    if isinstance(index, (ShardTensor, DTensor)):
        return index.to_local()
    return index


def _index_is_sharded(index: Any) -> bool:
    r"""Whether *index* is a ``Shard(0)`` ShardTensor (rank-local requests).

    Parameters
    ----------
    index : Any
        Index operand.

    Returns
    -------
    bool
        ``True`` for a ``Shard(0)`` ShardTensor, ``False`` for anything
        replicated or plain.

    Raises
    ------
    MissingShardPatch
        If the index is sharded on any other dim, or is Partial.
    """
    if not isinstance(index, ShardTensor):
        return False
    placement = index._spec.placements[0]
    if placement.is_partial():
        raise MissingShardPatch("a Partial index is not supported")
    if placement.is_shard() and placement.dim != 0:
        raise MissingShardPatch(
            f"an index sharded on dim {placement.dim} is not supported; "
            "shard the index on dim 0 or replicate it"
        )
    return placement.is_shard()


def _check_index_dtype(index: Any) -> None:
    r"""Integer indices only (``int32`` / ``int64``).

    Parameters
    ----------
    index : Any
        Index operand.

    Raises
    ------
    MissingShardPatch
        If the index dtype is not ``int32`` or ``int64``.
    """
    dtype = getattr(index, "dtype", None)
    if dtype not in _INDEX_DTYPES:
        raise MissingShardPatch(f"index must be int32 or int64, got {dtype}")


def _shard_sizes(spec: ShardTensorSpec, tensor_dim: int) -> list[int]:
    r"""Per-rank extents of a spec along one tensor dim.

    Parameters
    ----------
    spec : ShardTensorSpec
        Spec of a tensor on a 1-D mesh.
    tensor_dim : int
        Tensor dim to read the extents of.

    Returns
    -------
    list[int]
        One extent per rank, in rank order.
    """
    return [s[tensor_dim] for s in spec.sharding_shapes(0)]


def routed_gather(values: ShardTensor, index: Any, dim: int) -> ShardTensor:
    r"""Differentiable ``values.index_select(dim, index)`` for ``values`` sharded on *dim*.

    Parameters
    ----------
    values : ShardTensor
        Source tensor, ``Shard(dim)`` on a 1-D device mesh.
    index : ShardTensor or DTensor or torch.Tensor
        Integer tensor of any shape holding *global* positions along *dim*.
        A ``Shard(0)`` ShardTensor is rank-local requests; a replicated
        ShardTensor / DTensor / plain tensor is the same requests on every
        rank. Negative positions wrap.
    dim : int
        Dimension of *values* to gather along.

    Returns
    -------
    ShardTensor
        For a sharded index, ``Shard(dim)`` with the index's per-rank sizes;
        for a replicated index, ``Replicate`` (every rank holds the full
        result). The selected axis takes the place of *dim* and the remaining
        index dims are inserted there, matching
        ``full_values.movedim(dim, 0)[index].movedim(...)``; for a 1-D index
        this is exactly ``torch.index_select``.

    Raises
    ------
    MissingShardPatch
        On a multi-dimensional device mesh, a non-integer index, or an index
        sharded on a dim other than 0.
    """
    spec = values._spec
    if spec.mesh.ndim != 1:
        raise MissingShardPatch("routed gather supports 1-D device meshes only")
    _check_index_dtype(index)
    index_sharded = _index_is_sharded(index)
    # Plain-Python spec constants that ride into the custom op as ``int[]``:
    # shard boundaries of ``values`` and the request count on every rank. No
    # tensor is created and nothing is read back from the device.
    world_size = spec.mesh.size(0)
    offsets = [0, *itertools.accumulate(_shard_sizes(spec, dim))]
    idx_local = _local_index(index)
    if index_sharded:
        request_sizes = [math.prod(shape) for shape in index._spec.sharding_shapes(0)]
    else:
        request_sizes = [idx_local.numel()] * world_size

    # Global positions, wrapped once; exchanged once and reused by the backward.
    idx_local = _wrap_negative(idx_local.to(torch.int64), offsets[-1])
    group_name = resolve_group_name(spec.mesh)
    all_requests = _exchange_requests_op(
        idx_local.reshape(-1), request_sizes, group_name
    )

    # to_local / from_local are the differentiable ShardTensor <-> local bridges.
    local = values.to_local().movedim(dim, 0)
    out = _routed_gather_op(
        local,
        idx_local,
        all_requests,
        offsets,
        request_sizes,
        spec.mesh.get_local_rank(),
        group_name,
        not index_sharded,
    )
    # out: (*idx_local.shape, *trailing) with the selected axis first; put it at
    # dim. Materialize the permutation so the wrapped local owns its storage
    # (a view inside a ShardTensor does not survive dynamo fakeification).
    if dim != 0:
        out = out.movedim(
            tuple(range(idx_local.ndim)), tuple(range(dim, dim + idx_local.ndim))
        ).contiguous()

    global_shape = list(spec.tensor_meta.shape)
    if index_sharded:
        idx_sizes = _shard_sizes(index._spec, 0)
        idx_trailing = list(idx_local.shape[1:])
        shard_shapes = tuple(
            tuple(global_shape[:dim] + [n, *idx_trailing] + global_shape[dim + 1 :])
            for n in idx_sizes
        )
        return ShardTensor.from_local(
            out, spec.mesh, (Shard(dim),), sharding_shapes={0: shard_shapes}
        )
    # Explicit (empty) shard shapes: never ``"infer"`` on a traced path.
    return ShardTensor.from_local(out, spec.mesh, (Replicate(),), sharding_shapes={})


def sharded_index_select(tensor: ShardTensor, dim: int, index: Any) -> ShardTensor:
    r"""``torch.index_select`` on a ShardTensor.

    Parameters
    ----------
    tensor : ShardTensor
        Source tensor on a 1-D device mesh.
    dim : int
        Dimension to select along (negative values wrap).
    index : ShardTensor or DTensor or torch.Tensor
        1-D integer index of global positions along *dim*.

    Returns
    -------
    ShardTensor
        - ``tensor`` sharded on *dim*: routed gather (communication scales with
          the rows indexed); output sharded like the index.
        - ``tensor`` sharded on another dim: purely local. A sharded index is
          first gathered (eager only) so every rank selects the same rows of
          its shard; the output keeps ``tensor``'s placement.
        - ``tensor`` replicated: local select; a sharded index yields
          ``Shard(dim)`` (each rank selects its own rows), otherwise
          ``Replicate``.

    Raises
    ------
    MissingShardPatch
        On a multi-dimensional mesh, a Partial source, a non-integer index, an
        index sharded on a dim other than 0, or a sharded index on an
        off-shard select under ``torch.compile`` (needs a collective).
    """
    if dim < 0:
        dim += tensor.ndim
    spec = tensor._spec
    if spec.mesh.ndim != 1:
        raise MissingShardPatch("index_select supports 1-D device meshes only")
    placement = spec.placements[0]
    if placement.is_partial():
        raise MissingShardPatch(
            "index_select on a Partial ShardTensor is not supported"
        )
    _check_index_dtype(index)
    index_sharded = _index_is_sharded(index)

    if placement.is_shard() and placement.dim == dim:
        return routed_gather(tensor, index, dim)

    if placement.is_shard():
        if index_sharded:
            if _is_tracing((tensor, index)):
                raise MissingShardPatch(
                    "index_select off the sharded dim with a sharded index needs an "
                    "all-gather of the index; not supported under torch.compile"
                )
            idx = index.full_tensor()
        else:
            idx = _local_index(index)
        local = tensor.to_local().index_select(dim, idx)
        shapes = tuple(
            tuple(s[:dim] + (idx.numel(),) + s[dim + 1 :])
            for s in spec.sharding_shapes(0)
        )
        return ShardTensor.from_local(
            local, spec.mesh, spec.placements, sharding_shapes={0: shapes}
        )

    # Replicated source.
    idx = _local_index(index)
    local = tensor.to_local().index_select(dim, idx)
    if index_sharded:
        g = list(spec.tensor_meta.shape)
        shapes = tuple(
            tuple(g[:dim] + [n] + g[dim + 1 :]) for n in _shard_sizes(index._spec, 0)
        )
        return ShardTensor.from_local(
            local, spec.mesh, (Shard(dim),), sharding_shapes={0: shapes}
        )
    return ShardTensor.from_local(local, spec.mesh, (Replicate(),), sharding_shapes={})


def index_select_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""``torch.index_select`` / ``Tensor.index_select`` handler.

    Accepts the positional and keyword spellings of ``(input, dim, index)``.

    Parameters
    ----------
    func : Callable
        The intercepted function.
    types : tuple[Any, ...]
        Types involved in the dispatch.
    args : tuple[Any, ...]
        Positional arguments.
    kwargs : dict[str, Any]
        Keyword arguments.

    Returns
    -------
    ShardTensor
        See :func:`sharded_index_select`.

    Raises
    ------
    MissingShardPatch
        On unexpected extra arguments.
    """
    kwargs = dict(kwargs or {})
    params = list(args)
    tensor = params.pop(0) if params else kwargs.pop("input")
    dim = params.pop(0) if params else kwargs.pop("dim")
    index = params.pop(0) if params else kwargs.pop("index")
    if params or kwargs:
        raise MissingShardPatch(
            f"unexpected index_select arguments: {params}, {kwargs}"
        )
    return sharded_index_select(tensor, dim, index)


ShardTensor.register_function_handler(torch.index_select, index_select_wrapper)
ShardTensor.register_function_handler(torch.Tensor.index_select, index_select_wrapper)


def _is_int_tensor_key(key: Any) -> bool:
    r"""Whether *key* is a single integer tensor (advanced indexing).

    Parameters
    ----------
    key : Any
        ``__getitem__`` key.

    Returns
    -------
    bool
        ``True`` for an ``int32`` / ``int64`` tensor of any subclass.
    """
    return isinstance(key, torch.Tensor) and key.dtype in _INDEX_DTYPES


def getitem_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    r"""``torch.Tensor.__getitem__`` on a ShardTensor.

    A single integer-tensor key on a ``Shard(0)`` source is advanced indexing
    along the sharded dim and takes the routed gather. Every other key (ints,
    slices, tuples, masks, other placements) follows the default route, which
    mirrors the tail of ``ShardTensor.__torch_function__``: straight to
    dispatch under tracing, the DTensor fallback otherwise.

    Parameters
    ----------
    func : Callable
        The intercepted function.
    types : tuple[Any, ...]
        Types involved in the dispatch.
    args : tuple[Any, ...]
        ``(tensor, key)``.
    kwargs : dict[str, Any]
        Keyword arguments (none expected).

    Returns
    -------
    Any
        The indexed result; a ShardTensor on the routed path.
    """
    if len(args) == 2 and not kwargs:
        values, key = args
        if (
            isinstance(values, ShardTensor)
            and values._spec.mesh.ndim == 1
            and values._spec.placements == (Shard(0),)
            and _is_int_tensor_key(key)
        ):
            return routed_gather(values, key, 0)
    if _is_tracing(args, kwargs):
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
    return _torch_function_fallback_via_dtensor(func, args, kwargs)


ShardTensor.register_function_handler(torch.Tensor.__getitem__, getitem_wrapper)


def sharded_select_helper(tensor: ShardTensor, dim: int, index: int) -> ShardTensor:
    r"""Perform a select operation on a ShardTensor.

    Parameters
    ----------
    tensor : ShardTensor
        Input tensor to select from.
    dim : int
        Dimension along which to select.
    index : int
        Index to select.

    Returns
    -------
    ShardTensor
        Output tensor with the selected slice.

    Raises
    ------
    MissingShardPatch
        If selection is along a sharded axis or partial placement is used.
    """

    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.

    input_spec = tensor._spec

    input_placements = input_spec.placements

    shards = [s for s in input_placements if isinstance(s, Shard)]

    # We are reducing tensor rank and returning one sharding per tensor:
    original_shape = list(input_spec.shape)

    if dim in [i.dim for i in shards]:
        raise MissingShardPatch(
            "No implementation for aten.select.int along sharding axis yet."
        )

    else:
        # We are reducing tensor rank:
        original_shape.pop(dim)
        output_stride = _stride_from_contiguous_shape_C_style(original_shape)

        # Need to create a new global meta:
        new_meta = TensorMeta(
            torch.Size(tuple(original_shape)),
            stride=output_stride,
            dtype=input_spec.tensor_meta.dtype,
        )
        # The placements get adjusted too
        new_placements = []
        for p in input_spec.placements:
            if p.is_replicate():
                new_placements.append(p)
            elif p.is_shard():
                if p.dim > dim:
                    new_placements.append(Shard(p.dim - 1))
                else:
                    new_placements.append(p)
            elif p.is_partial():
                raise MissingShardPatch(
                    "Partial placement not supported yet for select"
                )

        # We can directly compute the sizes from the input spec sharding sizes:
        # Since the constraint above prevents selecting along a sharded dimension,
        # we can be sure that none of these adjusted shapes will be sharded.
        output_shard_sizes = {}
        for mesh_dim, index_shard_sizes in input_spec.sharding_shapes().items():
            output_shard_sizes[mesh_dim] = []
            for local_chunk_size in index_shard_sizes:
                local_chunk_size_list = list(local_chunk_size)
                local_chunk_size_list.pop(dim)
                # Plain int tuples (never torch.Size) for _sharding_shapes.
                output_shard_sizes[mesh_dim].append(tuple(local_chunk_size_list))
            output_shard_sizes[mesh_dim] = tuple(output_shard_sizes[mesh_dim])

        output_spec = ShardTensorSpec(
            mesh=input_spec.mesh,
            placements=tuple(new_placements),
            tensor_meta=new_meta,
            _sharding_shapes=output_shard_sizes,
        )
        # Finally, actually perform the select:
        local_result = aten.select.int(tensor._local_tensor, dim, index)

        return ShardTensor(
            local_result,
            output_spec,
            requires_grad=False,  # This will get adjusted after the dispatcher
        )


def sharded_select_backward_helper(
    grad_output: ShardTensor, input_sizes: torch.Size, dim: int, index: int
) -> ShardTensor:
    r"""Perform gradient computation for a select operation on a ShardTensor.

    We shard the gradients analogously to the output gradients.

    Parameters
    ----------
    grad_output : ShardTensor
        Gradient of the loss with respect to the output of the select operation.
    input_sizes : torch.Size
        Original input tensor sizes.
    dim : int
        Dimension along which the select was performed.
    index : int
        Index that was selected.

    Returns
    -------
    ShardTensor
        Gradient with respect to the input tensor.

    Raises
    ------
    Exception
        If partial placement is used (not supported).
    """

    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.

    input_placements = grad_output._spec.placements

    output_stride = _stride_from_contiguous_shape_C_style(input_sizes)

    # Need to create a new global meta:
    new_meta = TensorMeta(
        torch.Size(tuple(input_sizes)),
        stride=output_stride,
        dtype=grad_output._spec.tensor_meta.dtype,
    )

    new_placements = input_placements
    # The placements get adjusted too
    new_placements = []
    for p in grad_output._spec.placements:
        if p.is_replicate():
            new_placements.append(p)
        elif p.is_shard():
            if p.dim >= dim:
                new_placements.append(Shard(p.dim + 1))
            else:
                new_placements.append(p)
        elif p.is_partial():
            raise Exception("Partial placement not supported yet for select_backward")

    # Next, calculate the sharding sizes for the output tensor:
    output_shard_sizes = {}
    for mesh_dim, index_shard_sizes in grad_output._spec.sharding_shapes().items():
        output_shard_sizes[mesh_dim] = []
        for local_chunk_size in index_shard_sizes:
            # We need to insert input_sizes[dim] at index:
            local_chunk_size_list = list(local_chunk_size)
            local_chunk_size_list.insert(dim, input_sizes[dim])
            # Plain int tuples (never torch.Size) for _sharding_shapes.
            output_shard_sizes[mesh_dim].append(tuple(local_chunk_size_list))
        output_shard_sizes[mesh_dim] = tuple(output_shard_sizes[mesh_dim])

    output_spec = ShardTensorSpec(
        mesh=grad_output._spec.mesh,
        placements=tuple(new_placements),
        tensor_meta=new_meta,
        _sharding_shapes=output_shard_sizes,
    )

    # Finally, make sure we use the correct local size:
    mesh_rank = grad_output._spec.mesh.get_local_rank()
    if len(output_shard_sizes.keys()) > 0:
        local_output_size = output_shard_sizes[0][mesh_rank]
    else:
        # Fall back to the global shape if nothing is sharded:
        local_output_size = output_spec.tensor_meta.shape

    # Now, compute the local result:
    local_result = aten.select_backward(
        grad_output._local_tensor, local_output_size, dim, index
    )

    return ShardTensor(
        local_result,
        output_spec,
        requires_grad=False,  # This will get adjusted after the dispatcher
    )


ShardTensor.register_dispatch_handler(torch.ops.aten.select.int, sharded_select_helper)
ShardTensor.register_dispatch_handler(
    torch.ops.aten.select_backward.default, sharded_select_backward_helper
)
