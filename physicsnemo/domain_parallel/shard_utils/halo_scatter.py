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

r"""``torch.compile``-safe halo scatter-correction for a ``Shard(0)`` ShardTensor.

After a row scatter writes into borrowed-ghost rows of a ``[owned | ghost]`` layout,
this folds those contributions back into their owners and refreshes the ghost rows.
The correction is exposed as an AOT-traceable primitive (:func:`halo_scatter_correct`,
built from the :func:`halo_reverse_exchange` / :func:`halo_forward_exchange` halves)
with routing passed as a packed tensor so it survives graph breaks and per-step value
changes. Data movement is a pluggable backend chosen by :func:`select_halo_backend`;
:func:`register_halo_scatter_handlers` wires the correction onto ShardTensor scatter/add.
"""

from __future__ import annotations

import os
from typing import Protocol

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh

__all__ = [
    "funcol_all_to_all_v_rows",
    "halo_forward_exchange",
    "halo_reverse_exchange",
    "halo_scatter_correct",
    "pack_halo_routing",
    "register_halo_scatter_handlers",
    "select_halo_backend",
]


def _funcol_group_arg(group: object) -> object:
    r"""Return *group* in the form functional collectives accept (funcol rejects ``None``)."""
    if isinstance(group, DeviceMesh):
        return (group, 0)
    if group is None:
        return dist.distributed_c10d._get_default_group()
    return group


def _accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    r"""Higher-precision dtype for folding scatter contributions: ``float32`` accumulates
    in ``float64`` and reduced-precision (``float16``/``bfloat16``) in ``float32``; all
    other dtypes accumulate in place. Row folds sum many contributions, so accumulating in
    the input precision loses significance for the smaller float types."""
    if dtype == torch.float32:
        return torch.float64
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _halo_group_name(group: object) -> str:
    r"""Resolve *group* to a c10d group-name string (``""`` = default world group), the
    traceable token a ``custom_op`` can carry in place of a ``ProcessGroup``."""
    if group is None:
        return ""
    if isinstance(group, str):
        return group
    if isinstance(group, DeviceMesh):
        return group._dim_group_names[0]
    return group.group_name


def funcol_all_to_all_v_rows(
    send_rows: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: object = None,
) -> torch.Tensor:
    r"""AOT-traceable variable-sized row ``all_to_all`` via ``funcol.all_to_all_single``; send/recv buffers are destination/source-rank ordered."""
    trailing = tuple(send_rows.shape[1:])
    row_size = 1
    for d in trailing:
        row_size *= d
    flat_send = send_rows.contiguous().reshape(-1)
    send_flat = [c * row_size for c in send_counts]
    recv_flat = [c * row_size for c in recv_counts]
    total_recv = sum(recv_counts)
    flat_recv = funcol.wait_tensor(
        funcol.all_to_all_single(
            flat_send, recv_flat, send_flat, _funcol_group_arg(group)
        )
    )
    return flat_recv.reshape((total_recv,) + trailing)


# A transport backend owns the whole reverse/forward exchange, since the data-movement
# structure -- not just the collective call -- is transport-specific.


class _HaloBackend(Protocol):
    name: str

    def reverse(
        self,
        padded: torch.Tensor,
        n_owned: int,
        send_indices: list[torch.Tensor],
        send_sizes: list[list[int]],
        rank: int,
        world_size: int,
        group: object,
    ) -> torch.Tensor: ...

    def forward(
        self,
        owned: torch.Tensor,
        send_indices: list[torch.Tensor],
        send_sizes: list[list[int]],
        rank: int,
        world_size: int,
        group: object,
    ) -> torch.Tensor: ...


class _FuncolHaloBackend:
    r"""Functional-collective transport: dense ``all_to_all_single`` over the whole group; portable everywhere (incl. gloo/CPU) and the default fallback."""

    name = "funcol"

    def reverse(
        self, padded, n_owned, send_indices, send_sizes, rank, world_size, group
    ):
        r"""Fold ghost rows back into owners via a reverse row all-to-all-v."""
        ghost = padded[n_owned:].contiguous()

        # Send each ghost block back to the owner it was borrowed from.
        rev_indices: list[torch.Tensor] = []
        offset = 0
        for r in range(world_size):
            n = int(send_sizes[r][rank])
            rev_indices.append(
                torch.arange(
                    offset, offset + n, device=padded.device, dtype=torch.int64
                )
            )
            offset += n
        send_rows = torch.cat(
            [ghost.index_select(0, rev_indices[j]) for j in range(world_size)], dim=0
        )
        send_counts = [int(send_sizes[r][rank]) for r in range(world_size)]
        recv_counts = [int(send_sizes[rank][j]) for j in range(world_size)]
        received_back = funcol_all_to_all_v_rows(
            send_rows, send_counts, recv_counts, group
        )

        # Fold returned contributions into the lent owned rows in higher precision.
        acc_dtype = _accumulator_dtype(padded.dtype)
        owned = padded[:n_owned].to(acc_dtype)
        offset = 0
        for j in range(world_size):
            n = int(send_sizes[rank][j])
            if n == 0:
                continue
            owned = owned.index_add(
                0, send_indices[j], received_back[offset : offset + n].to(acc_dtype)
            )
            offset += n
        return owned.to(padded.dtype)

    def forward(self, owned, send_indices, send_sizes, rank, world_size, group):
        r"""Refresh ghost rows from owners via a forward row all-to-all-v."""
        send_rows = torch.cat(
            [owned.index_select(0, send_indices[j]) for j in range(world_size)], dim=0
        )
        send_counts = [int(send_sizes[rank][j]) for j in range(world_size)]
        recv_counts = [int(send_sizes[i][rank]) for i in range(world_size)]
        ghost_new = funcol_all_to_all_v_rows(send_rows, send_counts, recv_counts, group)
        return torch.cat([owned, ghost_new], dim=0)


def _symm_group_name(group: object) -> str:
    r"""Resolve *group* to a c10d group-name string for ``get_symm_mem_workspace``; ``None`` resolves to the *named* default world group, not ``""``."""
    if group is None:
        return dist.distributed_c10d._get_default_group().group_name
    if isinstance(group, str):
        return group
    if isinstance(group, DeviceMesh):
        return group._dim_group_names[0]
    return group.group_name


def _global_max_staged_rows(send_sizes: list[list[int]], world_size: int) -> int:
    r"""Rows the symmetric workspace must hold on every rank: the group-wide max of ``max(ghost rows, lent rows)``, so the symmetric allocation stays uniform."""
    m = 0
    for r in range(world_size):
        ghost = sum(int(send_sizes[i][r]) for i in range(world_size))
        lent = sum(int(send_sizes[r][j]) for j in range(world_size))
        m = max(m, ghost, lent)
    return m


def _require_symm_mem(tensor: torch.Tensor):
    r"""Return the ``_symmetric_memory`` module, or raise when it cannot serve *tensor* (CPU, or torch without it)."""
    if not tensor.is_cuda:
        raise RuntimeError(
            "the symmetric-memory halo backend requires CUDA tensors; "
            "set PHYSICSNEMO_HALO_BACKEND=funcol for CPU/gloo."
        )
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as exc:  # pragma: no cover - torch build without symm-mem
        raise RuntimeError(
            "torch.distributed._symmetric_memory is unavailable; "
            "set PHYSICSNEMO_HALO_BACKEND=funcol."
        ) from exc
    return symm_mem


# Per-neighbour readiness/completion fence channels; reverse and forward use disjoint
# channels so their signals are never confused.
_REV_READY, _REV_DONE, _FWD_READY, _FWD_DONE = 0, 1, 2, 3


class _SymmMemHaloBackend:
    r"""Intra-node symmetric-memory one-sided transport (CUDA-IPC ``get_buffer``).

    Each rank stages its exchange block into a symmetric workspace and *pulls* each
    neighbour's block, moving only real ghost/lent data. ``put_signal`` / ``wait_signal``
    fence each exchange peer-to-peer instead of a group-wide ``barrier``, so a rank
    synchronizes with O(neighbours) peers. Numerically identical to
    :class:`_FuncolHaloBackend`, whose dense destination-ordered layout the offsets mirror.
    """

    name = "symm_mem"

    @staticmethod
    def _row_numel(feat_shape: tuple[int, ...]) -> int:
        n = 1
        for d in feat_shape:
            n *= int(d)
        return n

    def reverse(
        self, padded, n_owned, send_indices, send_sizes, rank, world_size, group
    ):
        r"""Fold ghost rows back into owners via a one-sided reverse exchange."""
        symm_mem = _require_symm_mem(padded)
        feat = tuple(padded.shape[1:])
        row_numel = self._row_numel(feat)
        dtype = padded.dtype
        group_name = _symm_group_name(group)
        max_rows = _global_max_staged_rows(send_sizes, world_size)

        # Reverse sends ghost contributions back to owners: readers pull from my buffer
        # (peers I borrowed from); sources are the peers I pull from (peers I lent to).
        readers = [s for s in range(world_size) if int(send_sizes[s][rank]) > 0]
        sources = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]

        acc_dtype = _accumulator_dtype(dtype)
        owned = padded[:n_owned].to(acc_dtype)
        with torch.cuda.device(padded.device):
            handle = symm_mem.get_symm_mem_workspace(
                group_name, max(1, max_rows * row_numel * padded.element_size())
            )
            # Stage the whole ghost region, destination-ordered, and signal each reader.
            ghost = padded[n_owned:].contiguous()
            if ghost.shape[0]:
                handle.get_buffer(rank, tuple(ghost.shape), dtype).copy_(ghost)
            for r in readers:
                handle.put_signal(r, channel=_REV_READY)
            # Pull each lent-to peer's contributions, fold them in, and release its buffer.
            for j in sources:
                n = int(send_sizes[rank][j])
                off = sum(int(send_sizes[d][j]) for d in range(rank)) * row_numel
                handle.wait_signal(j, channel=_REV_READY)
                recv = handle.get_buffer(j, (n, *feat), dtype, storage_offset=off)
                owned = owned.index_add(0, send_indices[j], recv.to(acc_dtype))
                handle.put_signal(j, channel=_REV_DONE)
            # Hold until every reader has finished pulling before the buffer is reused.
            for r in readers:
                handle.wait_signal(r, channel=_REV_DONE)
        return owned.to(dtype)

    def forward(self, owned, send_indices, send_sizes, rank, world_size, group):
        r"""Refresh ghost rows from the corrected owners via a one-sided exchange."""
        symm_mem = _require_symm_mem(owned)
        feat = tuple(owned.shape[1:])
        row_numel = self._row_numel(feat)
        dtype = owned.dtype
        group_name = _symm_group_name(group)
        max_rows = _global_max_staged_rows(send_sizes, world_size)

        # Forward broadcasts owners to ghosts, so roles swap: readers are peers I lent
        # to; sources are peers I borrowed from.
        readers = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]
        sources = [i for i in range(world_size) if int(send_sizes[i][rank]) > 0]

        with torch.cuda.device(owned.device):
            handle = symm_mem.get_symm_mem_workspace(
                group_name, max(1, max_rows * row_numel * owned.element_size())
            )
            # Stage the rows lent to each peer, destination-ordered, and signal readers.
            send_rows = torch.cat(
                [owned.index_select(0, send_indices[j]) for j in range(world_size)],
                dim=0,
            )
            if send_rows.shape[0]:
                handle.get_buffer(rank, tuple(send_rows.shape), dtype).copy_(send_rows)
            for r in readers:
                handle.put_signal(r, channel=_FWD_READY)
            # Pull each refreshed ghost block from its owner and release its buffer.
            ghost_blocks = {}
            for i in sources:
                n = int(send_sizes[i][rank])
                off = sum(int(send_sizes[i][d]) for d in range(rank)) * row_numel
                handle.wait_signal(i, channel=_FWD_READY)
                gb = handle.get_buffer(i, (n, *feat), dtype, storage_offset=off)
                ghost_blocks[i] = gb.clone()
                handle.put_signal(i, channel=_FWD_DONE)
            for r in readers:
                handle.wait_signal(r, channel=_FWD_DONE)
        if not ghost_blocks:
            return owned
        ghost_new = torch.cat([ghost_blocks[i] for i in sources], dim=0)
        return torch.cat([owned, ghost_new], dim=0)


_FUNCOL_BACKEND = _FuncolHaloBackend()
_SYMM_MEM_BACKEND = _SymmMemHaloBackend()


_symm_capability_cache: dict[str, bool] = {}
_group_multinode_cache: dict[str, bool] = {}


def _resolve_pg(group: object):
    r"""Resolve *group* to a ``ProcessGroup`` (``None`` if it cannot be)."""
    if group is None:
        return dist.distributed_c10d._get_default_group()
    if isinstance(group, DeviceMesh):
        try:
            return group.get_group() if group.ndim == 1 else group.get_group(0)
        except Exception:
            return None
    if isinstance(group, str):
        try:
            return dist.distributed_c10d._resolve_process_group(group)
        except Exception:
            return None
    return group


def _check_symm_mem_ipc(symm_mem, group_name: str) -> bool:
    r"""Collective check that a symmetric workspace can be rendezvoused for *group_name*
    (CUDA-IPC / P2P available); every rank must call it together."""
    try:
        with torch.cuda.device(torch.cuda.current_device()):
            symm_mem.get_symm_mem_workspace(group_name, 1024)
        return True
    except Exception:  # pragma: no cover - runs only on real multi-GPU hardware
        return False


def _symm_mem_usable(group: object) -> bool:
    r"""Whether the symmetric-memory transport is auto-selectable for *group*: gated on a
    NCCL group plus a cached, per-group symmetric-workspace rendezvous check (which fails
    cleanly cross-node, where funcol is the always-correct fallback)."""
    if not torch.cuda.is_available():
        return False
    try:
        import torch.distributed as _dist
        import torch.distributed._symmetric_memory as symm_mem
    except Exception:  # pragma: no cover - torch build without symm-mem
        return False
    if not (_dist.is_available() and _dist.is_initialized()):
        return False
    # Substring test: a CUDA process registers a mixed backend like "cpu:gloo,cuda:nccl".
    pg = _resolve_pg(group)
    if pg is None:
        return False
    try:
        if "nccl" not in str(_dist.get_backend(pg)).lower():
            return False
    except Exception:
        return False
    group_name = _symm_group_name(group)
    cached = _symm_capability_cache.get(group_name)
    if cached is None:
        cached = _check_symm_mem_ipc(symm_mem, group_name)
        _symm_capability_cache[group_name] = cached
    return cached


def _group_is_multinode(group: object) -> bool:
    r"""Whether *group* spans more than one physical node, via a cached hostname all-gather; falls back to a world-size-vs-local-GPU-count heuristic if the gather cannot run."""
    import socket

    if not (dist.is_available() and dist.is_initialized()):
        return False
    pg = _resolve_pg(group)
    if pg is None:
        return False
    group_name = _symm_group_name(group)
    cached = _group_multinode_cache.get(group_name)
    if cached is None:
        # All ranks run the same collective and derive the same answer; a failure raises
        # (identically on every rank) rather than being swallowed, since a divergent
        # backend choice between ranks would deadlock the subsequent halo collectives.
        world = dist.get_world_size(pg)
        gathered: list[object] = [None] * world
        dist.all_gather_object(gathered, socket.gethostname(), group=pg)
        cached = len({str(h) for h in gathered}) > 1
        _group_multinode_cache[group_name] = cached
    return cached


def select_halo_backend(group: object = None, is_cuda: bool = True) -> _HaloBackend:
    r"""Return the halo transport backend for *group*, honouring
    ``PHYSICSNEMO_HALO_BACKEND`` and otherwise picking intra-node symm-mem when usable
    (funcol is the always-correct fallback; CPU work always routes to funcol)."""
    forced = os.environ.get("PHYSICSNEMO_HALO_BACKEND")
    if forced == "funcol":
        return _FUNCOL_BACKEND
    if forced == "symm_mem":
        return _SYMM_MEM_BACKEND
    if forced:
        raise ValueError(
            f"PHYSICSNEMO_HALO_BACKEND={forced!r} is not a known halo backend "
            "(expected 'funcol' or 'symm_mem')."
        )
    # CPU work can only use funcol (symm-mem backends are CUDA-only).
    if not is_cuda:
        return _FUNCOL_BACKEND
    # Only a single-node group can use symm-mem; multi-node routes to funcol, since the
    # CUDA-IPC symmetric workspace cannot rendezvous across nodes.
    if not _group_is_multinode(group) and _symm_mem_usable(group):
        return _SYMM_MEM_BACKEND
    return _FUNCOL_BACKEND


def halo_reverse_exchange(
    padded: torch.Tensor,
    n_owned: int,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    rank: int,
    world_size: int,
    group: object = None,
) -> torch.Tensor:
    r"""Fold borrowed ghost rows of a ``[owned | ghost]`` tensor back into their owners
    (transpose of the forward halo gather), returning the ``(n_owned, *F)`` owned block."""
    return select_halo_backend(group, padded.is_cuda).reverse(
        padded, n_owned, send_indices, send_sizes, rank, world_size, group
    )


def halo_forward_exchange(
    owned: torch.Tensor,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    rank: int,
    world_size: int,
    group: object = None,
) -> torch.Tensor:
    r"""Refresh ghost rows from the owners and append them, returning the
    ``[owned | ghost]`` layout (inverse of :func:`halo_reverse_exchange`)."""
    return select_halo_backend(group, owned.is_cuda).forward(
        owned, send_indices, send_sizes, rank, world_size, group
    )


def _scatter_correct_dense(
    padded: torch.Tensor,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    n_owned: int,
    rank: int,
    world_size: int,
    group: object,
) -> torch.Tensor:
    r"""``forward(reverse(padded))`` over *group* using a single selected backend."""
    backend = select_halo_backend(group, padded.is_cuda)
    owned = backend.reverse(
        padded, n_owned, send_indices, send_sizes, rank, world_size, group
    )
    return backend.forward(owned, send_indices, send_sizes, rank, world_size, group)


def pack_halo_routing(
    send_indices: list[list[int]] | list[torch.Tensor],
    send_sizes: list[list[int]],
    n_owned: int,
    rank: int,
    world_size: int,
    device: object = None,
    cap: int | None = None,
) -> torch.Tensor:
    r"""Pack halo routing into a 1-D int64 tensor for :func:`halo_scatter_correct`.

    The packed tensor rides as a graph input, so its values may change across steps and
    survive Dynamo graph breaks without recompiling. Layout is ``[world_size, n_owned,
    rank, n_flat, *send_sizes, *send_idx_lens, *send_idx_flat]``. If *cap* is given, the
    trailing index section is padded to a fixed length so the routing keeps a constant
    shape across steps; *cap* must be ``>=`` this rank's total lent-row indices. Index
    arrays stay as tensors throughout, so the pack triggers no value-dependent device sync.
    """
    idx_tensors = [
        idx.reshape(-1).to(torch.int64)
        if isinstance(idx, torch.Tensor)
        else torch.tensor(idx, dtype=torch.int64)
        for idx in send_indices
    ]
    lens = [int(t.numel()) for t in idx_tensors]  # shapes only -- no value sync
    ss = [int(send_sizes[i][j]) for i in range(world_size) for j in range(world_size)]
    if device is None and idx_tensors:
        device = idx_tensors[0].device
    header = torch.tensor(
        [world_size, n_owned, rank, sum(lens), *ss, *lens],
        dtype=torch.int64,
        device=device,
    )
    idx_flat = (
        torch.cat([t.to(device) for t in idx_tensors])
        if idx_tensors
        else torch.zeros(0, dtype=torch.int64, device=device)
    )
    if cap is not None:
        total = int(sum(lens))
        if total > cap:
            raise ValueError(
                f"pack_halo_routing: cap={cap} is smaller than the number of lent-row "
                f"indices this rank holds ({total}); set cap to the per-rank maximum."
            )
        if idx_flat.numel() < cap:
            idx_flat = torch.cat(
                [
                    idx_flat,
                    torch.zeros(
                        cap - idx_flat.numel(), dtype=torch.int64, device=device
                    ),
                ]
            )
    elif not idx_tensors:
        return header
    return torch.cat([header, idx_flat])


def _unpack_halo_routing(routing: torch.Tensor):
    r"""Inverse of :func:`pack_halo_routing`; materializes only the small header to host and keeps index arrays as on-device views, so trailing ``cap`` padding is never read."""
    world_size, n_owned, rank, _n_flat = routing[:4].tolist()
    body_len = world_size * world_size + world_size
    body = routing[4 : 4 + body_len].tolist()
    ss, lens = body[: world_size * world_size], body[world_size * world_size :]
    send_sizes = [
        [ss[i * world_size + j] for j in range(world_size)] for i in range(world_size)
    ]
    send_indices, o = [], 4 + body_len
    for length in lens:
        send_indices.append(routing[o : o + length])  # device view -- no sync
        o += length
    return send_indices, send_sizes, n_owned, rank, world_size


@torch.library.custom_op("physicsnemo::halo_scatter_correct", mutates_args=())
def _halo_scatter_correct_op(
    padded: torch.Tensor, routing: torch.Tensor, group_name: str
) -> torch.Tensor:
    r"""Dispatcher-visible ``forward(reverse(padded))``, opaque to fake mode; the body runs only at runtime (unpacking ``routing`` there) over the group named ``group_name`` (``""`` = default world group)."""
    group = group_name or None
    send_indices, send_sizes, n_owned, rank, world_size = _unpack_halo_routing(routing)
    return _scatter_correct_dense(
        padded, send_indices, send_sizes, n_owned, rank, world_size, group
    )


@_halo_scatter_correct_op.register_fake
def _halo_scatter_correct_fake(padded, routing, group_name):
    return torch.empty_like(padded)


def _halo_correct_setup_context(ctx, inputs, output):
    _padded, routing, group_name = inputs
    ctx.routing = routing
    ctx.group_name = group_name


def _halo_correct_backward(ctx, grad):
    # forward(reverse(.)) is self-adjoint, so the VJP is the op applied to grad.
    grad_in = _halo_scatter_correct_op(grad.contiguous(), ctx.routing, ctx.group_name)
    return grad_in, None, None


_halo_scatter_correct_op.register_autograd(
    _halo_correct_backward, setup_context=_halo_correct_setup_context
)


def halo_scatter_correct(
    padded: torch.Tensor,
    routing: torch.Tensor,
    group: object = None,
) -> torch.Tensor:
    r"""``torch.compile``-safe halo scatter-correction: fold borrowed-ghost
    contributions into their owners and refresh the ghost rows as a single
    AOT-traceable, differentiable graph node, with ``routing`` (from
    :func:`pack_halo_routing`) riding as a graph input to survive graph breaks."""
    return _halo_scatter_correct_op(padded, routing, _halo_group_name(group))


# ShardTensor scatter/index-add integration. A ShardTensor carrying packed routing as an
# inner tensor (``_halo_meta_packed``) gets its scatter_add / index_add corrected via a
# ``__torch_function__`` handler (chosen over ``__torch_dispatch__`` so the correction stays
# in the compiled backward). Tensors without routing fall through, so registering is opt-in.


def register_halo_scatter_handlers() -> None:
    r"""Register idempotent, opt-in halo-correction handlers for ``scatter_add`` /
    ``index_add`` (and their in-place forms) on ``ShardTensor``; only tensors carrying a
    non-empty ``_halo_meta_packed`` routing are corrected, all others fall through."""
    from torch.distributed.tensor import DTensor

    from physicsnemo.domain_parallel.shard_tensor import (
        ShardTensor,
        _torch_function_fallback_via_dtensor,
    )
    from physicsnemo.domain_parallel.shard_utils.patch_core import MissingShardPatch

    def _assert_single_mesh_dim(spec) -> None:
        # The routing (owner/ghost row indices) is 1-D; a multi-dim mesh is unsupported and
        # must fail loudly here rather than mis-route silently.
        if spec.mesh.ndim != 1:
            raise MissingShardPatch(
                "halo scatter-correction supports sharding over a single mesh dimension "
                f"only; got a {spec.mesh.ndim}-D mesh."
            )

    def _local(x):
        # ShardTensor args (the accumulator ``self``) return the raw ``_local_tensor``: the
        # correction's backward is supplied by the ``halo_scatter_correct`` custom op and
        # the ``_WrapLocalAsShard`` wrapper below, so ``to_local()`` here would splice a
        # redundant autograd node. ``index`` / ``src`` instead arrive promoted to a
        # Replicate DTensor (the base ``_promote_plain_handler_args`` runs first), and must
        # use the differentiable ``to_local()`` -- the raw ``_local_tensor`` would sever the
        # ``from_local`` bridge and drop the gradient back to the plain source.
        if isinstance(x, ShardTensor):
            return x._local_tensor
        if isinstance(x, DTensor):
            return x.to_local()
        return x

    def _routing(self):
        r = getattr(self, "_halo_meta_packed", None)
        return r if (r is not None and r.numel() > 0) else None

    def _needs_grad(*tensors):
        return torch.is_grad_enabled() and any(
            bool(getattr(t, "requires_grad", False))
            or getattr(t, "grad_fn", None) is not None
            for t in tensors
        )

    def _build(src_type, local, spec, routing, requires_grad):
        out = src_type.__new__(
            src_type, local_tensor=local, spec=spec, requires_grad=requires_grad
        )
        out._halo_meta_packed = routing
        return out

    class _WrapLocalAsShard(torch.autograd.Function):
        # Attach a grad_fn to the wrapper so the tangent flows wrapper -> local -> the
        # halo/scatter graph; a bare ``__new__`` wrapper is an autograd leaf and would
        # drop the correction's backward. Mirrors ``_FromTorchTensor``.
        @staticmethod
        def forward(ctx, local, src_type, spec, routing):
            return _build(src_type, local, spec, routing, local.requires_grad)

        @staticmethod
        def backward(ctx, grad_out):
            g = (
                grad_out._local_tensor
                if isinstance(grad_out, ShardTensor)
                else grad_out
            )
            return g, None, None, None

    def _wrap_like(src, local, routing, requires_grad):
        if requires_grad:
            return _WrapLocalAsShard.apply(local, type(src), src._spec, routing)
        return _build(type(src), local, src._spec, routing, False)

    def _apply_scatter(oop, local_self, dim, local_index, local_src, kwargs):
        # ``index_add`` takes an ``alpha`` scale that ``scatter_add`` lacks; thread it
        # through so a non-default ``alpha`` is honoured instead of silently dropped.
        if oop is torch.Tensor.index_add and "alpha" in kwargs:
            return oop(local_self, dim, local_index, local_src, alpha=kwargs["alpha"])
        return oop(local_self, dim, local_index, local_src)

    def _scatter_handler(f, types, args, kwargs):
        self = args[0]
        routing = _routing(self)
        if routing is None:
            return _torch_function_fallback_via_dtensor(f, args, kwargs)
        _assert_single_mesh_dim(self._spec)
        dim, index, src = args[1], args[2], args[3]
        local_result = _apply_scatter(
            f, _local(self), dim, _local(index), _local(src), kwargs
        )
        corrected = halo_scatter_correct(local_result, routing, group=self._spec.mesh)
        return _wrap_like(self, corrected, routing, _needs_grad(self, index, src))

    # In-place forms map to out-of-place so the correction's backward chains cleanly (an
    # in-place op on the inner tensor is dropped from the wrapper-subclass backward).
    _inplace_to_oop = {
        torch.Tensor.scatter_add_: torch.Tensor.scatter_add,
        torch.Tensor.index_add_: torch.Tensor.index_add,
    }

    def _scatter_inplace_handler(f, types, args, kwargs):
        self = args[0]
        routing = _routing(self)
        if routing is None:
            return _torch_function_fallback_via_dtensor(f, args, kwargs)
        _assert_single_mesh_dim(self._spec)
        dim, index, src = args[1], args[2], args[3]
        local_result = _apply_scatter(
            _inplace_to_oop[f], _local(self), dim, _local(index), _local(src), kwargs
        )
        corrected = halo_scatter_correct(local_result, routing, group=self._spec.mesh)
        out = _wrap_like(self, corrected, routing, _needs_grad(self, index, src))
        # Write corrected values into local storage so a caller reusing the accumulator sees
        # them; the detached no_grad copy avoids recording an in-place op on a leaf.
        with torch.no_grad():
            self._local_tensor.detach().copy_(corrected)
        return out

    for func in (torch.Tensor.scatter_add, torch.Tensor.index_add):
        ShardTensor.register_function_handler(func, _scatter_handler)
    for func in (torch.Tensor.scatter_add_, torch.Tensor.index_add_):
        ShardTensor.register_function_handler(func, _scatter_inplace_handler)
