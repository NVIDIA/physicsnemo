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

"""
Torch <-> Warp host glue for the ``sparse_fma_e2e`` radius-search backend.

This backend hashes *only occupied* cells (no dense per-batch grid, no
``total_cells``, no Morton sort, no query sort/gather, no host ``.item()``).
The build is count -> exclusive-scan -> scatter; the scan is the three-phase
async tile scan from ``_morton_dense_kernels.py`` (``wp.utils.array_scan``
injects a device sync, which this backend must avoid). Search reuses the
one-warp-per-query coalesced FMA loop.

Assumptions (current experimental scope):
* CUDA-only.
* ``Np = B * P < 200000`` so ``H = next_pow2(2 * Np) <= 2**19 < SCAN_BLOCK**2``
  and the two-level tile scan never overflows.

Entry point: :func:`radius_search_sparse_fma_e2e`, matching the
``(indices, points, distances, num_neighbors)`` contract of the other
``max_points`` backends.
"""

from __future__ import annotations

import os

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from . import _morton_dense_kernels as mk
from . import _sparse_hash_kernels as sk
from ._morton_dense_impl import (
    _alloc_outputs,
    _empty_outputs,
    _wp_outputs,
)
from .utils import validate_inputs

# Load factor <= 0.5: hash table has at least 2x as many slots as points.
_HASH_LOAD_FACTOR_INV = 2


def _nvtx_push(label: str) -> None:
    """Push an NVTX range when ``PROFILE_RUN=1`` (matches the dense e2e path)."""
    if os.environ.get("PROFILE_RUN", "0") == "1":
        torch.cuda.nvtx.range_push(label)


def _nvtx_pop() -> None:
    """Pop an NVTX range when ``PROFILE_RUN=1``."""
    if os.environ.get("PROFILE_RUN", "0") == "1":
        torch.cuda.nvtx.range_pop()


def _dbg(tag: str, tensor=None) -> None:
    """Sync + print a build-phase checkpoint when ``PHYSICSNEMO_SPARSE_DEBUG=1``.

    Temporary diagnostic to localise a hang/OOB to a specific kernel: the sync
    forces the just-launched kernel to complete before the next print, so the
    last tag printed names the stage that is stuck.
    """
    if os.environ.get("PHYSICSNEMO_SPARSE_DEBUG", "0") != "1":
        return
    import warp as _wp

    _wp.synchronize()
    msg = f"[sparse_fma_e2e] {tag}"
    if tensor is not None:
        t = tensor.detach()
        msg += (
            f"  shape={tuple(t.shape)} min={int(t.min())} "
            f"max={int(t.max())} sum={int(t.sum())}"
        )
    print(msg, flush=True)


def _next_power_of_two(n: int) -> int:
    """Smallest power of two >= max(1, n)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _warp_scan_exclusive(
    counts_wp,
    out_wp,
    n: int,
    device,
    wp_device,
    wp_stream,
) -> None:
    """Async exclusive prefix scan of ``counts_wp[:n]`` into ``out_wp[:n]``.

    Recursive, tile-free (no ``cudaDeviceSynchronize``, no CUB scratch, all
    launches on ``wp_stream``). Uses only scalar array ops -- ``wp.tile_scan_*``
    was observed to scan only a fraction of a block on this Warp build:

      1) sparse_block_reduce      block_sums[b] = sum(arr[block b])
      2) recurse                  exclusive-scan block_sums -> block_offsets
      3) sparse_block_scan_write  exclusive prefix within block b, seeded by
                                  block_offsets[b]

    Each level divides the length by ``SCAN_TILE`` (256), so any ``n`` is
    handled with O(log_256 n) levels (H+1 for Np < 200k needs two).
    """
    tile = int(sk.SCAN_TILE)
    num_blocks = (n + tile - 1) // tile

    block_sums = torch.zeros(num_blocks, dtype=torch.int32, device=device)
    block_offsets = torch.zeros(num_blocks, dtype=torch.int32, device=device)
    block_sums_wp = wp.from_torch(block_sums)
    block_offsets_wp = wp.from_torch(block_offsets)

    # 1) Per-block totals.
    wp.launch(
        sk.sparse_block_reduce,
        dim=num_blocks,
        inputs=[counts_wp, wp.int32(n), block_sums_wp],
        device=wp_device,
        stream=wp_stream,
    )

    # 2) Exclusive scan of the per-block totals -> block_offsets. Base case
    #    (num_blocks == 1) needs no scan: the single block's offset is 0.
    if num_blocks > 1:
        _warp_scan_exclusive(
            block_sums_wp, block_offsets_wp, num_blocks, device, wp_device, wp_stream
        )

    # 3) Write exclusive prefixes within each block, seeded by block_offsets.
    wp.launch(
        sk.sparse_block_scan_write,
        dim=num_blocks,
        inputs=[counts_wp, block_offsets_wp, out_wp, wp.int32(n)],
        device=wp_device,
        stream=wp_stream,
    )


def radius_search_sparse_fma_e2e(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse-hash-to-CSR radius search (CUDA-only, ``max_points`` path).

    Args:
        points: Reference points, ``(N, 3)`` or ``(B, N, 3)``.
        queries: Query points, ``(M, 3)`` or ``(B, M, 3)``.
        radius: Search radius.
        max_points: Maximum neighbours returned per query.
        return_dists: Whether to return neighbour distances.
        return_points: Whether to return neighbour coordinates.

    Returns:
        ``(indices, points, distances, num_neighbors)`` -- the shared
        ``max_points`` contract, with the batch dim squeezed for unbatched
        inputs and float outputs cast back to the input dtype.
    """
    input_dtype = points.dtype

    _nvtx_push("RADIUS SEARCH SPARSE FMA E2E")

    points, queries, was_unbatched = validate_inputs(points, queries)

    if points.dtype != torch.float32:
        points = points.to(torch.float32)
    if queries.dtype != torch.float32:
        queries = queries.to(torch.float32)
    _nvtx_push("points contiguous")
    points = points.contiguous()
    queries = queries.contiguous()
    _nvtx_pop()

    B, P, _ = points.shape
    Q = queries.shape[1]
    device = points.device

    if P == 0 or Q == 0:
        _nvtx_pop()  # RADIUS SEARCH SPARSE FMA E2E
        return _empty_outputs(
            B, Q, max_points, return_dists, return_points, was_unbatched,
            input_dtype, device,
        )

    Np = B * P
    Nq = B * Q
    H = _next_power_of_two(_HASH_LOAD_FACTOR_INV * Np)

    # Cell size strictly >= radius so the 27-neighbour stencil is exhaustive.
    cell_size = float(radius) * (1.0 + 1e-3)
    inv_cell = 1.0 / cell_size
    radius2 = float(radius) * float(radius)

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points)

    # -- Workspace (int32) -------------------------------------------------
    # cell_rep_plus1 doubles as the occupancy sentinel (flat+1) and is cleared
    # together with cell_count in a single zeroed allocation.
    _nvtx_push("alloc workspace")
    cell_rep_plus1 = torch.zeros(H, dtype=torch.int32, device=device)
    cell_count = torch.zeros(H + 1, dtype=torch.int32, device=device)  # [H]=0 pad
    cell_offsets = torch.empty(H + 1, dtype=torch.int32, device=device)
    cell_coords = torch.empty((H, 4), dtype=torch.int32, device=device)
    point_slot = torch.empty(Np, dtype=torch.int32, device=device)
    point_rank = torch.empty(Np, dtype=torch.int32, device=device)

    # Compact cell-contiguous point SoA.
    pc_x = torch.empty(Np, dtype=torch.float32, device=device)
    pc_y = torch.empty(Np, dtype=torch.float32, device=device)
    pc_z = torch.empty(Np, dtype=torch.float32, device=device)
    pc_orig = torch.empty(Np, dtype=torch.int32, device=device)

    out_idx, out_count, out_dist, out_pts = _alloc_outputs(
        B, Q, max_points, return_dists, return_points, device
    )

    # -- Warp views --------------------------------------------------------
    points_wp = wp.from_torch(points, dtype=wp.vec3)
    queries_wp = wp.from_torch(queries, dtype=wp.vec3)
    cell_rep_plus1_wp = wp.from_torch(cell_rep_plus1)
    cell_count_wp = wp.from_torch(cell_count)
    cell_coords_wp = wp.from_torch(cell_coords, dtype=wp.vec4i)
    point_slot_wp = wp.from_torch(point_slot)
    point_rank_wp = wp.from_torch(point_rank)
    pc_x_wp = wp.from_torch(pc_x)
    pc_y_wp = wp.from_torch(pc_y)
    pc_z_wp = wp.from_torch(pc_z)
    pc_orig_wp = wp.from_torch(pc_orig)
    cell_offsets_wp = wp.from_torch(cell_offsets)

    idx_wp, count_wp, dist_wp, pts_wp = _wp_outputs(
        out_idx, out_count, out_dist, out_pts, Nq, max_points
    )
    # _wp_outputs returns None for disabled optional outputs; the kernel still
    # needs valid arrays to bind, so wire zero-length placeholders.
    if dist_wp is None:
        dist_wp = wp.from_torch(
            torch.empty((Nq, 0), dtype=torch.float32, device=device)
        )
    if pts_wp is None:
        pts_wp = wp.from_torch(
            torch.empty((Nq, 0, 3), dtype=torch.float32, device=device),
            dtype=wp.vec3,
        )
    _nvtx_pop()  # alloc workspace

    
    
    _nvtx_push("SCOPED STREAM")
    with wp.ScopedStream(wp_stream, sync_enter=False, sync_exit=False):
        # 1) Count points per occupied cell + build the hash table.
        _nvtx_push("COUNT CELLS + BUILD HASH")
        wp.launch(
            sk.sparse_count_cells_kernel,
            dim=Np,
            inputs=[
                points_wp,
                wp.float32(inv_cell),
                wp.int32(P),
                wp.int32(H),
                cell_rep_plus1_wp,
                cell_count_wp,
                cell_coords_wp,
                point_slot_wp,
                point_rank_wp,
            ],
            device=wp_device,
            stream=wp_stream,
        )
        _nvtx_pop()  # COUNT CELLS + BUILD HASH
        # _dbg("after count: cell_count", cell_count)
        # _dbg("after count: point_slot", point_slot)

        # 2) Exclusive scan of per-cell counts -> exclusive cell starts.
        #    Two-level Warp tile scan: three kernel launches on wp_stream, no
        #    cudaDeviceSynchronize and no CUB scratch (torch.cumsum was confirmed
        #    to sync on this build). Scanning all H+1 entries (cell_count[H]==0)
        #    yields cell_offsets[H] == Np, so the cell_offsets[slot + 1] read for
        #    the last occupied slot is valid.
        _nvtx_push("Array Scan")
        # _dbg("before scan: cell_count", cell_count)
        # _warp_scan_exclusive(
        #     cell_count_wp, cell_offsets_wp, H + 1, device, wp_device, wp_stream
        # )
        wp.utils.array_scan(cell_count_wp[: H + 1], cell_offsets_wp[: H + 1], inclusive=False)

        _nvtx_pop()  # Array Scan
        # cell_offsets must be non-decreasing, offsets[0]==0, offsets[H]==Np.
        # _dbg("after scan: cell_offsets", cell_offsets)

        # 3) Scatter points into cell-contiguous slots. Each point's destination
        #    is cell_offsets[slot] + its precomputed intra-cell rank, so no write
        #    cursor and no cell_offsets clone are needed; cell_offsets stays the
        #    immutable CSR start array for search.
        _nvtx_push("Scatter Points to cells")
        wp.launch(
            sk.sparse_scatter_points_kernel,
            dim=Np,
            inputs=[
                points_wp,
                wp.int32(P),
                point_slot_wp,
                point_rank_wp,
                cell_offsets_wp,
                pc_x_wp,
                pc_y_wp,
                pc_z_wp,
                pc_orig_wp,
            ],
            device=wp_device,
            stream=wp_stream,
        )
        _nvtx_pop()  # Scatter Points to cells
        # _dbg("after scatter: pc_orig", pc_orig)

        # 4) One warp per query radius search (output padding fused in-kernel).
        # _dbg("before search (launching radius_search_sparse_fma_kernel)")
        _nvtx_push("RAD SEARCH KERNEL")
        wp.launch_tiled(
            sk.radius_search_sparse_fma_kernel,
            dim=Nq,
            block_dim=int(sk.FMA_WARP_SIZE),
            inputs=[
                queries_wp,
                wp.int32(Q),
                wp.float32(inv_cell),
                wp.float32(radius2),
                wp.int32(H),
                cell_rep_plus1_wp,
                cell_coords_wp,
                cell_offsets_wp,
                pc_x_wp,
                pc_y_wp,
                pc_z_wp,
                pc_orig_wp,
                wp.int32(max_points),
                return_dists,
                return_points,
                idx_wp,
                count_wp,
                dist_wp,
                pts_wp,
            ],
            device=wp_device,
            stream=wp_stream,
        )
        _nvtx_pop()  # RAD SEARCH KERNEL


    _nvtx_pop()
    
    num = out_count.view(B, Q)
    if out_pts is None:
        out_pts = torch.empty((0, max_points, 3), dtype=torch.float32, device=device)
    if out_dist is None:
        out_dist = torch.empty(0, dtype=torch.float32, device=device)

    if was_unbatched:
        out_idx = out_idx.squeeze(0)
        num = num.squeeze(0)
        if return_points:
            out_pts = out_pts.squeeze(0)
        if return_dists:
            out_dist = out_dist.squeeze(0)

    out_pts = out_pts.to(input_dtype)
    out_dist = out_dist.to(input_dtype)
    _nvtx_pop()  # RADIUS SEARCH SPARSE FMA E2E
    return out_idx, out_pts, out_dist, num
