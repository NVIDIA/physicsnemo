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
Torch <-> Warp host glue for the Morton-sorted compact dense-cell radius search.

This backend replaces the open-addressed cell hash of ``_morton_impl.py`` with a
dense, directly-indexed cell table: points are mapped to radius-sized cells laid
out in a per-batch row-major grid, the occupied bins are compacted in Morton-cell
order for locality, and a ``(begin, count)`` table indexed by dense row id gives
O(1) cell lookup at search time (no hash, no binary search).

Two search paths share the same index:

* ``dense_fma``  -- :func:`radius_search_morton_warp_fma`, one warp per query.
* ``dense_gemm`` -- :func:`radius_search_morton_tiled_gemm`, a tiled-matmul
  benchmark path.

:func:`radius_search_morton_dense` is the single entry point used by the dispatch
in ``_warp_impl.py``. It mirrors the ``max_points`` return contract of
``radius_search_impl``: ``(indices, points, distances, num_neighbors)`` with the
batch dimension squeezed for unbatched inputs and outputs cast back to the input
dtype. This backend is CUDA-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import warp as wp
import os
from physicsnemo.core.function_spec import FunctionSpec

from . import _morton_dense_gemm_kernels as mgk
from . import _morton_dense_kernels as mk
from .utils import validate_inputs

TILE_Q = mk.TILE_Q
TILE_P = mk.TILE_P
FMA_BLOCK_DIM = mk.FMA_BLOCK_DIM
MEM_OPT_2_BLOCK_DIM = mk.MEM_OPT_2_BLOCK_DIM

FMA_V2_BLOCK_DIM = mk.FMA_V2_BLOCK_DIM

DENSE_VARIANTS = (
    "dense_fma",
    "dense_fma_e2e",
    "dense_fma_store_opt",
    "dense_fma_mem_opt",
    "dense_fma_mm",
    "dense_gemm",
    # WIP experimental cell-centric mem-optimized launch; see
    # radius_search_mem_optimized_launch. Not runnable until its kernel is finished.
    "custom_grid_cell",
    "custom_grid_cell_2"
)

# Inflate the cell slightly past the nominal radius so the fixed 27-cell stencil
# never misses a boundary neighbor under fp rounding in the floor() quantization.
_CELL_INFLATION = 1.0 + 1e-3

# Dense cell ids and the radix-sort count are int32, so the grid cannot exceed the
# positive int32 range. Memory is proportional to total_cells (begin/count tables
# plus sort scratch), so a small radius relative to the domain extent can make the
# dense backend enormous; that is left to the caller to manage.
_MAX_TOTAL_CELLS = (1 << 31) - 1

def _grid_metadata_2(points:torch.Tensor, inv_radius: float):
    '''
    removes morton bits and ordering
    '''
    device = points.device
    B = points.shape[0]
    abs_cells = torch.floor(points * inv_radius).to(torch.int64)
    min_c = abs_cells.amin(dim=1)
    max_c = abs_cells.amax(dim=1)
    
    grid = max_c - min_c + 1

    cells_per_batch = grid[:, 0] * grid[:, 1] * grid[:, 2]
    grid_base = torch.zeros(B + 1, dtype=torch.int64, device=device)
    
    torch.cumsum(cells_per_batch, dim=0, out=grid_base[1:])

    total_cells_device = grid_base[B]

    return min_c, grid, grid_base, total_cells_device






    

def _grid_metadata(points: torch.Tensor, inv_radius: float):
    """Per-batch dense-grid extents and the Morton bit budget for ``points``.

    Returns ``(min_c, grid, grid_base, total_cells, bits, morton_bits)`` where
    ``min_c``/``grid`` are ``(B, 3)`` int64 cell minima/extents, ``grid_base`` is
    the ``(B + 1,)`` exclusive scan of per-batch cell counts, and ``bits`` is the
    Morton bits per axis.
    """
    device = points.device
    B = points.shape[0]
    abs_cells = torch.floor(points * inv_radius).to(torch.int64)
    min_c = abs_cells.amin(dim=1)
    max_c = abs_cells.amax(dim=1)
    
    grid = max_c - min_c + 1

    cells_per_batch = grid[:, 0] * grid[:, 1] * grid[:, 2]
    grid_base = torch.zeros(B + 1, dtype=torch.int64, device=device)
    
    torch.cumsum(cells_per_batch, dim=0, out=grid_base[1:])
    
    total_cells = int(grid_base[B].item())
 
    max_extent = int(grid.max().item())
    bits = max(1, max_extent.bit_length())
    batch_bits = 0 if B <= 1 else (B - 1).bit_length()
    morton_bits = 3 * bits
    if morton_bits + batch_bits > 63:
        raise ValueError(
            "Morton key would overflow int64: "
            f"{morton_bits} morton bits + {batch_bits} batch bits > 63. "
            "Use a larger radius or process fewer batches per call."
        )
    return min_c, grid, grid_base, total_cells, bits, morton_bits


def _neighbor_offsets_27(device: torch.device) -> torch.Tensor:
    """The 27 cell offsets in ``{-1, 0, 1}^3``, ordered near to far for early exit."""
    rng = (-1, 0, 1)
    offs = [(dx, dy, dz) for dx in rng for dy in rng for dz in rng]
    offs.sort(key=lambda o: o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
    return torch.tensor(offs, dtype=torch.int32, device=device)


@dataclass
class MortonCompactDenseIndex:
    """Reusable PC-side dense-cell grid + Morton-sorted compact point bins.

    Attributes:
        B: Batch size.
        P: Points per batch.
        Np: Total points (``B * P``).
        radius: Search radius.
        inv_radius: Reciprocal of the (slightly inflated) cell size.
        bits: Morton bits per axis.
        morton_bits: Total Morton bit width (``3 * bits``).
        total_cells: Number of dense cells across all batches.
        min_cx/min_cy/min_cz: Per-batch cell minima, shape ``(B,)`` int32.
        grid_x/grid_y/grid_z: Per-batch cell extents, shape ``(B,)`` int32.
        grid_base: Per-batch dense-cell base offsets, shape ``(B + 1,)`` int32.
        offsets: Near-to-far 27-cell offset table, shape ``(27, 3)`` int32.
        cell_begin_by_row: Compact start index per dense row, shape ``(total_cells,)``.
        cell_count_by_row: Point count per dense row, shape ``(total_cells,)``.
        pc_x_sorted/pc_y_sorted/pc_z_sorted: Morton-cell-sorted point coords as SoA,
            ``(Np,)`` each. The FMA search reads these per-lane; the SoA layout keeps
            those reads coalesced with no wasted bandwidth.
        pc_orig_sorted: Local point index per sorted slot, ``(Np,)``.
        pc_xyz4_sorted: Padded ``(x, y, z, 0)`` coords, ``(>=Np, 4)``, for the GEMM
            path's tile loads.
        pc_norm_sorted: Squared norm per sorted slot for the GEMM path.
    """

    B: int
    P: int
    Np: int
    radius: float
    inv_radius: float
    bits: int
    morton_bits: int
    total_cells: int
    min_cx: torch.Tensor
    min_cy: torch.Tensor
    min_cz: torch.Tensor
    grid_x: torch.Tensor
    grid_y: torch.Tensor
    grid_z: torch.Tensor
    grid_base: torch.Tensor
    offsets: torch.Tensor
    cell_begin_by_row: torch.Tensor
    cell_count_by_row: torch.Tensor
    pc_x_sorted: torch.Tensor
    pc_y_sorted: torch.Tensor
    pc_z_sorted: torch.Tensor
    pc_orig_sorted: torch.Tensor
    pc_xyz4_sorted: torch.Tensor
    pc_norm_sorted: torch.Tensor


@dataclass
class SortedQueryData:
    """Morton-sorted query arrays produced by :func:`prepare_morton_sorted_queries`.

    Attributes:
        B: Batch size.
        Q: Queries per batch.
        Nq: Total queries (``B * Q``).
        q_orig_sorted: Original flat query id (``b * Q + q``) per sorted slot. The
            batch id and absolute cell coords are derived from this and the coords
            where needed (``b = flat_q // Q``, cell = ``floor(coord / cell_size)``).
        q_x_sorted/q_y_sorted/q_z_sorted: Sorted query coords (SoA), read per-query
            by the FMA search.
        q_xyz4_sorted: Padded ``(x, y, z, 0)`` query coords, ``(>=Nq, 4)``, for the
            GEMM path's tile loads.
        q_norm_sorted: Squared norm per sorted query for the GEMM path.
        qkeys_sorted: Sorted packed query keys (for GEMM query-cell grouping).
    """

    B: int
    Q: int
    Nq: int
    q_orig_sorted: torch.Tensor
    q_x_sorted: torch.Tensor
    q_y_sorted: torch.Tensor
    q_z_sorted: torch.Tensor
    q_xyz4_sorted: torch.Tensor
    q_norm_sorted: torch.Tensor
    qkeys_sorted: torch.Tensor


def build_morton_compact_dense_index(
    points: torch.Tensor,
    radius: float,
    wp_device,
    wp_stream,
    sort_cells: bool = True,
    is_e2e:bool = False
) -> MortonCompactDenseIndex:
    """Build the dense-cell index: histogram, order cells, compact points.

    Args:
        points: Reference points of shape ``(B, N, 3)``, float32, CUDA.
        radius: Search radius.
        wp_device: Warp launch device (from ``warp_launch_context``).
        wp_stream: Warp launch stream (from ``warp_launch_context``).
        sort_cells: When ``True`` (default) the compact point bins are laid out in
            Morton-cell order via a ``radix_sort_pairs`` over all ``total_cells``.
            When ``False`` the bins are laid out in row-major dense-cell order via a
            single exclusive scan of ``cell_count_by_row`` -- no radix sort and no
            ``2 * total_cells`` scratch. Both orders are correct (the search kernel
            indexes cells by row-major dense id); Morton only changes locality.

    Returns:
        The populated :class:`MortonCompactDenseIndex`.

    Raises:
        ValueError: If the Morton key would overflow int64, or the dense grid
            exceeds the int32 cell-id limit.
    """
    device = points.device
    B, P, _ = points.shape
    Np = B * P

    cell_size = float(radius) * _CELL_INFLATION
    inv_radius = 1.0 / cell_size

    if os.environ.get('PROFILE_RUN', '0') == '1':
        torch.cuda.nvtx.range_push('GRID METADATA')
    
    if is_e2e:
        min_c, grid, grid_base_t, total_cells_device = _grid_metadata_2(points, inv_radius)
        bits = 0
        morton_bits =0
    else:
        min_c, grid, grid_base_t, total_cells, bits, morton_bits = _grid_metadata(
            points, inv_radius
        )

    if os.environ.get('PROFILE_RUN', '0') == '1':
        torch.cuda.nvtx.range_pop()
    
    if is_e2e:
        total_cells = int(total_cells_device.item())
        #total_cells = 845853361
        #total_cells = 100
    # print(f"TOTAL CELLS : {total_cells}")
    # import ipdb; ipdb.set_trace()
    if total_cells > _MAX_TOTAL_CELLS:
        raise ValueError(
            f"dense grid has {total_cells} cells, exceeding the int32 cell-id "
            f"limit ({_MAX_TOTAL_CELLS}). Use a larger radius."
        )

    min_cx = min_c[:, 0].to(torch.int32).contiguous()
    min_cy = min_c[:, 1].to(torch.int32).contiguous()
    min_cz = min_c[:, 2].to(torch.int32).contiguous()

    grid_x = grid[:, 0].to(torch.int32).contiguous()
    grid_y = grid[:, 1].to(torch.int32).contiguous()
    grid_z = grid[:, 2].to(torch.int32).contiguous()
    grid_base = grid_base_t.to(torch.int32).contiguous()

    points_wp = wp.from_torch(points, dtype=wp.vec3, return_ctype=True)
    min_cx_wp = wp.from_torch(min_cx)
    min_cy_wp = wp.from_torch(min_cy)
    min_cz_wp = wp.from_torch(min_cz)
    grid_x_wp = wp.from_torch(grid_x)
    grid_y_wp = wp.from_torch(grid_y)
    grid_z_wp = wp.from_torch(grid_z)
    grid_base_wp = wp.from_torch(grid_base)

    point_cell_row = torch.empty(Np, dtype=torch.int32, device=device)
    cell_count_by_row = torch.zeros(total_cells, dtype=torch.int32, device=device)
    if os.environ.get('PROFILE_RUN', '0') == '1':
        torch.cuda.nvtx.range_push('COUNT POINTS IN CELL')
    wp.launch(
        kernel=mk.count_points_in_cells,
        dim=Np,
        inputs=[
            points_wp,
            float(inv_radius),
            int(P),
            min_cx_wp,
            min_cy_wp,
            min_cz_wp,
            grid_x_wp,
            grid_y_wp,
            grid_base_wp,
            wp.from_torch(point_cell_row),
            wp.from_torch(cell_count_by_row),
        ],
        device=wp_device,
        stream=wp_stream,
    )
    if os.environ.get('PROFILE_RUN', '0') == '1':
        torch.cuda.nvtx.range_pop()

    # Per-row start offsets into the compact point arrays. Both layouts below are
    # correct because the search kernel looks cells up by row-major dense id (never
    # by Morton rank); Morton ordering only changes the spatial locality of the
    # compact bins.
    cell_begin_by_row = torch.empty(total_cells, dtype=torch.int32, device=device)
    if sort_cells:
        # Morton-cell order: sort every dense cell by its (batch, Morton) key, then
        # scatter the rank-order prefix offsets back into row-indexed starts. This
        # is the expensive radix_sort_pairs over all `total_cells` int64 keys.
        cell_key_tmp = torch.empty(2 * total_cells, dtype=torch.int64, device=device)
        cell_row_tmp = torch.empty(2 * total_cells, dtype=torch.int32, device=device)
        cell_key_wp = wp.from_torch(cell_key_tmp)
        cell_row_wp = wp.from_torch(cell_row_tmp)
        wp.launch(
            kernel=mk.enumerate_cell_morton_keys,
            dim=total_cells,
            inputs=[
                grid_x_wp,
                grid_y_wp,
                grid_base_wp,
                int(B),
                int(bits),
                int(morton_bits),
                cell_key_wp,
                cell_row_wp,
            ],
            device=wp_device,
            stream=wp_stream,
        )
        wp.utils.radix_sort_pairs(cell_key_wp, cell_row_wp, total_cells)

        cell_counts_rank = torch.empty(total_cells, dtype=torch.int32, device=device)
        cell_offsets_rank = torch.empty(total_cells, dtype=torch.int32, device=device)
        counts_rank_wp = wp.from_torch(cell_counts_rank)
        offsets_rank_wp = wp.from_torch(cell_offsets_rank)
        wp.launch(
            kernel=mk.gather_cell_counts_by_rank,
            dim=total_cells,
            inputs=[cell_row_wp, wp.from_torch(cell_count_by_row), counts_rank_wp],
            device=wp_device,
            stream=wp_stream,
        )
        wp.utils.array_scan(counts_rank_wp, offsets_rank_wp, inclusive=False)
        wp.launch(
            kernel=mk.fill_cell_row_ranges,
            dim=total_cells,
            inputs=[cell_row_wp, offsets_rank_wp, wp.from_torch(cell_begin_by_row)],
            device=wp_device,
            stream=wp_stream,
        )
    else:
        # Row-major order: a single exclusive scan of the per-row counts gives the
        # per-row starts directly. No radix sort, no key/rank scratch buffers.
        if os.environ.get('PROFILE_RUN', '0') == '1':
            torch.cuda.nvtx.range_push('Array Scan')
        # _warp_exclusive_scan(
        #     cell_count_by_row, cell_begin_by_row, wp_device, wp_stream
        # )
        wp.utils.array_scan(
            wp.from_torch(cell_count_by_row),
            wp.from_torch(cell_begin_by_row),
            inclusive=False,
        )
        if os.environ.get('PROFILE_RUN', '0') == '1':
            torch.cuda.nvtx.range_pop()

    n_pad = Np + TILE_P
    pc_x_sorted = torch.empty(Np, dtype=torch.float32, device=device)
    pc_y_sorted = torch.empty(Np, dtype=torch.float32, device=device)
    pc_z_sorted = torch.empty(Np, dtype=torch.float32, device=device)
    pc_orig_sorted = torch.empty(Np, dtype=torch.int32, device=device)
    pc_xyz4_sorted = torch.zeros((n_pad, 4), dtype=torch.float32, device=device)
    pc_norm_sorted = torch.zeros(n_pad, dtype=torch.float32, device=device)
    cell_write_by_row = cell_begin_by_row.clone()

    if os.environ.get('PROFILE_RUN', '0') == '1':
        torch.cuda.nvtx.range_push('Scatter Points to cells')
    wp.launch(
        kernel=mk.scatter_points_to_cells,
        dim=Np,
        inputs=[
            points_wp,
            wp.from_torch(point_cell_row),
            int(P),
            wp.from_torch(cell_write_by_row),
            wp.from_torch(pc_x_sorted),
            wp.from_torch(pc_y_sorted),
            wp.from_torch(pc_z_sorted),
            wp.from_torch(pc_orig_sorted),
            wp.from_torch(pc_xyz4_sorted),
            wp.from_torch(pc_norm_sorted),
        ],
        device=wp_device,
        stream=wp_stream,
    )
    if os.environ.get('PROFILE_RUN', '0') == '1':
        torch.cuda.nvtx.range_pop()

    return MortonCompactDenseIndex(
        B=B,
        P=P,
        Np=Np,
        radius=float(radius),
        inv_radius=float(inv_radius),
        bits=int(bits),
        morton_bits=int(morton_bits),
        total_cells=total_cells,
        min_cx=min_cx,
        min_cy=min_cy,
        min_cz=min_cz,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        grid_base=grid_base,
        offsets=_neighbor_offsets_27(device),
        cell_begin_by_row=cell_begin_by_row,
        cell_count_by_row=cell_count_by_row,
        pc_x_sorted=pc_x_sorted,
        pc_y_sorted=pc_y_sorted,
        pc_z_sorted=pc_z_sorted,
        pc_orig_sorted=pc_orig_sorted,
        pc_xyz4_sorted=pc_xyz4_sorted,
        pc_norm_sorted=pc_norm_sorted,
    )


def prepare_morton_sorted_queries(
    index: MortonCompactDenseIndex,
    queries: torch.Tensor,
    wp_device,
    wp_stream,
    sort_queries: bool = True,
) -> SortedQueryData:
    """Order the queries and gather their coords, absolute cells, and norms.

    Args:
        index: The PC-side dense index (provides the shared ``inv_radius`` / bits).
        queries: Query points of shape ``(B, Q, 3)``, float32, CUDA.
        wp_device: Warp launch device.
        wp_stream: Warp launch stream.
        sort_queries: When ``True`` (default) the queries are Morton-sorted via a
            ``radix_sort_pairs`` over ``Nq`` (for search-time locality), and
            ``qkeys_sorted`` is populated for the GEMM query-cell grouping. When
            ``False`` the queries are kept in original (identity) order -- no radix
            sort -- and ``qkeys_sorted`` is empty. The FMA search is
            order-independent (one block per query, output written at the original
            flat id), so only the GEMM path needs ``sort_queries=True``.

    Returns:
        The populated :class:`SortedQueryData`.
    """
    device = queries.device
    B, Q, _ = queries.shape
    Nq = B * Q
    inv_radius = index.inv_radius

    queries_wp = wp.from_torch(queries, dtype=wp.vec3, return_ctype=True)

    if sort_queries:
        # Morton-sort the queries so adjacent blocks touch overlapping cells (L2
        # locality); also yields qkeys_sorted for the GEMM query-cell grouping.
        q_abs = torch.floor(queries * inv_radius).to(torch.int64)
        q_min = q_abs.amin(dim=1)
        q_min_cx = q_min[:, 0].to(torch.int32).contiguous()
        q_min_cy = q_min[:, 1].to(torch.int32).contiguous()
        q_min_cz = q_min[:, 2].to(torch.int32).contiguous()

        q_key_tmp = torch.empty(2 * Nq, dtype=torch.int64, device=device)
        q_val_tmp = torch.empty(2 * Nq, dtype=torch.int32, device=device)
        q_key_wp = wp.from_torch(q_key_tmp)
        q_val_wp = wp.from_torch(q_val_tmp)
        wp.launch(
            kernel=mk.make_query_morton_keys,
            dim=Nq,
            inputs=[
                queries_wp,
                float(inv_radius),
                int(Q),
                wp.from_torch(q_min_cx),
                wp.from_torch(q_min_cy),
                wp.from_torch(q_min_cz),
                int(index.bits),
                int(index.morton_bits),
                q_key_wp,
                q_val_wp,
            ],
            device=wp_device,
            stream=wp_stream,
        )
        wp.utils.radix_sort_pairs(q_key_wp, q_val_wp, Nq)
        qkeys_sorted = q_key_tmp[:Nq].clone()
    else:
        # Identity order: no key build, no radix sort. gather_sorted_queries still
        # runs below to build the SoA/vec4/norm arrays the search kernel reads.
        # qkeys_sorted is consumed only by the GEMM task builder -> leave empty.
        q_val_tmp = torch.arange(Nq, dtype=torch.int32, device=device)
        q_val_wp = wp.from_torch(q_val_tmp)
        qkeys_sorted = torch.empty(0, dtype=torch.int64, device=device)

    bq_pad = Nq + TILE_Q
    q_orig_sorted = torch.empty(Nq, dtype=torch.int32, device=device)
    q_x_sorted = torch.empty(Nq, dtype=torch.float32, device=device)
    q_y_sorted = torch.empty(Nq, dtype=torch.float32, device=device)
    q_z_sorted = torch.empty(Nq, dtype=torch.float32, device=device)
    q_xyz4_sorted = torch.zeros((bq_pad, 4), dtype=torch.float32, device=device)
    q_norm_sorted = torch.zeros(bq_pad, dtype=torch.float32, device=device)
    wp.launch(
        kernel=mk.gather_sorted_queries,
        dim=Nq,
        inputs=[
            queries_wp,
            q_val_wp,
            int(Q),
            wp.from_torch(q_orig_sorted),
            wp.from_torch(q_x_sorted),
            wp.from_torch(q_y_sorted),
            wp.from_torch(q_z_sorted),
            wp.from_torch(q_xyz4_sorted),
            wp.from_torch(q_norm_sorted),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    return SortedQueryData(
        B=B,
        Q=Q,
        Nq=Nq,
        q_orig_sorted=q_orig_sorted,
        q_x_sorted=q_x_sorted,
        q_y_sorted=q_y_sorted,
        q_z_sorted=q_z_sorted,
        q_xyz4_sorted=q_xyz4_sorted,
        q_norm_sorted=q_norm_sorted,
        qkeys_sorted=qkeys_sorted,
    )


# def _warp_exclusive_scan(
#     counts: torch.Tensor,
#     begins: torch.Tensor,
#     wp_device: str,
#     wp_stream,
# ) -> None:
#     """
#     In-place async exclusive prefix scan with no cudaDeviceSynchronize.

#     Uses a three-phase Warp tile kernel (see exclusive_scan_phase{1,2,3} in
#     _morton_dense_kernels.py) that stays entirely on *wp_stream*:

#       Phase 1  per-block local exclusive scan + block totals
#       Phase 2  exclusive scan of block totals  (single block)
#       Phase 3  global[i] = local[i] + block_offset[block(i)]

#     Constraint: len(counts) <= SCAN_BLOCK^2 (= 1 048 576 with SCAN_BLOCK=1024).
#     """
#     n = counts.shape[0]
#     if n == 0:
#         return
#     BLOCK = int(mk.SCAN_BLOCK)  # wp.constant → Python int for arithmetic
#     num_blocks = math.ceil(n / BLOCK)
#     if num_blocks > BLOCK:
#         raise ValueError(
#             f"_warp_exclusive_scan: n={n} exceeds two-level limit "
#             f"SCAN_BLOCK^2={BLOCK * BLOCK}. Increase SCAN_BLOCK or use a "
#             "three-level hierarchy."
#         )

#     n_pad = num_blocks * BLOCK

#     # Zero-initialised padded input so the last tile load is always in-bounds.
#     arr_pad_wp = wp.zeros(n_pad, dtype=wp.int32, device=wp_device)
#     wp.launch(
#         mk._scan_copy_pad,
#         dim=n,
#         inputs=[wp.from_torch(counts), arr_pad_wp, n],
#         device=wp_device,
#         stream=wp_stream,
#     )

#     partial_wp = wp.empty(n_pad, dtype=wp.int32, device=wp_device)
#     block_sums_wp = wp.zeros(BLOCK, dtype=wp.int32, device=wp_device)
#     block_offsets_wp = wp.empty(BLOCK, dtype=wp.int32, device=wp_device)

#     wp.launch(
#         mk.exclusive_scan_phase1,
#         dim=num_blocks,
#         block_dim=BLOCK,
#         inputs=[arr_pad_wp, partial_wp, block_sums_wp],
#         device=wp_device,
#         stream=wp_stream,
#     )
#     wp.launch(
#         mk.exclusive_scan_phase2,
#         dim=1,
#         block_dim=BLOCK,
#         inputs=[block_sums_wp, block_offsets_wp],
#         device=wp_device,
#         stream=wp_stream,
#     )
#     wp.launch(
#         mk.exclusive_scan_phase3,
#         dim=n,
#         inputs=[partial_wp, block_offsets_wp, wp.from_torch(begins), n, BLOCK],
#         device=wp_device,
#         stream=wp_stream,
#     )


def _exclusive_scan(counts: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum of a 1D integer tensor, returned as ``int64``."""
    counts64 = counts.to(torch.int64)
    return torch.cumsum(counts64, dim=0) - counts64


def _alloc_outputs(B, Q, max_points, return_dists, return_points, device):
    """Allocate the output tensors (indices/count always; dists/pts optional)."""
    out_idx = torch.zeros((B, Q, max_points), dtype=torch.int32, device=device)
    out_count = torch.zeros(B * Q, dtype=torch.int32, device=device)
    out_dist = (
        torch.zeros((B, Q, max_points), dtype=torch.float32, device=device)
        if return_dists
        else None
    )
    out_pts = (
        torch.zeros((B, Q, max_points, 3), dtype=torch.float32, device=device)
        if return_points
        else None
    )
    return out_idx, out_count, out_dist, out_pts


def _wp_outputs(out_idx, out_count, out_dist, out_pts, Nq, max_points):
    """Wrap the (flattened) output tensors as Warp arrays for a search launch."""
    idx_wp = wp.from_torch(out_idx.view(Nq, max_points))
    count_wp = wp.from_torch(out_count)
    dist_wp = (
        wp.from_torch(out_dist.view(Nq, max_points)) if out_dist is not None else None
    )
    pts_wp = (
        wp.from_torch(out_pts.view(Nq, max_points, 3), dtype=wp.vec3)
        if out_pts is not None
        else None
    )
    return idx_wp, count_wp, dist_wp, pts_wp


def _gather_neighbor_points(
    points: torch.Tensor,
    out_idx: torch.Tensor,
    num: torch.Tensor,
) -> torch.Tensor:
    """Gather neighbor coordinates from ``out_idx`` on the host (Stage 1).

    Replaces the FMA kernel's old per-neighbor ``wp.vec3`` global scatter (a
    misaligned 12-byte store NCU flagged as inefficient) with a coalesced
    ``index_select`` gather from the reference points.

    Args:
        points: Reference points, ``(B, N, 3)`` float32.
        out_idx: Per-query neighbor indices -- local point ids in ``[0, N)`` with 0
            padding past the neighbor count -- ``(B, Q, max_points)`` int32.
        num: Neighbor count per query, ``(B, Q)`` int32.

    Returns:
        Neighbor coordinates ``(B, Q, max_points, 3)`` float32, with padding slots
        (``slot >= num``) zeroed to match the previous in-kernel output contract.
    """
    B, N, _ = points.shape
    _, Q, mp = out_idx.shape
    device = points.device
    # Flatten (batch, local-point) -> global row so a single index_select gathers
    # every neighbor coordinate in one coalesced pass.
    idx = out_idx.long().clamp_(0, N - 1)
    batch_off = (torch.arange(B, device=device, dtype=torch.long) * N).view(B, 1, 1)
    flat = (idx + batch_off).reshape(-1)
    gathered = points.reshape(B * N, 3).index_select(0, flat).reshape(B, Q, mp, 3)
    # Zero the padding slots (slot >= neighbor count) to preserve the old contract.
    slot = torch.arange(mp, device=device).view(1, 1, mp)
    valid = slot < num.view(B, Q, 1)
    return gathered * valid.unsqueeze(-1).to(gathered.dtype)


def radius_search_morton_warp_fma(
    index: MortonCompactDenseIndex,
    queries: SortedQueryData,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """Production search: one warp per query over the dense Morton-sorted cells.

    Returns ``(indices, points, distances, num_neighbors)`` with the batch dim
    intact; ``points``/``distances`` are ``None`` when not requested.
    """
    B, Q, Nq = queries.B, queries.Q, queries.Nq
    device = index.pc_xyz4_sorted.device
    
    if os.environ.get('PROFILE_RUN', '0') =='1':
        torch.cuda.nvtx.range_push('alloc outputs')
    
    out_idx, out_count, out_dist, out_pts = _alloc_outputs(
        B, Q, max_points, return_dists, return_points, device
    )
    idx_wp, count_wp, dist_wp, pts_wp = _wp_outputs(
        out_idx, out_count, out_dist, out_pts, Nq, max_points
    )

    if os.environ.get('PROFILE_RUN', '0') =='1':
        torch.cuda.nvtx.range_pop()

    if os.environ.get('PROFILE_RUN', '0') =='1':
        torch.cuda.nvtx.range_push('RAD SEARCH KERNEL')

    wp.launch_tiled(
        kernel=mk.radius_search_dense_fma_kernel_v2,
        dim=Nq,
        inputs=[
            wp.from_torch(queries.q_x_sorted),
            wp.from_torch(queries.q_y_sorted),
            wp.from_torch(queries.q_z_sorted),
            wp.from_torch(queries.q_orig_sorted),
            float(index.inv_radius),
            int(queries.Q),
            wp.from_torch(index.offsets),
            wp.from_torch(index.min_cx),
            wp.from_torch(index.min_cy),
            wp.from_torch(index.min_cz),
            wp.from_torch(index.grid_x),
            wp.from_torch(index.grid_y),
            wp.from_torch(index.grid_z),
            wp.from_torch(index.grid_base),
            wp.from_torch(index.cell_begin_by_row),
            wp.from_torch(index.cell_count_by_row),
            wp.from_torch(index.pc_x_sorted),
            wp.from_torch(index.pc_y_sorted),
            wp.from_torch(index.pc_z_sorted),
            wp.from_torch(index.pc_orig_sorted),
            float(radius) * float(radius),
            int(max_points),
            bool(return_dists),
            bool(return_points),
            idx_wp,
            count_wp,
            dist_wp,
            pts_wp,
        ],
        block_dim=FMA_V2_BLOCK_DIM,
        device=wp_device,
        stream=wp_stream,
    )

    if os.environ.get('PROFILE_RUN', '0') =='1':
        torch.cuda.nvtx.range_pop()
    return out_idx, out_pts, out_dist, out_count.view(B, Q)


def radius_search_morton_warp_fma_store_opt(
    index: MortonCompactDenseIndex,
    queries: SortedQueryData,
    points: torch.Tensor,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """Store-optimized FMA search: shared-mem staged flush + host-gathered points.

    Same one-block-per-query 27-cell scan as :func:`radius_search_morton_warp_fma`,
    but the kernel (Stage 2) stages indices/distances in a block-shared tile and
    flushes them in one coalesced pass, and (Stage 1) does not scatter neighbor
    coordinates -- those are gathered on the host from ``out_idx`` via
    :func:`_gather_neighbor_points`. Together these remove the misaligned
    ``wp.vec3`` global store and the per-chunk partial-warp scatters the profiler
    flagged on :func:`radius_search_morton_warp_fma`. Returns the ``(indices,
    points, distances, num_neighbors)`` 4-tuple with the batch dim intact.

    Raises:
        ValueError: If ``max_points`` exceeds the shared staging capacity
            (``FMA_STAGE_CAP``).
    """
    B, Q, Nq = queries.B, queries.Q, queries.Nq
    device = index.pc_xyz4_sorted.device
    if max_points > mk.FMA_STAGE_CAP:
        raise ValueError(
            f"dense_fma_store_opt supports max_points <= {mk.FMA_STAGE_CAP} "
            f"(got {max_points}); increase FMA_STAGE_CAP in _morton_dense_kernels.py "
            "or use the dense_fma variant."
        )

    # The kernel writes only indices/distances; coordinates are gathered below.
    out_idx = torch.zeros((B, Q, max_points), dtype=torch.int32, device=device)
    out_count = torch.zeros(B * Q, dtype=torch.int32, device=device)
    out_dist = (
        torch.zeros((B, Q, max_points), dtype=torch.float32, device=device)
        if return_dists
        else None
    )
    idx_wp = wp.from_torch(out_idx.view(Nq, max_points))
    count_wp = wp.from_torch(out_count)
    dist_wp = (
        wp.from_torch(out_dist.view(Nq, max_points)) if out_dist is not None else None
    )

    wp.launch_tiled(
        kernel=mk.radius_search_dense_fma_store_opt_kernel,
        dim=Nq,
        inputs=[
            wp.from_torch(queries.q_x_sorted),
            wp.from_torch(queries.q_y_sorted),
            wp.from_torch(queries.q_z_sorted),
            wp.from_torch(queries.q_orig_sorted),
            float(index.inv_radius),
            int(queries.Q),
            wp.from_torch(index.offsets),
            wp.from_torch(index.min_cx),
            wp.from_torch(index.min_cy),
            wp.from_torch(index.min_cz),
            wp.from_torch(index.grid_x),
            wp.from_torch(index.grid_y),
            wp.from_torch(index.grid_z),
            wp.from_torch(index.grid_base),
            wp.from_torch(index.cell_begin_by_row),
            wp.from_torch(index.cell_count_by_row),
            wp.from_torch(index.pc_x_sorted),
            wp.from_torch(index.pc_y_sorted),
            wp.from_torch(index.pc_z_sorted),
            wp.from_torch(index.pc_orig_sorted),
            float(radius) * float(radius),
            int(max_points),
            bool(return_dists),
            idx_wp,
            count_wp,
            dist_wp,
        ],
        block_dim=FMA_BLOCK_DIM,
        device=wp_device,
        stream=wp_stream,
    )
    num = out_count.view(B, Q)
    out_pts = _gather_neighbor_points(points, out_idx, num) if return_points else None
    return out_idx, out_pts, out_dist, num


def radius_search_morton_warp_fma_mem_opt(
    index: MortonCompactDenseIndex,
    queries: SortedQueryData,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """Memory-optimized FMA search: vec4 coord loads + in-kernel offset decode.

    Same one-block-per-query structure as :func:`radius_search_morton_warp_fma`,
    but reads query/point coords as single ``wp.vec4`` loads from the padded
    ``*_xyz4_sorted`` arrays and decodes the 27 neighbor offsets in-kernel. Returns
    the ``(indices, points, distances, num_neighbors)`` 4-tuple with batch dim
    intact.
    """
    B, Q, Nq = queries.B, queries.Q, queries.Nq
    device = index.pc_xyz4_sorted.device
    out_idx, out_count, out_dist, out_pts = _alloc_outputs(
        B, Q, max_points, return_dists, return_points, device
    )
    idx_wp, count_wp, dist_wp, pts_wp = _wp_outputs(
        out_idx, out_count, out_dist, out_pts, Nq, max_points
    )

    wp.launch_tiled(
        kernel=mk.radius_search_dense_fma_kernel_mem_opt,
        dim=Nq,
        inputs=[
            wp.from_torch(queries.q_xyz4_sorted, dtype=wp.vec4),
            wp.from_torch(queries.q_orig_sorted),
            float(index.inv_radius),
            int(queries.Q),
            wp.from_torch(index.min_cx),
            wp.from_torch(index.min_cy),
            wp.from_torch(index.min_cz),
            wp.from_torch(index.grid_x),
            wp.from_torch(index.grid_y),
            wp.from_torch(index.grid_z),
            wp.from_torch(index.grid_base),
            wp.from_torch(index.cell_begin_by_row),
            wp.from_torch(index.cell_count_by_row),
            wp.from_torch(index.pc_xyz4_sorted, dtype=wp.vec4),
            wp.from_torch(index.pc_orig_sorted),
            float(radius) * float(radius),
            int(max_points),
            bool(return_dists),
            bool(return_points),
            idx_wp,
            count_wp,
            dist_wp,
            pts_wp,
        ],
        block_dim=FMA_BLOCK_DIM,
        device=wp_device,
        stream=wp_stream,
    )
    return out_idx, out_pts, out_dist, out_count.view(B, Q)


def radius_search_morton_warp_fma_mm(
    index: MortonCompactDenseIndex,
    queries: SortedQueryData,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """In-kernel-gather matmul search: one block per query, per-chunk tile matmul.

    Same per-query 27-cell scan and tile-scan compaction as
    :func:`radius_search_morton_warp_fma`, but distances come from a tiled matmul
    of the query row against each candidate chunk. Launched at ``block_dim ==
    TILE_P``. Returns ``(indices, points, distances, num_neighbors)`` with the
    batch dim intact.
    """
    B, Q, Nq = queries.B, queries.Q, queries.Nq
    device = index.pc_xyz4_sorted.device
    out_idx, out_count, out_dist, out_pts = _alloc_outputs(
        B, Q, max_points, return_dists, return_points, device
    )
    idx_wp, count_wp, dist_wp, pts_wp = _wp_outputs(
        out_idx, out_count, out_dist, out_pts, Nq, max_points
    )

    wp.launch_tiled(
        kernel=mgk.radius_search_dense_fma_kernel_mm,
        dim=Nq,
        inputs=[
            wp.from_torch(queries.q_xyz4_sorted),
            wp.from_torch(queries.q_norm_sorted),
            wp.from_torch(queries.q_orig_sorted),
            float(index.inv_radius),
            int(queries.Q),
            wp.from_torch(index.offsets),
            wp.from_torch(index.min_cx),
            wp.from_torch(index.min_cy),
            wp.from_torch(index.min_cz),
            wp.from_torch(index.grid_x),
            wp.from_torch(index.grid_y),
            wp.from_torch(index.grid_z),
            wp.from_torch(index.grid_base),
            wp.from_torch(index.cell_begin_by_row),
            wp.from_torch(index.cell_count_by_row),
            wp.from_torch(index.pc_xyz4_sorted),
            wp.from_torch(index.pc_norm_sorted),
            wp.from_torch(index.pc_orig_sorted),
            float(radius) * float(radius),
            int(max_points),
            bool(return_dists),
            bool(return_points),
            idx_wp,
            count_wp,
            dist_wp,
            pts_wp,
        ],
        block_dim=TILE_P,
        device=wp_device,
        stream=wp_stream,
    )
    return out_idx, out_pts, out_dist, out_count.view(B, Q)


def _build_gemm_tasks(
    index: MortonCompactDenseIndex,
    queries: SortedQueryData,
    wp_device,
    wp_stream,
) -> dict:
    """Build the ``(query-tile, point-chunk)`` task list for the GEMM path.

    Splits each occupied query cell into ``TILE_Q`` chunks, then runs the
    two-pass (count -> exclusive scan -> fill) task generation against the dense
    cell table.
    """
    device = index.pc_xyz4_sorted.device
    Nq = queries.Nq
    qkeys = queries.qkeys_sorted

    flags = torch.ones(Nq, dtype=torch.bool, device=device)
    flags[1:] = qkeys[1:] != qkeys[:-1]
    qstarts = torch.nonzero(flags, as_tuple=False).flatten()
    num_qcells = int(qstarts.numel())
    qends = torch.empty_like(qstarts)
    if num_qcells > 1:
        qends[:-1] = qstarts[1:]
    qends[-1] = Nq
    qcell_len = qends - qstarts

    tiles_per_cell = (qcell_len + TILE_Q - 1) // TILE_Q
    num_qtiles = int(tiles_per_cell.sum().item())
    tile_off = _exclusive_scan(tiles_per_cell)
    cell_of_tile = torch.repeat_interleave(
        torch.arange(num_qcells, device=device), tiles_per_cell
    )
    local_tile = torch.arange(num_qtiles, device=device) - tile_off[cell_of_tile]
    tile_begin = qstarts[cell_of_tile] + local_tile * TILE_Q
    q_tile_begin = tile_begin.to(torch.int32).contiguous()
    q_tile_count = (
        torch.clamp(qends[cell_of_tile] - tile_begin, max=TILE_Q)
        .to(torch.int32)
        .contiguous()
    )
    inv_radius = index.inv_radius
    q_coords = queries.q_xyz4_sorted[: queries.Nq]
    q_batch = (queries.q_orig_sorted.to(torch.int64) // queries.Q).to(torch.int32)
    q_cx = torch.floor(q_coords[:, 0] * inv_radius).to(torch.int32)
    q_cy = torch.floor(q_coords[:, 1] * inv_radius).to(torch.int32)
    q_cz = torch.floor(q_coords[:, 2] * inv_radius).to(torch.int32)
    q_tile_b = q_batch[qstarts][cell_of_tile].contiguous()
    q_tile_cx = q_cx[qstarts][cell_of_tile].contiguous()
    q_tile_cy = q_cy[qstarts][cell_of_tile].contiguous()
    q_tile_cz = q_cz[qstarts][cell_of_tile].contiguous()

    offsets_wp = wp.from_torch(index.offsets)
    min_cx_wp = wp.from_torch(index.min_cx)
    min_cy_wp = wp.from_torch(index.min_cy)
    min_cz_wp = wp.from_torch(index.min_cz)
    grid_x_wp = wp.from_torch(index.grid_x)
    grid_y_wp = wp.from_torch(index.grid_y)
    grid_z_wp = wp.from_torch(index.grid_z)
    grid_base_wp = wp.from_torch(index.grid_base)
    cell_begin_wp = wp.from_torch(index.cell_begin_by_row)
    cell_count_wp = wp.from_torch(index.cell_count_by_row)
    q_tile_b_wp = wp.from_torch(q_tile_b)
    q_tile_cx_wp = wp.from_torch(q_tile_cx)
    q_tile_cy_wp = wp.from_torch(q_tile_cy)
    q_tile_cz_wp = wp.from_torch(q_tile_cz)

    task_count = torch.empty(num_qtiles, dtype=torch.int32, device=device)
    wp.launch(
        kernel=mgk.count_neighbor_tasks,
        dim=num_qtiles,
        inputs=[
            q_tile_b_wp,
            q_tile_cx_wp,
            q_tile_cy_wp,
            q_tile_cz_wp,
            offsets_wp,
            min_cx_wp,
            min_cy_wp,
            min_cz_wp,
            grid_x_wp,
            grid_y_wp,
            grid_z_wp,
            grid_base_wp,
            cell_count_wp,
            int(TILE_P),
            wp.from_torch(task_count),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    task_offset = _exclusive_scan(task_count).to(torch.int32).contiguous()
    num_tasks = int(task_count.sum().item())

    task_q_begin = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)
    task_q_count = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)
    task_p_begin = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)
    task_p_count = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)
    if num_tasks > 0:
        wp.launch(
            kernel=mgk.fill_neighbor_tasks,
            dim=num_qtiles,
            inputs=[
                wp.from_torch(q_tile_begin),
                wp.from_torch(q_tile_count),
                q_tile_b_wp,
                q_tile_cx_wp,
                q_tile_cy_wp,
                q_tile_cz_wp,
                wp.from_torch(task_offset),
                offsets_wp,
                min_cx_wp,
                min_cy_wp,
                min_cz_wp,
                grid_x_wp,
                grid_y_wp,
                grid_z_wp,
                grid_base_wp,
                cell_begin_wp,
                cell_count_wp,
                int(TILE_P),
                wp.from_torch(task_q_begin),
                wp.from_torch(task_q_count),
                wp.from_torch(task_p_begin),
                wp.from_torch(task_p_count),
            ],
            device=wp_device,
            stream=wp_stream,
        )

    return {
        "num_tasks": num_tasks,
        "task_q_begin": task_q_begin,
        "task_q_count": task_q_count,
        "task_p_begin": task_p_begin,
        "task_p_count": task_p_count,
    }


def radius_search_morton_tiled_gemm(
    index: MortonCompactDenseIndex,
    queries: SortedQueryData,
    tasks: dict,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """Benchmark search: tiled GEMM distances over the dense Morton-sorted cells.

    Returns ``(indices, points, distances, num_neighbors)`` with the batch dim
    intact; the per-query atomic counter is clamped to ``max_points`` afterward.
    """
    B, Q, Nq = queries.B, queries.Q, queries.Nq
    device = index.pc_xyz4_sorted.device
    out_idx, out_count, out_dist, out_pts = _alloc_outputs(
        B, Q, max_points, return_dists, return_points, device
    )
    idx_wp, count_wp, dist_wp, pts_wp = _wp_outputs(
        out_idx, out_count, out_dist, out_pts, Nq, max_points
    )

    num_tasks = tasks["num_tasks"]
    if num_tasks > 0:
        lane_iota = torch.arange(TILE_P, dtype=torch.int32, device=device)
        wp.launch_tiled(
            kernel=mgk.radius_search_dense_gemm_kernel,
            dim=num_tasks,
            inputs=[
                wp.from_torch(tasks["task_q_begin"]),
                wp.from_torch(tasks["task_q_count"]),
                wp.from_torch(tasks["task_p_begin"]),
                wp.from_torch(tasks["task_p_count"]),
                wp.from_torch(lane_iota),
                wp.from_torch(queries.q_xyz4_sorted),
                wp.from_torch(queries.q_norm_sorted),
                wp.from_torch(queries.q_orig_sorted),
                wp.from_torch(index.pc_xyz4_sorted),
                wp.from_torch(index.pc_norm_sorted),
                wp.from_torch(index.pc_orig_sorted),
                float(radius) * float(radius),
                int(max_points),
                bool(return_dists),
                bool(return_points),
                idx_wp,
                count_wp,
                dist_wp,
                pts_wp,
            ],
            block_dim=TILE_P,
            device=wp_device,
            stream=wp_stream,
        )

    return out_idx, out_pts, out_dist, out_count.clamp(max=max_points).view(B, Q)


def _empty_outputs(
    B, Q, max_points, return_dists, return_points, was_unbatched, dtype, device
):
    """Build a zero/empty 4-tuple matching the op contract (used for empty inputs)."""
    indices = torch.zeros((B, Q, max_points), dtype=torch.int32, device=device)
    num = torch.zeros((B, Q), dtype=torch.int32, device=device)
    pts = (
        torch.zeros((B, Q, max_points, 3), dtype=dtype, device=device)
        if return_points
        else torch.empty((0, max_points, 3), dtype=dtype, device=device)
    )
    dist = (
        torch.zeros((B, Q, max_points), dtype=dtype, device=device)
        if return_dists
        else torch.empty(0, dtype=dtype, device=device)
    )
    if was_unbatched:
        indices = indices.squeeze(0)
        num = num.squeeze(0)
        if return_points:
            pts = pts.squeeze(0)
        if return_dists:
            dist = dist.squeeze(0)
    return indices, pts, dist, num


def radius_search_morton_dense(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    return_dists: bool = False,
    return_points: bool = False,
    variant: str | None = "dense_fma",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Morton-sorted compact dense-cell radius search entry point (CUDA-only).

    Builds the reusable :class:`MortonCompactDenseIndex` from ``points``, prepares
    the queries, and dispatches to the requested ``variant``. Returns the same
    4-tuple contract as the ``max_points`` path of ``radius_search_impl``.

    Args:
        points: Reference points, ``(N, 3)`` or ``(B, N, 3)``.
        queries: Query points, ``(M, 3)`` or ``(B, M, 3)``.
        radius: Search radius.
        max_points: Maximum neighbors per query (must not be ``None``).
        return_dists: Whether to return neighbor distances.
        return_points: Whether to return neighbor coordinates.
        variant: One of ``"dense_fma"`` (production), ``"dense_fma_e2e"`` (sort-free
            ``dense_fma``: row-major cell bins + unsorted queries, no radix sort),
            ``"dense_fma_store_opt"`` (``dense_fma`` with shared-memory staged flush
            and host-gathered points -- optimizes the output stores),
            ``"dense_fma_mem_opt"``, ``"dense_fma_mm"``, or ``"dense_gemm"``
            (benchmark), or ``"custom_grid_cell"`` (WIP experimental cell-centric
            memory-optimized launch -- not runnable until its kernel is finished).
            ``None`` defaults to ``"dense_fma"``.

    Returns:
        ``(indices, points, distances, num_neighbors)`` mirroring
        ``radius_search_impl``.

    Raises:
        ValueError: If ``max_points`` is ``None``, inputs are not CUDA, or
            ``variant`` is unknown.
    """
    if max_points is None:
        raise ValueError("radius_search_morton_dense requires max_points (not None)")
    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")
    if points.device.type != "cuda":
        raise ValueError("radius_search_morton_dense requires CUDA tensors")

    variant = variant or "dense_fma"
    if variant not in DENSE_VARIANTS:
        raise ValueError(
            f"unknown dense morton variant '{variant}'; expected one of {DENSE_VARIANTS}"
        )

    # custom_grid_cell is self-search-only: its kernel (radius_search_mem_optimized)
    # ignores the query set and searches the points against themselves. Capture whether
    # points/queries alias the same storage now -- before validate_inputs reshapes them --
    # so that branch can reject cross-search calls instead of returning silently-wrong
    # neighbors.
    if os.environ.get('PROFILE_RUN','0') =='1':
        torch.cuda.nvtx.range_push('RADIUS SEARCH MORTON')

    is_self_search = points is queries or (
        points.shape == queries.shape
        and points.stride() == queries.stride()
        and points.data_ptr() == queries.data_ptr()
    )
    
    points, queries, was_unbatched = validate_inputs(points, queries)

    
    input_dtype = points.dtype
    if points.dtype != torch.float32:
        points = points.to(torch.float32)
    if queries.dtype != torch.float32:
        queries = queries.to(torch.float32)
    
    if os.environ.get('PROFILE_RUN','0') =='1':
        torch.cuda.nvtx.range_push('points contiguous')

    points = points.contiguous()
    queries = queries.contiguous()

    if os.environ.get('PROFILE_RUN','0') =='1':
        torch.cuda.nvtx.range_pop()
    
    B, N, _ = points.shape
    Q = queries.shape[1]
    if N == 0 or Q == 0:
        # Close the range opened above; the early return would otherwise
        # leave a dangling NVTX push (mirrors the sparse-hash e2e path).
        if os.environ.get('PROFILE_RUN', '0') == '1':
            torch.cuda.nvtx.range_pop()  # RADIUS SEARCH MORTON
        return _empty_outputs(
            B, Q, max_points, return_dists, return_points, was_unbatched,
            input_dtype, points.device,
        )

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points)

    # dense_fma_e2e is the sort-free twin of dense_fma: it drops both Morton radix
    # sorts (cell-bin ordering and query ordering), which are locality-only for the
    # one-block-per-query FMA search, and reuses the same search kernel.
    is_e2e = variant == "dense_fma_e2e" or variant == "custom_grid_cell" or variant == 'custom_grid_cell_2'

    with wp.ScopedStream(wp_stream, sync_enter= False):
        # import ipdb; ipdb.set_trace()
        if os.environ.get('PROFILE_RUN','0') =='1':
            torch.cuda.nvtx.range_push('build morton index')
        
        index = build_morton_compact_dense_index(
            points, radius, wp_device, wp_stream, sort_cells=not is_e2e, is_e2e= is_e2e
        )

        if os.environ.get('PROFILE_RUN','0') =='1':
            torch.cuda.nvtx.range_pop()

        # sorted_queries = None
        # if variant != "custom_grid_cell_2":
        #     sorted_queries = prepare_morton_sorted_queries(
        #         index, queries, wp_device, wp_stream, sort_queries=not is_e2e
        #     )

        if os.environ.get('PROFILE_RUN','0') =='1':
            torch.cuda.nvtx.range_push('prep morton sorted queries')
        
        # sorted_queries = queries
        # if not is_e2e:
        sorted_queries = prepare_morton_sorted_queries(
            index, queries, wp_device, wp_stream, sort_queries=not is_e2e
        )

        if os.environ.get('PROFILE_RUN','0') =='1':
            torch.cuda.nvtx.range_pop()
        
        if os.environ.get('PROFILE_RUN','0') =='1':
            torch.cuda.nvtx.range_push('Radius Search')
        
        if variant in ("dense_fma", "dense_fma_e2e"):
            indices, pts, dist, num = radius_search_morton_warp_fma(
                index, sorted_queries, radius, max_points, return_dists,
                return_points, wp_device, wp_stream,
            )
        elif variant == "dense_fma_store_opt":
            indices, pts, dist, num = radius_search_morton_warp_fma_store_opt(
                index, sorted_queries, points, radius, max_points, return_dists,
                return_points, wp_device, wp_stream,
            )
        elif variant == "dense_fma_mem_opt":
            indices, pts, dist, num = radius_search_morton_warp_fma_mem_opt(
                index, sorted_queries, radius, max_points, return_dists,
                return_points, wp_device, wp_stream,
            )
        elif variant == "dense_fma_mm":
            indices, pts, dist, num = radius_search_morton_warp_fma_mm(
                index, sorted_queries, radius, max_points, return_dists,
                return_points, wp_device, wp_stream,
            )
        elif variant == "custom_grid_cell":
            # WIP: experimental cell-centric memory-optimized launch (one block per
            # (batch, cell)), selected via PHYSICSNEMO_RADIUS_SEARCH_MORTON=custom_grid_cell
            # or implementation="custom_grid_cell". The launch and its
            # `radius_search_mem_optimized` kernel are still being authored (the kernel
            # is currently parked as an inert string in _morton_dense_kernels.py), so
            # this path is NOT runnable yet. The unpack order mirrors the launch's
            # current `(out_idx, out_count, out_dist, out_points)` return signature.
            # if not is_self_search:
            #     raise ValueError(
            #         "custom_grid_cell is a self-search-only radius-search variant: its "
            #         "kernel (radius_search_mem_optimized) ignores the query set and "
            #         "searches the points against themselves, so it is valid only when "
            #         "queries is the same tensor as points. This call has queries != "
            #         "points (a cross-search). Unset PHYSICSNEMO_RADIUS_SEARCH_MORTON "
            #         "(or select variant 'dense_fma') to use the cross-search path."
            #     )
            indices, num, dist, pts = radius_search_mem_optimized_launch(
                index, sorted_queries, radius, max_points, return_dists,
                return_points, wp_device, wp_stream,
            )
        elif variant == 'custom_grid_cell_2':
            indices, num, dist, pts = radius_search_mem_optimized_2_launch(
                index,
                radius,
                max_points,
                return_dists,
                return_points,
                wp_device,
                wp_stream,
            )
        else:
            tasks = _build_gemm_tasks(index, sorted_queries, wp_device, wp_stream)
            indices, pts, dist, num = radius_search_morton_tiled_gemm(
                index, sorted_queries, tasks, radius, max_points, return_dists,
                return_points, wp_device, wp_stream,
            )
        if os.environ.get('PROFILE_RUN','0') =='1':
            torch.cuda.nvtx.range_pop()

    if pts is None:
        pts = torch.empty((0, max_points, 3), dtype=torch.float32, device=points.device)
    if dist is None:
        dist = torch.empty(0, dtype=torch.float32, device=points.device)

    if was_unbatched:
        indices = indices.squeeze(0)
        num = num.squeeze(0)
        if return_points:
            pts = pts.squeeze(0)
        if return_dists:
            dist = dist.squeeze(0)

    pts = pts.to(input_dtype)
    dist = dist.to(input_dtype)
    if os.environ.get('PROFILE_RUN','0') =='1':
        torch.cuda.nvtx.range_pop()
    return indices, pts, dist, num


# def radius_search_mem_optimized(
#     index,
#     queries, 
#     radius, 
#     max_points, 
#     return_dists,
#     return_points,
#     wp_device,
#     wp_stream
# ):

#     B, Q, Nq = queries.B, queries.Q, queries.Nq
#     device = index.pc_xyz4_sorted.device

#     def __alloc_outputs(B, Q, max_points, return_dists, return_points, device):
#         pass 
    
#     grid_x, grid_y, grid_z = index.grid_x, index.grid_y, index.grid_z

#     int num_cells = grid_x * grid_y * grid_z
#     TILE_DIM = 16
#     BLOCK_DIM= 128
    
#     wp.launch_tiled(
#         kernel = ...,
#         dim = num_cells,
#         inputs = [

#             TILE_DIM
#         ],
#         block_dim = BLOCK_DIM,
#         device = wp_device,
#         stream= wp_stream  
#     )

#     return ....outputs


def radius_search_mem_optimized_launch(
    index, queries, radius, max_points, return_dists, return_points, wp_device, wp_stream,
    use_matmul=True,
):
    # The kernel caps neighbours at the compile-time MAX_POINTS and always writes
    # MAX_POINTS output slots per query (its shared staging tiles are sized
    # (TILE_DIM, MAX_POINTS)); the out_* arrays below are only ``max_points`` rows, so a
    # mismatch makes the kernel write out of bounds. Enforce equality instead of the OOB.
    # if int(max_points) != int(mk.MAX_POINTS.val):
    #     raise ValueError(
    #         f"custom_grid_cell requires max_points == compile-time MAX_POINTS "
    #         f"(={int(mk.MAX_POINTS.val)}); got {int(max_points)}. Rebuild the module with "
    #         f"MAX_POINTS set to your max_points (in _morton_dense_kernels.py), or use the "
    #         f"dense_fma variant, which supports arbitrary max_points."
    #     )

    # One block per *global* dense cell row. Cells are laid out per batch in a
    # variable-size row-major grid and concatenated, so the launch spans
    # ``total_cells`` (== grid_base[B]) and the kernel recovers the batch from
    # grid_base -- the same per-batch indexing the FMA search uses. ``Ntot`` is the
    # count of sorted points across all batches (the output row dimension).
    B           = index.B
    Ntot        = index.Np
    total_cells = index.total_cells
    MP          = max_points
    # import ipdb; ipdb.set_trace()
    # ---- allocate outputs TRANSPOSED (slot-major) so the kernel write coalesces on qg ----
    out_idx   = wp.full((MP, Ntot), -1, dtype=wp.int32,   device=wp_device)
    out_count = wp.zeros(Ntot,          dtype=wp.int32,   device=wp_device)
    out_dist  = (wp.zeros((MP, Ntot),    dtype=wp.float32, device=wp_device)
                 if return_dists  else wp.zeros((1, 1),    dtype=wp.float32, device=wp_device))
    out_points = (wp.zeros((MP, Ntot, 3), dtype=wp.float32, device=wp_device)
                  if return_points else wp.zeros((1, 1, 3), dtype=wp.float32, device=wp_device))

    BLOCK_DIM = 16
    kernel = mk.radius_search_mem_optimized if use_matmul else mk.radius_search_scan

    # Transposed coords (3, Ntot_pad) so the kernel loads Pᵀ as a real (K, N) tile_matmul
    # operand instead of a tile_transpose() view, which tile_matmul rejects (Warp #1527).
    pc_t = index.pc_xyz4_sorted[:, :3].t().contiguous()

    # One block per *occupied* cell, not per dense cell. total_cells scales as
    # (domain_extent / radius)^3 and is dominated by empty cells for small radii, so
    # dim=total_cells (~1e9 at radius 0.01) pegs the GPU scheduling empty-cell blocks
    # that immediately return. Occupied cells number <= Ntot, so restrict the launch grid
    # to them. NOTE: torch.nonzero is still one O(total_cells) scan of cell_count_by_row;
    # a cheaper source (unique of point_cell_row during index build) is a follow-up.
    # import ipdb; ipdb.set_trace()
    
    occupied_rows = (
        torch.nonzero(index.cell_count_by_row, as_tuple=False).squeeze(1).to(torch.int32).contiguous()
    )
    num_occupied = int(occupied_rows.numel())
    # TEMP diagnostic (remove once validated): confirms the launch-grid reduction.
    
    print(
        f"[custom_grid_cell] radius={float(radius):.4g} B={int(B)} Ntot={int(Ntot)} "
        f"total_cells={int(total_cells)} occupied={num_occupied} "
        f"(launching {num_occupied} blocks instead of {int(total_cells)})",
        flush=True,
    )
    if num_occupied > 0:
        # import ipdb; ipdb.set_trace()
        wp.launch_tiled(
            kernel=kernel,
            dim=num_occupied,                       # one block per OCCUPIED dense cell row
            inputs=[
                wp.from_torch(index.pc_xyz4_sorted),    # (Ntot_pad, 4) Morton-sorted coords
                wp.from_torch(pc_t),                    # (3, Ntot_pad) transposed coords for Pᵀ loads
                int(B),
                wp.from_torch(index.grid_x),            # (B,) per-batch cell extents
                wp.from_torch(index.grid_y),
                wp.from_torch(index.grid_z),
                wp.from_torch(index.grid_base),         # (B+1,) per-batch cell base offsets
                wp.from_torch(index.cell_begin_by_row),  # (total_cells,)
                wp.from_torch(index.cell_count_by_row),  # (total_cells,)
                wp.from_torch(occupied_rows),           # (num_occupied,) launch-grid -> global cell id
                float(radius),
                int(max_points),
                bool(return_dists),
                bool(return_points),
                out_idx,
                out_count,
                out_dist,
                out_points
            ],
            block_dim=BLOCK_DIM, device=wp_device, stream=wp_stream,
        )
    # ---- convert Warp outputs -> torch and reconcile with the dense_fma contract ----
    # The kernel writes in SORTED point order (row qg = sorted slot), with SORTED
    # neighbor ids, transposed (slot-major, (MP, Np)) and flat across batches. The
    # reference (dense_fma) returns (B, P, MP) torch tensors in ORIGINAL query order with
    # LOCAL original point ids. This variant is self-search (queries == points, enforced
    # by the caller's guard), so the query axis IS the point axis: points are grouped
    # batch-major, so reshape Np == B*P -> (B, P), then unsort each row to its original
    # position and remap neighbor ids -- both via pc_orig_sorted.
    Bx, Px = index.B, index.P
    dev = index.pc_xyz4_sorted.device
    orig = index.pc_orig_sorted.to(torch.long)                     # (Np,) sorted slot -> local id
    orig_bp = orig.view(Bx, Px)                                    # (B, P)
    batch_ix = torch.arange(Bx, device=dev).view(Bx, 1).expand(Bx, Px)  # (B, P)

    # out_idx: (MP, Np) -> (Np, MP); remap sorted neighbor slot -> local original id,
    # padding slots (-1) -> 0 (mirrors dense_fma's zero-initialized index buffer).
    idx_s = wp.to_torch(out_idx).transpose(0, 1).contiguous()      # (Np, MP) int32
    valid = idx_s >= 0
    remapped = orig[idx_s.clamp_min(0).long()].to(torch.int32)     # (Np, MP) local ids
    idx_local = torch.where(valid, remapped, torch.zeros_like(remapped)).view(Bx, Px, MP)
    cnt_s = wp.to_torch(out_count).view(Bx, Px)                    # (B, P)

    # Scatter sorted rows back to original query order: sorted slot s in batch b belongs
    # to original local query orig_bp[b, s] (a per-batch permutation of [0, P)).
    indices = idx_local.new_zeros((Bx, Px, MP))
    indices[batch_ix, orig_bp] = idx_local
    num = cnt_s.new_zeros((Bx, Px))
    num[batch_ix, orig_bp] = cnt_s

    dist = None
    if return_dists:
        # The kernel stores SQUARED distance; dense_fma returns Euclidean -> sqrt.
        dist_s = wp.to_torch(out_dist).transpose(0, 1).contiguous().view(Bx, Px, MP)
        dist_s = torch.sqrt(dist_s.clamp_min(0.0))
        dist = dist_s.new_zeros((Bx, Px, MP))
        dist[batch_ix, orig_bp] = dist_s

    pts = None
    if return_points:
        # out_points: (MP, Np, 3) -> (Np, MP, 3); values are actual neighbor coordinates.
        pts_s = wp.to_torch(out_points).transpose(0, 1).contiguous().view(Bx, Px, MP, 3)
        pts = pts_s.new_zeros((Bx, Px, MP, 3))
        pts[batch_ix, orig_bp] = pts_s

    return indices, num, dist, pts



    

def _build_custom_grid_cell_2_tasks(index: MortonCompactDenseIndex):
    """Build one launch task per ``(occupied cell, query tile)`` pair.

    The compacted point range of a cell is also its query range for this
    self-search variant. Dense cells therefore contribute multiple 32-query
    tasks, while empty cells contribute none.
    """
    cell_count = index.cell_count_by_row
    occupied_rows = torch.nonzero(cell_count, as_tuple=False).squeeze(1)

    if occupied_rows.numel() == 0:
        empty = torch.empty(0, dtype=torch.int32, device=cell_count.device)
        return empty, empty, empty

    points_per_cell = cell_count[occupied_rows]
    tiles_per_cell = torch.div(
        points_per_cell + MEM_OPT_2_BLOCK_DIM - 1,
        MEM_OPT_2_BLOCK_DIM,
        rounding_mode="floor",
    ).to(torch.long)

    task_cell = torch.repeat_interleave(
        occupied_rows.to(torch.int32), tiles_per_cell
    ).contiguous()
    num_tasks = task_cell.numel()

    # Convert each flattened task id into its tile number within the cell.
    first_task_for_cell = torch.cumsum(tiles_per_cell, dim=0) - tiles_per_cell
    task_local_tile = torch.arange(
        num_tasks, dtype=torch.long, device=cell_count.device
    ) - torch.repeat_interleave(first_task_for_cell, tiles_per_cell)

    task_cell_long = task_cell.to(torch.long)
    task_q_begin = (
        index.cell_begin_by_row[task_cell_long].to(torch.long)
        + task_local_tile * MEM_OPT_2_BLOCK_DIM
    ).to(torch.int32)
    task_q_count = torch.clamp(
        cell_count[task_cell_long]
        - task_local_tile.to(torch.int32) * MEM_OPT_2_BLOCK_DIM,
        max=MEM_OPT_2_BLOCK_DIM,
    ).to(torch.int32)

    return task_cell, task_q_begin.contiguous(), task_q_count.contiguous()


def radius_search_mem_optimized_2_launch(
    index: MortonCompactDenseIndex,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """Launch the direct-FMA cell/query-tile radius-search kernel.

    The Warp kernel writes compacted/sorted neighbor indices and optional squared
    distances in slot-major layout. This function remaps both query rows and
    neighbor ids to the public batch/local-id layout. Neighbor coordinates are
    gathered from the returned indices outside the search kernel.
    """
    batch_size = index.B
    points_per_batch = index.P
    num_points = index.Np
    num_slots = int(max_points)

    out_idx = wp.full(
        (num_slots, num_points), -1, dtype=wp.int32, device=wp_device
    )
    out_count = wp.zeros(num_points, dtype=wp.int32, device=wp_device)
    out_dist = (
        wp.zeros(
            (num_slots, num_points), dtype=wp.float32, device=wp_device
        )
        if return_dists
        else wp.zeros((1, 1), dtype=wp.float32, device=wp_device)
    )

    task_cell, task_q_begin, task_q_count = _build_custom_grid_cell_2_tasks(index)
    num_tasks = task_cell.numel()

    if num_tasks > 0:
        wp.launch_tiled(
            kernel=mk.radius_search_mem_optimized_3,
            dim=num_tasks,
            inputs=[
                wp.from_torch(index.pc_xyz4_sorted),
                int(batch_size),
                wp.from_torch(index.grid_x),
                wp.from_torch(index.grid_y),
                wp.from_torch(index.grid_z),
                wp.from_torch(index.grid_base),
                wp.from_torch(index.cell_begin_by_row),
                wp.from_torch(index.cell_count_by_row),
                wp.from_torch(index.offsets),
                wp.from_torch(task_cell),
                wp.from_torch(task_q_begin),
                wp.from_torch(task_q_count),
                float(radius),
                num_slots,
                bool(return_dists),
                out_idx,
                out_count,
                out_dist,
            ],
            block_dim=MEM_OPT_2_BLOCK_DIM,
            device=wp_device,
            stream=wp_stream,
        )

    # Convert slot-major Warp output to one row per sorted query.
    idx_sorted = wp.to_torch(out_idx).transpose(0, 1).contiguous()
    count_sorted = wp.to_torch(out_count)
    valid = idx_sorted >= 0
    safe_sorted_idx = idx_sorted.clamp_min(0).to(torch.long)

    # A compacted slot maps to a local original point id. Point bins are batch
    # contiguous, so each batch occupies exactly points_per_batch sorted slots.
    original_id = index.pc_orig_sorted.to(torch.long)
    original_id_by_batch = original_id.view(batch_size, points_per_batch)
    batch_index = torch.arange(
        batch_size, device=idx_sorted.device
    ).view(batch_size, 1).expand(batch_size, points_per_batch)

    remapped_idx = original_id[safe_sorted_idx].to(torch.int32)
    remapped_idx = torch.where(
        valid, remapped_idx, torch.zeros_like(remapped_idx)
    ).view(batch_size, points_per_batch, num_slots)
    count_sorted = count_sorted.view(batch_size, points_per_batch)

    indices = remapped_idx.new_zeros(
        (batch_size, points_per_batch, num_slots)
    )
    indices[batch_index, original_id_by_batch] = remapped_idx
    num = count_sorted.new_zeros((batch_size, points_per_batch))
    num[batch_index, original_id_by_batch] = count_sorted

    dist = None
    if return_dists:
        dist_sorted = wp.to_torch(out_dist).transpose(0, 1).contiguous()
        dist_sorted = torch.sqrt(dist_sorted.clamp_min(0.0)).view(
            batch_size, points_per_batch, num_slots
        )
        dist = dist_sorted.new_zeros(
            (batch_size, points_per_batch, num_slots)
        )
        dist[batch_index, original_id_by_batch] = dist_sorted

    pts = None
    if return_points:
        # Gather in sorted-index space before neighbor ids are remapped. Invalid
        # output slots contain -1, so clamp for the gather and mask them afterward.
        points_sorted = index.pc_xyz4_sorted[safe_sorted_idx, :3]
        points_sorted = torch.where(
            valid.unsqueeze(-1), points_sorted, torch.zeros_like(points_sorted)
        ).view(batch_size, points_per_batch, num_slots, 3)
        pts = points_sorted.new_zeros(
            (batch_size, points_per_batch, num_slots, 3)
        )
        pts[batch_index, original_id_by_batch] = points_sorted

    return indices, num, dist, pts