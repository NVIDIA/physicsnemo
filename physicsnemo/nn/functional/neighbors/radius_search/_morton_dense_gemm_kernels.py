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
GEMM benchmark kernels for the Morton-sorted compact dense-cell radius search.

These live in a separate module from ``_morton_dense_kernels.py`` because Warp
compiles a whole module for one ``block_dim`` at a time. The production FMA
kernel launches at ``block_dim=32`` while ``radius_search_dense_gemm_kernel``
needs ``block_dim=TILE_P`` (256) for its ``wp.untile`` over the point tile; if
they shared a module, compiling for the FMA launch would fail to build the GEMM
kernel. Keeping the GEMM path here means this module is only ever compiled at
``TILE_P``.

The query-tile / point-chunk task generation kernels live here too so the whole
GEMM path is self-contained.
"""

import warp as wp

from ._morton_dense_kernels import TILE_P, TILE_Q


@wp.kernel
def count_neighbor_tasks(
    q_tile_b: wp.array(dtype=wp.int32),
    q_tile_cx: wp.array(dtype=wp.int32),
    q_tile_cy: wp.array(dtype=wp.int32),
    q_tile_cz: wp.array(dtype=wp.int32),
    neighbor_offsets: wp.array2d(dtype=wp.int32),
    min_cx: wp.array(dtype=wp.int32),
    min_cy: wp.array(dtype=wp.int32),
    min_cz: wp.array(dtype=wp.int32),
    grid_x: wp.array(dtype=wp.int32),
    grid_y: wp.array(dtype=wp.int32),
    grid_z: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    cell_count_by_row: wp.array(dtype=wp.int32),
    tile_p: wp.int32,
    task_count: wp.array(dtype=wp.int32),
):
    """Count the ``(query-tile, point-chunk)`` tasks each query tile produces."""
    qt = wp.tid()
    b = q_tile_b[qt]
    cx = q_tile_cx[qt]
    cy = q_tile_cy[qt]
    cz = q_tile_cz[qt]
    gx = grid_x[b]
    gy = grid_y[b]
    gz = grid_z[b]
    total = wp.int32(0)
    for k in range(27):
        lx = cx + neighbor_offsets[k, 0] - min_cx[b]
        ly = cy + neighbor_offsets[k, 1] - min_cy[b]
        lz = cz + neighbor_offsets[k, 2] - min_cz[b]
        if lx >= 0 and lx < gx and ly >= 0 and ly < gy and lz >= 0 and lz < gz:
            row = grid_base[b] + lx + gx * (ly + gy * lz)
            plen = cell_count_by_row[row]
            if plen > 0:
                total += (plen + tile_p - 1) // tile_p
    task_count[qt] = total


@wp.kernel
def fill_neighbor_tasks(
    q_tile_begin: wp.array(dtype=wp.int32),
    q_tile_count: wp.array(dtype=wp.int32),
    q_tile_b: wp.array(dtype=wp.int32),
    q_tile_cx: wp.array(dtype=wp.int32),
    q_tile_cy: wp.array(dtype=wp.int32),
    q_tile_cz: wp.array(dtype=wp.int32),
    task_offset: wp.array(dtype=wp.int32),
    neighbor_offsets: wp.array2d(dtype=wp.int32),
    min_cx: wp.array(dtype=wp.int32),
    min_cy: wp.array(dtype=wp.int32),
    min_cz: wp.array(dtype=wp.int32),
    grid_x: wp.array(dtype=wp.int32),
    grid_y: wp.array(dtype=wp.int32),
    grid_z: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    cell_begin_by_row: wp.array(dtype=wp.int32),
    cell_count_by_row: wp.array(dtype=wp.int32),
    tile_p: wp.int32,
    task_q_begin: wp.array(dtype=wp.int32),
    task_q_count: wp.array(dtype=wp.int32),
    task_p_begin: wp.array(dtype=wp.int32),
    task_p_count: wp.array(dtype=wp.int32),
):
    """Materialize the task list at each query tile's exclusive-scan offset."""
    qt = wp.tid()
    out = task_offset[qt]
    qb = q_tile_begin[qt]
    qc = q_tile_count[qt]
    b = q_tile_b[qt]
    cx = q_tile_cx[qt]
    cy = q_tile_cy[qt]
    cz = q_tile_cz[qt]
    gx = grid_x[b]
    gy = grid_y[b]
    gz = grid_z[b]
    for k in range(27):
        lx = cx + neighbor_offsets[k, 0] - min_cx[b]
        ly = cy + neighbor_offsets[k, 1] - min_cy[b]
        lz = cz + neighbor_offsets[k, 2] - min_cz[b]
        if lx >= 0 and lx < gx and ly >= 0 and ly < gy and lz >= 0 and lz < gz:
            row = grid_base[b] + lx + gx * (ly + gy * lz)
            ps = cell_begin_by_row[row]
            pe = ps + cell_count_by_row[row]
            pb = ps
            while pb < pe:
                task_q_begin[out] = qb
                task_q_count[out] = qc
                task_p_begin[out] = pb
                task_p_count[out] = wp.min(tile_p, pe - pb)
                out += 1
                pb += tile_p


@wp.kernel
def radius_search_dense_gemm_kernel(
    task_q_begin: wp.array(dtype=wp.int32),
    task_q_count: wp.array(dtype=wp.int32),
    task_p_begin: wp.array(dtype=wp.int32),
    task_p_count: wp.array(dtype=wp.int32),
    lane_iota: wp.array(dtype=wp.int32),
    q_xyz4_sorted: wp.array2d(dtype=wp.float32),
    q_norm_sorted: wp.array(dtype=wp.float32),
    q_orig_sorted: wp.array(dtype=wp.int32),
    pc_xyz4_sorted: wp.array2d(dtype=wp.float32),
    pc_norm_sorted: wp.array(dtype=wp.float32),
    pc_orig_sorted: wp.array(dtype=wp.int32),
    radius2: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    return_points: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
    out_pts: wp.array2d(dtype=wp.vec3),
):
    """Tiled GEMM distances :math:`\\lVert Q\\rVert^2 + \\lVert P\\rVert^2 - 2 Q P^T`.

    Launched with ``wp.launch_tiled(dim=[num_tasks], block_dim=TILE_P)``; one block
    per task, one lane per candidate point. ``out_count`` is an atomic slot counter
    that can overshoot ``max_points``, so the host clamps it afterward.
    """
    task_id = wp.tid()
    qb = task_q_begin[task_id]
    qc = task_q_count[task_id]
    pb = task_p_begin[task_id]
    pcount = task_p_count[task_id]

    q_tile = wp.tile_load(q_xyz4_sorted, shape=(TILE_Q, 4), offset=(qb, 0), storage="shared")
    p_tile = wp.tile_load(pc_xyz4_sorted, shape=(TILE_P, 4), offset=(pb, 0), storage="shared")
    dot_tile = wp.tile_zeros(shape=(TILE_Q, TILE_P), dtype=wp.float32)
    wp.tile_matmul(q_tile, wp.tile_transpose(p_tile), dot_tile)
    dot = wp.untile(dot_tile)
    lane = wp.untile(wp.tile_load(lane_iota, shape=(TILE_P,)))

    if lane < pcount:
        sp = pb + lane
        p_norm = pc_norm_sorted[sp]
        porig = pc_orig_sorted[sp]
        for iq in range(TILE_Q):
            if iq < qc:
                flat_q = q_orig_sorted[qb + iq]
                if out_count[flat_q] < max_points:
                    d2 = q_norm_sorted[qb + iq] + p_norm - 2.0 * dot[iq]
                    if d2 <= radius2:
                        slot = wp.atomic_add(out_count, flat_q, 1)
                        if slot < max_points:
                            out_idx[flat_q, slot] = porig
                            if return_dists:
                                out_dist[flat_q, slot] = wp.sqrt(wp.max(d2, 0.0))
                            if return_points:
                                out_pts[flat_q, slot] = wp.vec3(
                                    pc_xyz4_sorted[sp, 0],
                                    pc_xyz4_sorted[sp, 1],
                                    pc_xyz4_sorted[sp, 2],
                                )


# ---------------------------------------------------------------------------
# In-kernel-gather matmul search: one block per query, per-chunk tiled matmul
# ---------------------------------------------------------------------------


@wp.kernel
def radius_search_dense_fma_kernel_mm(
    q_xyz4_sorted: wp.array2d(dtype=wp.float32),
    q_norm_sorted: wp.array(dtype=wp.float32),
    q_orig_sorted: wp.array(dtype=wp.int32),
    inv_radius: wp.float32,
    Q: wp.int32,
    neighbor_offsets: wp.array2d(dtype=wp.int32),
    min_cx: wp.array(dtype=wp.int32),
    min_cy: wp.array(dtype=wp.int32),
    min_cz: wp.array(dtype=wp.int32),
    grid_x: wp.array(dtype=wp.int32),
    grid_y: wp.array(dtype=wp.int32),
    grid_z: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    cell_begin_by_row: wp.array(dtype=wp.int32),
    cell_count_by_row: wp.array(dtype=wp.int32),
    pc_xyz4_sorted: wp.array2d(dtype=wp.float32),
    pc_norm_sorted: wp.array(dtype=wp.float32),
    pc_orig_sorted: wp.array(dtype=wp.int32),
    radius2: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    return_points: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
    out_pts: wp.array2d(dtype=wp.vec3),
):
    """One block per query: scan 27 dense cells, distances via per-chunk matmul.

    Same per-query 27-cell scan as ``radius_search_dense_fma_kernel``, but each
    ``TILE_P``-wide candidate chunk is loaded as a tile and its squared distances
    come from the norm expansion :math:`\\lVert q\\rVert^2 + \\lVert p\\rVert^2 -
    2 q p^T` with a tiled matmul of the single query row against the chunk, then a
    radius mask. Hits are compacted with the same block-wide tile scan as the FMA
    kernel. Launched with ``wp.launch_tiled(dim=[Nq], block_dim=TILE_P)``.
    """
    sorted_q, lane = wp.tid()
    flat_q = q_orig_sorted[sorted_q]
    b = flat_q // Q
    q_norm = q_norm_sorted[sorted_q]
    gx = grid_x[b]
    gy = grid_y[b]
    gz = grid_z[b]
    mcx = min_cx[b]
    mcy = min_cy[b]
    mcz = min_cz[b]
    gbase = grid_base[b]
    lqx = wp.int32(wp.floor(q_xyz4_sorted[sorted_q, 0] * inv_radius)) - mcx
    lqy = wp.int32(wp.floor(q_xyz4_sorted[sorted_q, 1] * inv_radius)) - mcy
    lqz = wp.int32(wp.floor(q_xyz4_sorted[sorted_q, 2] * inv_radius)) - mcz

    q_tile = wp.tile_load(q_xyz4_sorted, shape=(1, 4), offset=(sorted_q, 0), storage="shared")

    found = wp.int32(0)
    for n in range(27):
        lx = lqx + neighbor_offsets[n, 0]
        ly = lqy + neighbor_offsets[n, 1]
        lz = lqz + neighbor_offsets[n, 2]
        if lx >= 0 and lx < gx and ly >= 0 and ly < gy and lz >= 0 and lz < gz:
            row = gbase + lx + gx * (ly + gy * lz)
            start = cell_begin_by_row[row]
            end = start + cell_count_by_row[row]
            base = start
            while base < end:
                p_tile = wp.tile_load(
                    pc_xyz4_sorted, shape=(TILE_P, 4), offset=(base, 0), storage="shared"
                )
                dot_tile = wp.tile_zeros(shape=(1, TILE_P), dtype=wp.float32)
                wp.tile_matmul(q_tile, wp.tile_transpose(p_tile), dot_tile)
                dot = wp.untile(dot_tile)

                si = base + lane
                hit = wp.int32(0)
                d2 = wp.float32(0.0)
                if si < end:
                    d2 = q_norm + pc_norm_sorted[si] - 2.0 * dot[0]
                    if d2 <= radius2:
                        hit = wp.int32(1)
                hit_tile = wp.tile(hit)
                prefix = wp.untile(wp.tile_scan_exclusive(hit_tile))
                hits_tile = wp.tile_sum(hit_tile)
                write_hits = wp.min(hits_tile[0], max_points - found)
                if hit == 1 and prefix < write_hits:
                    slot = found + prefix
                    out_idx[flat_q, slot] = pc_orig_sorted[si]
                    if return_dists:
                        out_dist[flat_q, slot] = wp.sqrt(wp.max(d2, 0.0))
                    if return_points:
                        out_pts[flat_q, slot] = wp.vec3(
                            pc_xyz4_sorted[si, 0],
                            pc_xyz4_sorted[si, 1],
                            pc_xyz4_sorted[si, 2],
                        )
                found += write_hits
                if found >= max_points:
                    break
                base += TILE_P
        if found >= max_points:
            break
    if lane == 0:
        out_count[flat_q] = found
