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
Warp kernels for the Morton-sorted compact dense-cell radius search backend.

This module is pure Warp code (no PyTorch). It is self-contained and does not
share device helpers with the hash-based Morton backend in ``_morton_kernels.py``.

The backend maps points to radius-sized cells, lays them out in a dense
row-major grid indexed directly by cell id (no hash table), and sorts the
compacted point bins in Morton-cell order for locality. The host glue that
drives these kernels lives in ``_morton_dense_impl.py``.
"""

import warp as wp

# Query-tile and point-chunk widths for the GEMM benchmark path. Module-level
# Python ints so they are captured as compile-time tile shapes. ``TILE_P`` also
# doubles as the GEMM launch ``block_dim`` (one lane per candidate point).
TILE_Q = 8
TILE_P = 256

# Threads per query block for the production FMA search kernel. This is both the
# launch block_dim and the candidate-chunk stride, so each query block scans this
# many candidate points per chunk and compacts hits with a block-wide tile scan.
# 32 is one warp; larger (e.g. 128) scans more candidates per chunk for densely
# populated cells, at the cost of idle lanes when cells hold fewer points.
FMA_BLOCK_DIM = 64

# Compile-time capacity of the per-query shared-memory staging buffer used by the
# production FMA kernel (Stage 2). Neighbors from all 27 cells accumulate in this
# block-shared tile and are flushed to global memory once, coalesced, so the hot
# loop never issues the old per-chunk partial-warp global scatter. This caps the
# supported ``max_points`` (the host raises if ``max_points`` exceeds it); 256
# covers every realistic ball-query fan-out (production uses <= 32, tests <= 128).
# ``FMA_STAGE_SLOTS`` adds one dump slot at index ``FMA_STAGE_CAP`` where inactive
# lanes park their (never-read-back) write so the shared-tile store stays a single
# block-uniform statement instead of a divergent, sync-splitting branch. 
FMA_STAGE_CAP = 256
FMA_STAGE_SLOTS = FMA_STAGE_CAP + 1


# ---------------------------------------------------------------------------
# Device helper functions
# ---------------------------------------------------------------------------


@wp.func
def morton_encode(x: wp.int32, y: wp.int32, z: wp.int32, bits: wp.int32) -> wp.int64:
    """Interleave the low ``bits`` of three non-negative coords into a Morton code."""
    code = wp.int64(0)
    for i in range(bits):
        bx = wp.int64((x >> i) & 1)
        by = wp.int64((y >> i) & 1)
        bz = wp.int64((z >> i) & 1)
        s = wp.int64(3 * i)
        code = code | (bx << s) | (by << (s + wp.int64(1))) | (bz << (s + wp.int64(2)))
    return code


@wp.func
def pack_batch_key(
    b: wp.int32,
    cx: wp.int32,
    cy: wp.int32,
    cz: wp.int32,
    bits: wp.int32,
    morton_bits: wp.int32,
) -> wp.int64:
    """Pack a batch id (high bits) and a 3D Morton code (low bits) into an int64 key."""
    return (wp.int64(b) << wp.int64(morton_bits)) | morton_encode(cx, cy, cz, bits)


# ---------------------------------------------------------------------------
# PC-index build kernels
# ---------------------------------------------------------------------------


@wp.kernel
def count_points_in_cells(
    points: wp.array2d(dtype=wp.vec3),
    inv_radius: wp.float32,
    P: wp.int32,
    min_cx: wp.array(dtype=wp.int32),
    min_cy: wp.array(dtype=wp.int32),
    min_cz: wp.array(dtype=wp.int32),
    grid_x: wp.array(dtype=wp.int32),
    grid_y: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    point_cell_row: wp.array(dtype=wp.int32),
    cell_count_by_row: wp.array(dtype=wp.int32),
):
    """Histogram points into dense row-major cells and record each point's cell row."""
    flat_p = wp.tid()
    b = flat_p // P
    pt = points[b, flat_p % P]
    lx = wp.int32(wp.floor(pt[0] * inv_radius)) - min_cx[b]
    ly = wp.int32(wp.floor(pt[1] * inv_radius)) - min_cy[b]
    lz = wp.int32(wp.floor(pt[2] * inv_radius)) - min_cz[b]
    row = grid_base[b] + lx + grid_x[b] * (ly + grid_y[b] * lz)
    point_cell_row[flat_p] = row
    wp.atomic_add(cell_count_by_row, row, 1)


@wp.kernel
def enumerate_cell_morton_keys(
    grid_x: wp.array(dtype=wp.int32),
    grid_y: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    B: wp.int32,
    bits: wp.int32,
    morton_bits: wp.int32,
    cell_key_tmp: wp.array(dtype=wp.int64),
    cell_row_tmp: wp.array(dtype=wp.int32),
):
    """Emit a (batch, Morton) sort key for every dense cell row."""
    row = wp.tid()
    b = wp.int32(0)
    while b + 1 < B and grid_base[b + 1] <= row:
        b += 1
    gx = grid_x[b]
    gy = grid_y[b]
    local = row - grid_base[b]
    lx = local % gx
    ly = (local // gx) % gy
    lz = local // (gx * gy)
    cell_key_tmp[row] = pack_batch_key(b, lx, ly, lz, bits, morton_bits)
    cell_row_tmp[row] = row


@wp.kernel
def gather_cell_counts_by_rank(
    cell_row_tmp: wp.array(dtype=wp.int32),
    cell_count_by_row: wp.array(dtype=wp.int32),
    cell_counts_rank: wp.array(dtype=wp.int32),
):
    """Reorder per-cell counts into Morton-rank order for the prefix sum."""
    rank = wp.tid()
    cell_counts_rank[rank] = cell_count_by_row[cell_row_tmp[rank]]


@wp.kernel
def fill_cell_row_ranges(
    cell_row_tmp: wp.array(dtype=wp.int32),
    cell_offsets_rank: wp.array(dtype=wp.int32),
    cell_begin_by_row: wp.array(dtype=wp.int32),
):
    """Scatter the Morton-rank prefix offsets back into row-indexed cell starts."""
    rank = wp.tid()
    cell_begin_by_row[cell_row_tmp[rank]] = cell_offsets_rank[rank]


@wp.kernel
def scatter_points_to_cells(
    points: wp.array2d(dtype=wp.vec3),
    point_cell_row: wp.array(dtype=wp.int32),
    P: wp.int32,
    cell_write_by_row: wp.array(dtype=wp.int32),
    pc_x_sorted: wp.array(dtype=wp.float32),
    pc_y_sorted: wp.array(dtype=wp.float32),
    pc_z_sorted: wp.array(dtype=wp.float32),
    pc_orig_sorted: wp.array(dtype=wp.int32),
    pc_xyz4_sorted: wp.array2d(dtype=wp.float32),
    pc_norm_sorted: wp.array(dtype=wp.float32),
):
    """Scatter each point into its Morton-cell-sorted compact slot."""
    flat_p = wp.tid()
    pt = points[flat_p // P, flat_p % P]
    dst = wp.atomic_add(cell_write_by_row, point_cell_row[flat_p], 1)
    pc_x_sorted[dst] = pt[0]
    pc_y_sorted[dst] = pt[1]
    pc_z_sorted[dst] = pt[2]
    pc_orig_sorted[dst] = flat_p % P
    pc_xyz4_sorted[dst, 0] = pt[0]
    pc_xyz4_sorted[dst, 1] = pt[1]
    pc_xyz4_sorted[dst, 2] = pt[2]
    pc_xyz4_sorted[dst, 3] = 0.0
    pc_norm_sorted[dst] = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]


# ---------------------------------------------------------------------------
# Query build kernels
# ---------------------------------------------------------------------------


@wp.kernel
def make_query_morton_keys(
    queries: wp.array2d(dtype=wp.vec3),
    inv_radius: wp.float32,
    Q: wp.int32,
    q_min_cx: wp.array(dtype=wp.int32),
    q_min_cy: wp.array(dtype=wp.int32),
    q_min_cz: wp.array(dtype=wp.int32),
    bits: wp.int32,
    morton_bits: wp.int32,
    q_key_tmp: wp.array(dtype=wp.int64),
    q_val_tmp: wp.array(dtype=wp.int32),
):
    """Emit a (batch, Morton) sort key per query using per-batch query-cell minima."""
    flat_q = wp.tid()
    b = flat_q // Q
    pt = queries[b, flat_q % Q]
    lx = wp.int32(wp.floor(pt[0] * inv_radius)) - q_min_cx[b]
    ly = wp.int32(wp.floor(pt[1] * inv_radius)) - q_min_cy[b]
    lz = wp.int32(wp.floor(pt[2] * inv_radius)) - q_min_cz[b]
    q_key_tmp[flat_q] = pack_batch_key(b, lx, ly, lz, bits, morton_bits)
    q_val_tmp[flat_q] = flat_q


@wp.kernel
def gather_sorted_queries(
    queries: wp.array2d(dtype=wp.vec3),
    q_val_tmp: wp.array(dtype=wp.int32),
    Q: wp.int32,
    q_orig_sorted: wp.array(dtype=wp.int32),
    q_x_sorted: wp.array(dtype=wp.float32),
    q_y_sorted: wp.array(dtype=wp.float32),
    q_z_sorted: wp.array(dtype=wp.float32),
    q_xyz4_sorted: wp.array2d(dtype=wp.float32),
    q_norm_sorted: wp.array(dtype=wp.float32),
):
    """Reorder queries into Morton order and record coords + GEMM data.

    The per-query batch id and absolute cell coords are not stored; consumers
    derive them as ``b = flat_q // Q`` and ``floor(coord / cell_size)``.
    """
    sorted_q = wp.tid()
    flat_q = q_val_tmp[sorted_q]
    pt = queries[flat_q // Q, flat_q % Q]
    q_orig_sorted[sorted_q] = flat_q
    q_x_sorted[sorted_q] = pt[0]
    q_y_sorted[sorted_q] = pt[1]
    q_z_sorted[sorted_q] = pt[2]
    q_xyz4_sorted[sorted_q, 0] = pt[0]
    q_xyz4_sorted[sorted_q, 1] = pt[1]
    q_xyz4_sorted[sorted_q, 2] = pt[2]
    q_xyz4_sorted[sorted_q, 3] = 0.0
    q_norm_sorted[sorted_q] = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]


# ---------------------------------------------------------------------------
# Production search: one warp per query, direct-FMA distances
# ---------------------------------------------------------------------------


@wp.kernel
def radius_search_dense_fma_kernel(
    q_x_sorted: wp.array(dtype=wp.float32),
    q_y_sorted: wp.array(dtype=wp.float32),
    q_z_sorted: wp.array(dtype=wp.float32),
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
    pc_x_sorted: wp.array(dtype=wp.float32),
    pc_y_sorted: wp.array(dtype=wp.float32),
    pc_z_sorted: wp.array(dtype=wp.float32),
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
    """One block per query: scan 27 dense cells, block-compact in-radius hits.

    Launched with ``wp.launch_tiled(dim=[Nq], block_dim=FMA_BLOCK_DIM)``. All lanes
    reach the tile ops every chunk (inactive lanes contribute ``hit = 0``); the loop
    conditions are block-uniform so the tile scan/sum and the early breaks stay
    collective.
    """
    sorted_q, lane = wp.tid()
    flat_q = q_orig_sorted[sorted_q]
    b = flat_q // Q
    qx = q_x_sorted[sorted_q]
    qy = q_y_sorted[sorted_q]
    qz = q_z_sorted[sorted_q]
    gx = grid_x[b]
    gy = grid_y[b]
    gz = grid_z[b]
    mcx = min_cx[b]
    mcy = min_cy[b]
    mcz = min_cz[b]
    gbase = grid_base[b]
    lqx = wp.int32(wp.floor(qx * inv_radius)) - mcx
    lqy = wp.int32(wp.floor(qy * inv_radius)) - mcy
    lqz = wp.int32(wp.floor(qz * inv_radius)) - mcz

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
                si = base + lane
                hit = wp.int32(0)
                d2 = wp.float32(0.0)
                px = wp.float32(0.0)
                py = wp.float32(0.0)
                pz = wp.float32(0.0)
                if si < end:
                    px = pc_x_sorted[si]
                    py = pc_y_sorted[si]
                    pz = pc_z_sorted[si]
                    dx = px - qx
                    dy = py - qy
                    dz = pz - qz
                    d2 = dx * dx + dy * dy + dz * dz
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
                        out_dist[flat_q, slot] = wp.sqrt(d2)
                    if return_points:
                        out_pts[flat_q, slot] = wp.vec3(px, py, pz)
                found += write_hits
                if found >= max_points:
                    break
                base += FMA_BLOCK_DIM
        if found >= max_points:
            break
    if lane == 0:
        out_count[flat_q] = found


# ---------------------------------------------------------------------------
# Store-optimized search variant: shared-mem staged flush + host-gathered points
# ---------------------------------------------------------------------------


@wp.kernel
def radius_search_dense_fma_store_opt_kernel(
    q_x_sorted: wp.array(dtype=wp.float32),
    q_y_sorted: wp.array(dtype=wp.float32),
    q_z_sorted: wp.array(dtype=wp.float32),
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
    pc_x_sorted: wp.array(dtype=wp.float32),
    pc_y_sorted: wp.array(dtype=wp.float32),
    pc_z_sorted: wp.array(dtype=wp.float32),
    pc_orig_sorted: wp.array(dtype=wp.int32),
    radius2: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
):
    """Store-optimized twin of :func:`radius_search_dense_fma_kernel`.

    Same one-block-per-query 27-cell scan and block-wide hit compaction, but with
    two changes that target the inefficient global stores NCU flagged on the
    baseline kernel:

    * Stage 1 -- neighbor *coordinates* are not stored here. The baseline's
      ``out_pts[flat_q, slot] = wp.vec3(...)`` was a 12-byte (misaligned) global
      scatter; callers that need coordinates gather them on the host from
      ``out_idx`` (see ``_gather_neighbor_points``), so this kernel writes only
      indices/distances.
    * Stage 2 -- hits from all 27 cells accumulate in a per-query block-shared
      staging tile (``stage_idx``/``stage_dist``) instead of a per-chunk global
      write. After the scan a single coalesced pass flushes the compacted
      ``[0, found)`` run, replacing many small partial-warp scatters with one
      full-width store.

    ``max_points`` must be ``<= FMA_STAGE_CAP`` (enforced host-side); the shared
    tile carries one extra dump slot at index ``FMA_STAGE_CAP`` so the per-chunk
    shared-tile write stays a single block-uniform statement (inactive lanes write
    the dump slot, which is never read back).
    """
    sorted_q, lane = wp.tid()
    flat_q = q_orig_sorted[sorted_q]
    b = flat_q // Q
    qx = q_x_sorted[sorted_q]
    qy = q_y_sorted[sorted_q]
    qz = q_z_sorted[sorted_q]
    gx = grid_x[b]
    gy = grid_y[b]
    gz = grid_z[b]
    mcx = min_cx[b]
    mcy = min_cy[b]
    mcz = min_cz[b]
    gbase = grid_base[b]
    lqx = wp.int32(wp.floor(qx * inv_radius)) - mcx
    lqy = wp.int32(wp.floor(qy * inv_radius)) - mcy
    lqz = wp.int32(wp.floor(qz * inv_radius)) - mcz

    # Stage 2: block-shared staging buffers (one extra dump slot at FMA_STAGE_CAP).
    # Writing a tile element migrates the tile to shared memory and emits a
    # block-wide sync, so the per-chunk store below must be reached by every lane.
    stage_idx = wp.tile_zeros(shape=(FMA_STAGE_SLOTS,), dtype=wp.int32)
    stage_dist = wp.tile_zeros(shape=(FMA_STAGE_SLOTS,), dtype=wp.float32)

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
                si = base + lane
                hit = wp.int32(0)
                d2 = wp.float32(0.0)
                if si < end:
                    dx = pc_x_sorted[si] - qx
                    dy = pc_y_sorted[si] - qy
                    dz = pc_z_sorted[si] - qz
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 <= radius2:
                        hit = wp.int32(1)
                hit_tile = wp.tile(hit)
                prefix = wp.untile(wp.tile_scan_exclusive(hit_tile))
                hits_tile = wp.tile_sum(hit_tile)
                write_hits = wp.min(hits_tile[0], max_points - found)

                # Route inactive lanes to the dump slot so the shared-tile store is a
                # single block-uniform statement; the value read is guarded so only
                # in-radius lanes (si < end) touch pc_orig_sorted.
                dst = wp.int32(FMA_STAGE_CAP)
                idx_val = wp.int32(0)
                dist_val = wp.float32(0.0)
                if hit == 1 and prefix < write_hits:
                    dst = found + prefix
                    idx_val = pc_orig_sorted[si]
                    if return_dists:
                        dist_val = wp.sqrt(d2)
                stage_idx[dst] = idx_val
                if return_dists:
                    stage_dist[dst] = dist_val

                found += write_hits
                if found >= max_points:
                    break
                base += FMA_BLOCK_DIM
        if found >= max_points:
            break

    # Stage 2 flush: one coalesced pass writes the compacted [0, found) run.
    # Consecutive lanes write consecutive slots, so each warp store is fully packed.
    s = lane
    while s < found:
        out_idx[flat_q, s] = stage_idx[s]
        if return_dists:
            out_dist[flat_q, s] = stage_dist[s]
        s += FMA_BLOCK_DIM
    if lane == 0:
        out_count[flat_q] = found


# ---------------------------------------------------------------------------
# Memory-optimized search variant: vec4 coord loads + in-kernel offset decode
# ---------------------------------------------------------------------------


@wp.kernel
def radius_search_dense_fma_kernel_mem_opt(
    q_pos: wp.array(dtype=wp.vec4),
    q_orig_sorted: wp.array(dtype=wp.int32),
    inv_radius: wp.float32,
    Q: wp.int32,
    min_cx: wp.array(dtype=wp.int32),
    min_cy: wp.array(dtype=wp.int32),
    min_cz: wp.array(dtype=wp.int32),
    grid_x: wp.array(dtype=wp.int32),
    grid_y: wp.array(dtype=wp.int32),
    grid_z: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    cell_begin_by_row: wp.array(dtype=wp.int32),
    cell_count_by_row: wp.array(dtype=wp.int32),
    pc_pos: wp.array(dtype=wp.vec4),
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
    """Memory-optimized variant of :func:`radius_search_dense_fma_kernel`.

    Loads query and point coords as single ``wp.vec4`` (128-bit) loads and decodes
    the 27 neighbor offsets from the loop index (lexicographic) instead of reading
    an offset table. The vec4 load is one instruction but moves 16 B/point (incl.
    padding) vs the SoA path's coalesced 12 B; benchmark against ``dense_fma``.
    """
    sorted_q, lane = wp.tid()
    flat_q = q_orig_sorted[sorted_q]
    b = flat_q // Q
    qp = q_pos[sorted_q]
    qx = qp[0]
    qy = qp[1]
    qz = qp[2]
    gx = grid_x[b]
    gy = grid_y[b]
    gz = grid_z[b]
    mcx = min_cx[b]
    mcy = min_cy[b]
    mcz = min_cz[b]
    gbase = grid_base[b]
    lqx = wp.int32(wp.floor(qx * inv_radius)) - mcx
    lqy = wp.int32(wp.floor(qy * inv_radius)) - mcy
    lqz = wp.int32(wp.floor(qz * inv_radius)) - mcz

    found = wp.int32(0)
    for n in range(27):
        ox = n // 9 - 1
        oy = (n // 3) % 3 - 1
        oz = n % 3 - 1
        lx = lqx + ox
        ly = lqy + oy
        lz = lqz + oz
        if lx >= 0 and lx < gx and ly >= 0 and ly < gy and lz >= 0 and lz < gz:
            row = gbase + lx + gx * (ly + gy * lz)
            start = cell_begin_by_row[row]
            end = start + cell_count_by_row[row]
            base = start
            while base < end:
                si = base + lane
                hit = wp.int32(0)
                d2 = wp.float32(0.0)
                p = wp.vec4(0.0, 0.0, 0.0, 0.0)
                if si < end:
                    p = pc_pos[si]
                    dx = p[0] - qx
                    dy = p[1] - qy
                    dz = p[2] - qz
                    d2 = dx * dx + dy * dy + dz * dz
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
                        out_dist[flat_q, slot] = wp.sqrt(d2)
                    if return_points:
                        out_pts[flat_q, slot] = wp.vec3(p[0], p[1], p[2])
                found += write_hits
                if found >= max_points:
                    break
                base += FMA_BLOCK_DIM
        if found >= max_points:
            break
    if lane == 0:
        out_count[flat_q] = found
