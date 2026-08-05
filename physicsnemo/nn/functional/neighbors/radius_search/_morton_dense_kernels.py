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
FMA_V2_BLOCK_DIM = 256
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

_NEIGHBOR_X_2BIT = wp.constant(wp.int64(0x2A802A95402551))
_NEIGHBOR_Y_2BIT = wp.constant(wp.int64(0x28282528251945))
_NEIGHBOR_Z_2BIT = wp.constant(wp.int64(0x22221862185615))

# wp.config.mode = "debug"
# wp.config.lineinfo = False
# wp.config.line_directives = False

wp.set_module_options(
    {
        "mode": "release",
        "optimization_level": 3,
        "lineinfo": True,
        "fuse_fp": True,
        "fast_math": False,
    }
)
# Route wp.tile_matmul through Warp's scalar-GEMM fallback instead of cuBLASDx so
# the module never runs the libmathdx LTO device-link step. On this CUDA 13.2
# container that link fails to build ("invalid symbol"); the scalar fallback has
# no libmathdx dependency, so the kernel compiles. Guarded with hasattr so it is a
# harmless no-op on Warp versions that predate this config flag (added in #1228).
if hasattr(wp.config, "enable_mathdx_gemm"):
    wp.config.enable_mathdx_gemm = False
# ---------------------------------------------------------------------------
# Device helper functions
# ---------------------------------------------------------------------------

# wp.set_module_options({"mode": "release"}, module=_mk)

# wp.clear_kernel_cache()
# wp.clear_lto_cache()

# wp.load_module(_mk, device="cuda")

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


# ---------------------------------------------------------------------------
# Three-phase async exclusive prefix scan (no cudaDeviceSynchronize).
#
# The standard wp.utils.array_scan / torch.cumsum both inject a device sync
# because CUB's two-pass DeviceScan API requires a sync between the sizing
# call and the execution call.  These three kernels implement the same
# exclusive scan with three ordinary kernel launches on the caller's stream:
#
#   Phase 1  (num_blocks blocks × SCAN_BLOCK threads)
#            Each block runs a tile-local exclusive scan of its SCAN_BLOCK
#            elements and writes the block total to block_sums[b].
#
#   Phase 2  (1 block × SCAN_BLOCK threads)
#            Exclusive scan of the num_blocks block totals → block_offsets[b].
#            Works as long as num_blocks <= SCAN_BLOCK (i.e. n <= SCAN_BLOCK²).
#
#   Phase 3  (n threads, regular kernel)
#            global_exclusive[i] = local_exclusive[i] + block_offsets[block(i)]
#
# Caller must pad the input to a multiple of SCAN_BLOCK with zeros so the
# last tile load is always in-bounds; see _warp_exclusive_scan() in the impl.
# ---------------------------------------------------------------------------

SCAN_BLOCK = wp.constant(1024)  # threads per scan block; max n = SCAN_BLOCK^2 = 1 048 576


@wp.kernel
def _scan_copy_pad(
    src: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.int32),
    n: int,
):
    """Copy src[0:n] into dst; dst[n:] stays at the caller-provided zeros."""
    i = wp.tid()
    if i < n:
        dst[i] = src[i]


# @wp.kernel
# def exclusive_scan_phase1(
#     arr: wp.array(dtype=wp.int32),
#     partial: wp.array(dtype=wp.int32),
#     block_sums: wp.array(dtype=wp.int32),
# ):
#     """Per-block local exclusive scan; writes block totals to block_sums."""
#     b = wp.tid()
#     t = wp.tile_load(arr, b * SCAN_BLOCK, SCAN_BLOCK)
#     wp.tile_store(partial, b * SCAN_BLOCK, wp.tile_scan_exclusive(t))
#     block_sums[b] = wp.tile_sum(t)


# @wp.kernel
# def exclusive_scan_phase2(
#     block_sums: wp.array(dtype=wp.int32),
#     block_offsets: wp.array(dtype=wp.int32),
# ):
#     """Single-block exclusive scan of block_sums → block_offsets."""
#     wp.tile_store(
#         block_offsets, 0,
#         wp.tile_scan_exclusive(wp.tile_load(block_sums, 0, SCAN_BLOCK)),
#     )


# @wp.kernel
# def exclusive_scan_phase3(
#     partial: wp.array(dtype=wp.int32),
#     block_offsets: wp.array(dtype=wp.int32),
#     begins: wp.array(dtype=wp.int32),
#     n: int,
#     block_size: int,
# ):
#     """Combine: global_exclusive[i] = local_exclusive[i] + block_offset."""
#     i = wp.tid()
#     if i < n:
#         begins[i] = partial[i] + block_offsets[i // block_size]


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

@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    const unsigned hit_mask = __ballot_sync(0xffffffffu, hit != 0);
    const unsigned lower_lane_mask =
        (1u << static_cast<unsigned>(lane)) - 1u;
    return wp::vec2i(
        __popc(hit_mask & lower_lane_mask),
        __popc(hit_mask)
    );
#else
    return wp::vec2i(0, hit != 0 ? 1 : 0);
#endif
    """
)
def _warp_hit_prefix_and_count(
    hit: wp.int32, lane: wp.int32
) -> wp.vec2i:
    """Return ``(exclusive hit prefix, total warp hits)`` for one lane."""
    ...


@wp.func
def _decode_neighbor_offset_27(neighbor: wp.int32) -> wp.vec3i:
    """Decode one standard near-to-far 27-cell offset without a memory load."""
    shift = wp.int64(2 * neighbor)
    mask = wp.int64(3)
    offset_x = wp.int32((_NEIGHBOR_X_2BIT >> shift) & mask) - 1
    offset_y = wp.int32((_NEIGHBOR_Y_2BIT >> shift) & mask) - 1
    offset_z = wp.int32((_NEIGHBOR_Z_2BIT >> shift) & mask) - 1
    return wp.vec3i(offset_x, offset_y, offset_z)


@wp.kernel(enable_backward=False, launch_bounds=FMA_V2_BLOCK_DIM)
def radius_search_dense_fma_kernel_v2(
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
    """One-warp-per-query radius search with native hit compaction.

    This kernel has the exact input/output signature and result ordering of
    :func:`radius_search_dense_fma_kernel`, but it must be launched with
    ``block_dim=FMA_V2_BLOCK_DIM``. Each lane evaluates one candidate and a
    single native ballot computes both its exclusive output rank and the total
    number of hits in the 32-candidate chunk.

    Compared with the original block-wide ``tile_scan``/``tile_sum`` path, this
    design uses no Warp tiles, shared memory, or block barriers. It also checks
    ``max_points`` every 32 candidates instead of every 64/128 candidates, so a
    full query does less excess work before exiting.
    """
    sorted_q, lane = wp.tid()

    # All lanes in the warp own the same query. Warp keeps these block-uniform
    # values in uniform registers where the target architecture supports them.
    flat_q = q_orig_sorted[sorted_q]
    batch = flat_q // Q
    query_x = q_x_sorted[sorted_q]
    query_y = q_y_sorted[sorted_q]
    query_z = q_z_sorted[sorted_q]

    grid_size_x = grid_x[batch]
    grid_size_y = grid_y[batch]
    grid_size_z = grid_z[batch]
    batch_grid_base = grid_base[batch]

    query_cell_x = (
        wp.int32(wp.floor(query_x * inv_radius)) - min_cx[batch]
    )
    query_cell_y = (
        wp.int32(wp.floor(query_y * inv_radius)) - min_cy[batch]
    )
    query_cell_z = (
        wp.int32(wp.floor(query_z * inv_radius)) - min_cz[batch]
    )

    # ``found`` stays warp-uniform: every lane receives the same total hit count
    # from the ballot helper and adds the same accepted count after each chunk.
    found = wp.int32(0)

    for neighbor in range(27):
        # ``neighbor_offsets`` remains part of the signature so callers can
        # substitute v2 without rebuilding the launch argument list. The dense
        # backend always supplies its standard near-to-far stencil, which is
        # decoded from compile-time constants here to avoid 81 global loads per
        # query.
        neighbor_offset = _decode_neighbor_offset_27(neighbor)
        cell_x = query_cell_x + neighbor_offset[0]
        cell_y = query_cell_y + neighbor_offset[1]
        cell_z = query_cell_z + neighbor_offset[2]

        # Query-cell coordinates are uniform, so this bounds branch is taken by
        # the full warp and never creates intra-warp divergence.
        if (
            cell_x >= 0
            and cell_x < grid_size_x
            and cell_y >= 0
            and cell_y < grid_size_y
            and cell_z >= 0
            and cell_z < grid_size_z
        ):
            cell_row = (
                batch_grid_base
                + cell_x
                + grid_size_x * (cell_y + grid_size_y * cell_z)
            )
            point_begin = cell_begin_by_row[cell_row]
            point_end = point_begin + cell_count_by_row[cell_row]

            chunk_begin = point_begin
            while chunk_begin < point_end:
                point_index = chunk_begin + lane
                point_valid = point_index < point_end

                hit = wp.int32(0)
                distance2 = wp.float32(0.0)
                point_x = wp.float32(0.0)
                point_y = wp.float32(0.0)
                point_z = wp.float32(0.0)

                if point_valid:
                    # These structure-of-arrays reads are unit-stride across the
                    # warp, which preserves the original coalesced candidate path.
                    point_x = pc_x_sorted[point_index]
                    point_y = pc_y_sorted[point_index]
                    point_z = pc_z_sorted[point_index]

                    dx = point_x - query_x
                    dy = point_y - query_y
                    dz = point_z - query_z
                    distance2 = dx * dx + dy * dy + dz * dz
                    if distance2 <= radius2:
                        hit = wp.int32(1)

                compact = _warp_hit_prefix_and_count(hit, lane)
                hit_prefix = compact[0]
                chunk_hits = compact[1]
                accepted_hits = wp.min(chunk_hits, max_points - found)

                # Accepted hit lanes write a dense run of query-major slots.
                # ``hit_prefix`` preserves candidate order exactly as the tile
                # exclusive scan used by the original kernel.
                if hit == 1 and hit_prefix < accepted_hits:
                    output_slot = found + hit_prefix
                    out_idx[flat_q, output_slot] = pc_orig_sorted[point_index]
                    if return_dists:
                        out_dist[flat_q, output_slot] = wp.sqrt(distance2)
                    if return_points:
                        out_pts[flat_q, output_slot] = wp.vec3(
                            point_x, point_y, point_z
                        )

                found += accepted_hits
                if found >= max_points:
                    break

                chunk_begin += FMA_V2_BLOCK_DIM

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


# @wp.kernel
# def radius_search_mem_optimized(
#     grid_cell_idx,
#     inv_radius,
#     queries_per_batch,
#     min_cx,
#     min_cy, 
#     min_cz,
#     num_grid_x, 
#     num_grid_y, 
#     num_grid_z,
#     grid_base,
#     cell_begin_by_grid_odx,
#     cell_count_by_grid_idx,
#     radius,
#     max_points,
#     return_dists,
#     return_points,
#     out_idx, 
#     out_count, 
#     out_dist,
#     out_points,
#     TILE_DIM

# ):
#     curr_cell, lane = wp.tid()

#     num_cells = num_grid_x * num_grid_y * num_grid_z

#     num_points = cell_count_by_grid_idx[curr_cell]

#     gbase = 0
#     queries = .. shared global load of all the points in the cell

#     points = .. shared global load of all the points in the cell


#     if num_points < max_points + 1:
#         shared global load all points in the 27 neighbouring cells 
#         add to the points
#         for i in range(27):
#             ox = n//9 -1
#             oy = (n//3) %3 -1
#             oz = n%3 -1

#             lx = lqx + ox
#             ly = lqy + oy
#             lz = lqz + oz
#             if lx >= 0 and lx < gx and ly >= 0 and ly < gy and lz >= 0 and lz < gz:
#                 row = gbase + lx + gx * (ly + gy * lz)
#                 ....
    
#     num_tiles_q = (len(queries) // TILE_DIM) +1
#     num_tiles_pc = (len(points)//TILE_DIM) +1

#     for tile_q in range(num_tiles_q):
#         tile_idx = lane % TILE_DIM
#         shared_out_idx = ..allocate in shared_memory (TILE_DIM x max_points)
#         shared_out_count = ..allocate in shared memory (TILE_DIM x max_points)
#         shared_out_dist = ..allocate in shared memory (TILE_DIM x max_points)
#         dists_mat = ..allocate in shared memory ( TILE_DIM x TILE_DIM)
#         for tile_q in range(num_tiles_p):

#             q_points_tiled = wp.tiled_load
#             p_points_tiles = wp.transpose(wp.tiled_load())

#             dists_mat = -2.0 * wp.tiled_matmul(q_points_tiled, p_points_tiles)

#             dists_mat += ...

#             ..write to shared_out matrices
#             ..each thread writes to shared_out_count
        
#         ..from shared do a coalesed global write

TILE_DIM   = wp.constant(32)   # tile edge: queries per tile == candidates per tile
# Compile-time capacity of the per-query shared staging tiles (sh_idx/sh_dst are
# (TILE_DIM, MAX_POINTS)). This is an upper bound on the neighbor cap, NOT the cap
# itself: the actual per-call cap is the runtime ``max_points`` kernel argument,
# which the kernel clamps to MAX_POINTS so it can never stage/emit past the tile.
# Must be >= the largest ``max_points`` any caller requests (the GeoTransolver
# recipe uses neighbors_in_radius up to 128), otherwise results are truncated.
MAX_POINTS = wp.constant(128)  # shared-staging capacity / upper bound on max_points


# ===========================================================================
# PRIMARY: tiled / tile_matmul design, TRANSPOSED (coalesced) output
# ===========================================================================
@wp.kernel
def radius_search_mem_optimized(
    pts:           wp.array2d(dtype=wp.float32),   # (Ntot, 3+) sorted coords
    pcs_t:         wp.array2d(dtype=wp.float32),   # (3, Ntot) transposed coords for Pᵀ tile_matmul loads
    B:             wp.int32,                       # batch size
    num_grid_x:    wp.array(dtype=wp.int32),       # (B,) per-batch cell extents
    num_grid_y:    wp.array(dtype=wp.int32),       # (B,)
    num_grid_z:    wp.array(dtype=wp.int32),       # (B,)
    grid_base:     wp.array(dtype=wp.int32),       # (B+1,) per-batch dense-cell base offsets
    cell_begin:    wp.array(dtype=wp.int32),       # (total_cells,)
    cell_count:    wp.array(dtype=wp.int32),       # (total_cells,)
    occupied_rows: wp.array(dtype=wp.int32),       # (num_occupied,) global cell ids with count > 0 (the launch grid)
    radius:        wp.float32,
    max_points:    wp.int32,                       # runtime neighbor cap (== out_* row count)
    return_dists:  wp.bool,
    return_points: wp.bool,
    out_idx:       wp.array2d(dtype=wp.int32),     # (max_points, Ntot)   TRANSPOSED
    out_count:     wp.array(dtype=wp.int32),       # (Ntot,)
    out_dist:      wp.array2d(dtype=wp.float32),   # (max_points, Ntot)   TRANSPOSED (dummy (1,1) if off)
    out_points:    wp.array3d(dtype=wp.float32),   # (max_points, Ntot, 3)TRANSPOSED (dummy (1,1,3) if off)
):
    # One block per OCCUPIED cell: the launch grid spans only non-empty cells
    # (num_occupied <= Ntot), not the full dense grid (total_cells, dominated by empty
    # cells for small radii). Map the compact block id back to its global dense-cell row.
    blk, lane = wp.tid()                           # block ↔ occupied-cell slot ; lane in [0, BLOCK_DIM)
    curr_cell = occupied_rows[blk]                 # global row-major dense cell id (count > 0 by construction)
    # wp.print(f"inside_kernel : {lane}")
    # Recover the batch of this global row-major cell id from grid_base (the per-batch
    # exclusive scan of cell counts), mirroring enumerate_cell_morton_keys and the FMA
    # search. Cell extents and the row base are then read per-batch, so variable-size
    # per-batch grids index correctly (no uniform-grid b*num_cells assumption).
    b = wp.int32(0)
    # while b + 1 < B and grid_base[b + 1] <= curr_cell:
    #     b += 1
    gbase = grid_base[b]                           # per-batch cell base offset
    cell  = curr_cell - gbase                      # local row-major cell id within batch b
    gx = num_grid_x[b]
    gy = num_grid_y[b]
    gz = num_grid_z[b]

    n_self = cell_count[curr_cell]
    if n_self == 0:
        return
    q_start = cell_begin[curr_cell]
    r2 = radius * radius

    cell_x = cell % gx                             # lin = cell_x + gx*(cell_y + gy*cell_z)
    cell_y = (cell // gx) % gy
    cell_z = cell // (gx * gy)

    # Effective per-query cap: the caller's runtime max_points, clamped to the
    # shared-staging capacity so sh_idx/sh_dst[..cnt] and the out_* writes can never
    # run past their bounds (out_* have exactly max_points rows). max_points > MAX_POINTS
    # would silently truncate to MAX_POINTS (raise MAX_POINTS to avoid that).
    mp_eff = wp.min(max_points, MAX_POINTS)
    use_neighbors = n_self < (max_points + 1)      # self cell alone can't fill the quota → also scan the 27-cell stencil (approximate; note #6)


    
        
    num_q_tiles = (n_self + TILE_DIM - 1) // TILE_DIM

    for tq in range(num_q_tiles):
        q0 = q_start + tq * TILE_DIM
        Q  = wp.tile_load(pts, shape=(TILE_DIM, 3), offset=(q0, 0), storage="shared", bounds_check=True)
        qn = wp.tile_reshape(wp.tile_sum(Q * Q, axis=1), shape=(TILE_DIM, 1))  # (TILE_DIM,1) ‖x‖²
        #qn = wp.tile_zeros(shape=(TILE_DIM, 1),   storage="shared")

        # sh_idx = wp.tile_zeros(shape=(TILE_DIM, MAX_POINTS), dtype=wp.int32,   storage="register")
        # sh_dst = wp.tile_zeros(shape=(TILE_DIM, MAX_POINTS), dtype=wp.float32, storage="register")
        cnt = int(0)                               # per-lane running neighbour count (register)
        # if lane ==0:
        #     wp.breakpoint()
        for n in range(27):
            if (not use_neighbors):  # dense: self cell (offset 0,0,0 == n 13) only
                continue
            offset_x = n // 9 - 1
            offset_y = (n // 3) % 3 - 1
            offset_z = n % 3 - 1
            nx = cell_x + offset_x
            ny = cell_y + offset_y
            nz = cell_z + offset_z
            
            if nx < 0 or nx >= gx or ny < 0 or ny >= gy or nz < 0 or nz >= gz:
                continue
            
            ncell   = gbase + nx + gx * (ny + gy * nz)
            
            p_start = cell_begin[ncell]
            p_cnt   = cell_count[ncell]
            num_p_tiles = (p_cnt + TILE_DIM - 1) // TILE_DIM
            # if lane == 0:
            #     wp.breakpoint()
            for tp in range(num_p_tiles):
                p0 = p_start + tp * TILE_DIM
                # Load Pᵀ directly as (3, TILE_DIM) so tile_matmul gets a real (K, N)
                # operand: Warp's tile_matmul rejects a tile_transpose() view as an operand
                # (issue #1527), so the transpose is materialized here via the transposed
                # coord buffer instead of an in-kernel transpose.
                # if lane == 0:
                #     wp.breakpoint()
                Pt = wp.tile_load(pcs_t, shape=(3, TILE_DIM), offset=(0, p0), storage="shared", bounds_check=True)
                ## DEBUG
                #Pt = wp.tile_zeros(shape=(3, TILE_DIM),  storage="shared")
                pn = wp.tile_reshape(wp.tile_sum(Pt * Pt, axis=0), shape=(TILE_DIM, 1))  # (TILE_DIM,1) ‖y‖²
                #pn = wp.tile_zeros(shape=(TILE_DIM, 1),   storage="shared")
                # DEBUG
                #pn =  wp.tile_zeros(shape=(TILE_DIM,1),  storage="shared")
                
                dists = wp.tile_zeros(shape=(TILE_DIM, TILE_DIM), dtype=wp.float32, storage="shared")
                wp.tile_matmul(Q, Pt, dists, -2.0, 0.0)                           # dists = -2·Q·Pᵀ
                dists += wp.tile_broadcast(qn, shape=(TILE_DIM, TILE_DIM))
                dists += wp.tile_broadcast(wp.tile_transpose(pn), shape=(TILE_DIM, TILE_DIM))  # ‖x‖²+‖y‖²−2x·yᵀ
                # dists += qn 
                # dists += wp.tile_transpose(pn)
                # if lane == 0:
                #     wp.breakpoint()
                # if lane < TILE_DIM:                # per-lane selection (per-thread branch, not cooperative)
                #     qg = q0 + lane
                #     if qg < q_start + n_self:
                #         for c in range(TILE_DIM):
                #             pg = p0 + c
                #             if pg < p_start + p_cnt and pg != qg:
                #                 d2 = dists[lane, c]                # ⚠️ shared-tile element read (note #4)
                #                 if d2 <= r2 and cnt < mp_eff:
                #                     sh_idx[lane, cnt] = pg
                #                     sh_dst[lane, cnt] = d2
                #                     cnt += 1
                if lane < TILE_DIM:
                    qg = q0 + lane
                    if qg < q_start + n_self:
                        for c in range(TILE_DIM):
                            pg = p0 + c
                            if pg < p_start + p_cnt and pg != qg:
                                d2 = dists[lane, c]
                                if d2 <= r2 and cnt < max_points:
                                    out_idx[cnt, qg] = pg
                                    if return_dists != 0:
                                        out_dist[cnt, qg] = d2
                                    if return_points != 0:
                                        # out_points[cnt, qg, 0] = pts[pg, 0]
                                        # out_points[cnt, qg, 1] = pts[pg, 1]
                                        # out_points[cnt, qg, 2] = pts[pg, 2]
                                        out_points[cnt, qg, 0] = Q[lane, 0]
                                        out_points[cnt, qg, 1] = Q[lane, 1]
                                        out_points[cnt, qg, 2] = Q[lane, 2]
                                    cnt += 1



        # ---- COALESCED write: index [m, qg]; Ntot is innermost → consecutive lanes are unit-stride ----
        # if lane < TILE_DIM:
        #     qg = q0 + lane
        #     if qg < q_start + n_self:
        #         out_count[qg] = cnt
        #         for m in range(max_points):        # outer loop over slot (== out_* rows) → warp coalesces across qg
        #             if m < cnt:
        #                 j = sh_idx[lane, m]
        #                 out_idx[m, qg] = j
        #                 if return_dists != 0:
        #                     out_dist[m, qg] = sh_dst[lane, m]
        #                 if return_points != 0:
        #                     out_points[m, qg, 0] = pts[j, 0]
        #                     out_points[m, qg, 1] = pts[j, 1]
        #                     out_points[m, qg, 2] = pts[j, 2]
        #             else:
        #                 out_idx[m, qg] = -1             
        if lane < TILE_DIM:
            qg = q0 + lane
            if qg < q_start + n_self:
                out_count[qg] = cnt


MEM_OPT_2_BLOCK_DIM = 32

@wp.kernel
def radius_search_mem_optimized_2(
    pts: wp.array2d(dtype=wp.float32),
    B: wp.int32,
    num_grid_x: wp.array(dtype=wp.int32),
    num_grid_y: wp.array(dtype=wp.int32),
    num_grid_z: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    cell_begin: wp.array(dtype=wp.int32),
    cell_count: wp.array(dtype=wp.int32),
    neighbor_offsets: wp.array2d(dtype=wp.int32),
    task_cell: wp.array(dtype=wp.int32),
    task_q_begin: wp.array(dtype=wp.int32),
    task_q_count: wp.array(dtype=wp.int32),
    radius: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
):
    """Cell-centric self radius search using direct squared distances.

    Launch this kernel with ``block_dim=MEM_OPT_2_BLOCK_DIM`` and one task for
    each ``(occupied cell, query tile)`` pair. ``task_q_count`` must be no larger
    than ``MEM_OPT_2_BLOCK_DIM``. Each lane owns one query while the full block
    cooperatively loads a candidate tile into shared memory.

    ``pts`` contains compacted point coordinates in ``(x, y, z, 0)`` layout. It
    must have at least ``MEM_OPT_2_BLOCK_DIM - 1`` padded rows because full
    candidate tiles are loaded with bounds checking disabled. The dense index
    currently pads this array by ``TILE_P`` rows, so it satisfies that contract.

    Neighbor offsets should be ordered near-to-far, with the query cell first,
    to make the block-wide early exit useful. Output indices are compacted-point
    indices and distances are squared, matching ``radius_search_mem_optimized``.
    Neighbor coordinates are deliberately not written here: callers can gather
    them from ``pts`` using ``out_idx`` after the search has completed.
    """
    task, lane = wp.tid()

    curr_cell = task_cell[task]
    q_begin = task_q_begin[task]
    q_count = task_q_count[task]

    # All queries in a task belong to the same dense cell, so this batch lookup
    # and the subsequent cell-coordinate calculations are block-uniform.
    batch = wp.int32(0)
    while batch + 1 < B and grid_base[batch + 1] <= curr_cell:
        batch += 1

    cell_base = grid_base[batch]
    local_cell = curr_cell - cell_base
    grid_x = num_grid_x[batch]
    grid_y = num_grid_y[batch]
    grid_z = num_grid_z[batch]

    cell_x = local_cell % grid_x
    cell_y = (local_cell // grid_x) % grid_y
    cell_z = local_cell // (grid_x * grid_y)

    # One lane owns one query for the lifetime of this task. Invalid lanes in a
    # partial final tile participate in every cooperative operation as "done".
    query_valid = lane < q_count
    query_index = q_begin + lane
    query_x = wp.float32(0.0)
    query_y = wp.float32(0.0)
    query_z = wp.float32(0.0)
    if query_valid:
        query_x = pts[query_index, 0]
        query_y = pts[query_index, 1]
        query_z = pts[query_index, 2]

    radius2 = radius * radius
    found = wp.int32(0)
    all_queries_done = wp.bool(False)

    for neighbor in range(27):
        if all_queries_done:
            break

        # The reduction result is identical for every lane. Consequently the
        # break is block-uniform and no lane skips a cooperative tile operation.
        lane_done = wp.int32(0)
        if not query_valid or found >= max_points:
            lane_done = wp.int32(1)
        done_tile = wp.tile(lane_done)
        done_count = wp.tile_sum(done_tile)
        if done_count[0] == MEM_OPT_2_BLOCK_DIM:
            break

        neighbor_x = cell_x + neighbor_offsets[neighbor, 0]
        neighbor_y = cell_y + neighbor_offsets[neighbor, 1]
        neighbor_z = cell_z + neighbor_offsets[neighbor, 2]

        # Cell coordinates are shared by the block, so this branch is uniform.
        if (
            neighbor_x < 0
            or neighbor_x >= grid_x
            or neighbor_y < 0
            or neighbor_y >= grid_y
            or neighbor_z < 0
            or neighbor_z >= grid_z
        ):
            continue

        neighbor_cell = (
            cell_base
            + neighbor_x
            + grid_x * (neighbor_y + grid_y * neighbor_z)
        )
        point_begin = cell_begin[neighbor_cell]
        point_count = cell_count[neighbor_cell]
        num_point_tiles = (
            point_count + MEM_OPT_2_BLOCK_DIM - 1
        ) // MEM_OPT_2_BLOCK_DIM

        for point_tile in range(num_point_tiles):
            point_tile_begin = (
                point_begin + point_tile * MEM_OPT_2_BLOCK_DIM
            )
            valid_points = wp.min(
                MEM_OPT_2_BLOCK_DIM,
                point_begin + point_count - point_tile_begin,
            )

            # The last coordinate is padding. Loading 32x4 floats lets Warp use
            # vectorized float4 transactions and gives every lane fast shared
            # access to every candidate in this tile.
            candidate_tile = wp.tile_load(
                pts,
                shape=(MEM_OPT_2_BLOCK_DIM, 4),
                offset=(point_tile_begin, 0),
                storage="shared",
                bounds_check=False,
            )

            # Every lane performs the shared-tile reads in the same order. Only
            # the scalar distance and output work below is lane-divergent.
            for candidate in range(MEM_OPT_2_BLOCK_DIM):
                point_x = candidate_tile[candidate, 0]
                point_y = candidate_tile[candidate, 1]
                point_z = candidate_tile[candidate, 2]

                if (
                    query_valid
                    and found < max_points
                    and candidate < valid_points
                ):
                    point_index = point_tile_begin + candidate

                    # This kernel is for self-search, so exclude the query's own
                    # compacted point slot.
                    if point_index != query_index:
                        dx = point_x - query_x
                        dy = point_y - query_y
                        dz = point_z - query_z
                        distance2 = dx * dx + dy * dy + dz * dz

                        if distance2 <= radius2:
                            out_idx[found, query_index] = point_index
                            if return_dists:
                                out_dist[found, query_index] = distance2
                            found += 1

            # A second collective vote prevents this block from loading another
            # P tile (or visiting another neighbor cell) after every query in the
            # current Q tile has filled its output quota.
            lane_done = wp.int32(0)
            if not query_valid or found >= max_points:
                lane_done = wp.int32(1)
            done_tile = wp.tile(lane_done)
            done_count = wp.tile_sum(done_tile)
            if done_count[0] == MEM_OPT_2_BLOCK_DIM:
                all_queries_done = wp.bool(True)
                break

    if query_valid:
        out_count[query_index] = found


# ``radius_search_mem_optimized_3`` is intentionally a one-warp kernel. Warp
# does not currently expose CUDA's shuffle and vote intrinsics as ordinary
# kernel functions, so these two small native helpers provide the missing
# operations. The CPU branches keep the module source valid when Warp emits a
# CPU version; this search kernel is launched only on CUDA.
@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source_lane);
#else
    return value;
#endif
    """
)
def _warp_broadcast_f32(
    value: wp.float32, source_lane: wp.int32
) -> wp.float32:
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __all_sync(0xffffffffu, predicate != 0);
#else
    return predicate != 0;
#endif
    """
)
def _warp_all_i32(predicate: wp.int32) -> wp.int32:
    ...


@wp.kernel(enable_backward=False)
def radius_search_mem_optimized_3(
    pts: wp.array2d(dtype=wp.float32),
    B: wp.int32,
    num_grid_x: wp.array(dtype=wp.int32),
    num_grid_y: wp.array(dtype=wp.int32),
    num_grid_z: wp.array(dtype=wp.int32),
    grid_base: wp.array(dtype=wp.int32),
    cell_begin: wp.array(dtype=wp.int32),
    cell_count: wp.array(dtype=wp.int32),
    neighbor_offsets: wp.array2d(dtype=wp.int32),
    task_cell: wp.array(dtype=wp.int32),
    task_q_begin: wp.array(dtype=wp.int32),
    task_q_count: wp.array(dtype=wp.int32),
    radius: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
):
    """Register/shuffle version of the cell-centric self radius search.

    The signature and task layout match :func:`radius_search_mem_optimized_2`.
    Launch with ``block_dim=MEM_OPT_2_BLOCK_DIM`` so one CUDA warp owns one
    query tile and one lane owns each query.

    For every candidate tile, lane ``i`` loads candidate ``i`` into scalar
    registers. Warp shuffles then broadcast that candidate to the other lanes,
    retaining v2's one-global-load-per-candidate reuse without a Warp tile,
    shared-memory staging, or block synchronization. A warp-wide ``all`` vote
    stops the task as soon as every valid query has reached ``max_points``.

    Output indices are compacted-point indices and output distances are squared,
    exactly as in v2. The caller may gather neighbor coordinates from ``pts``
    after the search.
    """
    task, lane = wp.tid()

    curr_cell = task_cell[task]
    query_begin = task_q_begin[task]
    query_count = task_q_count[task]

    # All queries in this task come from one cell. The batch lookup and cell
    # coordinate calculations are therefore uniform across the warp.
    batch = wp.int32(0)
    while batch + 1 < B and grid_base[batch + 1] <= curr_cell:
        batch += 1

    cell_base = grid_base[batch]
    local_cell = curr_cell - cell_base
    grid_x = num_grid_x[batch]
    grid_y = num_grid_y[batch]
    grid_z = num_grid_z[batch]

    cell_x = local_cell % grid_x
    cell_y = (local_cell // grid_x) % grid_y
    cell_z = local_cell // (grid_x * grid_y)

    # Partial query tiles still execute as a full warp. Invalid lanes carry a
    # zero query and report themselves done in every warp-wide vote.
    query_valid = lane < query_count
    query_index = query_begin + lane
    query_x = wp.float32(0.0)
    query_y = wp.float32(0.0)
    query_z = wp.float32(0.0)
    if query_valid:
        query_x = pts[query_index, 0]
        query_y = pts[query_index, 1]
        query_z = pts[query_index, 2]

    radius2 = radius * radius
    found = wp.int32(0)

    # This initial vote handles max_points == 0 without entering the search.
    lane_done = wp.int32(0)
    if not query_valid or found >= max_points:
        lane_done = wp.int32(1)
    all_queries_done = _warp_all_i32(lane_done) != 0

    for neighbor in range(27):
        if all_queries_done:
            break

        neighbor_x = cell_x + neighbor_offsets[neighbor, 0]
        neighbor_y = cell_y + neighbor_offsets[neighbor, 1]
        neighbor_z = cell_z + neighbor_offsets[neighbor, 2]

        # Cell bounds are block-uniform, so every lane takes the same branch and
        # every active lane reaches the shuffle and vote intrinsics below.
        if (
            neighbor_x < 0
            or neighbor_x >= grid_x
            or neighbor_y < 0
            or neighbor_y >= grid_y
            or neighbor_z < 0
            or neighbor_z >= grid_z
        ):
            continue

        neighbor_cell = (
            cell_base
            + neighbor_x
            + grid_x * (neighbor_y + grid_y * neighbor_z)
        )
        point_begin = cell_begin[neighbor_cell]
        point_count = cell_count[neighbor_cell]
        num_point_tiles = (
            point_count + MEM_OPT_2_BLOCK_DIM - 1
        ) // MEM_OPT_2_BLOCK_DIM

        for point_tile in range(num_point_tiles):
            point_tile_begin = (
                point_begin + point_tile * MEM_OPT_2_BLOCK_DIM
            )
            valid_points = wp.min(
                MEM_OPT_2_BLOCK_DIM,
                point_begin + point_count - point_tile_begin,
            )

            # Each candidate is fetched once by its source lane. Initialize the
            # registers for lanes outside a partial P tile so all 32 lanes can
            # safely participate in every shuffle.
            lane_point_x = wp.float32(0.0)
            lane_point_y = wp.float32(0.0)
            lane_point_z = wp.float32(0.0)
            if lane < valid_points:
                lane_point_index = point_tile_begin + lane
                lane_point_x = pts[lane_point_index, 0]
                lane_point_y = pts[lane_point_index, 1]
                lane_point_z = pts[lane_point_index, 2]

            # A dynamic loop keeps the candidate state small. Shuffles occur
            # outside the query-dependent branch, which is required for a full
            # warp mask under CUDA independent thread scheduling.
            candidate = wp.int32(0)
            while candidate < valid_points:
                point_x = _warp_broadcast_f32(lane_point_x, candidate)
                point_y = _warp_broadcast_f32(lane_point_y, candidate)
                point_z = _warp_broadcast_f32(lane_point_z, candidate)

                if query_valid and found < max_points:
                    point_index = point_tile_begin + candidate

                    # The compacted points are both queries and candidates, so
                    # exclude the query's own compacted slot.
                    if point_index != query_index:
                        dx = point_x - query_x
                        dy = point_y - query_y
                        dz = point_z - query_z
                        distance2 = dx * dx + dy * dy + dz * dz

                        if distance2 <= radius2:
                            out_idx[found, query_index] = point_index
                            if return_dists:
                                out_dist[found, query_index] = distance2
                            found += 1

                candidate += 1

            # Check once per P tile: this avoids the cost of a vote for every
            # candidate while still preventing all later P tiles and neighbor
            # cells from being visited after the Q tile is full.
            lane_done = wp.int32(0)
            if not query_valid or found >= max_points:
                lane_done = wp.int32(1)
            if _warp_all_i32(lane_done) != 0:
                all_queries_done = wp.bool(True)
                break

    if query_valid:
        out_count[query_index] = found