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
Warp kernels for the ``sparse_fma_e2e`` radius-search backend.

Unlike the dense-grid backend (``_morton_dense_*``), this backend never
materialises a dense per-batch cell grid. It hashes *only the occupied* cells
into an open-addressed table sized ``H >= 2 * Np`` (load factor <= 0.5) and
compacts the points into cell-contiguous ranges via a count -> exclusive-scan
-> scatter build. Search reuses the one-warp-per-query, all-lanes-per-cell
coalesced loop of the dense v2 kernel; only the dense-row cell lookup is
replaced by a lane-zero hash probe broadcast to the warp.

The three-phase async prefix scan (no ``cudaDeviceSynchronize``) lives in
``_morton_dense_kernels.py`` and is reused verbatim by the host orchestration.
"""

import warp as wp

from ._morton_dense_kernels import (
    _NEIGHBOR_X_2BIT,
    _NEIGHBOR_Y_2BIT,
    _NEIGHBOR_Z_2BIT,
)

# One full warp per query. Every query launches exactly 32 lanes so the ballot
# / shuffle helpers below can use the full 0xffffffff mask (see design note).
FMA_WARP_SIZE = wp.constant(32)

# Elements processed serially per thread in the block-reduce / block-scan
# kernels. This is a plain grid-stride chunk size, NOT a tile/warp width: the
# scan below uses only scalar array ops and wp.tid(), never wp.tile*, because
# wp.tile_scan_exclusive / wp.tile_sum were observed to scan only a fraction of
# a block on this Warp build. Larger SCAN_TILE => fewer blocks but more serial
# work per thread; 256 keeps num_blocks small while staying cheap.
SCAN_TILE = wp.constant(256)


# ---------------------------------------------------------------------------
# Async exclusive prefix scan (no cudaDeviceSynchronize, no CUB, no tiles).
#
# Block-decomposed, tile-free. Each "block" owns SCAN_TILE contiguous elements
# and is handled by ONE thread that loops serially. The host recurses on the
# per-block totals (see _warp_scan_exclusive), so any length is supported.
#
#   phase1  dim=num_blocks, 1 thread/block: block_sums[b] = sum(arr[block b])
#   (recurse) exclusive-scan block_sums -> block_offsets
#   phase3  dim=num_blocks, 1 thread/block: write exclusive prefixes within the
#           block, seeded by block_offsets[b]
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sparse_block_reduce(
    arr: wp.array(dtype=wp.int32),
    n_pad: wp.int32,
    block_sums: wp.array(dtype=wp.int32),
):
    """One thread per block: serial sum of its SCAN_TILE-element chunk."""
    b = wp.tid()
    start = b * SCAN_TILE
    total = wp.int32(0)
    j = wp.int32(0)
    while j < SCAN_TILE:
        idx = start + j
        if idx < n_pad:
            total += arr[idx]
        j += 1
    block_sums[b] = total


@wp.kernel(enable_backward=False)
def sparse_block_scan_write(
    arr: wp.array(dtype=wp.int32),
    block_offsets: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
    n: wp.int32,
):
    """One thread per block: exclusive prefix within the block + block base."""
    b = wp.tid()
    start = b * SCAN_TILE
    running = block_offsets[b]
    j = wp.int32(0)
    while j < SCAN_TILE:
        idx = start + j
        if idx < n:
            out[idx] = running
            running += arr[idx]
        j += 1


# ---------------------------------------------------------------------------
# Quantization and hashing
# ---------------------------------------------------------------------------


@wp.func
def _quantize_cell(p: wp.vec3, inv_cell: wp.float32) -> wp.vec3i:
    """Absolute signed integer cell of a point (no per-batch min offset)."""
    return wp.vec3i(
        wp.int32(wp.floor(p[0] * inv_cell)),
        wp.int32(wp.floor(p[1] * inv_cell)),
        wp.int32(wp.floor(p[2] * inv_cell)),
    )


@wp.func
def _mix64(x: wp.uint64) -> wp.uint64:
    """splitmix64-style finalizer; good avalanche for spatial cell keys."""
    x = x ^ (x >> wp.uint64(33))
    x = x * wp.uint64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> wp.uint64(33))
    x = x * wp.uint64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> wp.uint64(33))
    return x


@wp.func
def _hash_sparse_cell(batch: wp.int32, cell: wp.vec3i) -> wp.uint64:
    """Hash a ``(batch, cx, cy, cz)`` cell key to 64 bits.

    The hash is *not* the cell identity: every insert/lookup compares the exact
    tuple, so hash collisions stay correct (they only cost extra probes).
    """
    xy = wp.uint64(wp.uint32(cell[0])) | (
        wp.uint64(wp.uint32(cell[1])) << wp.uint64(32)
    )
    zb = wp.uint64(wp.uint32(cell[2])) | (
        wp.uint64(wp.uint32(batch)) << wp.uint64(32)
    )
    return _mix64(xy ^ _mix64(zb))


@wp.func
def _decode_neighbor_offset_27(neighbor: wp.int32) -> wp.vec3i:
    """Decode one standard near-to-far 27-cell offset without a memory load."""
    shift = wp.int64(2 * neighbor)
    mask = wp.int64(3)
    offset_x = wp.int32((_NEIGHBOR_X_2BIT >> shift) & mask) - 1
    offset_y = wp.int32((_NEIGHBOR_Y_2BIT >> shift) & mask) - 1
    offset_z = wp.int32((_NEIGHBOR_Z_2BIT >> shift) & mask) - 1
    return wp.vec3i(offset_x, offset_y, offset_z)


# ---------------------------------------------------------------------------
# Native warp helpers (ballot compaction + int32 broadcast)
# ---------------------------------------------------------------------------


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
def _warp_hit_prefix_and_count(hit: wp.int32, lane: wp.int32) -> wp.vec2i:
    """Return ``(exclusive hit prefix, total warp hits)`` for one lane."""
    ...


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source_lane);
#else
    return value;
#endif
    """
)
def _warp_broadcast_i32(value: wp.int32, source_lane: wp.int32) -> wp.int32:
    """Broadcast ``value`` from ``source_lane`` to all lanes in the warp."""
    ...


# ---------------------------------------------------------------------------
# Search-time hash lookup
# ---------------------------------------------------------------------------


@wp.func
def _lookup_sparse_cell(
    batch: wp.int32,
    cell: wp.vec3i,
    hash_capacity: wp.int32,
    cell_rep_plus1: wp.array(dtype=wp.int32),
    cell_coords: wp.array(dtype=wp.vec4i),
) -> wp.int32:
    """Open-addressed probe for an occupied cell; ``-1`` if absent.

    ``cell_rep_plus1[slot] == 0`` marks an empty slot and terminates the probe
    (linear probing keeps a cell's whole run contiguous, so the first empty
    slot proves the cell was never inserted).
    """
    mask = hash_capacity - 1
    slot = wp.int32(_hash_sparse_cell(batch, cell) & wp.uint64(mask))

    probe = wp.int32(0)
    while probe < hash_capacity:
        if cell_rep_plus1[slot] == 0:
            return wp.int32(-1)

        stored = cell_coords[slot]
        if (
            stored[0] == batch
            and stored[1] == cell[0]
            and stored[2] == cell[1]
            and stored[3] == cell[2]
        ):
            return slot

        slot = (slot + 1) & mask
        probe += 1

    return wp.int32(-1)


# ---------------------------------------------------------------------------
# Build: count + hash construction
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sparse_count_cells_kernel(
    points: wp.array2d(dtype=wp.vec3),
    inv_cell: wp.float32,
    P: wp.int32,
    hash_capacity: wp.int32,
    cell_rep_plus1: wp.array(dtype=wp.int32),
    cell_count: wp.array(dtype=wp.int32),
    cell_coords: wp.array(dtype=wp.vec4i),
    point_slot: wp.array(dtype=wp.int32),
    point_rank: wp.array(dtype=wp.int32),
):
    """Insert each point's cell into the hash table and count per-slot points.

    Lock-free insert: a thread claims an empty slot with ``atomic_cas`` writing
    ``flat + 1`` (a nonzero sentinel that shares the zero-cleared allocation).
    On a CAS miss it re-derives the incumbent's cell from ``points[]`` -- never
    from ``cell_coords`` (which the winner may not have published yet) -- and
    compares the exact tuple, so the insert is race-safe without a fence.
    """
    flat = wp.tid()
    batch = flat // P
    local_point = flat - batch * P

    point = points[batch, local_point]
    cell = _quantize_cell(point, inv_cell)

    mask = hash_capacity - 1
    slot = wp.int32(_hash_sparse_cell(batch, cell) & wp.uint64(mask))

    resolved_slot = wp.int32(-1)
    probe = wp.int32(0)
    while probe < hash_capacity:
        old_plus1 = wp.atomic_cas(cell_rep_plus1, slot, wp.int32(0), flat + 1)

        if old_plus1 == 0:
            # This thread created the cell; publish its coordinates.
            cell_coords[slot] = wp.vec4i(batch, cell[0], cell[1], cell[2])
            resolved_slot = slot
            break

        # Re-derive the incumbent's cell from points[] (its cell_coords write may
        # not be visible yet). This makes the insert correct without a fence.
        representative = old_plus1 - 1
        rep_batch = representative // P
        rep_local = representative - rep_batch * P
        rep_point = points[rep_batch, rep_local]
        rep_cell = _quantize_cell(rep_point, inv_cell)

        if (
            rep_batch == batch
            and rep_cell[0] == cell[0]
            and rep_cell[1] == cell[1]
            and rep_cell[2] == cell[2]
        ):
            resolved_slot = slot
            break

        slot = (slot + 1) & mask
        probe += 1

    # H >= 2*Np guarantees a free slot is always found before probe exhausts,
    # so resolved_slot is always in [0, H). The guard keeps a hypothetical
    # table-full case a well-defined no-op instead of a cell_count[-1] OOB.
    point_slot[flat] = resolved_slot
    if resolved_slot >= 0:
        # Capture the intra-cell rank handed out by this atomic so scatter can
        # compute its destination as cell_offsets[slot] + rank directly, with no
        # second write-cursor atomic and no cell_offsets clone.
        rank = wp.atomic_add(cell_count, resolved_slot, wp.int32(1))
        point_rank[flat] = rank


# ---------------------------------------------------------------------------
# Build: scatter points into cell-contiguous SoA
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sparse_scatter_points_kernel(
    points: wp.array2d(dtype=wp.vec3),
    P: wp.int32,
    point_slot: wp.array(dtype=wp.int32),
    point_rank: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    pc_x: wp.array(dtype=wp.float32),
    pc_y: wp.array(dtype=wp.float32),
    pc_z: wp.array(dtype=wp.float32),
    pc_orig: wp.array(dtype=wp.int32),
):
    """Compact points into cell-contiguous order using precomputed intra-cell ranks.

    ``point_rank[flat]`` is the rank the count kernel handed out for this point
    within its cell, and ``cell_offsets[slot]`` is that cell's exclusive CSR
    start. Their sum is the point's unique destination -- no scatter-phase
    atomic and no ``cell_offsets`` clone. ``cell_offsets`` stays immutable so it
    serves directly as the CSR start array (``cell_offsets[slot] ..
    cell_offsets[slot + 1]``) at search time.
    """
    flat = wp.tid()
    batch = flat // P
    local_point = flat - batch * P

    slot = point_slot[flat]
    if slot < 0:
        return

    destination = cell_offsets[slot] + point_rank[flat]

    point = points[batch, local_point]
    pc_x[destination] = point[0]
    pc_y[destination] = point[1]
    pc_z[destination] = point[2]
    pc_orig[destination] = local_point


# ---------------------------------------------------------------------------
# Search: one warp per query
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False, launch_bounds=FMA_WARP_SIZE)
def radius_search_sparse_fma_kernel(
    queries: wp.array2d(dtype=wp.vec3),
    Q: wp.int32,
    inv_cell: wp.float32,
    radius2: wp.float32,
    hash_capacity: wp.int32,
    cell_rep_plus1: wp.array(dtype=wp.int32),
    cell_coords: wp.array(dtype=wp.vec4i),
    cell_offsets: wp.array(dtype=wp.int32),
    pc_x: wp.array(dtype=wp.float32),
    pc_y: wp.array(dtype=wp.float32),
    pc_z: wp.array(dtype=wp.float32),
    pc_orig: wp.array(dtype=wp.int32),
    max_points: wp.int32,
    return_dists: wp.bool,
    return_points: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
    out_pts: wp.array2d(dtype=wp.vec3),
):
    """One 32-lane warp per query; lane-0 hash probe, all lanes scan the cell.

    Mirrors the dense v2 FMA search (coalesced unit-stride point loads, native
    ballot compaction, ``max_points`` check every 32 candidates). The only
    change is that the neighbour cell's point range comes from a lane-0 hash
    probe broadcast to the warp instead of a dense-row lookup. Output padding
    is fused into this kernel so no separate memset launch is needed.
    """
    flat_query, lane = wp.tid()

    batch = flat_query // Q
    local_query = flat_query - batch * Q
    query = queries[batch, local_query]

    query_cell = _quantize_cell(query, inv_cell)

    # Fuse output-padding initialisation (strided across the warp).
    output_slot = lane
    while output_slot < max_points:
        out_idx[flat_query, output_slot] = 0
        if return_dists:
            out_dist[flat_query, output_slot] = 0.0
        if return_points:
            out_pts[flat_query, output_slot] = wp.vec3(0.0)
        output_slot += FMA_WARP_SIZE

    found = wp.int32(0)

    for neighbor in range(27):
        offset = _decode_neighbor_offset_27(neighbor)
        neighbor_cell = wp.vec3i(
            query_cell[0] + offset[0],
            query_cell[1] + offset[1],
            query_cell[2] + offset[2],
        )

        # Only lane zero walks the probe sequence; broadcast the result.
        cell_slot = wp.int32(-1)
        if lane == 0:
            cell_slot = _lookup_sparse_cell(
                batch, neighbor_cell, hash_capacity, cell_rep_plus1, cell_coords
            )
        cell_slot = _warp_broadcast_i32(cell_slot, wp.int32(0))

        if cell_slot >= 0:
            point_begin = wp.int32(0)
            point_end = wp.int32(0)
            if lane == 0:
                point_begin = cell_offsets[cell_slot]
                point_end = cell_offsets[cell_slot + 1]
            point_begin = _warp_broadcast_i32(point_begin, wp.int32(0))
            point_end = _warp_broadcast_i32(point_end, wp.int32(0))

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
                    point_x = pc_x[point_index]
                    point_y = pc_y[point_index]
                    point_z = pc_z[point_index]
                    dx = point_x - query[0]
                    dy = point_y - query[1]
                    dz = point_z - query[2]
                    distance2 = dx * dx + dy * dy + dz * dz
                    if distance2 <= radius2:
                        hit = wp.int32(1)

                compact = _warp_hit_prefix_and_count(hit, lane)
                hit_prefix = compact[0]
                chunk_hits = compact[1]
                accepted_hits = wp.min(chunk_hits, max_points - found)

                if hit == 1 and hit_prefix < accepted_hits:
                    destination = found + hit_prefix
                    out_idx[flat_query, destination] = pc_orig[point_index]
                    if return_dists:
                        out_dist[flat_query, destination] = wp.sqrt(distance2)
                    if return_points:
                        out_pts[flat_query, destination] = wp.vec3(
                            point_x, point_y, point_z
                        )

                found += accepted_hits
                if found >= max_points:
                    break
                chunk_begin += FMA_WARP_SIZE

        if found >= max_points:
            break

    if lane == 0:
        out_count[flat_query] = found
