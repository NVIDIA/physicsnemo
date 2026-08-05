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
This file contains warp kernels for the radius search operations.

It should be pure warp code, no pytorch here.
"""

import warp as wp


@wp.func
def check_distance(
    point: wp.vec3,
    neighbor: wp.vec3,
    radius_squared: wp.float32,
) -> wp.bool:
    """
    Check if a point is within a specified radius of a neighbor point.
    """
    return wp.dot(point - neighbor, point - neighbor) <= radius_squared


@wp.kernel
def radius_search_count(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    result_count: wp.array(dtype=wp.int32),
):
    """
    Warp kernel for counting the number of points within a specified radius
    for each query point, using a hash grid for spatial queries.

    Args:
        hashgrid: An array representing the hash grid.
        points: An array of points in space.
        queries: An array of query points.
        result_count: An array to store the count of neighboring points within the radius for each query point.
        radius: The search radius around each query point.
    """

    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count_tid = int(0)
    radius_squared = radius * radius

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        if check_distance(qp, neighbor, radius_squared):
            result_count_tid += 1

    result_count[tid] = result_count_tid


@wp.kernel
def radius_search_unlimited_select(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array2d(dtype=wp.int32),
    radius: wp.float32,
    return_dists: wp.bool,
    result_point_dist: wp.array(dtype=wp.float32),
    return_points: wp.bool,
    result_points: wp.array(dtype=wp.vec3),
):
    """
    Warp kernel for performing radius search queries on a set of points,
    storing the results of neighboring points within a specified radius.

    Optionally writes distances and/or neighbor coordinates based on the
    return_dists and return_points flags.

    Args:
        hashgrid: The hash grid for spatial queries.
        points: An array of points in space.
        queries: An array of query points.
        result_offset: Per-query offset into the flat output arrays.
        result_point_idx: Output array for (query_idx, point_idx) pairs.
        radius: The search radius around each query point.
        return_dists: Whether to write distances to result_point_dist.
        result_point_dist: Output array for distances (only written when return_dists is True).
        return_points: Whether to write neighbor coordinates to result_points.
        result_points: Output array for neighbor coordinates (only written when return_points is True).
    """
    tid = wp.tid()

    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    radius_squared = radius * radius

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        if check_distance(qp, neighbor, radius_squared):
            out_idx = offset_tid + result_count
            result_point_idx[0, out_idx] = tid
            result_point_idx[1, out_idx] = index
            if return_dists:
                result_point_dist[out_idx] = wp.length(qp - neighbor)
            if return_points:
                result_points[out_idx] = neighbor
            result_count += 1


@wp.kernel
def scatter_add_unlimited(
    indexes: wp.array2d(dtype=wp.int32),  # [num_inputs, num_indices]
    grad_outputs: wp.array(dtype=wp.vec3),  # [num_outputs, vec_dim]
    grad_inputs: wp.array(dtype=wp.vec3),  # [num_inputs, vec_dim]
):
    """
    For each input (thread), sum grad_outputs at the given indexes and atomically add to grad_inputs.
    Args:
        indexes: 2D array of indices into grad_outputs for each input.
        grad_outputs: 2D array of output gradients (vectors).
        grad_inputs: 2D array of input gradients (vectors) to be updated atomically.
    """

    # Indexes is a mapping, from the forward pass of the radius search.
    # It has shape [n_queries, max_points] and
    # represents the points selected from `points` for each query.

    # grad_outputs is the gradients on the selected points, of shape
    # [n_queries, max_points, 3]

    # grad_inputs is the to-be-updated gradient vector for the inputs.
    # Should be initialized before the kernel, from torch, with shape
    # [n_points, 3]

    # We use one thread per query point.
    # So this tid is used to index into `indexes` and `grad_outputs`

    tid = wp.tid()

    # Get the index for this query point:
    neighbor_pt_idx = indexes[1, tid]

    # Select the gradient from the output:
    grad = grad_outputs[tid]
    # Atomically add each component of the vector
    # for k in range(3):  # assuming vec3
    wp.atomic_add(grad_inputs, neighbor_pt_idx, grad)


# ---------------------------------------------------------------------------
# Batched kernel variants -- launched with dim=(B, N_queries)
# ---------------------------------------------------------------------------


@wp.kernel
def radius_search_limited_select_batched(
    hash_grids: wp.array(dtype=wp.uint64),
    points: wp.array2d(dtype=wp.vec3),
    queries: wp.array2d(dtype=wp.vec3),
    max_points: wp.int32,
    radius: wp.float32,
    mapping: wp.array3d(dtype=wp.int32),
    num_neighbors: wp.array2d(dtype=wp.int32),
    return_dists: wp.bool,
    distances: wp.array3d(dtype=wp.float32),
    return_points: wp.bool,
    result_points: wp.array3d(dtype=wp.vec3),
):
    """
    Batched ball query: finds up to max_points neighbors per query within radius.

    Launched with dim=(B, N_queries). Each thread handles one (batch, query) pair
    using the pre-built hash grid for its batch element.
    """
    b, tid = wp.tid()
    
    grid_id = hash_grids[b]

    pos = queries[b, tid]
    neighbors = wp.hash_grid_query(id=grid_id, point=pos, max_dist=radius)

    neighbors_found = wp.int32(0)
    radius_squared = radius * radius

    for index in neighbors:
        pos2 = points[b, index]
        if not check_distance(pos, pos2, radius_squared):
            continue

        mapping[b, tid, neighbors_found] = index
        if return_dists:
            distances[b, tid, neighbors_found] = wp.length(pos - pos2)
        if return_points:
            result_points[b, tid, neighbors_found] = pos2
        neighbors_found += 1

        if neighbors_found == max_points:
            num_neighbors[b, tid] = max_points
            break

    num_neighbors[b, tid] = neighbors_found


@wp.kernel
def scatter_add_batched(
    indexes: wp.array3d(dtype=wp.int32),
    num_neighbors: wp.array2d(dtype=wp.int32),
    grad_outputs: wp.array3d(dtype=wp.vec3),
    grad_inputs: wp.array2d(dtype=wp.vec3),
):
    """
    Batched backward scatter-add for the limited (max_points) path.

    Launched with dim=(B, N_queries). For each (batch, query) pair, scatters
    the gradient from grad_outputs back into grad_inputs using the index mapping.
    """
    b, tid = wp.tid()

    this_neighbors = num_neighbors[b, tid]

    for j in range(this_neighbors):
        idx = indexes[b, tid, j]
        grad = grad_outputs[b, tid, j]
        wp.atomic_add(grad_inputs, b, idx, grad)


# ---------------------------------------------------------------------------
# Morton ordering + tiled BVH ball query
# ---------------------------------------------------------------------------


@wp.func
def expand_bits(v: wp.uint32) -> wp.uint32:
    """
    Spread the 10 low bits of ``v`` so that each bit is separated by two zeros.

    This is the standard "magic bits" expansion used to build 30-bit 3D Morton
    codes from three 10-bit integer coordinates.
    """
    v = (v | (v << wp.uint32(16))) & wp.uint32(0x030000FF)
    v = (v | (v << wp.uint32(8))) & wp.uint32(0x0300F00F)
    v = (v | (v << wp.uint32(4))) & wp.uint32(0x030C30C3)
    v = (v | (v << wp.uint32(2))) & wp.uint32(0x09249249)
    return v


@wp.func
def morton3D(x: wp.int32, y: wp.int32, z: wp.int32) -> wp.int32:
    """
    Interleave three 10-bit integer coordinates into a 30-bit Morton code.

    The result fits in a non-negative ``int32`` so it can be sorted directly
    (e.g. with ``torch.argsort``) to order points along a Z-order curve.
    """
    xx = expand_bits(wp.uint32(x))
    yy = expand_bits(wp.uint32(y))
    zz = expand_bits(wp.uint32(z))
    code = (xx << wp.uint32(2)) | (yy << wp.uint32(1)) | zz
    return wp.int32(code)


@wp.kernel
def compute_morton_codes(
    coords: wp.array2d(dtype=wp.vec3),
    bbox_min: wp.array(dtype=wp.vec3),
    inv_extent: wp.array(dtype=wp.vec3),
    codes: wp.array2d(dtype=wp.int32),
):
    """
    Compute a 30-bit Morton code for each coordinate, batched over dim=(B, M).

    Each coordinate is normalized into the unit cube using the per-batch
    bounding box (``bbox_min`` and ``inv_extent = 1 / (bbox_max - bbox_min)``),
    quantized to a 10-bit-per-axis integer lattice, and interleaved.

    Args:
        coords: Coordinates to encode, shape (B, M).
        bbox_min: Per-batch lower corner of the bounding box, shape (B,).
        inv_extent: Per-batch reciprocal extent of the bounding box, shape (B,).
        codes: Output Morton codes, shape (B, M).
    """
    b, i = wp.tid()

    p = coords[b, i]
    mn = bbox_min[b]
    inv = inv_extent[b]

    # Normalize to the unit cube, then scale to the [0, 1023] integer lattice.
    nx = (p[0] - mn[0]) * inv[0]
    ny = (p[1] - mn[1]) * inv[1]
    nz = (p[2] - mn[2]) * inv[2]

    xi = wp.clamp(wp.int32(nx * 1024.0), 0, 1023)
    yi = wp.clamp(wp.int32(ny * 1024.0), 0, 1023)
    zi = wp.clamp(wp.int32(nz * 1024.0), 0, 1023)

    codes[b, i] = morton3D(xi, yi, zi)


@wp.kernel
def tiled_bvh_radius_search(
    bvhs: wp.array(dtype=wp.uint64),
    points: wp.array2d(dtype=wp.vec3),
    queries: wp.array2d(dtype=wp.vec3),
    max_points: wp.int32,
    radius: wp.float32,
    mapping: wp.array3d(dtype=wp.int32),
    num_neighbors: wp.array2d(dtype=wp.int32),
    return_dists: wp.bool,
    distances: wp.array3d(dtype=wp.float32),
    return_points: wp.bool,
    result_points: wp.array3d(dtype=wp.vec3),
):
    """
    Tiled (block-cooperative) BVH ball query: up to max_points neighbors / query.

    Launched as a flattened 1D tile grid of ``B * N_queries`` tiles with a
    non-trivial ``block_dim`` (via ``wp.launch_tiled``). Each thread block owns
    one (batch, query) pair -- decoded from the flat tile index so that ``(b, q)``
    is uniform across the block -- and its lanes cooperatively traverse the
    per-batch BVH for that query's axis-aligned bounding box :math:`[q - r, q + r]`
    using ``wp.tile_bvh_query_aabb``. Candidates are refined with an exact L2
    check and up to ``max_points`` first-found neighbors are recorded.

    A flat 1D launch grid (rather than a 2D ``dim=(B, Q)``) is used deliberately:
    it matches the only tiled-launch shape exercised by Warp's own BVH tests and
    avoids illegal memory accesses observed with 2D tiled launches.

    This is a drop-in analogue of :func:`radius_search_limited_select_batched`;
    the only signature difference is that the first argument carries BVH ids
    instead of hash-grid ids (both ``wp.array(dtype=wp.uint64)``).

    Note:
        The reference points and queries are expected to be Morton-ordered by the
        host (see :func:`radius_search_bvh`) so that nearby queries traverse
        similar BVH subtrees and point gathers are coalesced. ``num_neighbors`` is
        used as an atomic slot counter and may overshoot ``max_points``; the host
        clamps it after the launch.

    Args:
        bvhs: Per-batch BVH ids, shape (B,).
        points: Reference points, shape (B, N).
        queries: Query points, shape (B, Q).
        max_points: Maximum number of neighbors to record per query.
        radius: Search radius.
        mapping: Output neighbor indices (into ``points``), shape (B, Q, max_points).
        num_neighbors: Output neighbor counts per query, shape (B, Q).
        return_dists: Whether to write distances.
        distances: Output distances, shape (B, Q, max_points).
        return_points: Whether to write neighbor coordinates.
        result_points: Output neighbor coordinates, shape (B, Q, max_points).
    """
    # Flattened 1D tile grid: one thread block per (batch, query). Decode (b, q)
    # from the flat tile index; both are block-uniform, so the AABB passed to
    # wp.tile_bvh_query_aabb is identical across the block (a requirement of the
    # tiled query).
    tile = wp.tid()
    n_queries = queries.shape[1]
    b = tile // n_queries
    q = tile % n_queries

    bvh_id = bvhs[b]
    qp = queries[b, q]
    r = wp.vec3(radius, radius, radius)
    low = qp - r
    high = qp + r
    radius_squared = radius * radius

    # Cooperative traversal. The while-loop and the tile BVH-query builtins are
    # collective, so all lanes run the traversal to completion (no early break).
    # Each in-radius hit claims a unique output slot via an atomic counter kept
    # in-place in num_neighbors; slots past max_points are skipped.
    n_points = points.shape[1]
    query = wp.tile_bvh_query_aabb(bvh_id, low, high)
    while wp.tile_query_valid(query):
        # Each lane receives one candidate point index (or -1) this round.
        result_idx = wp.untile(wp.tile_bvh_query_next(query))
        # Guard the upper bound too: defends the points[] gather against any
        # out-of-range index the tiled traversal might hand back.
        if result_idx >= 0 and result_idx < n_points:
            pos2 = points[b, result_idx]
            if check_distance(qp, pos2, radius_squared):
                slot = wp.atomic_add(num_neighbors, b, q, wp.int32(1))
                if slot < max_points:
                    mapping[b, q, slot] = result_idx
                    if return_dists:
                        distances[b, q, slot] = wp.length(qp - pos2)
                    if return_points:
                        result_points[b, q, slot] = pos2
