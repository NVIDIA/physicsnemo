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
