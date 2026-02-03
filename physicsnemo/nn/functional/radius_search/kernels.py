# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
):
    """
    Warp kernel for performing radius search queries on a set of points,
    storing the results of neighboring points within a specified radius.

    Args:
        hashgrid: An array representing the hash grid.
        points: An array of points in space.
        queries: An array of query points.
        result_offset: An array to store the offset in the results array for each query point.
        result_point_idx: An array to store the indices of neighboring points found within the radius for each query point.
        result_point_dist: An array to store the distances to neighboring points within the radius for each query point.
        radius: The search radius around each query point.
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    radius_squared = radius * radius

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute full distance to neighbor point check avoiding the square root
        if check_distance(qp, neighbor, radius_squared):
            # Set the index as a matched pair from query set to points set:
            result_point_idx[0, offset_tid + result_count] = tid
            result_point_idx[1, offset_tid + result_count] = index
            result_count += 1


@wp.kernel
def radius_search_unlimited_select_with_points(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array2d(dtype=wp.int32),
    result_points: wp.array(dtype=wp.vec3),
    radius: wp.float32,
):
    """
    Warp kernel for performing radius search queries on a set of points,
    storing the results of neighboring points within a specified radius.

    Args:
        hashgrid: An array representing the hash grid.
        points: An array of points in space.
        queries: An array of query points.
        result_offset: An array to store the offset in the results array for each query point.
        result_point_idx: An array to store the indices of neighboring points found within the radius for each query point.
        result_point_dist: An array to store the distances to neighboring points within the radius for each query point.
        radius: The search radius around each query point.
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    radius_squared = radius * radius

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute full distance to neighbor point check avoiding the square root
        if check_distance(qp, neighbor, radius_squared):
            # Set the index as a matched pair from query set to points set:
            result_point_idx[0, offset_tid + result_count] = tid
            result_point_idx[1, offset_tid + result_count] = index
            result_points[offset_tid + result_count] = neighbor
            result_count += 1


@wp.kernel
def radius_search_unlimited_select_with_dists(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array2d(dtype=wp.int32),
    result_point_dist: wp.array(dtype=wp.float32),
    radius: wp.float32,
):
    """
    Warp kernel for performing radius search queries on a set of points,
    storing the results of neighboring points within a specified radius.

    Args:
        hashgrid: An array representing the hash grid.
        points: An array of points in space.
        queries: An array of query points.
        result_offset: An array to store the offset in the results array for each query point.
        result_point_idx: An array to store the indices of neighboring points found within the radius for each query point.
        result_point_dist: An array to store the distances to neighboring points within the radius for each query point.
        radius: The search radius around each query point.
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute full distance to neighbor point check avoiding the square root
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            # Set the index as a matched pair from query set to points set:
            result_point_idx[0, offset_tid + result_count] = tid
            result_point_idx[1, offset_tid + result_count] = index
            result_point_dist[offset_tid + result_count] = dist
            result_count += 1


@wp.kernel
def radius_search_unlimited_select_with_dists_and_points(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array2d(dtype=wp.int32),
    result_point_dist: wp.array(dtype=wp.float32),
    result_points: wp.array(dtype=wp.vec3),
    radius: wp.float32,
):
    """
    Warp kernel for performing radius search queries on a set of points,
    storing the results of neighboring points within a specified radius.

    Args:
        hashgrid: An array representing the hash grid.
        points: An array of points in space.
        queries: An array of query points.
        result_offset: An array to store the offset in the results array for each query point.
        result_point_idx: An array to store the indices of neighboring points found within the radius for each query point.
        result_point_dist: An array to store the distances to neighboring points within the radius for each query point.
        radius: The search radius around each query point.
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute full distance to neighbor point check avoiding the square root
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            # Set the index as a matched pair from query set to points set:
            result_point_idx[0, offset_tid + result_count] = tid
            result_point_idx[1, offset_tid + result_count] = index
            result_point_dist[offset_tid + result_count] = dist
            result_points[offset_tid + result_count] = neighbor
            result_count += 1


@wp.kernel
def radius_search_limited_select(
    hash_grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    max_points: wp.int32,
    radius: wp.float32,
    mapping: wp.array2d(dtype=wp.int32),
    num_neighbors: wp.array(dtype=wp.int32),
    return_dists: wp.bool,
    distances: wp.array2d(dtype=wp.float32),
    return_points: wp.bool,
    result_points: wp.array2d(dtype=wp.vec3),
):
    """
    Performs ball query operation to find neighboring points within a specified radius.

    For each point in points, finds up to k neighboring points from points2 that are
    within the specified radius. Uses a hash grid for efficient spatial queries.

    Note that the neighbors found are not strictly guaranteed to be the closest k neighbors,
    in the event that more than k neighbors are found within the radius.

    Args:
        points: Array of points to search
        queries: Array of query points
        grid: Pre-computed hash grid for accelerated spatial queries
        k: Maximum number of neighbors to find for each query point
        radius: Maximum search radius for finding neighbors
        mapping: Output array to store indices of neighboring points. Should be instantiated as zeros(1, len(points), k)
        num_neighbors: Output array to store the number of neighbors found for each query point. Should be instantiated as zeros(1, len(points))
    """
    tid = wp.tid()

    # Get position from points
    pos = queries[tid]

    # particle contact
    neighbors = wp.hash_grid_query(id=hash_grid, point=pos, max_dist=radius)

    # Keep track of the number of neighbors found
    neighbors_found = wp.int32(0)

    radius_squared = radius * radius

    # loop through neighbors to compute density
    for index in neighbors:
        # Check if outside the radius
        pos2 = points[index]
        if not check_distance(pos, pos2, radius_squared):
            continue

        # Add neighbor to the list
        mapping[tid, neighbors_found] = index
        if return_dists:
            distances[tid, neighbors_found] = wp.length(pos - pos2)
        if return_points:
            result_points[tid, neighbors_found] = pos2
        # Increment the number of neighbors found
        neighbors_found += 1

        # Break if we have found enough neighbors
        if neighbors_found == max_points:
            num_neighbors[tid] = max_points
            break

    # Set the number of neighbors
    num_neighbors[tid] = neighbors_found


@wp.kernel
def scatter_add(
    indexes: wp.array2d(dtype=wp.int32),  # [num_inputs, num_indices]
    num_neighbors: wp.array(dtype=wp.int32),  # [num_inputs]
    grad_outputs: wp.array2d(dtype=wp.vec3),  # [num_outputs, vec_dim]
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

    # How many indexes do we loop over?
    this_neighbors = num_neighbors[tid]

    for j in range(this_neighbors):
        # Get the index for this query point:
        idx = indexes[tid, j]
        # Select the gradient from the output:
        grad = grad_outputs[tid, j]
        # Atomically add each component of the vector
        # for k in range(3):  # assuming vec3
        wp.atomic_add(grad_inputs, idx, grad)


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
