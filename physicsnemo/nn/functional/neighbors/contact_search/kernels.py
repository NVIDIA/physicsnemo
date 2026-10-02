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

"""Warp kernels for topology-filtered radius contact search."""

import warp as wp


@wp.func
def _within_radius(
    point: wp.vec3,
    neighbor: wp.vec3,
    radius_squared: wp.float32,
) -> wp.bool:
    delta = point - neighbor
    return wp.dot(delta, delta) <= radius_squared


@wp.func
def _sorted_range_contains(
    value: wp.int32,
    neighbors: wp.array(dtype=wp.int32),
    lower: wp.int32,
    upper: wp.int32,
) -> wp.bool:
    """Binary-search a half-open range in a sorted array."""
    end = upper

    while lower < upper:
        middle = lower + (upper - lower) // 2
        candidate = neighbors[middle]
        if candidate < value:
            lower = middle + 1
        else:
            upper = middle

    if lower < end:
        return neighbors[lower] == value
    return False


@wp.kernel
def contact_search_count(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    exclude_neighbors: wp.array(dtype=wp.int32),
    exclude_offsets: wp.array(dtype=wp.int32),
    exclusion_row_base: wp.int32,
    radius: wp.float32,
    result_count: wp.array(dtype=wp.int32),
):
    """Count spatial neighbors not present in each query's exclusion row."""
    query_index = wp.tid()
    query_point = queries[query_index]
    spatial_query = wp.hash_grid_query(hashgrid, query_point, radius)
    point_index = wp.int32(0)
    count = wp.int32(0)
    radius_squared = radius * radius
    exclusion_row = exclusion_row_base + query_index
    exclusion_start = exclude_offsets[exclusion_row]
    exclusion_end = exclude_offsets[exclusion_row + 1]

    while wp.hash_grid_query_next(spatial_query, point_index):
        neighbor = points[point_index]
        if _within_radius(
            query_point, neighbor, radius_squared
        ) and not _sorted_range_contains(
            point_index, exclude_neighbors, exclusion_start, exclusion_end
        ):
            count += 1

    result_count[query_index] = count


@wp.kernel
def contact_search_unlimited_select(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    exclude_neighbors: wp.array(dtype=wp.int32),
    exclude_offsets: wp.array(dtype=wp.int32),
    exclusion_row_base: wp.int32,
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array2d(dtype=wp.int32),
    radius: wp.float32,
    return_dists: wp.bool,
    result_point_dist: wp.array(dtype=wp.float32),
    return_points: wp.bool,
    result_points: wp.array(dtype=wp.vec3),
):
    """Write all spatial neighbors not present in the exclusion CSR."""
    query_index = wp.tid()
    query_point = queries[query_index]
    spatial_query = wp.hash_grid_query(hashgrid, query_point, radius)
    point_index = wp.int32(0)
    count = wp.int32(0)
    output_offset = result_offset[query_index]
    radius_squared = radius * radius
    exclusion_row = exclusion_row_base + query_index
    exclusion_start = exclude_offsets[exclusion_row]
    exclusion_end = exclude_offsets[exclusion_row + 1]

    while wp.hash_grid_query_next(spatial_query, point_index):
        neighbor = points[point_index]
        if not _within_radius(query_point, neighbor, radius_squared):
            continue
        if _sorted_range_contains(
            point_index, exclude_neighbors, exclusion_start, exclusion_end
        ):
            continue

        output_index = output_offset + count
        result_point_idx[0, output_index] = query_index
        result_point_idx[1, output_index] = point_index
        if return_dists:
            result_point_dist[output_index] = wp.length(query_point - neighbor)
        if return_points:
            result_points[output_index] = neighbor
        count += 1


@wp.func
def _farther(
    distance_a: wp.float32,
    index_a: wp.int32,
    distance_b: wp.float32,
    index_b: wp.int32,
) -> wp.bool:
    """Compare heap keys; point IDs make Warp's exact-distance ties repeatable."""
    return distance_a > distance_b or (distance_a == distance_b and index_a > index_b)


@wp.func
def _heap_sift_down(
    mapping: wp.array(dtype=wp.int32),
    squared_distances: wp.array(dtype=wp.float32),
    base: wp.int32,
    count: wp.int32,
):
    """Restore a query's max-heap after replacing its farthest (root) entry."""
    point_index = mapping[base]
    distance = squared_distances[base]
    slot = wp.int32(0)
    while 2 * slot + 1 < count:
        child = 2 * slot + 1
        right = child + 1
        if right < count:
            if _farther(
                squared_distances[base + right],
                mapping[base + right],
                squared_distances[base + child],
                mapping[base + child],
            ):
                child = right
        if not _farther(
            squared_distances[base + child],
            mapping[base + child],
            distance,
            point_index,
        ):
            break
        mapping[base + slot] = mapping[base + child]
        squared_distances[base + slot] = squared_distances[base + child]
        slot = child
    mapping[base + slot] = point_index
    squared_distances[base + slot] = distance


@wp.func
def _heap_offer(
    mapping: wp.array(dtype=wp.int32),
    squared_distances: wp.array(dtype=wp.float32),
    base: wp.int32,
    count: wp.int32,
    capacity: wp.int32,
    point_index: wp.int32,
    distance: wp.float32,
) -> wp.int32:
    """Retain the nearest capacity candidates in O(log(capacity)) per update."""
    if count < capacity:
        slot = count
        while slot > 0:
            parent = (slot - 1) // 2
            if not _farther(
                distance,
                point_index,
                squared_distances[base + parent],
                mapping[base + parent],
            ):
                break
            mapping[base + slot] = mapping[base + parent]
            squared_distances[base + slot] = squared_distances[base + parent]
            slot = parent
        mapping[base + slot] = point_index
        squared_distances[base + slot] = distance
        return count + 1

    if _farther(squared_distances[base], mapping[base], distance, point_index):
        mapping[base] = point_index
        squared_distances[base] = distance
        _heap_sift_down(mapping, squared_distances, base, count)
    return count


@wp.func
def _heap_sort(
    mapping: wp.array(dtype=wp.int32),
    squared_distances: wp.array(dtype=wp.float32),
    base: wp.int32,
    count: wp.int32,
):
    """Sort retained candidates nearest-first without touching padded slots."""
    remaining = count
    while remaining > 1:
        remaining -= 1
        point_index = mapping[base]
        distance = squared_distances[base]
        mapping[base] = mapping[base + remaining]
        squared_distances[base] = squared_distances[base + remaining]
        mapping[base + remaining] = point_index
        squared_distances[base + remaining] = distance
        _heap_sift_down(mapping, squared_distances, base, remaining)


@wp.kernel
def contact_search_limited_select(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    exclude_neighbors: wp.array(dtype=wp.int32),
    exclude_offsets: wp.array(dtype=wp.int32),
    max_points: wp.int32,
    radius: wp.float32,
    mapping: wp.array(dtype=wp.int32),
    num_neighbors: wp.array(dtype=wp.int32),
    return_dists: wp.bool,
    distances: wp.array(dtype=wp.float32),
    return_points: wp.bool,
    result_points: wp.array2d(dtype=wp.vec3),
):
    """Select the nearest eligible neighbors using a bounded per-query heap."""
    query_index = wp.tid()
    query_point = queries[query_index]
    spatial_query = wp.hash_grid_query(hashgrid, query_point, radius)
    count = wp.int32(0)
    radius_squared = radius * radius
    exclusion_start = exclude_offsets[query_index]
    exclusion_end = exclude_offsets[query_index + 1]
    base = query_index * max_points

    for point_index in spatial_query:
        neighbor = points[point_index]
        if not _within_radius(query_point, neighbor, radius_squared):
            continue
        if _sorted_range_contains(
            point_index, exclude_neighbors, exclusion_start, exclusion_end
        ):
            continue

        delta = query_point - neighbor
        count = _heap_offer(
            mapping,
            distances,
            base,
            count,
            max_points,
            point_index,
            wp.dot(delta, delta),
        )

    _heap_sort(mapping, distances, base, count)
    for slot in range(count):
        if return_dists:
            distances[base + slot] = wp.sqrt(distances[base + slot])
        if return_points:
            result_points[query_index, slot] = points[mapping[base + slot]]
    num_neighbors[query_index] = count


@wp.kernel
def contact_search_limited_select_batched(
    hash_grids: wp.array(dtype=wp.uint64),
    points: wp.array2d(dtype=wp.vec3),
    queries: wp.array2d(dtype=wp.vec3),
    exclude_neighbors: wp.array(dtype=wp.int32),
    exclude_offsets: wp.array(dtype=wp.int32),
    exclusion_row_stride: wp.int32,
    max_points: wp.int32,
    radius: wp.float32,
    mapping: wp.array(dtype=wp.int32),
    num_neighbors: wp.array2d(dtype=wp.int32),
    return_dists: wp.bool,
    distances: wp.array(dtype=wp.float32),
    return_points: wp.bool,
    result_points: wp.array3d(dtype=wp.vec3),
):
    """Batched nearest-k search with independent per-query bounded heaps."""
    batch_index, query_index = wp.tid()
    grid_id = hash_grids[batch_index]
    query_point = queries[batch_index, query_index]
    spatial_query = wp.hash_grid_query(grid_id, query_point, radius)
    count = wp.int32(0)
    radius_squared = radius * radius
    exclusion_row = batch_index * exclusion_row_stride + query_index
    exclusion_start = exclude_offsets[exclusion_row]
    exclusion_end = exclude_offsets[exclusion_row + 1]
    base = (batch_index * queries.shape[1] + query_index) * max_points

    for point_index in spatial_query:
        neighbor = points[batch_index, point_index]
        if not _within_radius(query_point, neighbor, radius_squared):
            continue
        if _sorted_range_contains(
            point_index, exclude_neighbors, exclusion_start, exclusion_end
        ):
            continue

        delta = query_point - neighbor
        count = _heap_offer(
            mapping,
            distances,
            base,
            count,
            max_points,
            point_index,
            wp.dot(delta, delta),
        )

    _heap_sort(mapping, distances, base, count)
    for slot in range(count):
        if return_dists:
            distances[base + slot] = wp.sqrt(distances[base + slot])
        if return_points:
            result_points[batch_index, query_index, slot] = points[
                batch_index, mapping[base + slot]
            ]
    num_neighbors[batch_index, query_index] = count


__all__ = [
    "contact_search_count",
    "contact_search_limited_select",
    "contact_search_limited_select_batched",
    "contact_search_unlimited_select",
]
