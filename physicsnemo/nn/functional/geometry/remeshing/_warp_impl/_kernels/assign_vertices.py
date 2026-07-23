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

"""Warp kernel for assigning source vertices to remeshing clusters."""

import warp as wp


@wp.kernel
def assign_vertices(
    hash_grid_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    centroids: wp.array(dtype=wp.vec3f),
    vertex_areas: wp.array(dtype=wp.float32),
    labels: wp.array(dtype=wp.int32),
    centroid_sums: wp.array2d(dtype=wp.float32),
    centroid_areas: wp.array(dtype=wp.float32),
    search_radius: wp.float32,
    accumulate: wp.int32,
):
    """Assign vertices to their nearest centroid and optionally reduce them."""
    point_index = wp.tid()
    point = points[point_index]
    radius_sq = search_radius * search_radius

    best_index = int(-1)
    best_distance_sq = float(1.0e30)
    candidate_index = int(0)
    query = wp.hash_grid_query(hash_grid_id, point, search_radius)
    while wp.hash_grid_query_next(query, candidate_index):
        if candidate_index < centroids.shape[0]:
            delta = point - centroids[candidate_index]
            distance_sq = wp.dot(delta, delta)
            if distance_sq <= radius_sq and distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_index = candidate_index

    # The default radius and well-spaced initialization make this path rare on
    # ordinary connected surfaces. Each miss scans O(n_clusters), so a badly
    # undersized user radius can degrade assignment to O(n_points * n_clusters).
    # Retain the fallback for correctness on disconnected and highly nonuniform
    # inputs.
    if best_index < 0:
        for centroid_index in range(centroids.shape[0]):
            delta = point - centroids[centroid_index]
            distance_sq = wp.dot(delta, delta)
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_index = centroid_index

    labels[point_index] = best_index
    if accumulate != 0:
        weight = vertex_areas[point_index]
        weighted_point = weight * point
        wp.atomic_add(centroid_sums, best_index, 0, weighted_point[0])
        wp.atomic_add(centroid_sums, best_index, 1, weighted_point[1])
        wp.atomic_add(centroid_sums, best_index, 2, weighted_point[2])
        wp.atomic_add(centroid_areas, best_index, weight)
