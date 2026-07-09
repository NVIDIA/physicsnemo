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

"""Warp kernels used by GPU surface remeshing.

This module intentionally contains no PyTorch orchestration. Keeping kernels
separate makes their device-side contracts explicit and avoids importing Warp
from the public remeshing namespace until the GPU backend is selected.
"""

import warp as wp


@wp.kernel
def accumulate_vertex_areas(
    points: wp.array(dtype=wp.vec3f),
    cells: wp.array2d(dtype=wp.int32),
    vertex_areas: wp.array(dtype=wp.float32),
):
    """Accumulate one third of each triangle area at its vertices."""
    face = wp.tid()
    i0 = cells[face, 0]
    i1 = cells[face, 1]
    i2 = cells[face, 2]

    edge_1 = points[i1] - points[i0]
    edge_2 = points[i2] - points[i0]
    area_share = wp.length(wp.cross(edge_1, edge_2)) / float(6.0)

    wp.atomic_add(vertex_areas, i0, area_share)
    wp.atomic_add(vertex_areas, i1, area_share)
    wp.atomic_add(vertex_areas, i2, area_share)


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
    """Assign vertices to their nearest centroid and optionally reduce them.

    The hash-grid query is exact whenever it finds a centroid within
    ``search_radius``. A global scan handles sparse or disconnected regions,
    preserving correctness without forcing every query to inspect every
    centroid.
    """
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

    # A well-spaced initialization makes this path rare. It is deliberately
    # retained for arbitrary disconnected inputs and highly nonuniform meshes.
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


@wp.kernel
def update_centroids(
    centroids: wp.array(dtype=wp.vec3f),
    centroid_sums: wp.array2d(dtype=wp.float32),
    centroid_areas: wp.array(dtype=wp.float32),
):
    """Move nonempty centroids to their area-weighted cluster centers."""
    centroid_index = wp.tid()
    weight = centroid_areas[centroid_index]
    if weight > float(0.0):
        centroids[centroid_index] = wp.vec3f(
            centroid_sums[centroid_index, 0] / weight,
            centroid_sums[centroid_index, 1] / weight,
            centroid_sums[centroid_index, 2] / weight,
        )


@wp.kernel
def project_centroids_to_surface(
    mesh_id: wp.uint64,
    centroids: wp.array(dtype=wp.vec3f),
    max_distance: wp.float32,
):
    """Project centroids to their closest points on the source surface."""
    centroid_index = wp.tid()
    query = wp.mesh_query_point_sign_normal(
        mesh_id, centroids[centroid_index], max_distance
    )
    if query.result:
        mesh = wp.mesh_get(mesh_id)
        p0 = mesh.points[mesh.indices[3 * query.face + 0]]
        p1 = mesh.points[mesh.indices[3 * query.face + 1]]
        p2 = mesh.points[mesh.indices[3 * query.face + 2]]
        centroids[centroid_index] = (
            query.u * p0 + query.v * p1 + (float(1.0) - query.u - query.v) * p2
        )
