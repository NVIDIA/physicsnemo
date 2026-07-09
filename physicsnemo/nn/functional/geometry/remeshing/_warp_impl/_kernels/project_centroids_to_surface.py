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

"""Warp kernel for projecting centroids onto a triangle surface."""

import warp as wp


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
