# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warp kernels for shrinkwrap nearest-face searches."""

import warp as wp


@wp.kernel
def nearest_surface_faces_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3f),
    max_distance: wp.array(dtype=wp.float32),
    face_ids: wp.array(dtype=wp.int64),
):
    """Return the closest triangle index for every query point."""

    query_index = wp.tid()
    query = wp.mesh_query_point_no_sign(
        mesh_id,
        query_points[query_index],
        max_distance[0],
    )
    if query.result:
        face_ids[query_index] = wp.int64(query.face)
    else:
        face_ids[query_index] = wp.int64(-1)


__all__ = ["nearest_surface_faces_kernel"]
