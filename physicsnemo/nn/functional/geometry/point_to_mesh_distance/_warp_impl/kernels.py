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

"""Warp kernels for nearest-face BVH queries."""

import warp as wp


@wp.kernel
def nearest_face_indices_f32(
    mesh_id: wp.uint64,
    input_points: wp.array(dtype=wp.vec3f),
    max_distance: wp.float32,
    nearest_faces: wp.array(dtype=wp.int64),
):
    """Return the nearest triangle index for each query point."""

    query_index = wp.tid()
    result = wp.mesh_query_point_no_sign(
        mesh_id, input_points[query_index], max_distance
    )
    if result.result:
        nearest_faces[query_index] = wp.int64(result.face)
    else:
        # The public operation uses an unbounded search on a nonempty mesh, so
        # finite input coordinates always hit. Keep a valid sentinel for
        # nonfinite, out-of-contract inputs rather than emitting a negative
        # index that would wrap in the subsequent Torch gather.
        nearest_faces[query_index] = wp.int64(0)


__all__ = ["nearest_face_indices_f32"]
