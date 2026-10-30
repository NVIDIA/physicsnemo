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

"""Warp kernel for updating remeshing centroids."""

import warp as wp


@wp.kernel
def update_centroids(
    centroids: wp.array(dtype=wp.vec3f),
    centroid_sums: wp.array2d(dtype=wp.float32),
    centroid_areas: wp.array(dtype=wp.float32),
):
    """Move nonempty centroids and clear accumulators for the next iteration."""
    centroid_index = wp.tid()
    weight = centroid_areas[centroid_index]
    if weight > float(0.0):
        centroids[centroid_index] = wp.vec3f(
            centroid_sums[centroid_index, 0] / weight,
            centroid_sums[centroid_index, 1] / weight,
            centroid_sums[centroid_index, 2] / weight,
        )

    centroid_sums[centroid_index, 0] = float(0.0)
    centroid_sums[centroid_index, 1] = float(0.0)
    centroid_sums[centroid_index, 2] = float(0.0)
    centroid_areas[centroid_index] = float(0.0)
