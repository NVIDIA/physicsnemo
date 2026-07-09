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

"""Warp kernel for accumulating per-vertex surface area."""

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
