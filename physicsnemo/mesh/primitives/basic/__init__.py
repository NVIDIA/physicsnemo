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

"""Minimal test meshes for basic validation.

These meshes contain single cells or a few cells and are useful for unit testing
and validating basic mesh operations.
"""

from physicsnemo.mesh.primitives.basic import (
    single_edge_2d,
    single_edge_3d,
    single_point_2d,
    single_point_3d,
    single_tetrahedron,
    single_triangle_2d,
    single_triangle_3d,
    three_edges_2d,
    three_edges_3d,
    three_points_2d,
    three_points_3d,
    two_tetrahedra,
    two_triangles_2d,
    two_triangles_3d,
)
