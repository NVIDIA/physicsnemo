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

"""Tests for curve example meshes."""

import pytest

from physicsnemo.mesh import primitives


class TestCurvePrimitives:
    """Test all curve example meshes."""

    @pytest.mark.parametrize(
        "example_name,expected_manifold_dims,expected_spatial_dims",
        [
            # 1D → 1D
            ("line_segment_1d", 1, 1),
            ("line_segments_1d", 1, 1),
            # 1D → 2D
            ("straight_line_2d", 1, 2),
            ("circular_arc_2d", 1, 2),
            ("circle_2d", 1, 2),
            ("ellipse_2d", 1, 2),
            ("polyline_2d", 1, 2),
            ("spiral_2d", 1, 2),
            # 1D → 3D
            ("straight_line_3d", 1, 3),
            ("helix_3d", 1, 3),
            ("circle_3d", 1, 3),
            ("trefoil_knot_3d", 1, 3),
            ("spline_3d", 1, 3),
        ],
    )
    def test_curve_mesh(
        self, example_name, expected_manifold_dims, expected_spatial_dims
    ):
        """Test that curve mesh loads with correct dimensions."""
        primitives_module = getattr(primitives.curves, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == expected_manifold_dims
        assert mesh.n_spatial_dims == expected_spatial_dims
        assert mesh.n_points >= 2
        assert mesh.n_cells >= 1

    def test_parametric_control(self):
        """Test that parametric curves respond to resolution parameters."""
        # Test n_points parameter
        helix_coarse = primitives.curves.helix_3d.load(n_points=20)
        helix_fine = primitives.curves.helix_3d.load(n_points=100)

        assert helix_fine.n_points > helix_coarse.n_points
        assert helix_fine.n_cells > helix_coarse.n_cells

    def test_closed_curves(self):
        """Test that closed curves have correct topology."""
        # Circle should be closed (no boundary edges)
        circle = primitives.curves.circle_2d.load(n_points=32)

        # For a closed curve, each vertex should appear in exactly 2 edges
        from collections import Counter

        vertex_counts = Counter(circle.cells.flatten().tolist())
        assert all(count == 2 for count in vertex_counts.values())
