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

"""Tests for planar example meshes."""

import pytest
import torch

from physicsnemo.mesh import primitives


class TestPlanarPrimitives:
    """Test all planar example meshes (2D→2D)."""

    @pytest.mark.parametrize(
        "example_name",
        [
            "unit_square",
            "rectangle",
            "equilateral_triangle",
            "regular_polygon",
            "circle_2d",
            "annulus_2d",
            "l_shape",
            "structured_grid",
        ],
    )
    def test_planar_mesh(self, example_name):
        """Test that planar mesh loads with correct dimensions."""
        primitives_module = getattr(primitives.planar, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 2
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_subdivision_control(self):
        """Test that subdivision parameter works."""
        square_coarse = primitives.planar.unit_square.load(subdivisions=0)
        square_fine = primitives.planar.unit_square.load(subdivisions=2)

        assert square_fine.n_points > square_coarse.n_points
        assert square_fine.n_cells > square_coarse.n_cells

    def test_regular_polygon(self):
        """Test regular polygon creation."""
        # Triangle
        tri = primitives.planar.regular_polygon.load(n_sides=3)
        assert tri.n_cells >= 3

        # Hexagon
        hex = primitives.planar.regular_polygon.load(n_sides=6)
        assert hex.n_cells >= 6

    def test_annulus(self):
        """Test annulus (ring) creation."""
        annulus = primitives.planar.annulus_2d.load(inner_radius=0.5, outer_radius=1.0)

        # Check that points span the expected radial range
        radii = torch.norm(annulus.points, dim=1)
        assert radii.min() >= 0.5 - 0.1  # Allow some tolerance
        assert radii.max() <= 1.0 + 0.1
