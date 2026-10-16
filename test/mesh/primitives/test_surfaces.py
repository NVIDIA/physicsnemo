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

"""Tests for surface example meshes."""

import itertools

import pytest
import torch

from physicsnemo.mesh import primitives


class TestSurfacePrimitives:
    """Test all surface example meshes (2D→3D)."""

    @pytest.mark.parametrize(
        "example_name",
        [
            # Spheres
            # "sphere_icosahedral",
            "sphere_uv",
            # Cylinders
            "cylinder",
            "cylinder_open",
            # Other shapes
            "torus",
            "plane",
            "cone",
            "disk",
            "hemisphere",
            # Platonic solids
            "cube_surface",
            "tetrahedron_surface",
            "octahedron_surface",
            "icosahedron_surface",
            # Special
            "mobius_strip",
        ],
    )
    def test_surface_mesh(self, example_name):
        """Test that surface mesh loads with correct dimensions."""
        primitives_module = getattr(primitives.surfaces, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_torus_radii(self):
        """Test that torus has correct radii."""
        major_radius = 2.0
        minor_radius = 0.5
        torus = primitives.surfaces.torus.load(
            major_radius=major_radius,
            minor_radius=minor_radius,
            n_major=32,
            n_minor=16,
        )

        # Check that points are in expected range
        radii_xy = torch.norm(torus.points[:, :2], dim=1)
        assert radii_xy.min() >= major_radius - minor_radius - 0.1
        assert radii_xy.max() <= major_radius + minor_radius + 0.1

    def test_closed_vs_open(self):
        """Test that closed and open surfaces have correct topology."""
        # Closed cylinder should have no boundary
        cylinder_closed = primitives.surfaces.cylinder.load(n_circ=16, n_height=5)

        # Open cylinder should have boundary
        cylinder_open = primitives.surfaces.cylinder_open.load(n_circ=16, n_height=5)

        # Both should have points
        assert cylinder_closed.n_points > 0
        assert cylinder_open.n_points > 0

    def test_hemisphere_single_pole_vertex(self):
        """The pole is one vertex, so no triangle is degenerate."""
        mesh = primitives.surfaces.hemisphere.load(theta_resolution=30)

        assert mesh.validate()["valid"]
        tri = mesh.points.double()[mesh.cells]
        areas = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert (areas.norm(dim=1) > 0).all()

        # The only boundary is the 30-edge equator
        boundary = mesh.get_boundary_mesh()
        assert boundary.n_cells == 30
        z = boundary.points[boundary.cells][..., 2]
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)

    def test_cone_single_apex_vertex(self):
        """The apex is one vertex, so the capped cone is closed."""
        mesh = primitives.surfaces.cone.load(n_circ=32)

        assert mesh.validate()["valid"]
        assert mesh.is_watertight()
        tri = mesh.points.double()[mesh.cells]
        areas = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert (areas.norm(dim=1) > 0).all()

    def test_tetrahedron_surface_side_length(self):
        """Every edge has length side_length."""
        mesh = primitives.surfaces.tetrahedron_surface.load(side_length=2.0)

        for i, j in itertools.combinations(range(4), 2):
            edge = (mesh.points[i] - mesh.points[j]).norm()
            assert torch.isclose(edge, torch.tensor(2.0))

    @pytest.mark.parametrize("radius", [2.5, 1e20, 1e-25])
    @pytest.mark.parametrize("subdivisions", [0, 1, 2])
    def test_sphere_icosahedral_radius(self, subdivisions, radius):
        """All vertices lie at the requested radius, at every level and scale."""
        mesh = primitives.surfaces.sphere_icosahedral.load(
            radius=radius, subdivisions=subdivisions
        )

        radii = mesh.points.double().norm(dim=1) / radius
        assert torch.allclose(radii, torch.ones_like(radii))
