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

"""Tests for physicsnemo.mesh.generate.marching_cubes."""

import math

import pytest
import torch

from physicsnemo.mesh.generate import marching_cubes


def _sphere_sdf(
    resolution: int = 32, radius: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a sphere SDF on a [-1, 1]^3 grid.

    Returns the SDF field and the 1D coordinate vector (same for all axes).
    """
    coords = torch.linspace(-1, 1, resolution)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    sdf = torch.sqrt(xx**2 + yy**2 + zz**2) - radius
    return sdf, coords


class TestMarchingCubes:
    """Tests for marching cubes isosurface extraction."""

    def test_returns_triangle_mesh(self):
        sdf, _ = _sphere_sdf()
        mesh = marching_cubes(sdf)
        assert mesh.n_spatial_dims == 3
        assert mesh.n_manifold_dims == 2

    def test_nonempty_output(self):
        sdf, _ = _sphere_sdf()
        mesh = marching_cubes(sdf)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_cell_indices_in_range(self):
        sdf, _ = _sphere_sdf()
        mesh = marching_cubes(sdf)
        assert mesh.cells.min() >= 0
        assert mesh.cells.max() < mesh.n_points

    def test_dtypes(self):
        sdf, _ = _sphere_sdf()
        mesh = marching_cubes(sdf)
        assert mesh.points.dtype == torch.float32
        assert mesh.cells.dtype == torch.int64

    def test_custom_threshold(self):
        sdf, _ = _sphere_sdf(resolution=32, radius=0.5)
        mesh_small = marching_cubes(sdf, threshold=0.2)
        mesh_large = marching_cubes(sdf, threshold=-0.2)
        assert mesh_small.n_points > mesh_large.n_points


class TestCoords:
    """Tests for the coords parameter (physical coordinate mapping)."""

    def test_vertices_in_physical_space(self):
        """With coords, vertices should lie within the coordinate bounds."""
        sdf, coords = _sphere_sdf(resolution=32, radius=0.5)
        mesh = marching_cubes(sdf, coords=(coords, coords, coords))
        assert mesh.points.min() >= coords[0].item()
        assert mesh.points.max() <= coords[-1].item()

    def test_vertices_in_index_space_without_coords(self):
        """Without coords, vertices should be in grid-index space."""
        sdf, _ = _sphere_sdf(resolution=32, radius=0.5)
        mesh = marching_cubes(sdf)
        assert mesh.points.min() >= 0
        assert mesh.points.max() <= 31

    def test_coords_length_mismatch_raises(self):
        sdf, _ = _sphere_sdf(resolution=32)
        wrong = torch.linspace(0, 1, 64)
        with pytest.raises(ValueError, match="coords"):
            marching_cubes(sdf, coords=(wrong, wrong, wrong))

    def test_anisotropic_coords(self):
        """Different coordinate ranges per axis should scale accordingly."""
        sdf, _ = _sphere_sdf(resolution=32, radius=0.5)
        cx = torch.linspace(0, 10, 32)
        cy = torch.linspace(-5, 5, 32)
        cz = torch.linspace(0, 1, 32)
        mesh = marching_cubes(sdf, coords=(cx, cy, cz))
        assert mesh.points[:, 0].min() >= 0
        assert mesh.points[:, 0].max() <= 10
        assert mesh.points[:, 1].min() >= -5
        assert mesh.points[:, 1].max() <= 5
        assert mesh.points[:, 2].min() >= 0
        assert mesh.points[:, 2].max() <= 1

    def test_nonuniform_coords(self):
        """Non-uniform coords should place vertices via piecewise linear interp."""
        sdf, _ = _sphere_sdf(resolution=32, radius=0.5)
        uniform = torch.linspace(-1, 1, 32)
        # Quadratic spacing: denser near the center
        nonuniform = torch.sign(uniform) * uniform**2

        mesh_uniform = marching_cubes(sdf, coords=(uniform, uniform, uniform))
        mesh_nonuniform = marching_cubes(
            sdf, coords=(nonuniform, nonuniform, nonuniform)
        )

        # Both should produce valid meshes with the same topology
        assert mesh_nonuniform.n_cells == mesh_uniform.n_cells
        # Non-uniform vertices should be within the non-uniform coord bounds
        assert mesh_nonuniform.points.min() >= nonuniform[0].item()
        assert mesh_nonuniform.points.max() <= nonuniform[-1].item()
        # Vertices should differ (the mapping is different)
        assert not torch.allclose(mesh_uniform.points, mesh_nonuniform.points)


class TestGeometricAccuracy:
    """Geometric validation of extracted isosurfaces."""

    def test_sphere_surface_area(self):
        """Surface area of extracted sphere should approximate 4*pi*r^2."""
        radius = 0.5
        sdf, coords = _sphere_sdf(resolution=64, radius=radius)
        mesh = marching_cubes(sdf, coords=(coords, coords, coords))

        total_area = mesh.cell_areas.sum().item()
        expected_area = 4 * math.pi * radius**2

        assert total_area == pytest.approx(expected_area, rel=0.02)

    def test_sphere_is_watertight(self):
        """Extracted sphere should be a closed surface."""
        sdf, coords = _sphere_sdf(resolution=32, radius=0.5)
        mesh = marching_cubes(sdf, coords=(coords, coords, coords))
        assert mesh.is_watertight()

    def test_sphere_is_manifold(self):
        """Extracted sphere should be a valid 2-manifold."""
        sdf, coords = _sphere_sdf(resolution=32, radius=0.5)
        mesh = marching_cubes(sdf, coords=(coords, coords, coords))
        assert mesh.is_manifold()

    def test_sphere_centroid_near_origin(self):
        """Centroid of an origin-centered sphere should be near (0, 0, 0)."""
        sdf, coords = _sphere_sdf(resolution=64, radius=0.5)
        mesh = marching_cubes(sdf, coords=(coords, coords, coords))
        centroid = mesh.points.mean(dim=0)
        assert torch.allclose(centroid, torch.zeros(3), atol=0.05)

    def test_no_degenerate_cells(self):
        """Extracted mesh should have no zero-area triangles."""
        sdf, coords = _sphere_sdf(resolution=32, radius=0.5)
        mesh = marching_cubes(sdf, coords=(coords, coords, coords))
        assert (mesh.cell_areas > 0).all()


class TestMarchingCubesValidation:
    """Input validation and error handling."""

    def test_rejects_2d_input(self):
        with pytest.raises(NotImplementedError, match="3D scalar fields"):
            marching_cubes(torch.randn(10, 10))

    def test_rejects_4d_input(self):
        with pytest.raises(NotImplementedError, match="3D scalar fields"):
            marching_cubes(torch.randn(10, 10, 10, 10))
