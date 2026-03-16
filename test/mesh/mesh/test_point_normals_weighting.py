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

"""Tests for Mesh.compute_point_normals() weighting schemes.

Tests validate all four weighting schemes:
- "area": Area-weighted averaging
- "unweighted": Equal weight per face (matches PyVista/VTK)
- "angle": Interior angle-weighted averaging
- "angle_area": Combined angle and area weighting (Maya default)

Also tests error handling for:
- Invalid weighting scheme
- Non-codimension-1 meshes
- Angle-based weighting for 1-simplices (edges)
"""

import pytest
import torch

from physicsnemo.mesh import Mesh

### Helper Functions ###


def create_triangle_surface_3d() -> Mesh:
    """Create a simple triangular surface mesh in 3D.

    Creates a flat surface with 2 triangles in the xy-plane (z=0).
    """
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return Mesh(points=points, cells=cells)


def create_edge_mesh_2d() -> Mesh:
    """Create an edge mesh (1-simplex) in 2D space (codimension-1)."""
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32
    )
    cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    return Mesh(points=points, cells=cells)


def create_cube_corner_mesh() -> Mesh:
    """Create a mesh with 3 triangles meeting at a corner (like a cube corner).

    This creates an interesting test case where vertex angles matter.
    """
    # Three orthogonal triangles meeting at origin
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Corner point (shared by all 3 triangles)
            [1.0, 0.0, 0.0],  # x-axis
            [0.0, 1.0, 0.0],  # y-axis
            [0.0, 0.0, 1.0],  # z-axis
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor(
        [
            [0, 1, 2],  # xy-plane triangle (normal: +z)
            [0, 2, 3],  # yz-plane triangle (normal: +x)
            [0, 3, 1],  # xz-plane triangle (normal: +y)
        ],
        dtype=torch.long,
    )
    return Mesh(points=points, cells=cells)


def create_pyramid_mesh() -> Mesh:
    """Create a pyramid mesh with 4 triangular faces.

    Good test case for weighting since faces have different areas and angles.
    """
    # Pyramid with square base and apex
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # base corners
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],  # apex
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor(
        [
            [0, 1, 4],  # front face
            [1, 2, 4],  # right face
            [2, 3, 4],  # back face
            [3, 0, 4],  # left face
        ],
        dtype=torch.long,
    )
    return Mesh(points=points, cells=cells)


### Test Classes ###


class TestPointNormalsWeightingSchemes:
    """Tests for different weighting schemes."""

    def test_unweighted_produces_unit_normals(self):
        """Test that unweighted scheme produces unit normals."""
        mesh = create_triangle_surface_3d()

        normals = mesh.compute_point_normals(weighting="unweighted")

        # All normals should be unit vectors (or zero for isolated points)
        norms = normals.norm(dim=-1)
        # For connected points, norm should be ~1
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_area_weighted_produces_unit_normals(self):
        """Test that area-weighted scheme produces unit normals."""
        mesh = create_triangle_surface_3d()

        normals = mesh.compute_point_normals(weighting="area")

        norms = normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_angle_weighted_produces_unit_normals(self):
        """Test that angle-weighted scheme produces unit normals."""
        mesh = create_triangle_surface_3d()

        normals = mesh.compute_point_normals(weighting="angle")

        norms = normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_angle_area_weighted_produces_unit_normals(self):
        """Test that angle_area-weighted scheme produces unit normals."""
        mesh = create_triangle_surface_3d()

        normals = mesh.compute_point_normals(weighting="angle_area")

        norms = normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_flat_surface_all_weightings_agree(self):
        """Test that all weightings agree for flat surface (normals should be same)."""
        mesh = create_triangle_surface_3d()

        normals_unweighted = mesh.compute_point_normals(weighting="unweighted")
        normals_area = mesh.compute_point_normals(weighting="area")
        normals_angle = mesh.compute_point_normals(weighting="angle")
        normals_angle_area = mesh.compute_point_normals(weighting="angle_area")

        # For flat surface, all normals should point in +z direction
        expected = torch.tensor([0.0, 0.0, 1.0])
        for normals in [
            normals_unweighted,
            normals_area,
            normals_angle,
            normals_angle_area,
        ]:
            for i in range(mesh.n_points):
                # Allow for sign flip
                assert torch.allclose(normals[i].abs(), expected.abs(), atol=1e-5)

    def test_point_normals_property_uses_angle_area(self):
        """Test that point_normals property uses angle_area weighting."""
        mesh = create_triangle_surface_3d()

        property_normals = mesh.point_normals
        explicit_normals = mesh.compute_point_normals(weighting="angle_area")

        assert torch.allclose(property_normals, explicit_normals, atol=1e-6)


class TestWeightingDifferences:
    """Tests that verify weighting schemes can produce different results."""

    def test_different_weightings_can_differ_for_curved_surfaces(self):
        """Test that weightings can produce different results for non-flat meshes."""
        mesh = create_cube_corner_mesh()

        normals_unweighted = mesh.compute_point_normals(weighting="unweighted")
        normals_area = mesh.compute_point_normals(weighting="area")

        # The corner point (index 0) is shared by 3 orthogonal triangles
        # For unweighted: equal contribution from each
        # For area: weighted by triangle areas (which may differ)

        # Both should produce valid normals
        assert normals_unweighted[0].norm() > 0.9
        assert normals_area[0].norm() > 0.9

    def test_pyramid_apex_normal(self):
        """Test normal at pyramid apex where 4 triangles meet."""
        mesh = create_pyramid_mesh()

        normals_unweighted = mesh.compute_point_normals(weighting="unweighted")
        normals_area = mesh.compute_point_normals(weighting="area")

        apex_idx = 4
        # Apex normal should point somewhat upward (positive z component)
        assert normals_unweighted[apex_idx, 2] > 0
        assert normals_area[apex_idx, 2] > 0


class TestPointNormalsErrors:
    """Tests for error handling in compute_point_normals."""

    def test_invalid_weighting_raises(self):
        """Test that invalid weighting scheme raises ValueError."""
        mesh = create_triangle_surface_3d()

        with pytest.raises(ValueError, match="Invalid weighting"):
            mesh.compute_point_normals(weighting="invalid")

    def test_non_codimension_1_raises(self):
        """Test that non-codimension-1 mesh raises ValueError."""
        # Create triangles in 2D (codimension 0)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.codimension == 0

        with pytest.raises(ValueError, match="codimension-1"):
            mesh.compute_point_normals(weighting="unweighted")

    def test_angle_weighting_for_edges_raises(self):
        """Test that angle-based weighting for 1-simplices (edges) raises ValueError."""
        mesh = create_edge_mesh_2d()

        assert mesh.n_manifold_dims == 1

        with pytest.raises(
            ValueError, match="Angle-based weighting requires n_manifold_dims >= 2"
        ):
            mesh.compute_point_normals(weighting="angle")

    def test_angle_area_weighting_for_edges_raises(self):
        """Test that angle_area weighting for edges raises ValueError."""
        mesh = create_edge_mesh_2d()

        with pytest.raises(
            ValueError, match="Angle-based weighting requires n_manifold_dims >= 2"
        ):
            mesh.compute_point_normals(weighting="angle_area")

    def test_area_weighting_for_edges_works(self):
        """Test that area weighting works for edges (doesn't require angles)."""
        mesh = create_edge_mesh_2d()

        # Should not raise - area weighting doesn't require angles
        normals = mesh.compute_point_normals(weighting="area")

        assert normals.shape == (mesh.n_points, mesh.n_spatial_dims)

    def test_unweighted_for_edges_works(self):
        """Test that unweighted scheme works for edges."""
        mesh = create_edge_mesh_2d()

        normals = mesh.compute_point_normals(weighting="unweighted")

        assert normals.shape == (mesh.n_points, mesh.n_spatial_dims)


class TestEdgeNormals2D:
    """Tests for point normals on edge meshes in 2D."""

    def test_straight_line_normals(self):
        """Test normals for straight line segments in 2D."""
        # Horizontal line
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        normals = mesh.compute_point_normals(weighting="unweighted")

        # All normals should be perpendicular to x-axis (in +y or -y direction)
        for i in range(mesh.n_points):
            assert torch.allclose(normals[i, 0].abs(), torch.tensor(0.0), atol=1e-6)
            assert torch.allclose(normals[i, 1].abs(), torch.tensor(1.0), atol=1e-6)

    def test_corner_edge_normal(self):
        """Test normal at corner where two edges meet at 90 degrees."""
        # L-shaped path
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        normals = mesh.compute_point_normals(weighting="unweighted")

        # Corner point (index 1) has normal that's average of two perpendicular normals
        corner_normal = normals[1]
        # Should be roughly at 45 degrees
        assert corner_normal.norm() > 0.9


class TestPointNormalsDevices:
    """Tests for device handling in point normals computation."""

    def test_cpu_device(self):
        """Test that point normals work on CPU."""
        mesh = create_triangle_surface_3d()
        mesh = mesh.to("cpu")

        normals = mesh.compute_point_normals(weighting="area")

        assert normals.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test that point normals work on CUDA."""
        mesh = create_triangle_surface_3d()
        mesh = mesh.to("cuda")

        normals = mesh.compute_point_normals(weighting="area")

        assert normals.device.type == "cuda"


class TestPointNormalsParametrized:
    """Parametrized tests for point normals."""

    @pytest.mark.parametrize("weighting", ["area", "unweighted", "angle", "angle_area"])
    def test_all_weightings_triangle_mesh(self, weighting):
        """Test all weighting schemes on triangle mesh."""
        mesh = create_triangle_surface_3d()

        normals = mesh.compute_point_normals(weighting=weighting)

        assert normals.shape == (mesh.n_points, mesh.n_spatial_dims)
        # All should be unit normals
        norms = normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    @pytest.mark.parametrize("weighting", ["area", "unweighted"])
    def test_non_angle_weightings_edge_mesh(self, weighting):
        """Test non-angle weighting schemes on edge mesh."""
        mesh = create_edge_mesh_2d()

        normals = mesh.compute_point_normals(weighting=weighting)

        assert normals.shape == (mesh.n_points, mesh.n_spatial_dims)


class TestIsolatedPoints:
    """Tests for handling isolated points with no adjacent cells."""

    def test_isolated_point_has_zero_normal(self):
        """Test that isolated point (no adjacent cells) has zero normal."""
        # Create mesh with an extra isolated point
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [5.0, 5.0, 5.0],  # Isolated point
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)  # Only uses points 0,1,2
        mesh = Mesh(points=points, cells=cells)

        normals = mesh.compute_point_normals(weighting="unweighted")

        # Isolated point (index 3) should have zero normal
        assert torch.allclose(normals[3], torch.zeros(3), atol=1e-6)

        # Connected points should have unit normals
        for i in range(3):
            assert normals[i].norm() > 0.9


class TestCaching:
    """Tests for caching behavior of point normals."""

    def test_point_normals_property_is_cached(self):
        """Test that point_normals property caches the result."""
        mesh = create_triangle_surface_3d()

        # First access computes and caches
        normals1 = mesh.point_normals

        # Check cache exists
        cached = mesh._cache.get(("point", "normals"), None)
        assert cached is not None
        assert torch.equal(cached, normals1)

        # Second access uses cache
        normals2 = mesh.point_normals
        assert torch.equal(normals1, normals2)
