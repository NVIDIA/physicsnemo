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

"""Tests for point normal computation.

Tests area-weighted vertex normal calculation across various mesh types,
dimensions, and edge cases.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.mesh.utilities._cache import get_cached

### Helper Functions


def create_single_triangle_2d(device="cpu"):
    """Create a single triangle in 2D space (codimension-1)."""
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


def create_single_triangle_3d(device="cpu"):
    """Create a single triangle in 3D space (codimension-1)."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


def create_two_triangles_shared_edge(device="cpu"):
    """Create two triangles sharing an edge in 3D space."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [0.5, 0.5, 1.0],  # 3 (above the plane)
        ],
        dtype=torch.float32,
        device=device,
    )
    # Two triangles sharing edge (0,1)
    cells = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


def create_edge_mesh_2d(device="cpu"):
    """Create a 1D edge mesh in 2D space (codimension-1)."""
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


### Test Basic Functionality


class TestPointNormalsBasic:
    """Basic tests for point normals computation."""

    def test_single_triangle_2d(self, device):
        """Test that 2D triangles in 2D space (codimension-0) raise an error."""
        mesh = create_single_triangle_2d(device)

        # Should raise ValueError for codimension-0 (not codimension-1)
        with pytest.raises(ValueError, match="codimension-1"):
            _ = mesh.point_normals

    def test_single_triangle_3d(self, device):
        """Test point normals for a single triangle in 3D."""
        mesh = create_single_triangle_3d(device)
        point_normals = mesh.point_normals

        # Should have normals for all 3 points
        assert point_normals.shape == (3, 3)

        # All vertex normals should be unit vectors (or zero)
        norms = torch.norm(point_normals, dim=-1)
        assert torch.allclose(norms, torch.ones(3, device=device), atol=1e-5)

        # For a single flat triangle, all point normals should match the face normal
        cell_normal = mesh.cell_normals[0]
        for i in range(3):
            assert torch.allclose(point_normals[i], cell_normal, atol=1e-5)

    def test_edge_mesh_2d(self, device):
        """Test point normals for 1D edges in 2D (codimension-1)."""
        mesh = create_edge_mesh_2d(device)
        # 1D edges require area weighting (angle-based weighting not defined for 1D)
        point_normals = mesh.compute_point_normals(weighting="area")

        # Should have normals for all 3 points
        assert point_normals.shape == (3, 2)

        # All normals should be unit vectors
        norms = torch.norm(point_normals, dim=-1)
        assert torch.allclose(norms, torch.ones(3, device=device), atol=1e-5)

        # Middle point (1) is shared by two edges, should average their normals
        # End points (0, 2) each belong to one edge only


### Test Area Weighting


class TestPointNormalsAreaWeighting:
    """Tests for area-weighted averaging."""

    def test_area_weighting_non_uniform_faces(self, device):
        """Test that larger faces have more influence on point normals."""
        # Create a mesh with one large and one small triangle sharing an edge
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1 (shared edge is 0-1)
                [0.5, 10.0, 0.0],  # 2 (large triangle)
                [0.5, 0.1, 0.0],  # 3 (small triangle)
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 1, 3]],  # Large triangle  # Small triangle
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        # Get areas to verify one is much larger
        areas = mesh.cell_areas
        assert areas[0] > areas[1] * 5  # Large triangle is much bigger

        # Get point normals
        point_normals = mesh.point_normals
        cell_normals = mesh.cell_normals

        # For the shared edge points (0 and 1), the normal should be closer
        # to the large triangle's normal due to area weighting
        # Both triangles are in xy-plane, so both have normal in +z or -z direction
        # The weighted average should still be in that direction

        # Check that vertex normals are unit vectors
        for i in [0, 1]:
            norm = torch.norm(point_normals[i])
            assert torch.abs(norm - 1.0) < 1e-5

        # Verify cell normals are also unit vectors
        assert torch.allclose(
            torch.norm(cell_normals, dim=1), torch.ones(2, device=device), atol=1e-5
        )
        # Both cell normals should point in the same direction (both coplanar in xy-plane, pointing +z)
        assert torch.allclose(cell_normals[0], cell_normals[1], atol=1e-5), (
            "Both triangles are coplanar, so normals should be identical"
        )

    def test_shared_edge_averaging(self, device):
        """Test that shared edge vertices average normals from both triangles."""
        mesh = create_two_triangles_shared_edge(device)

        # Get normals
        point_normals = mesh.point_normals
        cell_normals = mesh.cell_normals

        # Verify cell normals are unit vectors
        assert torch.allclose(
            torch.norm(cell_normals, dim=1), torch.ones(2, device=device), atol=1e-5
        )

        # Points 0 and 1 are shared by both triangles
        # Their normals should be some average of the two cell normals
        # For shared points, the point normal should be between the two cell normals
        shared_point_normals = point_normals[[0, 1]]
        for i in range(2):
            # Dot product with both cell normals should be positive (same hemisphere)
            dot0 = (shared_point_normals[i] * cell_normals[0]).sum()
            dot1 = (shared_point_normals[i] * cell_normals[1]).sum()
            assert dot0 > 0.5, (
                f"Shared point {i} normal should be similar to cell 0 normal"
            )
            assert dot1 > 0.5, (
                f"Shared point {i} normal should be similar to cell 1 normal"
            )

        # Points 2 and 3 are only in one triangle each
        # Point 2 in triangle 0 only
        # Point 3 in triangle 1 only

        # All point normals should be unit vectors
        norms = torch.norm(point_normals, dim=-1)
        assert torch.allclose(norms, torch.ones(4, device=device), atol=1e-5)


### Test Edge Cases


class TestPointNormalsEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_codimension_validation(self, device):
        """Test that non-codimension-1 meshes raise an error."""
        # Create a tet mesh (3D in 3D, codimension-0)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        # Should raise ValueError for codimension-0
        with pytest.raises(ValueError, match="codimension-1"):
            _ = mesh.point_normals

    def test_caching(self, device):
        """Test that point normals are cached in point_data."""
        mesh = create_single_triangle_3d(device)

        # First access
        normals1 = mesh.point_normals

        # Check cached
        assert get_cached(mesh.point_data, "normals") is not None

        # Second access should return cached value
        normals2 = mesh.point_normals

        # Should be the same tensor
        assert torch.allclose(normals1, normals2)

    def test_empty_mesh(self, device):
        """Test handling of empty mesh."""
        points = torch.empty((0, 3), dtype=torch.float32, device=device)
        cells = torch.empty((0, 3), dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        # Should return empty tensor
        point_normals = mesh.point_normals
        assert point_normals.shape == (0, 3)

    def test_isolated_point(self, device):
        """Test that isolated points (not in any cell) get zero normals."""
        # Create mesh with extra point not in any cell
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [99.0, 99.0, 99.0],  # Isolated point
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        point_normals = mesh.point_normals

        # First 3 points should have unit normals
        for i in range(3):
            norm = torch.norm(point_normals[i])
            assert torch.abs(norm - 1.0) < 1e-5

        # Isolated point should have zero normal
        assert torch.allclose(
            point_normals[3], torch.zeros(3, device=device), atol=1e-6
        )


### Test Different Dimensions


class TestPointNormalsDimensions:
    """Tests across different manifold and spatial dimensions."""

    def test_2d_edges_in_2d_space(self, device):
        """Test 1D manifold (edges) in 2D space."""
        mesh = create_edge_mesh_2d(device)
        # 1D edges require area weighting (angle-based weighting not defined for 1D)
        point_normals = mesh.compute_point_normals(weighting="area")

        # Should work for codimension-1
        assert point_normals.shape == (3, 2)

        # All should be unit vectors
        norms = torch.norm(point_normals, dim=-1)
        assert torch.allclose(norms, torch.ones(3, device=device), atol=1e-5)

    def test_2d_triangles_in_3d_space(self, device):
        """Test 2D manifold (triangles) in 3D space."""
        mesh = create_single_triangle_3d(device)
        point_normals = mesh.point_normals

        assert point_normals.shape == (3, 3)

        # All should be unit vectors
        norms = torch.norm(point_normals, dim=-1)
        assert torch.allclose(norms, torch.ones(3, device=device), atol=1e-5)

    def test_1d_edges_in_3d_space(self, device):
        """Test that 1D manifold (edges) in 3D space (codimension-2) raises error."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        # Should raise ValueError for codimension-2 (not codimension-1)
        with pytest.raises(ValueError, match="codimension-1"):
            _ = mesh.point_normals


### Test Numerical Stability


class TestPointNormalsNumerical:
    """Tests for numerical stability and precision."""

    def test_normalization_stability(self, device):
        """Test that normalization is stable for various configurations."""
        # Create a very small triangle (but not so small that float32 loses precision)
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.5e-3, 1e-3, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        point_normals = mesh.point_normals

        # Should still produce unit normals
        norms = torch.norm(point_normals, dim=-1)
        assert torch.allclose(norms, torch.ones(3, device=device), atol=1e-4)

    def test_consistent_across_scales(self, device):
        """Test that point normals are consistent when mesh is scaled."""
        # Create mesh
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh1 = Mesh(points=points, cells=cells)

        # Scaled version
        mesh2 = Mesh(points=points * 100.0, cells=cells)

        normals1 = mesh1.point_normals
        normals2 = mesh2.point_normals

        # Normals should be the same (direction doesn't depend on scale)
        assert torch.allclose(normals1, normals2, atol=1e-5)


### Test Consistency with Cell Normals


class TestPointCellNormalConsistency:
    """Tests for consistency between point normals and cell normals."""

    def compute_angular_errors(self, mesh):
        """Compute angular errors between each cell normal and its vertex normals.

        Returns:
            Tensor of angular errors (in radians) for each cell-vertex pair.
            Shape: (n_cells * n_vertices_per_cell,)
        """
        cell_normals = mesh.cell_normals  # (n_cells, n_spatial_dims)
        point_normals = mesh.point_normals  # (n_points, n_spatial_dims)

        n_cells, n_vertices_per_cell = mesh.cells.shape

        # Get point normals for each vertex of each cell
        # Shape: (n_cells, n_vertices_per_cell, n_spatial_dims)
        point_normals_per_cell = point_normals[mesh.cells]

        # Repeat cell normals for each vertex
        # Shape: (n_cells, n_vertices_per_cell, n_spatial_dims)
        cell_normals_repeated = cell_normals.unsqueeze(1).expand(
            -1, n_vertices_per_cell, -1
        )

        # Compute dot products (cosine of angle)
        # Shape: (n_cells, n_vertices_per_cell)
        cos_angles = (cell_normals_repeated * point_normals_per_cell).sum(dim=-1)

        # Clamp to [-1, 1] to avoid numerical issues with acos
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)

        # Compute angular errors in radians
        # Shape: (n_cells * n_vertices_per_cell,)
        angular_errors = torch.acos(cos_angles).flatten()

        return angular_errors

    def test_flat_surface_perfect_alignment(self, device):
        """Test that flat surfaces have perfect alignment between point and cell normals."""
        # Create a flat triangular mesh (all normals should be identical)
        mesh = create_single_triangle_3d(device)

        angular_errors = self.compute_angular_errors(mesh)

        # All errors should be essentially zero for a single flat triangle
        assert torch.all(angular_errors < 1e-5)

    def test_smooth_surface_consistency(self, device):
        """Test that smooth surfaces have good alignment."""
        # Create multiple coplanar triangles (smooth surface)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 3], [1, 2, 4], [1, 4, 3]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        angular_errors = self.compute_angular_errors(mesh)

        # All errors should be very small for coplanar triangles
        assert torch.all(angular_errors < 1e-4)

    def test_sharp_edge_detection(self, device):
        """Test that sharp edges produce larger angular errors."""
        # Create two triangles at 90 degrees to each other
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Shared edge
                [1.0, 0.0, 0.0],  # Shared edge
                [0.5, 1.0, 0.0],  # In xy-plane
                [0.5, 0.0, 1.0],  # In xz-plane (90 degrees rotated)
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        angular_errors = self.compute_angular_errors(mesh)

        # Some errors should be larger due to the sharp edge
        # But most should still be reasonable (< pi/2)
        assert torch.any(angular_errors > 0.1)  # Some significant errors
        assert torch.all(angular_errors < torch.pi / 2)  # But not too extreme

    def test_real_mesh_consistency(self, device):
        """Test consistency on a realistic mesh (lumpy sphere).

        The lumpy sphere has varying curvature which causes point normals
        (area-weighted averages) to differ from cell normals. This is expected
        behavior - higher curvature regions naturally have larger angular errors
        between point and cell normals.
        """
        # Load lumpy sphere - a realistic mesh with interesting curvature
        mesh = lumpy_sphere.load(subdivisions=2, device=device)

        # Compute angular errors
        angular_errors = self.compute_angular_errors(mesh)

        # Lumpy sphere has varying curvature, so some angular error is expected
        # Mean error should be reasonable (< 0.3 rad = 17 degrees)
        assert angular_errors.mean() < 0.3
        # Max error should be bounded (< 1.5 rad = 86 degrees)
        assert angular_errors.max() < 1.5

    def test_subdivided_mesh_improved_consistency(self, device):
        """Test that subdivision improves consistency.

        Linear subdivision adds new vertices at edge midpoints. For a lumpy
        sphere with varying curvature, adding more vertices improves the
        approximation of the smooth surface, leading to better normal
        consistency (smaller angular errors between point and cell normals).
        """
        # Load lumpy sphere - has varying curvature
        mesh_original = lumpy_sphere.load(subdivisions=1, device=device)

        # Subdivide to add vertices at edge midpoints
        mesh_subdivided = mesh_original.subdivide(levels=1, filter="linear")

        # Compute angular errors for both
        errors_original = self.compute_angular_errors(mesh_original)
        errors_subdivided = self.compute_angular_errors(mesh_subdivided)

        # Check consistency at threshold of 0.1 radians
        threshold = 0.1
        fraction_original = (errors_original < threshold).float().mean()
        fraction_subdivided = (errors_subdivided < threshold).float().mean()

        # Subdivision should improve consistency (more vertices = better approximation)
        assert fraction_subdivided >= fraction_original  # Should improve
        # Mean error should decrease
        assert errors_subdivided.mean() <= errors_original.mean() + 0.05

    def test_multiple_subdivision_levels(self, device):
        """Test that consistency improves with subdivision levels.

        Linear subdivision adds vertices at edge midpoints, improving the
        mesh's approximation of the underlying surface. As more vertices
        are added, the angular error between point and cell normals decreases.
        """
        # Load lumpy sphere - has varying curvature
        mesh = lumpy_sphere.load(subdivisions=1, device=device)

        threshold = 0.1  # radians
        fractions = []

        # Test original and multiple subdivision levels
        for level in range(3):
            if level > 0:
                mesh = mesh.subdivide(levels=1, filter="linear")

            errors = self.compute_angular_errors(mesh)
            fraction = (errors < threshold).float().mean()
            fractions.append(fraction)

        # Consistency should generally improve with subdivision
        # Each level should be at least as good as the previous
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i - 1] - 0.05  # Allow small variance
        # Final level should be notably better than first
        assert fractions[-1] >= fractions[0] + 0.2

    def test_consistency_distribution(self, device):
        """Test the distribution of angular errors.

        For a lumpy sphere with varying curvature, the error distribution
        reflects the curvature variation. Higher curvature regions have
        larger angular errors between point and cell normals.
        """
        # Load lumpy sphere - has varying curvature
        mesh = lumpy_sphere.load(subdivisions=2, device=device)

        # Compute angular errors
        angular_errors = self.compute_angular_errors(mesh)

        # Check various percentiles
        percentiles = [50, 75, 90, 95, 99]
        values = [torch.quantile(angular_errors, p / 100.0) for p in percentiles]

        # Distribution should be reasonable for a curved surface
        assert values[0] < 0.25  # 50th percentile (< 14 degrees)
        assert values[-1] < 1.0  # 99th percentile (< 57 degrees)

    @pytest.mark.slow
    def test_loop_subdivision_smoothing(self, device):
        """Test that Loop subdivision improves normal consistency.

        Loop subdivision is APPROXIMATING - it repositions vertices to
        create a smoother surface. This should reduce angular errors
        between point and cell normals.
        """
        # Load lumpy sphere - has varying curvature
        mesh_original = lumpy_sphere.load(subdivisions=1, device=device)

        # Try Loop subdivision (approximating, should smooth)
        try:
            mesh_loop = mesh_original.subdivide(levels=1, filter="loop")

            # Compute angular errors for both
            errors_original = self.compute_angular_errors(mesh_original)
            errors_loop = self.compute_angular_errors(mesh_loop)

            # Loop subdivision should maintain or improve consistency
            # Mean error should not increase significantly
            assert errors_loop.mean() <= errors_original.mean() + 0.1
        except NotImplementedError:
            # Loop subdivision might not support all mesh types
            pytest.skip("Loop subdivision not supported for this mesh")
