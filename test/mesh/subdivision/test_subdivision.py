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

"""Comprehensive tests for mesh subdivision operations.

Tests linear, butterfly, and loop subdivision schemes across various
manifold dimensions, spatial dimensions, and codimensions.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

### Helper Functions


def create_line_mesh(device="cpu"):
    """Create a simple 1D line segment mesh (1-manifold in 2D space)."""
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


def create_triangle_mesh(device="cpu"):
    """Create a simple 2D triangle mesh (2-manifold in 2D space)."""
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


def create_triangle_mesh_3d(device="cpu"):
    """Create a simple 2D triangle mesh in 3D space (2-manifold in 3D, codim=1)."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


def create_tet_mesh(device="cpu"):
    """Create a simple 3D tetrahedral mesh (3-manifold in 3D space)."""
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
    return Mesh(points=points, cells=cells)


### Test Linear Subdivision


class TestLinearSubdivision:
    """Tests for linear subdivision across various manifold dimensions."""

    def test_line_subdivision_single_level(self, device):
        """Test 1D line subdivision - each edge splits into 2."""
        mesh = create_line_mesh(device)
        subdivided = mesh.subdivide(levels=1, filter="linear")

        # Check dimensions preserved
        assert subdivided.n_manifold_dims == 1
        assert subdivided.n_spatial_dims == 2

        # Original: 3 points, 2 edges
        # After 1 level: 3 + 2 = 5 points, 2 * 2 = 4 edges
        assert subdivided.n_points == 5
        assert subdivided.n_cells == 4

    def test_triangle_subdivision_single_level(self, device):
        """Test 2D triangle subdivision - each triangle splits into 4."""
        mesh = create_triangle_mesh(device)
        subdivided = mesh.subdivide(levels=1, filter="linear")

        # Check dimensions preserved
        assert subdivided.n_manifold_dims == 2
        assert subdivided.n_spatial_dims == 2

        # Original: 4 points, 2 triangles
        # Edges: Each triangle has 3 edges, shared edges counted once
        # Triangle 1: (0,1), (1,2), (2,0)
        # Triangle 2: (1,3), (3,2), (2,1) - (2,1) is shared
        # Unique edges: 5
        # After 1 level: 4 + 5 = 9 points, 2 * 4 = 8 triangles
        assert subdivided.n_points == 9
        assert subdivided.n_cells == 8

    def test_triangle_3d_subdivision_single_level(self, device):
        """Test 2D triangles in 3D space (codimension-1)."""
        mesh = create_triangle_mesh_3d(device)
        subdivided = mesh.subdivide(levels=1, filter="linear")

        # Codimension should be preserved
        assert subdivided.codimension == 1
        assert subdivided.n_manifold_dims == 2
        assert subdivided.n_spatial_dims == 3

        # Same topology as 2D triangle mesh
        assert subdivided.n_points == 9
        assert subdivided.n_cells == 8

    def test_tet_subdivision_single_level(self, device):
        """Test 3D tetrahedral subdivision - each tet splits into 8."""
        mesh = create_tet_mesh(device)
        subdivided = mesh.subdivide(levels=1, filter="linear")

        # Check dimensions preserved
        assert subdivided.n_manifold_dims == 3
        assert subdivided.n_spatial_dims == 3

        # Original: 4 points, 1 tet
        # Tet has C(4,2) = 6 edges
        # After 1 level: 4 + 6 = 10 points, 1 * 8 = 8 tets
        assert subdivided.n_points == 10
        assert subdivided.n_cells == 8

    def test_multi_level_subdivision(self, device):
        """Test multiple levels of subdivision."""
        mesh = create_triangle_mesh(device)

        # Level 1
        mesh_1 = mesh.subdivide(levels=1, filter="linear")
        n_points_1 = mesh_1.n_points
        n_cells_1 = mesh_1.n_cells

        # Level 2
        mesh_2 = mesh.subdivide(levels=2, filter="linear")
        assert mesh_2.n_cells == n_cells_1 * 4  # Each triangle splits into 4
        assert mesh_2.n_points > n_points_1  # More points added

        # Level 3
        mesh_3 = mesh.subdivide(levels=3, filter="linear")
        assert mesh_3.n_cells == mesh_2.n_cells * 4

    def test_edge_midpoints_correct(self, device):
        """Test that new vertices are at edge midpoints."""
        # Simple single edge
        points = torch.tensor(
            [[0.0, 0.0], [2.0, 4.0]], dtype=torch.float32, device=device
        )
        cells = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        subdivided = mesh.subdivide(levels=1, filter="linear")

        # Should have 3 points: original 2 + 1 midpoint
        assert subdivided.n_points == 3

        # Find the new point (not in original)
        new_point_mask = torch.ones(
            subdivided.n_points, dtype=torch.bool, device=device
        )
        for i in range(mesh.n_points):
            # Check if original point i is in subdivided mesh
            matches = torch.all(
                torch.isclose(subdivided.points, mesh.points[i].unsqueeze(0)), dim=1
            )
            if matches.any():
                # Find first matching index
                match_idx = torch.where(matches)[0][0]
                new_point_mask[match_idx] = False

        new_point = subdivided.points[new_point_mask][0]
        expected_midpoint = (points[0] + points[1]) / 2

        assert torch.allclose(new_point, expected_midpoint, atol=1e-6)

    def test_point_data_interpolation(self, device):
        """Test that point_data is interpolated to new vertices."""
        mesh = create_line_mesh(device)

        # Add point data
        mesh.point_data["scalar"] = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device=device
        )
        mesh.point_data["vector"] = torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32, device=device
        )

        subdivided = mesh.subdivide(levels=1, filter="linear")

        # Check point_data exists and has correct shape
        assert "scalar" in subdivided.point_data
        assert "vector" in subdivided.point_data
        assert subdivided.point_data["scalar"].shape == (5,)
        assert subdivided.point_data["vector"].shape == (5, 2)

        # Midpoint of first edge (0,1) should have interpolated values
        # Expected: (1.0 + 2.0) / 2 = 1.5 for scalar
        # Check that interpolation happened (values between originals exist)
        scalar_values = subdivided.point_data["scalar"]
        assert scalar_values.min() >= 1.0
        assert scalar_values.max() <= 3.0
        # Should have at least one value between 1 and 2 (midpoint of first edge)
        assert ((scalar_values > 1.0) & (scalar_values < 2.0)).any()

    def test_cell_data_propagation(self, device):
        """Test that cell_data is propagated from parent to children."""
        mesh = create_triangle_mesh(device)

        # Add cell data
        mesh.cell_data["pressure"] = torch.tensor(
            [100.0, 200.0], dtype=torch.float32, device=device
        )

        subdivided = mesh.subdivide(levels=1, filter="linear")

        # Each parent cell splits into 4 children
        # Original: 2 cells -> 8 cells after subdivision
        assert "pressure" in subdivided.cell_data
        assert subdivided.cell_data["pressure"].shape == (8,)

        # Each child should inherit parent's value
        # First 4 cells from first parent (pressure=100), next 4 from second (pressure=200)
        # Check that we have both values propagated
        assert (subdivided.cell_data["pressure"] == 100.0).sum() == 4
        assert (subdivided.cell_data["pressure"] == 200.0).sum() == 4

    def test_empty_mesh(self, device):
        """Test subdivision of empty mesh doesn't crash."""
        points = torch.empty((0, 2), dtype=torch.float32, device=device)
        cells = torch.empty((0, 2), dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        subdivided = mesh.subdivide(levels=1, filter="linear")
        assert subdivided.n_points == 0
        assert subdivided.n_cells == 0

    def test_single_simplex(self, device):
        """Test subdivision of single simplex."""
        # Single edge
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32, device=device
        )
        cells = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        subdivided = mesh.subdivide(levels=1, filter="linear")
        assert subdivided.n_points == 3  # 2 original + 1 midpoint
        assert subdivided.n_cells == 2  # 2 child edges

    @pytest.mark.parametrize("n_levels", [0, 1, 2, 3])
    def test_levels_parameter(self, device, n_levels):
        """Test that levels parameter works correctly."""
        mesh = create_triangle_mesh(device)

        if n_levels == 0:
            subdivided = mesh.subdivide(levels=0, filter="linear")
            # No subdivision should occur
            assert subdivided.n_points == mesh.n_points
            assert subdivided.n_cells == mesh.n_cells
        else:
            subdivided = mesh.subdivide(levels=n_levels, filter="linear")
            # Each level multiplies cells by 4 for triangles
            expected_cells = mesh.n_cells * (4**n_levels)
            assert subdivided.n_cells == expected_cells


### Test Butterfly Subdivision


class TestButterflySubdivision:
    """Tests for butterfly (interpolating) subdivision."""

    def test_triangle_butterfly_preserves_vertices(self, device):
        """Test that butterfly subdivision keeps original vertices unchanged."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        # Use sphere_icosahedral (80 triangles at subdivisions=1) for realistic test
        mesh = sphere_icosahedral.load(subdivisions=1, device=device)
        original_points = mesh.points.clone()

        subdivided = mesh.subdivide(levels=1, filter="butterfly")

        # Original vertices should still exist in subdivided mesh
        # Sample a subset for efficiency (checking all would be slow)
        sample_indices = torch.randperm(mesh.n_points, device=device)[:10]
        for i in sample_indices:
            # Find this point in subdivided mesh
            matches = torch.all(
                torch.isclose(subdivided.points, original_points[i].unsqueeze(0)),
                dim=1,
            )
            assert matches.any(), f"Original vertex {i} not found in subdivided mesh"

    def test_butterfly_topology_same_as_linear(self, device):
        """Test that butterfly has same connectivity as linear (interpolating scheme)."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        # Use lumpy_sphere for a more realistic mesh with varying geometry
        mesh = lumpy_sphere.load(subdivisions=1, device=device)

        linear = mesh.subdivide(levels=1, filter="linear")
        butterfly = mesh.subdivide(levels=1, filter="butterfly")

        # Same number of points and cells
        assert butterfly.n_points == linear.n_points
        assert butterfly.n_cells == linear.n_cells
        assert butterfly.n_manifold_dims == linear.n_manifold_dims

    def test_butterfly_2d_manifold_required(self, device):
        """Test that butterfly requires 2D manifold (or raises informative error)."""
        # This test checks if butterfly subdivision handles non-2D manifolds
        # It might error, or fall back to linear - either is acceptable
        mesh = create_line_mesh(device)

        # Butterfly was designed for 2D manifolds (triangles)
        # For 1D, it should either work or raise a clear error
        try:
            subdivided = mesh.subdivide(levels=1, filter="butterfly")
            # If it works, check it produces valid output
            assert subdivided.n_manifold_dims == 1
            assert subdivided.n_cells > mesh.n_cells
        except (ValueError, NotImplementedError) as e:
            # Acceptable to not support non-2D manifolds
            assert "manifold" in str(e).lower() or "dimension" in str(e).lower()


### Test Loop Subdivision


class TestLoopSubdivision:
    """Tests for Loop (approximating) subdivision."""

    def test_triangle_loop_modifies_vertices(self, device):
        """Test that Loop subdivision repositions original vertices."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        # Use sphere_icosahedral for a realistic closed surface test
        mesh = sphere_icosahedral.load(subdivisions=1, device=device)
        original_points = mesh.points.clone()

        subdivided = mesh.subdivide(levels=1, filter="loop")

        # Loop is approximating - original vertices get repositioned
        assert subdivided.n_points > mesh.n_points

        # Check that original vertices were modified in the subdivided mesh
        # The first n_original_points in the subdivided mesh are the repositioned originals
        n_original = original_points.shape[0]
        repositioned_points = subdivided.points[:n_original]

        # At least one vertex should have moved (Loop smoothing)
        max_displacement = torch.max(
            torch.norm(repositioned_points - original_points, dim=-1)
        )
        assert max_displacement > 1e-6, (
            "Loop subdivision should reposition at least some vertices"
        )

    def test_loop_topology_same_as_linear(self, device):
        """Test that Loop has same connectivity pattern as linear."""
        mesh = create_triangle_mesh(device)

        linear = mesh.subdivide(levels=1, filter="linear")
        loop = mesh.subdivide(levels=1, filter="loop")

        # Same topology, different geometry
        assert loop.n_points == linear.n_points
        assert loop.n_cells == linear.n_cells
        assert loop.n_manifold_dims == linear.n_manifold_dims

    def test_loop_smoothing_effect(self, device):
        """Test that Loop subdivision has smoothing effect on a realistic mesh."""
        from physicsnemo.mesh.primitives.surfaces import icosahedron_surface

        # Use icosahedron (20 triangles) as a more realistic test case
        mesh = icosahedron_surface.load(device=device)

        subdivided = mesh.subdivide(levels=1, filter="loop")

        # After Loop subdivision, mesh should still be valid
        assert subdivided.n_points > mesh.n_points
        assert subdivided.n_cells > mesh.n_cells

        # All cells should still be valid (positive area)
        areas = subdivided.cell_areas
        assert torch.all(areas > 0)

        # Loop subdivision should produce reasonable smoothing (areas should be consistent)
        area_std = areas.std() / areas.mean()
        assert area_std < 1.0, (
            "Loop subdivision should produce reasonably uniform cell areas"
        )


### Test Edge Cases and Validation


class TestSubdivisionValidation:
    """Tests for edge cases and input validation."""

    def test_negative_levels_error(self, device):
        """Test that negative levels raises error."""
        mesh = create_triangle_mesh(device)

        with pytest.raises((ValueError, RuntimeError)):
            mesh.subdivide(levels=-1, filter="linear")

    def test_invalid_filter_error(self, device):
        """Test that invalid filter name raises error."""
        mesh = create_triangle_mesh(device)

        with pytest.raises((ValueError, TypeError)):
            mesh.subdivide(levels=1, filter="invalid")  # type: ignore

    def test_manifold_dimension_preserved(self, device):
        """Test that manifold dimension is preserved across subdivision."""
        meshes = [
            create_line_mesh(device),
            create_triangle_mesh(device),
            create_tet_mesh(device),
        ]

        for mesh in meshes:
            original_n_manifold_dims = mesh.n_manifold_dims
            subdivided = mesh.subdivide(levels=1, filter="linear")
            assert subdivided.n_manifold_dims == original_n_manifold_dims

    def test_spatial_dimension_preserved(self, device):
        """Test that spatial dimension is preserved."""
        mesh = create_triangle_mesh_3d(device)
        assert mesh.n_spatial_dims == 3

        subdivided = mesh.subdivide(levels=1, filter="linear")
        assert subdivided.n_spatial_dims == 3

    def test_global_data_preserved(self, device):
        """Test that global_data is preserved during subdivision."""
        mesh = create_triangle_mesh(device)
        mesh.global_data["timestamp"] = torch.tensor(42.0, device=device)

        subdivided = mesh.subdivide(levels=1, filter="linear")

        assert "timestamp" in subdivided.global_data
        assert subdivided.global_data["timestamp"] == 42.0


### Performance and Scaling Tests


class TestSubdivisionScaling:
    """Tests for subdivision scaling and performance."""

    def test_exponential_cell_growth(self, device):
        """Test that cells grow exponentially with levels."""
        mesh = create_triangle_mesh(device)

        n_cells_0 = mesh.n_cells
        n_cells_1 = mesh.subdivide(levels=1, filter="linear").n_cells
        n_cells_2 = mesh.subdivide(levels=2, filter="linear").n_cells
        n_cells_3 = mesh.subdivide(levels=3, filter="linear").n_cells

        # For 2D triangles: 4x growth per level
        assert n_cells_1 == n_cells_0 * 4
        assert n_cells_2 == n_cells_0 * 16
        assert n_cells_3 == n_cells_0 * 64

    @pytest.mark.slow
    def test_large_mesh_subdivision(self, device):
        """Test subdivision on larger mesh."""
        # Create a moderately large triangle mesh
        n = 10

        # Vectorized grid point generation
        i_coords, j_coords = torch.meshgrid(
            torch.arange(n, dtype=torch.float32, device=device),
            torch.arange(n, dtype=torch.float32, device=device),
            indexing="ij",
        )
        points = torch.stack([i_coords, j_coords], dim=-1).reshape(-1, 2)

        # Vectorized cell generation: two triangles per quad
        i_idx, j_idx = torch.meshgrid(
            torch.arange(n - 1, device=device),
            torch.arange(n - 1, device=device),
            indexing="ij",
        )
        idx = (i_idx * n + j_idx).reshape(-1)  # (n-1)^2 quads
        # Triangle 1: [idx, idx+1, idx+n], Triangle 2: [idx+1, idx+n+1, idx+n]
        tri1 = torch.stack([idx, idx + 1, idx + n], dim=-1)
        tri2 = torch.stack([idx + 1, idx + n + 1, idx + n], dim=-1)
        cells = torch.cat([tri1, tri2], dim=0).to(torch.int64)
        mesh = Mesh(points=points, cells=cells)

        # Should handle reasonably large mesh
        subdivided = mesh.subdivide(levels=1, filter="linear")
        assert subdivided.n_cells == mesh.n_cells * 4
