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

"""Tests for spatial sampling functionality.

Tests validate barycentric coordinate computation and data sampling
across spatial dimensions and compute backends.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.sampling import (
    compute_barycentric_coordinates,
    find_all_containing_cells,
    find_containing_cells,
    sample_data_at_points,
)

### Helper Functions ###


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


class TestBarycentricCoordinates:
    """Tests for barycentric coordinate computation."""

    def test_barycentric_coords_2d_triangle(self):
        """Test barycentric coordinates for a 2D triangle."""
        ### Triangle with vertices at (0,0), (1,0), (0,1)
        vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

        ### Query point at centroid (1/3, 1/3)
        query = torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        ### All barycentric coordinates should be approximately 1/3
        assert bary.shape == (1, 1, 3)
        assert torch.allclose(
            bary, torch.tensor([[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]]), atol=1e-6
        )
        ### For codimension-0 (2D in 2D), reconstruction error should be 0
        assert recon_error.shape == (1, 1)
        assert torch.allclose(recon_error, torch.tensor([[0.0]]), atol=1e-6)

    def test_barycentric_coords_at_vertex(self):
        """Test barycentric coordinates when query point is at a vertex."""
        ### Triangle with vertices at (0,0), (1,0), (0,1)
        vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

        ### Query point at first vertex
        query = torch.tensor([[0.0, 0.0]])

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        ### Should be (1, 0, 0)
        assert torch.allclose(bary, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-6)
        assert torch.allclose(recon_error, torch.tensor([[0.0]]), atol=1e-6)

    def test_barycentric_coords_outside(self):
        """Test barycentric coordinates for point outside simplex."""
        ### Triangle with vertices at (0,0), (1,0), (0,1)
        vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

        ### Query point outside the triangle
        query = torch.tensor([[2.0, 2.0]])

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        ### At least one coordinate should be negative
        assert (bary < 0).any()
        ### Reconstruction error should still be 0 for codimension-0
        assert torch.allclose(recon_error, torch.tensor([[0.0]]), atol=1e-6)

    def test_barycentric_coords_3d_tetrahedron(self):
        """Test barycentric coordinates for a 3D tetrahedron."""
        ### Regular tetrahedron vertices
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        )

        ### Query point at centroid
        query = torch.tensor([[0.25, 0.25, 0.25]])

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        ### All barycentric coordinates should be 0.25
        assert bary.shape == (1, 1, 4)
        assert torch.allclose(
            bary, torch.tensor([[[0.25, 0.25, 0.25, 0.25]]]), atol=1e-6
        )
        ### Reconstruction error should be 0 for codimension-0
        assert torch.allclose(recon_error, torch.tensor([[0.0]]), atol=1e-6)

    def test_barycentric_coords_batch(self):
        """Test batched barycentric coordinate computation."""
        ### Two triangles
        vertices = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            ]
        )

        ### Two query points
        queries = torch.tensor([[0.5, 0.5], [1.0, 1.0]])

        bary, recon_error = compute_barycentric_coordinates(queries, vertices)

        ### Should have shape (2 queries, 2 cells, 3 vertices)
        assert bary.shape == (2, 2, 3)
        assert recon_error.shape == (2, 2)


class TestFindContainingCells:
    """Tests for finding containing cells."""

    def test_point_inside_single_triangle(self):
        """Test finding cell for point inside a single triangle."""
        ### Create a simple triangle mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Query point inside the triangle
        query = torch.tensor([[0.25, 0.25]])

        cell_indices, bary = find_containing_cells(mesh, query)

        ### Should find cell 0
        assert cell_indices[0] == 0
        ### Barycentric coords should all be positive and sum to 1
        assert (bary[0] >= 0).all()
        assert torch.allclose(bary[0].sum(), torch.tensor(1.0))

    def test_point_outside_mesh(self):
        """Test that point outside mesh returns -1."""
        ### Create a simple triangle mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Query point outside the triangle
        query = torch.tensor([[2.0, 2.0]])

        cell_indices, bary = find_containing_cells(mesh, query)

        ### Should return -1
        assert cell_indices[0] == -1
        ### Barycentric coords should be NaN
        assert torch.isnan(bary[0]).all()

    def test_multiple_query_points(self):
        """Test finding cells for multiple query points."""
        ### Create a mesh with two triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Query points in both triangles and one outside
        queries = torch.tensor(
            [
                [0.25, 0.25],  # In first triangle
                [0.75, 0.75],  # In second triangle
                [2.0, 2.0],  # Outside
            ]
        )

        cell_indices, bary = find_containing_cells(mesh, queries)

        ### Check results
        assert cell_indices[0] == 0
        assert cell_indices[1] == 1
        assert cell_indices[2] == -1


class TestFindAllContainingCells:
    """Tests for finding all containing cells."""

    def test_overlapping_cells(self):
        """Test finding multiple cells that contain a point."""
        ### Create overlapping triangles (degenerate case for testing)
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Query point that might be in multiple cells
        queries = torch.tensor([[0.1, 0.1]])

        containing = find_all_containing_cells(mesh, queries)

        ### Should find at least one cell (use to_list() for list-like access)
        containing_list = containing.to_list()
        assert len(containing_list[0]) >= 1


class TestSampleAtPoints:
    """Tests for sampling data at query points."""

    def test_sample_cell_data(self):
        """Test sampling cell data at query points."""
        ### Create a simple triangle mesh with cell data
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0, 200.0])},
        )

        ### Query points in each triangle
        queries = torch.tensor(
            [
                [0.25, 0.25],  # In first triangle
                [0.75, 0.75],  # In second triangle
            ]
        )

        result = sample_data_at_points(mesh, queries, data_source="cells")

        ### Should get cell data values
        assert torch.allclose(result["temperature"][0], torch.tensor(100.0))
        assert torch.allclose(result["temperature"][1], torch.tensor(200.0))

    def test_sample_point_data_interpolation(self):
        """Test interpolating point data using barycentric coordinates."""
        ### Create a triangle with point data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([0.0, 100.0, 200.0])},
        )

        ### Query point at centroid should get average
        queries = torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

        result = sample_data_at_points(mesh, queries, data_source="points")

        ### Should get average of point values
        expected = (0.0 + 100.0 + 200.0) / 3.0
        assert torch.allclose(result["value"][0], torch.tensor(expected), atol=1e-5)

    def test_sample_point_data_at_vertex(self):
        """Test interpolating point data when query is at a vertex."""
        ### Create a triangle with point data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([0.0, 100.0, 200.0])},
        )

        ### Query point at second vertex
        queries = torch.tensor([[1.0, 0.0]])

        result = sample_data_at_points(mesh, queries, data_source="points")

        ### Should get exact value at that vertex
        assert torch.allclose(result["value"][0], torch.tensor(100.0))

    def test_sample_outside_returns_nan(self):
        """Test that sampling outside mesh returns NaN."""
        ### Create a simple triangle
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0])},
        )

        ### Query point outside
        queries = torch.tensor([[2.0, 2.0]])

        result = sample_data_at_points(mesh, queries, data_source="cells")

        ### Should be NaN
        assert torch.isnan(result["temperature"][0])

    def test_sample_multidimensional_data(self):
        """Test sampling multi-dimensional data arrays."""
        ### Create a triangle with vector point data
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"velocity": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])},
        )

        ### Query at centroid
        queries = torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

        result = sample_data_at_points(mesh, queries, data_source="points")

        ### Should get averaged vector
        expected = torch.tensor([1.0 / 3.0, 1.0 / 3.0])
        assert torch.allclose(result["velocity"][0], expected, atol=1e-5)

    def test_multiple_cells_strategy_mean(self):
        """Test mean strategy when point is in multiple cells."""
        ### Create two overlapping triangles sharing an edge
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, -1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0, 200.0])},
        )

        ### Query point on shared edge (might be in both cells due to tolerance)
        queries = torch.tensor([[0.5, 0.0]])

        result = sample_data_at_points(
            mesh,
            queries,
            data_source="cells",
            multiple_cells_strategy="mean",
        )

        ### Should get a value (might be average if both cells contain it)
        assert not torch.isnan(result["temperature"][0])

    def test_skip_cached_properties(self):
        """Test that cached properties stored in _cache are skipped."""
        ### Create a mesh and trigger cached property computation
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Access a cached property to populate it
        _ = mesh.cell_centroids  # This populates mesh._cache["cell", "centroids"]

        ### Query point
        queries = torch.tensor([[0.25, 0.25]])

        ### Sample should not include cached properties
        result = sample_data_at_points(mesh, queries, data_source="cells")

        ### Result should not contain cached data
        assert "_cache" not in result.keys()

    def test_3d_tetrahedral_mesh(self):
        """Test sampling on a 3D tetrahedral mesh."""
        ### Create a tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"value": torch.tensor([0.0, 1.0, 2.0, 3.0])},
        )

        ### Query at centroid
        queries = torch.tensor([[0.25, 0.25, 0.25]])

        result = sample_data_at_points(mesh, queries, data_source="points")

        ### Should get average
        expected = (0.0 + 1.0 + 2.0 + 3.0) / 4.0
        assert torch.allclose(result["value"][0], torch.tensor(expected), atol=1e-5)


class TestProjectionOntoNearestCell:
    """Tests for projection onto nearest cell."""

    def test_project_onto_nearest_cell_2d(self):
        """Test projection onto nearest cell for 2D mesh."""
        ### Create a simple triangle
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0])},
        )

        ### Query point outside but close to triangle
        queries = torch.tensor([[0.5, 0.6]])  # Outside but close

        ### Sample with projection
        result = sample_data_at_points(
            mesh,
            queries,
            data_source="cells",
            project_onto_nearest_cell=True,
        )

        ### Should get a value (not NaN) because of projection
        assert not torch.isnan(result["temperature"][0])
        assert torch.allclose(result["temperature"][0], torch.tensor(100.0))


### Tests for Codimension != 0 Manifolds ###


class TestCodimensionNonZero:
    """Tests for barycentric coordinates and containment on codimension != 0 manifolds.

    These tests cover the case where the manifold dimension is less than the spatial
    dimension, e.g., 2D triangles embedded in 3D space. The key fix ensures that
    points far from the manifold are not incorrectly reported as "inside" a cell.
    """

    def test_triangle_in_3d_on_plane(self):
        """Test barycentric coordinates for 2D triangle in 3D, query on the plane."""
        ### Triangle in the z=0 plane
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        ### Query point at centroid, on the plane
        query = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 0.0]])

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        ### Barycentric coordinates should be approximately 1/3 each
        assert bary.shape == (1, 1, 3)
        assert torch.allclose(
            bary, torch.tensor([[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]]), atol=1e-6
        )
        ### Reconstruction error should be 0 (point is on the plane)
        assert recon_error.shape == (1, 1)
        assert torch.allclose(recon_error, torch.tensor([[0.0]]), atol=1e-6)

    def test_triangle_in_3d_slightly_off_plane(self):
        """Test barycentric coordinates for 2D triangle in 3D, query slightly off plane."""
        ### Triangle in the z=0 plane
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        ### Query point at centroid but slightly above the plane
        small_offset = 1e-7
        query = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, small_offset]])

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        ### Barycentric coordinates should still be approximately 1/3 each
        assert torch.allclose(
            bary, torch.tensor([[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]]), atol=1e-5
        )
        ### Reconstruction error should be equal to the z-offset
        assert torch.allclose(recon_error, torch.tensor([[small_offset]]), atol=1e-10)

    def test_triangle_in_3d_far_from_plane(self):
        """Test barycentric coordinates for 2D triangle in 3D, query far from plane."""
        ### Triangle in the z=0 plane
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        ### Query point at centroid projection but 1000 units above the plane
        large_offset = 1000.0
        query = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, large_offset]])

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        ### Barycentric coordinates should still be approximately 1/3 each
        # (they represent the projection onto the plane)
        assert torch.allclose(
            bary, torch.tensor([[[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]]), atol=1e-5
        )
        ### Reconstruction error should be large (equal to the z-offset)
        assert torch.allclose(recon_error, torch.tensor([[large_offset]]), atol=1e-3)

    def test_find_containing_cells_triangle_in_3d_rejects_far_points(self):
        """Test that find_containing_cells rejects points far from codim != 0 manifolds.

        This is the key test for the bug fix: points far from the manifold should
        not be reported as "inside" any cell, even if their projection onto the
        manifold would be inside.
        """
        ### Triangle mesh in z=0 plane
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Query point at centroid projection but 1000 units above
        query_far = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1000.0]])

        cell_indices, bary = find_containing_cells(mesh, query_far)

        ### Should NOT find a containing cell (point is too far from the plane)
        assert cell_indices[0] == -1
        assert torch.isnan(bary[0]).all()

    def test_find_containing_cells_triangle_in_3d_accepts_near_points(self):
        """Test that find_containing_cells accepts points very close to the manifold."""
        ### Triangle mesh in z=0 plane
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Query point at centroid, on the plane
        query_on_plane = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 0.0]])

        cell_indices, bary = find_containing_cells(mesh, query_on_plane)

        ### Should find the containing cell
        assert cell_indices[0] == 0
        assert (bary[0] >= 0).all()
        assert torch.allclose(bary[0].sum(), torch.tensor(1.0))

    def test_find_containing_cells_triangle_in_3d_with_tolerance(self):
        """Test that tolerance controls acceptance of slightly off-plane points."""
        ### Triangle mesh in z=0 plane
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        offset = 0.01  # 1cm offset
        query = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, offset]])

        ### With small tolerance, should reject
        cell_indices_small_tol, _ = find_containing_cells(mesh, query, tolerance=1e-6)
        assert cell_indices_small_tol[0] == -1

        ### With larger tolerance, should accept
        cell_indices_large_tol, bary = find_containing_cells(
            mesh,
            query,
            tolerance=0.1,  # 10cm tolerance
        )
        assert cell_indices_large_tol[0] == 0
        assert (bary[0] >= -0.1).all()

    def test_find_all_containing_cells_triangle_in_3d_rejects_far_points(self):
        """Test find_all_containing_cells rejects far points for codim != 0."""
        ### Triangle mesh in z=0 plane
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Query point far above the plane
        query_far = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1000.0]])

        containing = find_all_containing_cells(mesh, query_far)

        ### Should find no containing cells (use to_list() for list-like access)
        containing_list = containing.to_list()
        assert len(containing_list[0]) == 0

    def test_sample_data_triangle_in_3d_rejects_far_points(self):
        """Test that sample_data_at_points returns NaN for far points on codim != 0."""
        ### Triangle mesh in z=0 plane with cell data
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0])},
        )

        ### Query point far above the plane
        query_far = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1000.0]])

        result = sample_data_at_points(mesh, query_far, data_source="cells")

        ### Should be NaN (point is outside the mesh tolerance)
        assert torch.isnan(result["temperature"][0])

    def test_sample_data_triangle_in_3d_accepts_near_points(self):
        """Test that sample_data_at_points works for points on the manifold."""
        ### Triangle mesh in z=0 plane with cell data
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0])},
        )

        ### Query point on the plane inside the triangle
        query_on_plane = torch.tensor([[0.25, 0.25, 0.0]])

        result = sample_data_at_points(mesh, query_on_plane, data_source="cells")

        ### Should get the cell data value
        assert torch.allclose(result["temperature"][0], torch.tensor(100.0))


### Parametrized Tests for Exhaustive Coverage ###


class TestSamplingParametrized:
    """Parametrized tests for sampling across dimensions and backends."""

    @pytest.mark.parametrize("n_spatial_dims", [2, 3])
    def test_barycentric_coords_parametrized(self, n_spatial_dims, device):
        """Test barycentric coordinate computation across dimensions."""
        # Create simple simplex
        n_verts = n_spatial_dims + 1
        vertices = torch.eye(n_verts, n_spatial_dims, device=device)
        vertices = vertices.unsqueeze(0)  # Add batch dimension

        # Query at centroid
        query = torch.ones(1, n_spatial_dims, device=device) / n_verts

        bary, recon_error = compute_barycentric_coordinates(query, vertices)

        # All coords should be approximately 1/n_verts
        expected = torch.ones(1, 1, n_verts, device=device) / n_verts
        assert torch.allclose(bary, expected, atol=1e-5)

        # Verify device
        assert_on_device(bary, device)

        # Reconstruction error should be 0 for codimension-0
        assert torch.allclose(recon_error, torch.zeros(1, 1, device=device), atol=1e-6)

    @pytest.mark.parametrize("n_spatial_dims", [2, 3])
    def test_data_sampling_parametrized(self, n_spatial_dims, device):
        """Test data sampling across dimensions."""
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                device=device,
            )
            cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
            query = torch.tensor([[0.33, 0.33]], device=device)
        else:
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
            )
            cells = torch.tensor([[0, 1, 2, 3]], device=device, dtype=torch.int64)
            query = torch.tensor([[0.25, 0.25, 0.25]], device=device)

        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"value": torch.tensor([100.0], device=device)},
        )

        result = sample_data_at_points(mesh, query, data_source="cells")

        # Verify result
        assert "value" in result
        assert_on_device(result["value"], device)
        assert not torch.isnan(result["value"][0])
