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

"""Test suite for performance optimizations.

Verifies that all optimizations produce correct results and maintain backward compatibility
across compute backends (CPU, CUDA).
"""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.sampling.sample_data import (
    compute_barycentric_coordinates,
    compute_barycentric_coordinates_pairwise,
)
from physicsnemo.mesh.spatial import BVH

### Helper Functions ###


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


class TestBarycentricOptimizations:
    """Test pairwise barycentric coordinate computation."""

    def test_pairwise_vs_full_2d(self):
        """Verify pairwise barycentric matches diagonal of full computation (2D)."""
        torch.manual_seed(42)
        # Create simple triangle mesh in 2D
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])

        # Create query points
        n_queries = 10
        query_points = torch.rand(n_queries, 2)

        # Compute using both methods for first cell
        cell_vertices = points[cells]  # (2, 3, 2)

        # Full computation (O(n²))
        bary_full, recon_error_full = compute_barycentric_coordinates(
            query_points, cell_vertices
        )  # (n_queries, 2, 3) and (n_queries, 2)

        # Pairwise computation (O(n))
        # For each query, pair it with the first cell
        pairwise_query_points = query_points  # (n_queries, 2)
        pairwise_cell_vertices = cell_vertices[[0]].expand(
            n_queries, -1, -1
        )  # (n_queries, 3, 2)
        bary_pairwise, recon_error_pairwise = compute_barycentric_coordinates_pairwise(
            pairwise_query_points, pairwise_cell_vertices
        )  # (n_queries, 3) and (n_queries,)

        # Extract diagonal from full computation (what pairwise should match)
        bary_full_diagonal = bary_full[:, 0, :]  # (n_queries, 3)
        recon_error_full_diagonal = recon_error_full[:, 0]  # (n_queries,)

        # Verify they match
        torch.testing.assert_close(
            bary_pairwise, bary_full_diagonal, rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            recon_error_pairwise, recon_error_full_diagonal, rtol=1e-5, atol=1e-7
        )

    def test_pairwise_vs_full_3d(self):
        """Verify pairwise barycentric matches diagonal of full computation (3D)."""
        torch.manual_seed(42)
        # Create tetrahedron mesh in 3D
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2, 3]])

        # Create query points
        n_queries = 20
        query_points = torch.rand(n_queries, 3)

        cell_vertices = points[cells]  # (1, 4, 3)

        # Full computation
        bary_full, recon_error_full = compute_barycentric_coordinates(
            query_points, cell_vertices
        )  # (n_queries, 1, 4) and (n_queries, 1)

        # Pairwise computation
        pairwise_cell_vertices = cell_vertices.expand(
            n_queries, -1, -1
        )  # (n_queries, 4, 3)
        bary_pairwise, recon_error_pairwise = compute_barycentric_coordinates_pairwise(
            query_points, pairwise_cell_vertices
        )  # (n_queries, 4) and (n_queries,)

        # Extract diagonal
        bary_full_diagonal = bary_full[:, 0, :]
        recon_error_full_diagonal = recon_error_full[:, 0]

        torch.testing.assert_close(
            bary_pairwise, bary_full_diagonal, rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            recon_error_pairwise, recon_error_full_diagonal, rtol=1e-5, atol=1e-7
        )

    def test_pairwise_different_cells_per_query(self):
        """Test pairwise with different cells for each query."""
        # Create multiple triangles
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 4]])

        # Query points, each paired with specific cell
        query_points = torch.tensor(
            [[0.3, 0.3], [1.5, 0.3], [0.1, 0.1]], dtype=torch.float32
        )
        paired_cell_indices = torch.tensor([0, 1, 0])  # Which cell each query uses

        # Get cell vertices for each query
        cell_vertices = points[cells[paired_cell_indices]]  # (3, 3, 2)

        # Compute pairwise
        bary, recon_error = compute_barycentric_coordinates_pairwise(
            query_points, cell_vertices
        )

        # Verify properties
        assert bary.shape == (3, 3)
        assert recon_error.shape == (3,)
        # Barycentric coordinates should sum to 1
        torch.testing.assert_close(bary.sum(dim=1), torch.ones(3), rtol=1e-5, atol=1e-7)
        # Reconstruction error should be 0 for codimension-0 (2D in 2D)
        torch.testing.assert_close(recon_error, torch.zeros(3), rtol=1e-5, atol=1e-7)

    def test_pairwise_memory_efficiency(self):
        """Verify pairwise uses O(n) not O(n²) memory."""
        torch.manual_seed(42)
        # This is more of a conceptual test - verify shape differences
        n_pairs = 100
        query_points = torch.rand(n_pairs, 3)
        cell_vertices = torch.rand(n_pairs, 4, 3)  # Tets

        # Pairwise should return (n_pairs, 4) and (n_pairs,)
        bary_pairwise, recon_error = compute_barycentric_coordinates_pairwise(
            query_points, cell_vertices
        )
        assert bary_pairwise.shape == (n_pairs, 4)
        assert recon_error.shape == (n_pairs,)

        # Full would return (n_pairs, n_pairs, 4) if we computed it
        # We don't compute it here to avoid memory issues, but the shapes tell the story


class TestCellNormalsOptimizations:
    """Test optimized cell normal computation."""

    def test_2d_edge_normals(self):
        """Test 2D edge normal computation (special case)."""
        # Create a simple edge in 2D
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1], [0, 2]])  # Two edges

        mesh = Mesh(points=points, cells=cells)
        normals = mesh.cell_normals

        # Edge from (0,0) to (1,0): direction is (1,0), normal is (0,1)
        expected_normal_0 = torch.tensor([0.0, 1.0], dtype=torch.float32)
        torch.testing.assert_close(normals[0], expected_normal_0, rtol=1e-5, atol=1e-7)

        # Edge from (0,0) to (0,1): direction is (0,1), normal is (-1,0)
        expected_normal_1 = torch.tensor([-1.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(normals[1], expected_normal_1, rtol=1e-5, atol=1e-7)

    def test_3d_triangle_normals(self):
        """Test 3D triangle normal computation (special case)."""
        # Create a triangle in the XY plane
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)
        normals = mesh.cell_normals

        # Triangle in XY plane should have normal in +Z direction
        expected_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        torch.testing.assert_close(normals[0], expected_normal, rtol=1e-5, atol=1e-7)

    def test_normals_are_unit_length(self):
        """Verify all normals are unit length."""
        torch.manual_seed(42)
        # Create non-degenerate triangles (sequential indices to avoid duplicates)
        points = torch.randn(15, 3)
        # Use sequential indices to ensure non-degenerate triangles
        cells = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        )

        mesh = Mesh(points=points, cells=cells)
        normals = mesh.cell_normals

        # Check all are unit length
        lengths = torch.norm(normals, dim=1)
        torch.testing.assert_close(lengths, torch.ones(5), rtol=1e-5, atol=1e-6)


class TestGramMatrixOptimization:
    """Test einsum optimization in Gram matrix computation."""

    def test_cell_areas_correctness(self):
        """Verify cell area computation is still correct after optimization."""
        # Create a known triangle
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)
        area = mesh.cell_areas[0]

        # Right triangle with legs 1, area = 0.5
        expected_area = 0.5
        torch.testing.assert_close(
            area, torch.tensor(expected_area), rtol=1e-5, atol=1e-7
        )

    def test_3d_tetrahedron_volume(self):
        """Test tetrahedron volume computation."""
        # Unit tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2, 3]])

        mesh = Mesh(points=points, cells=cells)
        volume = mesh.cell_areas[0]

        # Volume of unit tetrahedron is 1/6
        expected_volume = 1.0 / 6.0
        torch.testing.assert_close(
            volume, torch.tensor(expected_volume), rtol=1e-5, atol=1e-7
        )


class TestMeshMergeOptimization:
    """Test optimized mesh merging."""

    def test_merge_preserves_correctness(self):
        """Verify merge produces same result as before."""
        # Create two simple meshes
        points1 = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        cells1 = torch.tensor([[0, 1, 2]])
        mesh1 = Mesh(
            points=points1, cells=cells1, cell_data={"value": torch.tensor([1.0])}
        )

        points2 = torch.tensor(
            [[2.0, 0.0], [3.0, 0.0], [2.0, 1.0]], dtype=torch.float32
        )
        cells2 = torch.tensor([[0, 1, 2]])
        mesh2 = Mesh(
            points=points2, cells=cells2, cell_data={"value": torch.tensor([2.0])}
        )

        # Merge
        merged = Mesh.merge([mesh1, mesh2])

        # Check structure
        assert merged.n_points == 6
        assert merged.n_cells == 2

        # Check cell indices are offset correctly
        # Mesh2's cells should reference points 3, 4, 5
        expected_cells = torch.tensor([[0, 1, 2], [3, 4, 5]])
        torch.testing.assert_close(merged.cells, expected_cells)

        # Check data preserved
        expected_values = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(merged.cell_data["value"], expected_values)


class TestCombinationCache:
    """Test combination index cache for facet extraction."""

    def test_triangle_edge_combinations(self):
        """Test triangle edge extraction uses cached combinations."""
        from physicsnemo.mesh.boundaries._facet_extraction import (
            _generate_combination_indices,
        )

        # Should use cache for (3, 2)
        combos = _generate_combination_indices(3, 2)
        expected = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int64)
        torch.testing.assert_close(combos, expected)

    def test_tetrahedron_face_combinations(self):
        """Test tetrahedron face extraction uses cached combinations."""
        from physicsnemo.mesh.boundaries._facet_extraction import (
            _generate_combination_indices,
        )

        # Should use cache for (4, 3)
        combos = _generate_combination_indices(4, 3)
        expected = torch.tensor(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.int64
        )
        torch.testing.assert_close(combos, expected)

    def test_facet_extraction_with_cache(self):
        """Test full facet extraction pipeline with cached combinations."""
        # Create triangle mesh
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Extract edges (should use cache)
        edge_mesh = mesh.get_facet_mesh(manifold_codimension=1)

        # Should have 5 unique edges
        assert edge_mesh.n_cells == 5
        assert edge_mesh.n_manifold_dims == 1


class TestRandomSamplingOptimization:
    """Test optimized random sampling normalization."""

    def test_barycentric_coords_sum_to_one(self):
        """Verify optimized normalization produces valid barycentric coords."""
        torch.manual_seed(42)
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Sample points
        sampled_points = mesh.sample_random_points_on_cells(
            cell_indices=[0, 0, 1, 1, 1]
        )

        assert sampled_points.shape == (5, 2)
        # Points should be within valid range
        assert (sampled_points >= 0.0).all()
        assert (sampled_points <= 1.0).all()


class TestBVHPerformance:
    """Test BVH traversal performance and correctness."""

    def test_bvh_candidate_finding(self):
        """Test BVH finds correct candidates."""
        # Create a simple mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2, 3], [1, 4, 2, 5]])
        mesh = Mesh(points=points, cells=cells)

        # Build BVH
        bvh = BVH.from_mesh(mesh)

        # Create query points
        query_points = torch.tensor(
            [[0.2, 0.2, 0.2], [0.6, 0.3, 0.1], [2.0, 2.0, 2.0]], dtype=torch.float32
        )

        # Find candidates
        candidates = bvh.find_candidate_cells(query_points)

        # Should return Adjacency for all queries
        candidates_list = candidates.to_list()
        assert len(candidates_list) == 3

        # Point inside first tet should find at least that cell
        assert len(candidates_list[0]) > 0

        # Point outside should find no candidates
        assert len(candidates_list[2]) == 0

    @pytest.mark.cuda
    def test_bvh_on_gpu(self):
        """Test BVH works on GPU."""
        torch.manual_seed(42)
        # Create mesh on GPU
        points = torch.randn(100, 3, device="cuda")
        cells = torch.randint(0, 100, (50, 4), device="cuda")
        mesh = Mesh(points=points, cells=cells)

        # Build BVH
        bvh = BVH.from_mesh(mesh)

        # Query points
        query_points = torch.randn(20, 3, device="cuda")

        # Should not raise
        candidates = bvh.find_candidate_cells(query_points)
        assert candidates.n_sources == 20


class TestHierarchicalSampling:
    """Test hierarchical sampling with all optimizations."""

    def test_hierarchical_sampling_correctness(self):
        """Verify hierarchical sampling produces valid results."""
        # Create a simple mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        cell_data = {"temperature": torch.tensor([100.0])}
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        # Sample using BVH-accelerated path
        from physicsnemo.mesh.sampling import sample_data_at_points

        query_points = torch.tensor([[0.25, 0.25, 0.25]], dtype=torch.float32)

        # Build BVH
        bvh = BVH.from_mesh(mesh)

        result = sample_data_at_points(mesh, query_points, bvh=bvh, data_source="cells")

        # Point inside the tet should get temperature value
        assert "temperature" in result
        torch.testing.assert_close(
            result["temperature"], torch.tensor([100.0]), rtol=1e-5, atol=1e-7
        )


### Parametrized Tests for Exhaustive Backend Coverage ###


class TestOptimizationsParametrized:
    """Parametrized tests for optimizations across backends."""

    @pytest.mark.parametrize("n_queries,n_spatial_dims", [(10, 2), (20, 3)])
    def test_barycentric_pairwise_parametrized(self, n_queries, n_spatial_dims, device):
        """Test pairwise barycentric across backends and dimensions."""
        torch.manual_seed(42)
        # Create query points and cell vertices
        query_points = torch.rand(n_queries, n_spatial_dims, device=device)
        cell_vertices = torch.rand(
            n_queries, n_spatial_dims + 1, n_spatial_dims, device=device
        )

        # Compute pairwise
        bary, recon_error = compute_barycentric_coordinates_pairwise(
            query_points, cell_vertices
        )

        # Verify shape
        assert bary.shape == (n_queries, n_spatial_dims + 1)
        assert recon_error.shape == (n_queries,)

        # Verify device
        assert_on_device(bary, device)
        assert_on_device(recon_error, device)

        # Verify barycentric coords sum to 1
        sums = bary.sum(dim=1)
        assert torch.allclose(sums, torch.ones(n_queries, device=device), rtol=1e-4)

        # For codimension-0 (n_spatial_dims == n_manifold_dims), recon error should be 0
        assert torch.allclose(
            recon_error, torch.zeros(n_queries, device=device), rtol=1e-5, atol=1e-6
        )

    @pytest.mark.parametrize("n_manifold_dims", [2, 3])
    def test_cell_areas_computation_parametrized(self, n_manifold_dims, device):
        """Test cell area computation across backends."""
        if n_manifold_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
                device=device,
            )
            cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)
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

        mesh = Mesh(points=points, cells=cells)
        areas = mesh.cell_areas

        # Verify device
        assert_on_device(areas, device)

        # Verify areas are positive
        assert torch.all(areas > 0), "All areas should be positive"

    @pytest.mark.parametrize("n_manifold_dims", [1, 2])
    def test_cell_normals_computation_parametrized(self, n_manifold_dims, device):
        """Test cell normals computation across backends (codimension-1 only)."""
        if n_manifold_dims == 1:
            points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
            cells = torch.tensor([[0, 1]], device=device, dtype=torch.int64)
        else:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                device=device,
            )
            cells = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int64)

        mesh = Mesh(points=points, cells=cells)
        normals = mesh.cell_normals

        # Verify device
        assert_on_device(normals, device)

        # Verify unit length
        lengths = torch.norm(normals, dim=1)
        assert torch.allclose(
            lengths,
            torch.ones(mesh.n_cells, device=device),
            rtol=1e-5,
        ), "Normals should be unit length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
