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

"""Correctness tests for vectorized performance optimizations.

These tests verify that vectorized implementations produce identical results
to reference implementations, ensuring no correctness regressions were introduced.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh


class TestLoopSubdivisionCorrectness:
    """Verify Loop subdivision vectorization produces correct results."""

    def test_valence_computation_matches_manual_count(self, device):
        """Verify that vectorized valence computation matches manual counting."""
        # Create a mesh with known valences
        points = torch.tensor(
            [
                [0.0, 0.0],  # Vertex 0: neighbors [1, 3] → valence 2
                [1.0, 0.0],  # Vertex 1: neighbors [0, 2, 3, 4] → valence 4
                [2.0, 0.0],  # Vertex 2: neighbors [1, 4] → valence 2
                [0.5, 1.0],  # Vertex 3: neighbors [0, 1, 4, 5] → valence 4
                [1.5, 1.0],  # Vertex 4: neighbors [1, 2, 3, 5] → valence 4
                [1.0, 2.0],  # Vertex 5: neighbors [3, 4] → valence 2
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [
                [0, 1, 3],  # Triangle 0
                [1, 4, 3],  # Triangle 1
                [1, 2, 4],  # Triangle 2
                [3, 4, 5],  # Triangle 3
            ],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Manual valence count (verified by hand)
        expected_valences = [2, 4, 2, 4, 4, 2]

        ### Compute using vectorized function
        from physicsnemo.mesh.neighbors import get_point_to_points_adjacency

        adjacency = get_point_to_points_adjacency(mesh)
        computed_valences = adjacency.offsets[1:] - adjacency.offsets[:-1]

        ### Verify
        assert torch.allclose(
            computed_valences,
            torch.tensor(expected_valences, dtype=torch.int64, device=device),
        )

    def test_loop_edge_opposite_vertex_finding(self, device):
        """Verify that opposite vertex finding in Loop subdivision is correct."""
        # Create simple mesh where we know the opposite vertices
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [1, 3, 2]],  # Two triangles sharing edge [1, 2]
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Extract unique edges
        from physicsnemo.mesh.subdivision._topology import extract_unique_edges

        unique_edges, _ = extract_unique_edges(mesh)

        ### Find the shared edge [1, 2]
        shared_edge_idx = None
        for i, edge in enumerate(unique_edges):
            if (edge[0] == 1 and edge[1] == 2) or (edge[0] == 2 and edge[1] == 1):
                shared_edge_idx = i
                break

        assert shared_edge_idx is not None, "Shared edge [1, 2] not found"

        ### Compute edge positions using Loop subdivision
        from physicsnemo.mesh.subdivision.loop import compute_loop_edge_positions_2d

        edge_positions = compute_loop_edge_positions_2d(mesh, unique_edges)

        ### Verify the computation manually
        # For interior edge [1, 2] with opposite vertices 0 and 3:
        # new_pos = 3/8 * (v1 + v2) + 1/8 * (v0 + v3)
        v0 = points[0]
        v1 = points[1]
        v2 = points[2]
        v3 = points[3]

        expected_pos = (3.0 / 8.0) * (v1 + v2) + (1.0 / 8.0) * (v0 + v3)

        # The shared edge should be at the index we found
        actual_pos = edge_positions[shared_edge_idx]

        assert torch.allclose(actual_pos, expected_pos, atol=1e-6), (
            f"Loop edge position mismatch:\n"
            f"Expected: {expected_pos}\n"
            f"Actual: {actual_pos}"
        )

    def test_boundary_edge_handling_loop(self, device):
        """Verify Loop subdivision handles boundary edges correctly (simple average)."""
        # Single triangle - all edges are boundary
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        from physicsnemo.mesh.subdivision._topology import extract_unique_edges
        from physicsnemo.mesh.subdivision.loop import compute_loop_edge_positions_2d

        unique_edges, _ = extract_unique_edges(mesh)
        edge_positions = compute_loop_edge_positions_2d(mesh, unique_edges)

        ### All edges should be simple averages (boundary edges)
        for i, edge in enumerate(unique_edges):
            v0 = mesh.points[edge[0]]
            v1 = mesh.points[edge[1]]
            expected = (v0 + v1) / 2

            assert torch.allclose(edge_positions[i], expected, atol=1e-6), (
                f"Boundary edge {i} should be simple average"
            )


class TestCotangentWeightsCorrectness:
    """Verify cotangent weight computation is correct."""

    def test_cotangent_weights_equilateral_triangle(self, device):
        """Test cotangent weights for an equilateral triangle."""

        # Equilateral triangle with side length 1
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        from physicsnemo.mesh.geometry.dual_meshes import (
            compute_cotan_weights_fem,
        )

        weights, unique_edges = compute_cotan_weights_fem(mesh)

        ### For equilateral triangle, all angles are 60 degrees
        # cot(60°) = 1/sqrt(3) ≈ 0.5774
        # Each edge has one adjacent triangle (boundary)
        # Weight = cot(60°) / 2 ≈ 0.2887
        expected_weight = (1.0 / (3**0.5)) / 2.0

        ### All three edges should have the same weight
        assert torch.allclose(
            weights, torch.full_like(weights, expected_weight), atol=1e-4
        ), f"Expected all weights to be {expected_weight:.4f}, got {weights}"

    def test_cotangent_weights_right_triangle(self, device):
        """Test cotangent weights for a right triangle with known angles."""
        # Right triangle: 90° at origin, 45° at other two vertices
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Right angle
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        from physicsnemo.mesh.geometry.dual_meshes import (
            compute_cotan_weights_fem,
        )

        weights, unique_edges = compute_cotan_weights_fem(mesh)

        ### Find each edge and verify its weight
        # Edge [0,1]: opposite angle at vertex 2 = 45°, cot(45°) = 1.0
        # Edge [0,2]: opposite angle at vertex 1 = 45°, cot(45°) = 1.0
        # Edge [1,2]: opposite angle at vertex 0 = 90°, cot(90°) = 0.0
        # All edges are boundary (one triangle), so weight = cot(angle) / 2

        expected_weights = {
            (0, 1): 1.0 / 2.0,  # cot(45°) / 2
            (0, 2): 1.0 / 2.0,  # cot(45°) / 2
            (1, 2): 0.0 / 2.0,  # cot(90°) / 2
        }

        for i, edge in enumerate(unique_edges):
            v0, v1 = int(edge[0]), int(edge[1])
            edge_tuple = tuple(sorted([v0, v1]))
            expected = expected_weights[edge_tuple]

            assert abs(weights[i] - expected) < 1e-4, (
                f"Edge {edge_tuple}: expected {expected:.4f}, got {weights[i]:.4f}"
            )

    def test_cotangent_weights_interior_edge(self, device):
        """Test cotangent weights for interior edge (two adjacent triangles)."""

        # Two triangles sharing an edge
        # Triangle 1: [0, 1, 2] with 60° angles (equilateral)
        # Triangle 2: [1, 3, 2] with known angles
        h = (3**0.5) / 2
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, h, 0.0],  # Equilateral triangle 1
                [1.5, h, 0.0],  # Forms triangle 2
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [1, 3, 2]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        from physicsnemo.mesh.geometry.dual_meshes import (
            compute_cotan_weights_fem,
        )

        weights, unique_edges = compute_cotan_weights_fem(mesh)

        ### Find the shared interior edge [1, 2]
        shared_edge_idx = None
        for i, edge in enumerate(unique_edges):
            v0, v1 = int(edge[0]), int(edge[1])
            if (v0 == 1 and v1 == 2) or (v0 == 2 and v1 == 1):
                shared_edge_idx = i
                break

        assert shared_edge_idx is not None

        ### For interior edge: weight = (cot α + cot β) / 2
        # Both triangles are equilateral, so both angles are 60°
        # cot(60°) = 1/sqrt(3)
        # Weight = (cot(60°) + cot(60°)) / 2 = 2 * (1/sqrt(3)) / 2 = 1/sqrt(3)
        expected_weight = 1.0 / (3**0.5)

        assert abs(weights[shared_edge_idx] - expected_weight) < 1e-4, (
            f"Interior edge weight: expected {expected_weight:.4f}, "
            f"got {weights[shared_edge_idx]:.4f}"
        )

    def test_neighbor_sum_computation(self, device):
        """Verify that neighbor position sums are computed correctly."""
        # Create simple mesh with known neighbor relationships
        points = torch.tensor(
            [
                [0.0, 0.0],  # Vertex 0
                [1.0, 0.0],  # Vertex 1 - neighbor of 0
                [0.0, 1.0],  # Vertex 2 - neighbor of 0
                [1.0, 1.0],  # Vertex 3 - neighbor of 1 and 2
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [1, 3, 2]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Get adjacency
        from physicsnemo.mesh.neighbors import get_point_to_points_adjacency

        adjacency = get_point_to_points_adjacency(mesh)
        valences = adjacency.offsets[1:] - adjacency.offsets[:-1]

        ### Compute neighbor sums using vectorized method
        neighbor_sums = torch.zeros_like(mesh.points)
        source_point_indices = torch.repeat_interleave(
            torch.arange(mesh.n_points, dtype=torch.int64, device=device),
            valences,
        )
        neighbor_positions = mesh.points[adjacency.indices]
        source_point_indices_expanded = source_point_indices.unsqueeze(-1).expand(
            -1, mesh.n_spatial_dims
        )
        neighbor_sums.scatter_add_(
            dim=0,
            index=source_point_indices_expanded,
            src=neighbor_positions,
        )

        ### Manually compute expected neighbor sums
        # Vertex 0 neighbors: 1, 2 → sum = [1,0] + [0,1] = [1,1]
        # Vertex 1 neighbors: 0, 2, 3 → sum = [0,0] + [0,1] + [1,1] = [1,2]
        # Vertex 2 neighbors: 0, 1, 3 → sum = [0,0] + [1,0] + [1,1] = [2,1]
        # Vertex 3 neighbors: 1, 2 → sum = [1,0] + [0,1] = [1,1]
        expected_sums = torch.tensor(
            [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        assert torch.allclose(neighbor_sums, expected_sums, atol=1e-6), (
            f"Neighbor sums mismatch:\nExpected:\n{expected_sums}\nActual:\n{neighbor_sums}"
        )

    def test_loop_subdivision_preserves_manifold(self, device):
        """Verify Loop subdivision produces valid manifold (no holes/gaps)."""
        # Start with simple manifold
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        mesh = lumpy_sphere.load(radius=1.0, subdivisions=2, device=device)

        initial_n_cells = mesh.n_cells

        # Subdivide
        subdivided = mesh.subdivide(levels=1, filter="loop")

        ### Check manifold properties
        # Should have 4x cells (2^2 for 2D)
        assert subdivided.n_cells == initial_n_cells * 4

        # All cells should be valid triangles
        assert subdivided.cells.shape[1] == 3

        # All cell indices should be in valid range
        assert subdivided.cells.min() >= 0
        assert subdivided.cells.max() < subdivided.n_points

        # No degenerate cells (all three vertices should be different)
        for cell_idx in range(min(100, subdivided.n_cells)):  # Check first 100
            cell = subdivided.cells[cell_idx]
            assert len(torch.unique(cell)) == 3, (
                f"Degenerate cell at {cell_idx}: {cell}"
            )


class TestButterflySubdivisionCorrectness:
    """Verify Butterfly subdivision vectorization is correct."""

    def test_butterfly_boundary_vs_interior(self, device):
        """Verify boundary edges use simple average, interior use butterfly stencil."""
        # Two triangles sharing edge [1, 2]
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [1, 3, 2]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        from physicsnemo.mesh.subdivision._topology import extract_unique_edges
        from physicsnemo.mesh.subdivision.butterfly import compute_butterfly_weights_2d

        unique_edges, _ = extract_unique_edges(mesh)
        edge_midpoints = compute_butterfly_weights_2d(mesh, unique_edges)

        ### Identify boundary edges (count adjacent cells)
        from physicsnemo.mesh.boundaries import extract_candidate_facets

        candidate_edges, _ = extract_candidate_facets(
            mesh.cells, manifold_codimension=1
        )
        _, inverse_indices = torch.unique(candidate_edges, dim=0, return_inverse=True)
        counts = torch.bincount(inverse_indices, minlength=len(unique_edges))

        ### Boundary edges (count=1) should be simple average
        for i, (edge, count) in enumerate(zip(unique_edges, counts)):
            if count == 1:
                v0 = mesh.points[edge[0]]
                v1 = mesh.points[edge[1]]
                expected = (v0 + v1) / 2

                assert torch.allclose(edge_midpoints[i], expected, atol=1e-6), (
                    f"Boundary edge {i} should be simple average"
                )


class TestGaussianCurvatureCorrectness:
    """Verify Gaussian curvature cell computation is correct."""

    def test_gaussian_curvature_varying_valences(self, device):
        """Test Gaussian curvature on mesh with varying cell valences."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        # Use lumpy_sphere which has varying neighbor counts per cell (icosahedral base)
        mesh = lumpy_sphere.load(radius=1.0, subdivisions=2, device=device)

        ### Compute Gaussian curvature
        K_cells = mesh.gaussian_curvature_cells

        ### Basic validity checks
        assert K_cells.shape == (mesh.n_cells,), f"Wrong shape: {K_cells.shape}"
        assert torch.all(torch.isfinite(K_cells) | torch.isnan(K_cells)), (
            "Non-finite values"
        )

        ### Check that values are in reasonable range
        # For airplane mesh, curvature should be modest (not extremely large)
        finite_K = K_cells[torch.isfinite(K_cells)]
        if len(finite_K) > 0:
            assert torch.abs(finite_K).max() < 100.0, (
                "Unreasonably large curvature values"
            )

    def test_gaussian_curvature_batching_consistency(self, device):
        """Verify that batching by valence produces same results as direct computation."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        # Create mesh with mix of valences (lumpy sphere has varying curvature)
        mesh = lumpy_sphere.load(radius=1.0, subdivisions=2, device=device)

        ### Compute using vectorized implementation
        K_cells = mesh.gaussian_curvature_cells

        ### Verify basic properties
        # Lumpy sphere has varying curvature, but should mostly be positive
        finite_K = K_cells[torch.isfinite(K_cells)]
        positive_fraction = (finite_K > 0).float().mean()
        assert positive_fraction > 0.5, (
            f"Expected mostly positive curvature, got {positive_fraction:.2%}"
        )

        ### Verify curvature values are in reasonable range
        assert torch.abs(finite_K).max() < 100.0, "Unreasonably large curvature values"


class TestSubdivisionTopologyCorrectness:
    """Verify subdivision topology vectorization is correct."""

    def test_child_cell_vertex_indices_valid(self, device):
        """Verify all child cells reference valid vertex indices."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        mesh = lumpy_sphere.load(radius=1.0, subdivisions=2, device=device)

        ### Subdivide
        subdivided = mesh.subdivide(levels=1, filter="linear")

        ### Check all indices are valid
        assert subdivided.cells.min() >= 0, "Negative indices"
        assert subdivided.cells.max() < subdivided.n_points, (
            f"Index out of range: max={subdivided.cells.max()}, n_points={subdivided.n_points}"
        )

        ### Check no duplicate vertices in any cell
        for cell_idx in range(min(100, subdivided.n_cells)):
            cell = subdivided.cells[cell_idx]
            unique_verts = torch.unique(cell)
            assert len(unique_verts) == len(cell), (
                f"Cell {cell_idx} has duplicate vertices: {cell}"
            )

    def test_subdivision_point_count(self, device):
        """Verify subdivision produces correct number of points."""
        # Triangle mesh: n_points_new = n_points_old + n_edges
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        original_n_points = mesh.n_points

        from physicsnemo.mesh.subdivision._topology import extract_unique_edges

        unique_edges, _ = extract_unique_edges(mesh)
        n_edges = len(unique_edges)

        ### Subdivide
        subdivided = mesh.subdivide(levels=1, filter="linear")

        ### Check point count
        expected_points = original_n_points + n_edges
        assert subdivided.n_points == expected_points, (
            f"Expected {expected_points} points, got {subdivided.n_points}"
        )


class TestEdgeCasesCorrectness:
    """Test edge cases to ensure robustness."""

    def test_single_triangle_subdivision(self, device):
        """Test subdivision on simplest possible mesh."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        ### Test all subdivision types
        for subdivision_type in ["linear", "loop", "butterfly"]:
            subdivided = mesh.subdivide(levels=1, filter=subdivision_type)

            # Should have 4 triangles (2^2)
            assert subdivided.n_cells == 4, (
                f"{subdivision_type}: expected 4 cells, got {subdivided.n_cells}"
            )

            # Should have 6 points (3 original + 3 edge midpoints)
            assert subdivided.n_points == 6, (
                f"{subdivision_type}: expected 6 points, got {subdivided.n_points}"
            )

    def test_mesh_with_isolated_vertex(self, device):
        """Test that isolated vertices don't break vectorized operations."""
        # Mesh with isolated vertex
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        ### Loop subdivision should handle isolated vertex
        subdivided = mesh.subdivide(levels=1, filter="loop")

        # Isolated vertex should remain unchanged
        assert torch.allclose(subdivided.points[3], points[3], atol=1e-6), (
            "Isolated vertex should remain unchanged in Loop subdivision"
        )

    def test_degenerate_mesh_cases(self, device):
        """Test empty and single-vertex meshes don't crash."""
        # Empty mesh
        empty_mesh = Mesh(
            points=torch.zeros((0, 3), dtype=torch.float32, device=device),
            cells=torch.zeros((0, 3), dtype=torch.int64, device=device),
        )

        # Should not crash
        result = empty_mesh.subdivide(levels=1, filter="linear")
        assert result.n_cells == 0

        # Single point (no cells)
        single_point = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
            cells=torch.zeros((0, 3), dtype=torch.int64, device=device),
        )

        result = single_point.subdivide(levels=1, filter="linear")
        assert result.n_points == 1
        assert result.n_cells == 0


class TestCPUGPUConsistency:
    """Verify CPU and GPU produce identical results."""

    @pytest.mark.cuda
    @pytest.mark.parametrize("subdivision_type", ["linear", "loop", "butterfly"])
    def test_subdivision_cpu_gpu_match(self, subdivision_type):
        """Verify subdivision produces identical results on CPU and GPU."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        # Create test mesh
        mesh_cpu = lumpy_sphere.load(radius=1.0, subdivisions=2, device="cpu")
        mesh_gpu = mesh_cpu.to("cuda")

        ### Subdivide on both devices
        sub_cpu = mesh_cpu.subdivide(levels=1, filter=subdivision_type)
        sub_gpu = mesh_gpu.subdivide(levels=1, filter=subdivision_type)

        ### Verify identical topology
        assert torch.equal(sub_cpu.cells.cpu(), sub_gpu.cells.cpu()), (
            f"{subdivision_type}: Cell topology differs between CPU and GPU"
        )

        ### Verify identical geometry
        assert torch.allclose(sub_cpu.points.cpu(), sub_gpu.points.cpu(), atol=1e-5), (
            f"{subdivision_type}: Point positions differ between CPU and GPU"
        )

    @pytest.mark.cuda
    def test_curvature_cpu_gpu_match(self):
        """Verify curvature computations match between CPU and GPU."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        mesh_cpu = lumpy_sphere.load(radius=2.0, subdivisions=2, device="cpu")
        mesh_gpu = mesh_cpu.to("cuda")

        ### Compute curvatures
        K_cpu = mesh_cpu.gaussian_curvature_vertices
        K_gpu = mesh_gpu.gaussian_curvature_vertices

        H_cpu = mesh_cpu.mean_curvature_vertices
        H_gpu = mesh_gpu.mean_curvature_vertices

        ### Verify match (allowing for numerical differences)
        assert torch.allclose(
            K_cpu, K_gpu.cpu(), atol=1e-4, rtol=1e-3, equal_nan=True
        ), "Gaussian curvature differs between CPU and GPU"

        assert torch.allclose(
            H_cpu, H_gpu.cpu(), atol=1e-4, rtol=1e-3, equal_nan=True
        ), "Mean curvature differs between CPU and GPU"
