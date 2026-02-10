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

"""Comprehensive tests for angle computation in all dimensions.

Tests coverage for:
- Total angle sums in watertight manifolds (topological invariants)
- Solid angle computation for 3D tetrahedra
- Multi-edge vertices in 1D manifolds
- Higher-dimensional angle computations
- Edge cases and numerical stability

This module consolidates tests from:
- Angle sum tests (topological invariants for closed curves and surfaces)
- Comprehensive angle tests (solid angles, higher dimensions, edge cases)
"""

import pytest
import torch

from physicsnemo.mesh.curvature._angles import compute_angles_at_vertices
from physicsnemo.mesh.curvature._utils import (
    compute_triangle_angles,
    stable_angle_between_vectors,
)
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

###############################################################################
# 1D Manifolds (Closed Curves)
###############################################################################


class TestClosedCurveAngleSums:
    """Tests for angle sums in closed 1D manifolds (circles)."""

    def test_circle_angle_sum_clean(self, device):
        """Test that clean circle has total angle sum = (n-2)π."""
        n_points = 40
        mesh = circle_2d.load(radius=1.0, n_points=n_points, device=device)

        # Compute angle sum at each vertex
        angle_sums = compute_angles_at_vertices(mesh)

        # Total sum of all angles
        total_angle = angle_sums.sum()

        # For a closed polygon with n vertices, sum of interior angles = (n-2)π
        # This is a topological invariant
        expected_total = (n_points - 2) * torch.pi

        # Should be close
        relative_error = torch.abs(total_angle - expected_total) / expected_total
        assert relative_error < 1e-5  # Essentially exact

    def test_circle_angle_sum_with_noise(self, device):
        """Test that noisy circle maintains topological angle sum = (n-2)π."""
        # Create clean circle
        n_points = 40
        mesh = circle_2d.load(radius=1.0, n_points=n_points, device=device)

        # Add radial noise: r_new = r_old + noise ∈ [0.5, 1.5]
        # This keeps all points outside origin and preserves topology
        torch.manual_seed(42)
        radial_noise = torch.rand(mesh.n_points, device=device) - 0.5  # [-0.5, 0.5]

        # Compute radial distance for each point
        radii = torch.norm(mesh.points, dim=-1)

        # Add noise to radii
        new_radii = radii + radial_noise

        # Update points with new radii (preserve direction)
        directions = mesh.points / radii.unsqueeze(-1)
        noisy_points = directions * new_radii.unsqueeze(-1)

        # Create noisy mesh
        noisy_mesh = Mesh(points=noisy_points, cells=mesh.cells)

        # Compute angles on noisy mesh
        angle_sums_noisy = compute_angles_at_vertices(noisy_mesh)
        total_angle_noisy = angle_sums_noisy.sum()

        # Should still be close to (n-2)π (topological property)
        expected_total = (n_points - 2) * torch.pi
        relative_error = torch.abs(total_angle_noisy - expected_total) / expected_total

        # Noisy perturbation changes geometry significantly for 1D curves
        # Angle sums are not purely topological for curves (depend on embedding)
        # With 1% noise, should still be essentially exact
        assert not torch.isnan(total_angle_noisy)
        assert total_angle_noisy > 0
        assert relative_error < 1e-5, (
            f"Relative error {relative_error:.3f} unexpectedly large for 1% noise"
        )


###############################################################################
# 2D Manifolds (Closed Surfaces)
###############################################################################


class TestClosedSurfaceAngleSums:
    """Tests for angle sums in closed 2D manifolds (spheres)."""

    def test_sphere_angle_sum_clean(self, device):
        """Test that clean sphere has total angle sum = 4π."""
        mesh = sphere_icosahedral.load(radius=1.0, subdivisions=1, device=device)

        # Compute angle sum at each vertex
        angle_sums = compute_angles_at_vertices(mesh)

        # Total sum of all angles at all vertices
        total_angle = angle_sums.sum()

        # For a closed surface (sphere), the total should relate to Euler characteristic
        # By Gauss-Bonnet: Σ(angle_defect) = 2π * χ
        # Σ(full_angle - angle_sum) = 2π * χ
        # N * full_angle - Σ(angle_sum) = 2π * χ
        # Σ(angle_sum) = N * 2π - 2π * χ

        # For sphere: χ = 2
        # Σ(angle_sum) = N * 2π - 2π * 2 = 2π(N - 2)

        n_points = mesh.n_points
        expected_total = 2 * torch.pi * (n_points - 2)

        # Should be close
        relative_error = torch.abs(total_angle - expected_total) / expected_total
        assert relative_error < 1e-5  # Essentially exact

    def test_sphere_angle_sum_with_noise(self, device):
        """Test that noisy sphere maintains topological angle sum."""
        # Create clean sphere
        mesh = sphere_icosahedral.load(radius=1.0, subdivisions=1, device=device)

        # Add radial noise to each vertex
        torch.manual_seed(42)
        radial_noise = torch.rand(mesh.n_points, device=device) - 0.5  # [-0.5, 0.5]

        # Compute radial distance for each point
        radii = torch.norm(mesh.points, dim=-1)

        # Add noise to radii (stays in range [0.5, 1.5])
        new_radii = radii + radial_noise
        new_radii = torch.clamp(new_radii, min=0.1)  # Ensure positive

        # Update points with new radii
        directions = mesh.points / radii.unsqueeze(-1)
        noisy_points = directions * new_radii.unsqueeze(-1)

        # Create noisy mesh (same connectivity)
        noisy_mesh = Mesh(points=noisy_points, cells=mesh.cells)

        # Compute angles on both meshes
        angle_sums_clean = compute_angles_at_vertices(mesh)
        angle_sums_noisy = compute_angles_at_vertices(noisy_mesh)

        total_clean = angle_sums_clean.sum()
        total_noisy = angle_sums_noisy.sum()

        # Topological invariant: should be approximately equal
        # (Some variation due to geometry change, but topology unchanged)
        relative_diff = torch.abs(total_clean - total_noisy) / total_clean

        # Should remain close despite geometric perturbation
        assert relative_diff < 0.1  # Within 10%

    def test_sphere_gauss_bonnet_relation(self, device):
        """Test discrete Gauss-Bonnet theorem holds."""
        mesh = sphere_icosahedral.load(radius=1.0, subdivisions=1, device=device)

        # Compute Gaussian curvature
        K = mesh.gaussian_curvature_vertices

        # Compute Voronoi areas
        from physicsnemo.mesh.geometry.dual_meshes import (
            compute_dual_volumes_0 as compute_voronoi_areas,
        )

        voronoi_areas = compute_voronoi_areas(mesh)

        # Integrate: ∫K dA ≈ Σ K_i * A_i
        total_curvature = (K * voronoi_areas).sum()

        # For sphere: χ = 2, so ∫K dA = 2π * 2 = 4π
        expected = 4 * torch.pi

        relative_error = torch.abs(total_curvature - expected) / expected
        assert relative_error < 0.1  # Within 10%

        # Now test with noise
        torch.manual_seed(42)
        radial_noise = torch.rand(mesh.n_points, device=device) - 0.5
        radii = torch.norm(mesh.points, dim=-1)
        new_radii = torch.clamp(radii + radial_noise, min=0.1)
        directions = mesh.points / radii.unsqueeze(-1)
        noisy_points = directions * new_radii.unsqueeze(-1)

        noisy_mesh = Mesh(points=noisy_points, cells=mesh.cells)

        K_noisy = noisy_mesh.gaussian_curvature_vertices
        voronoi_areas_noisy = compute_voronoi_areas(noisy_mesh)
        total_curvature_noisy = (K_noisy * voronoi_areas_noisy).sum()

        # Should still satisfy Gauss-Bonnet (topological invariant)
        relative_error_noisy = torch.abs(total_curvature_noisy - expected) / expected
        assert relative_error_noisy < 0.15  # Within 15% for noisy case


###############################################################################
# Triangle Angle Sum Property
###############################################################################


class TestTriangleAngleSum:
    """Test that triangle interior angles sum to π."""

    def test_triangle_angles_sum_to_pi(self, device):
        """Test that angles in a triangle sum to π."""
        # Create various triangles
        triangles = [
            # Equilateral
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, (3**0.5) / 2, 0.0]],
                device=device,
            ),
            # Right triangle
            torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
            ),
            # Scalene
            torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 1.5, 0.0]], device=device
            ),
        ]

        for triangle_points in triangles:
            # Compute all three angles
            angle_0 = compute_triangle_angles(
                triangle_points[0].unsqueeze(0),
                triangle_points[1].unsqueeze(0),
                triangle_points[2].unsqueeze(0),
            )[0]

            angle_1 = compute_triangle_angles(
                triangle_points[1].unsqueeze(0),
                triangle_points[2].unsqueeze(0),
                triangle_points[0].unsqueeze(0),
            )[0]

            angle_2 = compute_triangle_angles(
                triangle_points[2].unsqueeze(0),
                triangle_points[0].unsqueeze(0),
                triangle_points[1].unsqueeze(0),
            )[0]

            total = angle_0 + angle_1 + angle_2

            # Should sum to π
            assert torch.abs(total - torch.pi) < 1e-5


###############################################################################
# Angles 3D
###############################################################################


class TestAngles3D:
    """Tests for angle computation in 3D tetrahedral meshes."""

    def test_angles_at_vertices_3d_single_tet(self, device):
        """Test angle computation for single tetrahedron."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
                [0.5, (3**0.5) / 6, ((2 / 3) ** 0.5)],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Compute solid angles at all vertices
        angles = compute_angles_at_vertices(mesh)

        # All four vertices should have the same solid angle (regular tet)
        assert angles.shape == (4,)
        assert torch.all(angles > 0)

        # Verify they're approximately equal
        assert torch.std(angles) < 0.01  # Should be nearly identical

    def test_angles_at_vertices_3d_two_tets(self, device):
        """Test angle computation for two adjacent tetrahedra."""
        # Create two tets sharing a face
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [0.5, 1.0, 0.0],  # 2
                [0.5, 0.5, 1.0],  # 3 (above)
                [0.5, 0.5, -1.0],  # 4 (below)
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2, 3],  # Tet 1
                [0, 1, 2, 4],  # Tet 2 (shares face 0,1,2)
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # Vertices 0, 1, 2 should have sum of two solid angles
        # Vertices 3, 4 should have one solid angle each
        assert angles.shape == (5,)
        assert torch.all(angles > 0)

        # Shared vertices should have larger angles
        assert angles[0] > angles[3]
        assert angles[1] > angles[3]
        assert angles[2] > angles[3]


###############################################################################
# Multi-Edge Vertices 1D
###############################################################################


class TestMultiEdgeVertices1D:
    """Tests for vertices with more than 2 incident edges in 1D manifolds."""

    def test_junction_point_three_edges(self, device):
        """Test vertex where three edges meet (Y-junction)."""
        # Create Y-shaped curve
        points = torch.tensor(
            [
                [0.0, 0.0],  # Center (junction)
                [1.0, 0.0],  # Right
                [-0.5, (3**0.5) / 2],  # Upper left
                [-0.5, -(3**0.5) / 2],  # Lower left
            ],
            dtype=torch.float32,
            device=device,
        )

        # Three edges meeting at vertex 0
        cells = torch.tensor(
            [
                [0, 1],  # To right
                [0, 2],  # To upper left
                [0, 3],  # To lower left
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # Center vertex should have sum of pairwise angles
        # Between the three 120° separated rays: 3 * 120° = 360° = 2π
        assert angles[0] > 0

        # Each end vertex has angle from its single edge
        # (For open curves, this is not well-defined, so we just check it's computed)
        assert not torch.isnan(angles[1])
        assert not torch.isnan(angles[2])
        assert not torch.isnan(angles[3])

    def test_junction_point_four_edges(self, device):
        """Test vertex where four edges meet (cross junction)."""
        # Create cross-shaped curve
        points = torch.tensor(
            [
                [0.0, 0.0],  # Center (junction)
                [1.0, 0.0],  # Right
                [-1.0, 0.0],  # Left
                [0.0, 1.0],  # Up
                [0.0, -1.0],  # Down
            ],
            dtype=torch.float32,
            device=device,
        )

        # Four edges meeting at vertex 0
        cells = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # Center vertex with 4 edges at 90° intervals
        # Sum of pairwise angles should be computed
        assert angles[0] > 0
        assert not torch.isnan(angles[0])


###############################################################################
# Higher Dimensional Angles
###############################################################################


class TestHigherDimensionalAngles:
    """Tests for angle computation in higher dimensions."""

    def test_stable_angle_between_vectors_3d(self, device):
        """Test stable angle computation in 3D."""
        # Perpendicular vectors
        v1 = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        v2 = torch.tensor([[0.0, 1.0, 0.0]], device=device)

        angle = stable_angle_between_vectors(v1, v2)

        assert torch.abs(angle - torch.pi / 2) < 1e-6

    def test_stable_angle_between_vectors_parallel(self, device):
        """Test angle between parallel vectors."""
        v1 = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        v2 = torch.tensor([[2.0, 0.0, 0.0]], device=device)

        angle = stable_angle_between_vectors(v1, v2)

        assert torch.abs(angle) < 1e-6  # Should be 0

    def test_stable_angle_between_vectors_opposite(self, device):
        """Test angle between opposite vectors."""
        v1 = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        v2 = torch.tensor([[-1.0, 0.0, 0.0]], device=device)

        angle = stable_angle_between_vectors(v1, v2)

        assert torch.abs(angle - torch.pi) < 1e-6

    def test_stable_angle_4d(self, device):
        """Test angle computation in 4D space."""
        # Two 4D vectors
        v1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        v2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

        angle = stable_angle_between_vectors(v1, v2)

        assert torch.abs(angle - torch.pi / 2) < 1e-6

    def test_edges_in_higher_dim_space(self, device):
        """Test 1D manifold (edges) embedded in higher dimensional space."""
        # Create bent polyline in 4D space (not straight)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],  # Bent at 90 degrees
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # Middle vertex should have angle π/2 (90 degree bend)
        # Note: In higher dimensions, the angle computation uses stable_angle_between_vectors
        # Interior angle = π - exterior angle
        assert angles[1] > 0  # Should be computed

        # For a 90° bend, interior angle should be π/2
        assert torch.abs(angles[1] - torch.pi / 2) < 0.1


###############################################################################
# Angle Edge Cases
###############################################################################


class TestAngleEdgeCases:
    """Tests for edge cases in angle computation."""

    def test_empty_mesh(self, device):
        """Test angle computation on empty mesh."""
        points = torch.zeros((5, 3), device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # All angles should be zero (no incident cells)
        assert torch.allclose(angles, torch.zeros(5, device=device))

    def test_isolated_vertex(self, device):
        """Test that isolated vertices have zero angle."""
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

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # First three vertices have angles from triangle
        assert angles[0] > 0
        assert angles[1] > 0
        assert angles[2] > 0

        # Isolated vertex should have zero angle
        assert angles[3] == 0

    def test_single_edge_open_curve(self, device):
        """Test angle computation for single open edge."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # Each endpoint has only one incident edge
        # Angle is not well-defined for single edge, but should be computed
        assert angles.shape == (2,)
        # Both should be zero (no angle to measure)
        assert angles[0] == 0
        assert angles[1] == 0

    def test_nearly_degenerate_triangle(self, device):
        """Test angle computation for nearly degenerate triangle."""
        # Very flat triangle
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1e-6, 0.0],  # Nearly collinear
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # Should not produce NaN
        assert not torch.any(torch.isnan(angles))

        # Two vertices should have angles close to π/2 (nearly 90°)
        # One vertex should have angle close to 0 (nearly 0°)
        # Sum should still be close to π
        total = angles.sum()
        assert torch.abs(total - torch.pi) < 1e-3

    def test_2d_manifold_in_higher_dim(self, device):
        """Test triangle mesh embedded in higher dimensional space."""
        # Triangle in 4D space
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        angles = compute_angles_at_vertices(mesh)

        # Should compute angles correctly (equilateral triangle)
        # Each angle should be π/3
        expected = torch.pi / 3
        assert torch.allclose(
            angles, torch.full((3,), expected, device=device), atol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
