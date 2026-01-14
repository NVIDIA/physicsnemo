"""Tests for total angle sums in watertight manifolds.

Verifies fundamental topological properties: the sum of all angles at all
vertices should equal a constant determined by the mesh topology, regardless
of geometric perturbations (as long as the mesh remains valid).
"""

import pytest
import torch

from physicsnemo.mesh.curvature._angles import compute_angles_at_vertices
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

### Test 1D Manifolds (Closed Curves)


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


### Test 2D Manifolds (Closed Surfaces)


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


### Test Triangle Angle Sum Property


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

        from physicsnemo.mesh.curvature._utils import compute_triangle_angles

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
