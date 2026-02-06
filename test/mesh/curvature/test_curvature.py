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

"""Comprehensive tests for curvature computations.

Tests Gaussian and mean curvature on analytical test cases including
spheres, planes, cylinders, and tori. Validates convergence with subdivision.
"""

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import icosahedron_surface
from physicsnemo.mesh.utilities._cache import get_cached

### Mesh Generators


def create_sphere_mesh(radius=1.0, subdivisions=0, device="cpu"):
    """Create a triangulated sphere using icosahedron Loop subdivision.

    Args:
        radius: Sphere radius
        subdivisions: Number of Loop subdivision levels (0 = icosahedron)
        device: Device to create mesh on

    Returns:
        Mesh representing a sphere of given radius
    """
    mesh = icosahedron_surface.load(radius=1.0, device=device)
    mesh = mesh.subdivide(subdivisions, "loop")

    # Project to perfect sphere
    mesh.points = F.normalize(mesh.points, dim=-1) * radius

    return mesh


def create_plane_mesh(size=2.0, n_subdivisions=2, device="cpu"):
    """Create a flat triangulated plane."""
    n = 2**n_subdivisions + 1

    # Create grid of points
    x = torch.linspace(-size / 2, size / 2, n, device=device)
    y = torch.linspace(-size / 2, size / 2, n, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    points = torch.stack(
        [xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())], dim=1
    )

    # Create triangular cells
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            # Two triangles per quad
            cells.append([idx, idx + 1, idx + n])
            cells.append([idx + 1, idx + n + 1, idx + n])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)


def create_line_curve_2d(n_points=10, curvature=1.0, device="cpu"):
    """Create a 1D circular arc in 2D (for testing 1D curvature)."""
    # Circle of given curvature (κ = 1/r)
    radius = 1.0 / curvature
    theta = torch.linspace(0, torch.pi / 2, n_points, device=device)

    points = torch.stack(
        [
            radius * torch.cos(theta),
            radius * torch.sin(theta),
        ],
        dim=1,
    )

    # Create edge cells
    cells = torch.stack(
        [
            torch.arange(n_points - 1, device=device),
            torch.arange(1, n_points, device=device),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)


### Test Gaussian Curvature


class TestGaussianCurvature:
    """Tests for Gaussian curvature computation."""

    def test_sphere_gaussian_curvature(self, device):
        """Test that sphere has constant positive Gaussian curvature K = 1/r²."""
        radius = 2.0
        mesh = create_sphere_mesh(radius=radius, subdivisions=2, device=device)

        K_vertices = mesh.gaussian_curvature_vertices

        # Expected: K = 1/r² for all vertices
        expected_K = 1.0 / (radius**2)

        # With subdivision level 2, Loop subdivision gives excellent accuracy
        mean_K = K_vertices.mean()
        assert torch.abs(mean_K - expected_K) / expected_K < 0.02  # Within 2%

        # All should be positive
        assert torch.all(K_vertices > 0)

    def test_plane_gaussian_curvature(self, device):
        """Test that flat plane has zero Gaussian curvature at interior vertices."""
        mesh = create_plane_mesh(n_subdivisions=2, device=device)

        K_vertices = mesh.gaussian_curvature_vertices

        # Interior vertices should have zero curvature
        # For a 5x5 grid (n_subdivisions=2), interior vertices are those not on boundary
        # Grid size: 2^2 + 1 = 5
        n = 5

        # Find interior vertices (not on edges of grid)
        interior_mask = torch.zeros(mesh.n_points, dtype=torch.bool, device=device)
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                if 0 < i < n - 1 and 0 < j < n - 1:
                    interior_mask[idx] = True

        # Check interior vertices have zero curvature
        interior_K = K_vertices[interior_mask]
        assert torch.allclose(interior_K, torch.zeros_like(interior_K), atol=1e-5)

    def test_gaussian_curvature_convergence(self, device):
        """Test that Gaussian curvature converges with subdivision."""
        radius = 1.0
        expected_K = 1.0 / (radius**2)

        errors = []
        for subdivisions in [0, 1, 2]:
            mesh = create_sphere_mesh(
                radius=radius, subdivisions=subdivisions, device=device
            )
            K_vertices = mesh.gaussian_curvature_vertices
            mean_K = K_vertices.mean()
            error = torch.abs(mean_K - expected_K)
            errors.append(error.item())

        # Error should decrease with subdivision
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_gauss_bonnet_theorem(self, device):
        """Test discrete Gauss-Bonnet theorem: ∫K dA = 2πχ."""
        mesh = create_sphere_mesh(radius=1.0, subdivisions=1, device=device)

        K_vertices = mesh.gaussian_curvature_vertices

        # Compute Voronoi areas for integration
        from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0

        voronoi_areas = compute_dual_volumes_0(mesh)

        # Integrate: ∫K dA ≈ Σ K_i * A_i
        total_curvature = (K_vertices * voronoi_areas).sum()

        # For a sphere: χ = 2, so ∫K dA = 4π
        expected = 4 * torch.pi

        # Should be close (within a few percent for subdivision level 1)
        relative_error = torch.abs(total_curvature - expected) / expected
        assert relative_error < 0.1  # Within 10%

    def test_gaussian_curvature_cells(self, device):
        """Test cell-based Gaussian curvature (dual mesh)."""
        mesh = create_sphere_mesh(radius=1.0, subdivisions=1, device=device)

        K_cells = mesh.gaussian_curvature_cells

        # Should have curvature for all cells
        assert K_cells.shape == (mesh.n_cells,)

        # Should be positive for sphere
        assert torch.all(K_cells > 0)

    def test_pentagonal_vertex_convergence(self, device):
        """Test that pentagonal vertices converge correctly on icosphere.

        The icosahedron has 12 pentagonal vertices (valence 5) which remain
        pentagonal under Loop subdivision. With proper Voronoi areas, these
        should converge to the same curvature as hexagonal vertices (valence 6).

        This test verifies the fix for the systematic error at irregular vertices.
        """
        radius = 1.0
        expected_K = 1.0 / (radius**2)

        # Test at high subdivision level
        mesh = create_sphere_mesh(radius=radius, subdivisions=5, device=device)
        K_vertices = mesh.gaussian_curvature_vertices

        # Identify pentagonal vs hexagonal vertices by valence
        from physicsnemo.mesh.neighbors import get_point_to_cells_adjacency

        adjacency = get_point_to_cells_adjacency(mesh)
        valences = adjacency.offsets[1:] - adjacency.offsets[:-1]

        pentagonal_mask = valences == 5
        hexagonal_mask = valences == 6

        # Check that both types converge to K=1.0
        K_pent = K_vertices[pentagonal_mask]
        assert len(K_pent) == 12, "Icosphere should have exactly 12 pentagonal vertices"
        pent_error = torch.abs(K_pent.mean() - expected_K).item()
        assert pent_error < 0.02, f"Pentagonal vertex error too large: {pent_error:.6f}"

        K_hex = K_vertices[hexagonal_mask]
        hex_error = torch.abs(K_hex.mean() - expected_K).item()
        assert hex_error < 0.02, f"Hexagonal vertex error too large: {hex_error:.6f}"

        # Pentagonal and hexagonal vertices should have similar curvature
        pent_hex_diff = torch.abs(K_pent.mean() - K_hex.mean()).item()
        assert pent_hex_diff < 0.01, (
            f"Pentagonal and hexagonal vertices differ too much: {pent_hex_diff:.6f}"
        )

    def test_voronoi_areas_tile_surface(self, device):
        """Test that Voronoi areas perfectly tile the mesh surface.

        The sum of Voronoi areas should equal the sum of triangle areas,
        ensuring perfect tiling without gaps or overlaps (Meyer et al. 2003, Sec 3.4).
        """
        from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0

        for subdivisions in [0, 2, 4]:
            mesh = create_sphere_mesh(
                radius=1.0, subdivisions=subdivisions, device=device
            )
            voronoi_areas = compute_dual_volumes_0(mesh)

            # Sum of Voronoi areas should equal sum of triangle areas
            total_voronoi_area = voronoi_areas.sum().item()
            total_triangle_area = mesh.cell_areas.sum().item()
            relative_error = (
                abs(total_voronoi_area - total_triangle_area) / total_triangle_area
            )

            # Should be nearly exact (perfect tiling property)
            assert relative_error < 1e-6, (
                f"Voronoi areas don't perfectly tile mesh at subdivision {subdivisions}: "
                f"{relative_error:.9f} ({total_voronoi_area=:.6f}, {total_triangle_area=:.6f})"
            )


### Test Mean Curvature


class TestMeanCurvature:
    """Tests for mean curvature computation."""

    def test_sphere_mean_curvature(self, device):
        """Test that sphere has constant mean curvature H = 1/r."""
        radius = 2.0
        mesh = create_sphere_mesh(radius=radius, subdivisions=1, device=device)

        H_vertices = mesh.mean_curvature_vertices

        # Expected: H = 1/r for all vertices
        expected_H = 1.0 / radius

        # Should be close to expected
        mean_H = H_vertices.mean()
        assert torch.abs(mean_H - expected_H) / expected_H < 0.01  # Within 1%

        # All should be positive (outward normals)
        assert torch.all(H_vertices > 0)

    def test_plane_mean_curvature(self, device):
        """Test that flat plane has zero mean curvature."""
        mesh = create_plane_mesh(n_subdivisions=2, device=device)

        H_vertices = mesh.mean_curvature_vertices

        # Should be zero for interior vertices (boundary vertices are NaN)
        interior_H = H_vertices[~torch.isnan(H_vertices)]
        assert len(interior_H) > 0, "Should have interior vertices"
        assert torch.allclose(interior_H, torch.zeros_like(interior_H), atol=1e-6)

    def test_cylinder_mean_curvature(self, device):
        """Test that cylinder has H = 1/(2r) (curved in one direction only)."""
        from physicsnemo.mesh.primitives.surfaces import cylinder_open

        radius = 1.0
        mesh = cylinder_open.load(
            radius=radius,
            n_circ=64,
            n_height=32,
            device=device,  # Use finer mesh
        )

        H_vertices = mesh.mean_curvature_vertices

        # Expected: H = 1/(2r) for cylinder
        expected_H = 1.0 / (2 * radius)

        # Check interior vertices only (boundary vertices are NaN)
        interior_H = H_vertices[~torch.isnan(H_vertices)]

        assert len(interior_H) > 0, "Should have interior vertices"

        mean_H = interior_H.mean()
        relative_error = torch.abs(mean_H - expected_H) / expected_H

        # Interior vertices are perfect (0.0% error)
        assert relative_error < 0.001, (
            f"Mean curvature error {relative_error:.1%} exceeds 0.1% tolerance. "
            f"Got {mean_H:.4f}, expected {expected_H:.4f}"
        )

    def test_mean_curvature_convergence(self, device):
        """Test that mean curvature is accurate across subdivision levels."""
        radius = 1.0
        expected_H = 1.0 / radius

        for subdivisions in [0, 1, 2]:
            mesh = create_sphere_mesh(
                radius=radius, subdivisions=subdivisions, device=device
            )
            H_vertices = mesh.mean_curvature_vertices
            mean_H = H_vertices.mean()
            error = torch.abs(mean_H - expected_H)

            # Each subdivision level should maintain excellent accuracy
            assert error / expected_H < 0.01  # Within 1% at all levels

    def test_mean_curvature_codimension_error(self, device):
        """Test that mean curvature raises error for non-codimension-1."""
        # Create a tet mesh (codimension-0)
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="codimension-1"):
            _ = mesh.mean_curvature_vertices


### Test 1D Curvature (Curves)


class Test1DCurvature:
    """Tests for curvature of 1D curves."""

    def test_circular_arc_curvature(self, device):
        """Test curvature of circular arc (1D in 2D)."""
        curvature = 2.0  # κ = 1/r, r = 0.5
        mesh = create_line_curve_2d(n_points=20, curvature=curvature, device=device)

        K_vertices = mesh.gaussian_curvature_vertices

        # For 1D curves, Gaussian curvature is related to κ
        # Interior vertices should have consistent curvature
        # End vertices may differ (boundary effects)

        # Check that interior vertices have reasonable curvature
        interior_K = K_vertices[1:-1]  # Skip endpoints

        # Should all have same sign and similar magnitude
        assert torch.all(interior_K > 0) or torch.all(interior_K < 0)

    def test_straight_line_curvature(self, device):
        """Test that straight line has zero curvature."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        K_vertices = mesh.gaussian_curvature_vertices

        # Interior vertices should have zero curvature (straight line)
        # For 1D, interior vertices have angle sum = π (full angle for 1D)
        interior_K = K_vertices[1:-1]
        assert torch.allclose(interior_K, torch.zeros_like(interior_K), atol=1e-5)


### Test Edge Cases


class TestCurvatureEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_mesh(self, device):
        """Test curvature computation on empty mesh."""
        points = torch.empty((0, 3), dtype=torch.float32, device=device)
        cells = torch.empty((0, 3), dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        K_vertices = mesh.gaussian_curvature_vertices
        assert K_vertices.shape == (0,)

    def test_single_triangle(self, device):
        """Test curvature on single triangle."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        K_vertices = mesh.gaussian_curvature_vertices
        H_vertices = mesh.mean_curvature_vertices

        # Should compute without error
        assert K_vertices.shape == (3,)
        assert H_vertices.shape == (3,)

    def test_isolated_vertex(self, device):
        """Test that isolated vertices are handled gracefully."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [99.0, 99.0, 99.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        K_vertices = mesh.gaussian_curvature_vertices

        # Isolated vertex (index 3) should have zero or NaN curvature
        # Implementation choice - either is acceptable
        isolated_K = K_vertices[3]
        assert torch.isnan(isolated_K) or isolated_K == 0

    def test_caching(self, device):
        """Test that curvatures are cached."""
        mesh = create_sphere_mesh(radius=1.0, subdivisions=0, device=device)

        # First access
        K1 = mesh.gaussian_curvature_vertices
        H1 = mesh.mean_curvature_vertices

        # Check cached
        assert get_cached(mesh.point_data, "gaussian_curvature") is not None
        assert get_cached(mesh.point_data, "mean_curvature") is not None

        # Second access should return same values
        K2 = mesh.gaussian_curvature_vertices
        H2 = mesh.mean_curvature_vertices

        assert torch.allclose(K1, K2)
        assert torch.allclose(H1, H2)


### Test Dimension Coverage


class TestCurvatureDimensions:
    """Tests across different manifold dimensions."""

    def test_1d_curve_in_2d(self, device):
        """Test 1D curve curvature in 2D space."""
        mesh = create_line_curve_2d(n_points=10, curvature=1.0, device=device)

        K_vertices = mesh.gaussian_curvature_vertices

        assert K_vertices.shape == (mesh.n_points,)
        # Should have some non-zero curvature
        assert K_vertices.abs().max() > 0

    def test_2d_surface_in_3d(self, device):
        """Test 2D surface in 3D space (standard case)."""
        mesh = create_sphere_mesh(radius=1.0, subdivisions=0, device=device)

        K_vertices = mesh.gaussian_curvature_vertices
        H_vertices = mesh.mean_curvature_vertices

        assert K_vertices.shape == (mesh.n_points,)
        assert H_vertices.shape == (mesh.n_points,)

    def test_2d_surface_in_4d(self, device):
        """Test 2D surface in 4D space (higher codimension)."""
        # Create triangle in 4D
        points = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        # Gaussian curvature should work (intrinsic)
        K_vertices = mesh.gaussian_curvature_vertices
        assert K_vertices.shape == (3,)

        # Mean curvature should fail (requires codimension-1)
        with pytest.raises(ValueError, match="codimension-1"):
            _ = mesh.mean_curvature_vertices


### Test Principal Curvatures (Derived)


class TestPrincipalCurvatures:
    """Tests for principal curvatures derived from K and H."""

    def test_sphere_principal_curvatures(self, device):
        """Test that sphere has equal principal curvatures k1 = k2 = 1/r."""
        radius = 1.0
        mesh = create_sphere_mesh(radius=radius, subdivisions=2, device=device)

        K = mesh.gaussian_curvature_vertices
        H = mesh.mean_curvature_vertices

        # For sphere: k1 = k2 = 1/r
        # K = k1 * k2 = 1/r²
        # H = (k1 + k2)/2 = 1/r
        # Therefore: k1 = k2 = H

        expected_k = 1.0 / radius
        expected_K = expected_k**2

        # Mean curvature should match expected value
        mean_H = H.mean()
        mean_K = K.mean()

        H_rel_error = torch.abs(mean_H - expected_k) / expected_k
        K_rel_error = torch.abs(mean_K - expected_K) / expected_K

        # With subdivision level 1, should be within tight tolerance
        assert H_rel_error < 0.01, (
            f"Mean curvature error {H_rel_error:.1%} exceeds 1%. "
            f"Got {mean_H:.4f}, expected {expected_k:.4f}"
        )
        assert K_rel_error < 0.02, (
            f"Gaussian curvature error {K_rel_error:.1%} exceeds 2%. "
            f"Got {mean_K:.4f}, expected {expected_K:.4f}"
        )

        # Verify K ≈ H² for sphere (identity for sphere)
        K_from_H = H**2
        K_identity_error = (K - K_from_H).abs() / (K.abs() + 1e-10)
        assert K_identity_error.mean() < 0.02, (
            f"K vs H² relationship violated: mean error {K_identity_error.mean():.1%}"
        )

    def test_cylinder_principal_curvatures(self, device):
        """Test cylinder has k1 = 1/r, k2 = 0."""
        from physicsnemo.mesh.primitives.surfaces import cylinder_open

        radius = 1.0
        mesh = cylinder_open.load(radius=radius, n_circ=32, n_height=16, device=device)

        K = mesh.gaussian_curvature_vertices
        H = mesh.mean_curvature_vertices

        # For cylinder: k1 = 1/r, k2 = 0
        # K = k1 * k2 = 0
        # H = (k1 + k2)/2 = 1/(2r)

        # Filter to interior vertices (not on top/bottom boundary)
        # Top boundary: z > height/2 - epsilon
        # Bottom boundary: z < -height/2 + epsilon
        z_coords = mesh.points[:, 2]
        interior_mask = (z_coords > -0.9) & (z_coords < 0.9)

        K_interior = K[interior_mask]

        # Gaussian curvature should be near zero (intrinsically flat)
        assert torch.allclose(K_interior, torch.zeros_like(K_interior), atol=0.01)

        # Mean curvature should be positive
        H_interior = H[interior_mask]
        assert torch.all(H_interior > 0)


### Test Numerical Stability


class TestCurvatureNumerical:
    """Tests for numerical stability."""

    def test_small_radius_sphere(self, device):
        """Test curvature on very small sphere."""
        radius = 0.01
        mesh = create_sphere_mesh(radius=radius, subdivisions=2, device=device)

        K = mesh.gaussian_curvature_vertices
        H = mesh.mean_curvature_vertices

        # Should still compute valid curvatures
        assert not torch.any(torch.isnan(K))
        assert not torch.any(torch.isnan(H))

        # Should scale correctly with radius
        expected_K = 1.0 / (radius**2)
        expected_H = 1.0 / radius

        mean_K = K.mean()
        mean_H = H.mean()

        K_rel_error = torch.abs(mean_K - expected_K) / expected_K
        H_rel_error = torch.abs(mean_H - expected_H) / expected_H

        # Should be within tight tolerance even for small radius
        assert K_rel_error < 0.02, (
            f"Gaussian curvature error {K_rel_error:.1%} exceeds 2%. "
            f"Got {mean_K:.2f}, expected {expected_K:.2f}"
        )
        assert H_rel_error < 0.01, (
            f"Mean curvature error {H_rel_error:.1%} exceeds 1%. "
            f"Got {mean_H:.2f}, expected {expected_H:.2f}"
        )

    def test_large_radius_sphere(self, device):
        """Test curvature on very large sphere."""
        radius = 100.0
        mesh = create_sphere_mesh(radius=radius, subdivisions=2, device=device)

        K = mesh.gaussian_curvature_vertices
        H = mesh.mean_curvature_vertices

        # Should compute very small curvatures
        expected_K = 1.0 / (radius**2)
        expected_H = 1.0 / radius

        mean_K = K.mean()
        mean_H = H.mean()

        K_rel_error = torch.abs(mean_K - expected_K) / expected_K
        H_rel_error = torch.abs(mean_H - expected_H) / expected_H

        # Should be within tight tolerance even for large radius
        assert K_rel_error < 0.02, (
            f"Gaussian curvature error {K_rel_error:.1%} exceeds 2%. "
            f"Got {mean_K:.6f}, expected {expected_K:.6f}"
        )
        assert H_rel_error < 0.01, (
            f"Mean curvature error {H_rel_error:.1%} exceeds 1%. "
            f"Got {mean_H:.6f}, expected {expected_H:.6f}"
        )


### Test Sign Conventions


class TestCurvatureSigns:
    """Tests for sign conventions."""

    def test_positive_gaussian_curvature(self, device):
        """Test positive Gaussian curvature (elliptic point)."""
        # Sphere has positive curvature everywhere
        mesh = create_sphere_mesh(radius=1.0, subdivisions=0, device=device)
        K = mesh.gaussian_curvature_vertices

        assert torch.all(K > 0)

    def test_zero_gaussian_curvature(self, device):
        """Test zero Gaussian curvature (parabolic/flat) at interior vertices."""
        # Plane has zero curvature at interior vertices
        mesh = create_plane_mesh(n_subdivisions=2, device=device)
        K = mesh.gaussian_curvature_vertices

        # Check only interior vertices
        n = 5  # Grid size for n_subdivisions=2
        interior_mask = torch.zeros(mesh.n_points, dtype=torch.bool, device=device)
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                if 0 < i < n - 1 and 0 < j < n - 1:
                    interior_mask[idx] = True

        interior_K = K[interior_mask]
        assert torch.allclose(interior_K, torch.zeros_like(interior_K), atol=1e-5)

    def test_signed_mean_curvature_sphere(self, device):
        """Test that mean curvature sign depends on normal orientation."""
        mesh = create_sphere_mesh(radius=1.0, subdivisions=0, device=device)
        H = mesh.mean_curvature_vertices

        # With outward normals, sphere should have positive H
        # (All should have same sign)
        assert torch.all(H > 0) or torch.all(H < 0)


###############################################################################
# Regression: gaussian_curvature_cells on embedded manifolds
###############################################################################


class TestGaussianCurvatureCellsRegression:
    """Regression tests for gaussian_curvature_cells on embedded manifolds.

    The original implementation computed pairwise angles between centroid-to-
    centroid vectors in ambient 3D space instead of the manifold's tangent
    plane, causing spurious curvature on developable surfaces (e.g. cylinders)
    and divergence with mesh refinement.
    """

    def test_cylinder_curvature_near_zero(self, device):
        """Cylinder has zero intrinsic Gaussian curvature (developable surface).

        Interior cells should have |K| near zero.  The original implementation
        gave |K| ~ 3-40 on a cylinder, *increasing* with mesh refinement.
        """
        from physicsnemo.mesh.primitives.surfaces import cylinder_open

        cyl = cylinder_open.load(n_circ=32, n_height=16).to(device)
        K = cyl.gaussian_curvature_cells
        K_finite = K[~torch.isnan(K)]

        assert K_finite.abs().mean() < 0.05, (
            f"Cylinder K_cells abs mean = {K_finite.abs().mean():.4f}, "
            "expected near 0 for a developable surface"
        )

    def test_sphere_curvature_correct(self, device):
        """Sphere of radius r should have K = 1/r^2."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        for radius in (1.0, 2.0):
            sphere = sphere_icosahedral.load(radius=radius, subdivisions=3).to(device)
            K = sphere.gaussian_curvature_cells
            K_finite = K[~torch.isnan(K)]
            expected = 1.0 / radius**2

            assert abs(K_finite.mean().item() - expected) < 0.1 * expected, (
                f"Sphere r={radius}: K_cells mean = {K_finite.mean():.4f}, "
                f"expected {expected:.4f}"
            )

    def test_cylinder_convergence(self, device):
        """Cell curvature on a cylinder should not diverge with refinement."""
        from physicsnemo.mesh.primitives.surfaces import cylinder_open

        means = []
        for n in (16, 32, 64):
            cyl = cylinder_open.load(n_circ=n, n_height=n).to(device)
            K = cyl.gaussian_curvature_cells
            means.append(K[~torch.isnan(K)].abs().mean().item())

        assert means[-1] < means[0] + 0.01, (
            f"Cell curvature diverges with refinement: {means}"
        )

    def test_boundary_cells_are_nan(self, device):
        """Boundary cells (touching boundary vertices) should be NaN."""
        from physicsnemo.mesh.primitives.surfaces import cylinder_open

        cyl = cylinder_open.load(n_circ=16, n_height=8).to(device)
        K = cyl.gaussian_curvature_cells

        from physicsnemo.mesh.boundaries._detection import get_boundary_cells

        is_bnd = get_boundary_cells(cyl)
        assert torch.isnan(K[is_bnd]).all(), "Boundary cells should have NaN curvature"
