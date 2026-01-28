"""Comprehensive tests for Laplace-Beltrami operator.

Tests coverage for:
- Scalar fields (already mostly tested)
- Tensor fields (multi-dimensional point_values)
- Non-2D manifold error handling
- Edge cases and boundary conditions
"""

import pytest
import torch

from physicsnemo.mesh.calculus.laplacian import (
    compute_laplacian_points,
    compute_laplacian_points_dec,
)
from physicsnemo.mesh.mesh import Mesh


@pytest.fixture(params=["cpu"])
def device(request):
    """Test on CPU."""
    return request.param


class TestLaplacianTensorFields:
    """Tests for Laplacian of tensor (vector/matrix) fields."""

    def create_triangle_mesh(self, device="cpu"):
        """Create simple triangle mesh for testing."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, (3**0.5) / 2],
                [1.5, (3**0.5) / 2],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            dtype=torch.long,
            device=device,
        )

        return Mesh(points=points, cells=cells)

    def test_laplacian_vector_field(self, device):
        """Test Laplacian of vector field (n_points, n_dims)."""
        mesh = self.create_triangle_mesh(device)

        # Create vector field: velocity or position-like data
        # Use linear field for simplicity: v = [x, y]
        vector_values = mesh.points.clone()  # (n_points, 2)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, vector_values)

        # Should have same shape as input
        assert laplacian.shape == vector_values.shape
        assert laplacian.shape == (mesh.n_points, 2)

        # Laplacian should be computed (not NaN/Inf)
        assert not torch.any(torch.isnan(laplacian))
        assert not torch.any(torch.isinf(laplacian))

    def test_laplacian_3d_vector_field(self, device):
        """Test Laplacian of 3D vector field on 2D manifold."""
        mesh = self.create_triangle_mesh(device)

        # Create 3D vector field on 2D mesh
        # Each point has a 3D vector
        vector_values = torch.randn(mesh.n_points, 3, device=device)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, vector_values)

        # Should have same shape
        assert laplacian.shape == (mesh.n_points, 3)

        # No NaNs
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_matrix_field(self, device):
        """Test Laplacian of matrix field (n_points, d1, d2)."""
        mesh = self.create_triangle_mesh(device)

        # Create 2x2 matrix at each point
        matrix_values = torch.randn(mesh.n_points, 2, 2, device=device)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, matrix_values)

        # Should have same shape
        assert laplacian.shape == (mesh.n_points, 2, 2)

        # No NaNs
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_higher_order_tensor(self, device):
        """Test Laplacian of higher-order tensor field."""
        mesh = self.create_triangle_mesh(device)

        # Create 3D tensor at each point (e.g., stress tensor components)
        tensor_values = torch.randn(mesh.n_points, 3, 3, 3, device=device)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, tensor_values)

        # Should have same shape
        assert laplacian.shape == (mesh.n_points, 3, 3, 3)

        # No NaNs
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_vector_constant(self, device):
        """Test Laplacian of constant vector field is zero."""
        mesh = self.create_triangle_mesh(device)

        # Constant vector field
        constant_vector = torch.tensor([1.0, 2.0], device=device)
        vector_values = constant_vector.unsqueeze(0).expand(mesh.n_points, -1)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, vector_values)

        # Should be close to zero
        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1e-5)

    def test_laplacian_vector_linear_field(self, device):
        """Test Laplacian of linear vector field."""
        mesh = self.create_triangle_mesh(device)

        # Linear vector field: v(x,y) = [2x+y, x-y]
        x = mesh.points[:, 0]
        y = mesh.points[:, 1]

        vector_values = torch.stack(
            [
                2 * x + y,
                x - y,
            ],
            dim=1,
        )

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, vector_values)

        # Laplacian should be computed (not NaN/Inf)
        assert not torch.any(torch.isnan(laplacian))
        assert not torch.any(torch.isinf(laplacian))


class TestLaplacianManifoldDimensions:
    """Tests for Laplacian on different manifold dimensions."""

    def test_laplacian_not_implemented_for_1d(self, device):
        """Test that 1D manifolds raise NotImplementedError."""
        # Create 1D mesh (edges)
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
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

        # Should raise NotImplementedError
        scalar_values = torch.randn(mesh.n_points, device=device)

        with pytest.raises(NotImplementedError, match="only implemented for triangle meshes"):
            compute_laplacian_points_dec(mesh, scalar_values)

    def test_laplacian_not_implemented_for_3d(self, device):
        """Test that 3D manifolds raise NotImplementedError."""
        # Create single tetrahedron
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

        # Should raise NotImplementedError
        scalar_values = torch.randn(mesh.n_points, device=device)

        with pytest.raises(NotImplementedError, match="only implemented for triangle meshes"):
            compute_laplacian_points_dec(mesh, scalar_values)

    def test_laplacian_wrapper_function(self, device):
        """Test the wrapper function compute_laplacian_points."""
        # Create simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        scalar_values = torch.randn(mesh.n_points, device=device)

        # Test wrapper function
        laplacian1 = compute_laplacian_points(mesh, scalar_values)
        laplacian2 = compute_laplacian_points_dec(mesh, scalar_values)

        # Should be identical
        assert torch.allclose(laplacian1, laplacian2)


class TestLaplacianBoundaryAndEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def create_sphere_mesh(self, subdivisions=1, device="cpu"):
        """Create icosahedral sphere."""
        phi = (1.0 + (5.0**0.5)) / 2.0

        vertices = [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]

        points = torch.tensor(vertices, dtype=torch.float32, device=device)
        points = points / torch.norm(points, dim=-1, keepdim=True)

        faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        cells = torch.tensor(faces, dtype=torch.int64, device=device)
        mesh = Mesh(points=points, cells=cells)

        # Subdivide if requested
        for _ in range(subdivisions):
            mesh = mesh.subdivide(levels=1, filter="linear")
            mesh = Mesh(
                points=mesh.points / torch.norm(mesh.points, dim=-1, keepdim=True),
                cells=mesh.cells,
            )

        return mesh

    def test_laplacian_on_closed_surface(self, device):
        """Test Laplacian on closed surface (no boundary)."""
        mesh = self.create_sphere_mesh(subdivisions=0, device=device)

        # Create constant scalar field
        scalar_values = torch.ones(mesh.n_points, device=device)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        # For constant function, Laplacian should be zero
        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1e-5)

    def test_laplacian_empty_mesh(self, device):
        """Test Laplacian with no cells."""
        points = torch.randn(10, 2, device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        scalar_values = torch.randn(mesh.n_points, device=device)

        # With no cells, cotangent weights will be empty
        # This should handle gracefully (likely return zeros or small values)
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        # Should have correct shape
        assert laplacian.shape == scalar_values.shape

    def test_laplacian_single_triangle(self, device):
        """Test Laplacian on single isolated triangle."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Linear field
        scalar_values = mesh.points[:, 0]  # x-coordinate

        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        # Should compute without errors
        assert laplacian.shape == (3,)
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_degenerate_voronoi_area(self, device):
        """Test Laplacian handles very small Voronoi areas."""
        # Create mesh with very small triangle
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1e-8],  # Very small height
                [1.5, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        scalar_values = torch.ones(mesh.n_points, device=device)

        # Should handle small areas without producing NaN/Inf
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        assert not torch.any(torch.isnan(laplacian))
        assert not torch.any(torch.isinf(laplacian))


class TestLaplacianNumericalProperties:
    """Tests for numerical properties of the Laplacian."""

    def test_laplacian_symmetry(self, device):
        """Test that Laplacian operator is symmetric (self-adjoint)."""
        # Create mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        # Two different scalar fields
        f = torch.randn(mesh.n_points, device=device)
        g = torch.randn(mesh.n_points, device=device)

        # Compute Laplacians
        Lf = compute_laplacian_points_dec(mesh, f)
        Lg = compute_laplacian_points_dec(mesh, g)

        # For symmetric operator: <f, Lg> = <Lf, g>
        # (up to boundary terms, which don't exist for closed manifolds)

        # Get Voronoi areas for proper inner product
        from physicsnemo.mesh.calculus._circumcentric_dual import (
            get_or_compute_dual_volumes_0,
        )

        voronoi_areas = get_or_compute_dual_volumes_0(mesh)

        # Weighted inner products
        f_Lg = (f * Lg * voronoi_areas).sum()
        Lf_g = (Lf * g * voronoi_areas).sum()

        # Should be approximately equal (numerically)
        rel_diff = torch.abs(f_Lg - Lf_g) / (torch.abs(f_Lg) + torch.abs(Lf_g) + 1e-10)
        assert rel_diff < 0.01  # Within 1%


class TestDECLaplacianSphericalHarmonics:
    r"""Tests for DEC Laplacian using spherical harmonic eigenfunctions.

    Spherical harmonics Y_l^m are eigenfunctions of the Laplace-Beltrami operator
    on the unit sphere with eigenvalue \lambda = -l(l+1).

    These tests validate that the DEC implementation correctly recovers these
    eigenvalues, providing strong evidence for correctness.
    """

    def create_unit_sphere(self, subdivisions: int = 4) -> Mesh:
        """Create high-resolution unit sphere via icosahedral subdivision."""
        from physicsnemo.mesh.primitives.surfaces import sphere_uv

        # Use UV sphere for simplicity; high resolution for accuracy
        return sphere_uv.load(radius=1.0, theta_resolution=50, phi_resolution=50)

    def test_laplacian_constant_function_zero(self):
        r"""Verify \Delta(const) = 0 on closed surface.

        A constant function is a spherical harmonic with l=0 (Y_0^0),
        which has eigenvalue -0(0+1) = 0.
        """
        mesh = self.create_unit_sphere()
        phi = torch.ones(mesh.n_points, dtype=torch.float32)

        lap = compute_laplacian_points_dec(mesh, phi)

        assert lap.abs().max() < 1e-5, f"Laplacian of constant: max={lap.abs().max():.6f}"
        assert lap.abs().mean() < 1e-6, f"Laplacian of constant: mean={lap.abs().mean():.6f}"

    def test_laplacian_spherical_harmonic_Y10(self):
        r"""Verify \Delta_S(z) = -2z (eigenvalue -2 for l=1).

        Y_1^0 \propto z = cos(theta), with eigenvalue \lambda = -l(l+1) = -2.
        """
        mesh = self.create_unit_sphere()
        z = mesh.points[:, 2]
        phi = z.clone()

        lap = compute_laplacian_points_dec(mesh, phi)

        # Expected: Delta_S(z) = -2 * z
        expected = -2 * z

        # Verify eigenvalue relationship: lap / phi should be ~-2 (where phi != 0)
        mask = phi.abs() > 0.1  # Avoid division by near-zero
        ratio = lap[mask] / phi[mask]

        mean_eigenvalue = ratio.mean()
        assert (
            abs(mean_eigenvalue - (-2.0)) < 0.1
        ), f"Y_1^0 eigenvalue: {mean_eigenvalue:.4f}, expected -2.0"

        # Verify correlation with expected
        correlation = torch.corrcoef(torch.stack([lap, expected]))[0, 1]
        assert correlation > 0.999, f"Y_1^0 correlation: {correlation:.6f}"

    def test_laplacian_spherical_harmonic_Y20(self):
        r"""Verify \Delta_S(3z^2-1) = -6(3z^2-1) (eigenvalue -6 for l=2).

        Y_2^0 \propto (3cos^2(theta) - 1) = 3z^2 - 1, with eigenvalue -6.
        """
        mesh = self.create_unit_sphere()
        z = mesh.points[:, 2]
        phi = 3 * z**2 - 1

        lap = compute_laplacian_points_dec(mesh, phi)

        # Expected: Delta_S(3z^2 - 1) = -6 * (3z^2 - 1)
        expected = -6 * phi

        # Verify eigenvalue relationship
        mask = phi.abs() > 0.1
        ratio = lap[mask] / phi[mask]

        mean_eigenvalue = ratio.mean()
        assert (
            abs(mean_eigenvalue - (-6.0)) < 0.15
        ), f"Y_2^0 eigenvalue: {mean_eigenvalue:.4f}, expected -6.0"

        # Verify correlation
        correlation = torch.corrcoef(torch.stack([lap, expected]))[0, 1]
        assert correlation > 0.999, f"Y_2^0 correlation: {correlation:.6f}"

    def test_laplacian_spherical_harmonic_Y21(self):
        r"""Verify \Delta_S(xz) = -6(xz) (eigenvalue -6 for l=2, m=1).

        Y_2^1 \propto xz (real part) or yz (imaginary part), with eigenvalue -6.
        """
        mesh = self.create_unit_sphere()
        x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]

        # Test xz
        phi_xz = x * z
        lap_xz = compute_laplacian_points_dec(mesh, phi_xz)

        mask = phi_xz.abs() > 0.05
        ratio_xz = lap_xz[mask] / phi_xz[mask]
        mean_eigenvalue_xz = ratio_xz.mean()

        assert (
            abs(mean_eigenvalue_xz - (-6.0)) < 0.15
        ), f"Y_2^1 (xz) eigenvalue: {mean_eigenvalue_xz:.4f}, expected -6.0"

        # Test yz
        phi_yz = y * z
        lap_yz = compute_laplacian_points_dec(mesh, phi_yz)

        mask = phi_yz.abs() > 0.05
        ratio_yz = lap_yz[mask] / phi_yz[mask]
        mean_eigenvalue_yz = ratio_yz.mean()

        assert (
            abs(mean_eigenvalue_yz - (-6.0)) < 0.15
        ), f"Y_2^1 (yz) eigenvalue: {mean_eigenvalue_yz:.4f}, expected -6.0"

    def test_laplacian_spherical_harmonic_Y22(self):
        r"""Verify \Delta_S(x^2-y^2) = -6(x^2-y^2) (eigenvalue -6 for l=2, m=2).

        Y_2^2 \propto x^2-y^2 (real part) or xy (imaginary part), with eigenvalue -6.
        """
        mesh = self.create_unit_sphere()
        x, y = mesh.points[:, 0], mesh.points[:, 1]

        # Test x^2 - y^2
        phi_x2y2 = x**2 - y**2
        lap_x2y2 = compute_laplacian_points_dec(mesh, phi_x2y2)

        mask = phi_x2y2.abs() > 0.05
        ratio_x2y2 = lap_x2y2[mask] / phi_x2y2[mask]
        mean_eigenvalue_x2y2 = ratio_x2y2.mean()

        assert (
            abs(mean_eigenvalue_x2y2 - (-6.0)) < 0.15
        ), f"Y_2^2 (x^2-y^2) eigenvalue: {mean_eigenvalue_x2y2:.4f}, expected -6.0"

        # Test xy
        phi_xy = x * y
        lap_xy = compute_laplacian_points_dec(mesh, phi_xy)

        mask = phi_xy.abs() > 0.05
        ratio_xy = lap_xy[mask] / phi_xy[mask]
        mean_eigenvalue_xy = ratio_xy.mean()

        assert (
            abs(mean_eigenvalue_xy - (-6.0)) < 0.15
        ), f"Y_2^2 (xy) eigenvalue: {mean_eigenvalue_xy:.4f}, expected -6.0"

    def test_laplacian_z_squared_position_dependent(self):
        r"""Verify \Delta_S(z^2) = 2 - 6z^2 at all vertices.

        z^2 = cos^2(theta) decomposes into Y_0^0 and Y_2^0 components:
            z^2 = (1/3) + (2/3)(3z^2 - 1)/2 = (1/3) + (1/3)(3z^2 - 1)

        Applying the Laplacian:
            \Delta_S(z^2) = 0 + (-6)(2/3)(3z^2 - 1)/2 = 2 - 6z^2
        """
        mesh = self.create_unit_sphere()
        z = mesh.points[:, 2]
        phi = z**2

        lap = compute_laplacian_points_dec(mesh, phi)

        # Analytical: Delta_S(z^2) = 2 - 6z^2
        expected = 2 - 6 * z**2

        # Verify correlation
        correlation = torch.corrcoef(torch.stack([lap, expected]))[0, 1]
        assert correlation > 0.999, f"Correlation: {correlation:.6f}"

        # Verify mean absolute error
        mean_error = (lap - expected).abs().mean()
        assert mean_error < 0.03, f"Mean error: {mean_error:.4f}"

        # Verify max error is reasonable
        max_error = (lap - expected).abs().max()
        assert max_error < 0.1, f"Max error: {max_error:.4f}"

    def test_laplacian_flat_mesh_quadratic(self):
        r"""Verify \Delta(x^2+y^2) = 4 on flat 2D mesh.

        On a flat manifold, the Laplace-Beltrami reduces to the standard Laplacian.
        For phi = x^2 + y^2: \Delta phi = 2 + 2 = 4 (uniform everywhere).
        """
        # Create flat 2D mesh (unit square with interior vertex)
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],  # Interior vertex
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=torch.long,
        )
        mesh = Mesh(points=points, cells=cells)

        # phi = x^2 + y^2
        phi = points[:, 0] ** 2 + points[:, 1] ** 2

        lap = compute_laplacian_points_dec(mesh, phi)

        # Interior vertex (index 4) should have Laplacian = 4
        interior_lap = lap[4]
        assert (
            abs(interior_lap - 4.0) < 0.01
        ), f"Flat mesh Laplacian at interior: {interior_lap:.4f}, expected 4.0"
