"""Comprehensive tests for Laplace-Beltrami operator.

Tests coverage for:
- Scalar fields (already mostly tested)
- Tensor fields (multi-dimensional point_values)
- Non-2D manifold error handling
- Edge cases and boundary conditions
"""

import torch
import pytest

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.calculus.laplacian import (
    compute_laplacian_points_dec,
    compute_laplacian_points,
)


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

        with pytest.raises(NotImplementedError, match="triangle meshes"):
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

        with pytest.raises(NotImplementedError, match="triangle meshes"):
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
        from physicsnemo.mesh.calculus._circumcentric_dual import get_or_compute_dual_volumes_0

        voronoi_areas = get_or_compute_dual_volumes_0(mesh)

        # Weighted inner products
        f_Lg = (f * Lg * voronoi_areas).sum()
        Lf_g = (Lf * g * voronoi_areas).sum()

        # Should be approximately equal (numerically)
        rel_diff = torch.abs(f_Lg - Lf_g) / (torch.abs(f_Lg) + torch.abs(Lf_g) + 1e-10)
        assert rel_diff < 0.01  # Within 1%
