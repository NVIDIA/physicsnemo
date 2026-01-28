"""Comprehensive tests for discrete calculus operators.

Tests gradient, divergence, curl, and Laplacian operators using analytical
fields with known derivatives. Verifies fundamental calculus identities.
"""

import pytest
import torch

from physicsnemo.mesh.primitives import procedural, volumes


### Analytical field generators
def make_constant_field(value=5.0):
    """Constant scalar field."""
    return lambda r: torch.full((r.shape[0],), value, dtype=r.dtype, device=r.device)


def make_linear_field(coeffs):
    """Linear field: φ = a·r where a = coeffs."""
    coeffs_tensor = torch.tensor(coeffs)
    return lambda r: (r * coeffs_tensor.to(r.device)).sum(dim=-1)


def make_quadratic_field():
    """Quadratic field: φ = ||r||² = x² + y² + z²."""
    return lambda r: (r**2).sum(dim=-1)


def make_polynomial_field_3d():
    """Polynomial: φ = x²y + yz² - 2xz."""

    def phi(r):
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        return x**2 * y + y * z**2 - 2 * x * z

    return phi


def make_uniform_divergence_field_3d():
    """Vector field v = [x, y, z], div(v) = 3."""
    return lambda r: r.clone()


def make_scaled_divergence_field_3d(scale_factors):
    """Vector field v = [a×x, b×y, c×z], div(v) = a+b+c."""
    a, b, c = scale_factors

    def v(r):
        result = r.clone()
        result[:, 0] *= a
        result[:, 1] *= b
        result[:, 2] *= c
        return result

    return v


def make_zero_divergence_rotation_3d():
    """Vector field v = [-y, x, 0], div(v) = 0."""

    def v(r):
        result = torch.zeros_like(r)
        result[:, 0] = -r[:, 1]  # -y
        result[:, 1] = r[:, 0]  # x
        result[:, 2] = 0.0
        return result

    return v


def make_zero_divergence_field_3d():
    """Vector field v = [yz, xz, xy], div(v) = 0."""

    def v(r):
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        result = torch.zeros_like(r)
        result[:, 0] = y * z
        result[:, 1] = x * z
        result[:, 2] = x * y
        return result

    return v


def make_radial_field():
    """Radial field v = r, div(v) = n (spatial dims)."""
    return lambda r: r.clone()


def make_uniform_curl_field_3d():
    """Vector field v = [-y, x, 0], curl(v) = [0, 0, 2]."""
    return make_zero_divergence_rotation_3d()  # Same field


def make_zero_curl_field_3d():
    """Conservative field v = [x, y, z] = ∇(½||r||²), curl(v) = 0."""
    return lambda r: r.clone()


def make_helical_field_3d():
    """Helical field v = [-y, x, z], curl(v) = [0, 0, 2]."""

    def v(r):
        result = torch.zeros_like(r)
        result[:, 0] = -r[:, 1]
        result[:, 1] = r[:, 0]
        result[:, 2] = r[:, 2]
        return result

    return v


def make_polynomial_curl_field_3d():
    """v = [yz, -xz, 0], curl(v) = [-x, -y, -2z]."""

    def v(r):
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        result = torch.zeros_like(r)
        result[:, 0] = y * z
        result[:, 1] = -x * z
        result[:, 2] = 0.0
        return result

    return v


def make_harmonic_field_2d():
    """Harmonic field φ = x² - y² in 2D, Δφ = 0."""

    def phi(r):
        if r.shape[-1] >= 2:
            return r[:, 0] ** 2 - r[:, 1] ** 2
        else:
            raise ValueError("Need at least 2D for this field")

    return phi


def make_harmonic_field_xy():
    """Harmonic field φ = xy, Δφ = 0."""

    def phi(r):
        if r.shape[-1] >= 2:
            return r[:, 0] * r[:, 1]
        else:
            raise ValueError("Need at least 2D")

    return phi


### Mesh fixtures
@pytest.fixture
def simple_triangle_mesh_2d():
    """Simple 2D triangle mesh for basic tests."""
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    cells = torch.tensor(
        [
            [0, 1, 4],
            [0, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
        ]
    )
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(points=points, cells=cells)


### Test Classes


class TestGradient:
    """Test gradient computation."""

    def test_gradient_of_constant_is_zero(self):
        """∇(const) = 0."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Create constant field
        const_value = 5.0
        mesh.point_data["const"] = torch.full(
            (mesh.n_points,), const_value, dtype=torch.float32
        )

        # Compute gradient
        mesh_grad = mesh.compute_point_derivatives(keys="const", method="lsq")

        gradient = mesh_grad.point_data["const_gradient"]

        # Should be zero everywhere
        assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-6)

    def test_gradient_of_linear_is_exact(self):
        """∇(a·r) = a exactly for linear fields."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Linear field: φ = 2x + 3y - z
        coeffs = torch.tensor([2.0, 3.0, -1.0])
        phi = (mesh.points * coeffs).sum(dim=-1)

        mesh.point_data["linear"] = phi

        # Compute gradient
        mesh_grad = mesh.compute_point_derivatives(keys="linear", method="lsq")
        gradient = mesh_grad.point_data["linear_gradient"]

        # Should equal coeffs everywhere
        expected = coeffs.unsqueeze(0).expand(mesh.n_points, -1)

        # Linear functions should be reconstructed exactly by LSQ
        assert torch.allclose(gradient, expected, atol=1e-4)

    @pytest.mark.parametrize("method", ["lsq"])
    def test_quadratic_hessian_uniformity(self, method):
        """φ = ||r||² has uniform Laplacian (Hessian trace is constant).

        This tests the KEY property: Laplacian of ||r||² should be spatially uniform.
        The absolute value may have systematic bias in first-order methods, but
        the spatial variation (std dev) should be small relative to mean.
        """
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Quadratic field
        phi = (mesh.points**2).sum(dim=-1)
        mesh.point_data["quadratic"] = phi

        # Compute Laplacian via div(grad(φ))
        mesh_grad = mesh.compute_point_derivatives(keys="quadratic", method=method)
        grad = mesh_grad.point_data["quadratic_gradient"]

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        laplacian = compute_divergence_points_lsq(mesh_grad, grad)

        # Key test: Laplacian should be UNIFORM (low std dev relative to mean)
        mean_lap = laplacian.mean()
        std_lap = laplacian.std()

        # Coefficient of variation should be small
        cv = std_lap / mean_lap.abs().clamp(min=1e-10)

        assert cv < 0.5, (
            f"Laplacian not uniform: CV={cv:.3f}, mean={mean_lap:.3f}, std={std_lap:.3f}"
        )

        # Laplacian should be positive (correct sign)
        assert mean_lap > 0, "Laplacian should be positive for convex function"


class TestDivergence:
    """Test divergence computation with analytical fields."""

    def test_uniform_divergence_3d(self):
        """v = [x,y,z], div(v) = 3 (constant everywhere)."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Vector field v = r
        v = mesh.points.clone()

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        divergence = compute_divergence_points_lsq(mesh, v)

        # LSQ should exactly recover divergence of linear field
        expected = 3.0
        assert torch.allclose(
            divergence, torch.full_like(divergence, expected), atol=1e-4
        ), f"Divergence mean={divergence.mean():.6f}, expected={expected}"

    def test_scaled_divergence_field(self):
        """v = [2x, 3y, 4z], div(v) = 2+3+4 = 9."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        v = mesh.points.clone()
        v[:, 0] *= 2.0
        v[:, 1] *= 3.0
        v[:, 2] *= 4.0

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        divergence = compute_divergence_points_lsq(mesh, v)

        # Should be exactly 9
        assert torch.allclose(divergence, torch.full_like(divergence, 9.0), atol=1e-4)

    def test_zero_divergence_rotation(self):
        """v = [-y,x,0], div(v) = 0 (solenoidal field)."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Rotation field
        v = torch.zeros_like(mesh.points)
        v[:, 0] = -mesh.points[:, 1]  # -y
        v[:, 1] = mesh.points[:, 0]  # x
        v[:, 2] = 0.0

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        divergence = compute_divergence_points_lsq(mesh, v)

        # Should be exactly zero (linear field components)
        assert torch.allclose(divergence, torch.zeros_like(divergence), atol=1e-6)

    def test_zero_divergence_field_xyz(self):
        """v = [yz, xz, xy], div(v) = 0."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]
        v = torch.stack([y * z, x * z, x * y], dim=-1)

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        divergence = compute_divergence_points_lsq(mesh, v)

        # ∂(yz)/∂x + ∂(xz)/∂y + ∂(xy)/∂z = 0 + 0 + 0 = 0
        # But these are quadratic, so expect some error
        assert divergence.abs().mean() < 0.5


class TestCurl:
    """Test curl computation with analytical fields."""

    def test_uniform_curl_3d(self):
        """v = [-y,x,0], curl(v) = [0,0,2] (uniform curl)."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Rotation field
        v = torch.zeros_like(mesh.points)
        v[:, 0] = -mesh.points[:, 1]
        v[:, 1] = mesh.points[:, 0]
        v[:, 2] = 0.0

        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        curl_v = compute_curl_points_lsq(mesh, v)

        # LSQ should exactly recover curl of linear field
        expected = torch.zeros_like(curl_v)
        expected[:, 2] = 2.0

        assert torch.allclose(curl_v, expected, atol=1e-4)

    def test_zero_curl_conservative_field(self):
        """v = r = ∇(½||r||²), curl(v) = 0 (irrotational)."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Conservative field (gradient of potential)
        v = mesh.points.clone()

        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        curl_v = compute_curl_points_lsq(mesh, v)

        # Should be exactly zero (curl of gradient of linear function)
        assert torch.allclose(curl_v, torch.zeros_like(curl_v), atol=1e-6)

    def test_helical_field(self):
        """v = [-y, x, z], curl(v) = [0, 0, 2]."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        v = torch.zeros_like(mesh.points)
        v[:, 0] = -mesh.points[:, 1]
        v[:, 1] = mesh.points[:, 0]
        v[:, 2] = mesh.points[:, 2]

        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        curl_v = compute_curl_points_lsq(mesh, v)

        expected = torch.zeros_like(curl_v)
        expected[:, 2] = 2.0

        assert torch.allclose(curl_v, expected, atol=1e-4)

    def test_curl_multiple_axes(self):
        """Test curl with rotation about different axes (all linear fields)."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Test 1: Rotation about z-axis: v = [-y, x, 0], curl = [0, 0, 2]
        v_z = torch.zeros_like(mesh.points)
        v_z[:, 0] = -mesh.points[:, 1]
        v_z[:, 1] = mesh.points[:, 0]

        # Test 2: Rotation about x-axis: v = [0, -z, y], curl = [2, 0, 0]
        v_x = torch.zeros_like(mesh.points)
        v_x[:, 1] = -mesh.points[:, 2]
        v_x[:, 2] = mesh.points[:, 1]

        # Test 3: Rotation about y-axis: v = [z, 0, -x], curl = [0, 2, 0]
        v_y = torch.zeros_like(mesh.points)
        v_y[:, 0] = mesh.points[:, 2]
        v_y[:, 2] = -mesh.points[:, 0]

        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        curl_z = compute_curl_points_lsq(mesh, v_z)
        curl_x = compute_curl_points_lsq(mesh, v_x)
        curl_y = compute_curl_points_lsq(mesh, v_y)

        # All should be exact (linear fields)
        expected_z = torch.zeros_like(curl_z)
        expected_z[:, 2] = 2.0

        expected_x = torch.zeros_like(curl_x)
        expected_x[:, 0] = 2.0

        expected_y = torch.zeros_like(curl_y)
        expected_y[:, 1] = 2.0

        assert torch.allclose(curl_z, expected_z, atol=1e-4), "Curl about z-axis failed"
        assert torch.allclose(curl_x, expected_x, atol=1e-4), "Curl about x-axis failed"
        assert torch.allclose(curl_y, expected_y, atol=1e-4), "Curl about y-axis failed"


class TestLaplacian:
    """Test Laplace-Beltrami operator."""

    def test_harmonic_function_laplacian_zero(self, simple_triangle_mesh_2d):
        """Harmonic function φ = x² - y² should have Δφ ≈ 0 in 2D."""
        mesh = simple_triangle_mesh_2d

        # Harmonic function in 2D
        phi = mesh.points[:, 0] ** 2 - mesh.points[:, 1] ** 2
        mesh.point_data["harmonic"] = phi

        # Compute Laplacian
        mesh_grad = mesh.compute_point_derivatives(keys="harmonic", method="lsq")
        grad = mesh_grad.point_data["harmonic_gradient"]

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        laplacian = compute_divergence_points_lsq(mesh_grad, grad)

        # For a true harmonic function, Laplacian = 0
        # Interior points should have |Δφ| << |φ|
        assert laplacian.abs().mean() < 0.5, (
            f"Harmonic function Laplacian should be ~0, got mean={laplacian.mean():.4f}"
        )

    def test_dec_laplacian_linear_function_zero(self):
        """DEC Laplacian of linear function should be exactly zero."""
        # Simple 2D mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

        from physicsnemo.mesh.mesh import Mesh

        mesh = Mesh(points=points, cells=cells)

        # Linear function
        phi = 2 * points[:, 0] + 3 * points[:, 1]

        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        lap = compute_laplacian_points_dec(mesh, phi)

        # Interior point (index 4) should have Laplacian = 0
        assert torch.abs(lap[4]) < 1e-6, (
            f"Laplacian of linear function at interior: {lap[4]:.6f}"
        )

    def test_dec_laplacian_quadratic_reasonable(self):
        """DEC Laplacian of φ=||r||² gives reasonable approximation.

        Note: Uses a Delaunay-quality mesh. Circumcentric duals work best on
        well-centered meshes where circumcenters lie inside triangles. Axis-aligned
        grids create poorly-conditioned duals.
        """
        from physicsnemo.mesh.primitives.surfaces import sphere_uv

        # Use a sphere mesh which is naturally well-centered (close to Delaunay)
        mesh = sphere_uv.load(radius=1.0, theta_resolution=20, phi_resolution=20)

        # Test function: φ = z²
        # On a sphere, this is NOT constant, so we get a non-trivial Laplacian
        # Analytical: ∂²(z²)/∂z² = 2
        phi = mesh.points[:, 2] ** 2

        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        lap = compute_laplacian_points_dec(mesh, phi)

        # Expected: 4 (∇²(x²+y²) = 2+2)
        expected = 4.0
        assert torch.abs(lap[4] - expected) < expected * 0.01, (
            f"Laplacian at interior: {lap[4]:.3f}, expected ≈{expected}"
        )


class TestManifolds:
    """Test calculus on manifolds (surfaces in higher dimensions)."""

    def test_intrinsic_gradient_orthogonal_to_normal(self):
        """Intrinsic gradient should be perpendicular to surface normal."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=3)

        # Any scalar field
        phi = (mesh.points**2).sum(dim=-1)
        mesh.point_data["test_field"] = phi

        # Compute intrinsic and extrinsic gradients
        mesh_grad = mesh.compute_point_derivatives(
            keys="test_field", method="lsq", gradient_type="both"
        )

        grad_intrinsic = mesh_grad.point_data["test_field_gradient_intrinsic"]
        grad_extrinsic = mesh_grad.point_data["test_field_gradient_extrinsic"]

        # Get normals at points (use mesh's area-weighted normals)
        point_normals = mesh.point_normals

        # Intrinsic gradient should be orthogonal to normal
        dot_products_intrinsic = (grad_intrinsic * point_normals).sum(dim=-1)

        assert dot_products_intrinsic.abs().max() < 1e-2, (
            f"Intrinsic gradient not orthogonal to normal: max dot product = {dot_products_intrinsic.abs().max():.6f}"
        )

        # Extrinsic gradient should be finite and have correct shape
        assert torch.all(torch.isfinite(grad_extrinsic))
        assert grad_extrinsic.shape == grad_intrinsic.shape


class TestCalculusIdentities:
    """Test fundamental calculus identities."""

    def test_curl_of_gradient_is_zero(self):
        """curl(∇φ) = 0 for any scalar field."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Should be zero (curl of conservative field)
        # For LINEAR potential, curl of gradient should be near-exact zero
        # Use phi = x + y for exact test (quadratic fields have O(h) discretization error)
        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        phi_linear = mesh.points[:, 0] + mesh.points[:, 1]
        mesh.point_data["phi_linear"] = phi_linear
        mesh_grad_linear = mesh.compute_point_derivatives(
            keys="phi_linear", method="lsq"
        )
        grad_linear = mesh_grad_linear.point_data["phi_linear_gradient"]
        curl_of_grad_linear = compute_curl_points_lsq(mesh_grad_linear, grad_linear)

        assert torch.allclose(
            curl_of_grad_linear, torch.zeros_like(curl_of_grad_linear), atol=1e-5
        )

    def test_divergence_of_curl_is_zero(self):
        """div(curl(v)) = 0 for any vector field."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        # Use rotation field
        v = torch.zeros_like(mesh.points)
        v[:, 0] = -mesh.points[:, 1]
        v[:, 1] = mesh.points[:, 0]
        v[:, 2] = mesh.points[:, 2]  # Helical

        # Compute curl
        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        curl_v = compute_curl_points_lsq(mesh, v)

        # Compute divergence of curl
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        div_curl = compute_divergence_points_lsq(mesh, curl_v)

        # Should be zero
        assert torch.allclose(div_curl, torch.zeros_like(div_curl), atol=1e-5)


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize("field_type", ["constant", "linear"])
    @pytest.mark.parametrize("method", ["lsq"])
    def test_gradient_exact_recovery(self, field_type, method):
        """Gradient of constant/linear fields should be exact."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)

        if field_type == "constant":
            phi = torch.full((mesh.n_points,), 5.0)
            expected_grad = torch.zeros((mesh.n_points, mesh.n_spatial_dims))
            tol = 1e-6
        else:  # linear
            coeffs = torch.tensor([2.0, 3.0, -1.0])
            phi = (mesh.points * coeffs).sum(dim=-1)
            expected_grad = coeffs.unsqueeze(0).expand(mesh.n_points, -1)
            tol = 1e-4

        mesh.point_data["test"] = phi
        mesh_grad = mesh.compute_point_derivatives(keys="test", method=method)
        grad = mesh_grad.point_data["test_gradient"]

        assert torch.allclose(grad, expected_grad, atol=tol)

    @pytest.mark.parametrize("divergence_value", [1.0, 3.0, 9.0])
    def test_uniform_divergence_recovery(self, divergence_value):
        """Divergence of scaled identity field should be exact."""
        mesh = volumes.cube_volume.load(size=1.0, n_subdivisions=5)
        scale = divergence_value / mesh.n_spatial_dims
        v = mesh.points * scale

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        div_v = compute_divergence_points_lsq(mesh, v)

        assert torch.allclose(
            div_v, torch.full_like(div_v, divergence_value), atol=1e-4
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
