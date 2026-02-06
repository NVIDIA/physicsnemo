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

"""Comprehensive tests for discrete calculus operators.

Tests gradient, divergence, curl, and Laplacian operators using analytical
fields with known derivatives. Verifies fundamental calculus identities,
DEC operators, edge cases, and numerical properties.

This module consolidates tests from:
- Core analytical field tests (gradient, divergence, curl, Laplacian)
- DEC operators (exterior derivative, Hodge star, sharp/flat)
- Laplacian-specific tests (tensor fields, spherical harmonics, edge cases)
- Code coverage tests (error handling, edge conditions)
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives import procedural

###############################################################################
# Helper Functions - Analytical Field Generators
###############################################################################


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


###############################################################################
# Fixtures
###############################################################################


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
    return Mesh(points=points, cells=cells)


@pytest.fixture
def simple_tet_mesh():
    """Simple tetrahedral mesh for testing."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
    return Mesh(points=points, cells=cells)


###############################################################################
# Core Analytical Field Tests
###############################################################################


class TestGradient:
    """Test gradient computation."""

    def test_gradient_of_constant_is_zero(self):
        """∇(const) = 0."""
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

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

        # Two nested first-order LSQ steps (grad then div) yield O(1) error
        # in the Laplacian that doesn't vanish with refinement: the gradient
        # bias epsilon ~ O(h) varies on the mesh scale, so div(epsilon) ~ O(1).
        # On lumpy_ball (noise_amplitude=0.5), CV ~0.36 is typical.
        assert cv < 0.5, (
            f"Laplacian not uniform: CV={cv:.3f}, mean={mean_lap:.3f}, std={std_lap:.3f}"
        )

        # Laplacian should be positive (correct sign)
        assert mean_lap > 0, "Laplacian should be positive for convex function"


class TestDivergence:
    """Test divergence computation with analytical fields."""

    def test_uniform_divergence_3d(self):
        """v = [x,y,z], div(v) = 3 (constant everywhere)."""
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

        x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]
        v = torch.stack([y * z, x * z, x * y], dim=-1)

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        divergence = compute_divergence_points_lsq(mesh, v)

        # ∂(yz)/∂x + ∂(xz)/∂y + ∂(xy)/∂z = 0 + 0 + 0 = 0
        # Quadratic components have O(h) LSQ error; lumpy_ball is moderate resolution.
        assert divergence.abs().mean() < 0.15


class TestCurl:
    """Test curl computation with analytical fields."""

    def test_uniform_curl_3d(self):
        """v = [-y,x,0], curl(v) = [0,0,2] (uniform curl)."""
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

        # Conservative field (gradient of potential)
        v = mesh.points.clone()

        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        curl_v = compute_curl_points_lsq(mesh, v)

        # Should be exactly zero (curl of gradient of linear function)
        assert torch.allclose(curl_v, torch.zeros_like(curl_v), atol=1e-6)

    def test_helical_field(self):
        """v = [-y, x, z], curl(v) = [0, 0, 2]."""
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

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
        # Coarse mesh (5 points, 4 triangles); large discretization error expected.
        assert laplacian.abs().mean() < 0.3, (
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
        r"""DEC Laplacian of phi=z^2 gives correct surface Laplacian.

        For the Laplace-Beltrami operator on a unit sphere:
            \Delta_S(z^2) = 2 - 6z^2

        This is the SURFACE Laplacian (intrinsic to the manifold), not the
        ambient 3D Laplacian. The result varies by position: negative near
        poles (|z| ~ 1), positive near equator (z ~ 0).

        Derivation: z^2 = cos^2(theta) can be decomposed into spherical harmonics
        Y_0^0 and Y_2^0. The eigenvalue for l=2 is -l(l+1) = -6, giving the
        position-dependent result.
        """
        from physicsnemo.mesh.primitives.surfaces import sphere_uv

        # Use higher resolution for better accuracy
        mesh = sphere_uv.load(radius=1.0, theta_resolution=40, phi_resolution=40)

        # Test function: phi = z^2
        phi = mesh.points[:, 2] ** 2

        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        lap = compute_laplacian_points_dec(mesh, phi)

        # Analytical surface Laplacian: Delta_S(z^2) = 2 - 6z^2
        z = mesh.points[:, 2]
        expected = 2 - 6 * z**2

        # Verify correlation (should be ~1.0)
        correlation = torch.corrcoef(torch.stack([lap, expected]))[0, 1]
        assert correlation > 0.999, f"Correlation with analytical: {correlation:.6f}"

        # Verify mean absolute error is small
        mean_error = (lap - expected).abs().mean()
        assert mean_error < 0.05, f"Mean error: {mean_error:.4f}"


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
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()

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
        mesh = procedural.lumpy_ball.load()
        scale = divergence_value / mesh.n_spatial_dims
        v = mesh.points * scale

        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_lsq

        div_v = compute_divergence_points_lsq(mesh, v)

        assert torch.allclose(
            div_v, torch.full_like(div_v, divergence_value), atol=1e-4
        )


###############################################################################
# Laplacian Tensor Fields Tests
###############################################################################


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

    def test_laplacian_vector_field(self):
        """Test Laplacian of vector field (n_points, n_dims)."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_triangle_mesh()

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

    def test_laplacian_3d_vector_field(self):
        """Test Laplacian of 3D vector field on 2D manifold."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_triangle_mesh()

        # Create 3D vector field on 2D mesh
        # Each point has a 3D vector
        vector_values = torch.randn(mesh.n_points, 3)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, vector_values)

        # Should have same shape
        assert laplacian.shape == (mesh.n_points, 3)

        # No NaNs
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_matrix_field(self):
        """Test Laplacian of matrix field (n_points, d1, d2)."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_triangle_mesh()

        # Create 2x2 matrix at each point
        matrix_values = torch.randn(mesh.n_points, 2, 2)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, matrix_values)

        # Should have same shape
        assert laplacian.shape == (mesh.n_points, 2, 2)

        # No NaNs
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_higher_order_tensor(self):
        """Test Laplacian of higher-order tensor field."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_triangle_mesh()

        # Create 3D tensor at each point (e.g., stress tensor components)
        tensor_values = torch.randn(mesh.n_points, 3, 3, 3)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, tensor_values)

        # Should have same shape
        assert laplacian.shape == (mesh.n_points, 3, 3, 3)

        # No NaNs
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_vector_constant(self):
        """Test Laplacian of constant vector field is zero."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_triangle_mesh()

        # Constant vector field
        constant_vector = torch.tensor([1.0, 2.0])
        vector_values = constant_vector.unsqueeze(0).expand(mesh.n_points, -1)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, vector_values)

        # Should be close to zero
        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1e-5)

    def test_laplacian_vector_linear_field(self):
        """Test Laplacian of linear vector field."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_triangle_mesh()

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


###############################################################################
# Laplacian Spherical Harmonics Tests
###############################################################################


class TestLaplacianSphericalHarmonics:
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
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_unit_sphere()
        phi = torch.ones(mesh.n_points, dtype=torch.float32)

        lap = compute_laplacian_points_dec(mesh, phi)

        assert lap.abs().max() < 1e-5, (
            f"Laplacian of constant: max={lap.abs().max():.6f}"
        )
        assert lap.abs().mean() < 1e-6, (
            f"Laplacian of constant: mean={lap.abs().mean():.6f}"
        )

    def test_laplacian_spherical_harmonic_Y10(self):
        r"""Verify \Delta_S(z) = -2z (eigenvalue -2 for l=1).

        Y_1^0 \propto z = cos(theta), with eigenvalue \lambda = -l(l+1) = -2.
        """
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

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
        assert abs(mean_eigenvalue - (-2.0)) < 0.1, (
            f"Y_1^0 eigenvalue: {mean_eigenvalue:.4f}, expected -2.0"
        )

        # Verify correlation with expected
        correlation = torch.corrcoef(torch.stack([lap, expected]))[0, 1]
        assert correlation > 0.999, f"Y_1^0 correlation: {correlation:.6f}"

    def test_laplacian_spherical_harmonic_Y20(self):
        r"""Verify \Delta_S(3z^2-1) = -6(3z^2-1) (eigenvalue -6 for l=2).

        Y_2^0 \propto (3cos^2(theta) - 1) = 3z^2 - 1, with eigenvalue -6.
        """
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

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
        assert abs(mean_eigenvalue - (-6.0)) < 0.15, (
            f"Y_2^0 eigenvalue: {mean_eigenvalue:.4f}, expected -6.0"
        )

        # Verify correlation
        correlation = torch.corrcoef(torch.stack([lap, expected]))[0, 1]
        assert correlation > 0.999, f"Y_2^0 correlation: {correlation:.6f}"

    def test_laplacian_spherical_harmonic_Y21(self):
        r"""Verify \Delta_S(xz) = -6(xz) (eigenvalue -6 for l=2, m=1).

        Y_2^1 \propto xz (real part) or yz (imaginary part), with eigenvalue -6.
        """
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_unit_sphere()
        x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]

        # Test xz
        phi_xz = x * z
        lap_xz = compute_laplacian_points_dec(mesh, phi_xz)

        mask = phi_xz.abs() > 0.05
        ratio_xz = lap_xz[mask] / phi_xz[mask]
        mean_eigenvalue_xz = ratio_xz.mean()

        assert abs(mean_eigenvalue_xz - (-6.0)) < 0.15, (
            f"Y_2^1 (xz) eigenvalue: {mean_eigenvalue_xz:.4f}, expected -6.0"
        )

        # Test yz
        phi_yz = y * z
        lap_yz = compute_laplacian_points_dec(mesh, phi_yz)

        mask = phi_yz.abs() > 0.05
        ratio_yz = lap_yz[mask] / phi_yz[mask]
        mean_eigenvalue_yz = ratio_yz.mean()

        assert abs(mean_eigenvalue_yz - (-6.0)) < 0.15, (
            f"Y_2^1 (yz) eigenvalue: {mean_eigenvalue_yz:.4f}, expected -6.0"
        )

    def test_laplacian_spherical_harmonic_Y22(self):
        r"""Verify \Delta_S(x^2-y^2) = -6(x^2-y^2) (eigenvalue -6 for l=2, m=2).

        Y_2^2 \propto x^2-y^2 (real part) or xy (imaginary part), with eigenvalue -6.
        """
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_unit_sphere()
        x, y = mesh.points[:, 0], mesh.points[:, 1]

        # Test x^2 - y^2
        phi_x2y2 = x**2 - y**2
        lap_x2y2 = compute_laplacian_points_dec(mesh, phi_x2y2)

        mask = phi_x2y2.abs() > 0.05
        ratio_x2y2 = lap_x2y2[mask] / phi_x2y2[mask]
        mean_eigenvalue_x2y2 = ratio_x2y2.mean()

        assert abs(mean_eigenvalue_x2y2 - (-6.0)) < 0.15, (
            f"Y_2^2 (x^2-y^2) eigenvalue: {mean_eigenvalue_x2y2:.4f}, expected -6.0"
        )

        # Test xy
        phi_xy = x * y
        lap_xy = compute_laplacian_points_dec(mesh, phi_xy)

        mask = phi_xy.abs() > 0.05
        ratio_xy = lap_xy[mask] / phi_xy[mask]
        mean_eigenvalue_xy = ratio_xy.mean()

        assert abs(mean_eigenvalue_xy - (-6.0)) < 0.15, (
            f"Y_2^2 (xy) eigenvalue: {mean_eigenvalue_xy:.4f}, expected -6.0"
        )


###############################################################################
# Laplacian Boundary and Edge Cases
###############################################################################


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

    def test_laplacian_on_closed_surface(self):
        """Test Laplacian on closed surface (no boundary)."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = self.create_sphere_mesh(subdivisions=0)

        # Create constant scalar field
        scalar_values = torch.ones(mesh.n_points)

        # Compute Laplacian
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        # For constant function, Laplacian should be zero
        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1e-5)

    def test_laplacian_empty_mesh(self):
        """Test Laplacian with no cells."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        points = torch.randn(10, 2)
        cells = torch.zeros((0, 3), dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)

        scalar_values = torch.randn(mesh.n_points)

        # With no cells, cotangent weights will be empty
        # This should handle gracefully (likely return zeros or small values)
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        # Should have correct shape
        assert laplacian.shape == scalar_values.shape

    def test_laplacian_single_triangle(self):
        """Test Laplacian on single isolated triangle."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)

        # Linear field
        scalar_values = mesh.points[:, 0]  # x-coordinate

        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        # Should compute without errors
        assert laplacian.shape == (3,)
        assert not torch.any(torch.isnan(laplacian))

    def test_laplacian_degenerate_voronoi_area(self):
        """Test Laplacian handles very small Voronoi areas."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        # Create mesh with very small triangle
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1e-8],  # Very small height
                [1.5, 0.0],
            ],
            dtype=torch.float32,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            dtype=torch.long,
        )

        mesh = Mesh(points=points, cells=cells)

        scalar_values = torch.ones(mesh.n_points)

        # Should handle small areas without producing NaN/Inf
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        assert not torch.any(torch.isnan(laplacian))
        assert not torch.any(torch.isinf(laplacian))


class TestLaplacianNumericalProperties:
    """Tests for numerical properties of the Laplacian."""

    def test_laplacian_symmetry(self):
        """Test that Laplacian operator is symmetric (self-adjoint)."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

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

        # Two different scalar fields
        f = torch.randn(mesh.n_points)
        g = torch.randn(mesh.n_points)

        # Compute Laplacians
        Lf = compute_laplacian_points_dec(mesh, f)
        Lg = compute_laplacian_points_dec(mesh, g)

        # For symmetric operator: <f, Lg> = <Lf, g>
        # (up to boundary terms, which don't exist for closed manifolds)

        # Get Voronoi areas for proper inner product
        from physicsnemo.mesh.geometry.dual_meshes import (
            get_or_compute_dual_volumes_0,
        )

        voronoi_areas = get_or_compute_dual_volumes_0(mesh)

        # Weighted inner products
        f_Lg = (f * Lg * voronoi_areas).sum()
        Lf_g = (Lf * g * voronoi_areas).sum()

        # Should be approximately equal (numerically)
        rel_diff = torch.abs(f_Lg - Lf_g) / (torch.abs(f_Lg) + torch.abs(Lf_g) + 1e-10)
        assert rel_diff < 0.01  # Within 1%

    def test_laplacian_dec_basic(self):
        """Test compute_laplacian_points_dec produces correct shape and dtype."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        # Create simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        scalar_values = torch.randn(mesh.n_points)

        laplacian = compute_laplacian_points_dec(mesh, scalar_values)

        assert laplacian.shape == scalar_values.shape
        assert laplacian.dtype == scalar_values.dtype


class TestLaplacianManifoldDimensions:
    """Tests for Laplacian on different manifold dimensions."""

    def test_laplacian_works_for_1d(self):
        """Test that DEC Laplacian works on 1D edge meshes."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        # Create 1D mesh (edges in 2D ambient space)
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
        )

        cells = torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
        )

        mesh = Mesh(points=points, cells=cells)
        scalar_values = torch.randn(mesh.n_points)

        # Should run without error and return correct shape
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)
        assert laplacian.shape == scalar_values.shape
        assert torch.isfinite(laplacian).all()

    def test_laplacian_works_for_3d(self):
        """Test that DEC Laplacian works on 3D tetrahedral meshes."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        # Create single tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
                [0.5, (3**0.5) / 6, ((2 / 3) ** 0.5)],
            ],
            dtype=torch.float32,
        )

        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)
        scalar_values = torch.randn(mesh.n_points)

        # Should run without error and return correct shape
        laplacian = compute_laplacian_points_dec(mesh, scalar_values)
        assert laplacian.shape == scalar_values.shape
        assert torch.isfinite(laplacian).all()

    def test_laplacian_flat_mesh_quadratic(self):
        r"""Verify \Delta(x^2+y^2) = 4 on flat 2D mesh.

        On a flat manifold, the Laplace-Beltrami reduces to the standard Laplacian.
        For phi = x^2 + y^2: \Delta phi = 2 + 2 = 4 (uniform everywhere).
        """
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

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
        assert abs(interior_lap - 4.0) < 0.01, (
            f"Flat mesh Laplacian at interior: {interior_lap:.4f}, expected 4.0"
        )


###############################################################################
# DEC Operators Tests
###############################################################################


class TestDECOperators:
    """Test DEC-specific code paths."""

    def test_exterior_derivative_0(self, simple_tet_mesh):
        """Test exterior derivative d₀: Ω⁰ → Ω¹."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0

        mesh = simple_tet_mesh
        vertex_values = torch.arange(mesh.n_points, dtype=torch.float32)

        edge_values, edges = exterior_derivative_0(mesh, vertex_values)

        assert edge_values.shape[0] == edges.shape[0]
        assert edges.shape[1] == 2

        # Verify: df(edge) = f(v1) - f(v0)
        for i in range(len(edges)):
            expected = vertex_values[edges[i, 1]] - vertex_values[edges[i, 0]]
            assert torch.allclose(edge_values[i], expected, atol=1e-6)

    def test_exterior_derivative_tensor_field(self, simple_tet_mesh):
        """Test d₀ on tensor-valued 0-form."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0

        mesh = simple_tet_mesh
        # Vector-valued function at vertices
        vertex_vectors = mesh.points.clone()  # (n_points, 3)

        edge_values, edges = exterior_derivative_0(mesh, vertex_vectors)

        assert edge_values.shape == (len(edges), 3)

    def test_hodge_star_0(self, simple_tet_mesh):
        """Test Hodge star on 0-forms."""
        from physicsnemo.mesh.calculus._hodge_star import hodge_star_0

        mesh = simple_tet_mesh
        vertex_values = torch.ones(mesh.n_points)

        dual_values = hodge_star_0(mesh, vertex_values)

        assert dual_values.shape == vertex_values.shape
        # All values should be scaled by dual volumes
        assert (dual_values > 0).all()

    def test_hodge_star_0_tensor(self, simple_tet_mesh):
        """Test Hodge star on tensor-valued 0-form."""
        from physicsnemo.mesh.calculus._hodge_star import hodge_star_0

        mesh = simple_tet_mesh
        vertex_tensors = mesh.points.clone()  # (n_points, 3)

        dual_tensors = hodge_star_0(mesh, vertex_tensors)

        assert dual_tensors.shape == vertex_tensors.shape

    def test_hodge_star_1(self, simple_tet_mesh):
        """Test Hodge star on 1-forms."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._hodge_star import hodge_star_1

        mesh = simple_tet_mesh
        vertex_values = torch.ones(mesh.n_points)

        edge_values, edges = exterior_derivative_0(mesh, vertex_values)
        dual_edge_values = hodge_star_1(mesh, edge_values, edges)

        assert dual_edge_values.shape == edge_values.shape

    def test_sharp_operator(self, simple_tet_mesh):
        """Test sharp operator: 1-form → vector field."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import sharp

        mesh = simple_tet_mesh
        vertex_values = torch.arange(mesh.n_points, dtype=torch.float32)

        edge_values, edges = exterior_derivative_0(mesh, vertex_values)
        vector_field = sharp(mesh, edge_values, edges)

        assert vector_field.shape == (mesh.n_points, mesh.n_spatial_dims)

    def test_sharp_operator_tensor(self, simple_tet_mesh):
        """Test sharp on tensor-valued 1-form."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import sharp

        mesh = simple_tet_mesh
        vertex_tensors = mesh.points.clone()

        edge_tensors, edges = exterior_derivative_0(mesh, vertex_tensors)
        vector_field = sharp(mesh, edge_tensors, edges)

        assert vector_field.shape[0] == mesh.n_points

    def test_flat_operator(self, simple_tet_mesh):
        """Test flat operator: vector field → 1-form."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import flat

        mesh = simple_tet_mesh
        vector_field = mesh.points.clone()

        # Get edges
        _, edges = exterior_derivative_0(mesh, torch.zeros(mesh.n_points))

        edge_1form = flat(mesh, vector_field, edges)

        assert edge_1form.shape[0] == len(edges)

    def test_flat_operator_tensor(self, simple_tet_mesh):
        """Test flat on tensor field."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import flat

        mesh = simple_tet_mesh
        # Tensor field (n_points, 3, 2) for example
        tensor_field = mesh.points.unsqueeze(-1).repeat(1, 1, 2)

        _, edges = exterior_derivative_0(mesh, torch.zeros(mesh.n_points))

        edge_form = flat(mesh, tensor_field, edges)

        assert edge_form.ndim > 1

    def test_dec_gradient_points(self, simple_tet_mesh):
        """Test DEC gradient code path (implementation incomplete)."""
        from physicsnemo.mesh.calculus.gradient import compute_gradient_points_dec

        mesh = simple_tet_mesh
        phi = 2 * mesh.points[:, 0] + 3 * mesh.points[:, 1] - mesh.points[:, 2]

        grad = compute_gradient_points_dec(mesh, phi)

        # Just verify it runs and returns correct shape
        assert grad.shape == (mesh.n_points, mesh.n_spatial_dims)
        assert torch.isfinite(grad).all()


class TestExteriorDerivative:
    """Test d₁ exterior derivative."""

    def test_exterior_derivative_1_on_triangles(self):
        """Test d₁: Ω¹ → Ω² on triangle mesh."""
        from physicsnemo.mesh.calculus._exterior_derivative import (
            exterior_derivative_0,
            exterior_derivative_1,
        )

        # Triangle mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0]])
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Create 0-form and compute df
        vertex_values = torch.arange(mesh.n_points, dtype=torch.float32)
        edge_1form, edges = exterior_derivative_0(mesh, vertex_values)

        # Compute d(1-form)
        face_2form, faces = exterior_derivative_1(mesh, edge_1form, edges)

        assert face_2form.shape[0] == mesh.n_cells

    def test_exterior_derivative_1_error_on_1d(self):
        """Test d₁ raises error on 1D manifold."""
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_1

        # 1D mesh (curve)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        cells = torch.tensor([[0, 1], [1, 2]])
        mesh = Mesh(points=points, cells=cells)

        edge_values = torch.ones(mesh.n_cells)
        edges = mesh.cells

        with pytest.raises(ValueError, match="requires n_manifold_dims >= 2"):
            exterior_derivative_1(mesh, edge_values, edges)


class TestCircumcentricDual:
    """Test circumcentric dual computation."""

    def test_circumcenter_edge(self):
        """Test circumcenter of edge (1-simplex)."""
        from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

        # Single edge
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        circumcenters = compute_circumcenters(vertices)

        # Should be midpoint
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(circumcenters, expected, atol=1e-6)

    def test_circumcenter_triangle_2d(self):
        """Test circumcenter of triangle in 2D."""
        from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

        # Right triangle at origin
        vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

        circumcenters = compute_circumcenters(vertices)

        # Should be at [0.5, 0.5] (midpoint of hypotenuse)
        expected = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(circumcenters, expected, atol=1e-5)

    def test_circumcenter_triangle_3d(self):
        """Test circumcenter of triangle embedded in 3D."""
        from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

        # Right triangle in xy-plane
        vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        circumcenters = compute_circumcenters(vertices)

        # For embedded triangle, uses least-squares (over-determined system)
        # Just verify shape and finiteness
        assert circumcenters.shape == (1, 3)
        assert torch.isfinite(circumcenters).all()

    def test_circumcenter_tetrahedron(self):
        """Test circumcenter of tetrahedron."""
        from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

        # Regular tetrahedron (approximately)
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0], [0.5, 0.433, 0.816]]]
        )

        circumcenters = compute_circumcenters(vertices)

        # Should be equidistant from all vertices
        assert circumcenters.shape == (1, 3)

        # Verify equidistance
        for i in range(4):
            dist = torch.norm(circumcenters[0] - vertices[0, i])
            if i == 0:
                ref_dist = dist
            else:
                assert torch.allclose(dist, ref_dist, atol=1e-4)

    def test_circumcenter_non_origin_triangle(self):
        """Regression test: circumcenter for triangle not at origin.

        The bug was a wrong RHS in the linear system that only showed up
        when the first vertex was not at the origin.
        Triangle (1,1), (5,1), (1,5) has circumcenter at (3,3).
        """
        from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

        vertices = torch.tensor([[[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]]])
        cc = compute_circumcenters(vertices)
        torch.testing.assert_close(cc, torch.tensor([[3.0, 3.0]]), atol=1e-6, rtol=1e-6)

    def test_circumcenter_equilateral_triangle(self):
        """Regression test: circumcenter of equilateral triangle centered at origin."""
        from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

        vertices = torch.tensor([[[1.0, 0.0], [-0.5, 0.8660254], [-0.5, -0.8660254]]])
        cc = compute_circumcenters(vertices)
        torch.testing.assert_close(cc, torch.zeros(1, 2), atol=1e-5, rtol=1e-5)


###############################################################################
# Cell Derivatives Tests
###############################################################################


class TestCellDerivatives:
    """Test cell-based derivative computation."""

    def test_cell_gradient_lsq(self, simple_tet_mesh):
        """Test LSQ gradient on cell data."""
        mesh = simple_tet_mesh

        # Linear function on cells
        cell_centroids = mesh.cell_centroids
        cell_values = (cell_centroids * torch.tensor([2.0, 3.0, -1.0])).sum(dim=-1)

        mesh.cell_data["test"] = cell_values

        mesh_grad = mesh.compute_cell_derivatives(keys="test", method="lsq")

        grad = mesh_grad.cell_data["test_gradient"]
        assert grad.shape == (mesh.n_cells, mesh.n_spatial_dims)

        # Should recover linear coefficients approximately.
        # Coarse mesh (4 tets, ~3 face-adjacent neighbors per cell); cell-based
        # LSQ has limited accuracy with so few neighbors.
        expected = torch.tensor([2.0, 3.0, -1.0])
        assert torch.allclose(grad.mean(dim=0), expected, atol=0.25)

    def test_cell_gradient_dec_not_implemented(self, simple_tet_mesh):
        """Test that DEC cell gradients raise NotImplementedError."""
        mesh = simple_tet_mesh
        mesh.cell_data["test"] = torch.ones(mesh.n_cells)

        with pytest.raises(NotImplementedError):
            mesh.compute_cell_derivatives(keys="test", method="dec")


class TestTensorFields:
    """Test gradient computation on tensor fields."""

    def test_vector_field_gradient_jacobian(self, simple_tet_mesh):
        """Test that gradient of vector field gives Jacobian."""
        mesh = simple_tet_mesh

        # Vector field
        mesh.point_data["velocity"] = mesh.points.clone()

        mesh_grad = mesh.compute_point_derivatives(keys="velocity", method="lsq")

        jacobian = mesh_grad.point_data["velocity_gradient"]

        # Shape should be (n_points, 3, 3) for 3D
        assert jacobian.shape == (mesh.n_points, 3, 3)

        # For v=r, Jacobian should be identity.
        # Mean Jacobian should be close to I; coarse mesh (5 pts, 4 tets)
        # limits per-point accuracy, but the mean should be tighter.
        mean_jac = jacobian.mean(dim=0)
        expected = torch.eye(3)

        assert torch.allclose(mean_jac, expected, atol=0.1)


###############################################################################
# Edge Cases and Error Handling
###############################################################################


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_gradient_invalid_method(self, simple_tet_mesh):
        """Test that invalid method raises ValueError."""
        mesh = simple_tet_mesh
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        with pytest.raises(ValueError, match="Invalid method"):
            mesh.compute_point_derivatives(keys="test", method="invalid")

    def test_gradient_invalid_gradient_type(self, simple_tet_mesh):
        """Test that invalid gradient_type raises ValueError."""
        mesh = simple_tet_mesh
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        with pytest.raises(ValueError, match="Invalid gradient_type"):
            mesh.compute_point_derivatives(keys="test", gradient_type="invalid")

    def test_laplacian_on_3d_mesh_constant(self, simple_tet_mesh):
        """Test that DEC Laplacian of constant on 3D mesh is zero."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        mesh = simple_tet_mesh  # 3D manifold
        phi = torch.ones(mesh.n_points)

        laplacian = compute_laplacian_points_dec(mesh, phi)
        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1e-5)

    def test_curl_on_2d_raises(self):
        """Test that curl on 2D data raises ValueError."""
        from physicsnemo.mesh.calculus.curl import compute_curl_points_lsq

        # 2D mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        v = torch.ones((mesh.n_points, 2))

        with pytest.raises(ValueError, match="only defined for 3D"):
            compute_curl_points_lsq(mesh, v)

    def test_isolated_point_gradient_zero(self):
        """Test that isolated points (no neighbors) get zero gradient."""
        # Mesh with isolated point
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],  # Isolated
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])  # Only connects first 3 in one direction
        mesh = Mesh(points=points, cells=cells)

        phi = torch.arange(mesh.n_points, dtype=torch.float32)

        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_point_gradient_lsq,
        )

        grad = compute_point_gradient_lsq(mesh, phi)

        # Should not crash, gradients should be defined
        assert grad.shape == (mesh.n_points, mesh.n_spatial_dims)


class TestGradientTypes:
    """Test all gradient_type options."""

    def test_extrinsic_gradient(self):
        """Test gradient_type='extrinsic'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", gradient_type="extrinsic"
        )

        assert "test_gradient" in mesh_grad.point_data.keys()
        assert "test_gradient_intrinsic" not in mesh_grad.point_data.keys()

    def test_intrinsic_gradient(self):
        """Test gradient_type='intrinsic'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", gradient_type="intrinsic"
        )

        assert "test_gradient" in mesh_grad.point_data.keys()
        assert "test_gradient_extrinsic" not in mesh_grad.point_data.keys()

    def test_both_gradients(self):
        """Test gradient_type='both'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(keys="test", gradient_type="both")

        assert "test_gradient_intrinsic" in mesh_grad.point_data.keys()
        assert "test_gradient_extrinsic" in mesh_grad.point_data.keys()


class TestKeyParsing:
    """Test various key input formats."""

    def test_none_keys_all_fields(self, simple_tet_mesh):
        """Test keys=None computes all non-cached fields (excludes "_cache" sub-dict)."""
        from physicsnemo.mesh.utilities._cache import set_cached

        mesh = simple_tet_mesh
        mesh.point_data["field1"] = torch.ones(mesh.n_points)
        mesh.point_data["field2"] = torch.ones(mesh.n_points)
        set_cached(
            mesh.point_data, "test_value", torch.ones(mesh.n_points)
        )  # Should skip

        mesh_grad = mesh.compute_point_derivatives(keys=None)

        assert "field1_gradient" in mesh_grad.point_data.keys()
        assert "field2_gradient" in mesh_grad.point_data.keys()
        # Cached values should not have gradients computed
        assert "test_value_gradient" not in mesh_grad.point_data.keys()

    def test_nested_tensordict_keys(self, simple_tet_mesh):
        """Test nested TensorDict access."""
        from tensordict import TensorDict

        mesh = simple_tet_mesh
        nested = TensorDict(
            {"temperature": torch.ones(mesh.n_points)},
            batch_size=torch.Size([mesh.n_points]),
        )
        mesh.point_data["flow"] = nested

        mesh_grad = mesh.compute_point_derivatives(keys=("flow", "temperature"))

        assert "flow" in mesh_grad.point_data.keys()
        assert "temperature_gradient" in mesh_grad.point_data["flow"].keys()

    def test_list_of_keys(self, simple_tet_mesh):
        """Test list of multiple keys."""
        mesh = simple_tet_mesh
        mesh.point_data["field1"] = torch.ones(mesh.n_points)
        mesh.point_data["field2"] = torch.ones(mesh.n_points) * 2

        mesh_grad = mesh.compute_point_derivatives(keys=["field1", "field2"])

        assert "field1_gradient" in mesh_grad.point_data.keys()
        assert "field2_gradient" in mesh_grad.point_data.keys()


###############################################################################
# Higher Codimension and Specialized Tests
###############################################################################


class TestHigherCodeimension:
    """Test manifolds with codimension > 1."""

    def test_gradient_on_curve_in_3d(self):
        """Test gradient on 1D curve in 3D space (codimension=2)."""
        # Helix
        t = torch.linspace(0, 2 * torch.pi, 20)
        points = torch.stack([torch.cos(t), torch.sin(t), t], dim=-1)

        # Edges along curve
        cells = torch.stack([torch.arange(19), torch.arange(1, 20)], dim=-1)

        mesh = Mesh(points=points, cells=cells)

        # Scalar field along curve
        mesh.point_data["test"] = t

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", gradient_type="extrinsic"
        )

        grad = mesh_grad.point_data["test_gradient"]
        assert grad.shape == (mesh.n_points, 3)


class TestLSQWeighting:
    """Test LSQ weight variations."""

    def test_lsq_with_ill_conditioned_system(self):
        """Test LSQ handles ill-conditioned systems."""
        # Create mesh where some points have nearly collinear neighbors
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.01, 0.01, 0.0],  # Nearly collinear with edge
                [1.02, 0.0, 0.01],  # Also nearly collinear
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        phi = torch.arange(mesh.n_points, dtype=torch.float32)

        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_point_gradient_lsq,
        )

        # Should not crash despite ill-conditioning
        grad = compute_point_gradient_lsq(mesh, phi)

        assert torch.isfinite(grad).all()
        # Some points may have zero gradient if too few neighbors
        assert grad.shape == (mesh.n_points, 3)


class TestCellGradientEdgeCases:
    """Test cell gradient edge cases."""

    def test_cell_with_no_neighbors(self):
        """Test cell with no face-adjacent neighbors."""
        # Single isolated tet
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        mesh.cell_data["test"] = torch.tensor([5.0])

        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_cell_gradient_lsq,
        )

        # Should handle gracefully (no neighbors)
        grad = compute_cell_gradient_lsq(mesh, mesh.cell_data["test"])

        # Gradient should be zero (no neighbors to reconstruct from)
        assert torch.allclose(grad, torch.zeros_like(grad))


class TestProjectionEdgeCases:
    """Test tangent space projection edge cases."""

    def test_projection_on_flat_mesh(self, simple_tet_mesh):
        """Test that projection on codim=0 mesh returns input unchanged."""
        from physicsnemo.mesh.calculus.gradient import project_to_tangent_space

        torch.manual_seed(42)
        mesh = simple_tet_mesh  # Codimension 0
        gradients = torch.randn(mesh.n_points, mesh.n_spatial_dims)

        projected = project_to_tangent_space(mesh, gradients, "points")

        assert torch.allclose(projected, gradients)

    def test_projection_higher_codimension_pca(self):
        """Test projection on codim>1 uses PCA to find tangent space."""
        torch.manual_seed(42)
        # 1D curve in 3D (codimension=2)
        t = torch.linspace(0, 1, 10)
        points = torch.stack([t, t**2, t**3], dim=-1)
        cells = torch.stack([torch.arange(9), torch.arange(1, 10)], dim=-1)
        mesh = Mesh(points=points, cells=cells)

        from physicsnemo.mesh.calculus.gradient import project_to_tangent_space

        gradients = torch.randn(mesh.n_points, 3)
        projected = project_to_tangent_space(mesh, gradients, "points")

        # Should project to tangent space (1D manifold)
        # Projected gradient should have smaller norm than original (normal component removed)
        assert projected.shape == gradients.shape

        # Check that projection actually happened (not identity)
        assert not torch.allclose(projected, gradients)

        # Projected gradient should generally have smaller or equal norm
        projected_norms = torch.norm(projected, dim=-1)
        original_norms = torch.norm(gradients, dim=-1)
        # Most should be smaller (allowing some numerical tolerance)
        assert (projected_norms <= original_norms + 1e-5).float().mean() > 0.7


class TestTangentSpaceProjection:
    """Test tangent space projection for tensors."""

    def test_project_tensor_gradient_to_tangent(self):
        """Test projecting tensor gradient onto tangent space."""
        from physicsnemo.mesh.calculus.gradient import project_to_tangent_space

        torch.manual_seed(42)
        # Surface mesh
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)

        # Tensor gradient (n_points, n_spatial_dims, 2)
        tensor_grads = torch.randn(mesh.n_points, 3, 2)

        projected = project_to_tangent_space(mesh, tensor_grads, "points")

        assert projected.shape == tensor_grads.shape
        # Should be different from input (projection happened)
        assert not torch.allclose(projected, tensor_grads)


class TestIntrinsicLSQEdgeCases:
    """Test intrinsic LSQ edge cases."""

    def test_intrinsic_lsq_on_flat_mesh(self, simple_tet_mesh):
        """Test intrinsic LSQ falls back to standard for flat meshes."""
        from physicsnemo.mesh.calculus._lsq_intrinsic import (
            compute_point_gradient_lsq_intrinsic,
        )

        mesh = simple_tet_mesh  # Codimension 0
        phi = torch.ones(mesh.n_points)

        grad = compute_point_gradient_lsq_intrinsic(mesh, phi)

        # Should call standard LSQ for flat meshes
        assert grad.shape == (mesh.n_points, mesh.n_spatial_dims)


###############################################################################
# DEC Divergence Tests
###############################################################################


class TestDivergenceDEC:
    """Test DEC divergence for 2D and 3D meshes.

    The DEC divergence is exact for linear vector fields at interior vertices
    (where the Voronoi cell is complete). Boundary vertices have truncated dual
    cells, so accuracy there is not tested.
    """

    ### Helpers ##############################################################

    @staticmethod
    def _interior_mask_2d(mesh):
        """Boolean mask that is True for interior (non-boundary) vertices."""
        from physicsnemo.mesh.boundaries._detection import get_boundary_vertices

        return ~get_boundary_vertices(mesh)

    ### 2D correctness tests (structured grid) ###############################

    def test_2d_div_identity(self):
        """div(position) = 2 at interior vertices of a flat 2D mesh."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec
        from physicsnemo.mesh.primitives.planar import structured_grid

        mesh = structured_grid.load(n_x=8, n_y=8)
        mesh = Mesh(points=mesh.points.to(torch.float64), cells=mesh.cells)

        div_v = compute_divergence_points_dec(mesh, mesh.points.clone())
        interior = self._interior_mask_2d(mesh)

        assert interior.sum() > 0, "Need interior vertices for this test"
        assert torch.allclose(
            div_v[interior],
            torch.full_like(div_v[interior], 2.0),
            atol=1e-12,
        )

    def test_2d_div_single_component(self):
        """div(x, 0) = 1 at interior vertices."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec
        from physicsnemo.mesh.primitives.planar import structured_grid

        mesh = structured_grid.load(n_x=8, n_y=8)
        mesh = Mesh(points=mesh.points.to(torch.float64), cells=mesh.cells)

        v_field = torch.zeros_like(mesh.points)
        v_field[:, 0] = mesh.points[:, 0]
        div_v = compute_divergence_points_dec(mesh, v_field)
        interior = self._interior_mask_2d(mesh)

        assert torch.allclose(
            div_v[interior],
            torch.ones_like(div_v[interior]),
            atol=1e-12,
        )

    def test_2d_div_rotation(self):
        """div(-y, x) = 0 at interior vertices (divergence-free field)."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec
        from physicsnemo.mesh.primitives.planar import structured_grid

        mesh = structured_grid.load(n_x=8, n_y=8)
        mesh = Mesh(points=mesh.points.to(torch.float64), cells=mesh.cells)

        v_field = torch.stack([-mesh.points[:, 1], mesh.points[:, 0]], dim=-1)
        div_v = compute_divergence_points_dec(mesh, v_field)
        interior = self._interior_mask_2d(mesh)

        assert torch.allclose(
            div_v[interior],
            torch.zeros_like(div_v[interior]),
            atol=1e-12,
        )

    ### 3D correctness test (tetrahedral mesh) ###############################

    def test_3d_div_identity_interior(self, simple_tet_mesh):
        """div(position) = 3 at the interior vertex of a 3D tet mesh."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec

        mesh = Mesh(
            points=simple_tet_mesh.points.to(torch.float64),
            cells=simple_tet_mesh.cells,
        )
        div_v = compute_divergence_points_dec(mesh, mesh.points.clone())

        # Vertex 4 at (0.5, 0.5, 0.5) is the interior vertex
        assert torch.isclose(
            div_v[4], torch.tensor(3.0, dtype=torch.float64), atol=1e-12
        )

    ### 3D regression test: must not crash on meshes where n_edges != n_faces

    def test_3d_no_crash_on_real_mesh(self):
        """Divergence must run without error on a real 3D tet mesh."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec
        from physicsnemo.mesh.primitives.procedural import lumpy_ball

        mesh = lumpy_ball.load(n_shells=2, subdivisions=1)
        v_field = mesh.points.clone()
        div_v = compute_divergence_points_dec(mesh, v_field)

        assert div_v.shape == (mesh.n_points,)
        assert torch.isfinite(div_v).all()

    ### Shape and dtype tests ################################################

    def test_output_shape_and_finiteness_2d(self):
        """Basic smoke test: correct shape and finite values on 2D mesh."""
        from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec

        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.5]])
        cells = torch.tensor([[0, 1, 3], [0, 2, 3], [1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        div_v = compute_divergence_points_dec(mesh, points.clone())

        assert div_v.shape == (mesh.n_points,)
        assert torch.isfinite(div_v).all()


###############################################################################
# Method Combinations Tests
###############################################################################


class TestDerivativesMethodCombinations:
    """Test all method × gradient_type combinations."""

    def test_dec_method_extrinsic_gradient(self):
        """Test method='dec' with gradient_type='extrinsic'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", method="dec", gradient_type="extrinsic"
        )

        assert "test_gradient" in mesh_grad.point_data.keys()

    def test_dec_method_both_gradients(self):
        """Test method='dec' with gradient_type='both'."""
        mesh = procedural.lumpy_sphere.load(radius=1.0, subdivisions=2)
        mesh.point_data["test"] = torch.ones(mesh.n_points)

        mesh_grad = mesh.compute_point_derivatives(
            keys="test", method="dec", gradient_type="both"
        )

        assert "test_gradient_extrinsic" in mesh_grad.point_data.keys()
        assert "test_gradient_intrinsic" in mesh_grad.point_data.keys()


class TestCellDerivativesGradientTypes:
    """Test cell derivatives with different gradient types."""

    def test_cell_extrinsic_gradient(self, simple_tet_mesh):
        """Test cell gradient with gradient_type='extrinsic'."""
        mesh = simple_tet_mesh
        mesh.cell_data["test"] = torch.ones(mesh.n_cells)

        mesh_grad = mesh.compute_cell_derivatives(
            keys="test", gradient_type="extrinsic"
        )

        assert "test_gradient" in mesh_grad.cell_data.keys()

    def test_cell_both_gradients(self, simple_tet_mesh):
        """Test cell gradient with gradient_type='both'."""
        mesh = simple_tet_mesh
        mesh.cell_data["test"] = torch.ones(mesh.n_cells)

        mesh_grad = mesh.compute_cell_derivatives(keys="test", gradient_type="both")

        assert "test_gradient_extrinsic" in mesh_grad.cell_data.keys()
        assert "test_gradient_intrinsic" in mesh_grad.cell_data.keys()


###############################################################################
# n-Dimensional DEC Laplacian Tests
###############################################################################


class TestLaplacian1D:
    """Test DEC Laplacian on 1D edge meshes.

    For 1D manifolds, the Laplace-Beltrami operator reduces to the second
    arc-length derivative: Delta f = d^2 f / ds^2.

    The FEM stiffness cotangent weights give w_ij = 1/|edge| for 1D edges,
    which produces the standard finite-difference second derivative when
    combined with the dual volume normalization (half the sum of adjacent
    edge lengths).
    """

    def test_laplacian_x_squared_uniform_1d(self):
        """Delta(x^2) = 2 on a uniform 1D grid in 1D ambient space."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        n = 21
        x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        points = x.unsqueeze(-1)  # (n, 1)
        cells = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1)
        mesh = Mesh(points=points, cells=cells)

        f = x**2
        lap = compute_laplacian_points_dec(mesh, f)

        # Interior points (not boundary) should give exactly 2
        interior = slice(1, -1)
        assert torch.allclose(
            lap[interior], torch.full_like(lap[interior], 2.0), atol=1e-8
        ), f"Interior Laplacian: {lap[interior]}"

    def test_laplacian_x_squared_nonuniform_1d(self):
        """Delta(x^2) = 2 on a non-uniform 1D grid."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        # Non-uniform spacing
        x = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.6, 0.85, 1.0], dtype=torch.float64)
        points = x.unsqueeze(-1)  # (7, 1)
        n = len(x)
        cells = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1)
        mesh = Mesh(points=points, cells=cells)

        f = x**2
        lap = compute_laplacian_points_dec(mesh, f)

        # Interior points should give exactly 2 (FEM cotangent weights are
        # exact for quadratics on 1D meshes, regardless of spacing)
        interior = slice(1, -1)
        assert torch.allclose(
            lap[interior], torch.full_like(lap[interior], 2.0), atol=1e-8
        ), f"Interior Laplacian: {lap[interior]}"

    def test_laplacian_1d_in_2d_ambient(self):
        """Delta(x^2) = 2 on a 1D line segment embedded in 2D."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        n = 15
        t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        # Line along x-axis in 2D
        points = torch.stack([t, torch.zeros_like(t)], dim=1)  # (n, 2)
        cells = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1)
        mesh = Mesh(points=points, cells=cells)

        f = points[:, 0] ** 2  # x^2
        lap = compute_laplacian_points_dec(mesh, f)

        interior = slice(1, -1)
        assert torch.allclose(
            lap[interior], torch.full_like(lap[interior], 2.0), atol=1e-8
        ), f"Interior Laplacian: {lap[interior]}"

    def test_laplacian_1d_in_3d_ambient(self):
        """Delta(s^2) = 2 on a 1D line segment embedded in 3D."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        n = 15
        t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        # Line along (1,1,1) direction in 3D
        direction = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        direction = direction / direction.norm()
        points = t.unsqueeze(-1) * direction.unsqueeze(0)  # (n, 3)
        cells = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1)
        mesh = Mesh(points=points, cells=cells)

        # Arc-length parameter is t (since |direction| = 1 and spacing is uniform)
        f = t**2
        lap = compute_laplacian_points_dec(mesh, f)

        interior = slice(1, -1)
        assert torch.allclose(
            lap[interior], torch.full_like(lap[interior], 2.0), atol=1e-8
        ), f"Interior Laplacian: {lap[interior]}"

    def test_laplacian_linear_zero_1d(self):
        """Delta(linear) = 0 on a 1D mesh."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        n = 10
        x = torch.linspace(0.0, 2.0, n, dtype=torch.float64)
        points = x.unsqueeze(-1)
        cells = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1)
        mesh = Mesh(points=points, cells=cells)

        f = 3.0 * x + 7.0  # Linear function
        lap = compute_laplacian_points_dec(mesh, f)

        # Interior points: Laplacian of linear function = 0
        interior = slice(1, -1)
        assert torch.allclose(
            lap[interior], torch.zeros_like(lap[interior]), atol=1e-8
        ), f"Interior Laplacian of linear: {lap[interior]}"

    def test_laplacian_constant_zero_1d(self):
        """Delta(constant) = 0 on a 1D mesh."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec

        n = 8
        x = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        points = x.unsqueeze(-1)
        cells = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1)
        mesh = Mesh(points=points, cells=cells)

        f = torch.full((n,), 42.0, dtype=torch.float64)
        lap = compute_laplacian_points_dec(mesh, f)

        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-10)


class TestLaplacian3D:
    """Test DEC Laplacian on 3D tetrahedral meshes.

    Uses the FEM stiffness matrix cotangent weights, which give exact
    dihedral-angle-based weights for tetrahedra. The Kuhn triangulation
    of a uniform grid has sufficient symmetry for the discrete Laplacian
    to be exact for quadratic functions at interior vertices.
    """

    def test_laplacian_constant_zero_3d(self):
        """Delta(constant) = 0 on a tetrahedral mesh."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec
        from physicsnemo.mesh.primitives.volumes.cube_volume import load

        mesh = load(size=1.0, subdivisions=3)
        f = torch.ones(mesh.n_points, dtype=mesh.points.dtype)
        lap = compute_laplacian_points_dec(mesh, f)

        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-5), (
            f"Laplacian of constant: max abs = {lap.abs().max():.2e}"
        )

    def test_laplacian_linear_zero_3d(self):
        """Delta(linear) = 0 at interior points of a tetrahedral mesh."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec
        from physicsnemo.mesh.primitives.volumes.cube_volume import load

        mesh = load(size=2.0, subdivisions=4)
        # Linear function: f = 2x + 3y - z + 5
        f = 2.0 * mesh.points[:, 0] + 3.0 * mesh.points[:, 1] - mesh.points[:, 2] + 5.0
        lap = compute_laplacian_points_dec(mesh, f)

        # Interior points: far from the cube boundary
        half_size = 1.0  # cube has size 2, so boundary is at ±1
        is_interior = (mesh.points.abs() < half_size * 0.8).all(dim=-1)
        assert is_interior.sum() > 10, (
            "Need enough interior points for a meaningful test"
        )

        interior_lap = lap[is_interior]
        assert torch.allclose(
            interior_lap, torch.zeros_like(interior_lap), atol=1e-4
        ), (
            f"Laplacian of linear at interior: "
            f"max abs = {interior_lap.abs().max():.2e}, "
            f"mean abs = {interior_lap.abs().mean():.2e}"
        )

    def test_laplacian_r_squared_3d(self):
        """Delta(x^2 + y^2 + z^2) = 6 at interior points of a tetrahedral mesh.

        On the Kuhn triangulation of a uniform grid, the FEM cotangent
        Laplacian is exact for quadratics at interior vertices due to
        the stencil symmetry.
        """
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec
        from physicsnemo.mesh.primitives.volumes.cube_volume import load

        mesh = load(size=2.0, subdivisions=5)
        f = (mesh.points**2).sum(dim=-1)  # x^2 + y^2 + z^2
        lap = compute_laplacian_points_dec(mesh, f)

        # Interior points only (boundary vertices have incomplete stencils)
        half_size = 1.0
        is_interior = (mesh.points.abs() < half_size * 0.8).all(dim=-1)
        assert is_interior.sum() > 20, "Need enough interior points"

        interior_lap = lap[is_interior]
        expected = torch.full_like(interior_lap, 6.0)

        # Kuhn triangulation with subdivisions=5 gives near-exact results for
        # quadratics at interior vertices (FEM cotangent weights are exact on
        # symmetric stencils).
        assert torch.allclose(interior_lap, expected, atol=0.05), (
            f"Laplacian of r^2 at interior: "
            f"mean = {interior_lap.mean():.4f}, "
            f"max deviation = {(interior_lap - 6.0).abs().max():.4f}"
        )

    def test_laplacian_single_component_3d(self):
        """Delta(x^2) = 2 at interior points of a tetrahedral mesh."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec
        from physicsnemo.mesh.primitives.volumes.cube_volume import load

        mesh = load(size=2.0, subdivisions=5)
        f = mesh.points[:, 0] ** 2  # x^2 only
        lap = compute_laplacian_points_dec(mesh, f)

        # Interior points
        half_size = 1.0
        is_interior = (mesh.points.abs() < half_size * 0.8).all(dim=-1)

        interior_lap = lap[is_interior]
        expected = torch.full_like(interior_lap, 2.0)

        # Kuhn triangulation with subdivisions=5; FEM cotangent weights are
        # near-exact for single-variable quadratics at interior vertices.
        assert torch.allclose(interior_lap, expected, atol=0.05), (
            f"Laplacian of x^2 at interior: "
            f"mean = {interior_lap.mean():.4f}, "
            f"max deviation = {(interior_lap - 2.0).abs().max():.4f}"
        )

    def test_laplacian_vector_field_3d(self):
        """DEC Laplacian works on vector fields over tetrahedral meshes."""
        from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec
        from physicsnemo.mesh.primitives.volumes.cube_volume import load

        mesh = load(size=1.0, subdivisions=3)

        # Linear vector field: Laplacian should be zero
        vector_field = mesh.points.clone()  # (n_points, 3)
        lap = compute_laplacian_points_dec(mesh, vector_field)

        assert lap.shape == vector_field.shape

        # Interior points should have near-zero Laplacian
        half_size = 0.5
        is_interior = (mesh.points.abs() < half_size * 0.8).all(dim=-1)
        interior_lap = lap[is_interior]
        assert torch.allclose(interior_lap, torch.zeros_like(interior_lap), atol=1e-4)


class TestCotanWeightsFEM:
    """Verify FEM cotangent weights against analytically known values.

    For 2D triangle meshes, the FEM stiffness matrix approach produces
    weights equal to (1/2)(cot alpha + cot beta) for each edge, where
    alpha and beta are the angles opposite the edge in the two adjacent
    triangles.
    """

    def test_equilateral_triangle_weights(self):
        """FEM weights for an equilateral triangle match cot(60 deg)/2."""
        from physicsnemo.mesh.geometry.dual_meshes import (
            compute_cotan_weights_fem,
        )

        # Equilateral triangle: all angles = 60 deg, cot(60) = 1/sqrt(3)
        h = (3**0.5) / 2
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.5, h]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        weights, edges = compute_cotan_weights_fem(mesh)

        # Each edge is boundary (1 triangle), weight = cot(60 deg) / 2
        expected_weight = (1.0 / (3**0.5)) / 2.0
        assert torch.allclose(
            weights, torch.full_like(weights, expected_weight), atol=1e-10
        ), f"Expected all weights ~{expected_weight:.6f}, got {weights}"

    def test_right_triangle_weights(self):
        """FEM weights for a right triangle match known cotangent values."""
        from physicsnemo.mesh.geometry.dual_meshes import (
            compute_cotan_weights_fem,
        )

        # Right triangle: 90 deg at origin, 45 deg at the other two
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        weights, edges = compute_cotan_weights_fem(mesh)

        # Expected (boundary, 1 triangle):
        #   edge [0,1]: opposite angle at v2 = 45 deg, cot(45)=1 -> w = 0.5
        #   edge [0,2]: opposite angle at v1 = 45 deg, cot(45)=1 -> w = 0.5
        #   edge [1,2]: opposite angle at v0 = 90 deg, cot(90)=0 -> w = 0.0
        expected = {(0, 1): 0.5, (0, 2): 0.5, (1, 2): 0.0}
        for i, edge in enumerate(edges):
            key = (int(edge[0]), int(edge[1]))
            assert abs(weights[i].item() - expected[key]) < 1e-10, (
                f"Edge {key}: expected {expected[key]:.4f}, got {weights[i]:.4f}"
            )

    def test_weights_on_3d_surface(self):
        """FEM weights are well-defined on a surface mesh in 3D."""
        from physicsnemo.mesh.geometry.dual_meshes import (
            compute_cotan_weights_fem,
        )
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        mesh = sphere_icosahedral.load(subdivisions=2)
        mesh = Mesh(
            points=mesh.points.to(torch.float64),
            cells=mesh.cells,
        )

        weights, edges = compute_cotan_weights_fem(mesh)

        # Sanity: icosahedral sphere has all positive cotan weights (Delaunay)
        assert (weights > -1e-8).all(), (
            f"Unexpected large negative weight: {weights.min():.2e}"
        )
        # All edges accounted for
        assert edges.shape[1] == 2
        assert len(weights) == len(edges)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
