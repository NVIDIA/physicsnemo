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

"""Tests for PCA-based tangent space estimation."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus._pca_tangent import (
    estimate_tangent_space_pca,
    project_gradient_to_tangent_space_pca,
)


@pytest.fixture
def device():
    """Test on CPU."""
    return "cpu"


class TestPCATangentSpace:
    """Tests for PCA tangent space estimation."""

    def test_curve_in_3d_tangent_space(self, device):
        """Test tangent space estimation for curve in 3D."""
        # Straight line in 3D
        t = torch.linspace(0, 1, 10, device=device)
        points = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)

        cells = torch.stack(
            [torch.arange(9, device=device), torch.arange(1, 10, device=device)], dim=-1
        )

        mesh = Mesh(points=points, cells=cells)

        tangent_basis, normal_basis = estimate_tangent_space_pca(mesh)

        # Should have shape (n_points, 1, 3) for tangent
        assert tangent_basis.shape == (10, 1, 3)
        assert normal_basis.shape == (10, 2, 3)

        # Tangent should align with x-axis for straight line
        # (up to sign and for interior points with enough neighbors)
        # Check middle points
        for i in range(2, 8):
            tangent = tangent_basis[i, 0]
            # Should be primarily in x direction
            assert torch.abs(tangent[0]) > 0.9  # Mostly x-component

    def test_circle_in_3d_tangent_space(self, device):
        """Test tangent space for circle in 3D."""
        n = 20
        theta = torch.linspace(0, 2 * torch.pi, n + 1, device=device)[:-1]

        # Circle in xy-plane
        points = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                torch.zeros_like(theta),
            ],
            dim=-1,
        )

        # Closed loop
        cells = torch.stack(
            [
                torch.arange(n, device=device),
                torch.roll(torch.arange(n, device=device), -1),
            ],
            dim=-1,
        )

        mesh = Mesh(points=points, cells=cells)

        tangent_basis, normal_basis = estimate_tangent_space_pca(mesh)

        # Tangent space is 1D (curve)
        assert tangent_basis.shape == (n, 1, 3)
        assert normal_basis.shape == (n, 2, 3)

        # Tangents should be unit vectors
        tangent_norms = torch.norm(tangent_basis, dim=-1)
        assert torch.allclose(tangent_norms, torch.ones_like(tangent_norms), atol=1e-5)

        # Normal space should span the perpendicular plane
        # For circle in xy-plane, one normal should be mostly z
        for i in range(n):
            normals = normal_basis[i]  # (2, 3)
            # Check that at least one normal has significant z-component
            z_components = torch.abs(normals[:, 2])
            assert z_components.max() > 0.7  # One normal ~aligned with z

    def test_helix_tangent_space(self, device):
        """Test tangent space for helix (curve in 3D)."""
        t = torch.linspace(0, 4 * torch.pi, 50, device=device)

        # Helix
        points = torch.stack(
            [
                torch.cos(t),
                torch.sin(t),
                0.1 * t,  # Pitch
            ],
            dim=-1,
        )

        cells = torch.stack(
            [
                torch.arange(49, device=device),
                torch.arange(1, 50, device=device),
            ],
            dim=-1,
        )

        mesh = Mesh(points=points, cells=cells)

        tangent_basis, normal_basis = estimate_tangent_space_pca(mesh)

        # Shapes
        assert tangent_basis.shape == (50, 1, 3)
        assert normal_basis.shape == (50, 2, 3)

        # All tangents should be unit vectors
        tangent_norms = torch.norm(tangent_basis, dim=-1)
        assert torch.allclose(tangent_norms, torch.ones_like(tangent_norms), atol=1e-4)

    def test_surface_in_3d_tangent_space(self, device):
        """Test PCA tangent space estimation works for surface in 3D (codimension-1).

        Note: While codimension-1 has more efficient normal-based methods,
        the PCA method should still work correctly for these cases.
        """

        ### Create an equilateral triangle in the XY plane
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        mesh = Mesh(points=points, cells=cells)

        ### Estimate tangent space using PCA
        tangent_basis, normal_basis = estimate_tangent_space_pca(mesh, k_neighbors=2)

        ### Verify shapes
        assert tangent_basis.shape == (
            3,
            2,
            3,
        )  # (n_points, n_manifold_dims, n_spatial_dims)
        assert normal_basis.shape == (
            3,
            1,
            3,
        )  # (n_points, codimension, n_spatial_dims)

        ### For a triangle in the XY plane:
        # Tangent space should span XY directions
        # Normal space should point in Z direction

        ### Check that tangent vectors are orthogonal to normal
        for i in range(3):
            for j in range(2):
                tangent_vec = tangent_basis[i, j]
                normal_vec = normal_basis[i, 0]

                # Tangent and normal should be orthogonal
                dot_product = torch.dot(tangent_vec, normal_vec)
                assert torch.abs(dot_product) < 1e-4, (
                    f"Tangent {j} at point {i} not orthogonal to normal"
                )

        ### Check that normal points primarily in Z direction (since triangle is in XY plane)
        for i in range(3):
            normal_vec = normal_basis[i, 0]
            # Z component should dominate
            assert torch.abs(normal_vec[2]) > 0.9, (
                f"Normal at point {i} should point in Z direction"
            )

        ### Check that tangent vectors are unit length
        tangent_norms = torch.norm(tangent_basis, dim=-1)
        assert torch.allclose(tangent_norms, torch.ones_like(tangent_norms), atol=1e-4)


class TestGradientProjection:
    """Tests for gradient projection to tangent space."""

    def test_project_gradient_curve_3d(self, device):
        """Test projecting gradient onto curve tangent space."""
        t = torch.linspace(0, 1, 10, device=device)
        points = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)

        cells = torch.stack(
            [
                torch.arange(9, device=device),
                torch.arange(1, 10, device=device),
            ],
            dim=-1,
        )

        mesh = Mesh(points=points, cells=cells)

        # Random 3D gradient
        torch.manual_seed(42)
        gradient = torch.randn(10, 3, device=device)

        # Project to tangent space
        projected = project_gradient_to_tangent_space_pca(mesh, gradient)

        # Should have same shape
        assert projected.shape == gradient.shape

        # Projected gradient should have smaller or equal norm
        proj_norms = torch.norm(projected, dim=-1)
        orig_norms = torch.norm(gradient, dim=-1)

        # All projections should not increase norm
        assert torch.all(proj_norms <= orig_norms + 1e-5)

    def test_project_gradient_reduces_norm(self, device):
        """Test that projection removes normal component."""
        # Circle in xy-plane
        n = 20
        theta = torch.linspace(0, 2 * torch.pi, n + 1, device=device)[:-1]

        points = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                torch.zeros_like(theta),
            ],
            dim=-1,
        )

        cells = torch.stack(
            [
                torch.arange(n, device=device),
                torch.roll(torch.arange(n, device=device), -1),
            ],
            dim=-1,
        )

        mesh = Mesh(points=points, cells=cells)

        # Gradient with z-component (perpendicular to circle)
        gradient = torch.randn(n, 3, device=device)
        gradient[:, 2] = 1.0  # Add significant z-component

        # Project
        projected = project_gradient_to_tangent_space_pca(mesh, gradient)

        # Z-component should be significantly reduced for interior points
        # (boundary points may not have enough neighbors)
        for i in range(5, 15):  # Check interior points
            assert torch.abs(projected[i, 2]) < torch.abs(gradient[i, 2])

    def test_projection_orthogonality(self, device):
        """Test that projected gradient is orthogonal to normal space."""
        t = torch.linspace(0, 1, 10, device=device)
        points = torch.stack([t, t**2, torch.zeros_like(t)], dim=-1)

        cells = torch.stack(
            [
                torch.arange(9, device=device),
                torch.arange(1, 10, device=device),
            ],
            dim=-1,
        )

        mesh = Mesh(points=points, cells=cells)

        # Get tangent and normal bases
        tangent_basis, normal_basis = estimate_tangent_space_pca(mesh)

        # Check orthogonality: tangent · normal ≈ 0
        for i in range(1, 9):  # Interior points
            tangent = tangent_basis[i, 0]  # (3,)
            normals = normal_basis[i]  # (2, 3)

            # Dot products should be near zero
            dots = torch.abs((normals @ tangent))
            assert torch.all(dots < 0.1)  # Should be nearly orthogonal


class TestPCAEdgeCases:
    """Edge case tests for PCA tangent space."""

    def test_insufficient_neighbors(self, device):
        """Test PCA with insufficient neighbors."""
        # Single edge (points have at most 1 neighbor)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Should handle gracefully (may use fallback)
        tangent_basis, normal_basis = estimate_tangent_space_pca(mesh)

        assert tangent_basis.shape == (2, 1, 3)
        assert normal_basis.shape == (2, 2, 3)

    def test_degenerate_neighborhood(self, device):
        """Test PCA with degenerate neighborhood (collinear neighbors)."""
        # Points all along x-axis
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        # Should compute tangent space even for degenerate case
        tangent_basis, normal_basis = estimate_tangent_space_pca(mesh)

        assert not torch.any(torch.isnan(tangent_basis))
        assert not torch.any(torch.isnan(normal_basis))
