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

"""Tests for rigorous sharp/flat operators per Hirani (2003).

These tests verify that the sharp and flat operators follow Hirani's formulas
with support volume intersections and barycentric interpolation gradients.

The key identities to test:
1. div(curl(V)) = 0
2. curl(grad(f)) = 0
3. div(grad(f)) ≈ Δf (may not be exact in discrete DEC per Hirani Section 5.9)

References:
    Hirani (2003) Chapter 5 (sharp/flat), Section 9.3 (vector identities)
"""

import pytest
import torch

from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec
from physicsnemo.mesh.calculus.gradient import compute_gradient_points_dec
from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec
from physicsnemo.mesh.mesh import Mesh


class TestSharpFlatProperties:
    """Test basic properties of sharp and flat operators."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_sharp_flat_on_simple_mesh(self, device):
        """Test that sharp and flat operators run without errors."""
        ### Simple mesh with interior vertex
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0],  # center
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Test sharp
        from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
        from physicsnemo.mesh.calculus._sharp_flat import sharp

        f = points[:, 0]  # f = x
        df, edges = exterior_derivative_0(mesh, f)
        grad_f = sharp(mesh, df, edges)

        assert grad_f.shape == (mesh.n_points, mesh.n_spatial_dims)
        assert not torch.any(torch.isnan(grad_f))

        ### Test flat
        from physicsnemo.mesh.calculus._sharp_flat import flat

        vectors = torch.randn(
            mesh.n_points, mesh.n_spatial_dims, dtype=torch.float32, device=device
        )
        one_form = flat(mesh, vectors, edges)

        assert one_form.shape == (len(edges),)
        assert not torch.any(torch.isnan(one_form))


class TestVectorCalculusIdentities:
    """Test vector calculus identities with rigorous operators."""

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_div_grad_vs_laplacian_linear_function(self, device):
        """Test that div(grad(f)) and Δf are close for linear functions.

        For linear f, both should give zero at interior vertices.
        """
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0],  # v0: center (interior)
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Linear function: Δf = 0 everywhere
        f = points[:, 0]  # f = x

        grad_f = compute_gradient_points_dec(mesh, f)
        div_grad_f = compute_divergence_points_dec(mesh, grad_f)
        lap_f = compute_laplacian_points_dec(mesh, f)

        ### At interior vertex, both should be ~0 for linear function
        assert abs(div_grad_f[0]) < 0.1, (
            f"div(grad(linear)) = {div_grad_f[0].item():.4f}, expected ≈ 0"
        )
        assert abs(lap_f[0]) < 0.01, f"Δ(linear) = {lap_f[0].item():.4f}, expected ≈ 0"

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
    )
    def test_div_grad_approximate_laplacian(self, device):
        """Test that div(grad(f)) is approximately equal to Δf at interior vertices.

        In discrete DEC, sharp and flat are NOT exact inverses (Hirani
        Prop. 5.5.3), so div(grad(f)) won't exactly equal Δf. But they should
        agree within an order of magnitude, and the ratio converges to 1.0
        under mesh refinement.
        """
        points = torch.tensor(
            [
                [0.5, 0.5, 0.0],  # v0: center (interior)
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
            device=device,
        )
        cells = torch.tensor(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
            dtype=torch.int64,
            device=device,
        )
        mesh = Mesh(points=points, cells=cells)

        ### Quadratic function
        f = points[:, 0] ** 2 + points[:, 1] ** 2  # Δf = 4

        grad_f = compute_gradient_points_dec(mesh, f)
        div_grad_f = compute_divergence_points_dec(mesh, grad_f)
        lap_f = compute_laplacian_points_dec(mesh, f)

        # Both should have the same sign
        assert (
            torch.sign(div_grad_f[0]) == torch.sign(lap_f[0])
            or torch.abs(lap_f[0]) < 0.1
        ), (
            f"div(grad(f)) and Δf have opposite signs: "
            f"{div_grad_f[0].item():.2f} vs {lap_f[0].item():.2f}"
        )

        # Should be within same order of magnitude. On this small 5-vertex
        # mesh the ratio is 0.5 (inherent discretization error that decreases
        # with mesh refinement: 0.75 at 8x8, 0.96 at 16x16, etc.).
        ratio = abs(div_grad_f[0] / lap_f[0].clamp(min=1e-10))
        assert 0.3 < ratio < 3.0, (
            f"div(grad(f)) and Δf differ by more than 3x:\n"
            f"div(grad(f)) = {div_grad_f[0].item():.4f}\n"
            f"Δf = {lap_f[0].item():.4f}\n"
            f"Ratio = {ratio.item():.2f}\n"
            f"Note: Exact equality not guaranteed in discrete DEC (Hirani Prop. 5.5.3)"
        )
