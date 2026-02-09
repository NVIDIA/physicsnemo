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
#
# This file contains code derived from `fairchem` found at
# https://github.com/facebookresearch/fairchem.
# Copyright (c) [2025] Meta, Inc. and its affiliates.
# Licensed under MIT License.

r"""Test suite for EdgeRotation module.

Tests the EdgeRotation class for computing Wigner D-matrices from edge
direction vectors, which is used for rotating spherical harmonic embeddings
to edge-aligned local frames in equivariant networks.

Testing scope includes a bunch of hard-coded regression tests,
both against serialized `sympy` results (in `wigner_reference`),
as well as properties (e.g. orthogonality) in addition to the
usual `nn.Module` tests like shapes and whatnot.
"""

import math

import pytest
import torch

from physicsnemo.experimental.nn.symmetry.wigner import (
    EdgeRotation,
    _compute_d_matrix,
    _compute_J_matrix,
    edge_vectors_to_euler_angles,
)

# Import shared test utilities from conftest
from test.experimental.nn.symmetry.conftest import get_rtol_atol, is_half_precision
from test.experimental.nn.symmetry.wigner_reference import (
    D_MATRICES_BETA_0,
    D_MATRICES_BETA_PI,
    D_MATRICES_BETA_PI_2,
    J_MATRICES,
    verify_d_matrix_orthogonality,
    verify_d_matrix_symmetry,
    verify_j_matrix_from_d,
    verify_j_matrix_involution,
)


def compute_expected_dims(lmax: int, mmax: int) -> tuple:
    """Compute expected (full_dim, reduced_dim) for given lmax and mmax.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum degree.
    mmax : int
        Maximum azimuthal quantum number (m-truncation).

    Returns
    -------
    tuple
        (full_dim, reduced_dim) where:
        - full_dim = (lmax + 1)^2
        - reduced_dim = sum over l of min(2*mmax + 1, 2*l + 1)
    """
    full_dim = (lmax + 1) ** 2
    reduced_dim = sum(min(2 * mmax + 1, 2 * ell + 1) for ell in range(lmax + 1))
    return full_dim, reduced_dim


# Note: dtype, device, and lmax_mmax fixtures are provided by conftest.py


# =============================================================================
# Tests for Internal Functions
# =============================================================================


class TestWignerInternalFunctions:
    """Test suite for internal Wigner d-matrix and J-matrix functions.

    Note: The `_compute_d_matrix_from_lower` function (used for l >= 3) has
    numerical instabilities at edge cases (beta=0 or beta=pi) due to the
    factorial-based formula involving 0^negative exponents. Tests for l=3
    are limited to beta values where the formula is numerically stable.
    The closed-form implementations for l=0,1,2 are tested at all beta values.
    """

    # =========================================================================
    # _compute_d_matrix tests
    # =========================================================================

    @pytest.mark.parametrize("ell", [0, 1, 2])
    def test_d_matrix_beta_zero(self, ell: int) -> None:
        r"""d-matrix at beta=0 should be identity.

        Note: Only tested for l=0,1,2 which use closed-form formulas.
        The generic formula for l>=3 has numerical issues at beta=0.
        """
        beta = torch.tensor([0.0], dtype=torch.float64)
        d = _compute_d_matrix(ell, beta).squeeze(0)

        expected = D_MATRICES_BETA_0[ell]
        torch.testing.assert_close(
            d,
            expected,
            rtol=1e-10,
            atol=1e-10,
            msg=f"d-matrix at beta=0 for l={ell} does not match identity",
        )

    @pytest.mark.parametrize("ell", [0, 1, 2, 3])
    def test_d_matrix_beta_pi_2(self, ell: int) -> None:
        r"""d-matrix at beta=pi/2 should match SymPy reference.

        This is the critical case for J-matrix computation, tested for all l.
        """
        beta = torch.tensor([math.pi / 2], dtype=torch.float64)
        d = _compute_d_matrix(ell, beta).squeeze(0)

        expected = D_MATRICES_BETA_PI_2[ell]
        torch.testing.assert_close(
            d,
            expected,
            rtol=1e-10,
            atol=1e-10,
            msg=f"d-matrix at beta=pi/2 for l={ell} does not match reference",
        )

    @pytest.mark.parametrize("ell", [0, 1, 2])
    def test_d_matrix_beta_pi(self, ell: int) -> None:
        r"""d-matrix at beta=pi should have anti-diagonal pattern.

        Note: Only tested for l=0,1,2 which use closed-form formulas.
        The generic formula for l>=3 has numerical issues at beta=pi.
        """
        beta = torch.tensor([math.pi], dtype=torch.float64)
        d = _compute_d_matrix(ell, beta).squeeze(0)

        expected = D_MATRICES_BETA_PI[ell]
        torch.testing.assert_close(
            d,
            expected,
            rtol=1e-10,
            atol=1e-10,
            msg=f"d-matrix at beta=pi for l={ell} does not match reference",
        )

    @pytest.mark.parametrize("ell", [0, 1, 2, 3])
    def test_d_matrix_orthogonality(self, ell: int) -> None:
        r"""d-matrix should be orthogonal: D^T @ D = I at beta=pi/2."""
        beta = torch.tensor([math.pi / 2], dtype=torch.float64)
        d = _compute_d_matrix(ell, beta).squeeze(0)

        assert verify_d_matrix_orthogonality(d, tol=1e-10), (
            f"d-matrix for l={ell} is not orthogonal"
        )

    @pytest.mark.parametrize("ell", [0, 1, 2])
    @pytest.mark.parametrize(
        "beta_val", [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi]
    )
    def test_d_matrix_orthogonality_various_beta(
        self, ell: int, beta_val: float
    ) -> None:
        r"""d-matrix should be orthogonal for various beta values.

        Note: Only tested for l=0,1,2 which use closed-form formulas.
        """
        beta = torch.tensor([beta_val], dtype=torch.float64)
        d = _compute_d_matrix(ell, beta).squeeze(0)

        assert verify_d_matrix_orthogonality(d, tol=1e-10), (
            f"d-matrix for l={ell}, beta={beta_val} is not orthogonal"
        )

    @pytest.mark.parametrize("ell", [0, 1, 2])
    def test_d_matrix_symmetry(self, ell: int) -> None:
        r"""d^l_{m,m'}(beta) = (-1)^{m-m'} * d^l_{m',m}(beta).

        Note: Only tested for l=0,1,2 which use closed-form formulas.
        """
        beta = torch.tensor([math.pi / 3], dtype=torch.float64)
        d = _compute_d_matrix(ell, beta).squeeze(0)

        assert verify_d_matrix_symmetry(ell, d, tol=1e-10), (
            f"d-matrix for l={ell} does not satisfy symmetry relation"
        )

    # =========================================================================
    # _compute_J_matrix tests
    # =========================================================================

    @pytest.mark.parametrize("ell", [0, 1, 2, 3])
    def test_j_matrix_values(self, ell: int) -> None:
        r"""J matrix should match SymPy reference."""
        J = _compute_J_matrix(ell, dtype=torch.float64)

        expected = J_MATRICES[ell]
        torch.testing.assert_close(
            J,
            expected,
            rtol=1e-10,
            atol=1e-10,
            msg=f"J matrix for l={ell} does not match reference",
        )

    @pytest.mark.parametrize("ell", [0, 1, 2, 3])
    def test_j_matrix_involution(self, ell: int) -> None:
        r"""J @ J should equal identity (involution property)."""
        J = _compute_J_matrix(ell, dtype=torch.float64)

        assert verify_j_matrix_involution(J, tol=1e-12), (
            f"J matrix for l={ell} is not an involution (J @ J != I)"
        )

    @pytest.mark.parametrize("ell", [0, 1, 2, 3])
    def test_j_matrix_from_d(self, ell: int) -> None:
        r"""J should equal diag((-1)^i) @ d(pi/2)."""
        J = _compute_J_matrix(ell, dtype=torch.float64)

        beta = torch.tensor([math.pi / 2], dtype=torch.float64)
        d_pi2 = _compute_d_matrix(ell, beta).squeeze(0)

        assert verify_j_matrix_from_d(ell, J, d_pi2, tol=1e-12), (
            f"J matrix for l={ell} does not satisfy J = diag((-1)^i) @ d(pi/2)"
        )

    # =========================================================================
    # edge_vectors_to_euler_angles tests
    # =========================================================================

    def test_euler_y_axis(self) -> None:
        r"""y-axis direction should give beta=0."""
        edge = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge)

        torch.testing.assert_close(
            beta,
            torch.zeros_like(beta),
            atol=1e-10,
            rtol=0,
            msg=f"y-axis should give beta=0, got {beta.item()}",
        )
        torch.testing.assert_close(
            gamma,
            torch.zeros_like(gamma),
            atol=1e-10,
            rtol=0,
            msg=f"gamma should always be 0, got {gamma.item()}",
        )

    def test_euler_negative_y_axis(self) -> None:
        r"""Negative y-axis should give beta=pi."""
        edge = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge)

        expected_beta = torch.tensor([math.pi], dtype=torch.float64)
        torch.testing.assert_close(
            beta,
            expected_beta,
            atol=1e-10,
            rtol=0,
            msg=f"Negative y-axis should give beta=pi, got {beta.item()}",
        )

    def test_euler_x_axis(self) -> None:
        r"""x-axis direction: beta=pi/2, alpha=pi/2."""
        edge = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge)

        expected_beta = torch.tensor([math.pi / 2], dtype=torch.float64)
        expected_alpha = torch.tensor([math.pi / 2], dtype=torch.float64)

        torch.testing.assert_close(
            beta,
            expected_beta,
            atol=1e-10,
            rtol=0,
            msg=f"x-axis should give beta=pi/2, got {beta.item()}",
        )
        torch.testing.assert_close(
            alpha,
            expected_alpha,
            atol=1e-10,
            rtol=0,
            msg=f"x-axis should give alpha=pi/2, got {alpha.item()}",
        )

    def test_euler_z_axis(self) -> None:
        r"""z-axis direction: beta=pi/2, alpha=0."""
        edge = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge)

        expected_beta = torch.tensor([math.pi / 2], dtype=torch.float64)
        expected_alpha = torch.tensor([0.0], dtype=torch.float64)

        torch.testing.assert_close(
            beta,
            expected_beta,
            atol=1e-10,
            rtol=0,
            msg=f"z-axis should give beta=pi/2, got {beta.item()}",
        )
        torch.testing.assert_close(
            alpha,
            expected_alpha,
            atol=1e-10,
            rtol=0,
            msg=f"z-axis should give alpha=0, got {alpha.item()}",
        )

    def test_euler_negative_x_axis(self) -> None:
        r"""Negative x-axis direction: beta=pi/2, alpha=-pi/2."""
        edge = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge)

        expected_beta = torch.tensor([math.pi / 2], dtype=torch.float64)
        expected_alpha = torch.tensor([-math.pi / 2], dtype=torch.float64)

        torch.testing.assert_close(
            beta,
            expected_beta,
            atol=1e-10,
            rtol=0,
            msg=f"Negative x-axis should give beta=pi/2, got {beta.item()}",
        )
        torch.testing.assert_close(
            alpha,
            expected_alpha,
            atol=1e-10,
            rtol=0,
            msg=f"Negative x-axis should give alpha=-pi/2, got {alpha.item()}",
        )

    def test_euler_negative_z_axis(self) -> None:
        r"""Negative z-axis direction: beta=pi/2, alpha=pi."""
        edge = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge)

        expected_beta = torch.tensor([math.pi / 2], dtype=torch.float64)
        expected_alpha = torch.tensor([math.pi], dtype=torch.float64)

        torch.testing.assert_close(
            beta,
            expected_beta,
            atol=1e-10,
            rtol=0,
            msg=f"Negative z-axis should give beta=pi/2, got {beta.item()}",
        )
        torch.testing.assert_close(
            alpha,
            expected_alpha,
            atol=1e-10,
            rtol=0,
            msg=f"Negative z-axis should give alpha=pi, got {alpha.item()}",
        )

    def test_euler_unnormalized_vectors(self) -> None:
        r"""Unnormalized edge vectors should give same result as normalized."""
        edge_unnorm = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
        edge_norm = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

        alpha1, beta1, gamma1 = edge_vectors_to_euler_angles(edge_unnorm)
        alpha2, beta2, gamma2 = edge_vectors_to_euler_angles(edge_norm)

        torch.testing.assert_close(alpha1, alpha2, atol=1e-10, rtol=0)
        torch.testing.assert_close(beta1, beta2, atol=1e-10, rtol=0)
        torch.testing.assert_close(gamma1, gamma2, atol=1e-10, rtol=0)

    def test_euler_batch_processing(self) -> None:
        r"""Test batch processing of edge vectors."""
        edges = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # y-axis
                [1.0, 0.0, 0.0],  # x-axis
                [0.0, 0.0, 1.0],  # z-axis
            ],
            dtype=torch.float64,
        )
        alpha, beta, gamma = edge_vectors_to_euler_angles(edges)

        assert alpha.shape == (3,)
        assert beta.shape == (3,)
        assert gamma.shape == (3,)

        # Check individual results
        torch.testing.assert_close(
            beta[0], torch.tensor(0.0, dtype=torch.float64), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(
            beta[1], torch.tensor(math.pi / 2, dtype=torch.float64), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(
            beta[2], torch.tensor(math.pi / 2, dtype=torch.float64), atol=1e-10, rtol=0
        )


# =============================================================================
# EdgeRotation Regression Tests
# =============================================================================


class TestEdgeRotationRegression:
    """Regression tests with hardcoded expected values."""

    def test_lmax2_y_axis_values(self) -> None:
        """Hardcoded test: y-axis edge with lmax=2 should give identity."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        edge_vecs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        # For y-axis (beta=0, alpha=0, gamma=0), the rotation should be identity
        expected = torch.eye(9, dtype=torch.float64).unsqueeze(0).unsqueeze(0)

        torch.testing.assert_close(
            D,
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg="D matrix for y-axis should be identity",
        )

    def test_lmax2_x_axis_values(self) -> None:
        """Hardcoded test: x-axis edge with lmax=2."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        edge_vecs = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        # For x-axis: alpha=pi/2, beta=pi/2, gamma=0
        # The D matrix should be orthogonal
        D_squeezed = D[0, 0]
        product = torch.matmul(D_squeezed, D_squeezed.T)
        identity = torch.eye(9, dtype=torch.float64)

        torch.testing.assert_close(
            product,
            identity,
            rtol=1e-5,
            atol=1e-5,
            msg="D @ D^T should be identity for x-axis",
        )

        # The D matrix should not be identity (alpha and beta are non-zero)
        assert not torch.allclose(D_squeezed, identity, rtol=1e-3, atol=1e-3), (
            "D for x-axis should not be identity"
        )

    def test_lmax2_z_axis_values(self) -> None:
        """Hardcoded test: z-axis edge with lmax=2."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        edge_vecs = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        # For z-axis: alpha=0, beta=pi/2, gamma=0
        D_squeezed = D[0, 0]
        product = torch.matmul(D_squeezed, D_squeezed.T)
        identity = torch.eye(9, dtype=torch.float64)

        torch.testing.assert_close(
            product,
            identity,
            rtol=1e-5,
            atol=1e-5,
            msg="D @ D^T should be identity for z-axis",
        )

    def test_specific_edge_vectors(self, dtype, device) -> None:
        """Test specific edge vectors against precomputed D matrices."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)
        model = model.to(dtype=dtype, device=device)

        # Test with normalized diagonal direction
        edge_vecs = torch.tensor(
            [[[1.0 / math.sqrt(3), 1.0 / math.sqrt(3), 1.0 / math.sqrt(3)]]],
            dtype=dtype,
            device=device,
        )

        D = model.get_wigner_matrices(edge_vecs)

        # D should be orthogonal
        D_squeezed = D[0, 0]
        product = torch.matmul(D_squeezed, D_squeezed.T).abs()
        identity = torch.eye(9, dtype=product.dtype, device=device)

        # Use looser tolerances for orthogonality check
        if is_half_precision(dtype):
            rtol, atol = get_rtol_atol(dtype, scale=20.0)
        else:
            rtol, atol = get_rtol_atol(dtype, scale=10.0)
        torch.testing.assert_close(
            product,
            identity,
            rtol=rtol,
            atol=atol,
            msg=f"D @ D^T should be identity for diagonal vector"
            f" (dtype={dtype}, device={device}, rtol={rtol}, atol={atol})",
        )

    def test_lmax3_regression(self) -> None:
        """Regression test for lmax=3."""
        lmax = 3
        model = EdgeRotation(lmax=lmax)

        # Test y-axis (identity case)
        edge_vecs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        # Should be identity for y-axis
        expected = torch.eye(16, dtype=torch.float64).unsqueeze(0).unsqueeze(0)

        torch.testing.assert_close(
            D,
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg="D matrix for y-axis at lmax=3 should be identity",
        )

    def test_mmax_reduction_regression(self) -> None:
        """Regression test for mmax < lmax."""
        lmax = 3
        mmax = 1
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        edge_vecs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        # Check shape: reduced_dim = 1 + 3 + 3 + 3 = 10, full_dim = 16
        assert D.shape == (1, 1, 10, 16), (
            f"Expected shape (1, 1, 10, 16), got {D.shape}"
        )


# =============================================================================
# TestEdgeRotationParameterized - Parameterized lmax/mmax tests
# =============================================================================


class TestEdgeRotationParameterized:
    """End-to-end functionality tests with parameterized lmax/mmax combinations."""

    def test_output_shape(self, lmax_mmax) -> None:
        """Verify output tensor shape is correct for all lmax/mmax combinations."""
        lmax, mmax = lmax_mmax
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        num_nodes, max_neighbors = 3, 4
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        D = model.get_wigner_matrices(edge_vecs)

        full_dim, reduced_dim = compute_expected_dims(lmax, mmax)
        expected_shape = (num_nodes, max_neighbors, reduced_dim, full_dim)

        assert D.shape == expected_shape, (
            f"lmax={lmax}, mmax={mmax}: expected shape {expected_shape}, got {D.shape}"
        )

    def test_dimensions_stored(self, lmax_mmax) -> None:
        """Verify _full_dim and _reduced_dim attributes are correct."""
        lmax, mmax = lmax_mmax
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        full_dim, reduced_dim = compute_expected_dims(lmax, mmax)

        assert model._full_dim == full_dim, (
            f"lmax={lmax}, mmax={mmax}: expected _full_dim={full_dim}, got {model._full_dim}"
        )
        assert model._reduced_dim == reduced_dim, (
            f"lmax={lmax}, mmax={mmax}: expected _reduced_dim={reduced_dim}, got {model._reduced_dim}"
        )

    def test_orthogonality_full_mmax(self, lmax_mmax) -> None:
        """When mmax=lmax, D @ D^T should be identity (orthogonal matrix)."""
        lmax, mmax = lmax_mmax
        if mmax != lmax:
            pytest.skip(
                "Orthogonality test only applies when mmax=lmax (square matrix)"
            )

        model = EdgeRotation(lmax=lmax, mmax=mmax)
        model = model.to(dtype=torch.float64)

        num_edges = 5
        edge_vecs = torch.randn(num_edges, 1, 3, dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        full_dim = (lmax + 1) ** 2
        identity = torch.eye(full_dim, dtype=torch.float64)

        for i in range(num_edges):
            D_i = D[i, 0]
            product = torch.matmul(D_i, D_i.T)
            torch.testing.assert_close(
                product,
                identity,
                rtol=1e-5,
                atol=1e-5,
                msg=f"lmax={lmax}: D @ D^T not identity for edge {i}",
            )

    def test_y_axis_identity(self, lmax_mmax) -> None:
        """Y-axis edge should give identity rotation for all lmax values."""
        lmax, mmax = lmax_mmax
        if mmax != lmax:
            pytest.skip("Identity test only applies when mmax=lmax (square matrix)")

        model = EdgeRotation(lmax=lmax, mmax=mmax)
        model = model.to(dtype=torch.float64)

        # Y-axis edge (identity direction in this convention)
        edge_vecs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        full_dim = (lmax + 1) ** 2
        expected = torch.eye(full_dim, dtype=torch.float64).unsqueeze(0).unsqueeze(0)

        torch.testing.assert_close(
            D,
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg=f"lmax={lmax}: D matrix for y-axis should be identity",
        )

    def test_j_matrices_registered(self, lmax_mmax) -> None:
        """Verify all J matrices are registered as buffers."""
        lmax, mmax = lmax_mmax
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        for ell in range(lmax + 1):
            buffer_name = f"_J_{ell}"
            assert hasattr(model, buffer_name), f"Missing buffer {buffer_name}"
            J_l = getattr(model, buffer_name)
            expected_shape = (2 * ell + 1, 2 * ell + 1)
            assert J_l.shape == expected_shape, (
                f"J_{ell} has shape {J_l.shape}, expected {expected_shape}"
            )

    def test_gradient_flow(self, lmax_mmax) -> None:
        """Verify gradients flow through for all lmax/mmax combinations."""
        lmax, mmax = lmax_mmax
        model = EdgeRotation(lmax=lmax, mmax=mmax)
        model = model.to(dtype=torch.float64)

        edge_vecs = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)

        D = model.get_wigner_matrices(edge_vecs)
        loss = D.sum()
        loss.backward()

        assert edge_vecs.grad is not None, (
            f"lmax={lmax}, mmax={mmax}: gradients should flow to edge_vecs"
        )
        assert torch.isfinite(edge_vecs.grad).all(), (
            f"lmax={lmax}, mmax={mmax}: gradients should be finite"
        )

    def test_dtype_device_consistency(self, lmax_mmax, dtype, device) -> None:
        """Verify dtype and device are preserved for all combinations."""
        lmax, mmax = lmax_mmax
        model = EdgeRotation(lmax=lmax, mmax=mmax)
        model = model.to(dtype=dtype, device=device)

        edge_vecs = torch.randn(2, 2, 3, dtype=dtype, device=device)

        D = model.get_wigner_matrices(edge_vecs)

        assert D.dtype == dtype, (
            f"lmax={lmax}, mmax={mmax}: expected dtype {dtype}, got {D.dtype}"
        )
        assert D.device.type == device.type, (
            f"lmax={lmax}, mmax={mmax}: expected device {device.type}, got {D.device.type}"
        )

    def test_mask_identity_shape(self, lmax_mmax) -> None:
        """Verify masked edges get identity with correct shape."""
        lmax, mmax = lmax_mmax
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        num_nodes, max_neighbors = 2, 3
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        # Mask out first edge
        mask = torch.ones(num_nodes, max_neighbors, dtype=torch.bool)
        mask[0, 0] = False

        D = model.get_wigner_matrices(edge_vecs, mask=mask)

        # Verify shape is still correct
        full_dim, reduced_dim = compute_expected_dims(lmax, mmax)
        expected_shape = (num_nodes, max_neighbors, reduced_dim, full_dim)

        assert D.shape == expected_shape, (
            f"lmax={lmax}, mmax={mmax}: masked output shape {D.shape} != expected {expected_shape}"
        )

        # Verify masked edge has identity
        identity = model._get_identity_reduced(1, D.dtype, D.device)
        torch.testing.assert_close(
            D[0, 0],
            identity[0],
            rtol=1e-5,
            atol=1e-5,
            msg=f"lmax={lmax}, mmax={mmax}: masked edge should have identity",
        )


# =============================================================================
# TestEdgeRotation - Misc. infra
# =============================================================================


class TestEdgeRotation:
    r"""Test suite for EdgeRotation module."""

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_init_buffers_registered(self) -> None:
        r"""Verify J matrices are registered as persistent buffers."""
        lmax = 3
        model = EdgeRotation(lmax=lmax)

        # Check all J matrices are registered
        for ell in range(lmax + 1):
            assert hasattr(model, f"_J_{ell}"), f"Missing buffer _J_{ell}"
            J_l = getattr(model, f"_J_{ell}")
            assert J_l.shape == (2 * ell + 1, 2 * ell + 1), f"Wrong shape for _J_{ell}"

        # Check they appear in state_dict
        state_dict = model.state_dict()
        for ell in range(lmax + 1):
            assert f"_J_{ell}" in state_dict, f"_J_{ell} not in state_dict"

    def test_init_default_mmax(self) -> None:
        r"""Verify mmax defaults to lmax when not specified."""
        lmax = 4
        model = EdgeRotation(lmax=lmax)

        assert model.lmax == lmax
        assert model.mmax == lmax, "mmax should default to lmax"

    def test_init_invalid_mmax_raises(self) -> None:
        r"""Verify ValueError when mmax > lmax."""
        with pytest.raises(ValueError, match="mmax must be <= lmax"):
            EdgeRotation(lmax=2, mmax=3)

    def test_init_dimensions_computed(self) -> None:
        r"""Verify _full_dim and _reduced_dim are correct."""
        # Test case 1: lmax=2, mmax=2 (no reduction)
        model = EdgeRotation(lmax=2, mmax=2)
        assert model._full_dim == 9, "full_dim should be (2+1)^2 = 9"
        # reduced_dim = min(5, 1) + min(5, 3) + min(5, 5) = 1 + 3 + 5 = 9
        assert model._reduced_dim == 9, "reduced_dim should be 9 when mmax=lmax"

        # Test case 2: lmax=2, mmax=1 (reduction)
        model = EdgeRotation(lmax=2, mmax=1)
        assert model._full_dim == 9, "full_dim should be (2+1)^2 = 9"
        # reduced_dim = min(3, 1) + min(3, 3) + min(3, 5) = 1 + 3 + 3 = 7
        assert model._reduced_dim == 7, "reduced_dim should be 7 when mmax=1"

        # Test case 3: lmax=3, mmax=1
        model = EdgeRotation(lmax=3, mmax=1)
        assert model._full_dim == 16, "full_dim should be (3+1)^2 = 16"
        # reduced_dim = min(3, 1) + min(3, 3) + min(3, 5) + min(3, 7) = 1 + 3 + 3 + 3 = 10
        assert model._reduced_dim == 10, "reduced_dim should be 10 when lmax=3, mmax=1"

        # Test case 4: lmax=3, mmax=0
        model = EdgeRotation(lmax=3, mmax=0)
        # reduced_dim = min(1, 1) + min(1, 3) + min(1, 5) + min(1, 7) = 1 + 1 + 1 + 1 = 4
        assert model._reduced_dim == 4, "reduced_dim should be 4 when mmax=0"

    # =========================================================================
    # Forward Shape Tests
    # =========================================================================

    def test_forward_shape_full(self) -> None:
        r"""Test output shape when mmax = lmax."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 4, 5
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        D = model.get_wigner_matrices(edge_vecs)

        # full_dim = (lmax+1)^2 = 9
        # reduced_dim = 1 + 3 + 5 = 9 (same as full when mmax=lmax)
        assert D.shape == (4, 5, 9, 9), f"Expected shape (4, 5, 9, 9), got {D.shape}"

    def test_forward_shape_reduced(self) -> None:
        r"""Test output shape when mmax < lmax."""
        lmax = 3
        mmax = 1
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        num_nodes, max_neighbors = 4, 5
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        D = model.get_wigner_matrices(edge_vecs)

        # full_dim = (3+1)^2 = 16
        # reduced_dim = 1 + 3 + 3 + 3 = 10
        assert D.shape == (4, 5, 10, 16), (
            f"Expected shape (4, 5, 10, 16), got {D.shape}"
        )

    # =========================================================================
    # Mathematical Correctness Tests
    # =========================================================================

    def test_y_axis_gives_identity(self) -> None:
        r"""Edge [0,1,0] should give D = I.

        In the convention used, beta = acos(y), so pointing along y-axis
        (y=1) gives beta=0. Combined with alpha=0 when x=z=0, this
        results in the identity rotation.
        """
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        # Single edge pointing along y-axis (identity direction in this convention)
        edge_vecs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        # For y-axis (beta=0, alpha=0, gamma=0), the rotation should be identity
        expected = torch.eye(9, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        torch.testing.assert_close(
            D,
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg="D matrix for y-axis should be identity",
        )

    def test_orthogonality_parameterized(self, lmax_mmax, dtype, device) -> None:
        r"""Test orthogonality with parameterized dtype and device."""
        model = EdgeRotation(*lmax_mmax)
        model = model.to(dtype=dtype, device=device)

        num_nodes, max_neighbors = 2, 3
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=dtype, device=device)

        D = model.get_wigner_matrices(edge_vecs)

        identity = torch.eye(model._reduced_dim, dtype=dtype, device=device)
        # rescale tolerance
        match dtype:
            case torch.float32:
                scaling = 10.0
            case torch.float16:
                scaling = 1e3
            case torch.bfloat16:
                scaling = 1e3
            case _:
                scaling = 1.0
        rtol, atol = get_rtol_atol(dtype, scaling)

        for i in range(num_nodes):
            for j in range(max_neighbors):
                D_ij = D[i, j]
                product = torch.matmul(D_ij, D_ij.T)
                torch.testing.assert_close(
                    product,
                    identity,
                    rtol=rtol,
                    atol=atol,
                    msg=f"D @ D^T not identity for edge [{i}, {j}] (dtype={dtype}, device={device})",
                )

    def test_negative_y_axis(self) -> None:
        r"""Edge [0,-1,0] should give expected pattern.

        Negative y-axis corresponds to beta=pi (180 degree rotation about y).
        The D matrix should still be orthogonal but not identity.
        """
        lmax = 1
        model = EdgeRotation(lmax=lmax)

        # Edge pointing along negative y-axis
        edge_vecs = torch.tensor([[[0.0, -1.0, 0.0]]], dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        D = model.get_wigner_matrices(edge_vecs)

        # Verify it's still orthogonal
        D_squeezed = D[0, 0]
        product = torch.matmul(D_squeezed, D_squeezed.T)
        identity = torch.eye(4, dtype=torch.float64)  # (1+1)^2 = 4 for lmax=1
        torch.testing.assert_close(
            product,
            identity,
            rtol=1e-5,
            atol=1e-5,
            msg="D @ D^T not identity for negative y-axis",
        )

        # Verify it's not just the identity
        assert not torch.allclose(D_squeezed, identity, rtol=1e-3, atol=1e-3), (
            "D for negative y should not be identity"
        )

    # =========================================================================
    # Mask Tests
    # =========================================================================

    def test_mask_applied_identity(self) -> None:
        r"""Masked (False) edges get identity matrix."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 2, 3
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)

        # Mask out specific edges
        mask = torch.ones(num_nodes, max_neighbors, dtype=torch.bool)
        mask[0, 0] = False
        mask[1, 2] = False

        D = model.get_wigner_matrices(edge_vecs, mask=mask)

        # Get expected identity in reduced form
        identity = model._get_identity_reduced(1, D.dtype, D.device)

        # Check masked edges have identity
        torch.testing.assert_close(
            D[0, 0],
            identity[0],
            rtol=1e-5,
            atol=1e-5,
            msg="Masked edge [0,0] should have identity",
        )
        torch.testing.assert_close(
            D[1, 2],
            identity[0],
            rtol=1e-5,
            atol=1e-5,
            msg="Masked edge [1,2] should have identity",
        )

    def test_mask_none_all_computed(self) -> None:
        r"""Without mask, all edges are computed normally."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 2, 3
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=torch.float64)
        model = model.to(dtype=torch.float64)

        # Without mask
        D_no_mask = model.get_wigner_matrices(edge_vecs, mask=None)

        # With all-True mask
        mask = torch.ones(num_nodes, max_neighbors, dtype=torch.bool)
        D_all_true = model.get_wigner_matrices(edge_vecs, mask=mask)

        torch.testing.assert_close(
            D_no_mask,
            D_all_true,
            rtol=1e-10,
            atol=1e-10,
            msg="mask=None should behave same as all-True mask",
        )

    def test_mask_parameterized(self, dtype, device) -> None:
        r"""Test masking with parameterized dtype and device."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)
        model = model.to(dtype=dtype, device=device)

        num_nodes, max_neighbors = 2, 3
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=dtype, device=device)

        mask = torch.ones(num_nodes, max_neighbors, dtype=torch.bool, device=device)
        mask[0, 0] = False

        D = model.get_wigner_matrices(edge_vecs, mask=mask)

        identity = model._get_identity_reduced(1, dtype, device)

        rtol, atol = get_rtol_atol(
            dtype, scale=10.0 if is_half_precision(dtype) else 1.0
        )

        torch.testing.assert_close(
            D[0, 0],
            identity[0],
            rtol=rtol,
            atol=atol,
            msg=f"Masked edge should have identity (dtype={dtype}, device={device})",
        )

    # =========================================================================
    # State Dict Tests
    # =========================================================================

    def test_state_dict_contains_J_matrices(self) -> None:
        r"""Save state_dict and verify J buffers present."""
        lmax = 3
        model = EdgeRotation(lmax=lmax)

        state_dict = model.state_dict()

        # Check all J matrices are in state_dict
        for ell in range(lmax + 1):
            key = f"_J_{ell}"
            assert key in state_dict, f"{key} not found in state_dict"
            assert state_dict[key].shape == (2 * ell + 1, 2 * ell + 1), (
                f"Wrong shape for {key} in state_dict"
            )

    def test_state_dict_roundtrip(self, lmax_mmax) -> None:
        r"""Save/load model and verify identical output."""
        model1 = EdgeRotation(*lmax_mmax)

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load state dict
        model2 = EdgeRotation(*lmax_mmax)
        model2.load_state_dict(state_dict)

        # Test with same input
        edge_vecs = torch.randn(3, 4, 3)

        D1 = model1.get_wigner_matrices(edge_vecs)
        D2 = model2.get_wigner_matrices(edge_vecs)

        torch.testing.assert_close(
            D1,
            D2,
            rtol=1e-10,
            atol=1e-10,
            msg="Model output should be identical after state_dict roundtrip",
        )

    def test_dtype_device(self, dtype, device, lmax_mmax) -> None:
        r"""Test dtype and device handling with fixtures."""
        model = EdgeRotation(*lmax_mmax)
        model = model.to(dtype=dtype, device=device)

        edge_vecs = torch.randn(2, 3, 3, dtype=dtype, device=device)
        D = model.get_wigner_matrices(edge_vecs)

        assert D.dtype == dtype, f"Expected {dtype} output, got {D.dtype}"
        assert D.device.type == device.type, (
            f"Expected {device.type} output, got {D.device.type}"
        )

    # =========================================================================
    # Gradient Tests
    # =========================================================================

    def test_gradient_flow(self, dtype, device, lmax_mmax) -> None:
        r"""Test gradient flow with parameterized dtype and device."""
        model = EdgeRotation(*lmax_mmax)
        model = model.to(dtype=dtype, device=device)

        edge_vecs = torch.randn(2, 3, 3, dtype=dtype, device=device, requires_grad=True)

        D = model.get_wigner_matrices(edge_vecs)

        loss = D.sum()
        loss.backward()

        assert edge_vecs.grad is not None, (
            f"Gradients should flow to edge_vecs (dtype={dtype}, device={device})"
        )
        assert torch.isfinite(edge_vecs.grad).all(), (
            f"Gradients should be finite (dtype={dtype}, device={device})"
        )

    # =========================================================================
    # Apply Rotation Tests
    # =========================================================================

    def test_apply_rotation_shape(self) -> None:
        r"""Test apply_rotation output shape."""
        lmax = 2
        mmax = 1
        model = EdgeRotation(lmax=lmax, mmax=mmax)

        num_nodes, max_neighbors = 3, 4
        channels = 8
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)
        x = torch.randn(num_nodes, max_neighbors, 9, channels)  # full_dim = 9

        # Forward rotation
        model.get_wigner_matrices(edge_vecs)
        x_rotated = model(x)

        # Should have reduced_dim = 7
        assert x_rotated.shape == (num_nodes, max_neighbors, 7, channels), (
            f"Expected shape (3, 4, 7, 8), got {x_rotated.shape}"
        )

    def test_apply_rotation_with_precomputed_wigner(self) -> None:
        r"""Test apply_rotation with pre-computed Wigner matrices."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        num_nodes, max_neighbors = 2, 3
        channels = 4
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3)
        x = torch.randn(num_nodes, max_neighbors, 9, channels)

        # Compute Wigner matrices separately
        _ = model.get_wigner_matrices(edge_vecs)

        # Apply rotation using pre-computed matrices
        x_rotated = model(x)

        assert x_rotated.shape == (num_nodes, max_neighbors, 9, channels)

    def test_apply_rotation_roundtrip(self) -> None:
        r"""Test that forward + inverse rotation returns original (mmax=lmax)."""
        lmax = 2
        model = EdgeRotation(lmax=lmax, mmax=lmax)
        model = model.to(dtype=torch.float64)

        num_nodes, max_neighbors = 4, 5
        channels = 8
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=torch.float64)
        x = torch.randn(num_nodes, max_neighbors, 9, channels, dtype=torch.float64)

        # Forward rotation
        _ = model.get_wigner_matrices(edge_vecs)
        x_rotated = model(x)

        # Inverse rotation
        x_back = model(x_rotated, inverse=True)

        # Should recover original
        torch.testing.assert_close(
            x,
            x_back,
            rtol=1e-5,
            atol=1e-5,
            msg="Forward + inverse rotation should return original",
        )

    def test_apply_rotation_roundtrip_reduced(self) -> None:
        r"""Test that forward + inverse rotation works with mmax < lmax.

        Note: When mmax < lmax, the rotation is lossy (information is discarded),
        so we can't expect perfect round-trip recovery. This test just verifies
        the shapes and that the operation runs without error.
        """
        lmax = 3
        mmax = 1
        model = EdgeRotation(lmax=lmax, mmax=mmax)
        model = model.to(dtype=torch.float64)

        num_nodes, max_neighbors = 2, 3
        channels = 4
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=torch.float64)
        x = torch.randn(num_nodes, max_neighbors, 16, channels, dtype=torch.float64)

        # Forward rotation
        _ = model.get_wigner_matrices(edge_vecs)
        x_rotated = model(x)

        assert x_rotated.shape == (num_nodes, max_neighbors, 10, channels)

        # Inverse rotation
        x_back = model(x_rotated, inverse=True)

        assert x_back.shape == (num_nodes, max_neighbors, 16, channels)

        # Note: We cannot expect x == x_back because information was lost
        # in the forward rotation (reduced representation discards high-order modes)

    def test_forward_requires_cache(self) -> None:
        r"""Test that forward() raises error when cache is empty."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        x = torch.randn(2, 3, 9, 4)

        with pytest.raises(RuntimeError, match="No cached Wigner D-matrices"):
            model(x)

    def test_apply_rotation_dtype_device(self, dtype, device) -> None:
        r"""Test apply_rotation with parameterized dtype and device."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)
        model = model.to(dtype=dtype, device=device)

        num_nodes, max_neighbors = 2, 3
        channels = 4
        edge_vecs = torch.randn(num_nodes, max_neighbors, 3, dtype=dtype, device=device)
        x = torch.randn(
            num_nodes, max_neighbors, 9, channels, dtype=dtype, device=device
        )

        model.get_wigner_matrices(edge_vecs)
        x_rotated = model(x)

        assert x_rotated.dtype == dtype
        assert x_rotated.device.type == device.type

    def test_get_wigner_matrices_caching(self) -> None:
        r"""Test that get_wigner_matrices caches D-matrices for forward()."""
        lmax = 2
        model = EdgeRotation(lmax=lmax)

        edge_vecs = torch.randn(3, 4, 3)

        # Compute and cache D-matrices
        D = model.get_wigner_matrices(edge_vecs)

        # Verify cache is populated
        assert model._cached_wigner is not None
        torch.testing.assert_close(
            D,
            model._cached_wigner,
            rtol=1e-10,
            atol=1e-10,
            msg="get_wigner_matrices should cache the computed D-matrices",
        )

    def test_apply_rotation_gradient_flow(self, dtype, device) -> None:
        r"""Test gradient flow through apply_rotation."""
        if dtype in [torch.float16, torch.bfloat16]:
            pytest.skip("Skipping gradient test for half precision dtypes")

        lmax = 2
        model = EdgeRotation(lmax=lmax)
        model = model.to(dtype=dtype, device=device)

        edge_vecs = torch.randn(2, 3, 3, dtype=dtype, device=device, requires_grad=True)
        x = torch.randn(2, 3, 9, 4, dtype=dtype, device=device, requires_grad=True)

        model.get_wigner_matrices(edge_vecs)
        x_rotated = model(x)
        loss = x_rotated.sum()
        loss.backward()

        assert edge_vecs.grad is not None
        assert x.grad is not None
        assert torch.isfinite(edge_vecs.grad).all()
        assert torch.isfinite(x.grad).all()

    # =========================================================================
    # Cache Management Tests
    # =========================================================================

    def test_clear_cache(self, dtype, device) -> None:
        r"""Verify clear_cache() removes cached D-matrices."""
        model = EdgeRotation(lmax=2).to(device=device, dtype=dtype)
        edge_vecs = torch.randn(4, 5, 3, device=device, dtype=dtype)

        # Populate cache
        model.get_wigner_matrices(edge_vecs)
        assert model._cached_wigner is not None

        # Clear cache
        model.clear_cache()
        assert model._cached_wigner is None
        assert model._cache_batch_shape is None

    def test_cache_reuse(self, dtype, device) -> None:
        r"""Verify cached D-matrices are reused for multiple forward calls."""
        model = EdgeRotation(lmax=2).to(device=device, dtype=dtype)
        edge_vecs = torch.randn(4, 5, 3, device=device, dtype=dtype)
        x1 = torch.randn(4, 5, 9, 64, device=device, dtype=dtype)
        x2 = torch.randn(4, 5, 9, 32, device=device, dtype=dtype)

        # Compute D-matrices once
        D = model.get_wigner_matrices(edge_vecs)

        # Apply to multiple tensors (should reuse cached D)
        _ = model(x1)
        _ = model(x2)

        # Verify cache wasn't cleared
        assert model._cached_wigner is not None
        torch.testing.assert_close(model._cached_wigner, D)

    # =========================================================================
    # Batch Consistency Tests
    # =========================================================================

    def test_batch_consistency(self, dtype, device, lmax_mmax) -> None:
        r"""Test batch consistency with parameterized dtype and device."""
        model = EdgeRotation(*lmax_mmax)
        model = model.to(dtype=dtype, device=device)

        single_edge = torch.randn(3, dtype=dtype, device=device)

        edge_vecs = torch.randn(3, 4, 3, dtype=dtype, device=device)
        edge_vecs[0, 1] = single_edge
        edge_vecs[2, 3] = single_edge

        D = model.get_wigner_matrices(edge_vecs)

        rtol, atol = get_rtol_atol(
            dtype, scale=10.0 if is_half_precision(dtype) else 1.0
        )

        torch.testing.assert_close(
            D[0, 1],
            D[2, 3],
            rtol=rtol,
            atol=atol,
            msg=f"Same edge vector should produce same D matrix (dtype={dtype}, device={device})",
        )


class TestEdgeRotationComputationDtype:
    r"""Test suite for EdgeRotation.computation_dtype type promotion.

    Tests the `computation_dtype` parameter that allows users to specify a
    higher-precision dtype for internal Wigner D-matrix computations to
    improve numerical accuracy for half-precision inputs.
    """

    # =========================================================================
    # Basic Functionality Tests
    # =========================================================================

    def test_computation_dtype_stored(self) -> None:
        r"""Verify that computation_dtype is stored correctly."""
        model = EdgeRotation(lmax=2, computation_dtype=torch.float32)
        assert model.computation_dtype == torch.float32, (
            "computation_dtype should be stored as torch.float32"
        )

        model = EdgeRotation(lmax=2, computation_dtype=torch.float64)
        assert model.computation_dtype == torch.float64, (
            "computation_dtype should be stored as torch.float64"
        )

    # =========================================================================
    # Numerical Precision Tests
    # =========================================================================

    def test_computation_dtype_improves_precision(self) -> None:
        r"""Verify that using computation_dtype improves numerical precision.

        Compare orthogonality error (||D @ D^T - I||) between models with and
        without computation_dtype promotion for half-precision inputs.
        """
        lmax = 2
        model_no_promotion = EdgeRotation(lmax=lmax, computation_dtype=None)
        model_with_promotion = EdgeRotation(lmax=lmax, computation_dtype=torch.float32)

        # Use float16 input where precision issues are noticeable
        num_edges = 10
        edge_vecs = torch.randn(num_edges, 1, 3, dtype=torch.float16)

        D_no_promotion = model_no_promotion.get_wigner_matrices(edge_vecs)
        D_with_promotion = model_with_promotion.get_wigner_matrices(edge_vecs)

        # Both should be float16
        assert D_no_promotion.dtype == torch.float16
        assert D_with_promotion.dtype == torch.float16

        # Compute orthogonality errors: ||D @ D^T - I||_F
        full_dim = (lmax + 1) ** 2
        identity = torch.eye(full_dim, dtype=torch.float16)

        errors_no_promotion = []
        errors_with_promotion = []

        for i in range(num_edges):
            D_i_no = D_no_promotion[i, 0]
            D_i_with = D_with_promotion[i, 0]

            error_no = torch.norm(torch.matmul(D_i_no, D_i_no.T) - identity)
            error_with = torch.norm(torch.matmul(D_i_with, D_i_with.T) - identity)

            errors_no_promotion.append(error_no.item())
            errors_with_promotion.append(error_with.item())

        avg_error_no = sum(errors_no_promotion) / len(errors_no_promotion)
        avg_error_with = sum(errors_with_promotion) / len(errors_with_promotion)

        # The model with computation_dtype should have better (lower) error
        assert avg_error_with < avg_error_no, (
            f"computation_dtype should improve precision: "
            f"avg_error_no_promotion={avg_error_no:.6f}, "
            f"avg_error_with_promotion={avg_error_with:.6f}"
        )

    # =========================================================================
    # Type Promotion Tests
    # =========================================================================

    @pytest.mark.parametrize(
        "first_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    @pytest.mark.parametrize(
        "second_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_computation_with_casting(self, first_dtype, second_dtype) -> None:
        r"""Test different data type casting"""
        model = EdgeRotation(lmax=2, computation_dtype=first_dtype)

        edge_vecs = torch.rand(3, 4, 3, dtype=second_dtype)
        # computation should be done with first_dtype, but result is
        # in second_dtype
        D = model.get_wigner_matrices(edge_vecs)
        assert D.dtype == second_dtype

        # Verify computation completes without error (no NaN/Inf)
        assert torch.isfinite(D).all(), "Output should not contain NaN or Inf"

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_computation_dtype_consistency_across_batches(self) -> None:
        r"""Verify that computation_dtype produces consistent results across different batch sizes."""
        model = EdgeRotation(lmax=2, computation_dtype=torch.float32)

        # Create a single edge vector
        single_edge = torch.randn(3, dtype=torch.float16)

        # Test with different batch arrangements
        edge_vecs_1x1 = single_edge.reshape(1, 1, 3)
        edge_vecs_2x1 = torch.stack([single_edge, single_edge]).unsqueeze(1)

        D_1x1 = model.get_wigner_matrices(edge_vecs_1x1)
        D_2x1 = model.get_wigner_matrices(edge_vecs_2x1)

        # Both batch entries in 2x1 should match the 1x1 result
        rtol, atol = get_rtol_atol(torch.float16, scale=2.0)
        torch.testing.assert_close(
            D_1x1[0, 0],
            D_2x1[0, 0],
            rtol=rtol,
            atol=atol,
            msg="computation_dtype should produce consistent results across batches",
        )
        torch.testing.assert_close(
            D_1x1[0, 0],
            D_2x1[1, 0],
            rtol=rtol,
            atol=atol,
            msg="computation_dtype should produce consistent results across batches",
        )
