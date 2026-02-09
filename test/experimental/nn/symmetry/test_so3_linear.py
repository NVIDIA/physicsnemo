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

"""Comprehensive unit tests for SO(3) linear layer implementation.

This module provides tests for:
- SO3LinearGrid: SO(3) equivariant linear layer
- rotate_grid_coefficients: Wigner D-matrix rotation utility
- SO(3) equivariance verification

Test Structure
--------------
TestSO3LinearGrid
    Tests for the SO3LinearGrid layer (shapes, masking, backward pass).
TestRotateGridCoefficients
    Tests for the rotation utility function.
TestSO3Equivariance
    Key equivariance tests verifying SO(3) symmetry preservation.
TestIntegrationWithEdgeRotation
    Integration tests combining SO3LinearGrid with EdgeRotation.

Notes
-----
The SO(3) equivariance property states that for any rotation R:
    Layer(rotate(x, R)) = rotate(Layer(x), R)

This is the fundamental symmetry that the layer must preserve.
"""

from __future__ import annotations

import math

import pytest
import torch

from physicsnemo.experimental.nn.symmetry import make_grid_mask
from physicsnemo.experimental.nn.symmetry.so3_linear import SO3LinearGrid
from physicsnemo.experimental.nn.symmetry.wigner import (
    edge_vectors_to_euler_angles,
    rotate_grid_coefficients,
)
from test.experimental.nn.symmetry.conftest import get_rtol_atol

# =============================================================================
# Fixtures
# =============================================================================

# Note: `dtype` and `device` fixtures are provided by conftest.py
# The dtype fixture includes: float16, bfloat16, float32, float64
# The device fixture includes: cpu, cuda (with automatic skip if unavailable)


@pytest.fixture(params=[(2, 2), (2, 1), (4, 4)])
def lmax_mmax(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterized fixture for testing with different lmax/mmax configurations.

    Includes:
    - (2, 2): Small lmax with mmax == lmax (low complexity)
    - (4, 4): Larger lmax with mmax == lmax (full grid, higher complexity)

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    tuple[int, int]
        Tuple of (lmax, mmax) values.
    """
    return request.param


# =============================================================================
# TestSO3LinearGrid
# =============================================================================


class TestSO3LinearGrid:
    """Tests for the SO3LinearGrid layer.

    This layer applies degree-wise linear transformations to spherical harmonic
    coefficients while preserving SO(3) equivariance.
    """

    def test_output_shape(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify correct output dimensions.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 32
        out_channels = 64
        batch_size = 50

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        y = layer(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"

    def test_masked_positions_zero(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify invalid positions remain zero in output.

        Positions where m > l should be zero in the output after the
        layer applies the validity mask.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 16
        out_channels = 16
        batch_size = 10

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        # Create input with non-zero values everywhere
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        x = x + 1.0  # Shift to ensure non-zero

        with torch.no_grad():
            y = layer(x)

        # Check invalid positions are zero
        mask = make_grid_mask(lmax, mmax).to(device=device)

        for ell in range(lmax + 1):
            for m in range(mmax + 1):
                if not mask[ell, m]:
                    torch.testing.assert_close(
                        y[:, ell, m, :, :],
                        torch.zeros_like(y[:, ell, m, :, :]),
                        rtol=0,
                        atol=0,
                        msg=f"Invalid position (l={ell}, m={m}) is not zero in output",
                    )

    def test_backward_pass(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify gradients are computed correctly.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 16
        out_channels = 16
        batch_size = 10

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            in_channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert layer.weight.grad is not None, "Weight gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    def test_deterministic(self, dtype: torch.dtype, device: torch.device) -> None:
        """Verify same input gives same output.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        in_channels = 32
        out_channels = 32
        batch_size = 20

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer.eval()

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y1 = layer(x)
            y2 = layer(x)

        torch.testing.assert_close(
            y1, y2, atol=1e-6, rtol=0, msg="Output should be deterministic"
        )

    def test_bias_only_on_scalar(
        self, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify bias is only applied to l=0, m=0, real component.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        batch_size = 10

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            bias=True,
        ).to(device=device, dtype=dtype)

        # Set bias to a known non-zero value
        with torch.no_grad():
            layer.bias.fill_(1.0)
            layer.weight.zero_()

        # Zero input
        x = torch.zeros(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y = layer(x)

        # Only l=0, m=0, real should have the bias
        torch.testing.assert_close(
            y[:, 0, 0, 0, :],
            torch.ones_like(y[:, 0, 0, 0, :]),
            rtol=0,
            atol=0,
            msg="Bias should be applied to (l=0, m=0, real)",
        )

        # All other positions should be zero
        y_copy = y.clone()
        y_copy[:, 0, 0, 0, :] = 0.0
        torch.testing.assert_close(
            y_copy,
            torch.zeros_like(y_copy),
            rtol=0,
            atol=0,
            msg="Bias should only be applied to (l=0, m=0, real)",
        )


# =============================================================================
# TestRotateGridCoefficients
# =============================================================================


class TestRotateGridCoefficients:
    """Tests for the rotate_grid_coefficients function.

    This function applies Wigner D-matrix rotations to spherical harmonic
    coefficients in the grid layout.
    """

    def test_identity_rotation(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Zero rotation should return input unchanged.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax

        channels = 8
        batch_size = 4

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        # Apply mask to ensure valid input
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        # Enforce m=0 imaginary = 0 constraint (real spherical harmonics property)
        x[:, :, 0, 1, :] = 0.0

        # Zero rotation
        alpha = torch.zeros(batch_size, device=device, dtype=dtype)
        beta = torch.zeros(batch_size, device=device, dtype=dtype)
        gamma = torch.zeros(batch_size, device=device, dtype=dtype)

        x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))

        # rescale tolerance
        match dtype:
            case torch.float32:
                scaling = 10.0
            case torch.float16:
                scaling = 1e4
            case torch.bfloat16:
                scaling = 1e4
            case _:
                scaling = 1.0
        rtol, atol = get_rtol_atol(dtype, scaling)
        torch.testing.assert_close(
            x,
            x_rotated,
            rtol=rtol,
            atol=atol,
            msg=(
                f"Identity rotation should return input unchanged, "
                f"max diff: {torch.abs(x - x_rotated).max().item():.2e}"
            ),
        )

    def test_scalar_invariance(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """l=0 coefficients should be invariant under any rotation.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        channels = 8
        batch_size = 4

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        # Apply mask
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        # Enforce m=0 imaginary = 0 constraint (real spherical harmonics property)
        x[:, :, 0, 1, :] = 0.0

        # Random rotation
        alpha = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi
        beta = torch.rand(batch_size, device=device, dtype=dtype) * math.pi
        gamma = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi

        x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))

        # l=0 coefficients should be unchanged
        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(
            x[:, 0, :, :, :],
            x_rotated[:, 0, :, :, :],
            rtol=rtol,
            atol=atol,
            msg=(
                f"l=0 coefficients should be invariant under rotation, "
                f"max diff: {torch.abs(x[:, 0, :, :, :] - x_rotated[:, 0, :, :, :]).max().item():.2e}"
            ),
        )

    def test_scalar_angle_input(self, dtype: torch.dtype, device: torch.device) -> None:
        """Verify that scalar angle inputs work correctly.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 3, 2
        channels = 4
        batch_size = 2

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        # Scalar angles
        alpha = 0.5
        beta = 0.3
        gamma = 0.7

        # Should not raise
        x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))

        assert x_rotated.shape == x.shape, "Output shape should match input shape"
        assert torch.isfinite(x_rotated).all(), "Output should be finite"

    def test_both_rotation_modes(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify Euler angle mode and D-matrix mode produce the same result.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        from physicsnemo.experimental.nn.symmetry.wigner import (
            compute_wigner_d_matrices,
        )

        lmax, mmax = lmax_mmax
        channels = 8
        batch_size = 4

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        # Apply mask to ensure valid input
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        # Enforce m=0 imaginary = 0 constraint (real spherical harmonics property)
        x[:, :, 0, 1, :] = 0.0

        # Random Euler angles
        alpha = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi
        beta = torch.rand(batch_size, device=device, dtype=dtype) * math.pi
        gamma = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi

        # Mode 1: Use Euler angles directly
        x_rot1 = rotate_grid_coefficients(x, (alpha, beta, gamma))

        # Mode 2: Pre-compute D-matrix
        D = compute_wigner_d_matrices(alpha, beta, gamma, lmax)
        x_rot2 = rotate_grid_coefficients(x, D)

        # Both modes should produce identical results
        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(
            x_rot1,
            x_rot2,
            rtol=rtol,
            atol=atol,
            msg=(
                f"Euler angle mode and D-matrix mode should produce identical results, "
                f"max diff: {torch.abs(x_rot1 - x_rot2).max().item():.2e}"
            ),
        )


# =============================================================================
# TestSO3Equivariance
# =============================================================================


class TestSO3Equivariance:
    """Key symmetry tests: SO(3) equivariance of SO3LinearGrid.

    For any rotation R:
        Layer(rotate(x, R)) = rotate(Layer(x), R)

    This is the fundamental symmetry that the layer must preserve.
    """

    @pytest.mark.parametrize(
        "alpha_val,beta_val,gamma_val",
        [
            (0.1, 0.2, 0.3),  # Small rotation (near identity)
            (math.pi, math.pi / 2, 0.0),  # Large rotation (boundary case with π)
        ],
        ids=["small", "large"],
    )
    def test_so3_linear_equivariance(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
    ) -> None:
        """Test that SO3LinearGrid is equivariant under 3D rotations.

        For any rotation R represented by Euler angles (α, β, γ):
            Layer(D(R) @ x) = D(R) @ Layer(x)

        where D(R) is the Wigner D-matrix.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        alpha_val : float
            First Euler angle (radians).
        beta_val : float
            Second Euler angle (radians).
        gamma_val : float
            Third Euler angle (radians).
        """
        lmax, mmax = lmax_mmax
        in_channels = 16
        out_channels = 16
        batch_size = 10

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer.eval()

        # Create valid input (zero at invalid positions)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]

        alpha = torch.full((batch_size,), alpha_val, device=device, dtype=dtype)
        beta = torch.full((batch_size,), beta_val, device=device, dtype=dtype)
        gamma = torch.full((batch_size,), gamma_val, device=device, dtype=dtype)

        with torch.no_grad():
            # Method 1: Rotate input, then apply layer
            x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
            y1 = layer(x_rotated)

            # Method 2: Apply layer, then rotate output
            y = layer(x)
            y2 = rotate_grid_coefficients(y, (alpha, beta, gamma))

        # Rescale tolerance based on dtype
        match dtype:
            case torch.float32:
                scaling = 10.0
            case torch.float16:
                scaling = 1e4
            case torch.bfloat16:
                scaling = 1e4
            case _:
                scaling = 1.0
        rtol, atol = get_rtol_atol(dtype, scaling)

        torch.testing.assert_close(
            y1,
            y2,
            rtol=rtol,
            atol=atol,
            msg=(
                f"SO(3) equivariance violated at angles "
                f"(α={alpha_val:.3f}, β={beta_val:.3f}, γ={gamma_val:.3f}): "
                f"max diff={torch.abs(y1 - y2).max().item():.2e}"
            ),
        )

    def test_so3_linear_equivariance_random_rotations(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Test SO(3) equivariance with random rotations.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 16
        out_channels = 16
        batch_size = 10
        num_rotations = 3

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer.eval()

        # Create valid input
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]

        for _ in range(num_rotations):
            # Random Euler angles
            alpha = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi
            beta = torch.rand(batch_size, device=device, dtype=dtype) * math.pi
            gamma = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi

            with torch.no_grad():
                # Method 1: Rotate then layer
                x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
                y1 = layer(x_rotated)

                # Method 2: Layer then rotate
                y = layer(x)
                y2 = rotate_grid_coefficients(y, (alpha, beta, gamma))

            # Rescale tolerance based on dtype
            match dtype:
                case torch.float32:
                    scaling = 10.0
                case torch.float16:
                    scaling = 1e4
                case torch.bfloat16:
                    scaling = 1e4
                case _:
                    scaling = 1.0
            rtol, atol = get_rtol_atol(dtype, scaling)

            torch.testing.assert_close(
                y1,
                y2,
                rtol=rtol,
                atol=atol,
                msg=(
                    f"SO(3) equivariance violated with random rotation: "
                    f"max diff={torch.abs(y1 - y2).max().item():.2e}"
                ),
            )

    def test_non_equivariant_layer_fails(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify that a non-equivariant layer fails the equivariance test.

        This is a negative test that confirms our equivariance testing methodology
        is actually detecting violations. A regular nn.Linear that mixes all (l, m)
        coefficients inappropriately should NOT be equivariant.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 8
        batch_size = 5

        # Create a non-equivariant layer: flatten -> Linear -> reshape
        # This mixes all (l, m, real/imag) coefficients, breaking equivariance
        flat_dim = (lmax + 1) * (mmax + 1) * 2 * in_channels

        class NonEquivariantLayer(torch.nn.Module):
            """A layer that intentionally breaks SO(3) equivariance."""

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(flat_dim, flat_dim)

            def forward(self, x):
                shape = x.shape
                x_flat = x.reshape(shape[0], -1)
                y_flat = self.linear(x_flat)
                return y_flat.reshape(shape)

        layer = NonEquivariantLayer().to(device=device, dtype=dtype)
        layer.eval()

        # Create valid input
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]

        # Test with a non-trivial rotation
        alpha = torch.full((batch_size,), math.pi / 4, device=device, dtype=dtype)
        beta = torch.full((batch_size,), math.pi / 3, device=device, dtype=dtype)
        gamma = torch.full((batch_size,), math.pi / 6, device=device, dtype=dtype)

        with torch.no_grad():
            # Method 1: Rotate input, then apply layer
            x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
            y1 = layer(x_rotated)

            # Method 2: Apply layer, then rotate output
            y = layer(x)
            y2 = rotate_grid_coefficients(y, (alpha, beta, gamma))

        # The non-equivariant layer should FAIL the equivariance test
        # We check that the outputs are NOT close
        is_equivariant = torch.allclose(y1, y2, rtol=1e-2, atol=1e-2)
        assert not is_equivariant, (
            "Non-equivariant layer should NOT satisfy equivariance property! "
            "This indicates a bug in the equivariance testing methodology."
        )


# =============================================================================
# TestIntegrationWithEdgeRotation
# =============================================================================


class TestIntegrationWithEdgeRotation:
    """Integration tests combining SO3LinearGrid with EdgeRotation.

    These tests verify that the rotation utilities work correctly with
    the edge-based rotation machinery used in equivariant GNNs.
    """

    def test_edge_rotation_produces_valid_angles(
        self, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify EdgeRotation produces valid Euler angles from edge vectors.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        num_edges = 20

        # Random edge vectors
        edge_vecs = torch.randn(num_edges, 3, device=device, dtype=dtype)

        # Get Euler angles
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge_vecs)

        # Verify shapes
        assert alpha.shape == (num_edges,), (
            f"Expected shape ({num_edges},), got {alpha.shape}"
        )
        assert beta.shape == (num_edges,), (
            f"Expected shape ({num_edges},), got {beta.shape}"
        )
        assert gamma.shape == (num_edges,), (
            f"Expected shape ({num_edges},), got {gamma.shape}"
        )

        # Verify finite values
        assert torch.isfinite(alpha).all(), "Alpha contains non-finite values"
        assert torch.isfinite(beta).all(), "Beta contains non-finite values"
        assert torch.isfinite(gamma).all(), "Gamma contains non-finite values"

        # Verify gamma is always zero for edge rotations
        torch.testing.assert_close(
            gamma,
            torch.zeros_like(gamma),
            rtol=0,
            atol=0,
            msg="Gamma should be zero for edge rotations",
        )

    def test_so3linear_with_edge_derived_rotation(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Test SO3LinearGrid equivariance with edge-derived rotations.

        This simulates the use case in equivariant GNNs where rotations
        are derived from edge directions.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 8
        out_channels = 8
        batch_size = 10

        layer = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer.eval()

        # Create input
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]

        # Derive rotation from random edge vectors
        edge_vecs = torch.randn(batch_size, 3, device=device, dtype=dtype)
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge_vecs)

        with torch.no_grad():
            # Apply rotation, then layer
            x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
            y1 = layer(x_rotated)

            # Apply layer, then rotation
            y = layer(x)
            y2 = rotate_grid_coefficients(y, (alpha, beta, gamma))

        # Rescale tolerance based on dtype
        match dtype:
            case torch.float32:
                scaling = 10.0
            case torch.float16:
                scaling = 1e4
            case torch.bfloat16:
                scaling = 1e4
            case _:
                scaling = 1.0
        rtol, atol = get_rtol_atol(dtype, scaling)

        torch.testing.assert_close(
            y1,
            y2,
            rtol=rtol,
            atol=atol,
            msg=(
                f"SO(3) equivariance violated with edge-derived rotation: "
                f"max diff={torch.abs(y1 - y2).max().item():.2e}"
            ),
        )


# =============================================================================
# TestValidation
# =============================================================================


class TestValidation:
    """Tests for input validation in SO3LinearGrid."""

    def test_invalid_lmax_negative(self) -> None:
        """Verify error when constructor receives negative lmax."""
        with pytest.raises(ValueError, match=r"lmax must be non-negative"):
            SO3LinearGrid(
                in_channels=16,
                out_channels=16,
                lmax=-1,
                mmax=2,
            )

    def test_invalid_mmax_negative(self) -> None:
        """Verify error when constructor receives negative mmax."""
        with pytest.raises(ValueError, match=r"mmax must be non-negative"):
            SO3LinearGrid(
                in_channels=16,
                out_channels=16,
                lmax=4,
                mmax=-1,
            )

    def test_invalid_mmax_gt_lmax(self) -> None:
        """Verify error when constructor receives mmax > lmax."""
        with pytest.raises(ValueError, match=r"mmax.*must be <= lmax"):
            SO3LinearGrid(
                in_channels=16,
                out_channels=16,
                lmax=2,
                mmax=4,
            )

    def test_invalid_in_channels(self) -> None:
        """Verify error when constructor receives non-positive in_channels."""
        with pytest.raises(ValueError, match=r"in_channels must be positive"):
            SO3LinearGrid(
                in_channels=0,
                out_channels=16,
                lmax=4,
                mmax=2,
            )

    def test_invalid_out_channels(self) -> None:
        """Verify error when constructor receives non-positive out_channels."""
        with pytest.raises(ValueError, match=r"out_channels must be positive"):
            SO3LinearGrid(
                in_channels=16,
                out_channels=0,
                lmax=4,
                mmax=2,
            )
