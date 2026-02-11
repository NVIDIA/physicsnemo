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

"""Comprehensive unit tests for SO3ConvolutionBlock implementation.

This module provides tests for:
- SO3ConvolutionBlock: SO(3) equivariant feed-forward block for features
- Shape preservation
- Gradient flow
- SO(3) equivariance verification

Test Structure
--------------
TestSO3ConvolutionBlock
    Tests for the SO3ConvolutionBlock layer (shapes, backward pass, determinism).
TestSO3BlockEquivariance
    Key equivariance tests verifying SO(3) symmetry preservation.
TestValidation
    Tests for input validation and error handling.

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

from physicsnemo.experimental.nn.symmetry import SO3ConvolutionBlock, make_grid_mask
from physicsnemo.experimental.nn.symmetry.wigner import rotate_grid_coefficients
from test.experimental.nn.symmetry.conftest import get_rtol_atol

# =============================================================================
# Fixtures
# =============================================================================

# Note: `dtype` and `device` fixtures are provided by conftest.py
# Note: `is_half_precision` helper is provided by conftest.py


@pytest.fixture(params=[(2, 2), (2, 1), (4, 4)])
def lmax_mmax(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterized fixture for testing with different lmax/mmax configurations.

    Note: lmax must be >= 1 for SO3ConvolutionBlock (requires gates for l>0).

    Includes:
    - (2, 2): Small lmax with mmax == lmax (low complexity)
    - (2, 1): Small lmax with mmax != lmax (low complexity)
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
# TestSO3ConvolutionBlock
# =============================================================================


class TestSO3ConvolutionBlock:
    """Tests for the SO3ConvolutionBlock layer.

    This layer applies block-wise feed-forward transformations in the spherical
    harmonic domain while preserving SO(3) equivariance.
    """

    def test_output_shape(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify output dimensions match input (residual-friendly).

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 32
        hidden_channels = 64
        batch_size = 50

        layer = SO3ConvolutionBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        y = layer(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, in_channels)
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
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 16
        hidden_channels = 32
        batch_size = 10

        layer = SO3ConvolutionBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
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
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 16
        hidden_channels = 32
        batch_size = 10

        layer = SO3ConvolutionBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
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
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

        # Check gradients for key parameters
        assert layer.so3_linear_1.weight.grad is not None, (
            "so3_linear_1 weight gradients not computed"
        )
        assert layer.so3_linear_2.weight.grad is not None, (
            "so3_linear_2 weight gradients not computed"
        )
        assert layer.scalar_mlp.layers[0].linear.weight.grad is not None, (
            "scalar_mlp weight gradients not computed"
        )

    def test_deterministic(self, dtype: torch.dtype, device: torch.device) -> None:
        """Verify same input gives same output.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        in_channels = 32
        hidden_channels = 64
        batch_size = 20

        layer = SO3ConvolutionBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
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

    def test_extra_repr(self, dtype: torch.dtype, device: torch.device) -> None:
        """Verify extra_repr contains expected parameters.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        layer = SO3ConvolutionBlock(
            in_channels=32,
            hidden_channels=64,
            lmax=4,
            mmax=2,
        ).to(device=device, dtype=dtype)

        repr_str = layer.extra_repr()
        assert "in_channels=32" in repr_str
        assert "hidden_channels=64" in repr_str
        assert "lmax=4" in repr_str
        assert "mmax=2" in repr_str

    @pytest.mark.parametrize(
        "compile_backend,compile_mode",
        [
            ("inductor", "default"),
            ("inductor", "reduce-overhead"),
            ("cudagraphs", None),  # cudagraphs doesn't use mode
        ],
        ids=["inductor-default", "inductor-reduce-overhead", "cudagraphs"],
    )
    def test_torch_compile(
        self,
        lmax_mmax: tuple[int, int],
        compile_backend: str,
        compile_mode: str | None,
    ):
        """Ensure that compilation of the block is functional for forward and backward.

        Notes
        -----
        Half-precision dtypes may have larger numerical differences between
        compiled and eager execution due to operation reordering and fusion.
        """
        lmax, mmax = lmax_mmax
        batch_size = 16
        in_channels = 32
        # compilation shouldn't be too sensitive to dtype and device
        dtype = torch.float32
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        layer = SO3ConvolutionBlock(
            in_channels=in_channels, hidden_channels=64, lmax=lmax, mmax=mmax
        ).to(device=device, dtype=dtype)

        if compile_mode is None:
            compiled = torch.compile(layer, backend=compile_backend)
        else:
            compiled = torch.compile(layer, mode=compile_mode, backend=compile_backend)

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

        # Test forward pass (no grad) - verify numerical equivalence
        layer.eval()
        with torch.no_grad():
            ref_output = layer(x.detach())
            compiled_output = compiled(x.detach())
            rtol, atol = get_rtol_atol(dtype)
            torch.testing.assert_close(
                ref_output, compiled_output, rtol=rtol, atol=atol
            )

        # Test backward pass (with grad) - verify gradients flow
        layer.train()
        output = compiled(x)
        loss = ((torch.randn_like(output) - output) ** 2.0).mean()
        loss.backward()
        example_grad = getattr(
            compiled.scalar_mlp.layers[0].linear.weight, "grad", None
        )
        assert example_grad is not None, "No gradients attached after backward."
        assert torch.isfinite(example_grad).all()


# =============================================================================
# TestSO3BlockEquivariance
# =============================================================================


class TestSO3BlockEquivariance:
    """Key symmetry tests: SO(3) equivariance of SO3ConvolutionBlock.

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
    def test_so3_block_equivariance(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
    ) -> None:
        """Test that SO3ConvolutionBlock is equivariant under 3D rotations.

        For any rotation R represented by Euler angles (alpha, beta, gamma):
            Layer(D(R) @ x) = D(R) @ Layer(x)

        where D(R) is the Wigner D-matrix.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
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
        hidden_channels = 32
        batch_size = 10

        layer = SO3ConvolutionBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer.eval()

        # Create valid input (zero at invalid positions and m=0 imaginary)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        # Enforce m=0 imaginary = 0 constraint (real spherical harmonics property)
        x[:, :, 0, 1, :] = 0.0

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
                f"(alpha={alpha_val:.3f}, beta={beta_val:.3f}, gamma={gamma_val:.3f}): "
                f"max diff={torch.abs(y1 - y2).max().item():.2e}"
            ),
        )

    def test_so3_block_equivariance_random_rotations(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Test SO(3) equivariance with random rotations.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax
        in_channels = 16
        hidden_channels = 32
        batch_size = 10
        num_rotations = 3

        layer = SO3ConvolutionBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer.eval()

        # Create valid input (zero at invalid positions and m=0 imaginary)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        # Enforce m=0 imaginary = 0 constraint (real spherical harmonics property)
        x[:, :, 0, 1, :] = 0.0

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

    def test_scalar_features_pass_through_gates(
        self, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Verify scalar features (l=0) influence gates but remain independent of rotation.

        This tests that the gate computation from scalar features works correctly
        and that scalar features are invariant under rotation.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        in_channels = 16
        hidden_channels = 32
        batch_size = 5

        layer = SO3ConvolutionBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer.eval()

        # Create input with non-zero scalar features (zero at invalid positions and m=0 imaginary)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        # Enforce m=0 imaginary = 0 constraint (real spherical harmonics property)
        x[:, :, 0, 1, :] = 0.0

        # Verify num_gates attribute
        expected_num_gates = lmax * hidden_channels
        assert layer.num_gates == expected_num_gates, (
            f"Expected num_gates={expected_num_gates}, got {layer.num_gates}"
        )

        # Run forward pass and check output shape
        with torch.no_grad():
            y = layer(x)

        assert y.shape == x.shape, (
            f"Output shape {y.shape} doesn't match input shape {x.shape}"
        )


# =============================================================================
# TestValidation
# =============================================================================


class TestValidation:
    """Tests for input validation in SO3ConvolutionBlock."""

    def test_invalid_lmax_too_small(self) -> None:
        """Verify error when constructor receives lmax < 1."""
        with pytest.raises(ValueError, match=r"lmax must be >= 1"):
            SO3ConvolutionBlock(
                in_channels=16,
                hidden_channels=32,
                lmax=0,
                mmax=0,
            )

    def test_invalid_mmax_negative(self) -> None:
        """Verify error when constructor receives negative mmax."""
        with pytest.raises(ValueError, match=r"mmax must be non-negative"):
            SO3ConvolutionBlock(
                in_channels=16,
                hidden_channels=32,
                lmax=4,
                mmax=-1,
            )

    def test_invalid_mmax_gt_lmax(self) -> None:
        """Verify error when constructor receives mmax > lmax."""
        with pytest.raises(ValueError, match=r"mmax.*must be <= lmax"):
            SO3ConvolutionBlock(
                in_channels=16,
                hidden_channels=32,
                lmax=2,
                mmax=4,
            )

    def test_invalid_in_channels(self) -> None:
        """Verify error when constructor receives non-positive in_channels."""
        with pytest.raises(ValueError, match=r"in_channels must be positive"):
            SO3ConvolutionBlock(
                in_channels=0,
                hidden_channels=32,
                lmax=4,
                mmax=2,
            )

    def test_invalid_hidden_channels(self) -> None:
        """Verify error when constructor receives non-positive hidden_channels."""
        with pytest.raises(ValueError, match=r"hidden_channels must be positive"):
            SO3ConvolutionBlock(
                in_channels=16,
                hidden_channels=0,
                lmax=4,
                mmax=2,
            )


# =============================================================================
# TestIntegrationWithRotation
# =============================================================================


class TestIntegrationWithRotation:
    """Integration tests combining SO3ConvolutionBlock with rotation utilities."""

    def test_composition_of_layers(
        self, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Test stacking multiple SO3ConvolutionBlock layers preserves equivariance.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 5

        # Stack two layers
        layer1 = SO3ConvolutionBlock(
            in_channels=channels,
            hidden_channels=channels * 2,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer2 = SO3ConvolutionBlock(
            in_channels=channels,
            hidden_channels=channels * 2,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        layer1.eval()
        layer2.eval()

        # Create valid input (zero at invalid positions and m=0 imaginary)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        # Enforce m=0 imaginary = 0 constraint (real spherical harmonics property)
        x[:, :, 0, 1, :] = 0.0

        # Random rotation
        alpha = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi
        beta = torch.rand(batch_size, device=device, dtype=dtype) * math.pi
        gamma = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi

        with torch.no_grad():
            # Path 1: Rotate input, then apply layers
            x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
            y1 = layer2(layer1(x_rotated))

            # Path 2: Apply layers, then rotate output
            y = layer2(layer1(x))
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
        rtol *= 10  # Looser for composed layers
        atol *= 10

        torch.testing.assert_close(
            y1,
            y2,
            rtol=rtol,
            atol=atol,
            msg=(
                f"Composed layers violate SO(3) equivariance: "
                f"max diff={torch.abs(y1 - y2).max().item():.2e}"
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
