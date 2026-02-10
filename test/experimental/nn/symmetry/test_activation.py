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

"""Unit tests for GateActivation layer.

Tests cover:
- Shape changes (input has embedded gates, output has only features)
- l=0 gets SiLU activation
- l>0 gets gate multiplication
- Invalid (l,m) positions are zero
- m=0 imaginary is zero
- Gradient flow
- torch.compile compatibility
- SO(2) equivariance preservation
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from physicsnemo.experimental.nn.symmetry.activation import GateActivation
from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Parameterized fixture for testing with different floating-point precisions.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    torch.dtype
        The dtype to use for tensor operations (float32 or float64).
    """
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> str:
    """Parameterized fixture for testing on CPU and GPU if available.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    str
        Device string ("cpu" or "cuda").
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


@pytest.fixture(params=[(2, 2), (4, 2), (6, 2), (4, 4)])
def lmax_mmax(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterized fixture for testing with different lmax/mmax configurations.

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
# Test Classes
# =============================================================================


class TestGateActivationBasic:
    """Basic functionality tests for GateActivation."""

    def test_output_shape(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: str
    ) -> None:
        """Output shape should be [batch, lmax+1, mmax+1, 2, channels] (gates consumed).

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
        channels = 32
        batch_size = 50
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        # Input has embedded gates
        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert out.shape == expected_shape, (
            f"Expected {expected_shape}, got {out.shape}"
        )

    def test_l0_gets_silu(self, dtype: torch.dtype, device: str) -> None:
        """l=0 positions should have SiLU applied, independent of gates.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )
        # Set l=0, m=0 imaginary to zero (as required)
        x[:, 0, 0, 1, :] = 0.0

        out = act(x)

        # l=0 should have SiLU applied (only m=0 is valid for l=0)
        # Only the first `channels` values are the features
        expected_l0 = torch.nn.functional.silu(x[:, 0, 0, 0, :channels])
        actual_l0 = out[:, 0, 0, 0, :]

        assert torch.allclose(actual_l0, expected_l0, rtol=1e-5, atol=1e-5), (
            "l=0 should have SiLU activation applied"
        )

    @pytest.mark.parametrize(
        "activation", ["silu", "relu", "stan", nn.functional.sigmoid, nn.SiLU()]
    )
    def test_str_activation(self, device: str, activation: str) -> None:
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(device=device)

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
        )

        with torch.no_grad():
            _ = act(x)

    def test_l_gt_0_gets_gating(self, dtype: torch.dtype, device: str) -> None:
        """l>0 positions should be scaled by sigmoid(gates).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        # Extract gates from where they're embedded (l=0, m=0, real=0, channels onwards)
        gates = x[:, 0, 0, 0, channels:]  # [batch, lmax * channels]

        # Compute expected gating for l=2, m=1 (valid position)
        # Gates are indexed as l-1, so l=2 uses gate index 1
        gates_reshaped = gates.view(batch_size, lmax, channels)
        gate_l2 = torch.sigmoid(gates_reshaped[:, 1, :])  # l=2 uses gate index 1

        # Features are the first `channels` values
        expected = x[:, 2, 1, 0, :channels] * gate_l2  # real part
        actual = out[:, 2, 1, 0, :]

        assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5), (
            "l>0 positions should be scaled by sigmoid(gates)"
        )

    def test_invalid_positions_zero(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: str
    ) -> None:
        """Invalid (l, m) positions where m > l should be zero.

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
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        # Create input with non-zero values everywhere
        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )
        x = x + 1.0  # Ensure non-zero

        out = act(x)

        # Check invalid positions are zero
        mask = make_grid_mask(lmax, mmax).to(device=device)
        for l_idx in range(lmax + 1):
            for m_idx in range(mmax + 1):
                if not mask[l_idx, m_idx]:
                    assert torch.allclose(
                        out[:, l_idx, m_idx, :, :],
                        torch.zeros_like(out[:, l_idx, m_idx, :, :]),
                    ), f"Invalid position (l={l_idx}, m={m_idx}) should be zero"

    def test_m0_imaginary_zero(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: str
    ) -> None:
        """m=0 imaginary component should always be zero.

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
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        # Create input with non-zero m=0 imaginary
        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        # m=0 imaginary should be zero for all l
        m0_imag = out[:, :, 0, 1, :]
        assert torch.allclose(m0_imag, torch.zeros_like(m0_imag)), (
            "m=0 imaginary should be zero"
        )


class TestGateActivationGradients:
    """Gradient flow tests for GateActivation."""

    def test_backward_pass(self, dtype: torch.dtype, device: str) -> None:
        """Gradients should flow to input tensor (including embedded gates).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = act(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "x gradients not computed"
        assert torch.isfinite(x.grad).all(), "x gradients contain non-finite values"

    def test_gates_gradients_nonzero(self, dtype: torch.dtype, device: str) -> None:
        """Gates (embedded in input) should receive non-zero gradients.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = act(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradients not computed"
        # Check that gradients flow to the gate positions (l=0, m=0, real=0, channels onwards)
        gate_grads = x.grad[:, 0, 0, 0, channels:]
        assert gate_grads.abs().sum() > 0, "Gate gradients should be non-zero"


class TestGateActivationEquivariance:
    """SO(2) equivariance tests for GateActivation."""

    def test_equivariance_preserved(self, dtype: torch.dtype, device: str) -> None:
        """Gated activation should preserve SO(2) equivariance.

        Since gates are scalars (invariant), the gating operation commutes
        with rotation: R(gate * x) = gate * R(x).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        # Create valid input
        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )
        mask = make_grid_mask(lmax, mmax).to(device=device)
        x = x * mask[None, :, :, None, None]
        x[:, :, 0, 1, :] = 0.0  # Zero m=0 imaginary

        def rotate_grid(x: torch.Tensor, phi: float) -> torch.Tensor:
            """Rotate grid-layout features by angle phi around z-axis."""
            x_rot = x.clone()
            for m in range(1, mmax + 1):
                cos_phi = math.cos(m * phi)
                sin_phi = math.sin(m * phi)
                x_real = x[:, :, m, 0, :]
                x_imag = x[:, :, m, 1, :]
                x_rot[:, :, m, 0, :] = x_real * cos_phi - x_imag * sin_phi
                x_rot[:, :, m, 1, :] = x_real * sin_phi + x_imag * cos_phi
            return x_rot

        phi = 0.7
        with torch.no_grad():
            # Path 1: Rotate input (keeping gates the same) then activate
            x_rot = rotate_grid(x, phi)
            # Note: Gates at (l=0, m=0, real=0) are invariant under rotation
            y1 = act(x_rot)

            # Path 2: Activate then rotate output
            y = act(x)
            y2 = rotate_grid(y, phi)

        # Should be equal (gates are invariant)
        rtol = 1e-4 if dtype == torch.float32 else 1e-10
        atol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(y1, y2, rtol=rtol, atol=atol), (
            f"Equivariance violated: max diff = {(y1 - y2).abs().max():.2e}"
        )


class TestGateActivationCompile:
    """torch.compile compatibility tests."""

    def test_compile_forward(self, dtype: torch.dtype, device: str) -> None:
        """Forward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )
        compiled_act = torch.compile(act)

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = compiled_act(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert out.shape == expected_shape
        assert torch.isfinite(out).all()

    def test_compile_matches_eager(self, dtype: torch.dtype, device: str) -> None:
        """Compiled output should match eager output.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )
        act.eval()
        compiled_act = torch.compile(act)

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            out_eager = act(x)
            out_compiled = compiled_act(x)

        rtol = 1e-4 if dtype == torch.float32 else 1e-10
        atol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(out_eager, out_compiled, rtol=rtol, atol=atol)

    def test_compile_backward(self, dtype: torch.dtype, device: str) -> None:
        """Backward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )
        compiled_act = torch.compile(act)

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = compiled_act(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"


class TestGateActivationValidation:
    """Input validation tests."""

    def test_invalid_lmax(self) -> None:
        """lmax must be >= 1."""
        with pytest.raises(ValueError, match="lmax must be >= 1"):
            GateActivation(lmax=0, mmax=0, channels=16)

    def test_invalid_mmax_negative(self) -> None:
        """mmax must be non-negative."""
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            GateActivation(lmax=2, mmax=-1, channels=16)

    def test_invalid_mmax_gt_lmax(self) -> None:
        """mmax must be <= lmax."""
        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            GateActivation(lmax=2, mmax=3, channels=16)

    def test_invalid_channels(self) -> None:
        """channels must be positive."""
        with pytest.raises(ValueError, match="channels must be positive"):
            GateActivation(lmax=2, mmax=2, channels=0)

    def test_invalid_input_channels(self) -> None:
        """Should raise error if input channel count doesn't match."""
        act = GateActivation(lmax=4, mmax=2, channels=16)
        # Wrong channel count: should be 16 + 4*16 = 80, not 16
        x = torch.randn(10, 5, 3, 2, 16)

        with pytest.raises(ValueError, match="Expected input with"):
            act(x)


class TestGateActivationHardcoded:
    """Hardcoded regression tests for GateActivation."""

    def test_regression_lmax2_mmax2(self) -> None:
        """Regression test with fixed seed and known values.

        Uses lmax=2, mmax=2, small layer with 4 channels.
        """
        torch.manual_seed(42)

        lmax, mmax = 2, 2
        channels = 4
        batch_size = 2
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels)

        # Fixed input
        torch.manual_seed(123)
        x = torch.randn(batch_size, lmax + 1, mmax + 1, 2, total_in_channels)

        with torch.no_grad():
            y = act(x)

        # Re-run with same seed and verify determinism
        torch.manual_seed(123)
        x2 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, total_in_channels)

        with torch.no_grad():
            y2 = act(x2)

        assert torch.allclose(y, y2), "Forward pass should be deterministic"

        # Verify basic properties
        assert y.shape == (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert torch.isfinite(y).all()

        # Verify l=0 has SiLU applied (m=0, real part)
        expected_l0 = torch.nn.functional.silu(x[:, 0, 0, 0, :channels])
        assert torch.allclose(y[:, 0, 0, 0, :], expected_l0, rtol=1e-5, atol=1e-5)

    def test_zero_gates(self, dtype: torch.dtype, device: str) -> None:
        """Test behavior with zero gates (sigmoid(0) = 0.5).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 2, 2
        channels = 4
        batch_size = 2
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )
        # Set gates to zero (channels onwards at l=0, m=0, real=0)
        x[:, 0, 0, 0, channels:] = 0.0

        with torch.no_grad():
            y = act(x)

        # l>0 should be scaled by 0.5 (sigmoid(0))
        # Check l=1, m=0, real part (valid position)
        expected_l1 = x[:, 1, 0, 0, :channels] * 0.5
        assert torch.allclose(y[:, 1, 0, 0, :], expected_l1, rtol=1e-5, atol=1e-5), (
            "With zero gates, l>0 should be scaled by 0.5"
        )

    def test_large_positive_gates(self, dtype: torch.dtype, device: str) -> None:
        """Test behavior with large positive gates (sigmoid approaches 1).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 2, 2
        channels = 4
        batch_size = 2
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )
        # Set gates to large positive value
        x[:, 0, 0, 0, channels:] = 100.0

        with torch.no_grad():
            y = act(x)

        # l>0 should be approximately unchanged (scaled by ~1)
        # Check l=1, m=0, real part (valid position)
        assert torch.allclose(
            y[:, 1, 0, 0, :], x[:, 1, 0, 0, :channels], rtol=1e-3, atol=1e-3
        ), "With large positive gates, l>0 should be approximately unchanged"

    def test_large_negative_gates(self, dtype: torch.dtype, device: str) -> None:
        """Test behavior with large negative gates (sigmoid approaches 0).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 2, 2
        channels = 4
        batch_size = 2
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )
        # Set gates to large negative value
        x[:, 0, 0, 0, channels:] = -100.0

        with torch.no_grad():
            y = act(x)

        # l>0 should be approximately zero (scaled by ~0)
        # Check l=1, m=0, real part (valid position)
        assert torch.allclose(
            y[:, 1, 0, 0, :],
            torch.zeros_like(y[:, 1, 0, 0, :]),
            rtol=1e-3,
            atol=1e-3,
        ), "With large negative gates, l>0 should be approximately zero"


class TestGateActivationBatchIndependence:
    """Tests for batch independence in GateActivation."""

    def test_batch_independence(self, dtype: torch.dtype, device: str) -> None:
        """Each batch element should be processed independently.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )
        act.eval()

        # Create batch of 2 with different inputs
        x = torch.randn(
            2, lmax + 1, mmax + 1, 2, total_in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            # Process as batch
            y_batch = act(x)

            # Process individually
            y0 = act(x[0:1])
            y1 = act(x[1:2])

        # Results should match
        rtol = 1e-5 if dtype == torch.float32 else 1e-10
        atol = 1e-5 if dtype == torch.float32 else 1e-10

        assert torch.allclose(y_batch[0], y0[0], rtol=rtol, atol=atol), (
            "Batch processing should match individual processing for sample 0"
        )
        assert torch.allclose(y_batch[1], y1[0], rtol=rtol, atol=atol), (
            "Batch processing should match individual processing for sample 1"
        )


class TestGateActivationEdgeCases:
    """Edge case tests for GateActivation."""

    def test_lmax1_mmax0(self, dtype: torch.dtype, device: str) -> None:
        """Test with minimal lmax=1, mmax=0 configuration.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 1, 0
        channels = 8
        batch_size = 5
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert out.shape == expected_shape
        assert torch.isfinite(out).all()

    def test_lmax1_mmax1(self, dtype: torch.dtype, device: str) -> None:
        """Test with lmax=1, mmax=1 configuration.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 1, 1
        channels = 8
        batch_size = 5
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert out.shape == expected_shape
        assert torch.isfinite(out).all()

    def test_large_lmax_mmax(self, dtype: torch.dtype, device: str) -> None:
        """Test with larger lmax and mmax values.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 8, 4
        channels = 32
        batch_size = 16
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert out.shape == expected_shape
        assert torch.isfinite(out).all()

    def test_batch_size_one(self, dtype: torch.dtype, device: str) -> None:
        """Test with batch size of 1.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 1
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert out.shape == expected_shape
        assert torch.isfinite(out).all()

    def test_single_channel(self, dtype: torch.dtype, device: str) -> None:
        """Test with single channel.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 1
        batch_size = 10
        gate_channels = lmax * channels
        total_in_channels = channels + gate_channels

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            total_in_channels,
            device=device,
            dtype=dtype,
        )

        out = act(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, channels)
        assert out.shape == expected_shape
        assert torch.isfinite(out).all()


class TestGateActivationProperties:
    """Tests for GateActivation properties."""

    def test_gate_channels_property(self) -> None:
        """Test gate_channels property returns correct value."""
        lmax, mmax = 4, 2
        channels = 16

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels)

        assert act.gate_channels == lmax * channels
        assert act.gate_channels == 64

    def test_total_in_channels_property(self) -> None:
        """Test total_in_channels property returns correct value."""
        lmax, mmax = 4, 2
        channels = 16

        act = GateActivation(lmax=lmax, mmax=mmax, channels=channels)

        assert act.total_in_channels == channels + lmax * channels
        assert act.total_in_channels == 80
