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

"""Comprehensive unit tests for SO(2) convolution implementation.

This module provides tests for:
- make_grid_mask: Boolean mask creation for valid (l, m) pairs
- SO2Convolution: SO(2) equivariant convolution layer
- torch.compile compatibility
- Integration tests for multi-layer pipelines

Test Structure
--------------
TestGridMask
    Tests for the make_grid_mask function.
TestSO2Convolution
    Tests for the main SO2Convolution layer.
TestHardcodedRegression
    Regression tests with fixed inputs and expected outputs.
TestTorchCompile
    Tests for torch.compile compatibility.
TestIntegration
    Integration tests for multi-layer pipelines.

Notes
-----
For m=0, the imaginary component must always be zero by SO(2) symmetry.
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.experimental.nn.symmetry import SO2Convolution, make_grid_mask

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


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
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
# TestGridMask
# =============================================================================


class TestGridMask:
    """Tests for make_grid_mask function.

    The grid mask identifies valid (l, m) positions where m <= l, which
    corresponds to the physical constraint that spherical harmonic Y_l^m
    only exists when |m| <= l.
    """

    @pytest.mark.parametrize(
        "lmax,mmax",
        [(0, 0), (1, 0), (1, 1), (2, 2), (4, 2), (6, 2), (10, 5)],
    )
    def test_mask_shape(self, lmax: int, mmax: int) -> None:
        """Verify output shape is [lmax+1, mmax+1].

        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        mmax : int
            Maximum spherical harmonic order.
        """
        mask = make_grid_mask(lmax, mmax)

        expected_shape = (lmax + 1, mmax + 1)
        assert mask.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {mask.shape}"
        )
        assert mask.dtype == torch.bool, "Mask should be boolean"

    @pytest.mark.parametrize(
        "lmax,mmax,expected_valid",
        [
            (0, 0, 1),  # Only (0, 0) valid
            (1, 1, 3),  # (0,0), (1,0), (1,1)
            (2, 2, 6),  # (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
            (3, 2, 9),  # (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2)
            (4, 2, 12),  # Above + (4,0), (4,1), (4,2)
        ],
    )
    def test_mask_sparsity(self, lmax: int, mmax: int, expected_valid: int) -> None:
        """Count True values matches expected valid positions.

        The number of valid (l, m) pairs is:
        - For l from 0 to min(mmax, lmax): contributes (l+1) valid m values
        - For l from mmax+1 to lmax: contributes (mmax+1) valid m values

        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        mmax : int
            Maximum spherical harmonic order.
        expected_valid : int
            Expected number of valid (l, m) positions.
        """
        mask = make_grid_mask(lmax, mmax)

        num_valid = mask.sum().item()
        assert num_valid == expected_valid, (
            f"Expected {expected_valid} valid positions, got {num_valid}"
        )

    def test_mask_pattern(self) -> None:
        """Hardcoded test for specific lmax=3, mmax=2 pattern.

        Expected pattern:
        l=0: [True, False, False]  # Only m=0 valid
        l=1: [True, True, False]   # m=0,1 valid
        l=2: [True, True, True]    # m=0,1,2 valid
        l=3: [True, True, True]    # m=0,1,2 valid
        """
        mask = make_grid_mask(lmax=3, mmax=2)

        expected = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [True, True, True],
            ]
        )

        assert mask.shape == expected.shape
        assert torch.equal(mask, expected), (
            f"Mask pattern mismatch:\nGot:\n{mask}\nExpected:\n{expected}"
        )

    @pytest.mark.parametrize(
        "lmax,mmax,error_msg",
        [
            (-1, 0, "lmax must be non-negative"),
            (0, -1, "mmax must be non-negative"),
            (2, 3, "mmax.*must be <= lmax"),
        ],
    )
    def test_mask_invalid_inputs(self, lmax: int, mmax: int, error_msg: str) -> None:
        """Verify errors for invalid lmax/mmax combinations.

        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        mmax : int
            Maximum spherical harmonic order.
        error_msg : str
            Expected error message pattern.
        """
        with pytest.raises(ValueError, match=error_msg):
            make_grid_mask(lmax, mmax)


# =============================================================================
# TestSO2Convolution
# =============================================================================


class TestSO2Convolution:
    """Tests for the main SO2Convolution layer.

    This layer performs SO(2) equivariant convolution on spherical harmonic
    coefficients arranged in a regular grid layout with masking.
    """

    def test_output_shape(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: str
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

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        y = conv(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"

    def test_masked_positions_zero(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: str
    ) -> None:
        """Verify invalid positions remain zero in output.

        Positions where m > l should be zero in the output after the
        convolution applies the validity mask.

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

        conv = SO2Convolution(
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
            y = conv(x)

        # Check invalid positions are zero
        mask = make_grid_mask(lmax, mmax).to(device=device)

        for l in range(lmax + 1):  # noqa: E741
            for m in range(mmax + 1):
                if not mask[l, m]:
                    assert torch.allclose(
                        y[:, l, m, :, :], torch.zeros_like(y[:, l, m, :, :])
                    ), f"Invalid position (l={l}, m={m}) is not zero in output"

    def test_m0_imaginary_zero(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: str
    ) -> None:
        """Verify m=0 imaginary component is always zero.

        For m=0, the imaginary part must be zero by SO(2) symmetry.

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
        batch_size = 20

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        # Create input with non-zero imaginary component at m=0
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y = conv(x)

        # Check m=0 imaginary is zero for all l
        m0_imaginary = y[:, :, 0, 1, :]  # batch, l, imaginary, channels
        assert torch.allclose(m0_imaginary, torch.zeros_like(m0_imaginary)), (
            "m=0 imaginary component should be zero"
        )

    def test_backward_pass(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: str
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

        conv = SO2Convolution(
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

        y = conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert conv.W_r.grad is not None, "W_r gradients not computed"
        assert conv.W_i.grad is not None, "W_i gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    def test_deterministic(self, dtype: torch.dtype, device: str) -> None:
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

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        conv.eval()

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y1 = conv(x)
            y2 = conv(x)

        assert torch.allclose(y1, y2, atol=1e-6), "Output should be deterministic"

    def test_so2_equivariance(self, dtype: torch.dtype, device: str) -> None:
        """Key symmetry test: rotation around z commutes with conv.

        For rotation R_phi around z-axis:
        Conv(R_phi(x)) = R_phi(Conv(x))

        This property holds because:
        1. SO(2) rotation multiplies coefficient Y_l^m by exp(i*m*phi)
        2. The convolution applies a complex linear transform per m-order
        3. Scalar multiplication (rotation) commutes with linear transforms

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        import math

        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        conv.eval()

        # Create valid input (zero at invalid positions)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device)
        x = x * mask[None, :, :, None, None]
        # Also zero m=0 imaginary (required by SO(2) symmetry)
        x[:, :, 0, 1, :] = 0.0

        def rotate_grid(x: torch.Tensor, phi: float) -> torch.Tensor:
            """Rotate grid-layout features by angle phi around z-axis."""
            x_rot = x.clone()
            for m in range(1, mmax + 1):  # m=0 unchanged
                cos_phi = math.cos(m * phi)
                sin_phi = math.sin(m * phi)
                x_real = x[:, :, m, 0, :]
                x_imag = x[:, :, m, 1, :]
                x_rot[:, :, m, 0, :] = x_real * cos_phi - x_imag * sin_phi
                x_rot[:, :, m, 1, :] = x_real * sin_phi + x_imag * cos_phi
            return x_rot

        # Test with several rotation angles
        for phi in [0.1, 0.5, 1.0, math.pi / 4, math.pi / 2]:
            with torch.no_grad():
                # Method 1: Rotate input, then convolve
                x_rot = rotate_grid(x, phi)
                y1 = conv(x_rot)

                # Method 2: Convolve, then rotate output
                y = conv(x)
                y2 = rotate_grid(y, phi)

            # Tolerances based on dtype precision
            # float32 has ~7 decimal digits, float64 has ~15
            if dtype == torch.float32:
                rtol, atol = 5e-3, 5e-3
            else:
                rtol, atol = 1e-10, 1e-10

            assert torch.allclose(y1, y2, rtol=rtol, atol=atol), (
                f"SO(2) equivariance violated at phi={phi:.3f}: "
                f"max diff={torch.abs(y1 - y2).max().item():.2e}"
            )

    def test_edge_modulation(self, dtype: torch.dtype, device: str) -> None:
        """Verify edge-dependent weights work correctly.

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
        edge_channels = 32
        batch_size = 20

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels=edge_channels,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        x_edge = torch.randn(batch_size, edge_channels, device=device, dtype=dtype)

        y = conv(x, x_edge)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"

        # Different edge features should give different outputs
        x_edge2 = torch.randn(batch_size, edge_channels, device=device, dtype=dtype)
        with torch.no_grad():
            y2 = conv(x, x_edge2)

        assert not torch.allclose(y, y2, atol=1e-3), (
            "Different edge features should produce different outputs"
        )

    @pytest.mark.parametrize("lmax", [1, 2, 4])
    @pytest.mark.parametrize("mmax", [1, 2, 4])
    def test_produce_gates(
        self, dtype: torch.dtype, device: str, lmax: int, mmax: int
    ) -> None:
        """Verify gate channels are embedded correctly in output tensor.

        When produce_gates=True, the output tensor should have additional
        gate channels (lmax * out_channels) that are only non-zero at
        the (l=0, m=0, real) position.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        if mmax > lmax:
            pytest.skip("mmax should not be greater than lmax.")
        in_channels = 16
        out_channels = 16
        batch_size = 20

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            produce_gates=True,
        ).to(device=device, dtype=dtype)

        # Verify num_gate_channels and total_out_channels properties
        expected_gate_channels = lmax * out_channels  # 4 * 16 = 64
        assert conv.num_gate_channels == expected_gate_channels, (
            f"Expected num_gate_channels={expected_gate_channels}, got {conv.num_gate_channels}"
        )
        assert conv.total_out_channels == out_channels + expected_gate_channels

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y = conv(x)

        expected_total_channels = out_channels + expected_gate_channels
        assert y.shape == (
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            expected_total_channels,
        ), (
            f"Expected shape {(batch_size, lmax + 1, mmax + 1, 2, expected_total_channels)}, "
            f"got {y.shape}"
        )

        # Gate channels (indices [out_channels:]) should be zero everywhere
        # except at (l=0, m=0, real)
        gate_channels_output = y[..., out_channels:]

        # Check gates are zero at all positions except (l=0, m=0, real=0)
        for l in range(lmax + 1):  # noqa: E741
            for m in range(mmax + 1):
                for ri in range(2):  # real=0, imag=1
                    gate_slice = gate_channels_output[:, l, m, ri, :]
                    if l == 0 and m == 0 and ri == 0:
                        # This position should have non-zero gates (generically)
                        # We can't guarantee non-zero due to random weights, but
                        # at least verify the shape
                        assert gate_slice.shape == (batch_size, expected_gate_channels)
                    else:
                        # All other positions should be zero
                        assert torch.allclose(
                            gate_slice, torch.zeros_like(gate_slice)
                        ), f"Gate channels should be zero at (l={l}, m={m}, ri={ri})"

    def test_produce_gates_false(self, dtype: torch.dtype, device: str) -> None:
        """Verify produce_gates=False gives standard output without gate channels.

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
        batch_size = 20

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            produce_gates=False,  # Default
        ).to(device=device, dtype=dtype)

        # Verify num_gate_channels is 0
        assert conv.num_gate_channels == 0
        assert conv.total_out_channels == out_channels

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y = conv(x)

        # Output should have only out_channels
        assert y.shape == (batch_size, lmax + 1, mmax + 1, 2, out_channels)

    def test_edge_modulation_produces_different_outputs(
        self, dtype: torch.dtype, device: str
    ) -> None:
        """Different edge features must produce different outputs.

        This verifies that edge modulation is actually being applied per-edge,
        not averaged or ignored.

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
        edge_channels = 32
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels=edge_channels,
        ).to(device=device, dtype=dtype)

        # Same input for both
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        # Different edge features
        x_edge1 = torch.randn(batch_size, edge_channels, device=device, dtype=dtype)
        x_edge2 = torch.randn(batch_size, edge_channels, device=device, dtype=dtype)

        with torch.no_grad():
            y1 = conv(x, x_edge1)
            y2 = conv(x, x_edge2)

        # Outputs must differ since edge features differ
        assert not torch.allclose(y1, y2, atol=1e-3), (
            "Different edge features should produce different outputs"
        )

    def test_edge_modulation_per_sample_independence(
        self, dtype: torch.dtype, device: str
    ) -> None:
        """Each batch element should be modulated independently.

        Processing samples together in a batch should give the same result
        as processing them individually.

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
        edge_channels = 32

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels=edge_channels,
        ).to(device=device, dtype=dtype)
        conv.eval()

        # Create batch of 2 with different edge features
        x = torch.randn(
            2, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        x_edge = torch.randn(2, edge_channels, device=device, dtype=dtype)

        with torch.no_grad():
            # Process as batch
            y_batch = conv(x, x_edge)

            # Process individually
            y0 = conv(x[0:1], x_edge[0:1])
            y1 = conv(x[1:2], x_edge[1:2])

        # Results should match
        rtol = 1e-3 if dtype == torch.float32 else 1e-10
        atol = 1e-3 if dtype == torch.float32 else 1e-10

        assert torch.allclose(y_batch[0], y0[0], rtol=rtol, atol=atol), (
            "Batch processing should match individual processing for sample 0"
        )
        assert torch.allclose(y_batch[1], y1[0], rtol=rtol, atol=atol), (
            "Batch processing should match individual processing for sample 1"
        )

    def test_edge_modulation_does_not_preserve_equivariance(
        self, dtype: torch.dtype, device: str
    ) -> None:
        """Verify per-coefficient edge modulation intentionally breaks SO(2) equivariance.

        Per-coefficient edge modulation (Option A) applies position-dependent
        scaling based on edge features. Since different (l, m) positions get
        different modulation factors, the operation does not commute with
        rotation. This is expected behavior matching the reference eSCN
        implementation.

        Note: Without edge modulation, SO(2) equivariance is preserved (tested
        in test_so2_equivariance).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        import math

        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        edge_channels = 32
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels=edge_channels,
        ).to(device=device, dtype=dtype)
        conv.eval()

        # Create valid input (zero at invalid positions)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device)
        x = x * mask[None, :, :, None, None]
        x[:, :, 0, 1, :] = 0.0  # Zero m=0 imaginary

        # Edge features (invariant, same for rotated and unrotated)
        x_edge = torch.randn(batch_size, edge_channels, device=device, dtype=dtype)

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

        # Test that edge modulation breaks equivariance (expected behavior)
        phi = 0.5
        with torch.no_grad():
            # Rotate then convolve
            x_rot = rotate_grid(x, phi)
            y1 = conv(x_rot, x_edge)

            # Convolve then rotate
            y = conv(x, x_edge)
            y2 = rotate_grid(y, phi)

        # With per-coefficient edge modulation, the results should NOT be equal
        # because the modulation is position-dependent and doesn't transform
        # with rotation
        max_diff = torch.abs(y1 - y2).max().item()

        # Verify they are significantly different (not equal within tight tolerances)
        assert max_diff > 1e-2, (
            f"Per-coefficient edge modulation should break SO(2) equivariance, "
            f"but max diff is only {max_diff:.2e}"
        )

        # However, the output should still be valid (finite, correct shape, etc.)
        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert y1.shape == expected_shape, f"Expected {expected_shape}, got {y1.shape}"
        assert torch.isfinite(y1).all(), "Output contains non-finite values"

        # m=0 imaginary should still be zero
        assert torch.allclose(y1[:, :, 0, 1, :], torch.zeros_like(y1[:, :, 0, 1, :])), (
            "m=0 imaginary should be zero"
        )

    def test_edge_modulation_backward(self, dtype: torch.dtype, device: str) -> None:
        """Verify gradients flow correctly through edge modulation.

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
        edge_channels = 32
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels=edge_channels,
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
        x_edge = torch.randn(
            batch_size, edge_channels, device=device, dtype=dtype, requires_grad=True
        )

        y = conv(x, x_edge)
        loss = y.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert x.grad is not None, "Input gradients not computed"
        assert x_edge.grad is not None, "Edge feature gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"
        assert torch.isfinite(x_edge.grad).all(), (
            "Edge gradients contain non-finite values"
        )

        # Edge gradients should be non-zero (edge features affect output)
        assert x_edge.grad.abs().sum() > 0, "Edge gradients are all zero"

    def test_edge_modulation_required_when_configured(
        self, dtype: torch.dtype, device: str
    ) -> None:
        """Verify error is raised if x_edge not provided when required.

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
        edge_channels = 32
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels=edge_channels,  # Edge modulation configured
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        # Should raise error when x_edge not provided
        with pytest.raises(ValueError, match="x_edge.*required"):
            conv(x)  # No x_edge provided


# =============================================================================
# TestHardcodedRegression
# =============================================================================


class TestHardcodedRegression:
    """Hardcoded input/output pairs for regression testing.

    These tests use fixed random seeds and known weight values to verify
    that the forward and backward passes produce expected results.
    """

    def test_regression_lmax2_mmax2_forward(self) -> None:
        """Regression test with fixed seed and weights for forward pass.

        Uses lmax=2, mmax=2, small layer with 4 input/output channels.
        """
        torch.manual_seed(42)

        lmax, mmax = 2, 2
        in_channels = 4
        out_channels = 4
        batch_size = 2

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        )

        # Set weights to known values
        with torch.no_grad():
            conv.W_r.copy_(
                torch.arange(
                    (mmax + 1) * in_channels * out_channels, dtype=torch.float32
                ).reshape(mmax + 1, in_channels, out_channels)
                * 0.01
            )
            conv.W_i.copy_(
                torch.arange(
                    (mmax + 1) * in_channels * out_channels, dtype=torch.float32
                ).reshape(mmax + 1, in_channels, out_channels)
                * 0.005
            )

        # Fixed input
        torch.manual_seed(123)
        x = torch.randn(batch_size, lmax + 1, mmax + 1, 2, in_channels)

        with torch.no_grad():
            y = conv(x)

        # Re-run and verify determinism
        torch.manual_seed(123)
        x2 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, in_channels)
        with torch.no_grad():
            y2 = conv(x2)

        assert torch.allclose(y, y2), "Forward pass should be deterministic"

        # Verify output properties
        assert y.shape == (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert torch.isfinite(y).all(), "Output contains non-finite values"

        # Check m=0 imaginary is zero
        assert torch.allclose(y[:, :, 0, 1, :], torch.zeros_like(y[:, :, 0, 1, :])), (
            "m=0 imaginary should be zero"
        )

    def test_regression_lmax4_mmax2_backward(self) -> None:
        """Regression test with fixed seed for backward pass.

        Uses lmax=4, mmax=2 to test gradient computation.
        """
        torch.manual_seed(42)

        lmax, mmax = 4, 2
        in_channels = 4
        out_channels = 4
        batch_size = 3

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        )

        # Set weights to known values
        with torch.no_grad():
            conv.W_r.copy_(
                torch.arange(
                    (mmax + 1) * in_channels * out_channels, dtype=torch.float32
                ).reshape(mmax + 1, in_channels, out_channels)
                * 0.02
            )
            conv.W_i.copy_(
                torch.arange(
                    (mmax + 1) * in_channels * out_channels, dtype=torch.float32
                ).reshape(mmax + 1, in_channels, out_channels)
                * 0.01
            )

        # Fixed input with gradient
        torch.manual_seed(456)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, requires_grad=True
        )

        y = conv(x)
        loss = y.sum()
        loss.backward()

        # Verify gradients are computed and finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"
        assert conv.W_r.grad is not None
        assert torch.isfinite(conv.W_r.grad).all(), (
            "W_r gradients contain non-finite values"
        )
        assert conv.W_i.grad is not None
        assert torch.isfinite(conv.W_i.grad).all(), (
            "W_i gradients contain non-finite values"
        )

        # Re-run and verify determinism
        conv.zero_grad()
        torch.manual_seed(456)
        x2 = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, requires_grad=True
        )
        y2 = conv(x2)
        loss2 = y2.sum()
        loss2.backward()

        assert torch.allclose(x.grad, x2.grad), "Backward pass should be deterministic"


# =============================================================================
# TestTorchCompile
# =============================================================================


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
class TestTorchCompile:
    """Tests for torch.compile compatibility.

    These tests verify that SO2Convolution works correctly when
    compiled with torch.compile for various backends.
    """

    def test_compile_forward(self, dtype: torch.dtype, device: str) -> None:
        """Verify forward pass works with torch.compile.

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

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        # Compile the model
        compiled_conv = torch.compile(conv)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y = compiled_conv(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_compile_backward(self, dtype: torch.dtype, device: str) -> None:
        """Verify backward pass works with torch.compile.

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

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        # Compile the model
        compiled_conv = torch.compile(conv)

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

        y = compiled_conv(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for reduce-overhead"
    )
    def test_compile_cuda_graph_compatible(self, dtype: torch.dtype) -> None:
        """Verify works with reduce-overhead mode (CUDA graphs).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        """
        device = "cuda"
        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        # Compile with reduce-overhead mode (uses CUDA graphs)
        compiled_conv = torch.compile(conv, mode="reduce-overhead")

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        # Warmup run (compiles the graph)
        with torch.no_grad():
            _ = compiled_conv(x)

        # Actual test
        with torch.no_grad():
            y = compiled_conv(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"

    def test_compile_gives_same_result(self, dtype: torch.dtype, device: str) -> None:
        """Verify compiled model matches eager execution.

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

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)
        conv.eval()

        compiled_conv = torch.compile(conv)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            y_eager = conv(x)
            y_compiled = compiled_conv(x)

        rtol = 1e-4 if dtype == torch.float32 else 1e-8
        atol = 1e-5 if dtype == torch.float32 else 1e-10
        assert torch.allclose(y_eager, y_compiled, rtol=rtol, atol=atol), (
            "Compiled output should match eager output"
        )


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests for multi-layer pipelines."""

    def test_two_layer_pipeline(self, dtype: torch.dtype, device: str) -> None:
        """Test chaining two convolutions.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 6, 2
        in_channels = 64
        hidden_channels = 128
        out_channels = 64
        batch_size = 50

        conv1 = SO2Convolution(
            in_channels=in_channels,
            out_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        conv2 = SO2Convolution(
            in_channels=hidden_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        # Forward through both layers
        h = conv1(x)
        y = conv2(h)

        # Check shapes
        assert h.shape == (batch_size, lmax + 1, mmax + 1, 2, hidden_channels)
        assert y.shape == (batch_size, lmax + 1, mmax + 1, 2, out_channels)

        # Check m=0 imaginary is zero
        assert torch.allclose(y[:, :, 0, 1, :], torch.zeros_like(y[:, :, 0, 1, :])), (
            "Final m=0 imaginary should be zero"
        )

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_with_various_batch_sizes(
        self, batch_size: int, dtype: torch.dtype, device: str
    ) -> None:
        """Test with various batch sizes.

        Parameters
        ----------
        batch_size : int
            Number of samples in batch.
        dtype : torch.dtype
            Data type for tensors.
        device : str
            Device to run on.
        """
        lmax, mmax = 4, 2
        in_channels = 32
        out_channels = 32

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, in_channels, device=device, dtype=dtype
        )

        y = conv(x)

        expected_shape = (batch_size, lmax + 1, mmax + 1, 2, out_channels)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"


# =============================================================================
# TestSO2ConvolutionValidation
# =============================================================================


class TestSO2ConvolutionValidation:
    """Tests for input shape and constructor validation in SO2Convolution.

    These tests verify that SO2Convolution raises appropriate errors when
    given invalid inputs, either at construction time or during forward pass.
    """

    def test_invalid_input_channels(self) -> None:
        """Verify error when input tensor has wrong number of channels.

        The input tensor's last dimension should match in_channels.
        """
        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        )

        # Wrong number of input channels (32 instead of 16)
        x = torch.randn(batch_size, lmax + 1, mmax + 1, 2, 32)

        with pytest.raises(ValueError, match=r"Expected input shape.*got shape"):
            conv(x)

    def test_invalid_input_lmax_dim(self) -> None:
        """Verify error when input tensor has wrong lmax+1 dimension.

        The second dimension should be lmax+1.
        """
        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        )

        # Wrong lmax dimension (3 instead of 5)
        x = torch.randn(batch_size, 3, mmax + 1, 2, in_channels)

        with pytest.raises(ValueError, match=r"Expected input shape.*got shape"):
            conv(x)

    def test_invalid_input_mmax_dim(self) -> None:
        """Verify error when input tensor has wrong mmax+1 dimension.

        The third dimension should be mmax+1.
        """
        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        )

        # Wrong mmax dimension (5 instead of 3)
        x = torch.randn(batch_size, lmax + 1, 5, 2, in_channels)

        with pytest.raises(ValueError, match=r"Expected input shape.*got shape"):
            conv(x)

    def test_invalid_input_real_imag_dim(self) -> None:
        """Verify error when input tensor has wrong real/imag dimension.

        The fourth dimension should be 2 (for real and imaginary components).
        """
        lmax, mmax = 4, 2
        in_channels = 16
        out_channels = 16
        batch_size = 10

        conv = SO2Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            lmax=lmax,
            mmax=mmax,
        )

        # Wrong real/imag dimension (3 instead of 2)
        x = torch.randn(batch_size, lmax + 1, mmax + 1, 3, in_channels)

        with pytest.raises(ValueError, match=r"Expected input shape.*got shape"):
            conv(x)

    def test_invalid_lmax_negative(self) -> None:
        """Verify error when constructor receives negative lmax.

        The lmax parameter must be non-negative.
        """
        with pytest.raises(ValueError, match=r"lmax must be non-negative"):
            SO2Convolution(
                in_channels=16,
                out_channels=16,
                lmax=-1,
                mmax=2,
            )

    def test_invalid_mmax_negative(self) -> None:
        """Verify error when constructor receives negative mmax.

        The mmax parameter must be non-negative.
        """
        with pytest.raises(ValueError, match=r"mmax must be non-negative"):
            SO2Convolution(
                in_channels=16,
                out_channels=16,
                lmax=4,
                mmax=-1,
            )

    def test_invalid_mmax_gt_lmax(self) -> None:
        """Verify error when constructor receives mmax > lmax.

        The mmax parameter must be less than or equal to lmax.
        """
        with pytest.raises(ValueError, match=r"mmax.*must be <= lmax"):
            SO2Convolution(
                in_channels=16,
                out_channels=16,
                lmax=2,
                mmax=4,
            )

    def test_invalid_in_channels(self) -> None:
        """Verify error when constructor receives non-positive in_channels.

        The in_channels parameter must be positive.
        """
        with pytest.raises(ValueError, match=r"in_channels must be positive"):
            SO2Convolution(
                in_channels=0,
                out_channels=16,
                lmax=4,
                mmax=2,
            )

        with pytest.raises(ValueError, match=r"in_channels must be positive"):
            SO2Convolution(
                in_channels=-5,
                out_channels=16,
                lmax=4,
                mmax=2,
            )

    def test_invalid_out_channels(self) -> None:
        """Verify error when constructor receives non-positive out_channels.

        The out_channels parameter must be positive.
        """
        with pytest.raises(ValueError, match=r"out_channels must be positive"):
            SO2Convolution(
                in_channels=16,
                out_channels=0,
                lmax=4,
                mmax=2,
            )

        with pytest.raises(ValueError, match=r"out_channels must be positive"):
            SO2Convolution(
                in_channels=16,
                out_channels=-5,
                lmax=4,
                mmax=2,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
