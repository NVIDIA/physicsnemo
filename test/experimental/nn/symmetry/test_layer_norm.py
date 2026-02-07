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

"""Unit tests for equivariant normalization layers.

Tests cover:
- Shape preservation and validation
- Invalid (l, m) positions remain zero
- m=0 imaginary component remains zero
- l=0 mean subtraction behavior
- l>0 scaling-only behavior (no mean subtraction)
- Degree balancing
- Multi-precision support (float16, bfloat16, float32, float64)
- SO(2) equivariance preservation
- Gradient flow
- torch.compile compatibility
- Determinism
"""

from __future__ import annotations

import math

import pytest
import torch

from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask
from physicsnemo.experimental.nn.symmetry.layer_norm import (
    EquivariantLayerNormGrid,
    EquivariantLayerNormSHGrid,
    EquivariantRMSNormSHGrid,
    make_degree_balance_weight,
    make_m0_imag_mask,
)
from physicsnemo.experimental.nn.symmetry.wigner import rotate_grid_coefficients
from test.experimental.nn.symmetry.conftest import get_rtol_atol

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=[(2, 2), (4, 2), (4, 4), (6, 3)])
def lmax_mmax(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterized fixture for lmax/mmax configurations.

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


@pytest.fixture(params=[(1, 0), (1, 1), (2, 1), (2, 2), (4, 2), (4, 4)])
def lmax_mmax_layernorm_sh(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterized fixture for lmax/mmax configurations for EquivariantLayerNormSHGrid.

    Note: EquivariantLayerNormSHGrid requires lmax >= 1.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    tuple[int, int]
        Tuple of (lmax, mmax) values where lmax >= 1.
    """
    return request.param


@pytest.fixture(params=[(0, 0), (1, 0), (1, 1), (2, 1)])
def lmax_mmax_small(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Small lmax/mmax configurations for EquivariantLayerNormGrid tests.

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
# Test Helper Utilities
# =============================================================================


class TestMakeDegreeBalanceWeight:
    """Tests for make_degree_balance_weight utility function."""

    def test_output_shape(self) -> None:
        """Output shape should be (lmax+1, mmax+1)."""
        lmax, mmax = 4, 2
        weights = make_degree_balance_weight(lmax, mmax)
        assert weights.shape == (lmax + 1, mmax + 1)

    def test_invalid_positions_zero(self) -> None:
        """Invalid (l, m) positions should have zero weight."""
        lmax, mmax = 4, 2
        weights = make_degree_balance_weight(lmax, mmax)
        mask = make_grid_mask(lmax, mmax)

        for l_idx in range(lmax + 1):
            for m_idx in range(mmax + 1):
                if not mask[l_idx, m_idx]:
                    assert weights[l_idx, m_idx] == 0.0, (
                        f"Invalid position ({l_idx}, {m_idx}) should have zero weight"
                    )

    def test_weights_sum_to_one(self) -> None:
        """Valid weights should sum to approximately 1.0."""
        for lmax in range(5):
            for mmax in range(lmax + 1):
                weights = make_degree_balance_weight(lmax, mmax)
                total = weights.sum().item()
                assert abs(total - 1.0) < 1e-5, (
                    f"Weights should sum to 1.0, got {total} for lmax={lmax}, mmax={mmax}"
                )

    def test_degree_balance(self) -> None:
        """Each degree should contribute equally when weighted."""
        lmax, mmax = 4, 4
        weights = make_degree_balance_weight(lmax, mmax)

        # Sum weights for each degree
        degree_contributions = []
        for l_idx in range(lmax + 1):
            # Sum over valid m for this l (which is min(l, mmax) + 1 entries)
            num_valid_m = min(l_idx, mmax) + 1
            degree_sum = weights[l_idx, :num_valid_m].sum().item()
            degree_contributions.append(degree_sum)

        # Each degree should contribute 1/(lmax+1)
        expected = 1.0 / (lmax + 1)
        for deg_idx, contrib in enumerate(degree_contributions):
            assert abs(contrib - expected) < 1e-5, (
                f"Degree {deg_idx} contribution should be {expected}, got {contrib}"
            )

    def test_validation_errors(self) -> None:
        """Should raise ValueError for invalid parameters."""
        with pytest.raises(ValueError, match="lmax must be non-negative"):
            make_degree_balance_weight(-1, 0)

        with pytest.raises(ValueError, match="mmax must be non-negative"):
            make_degree_balance_weight(2, -1)

        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            make_degree_balance_weight(2, 3)


class TestMakeM0ImagMask:
    """Tests for make_m0_imag_mask utility function."""

    def test_output_shape(self) -> None:
        """Output shape should be (1, 1, mmax+1, 2, 1)."""
        mmax = 3
        mask = make_m0_imag_mask(mmax)
        assert mask.shape == (1, 1, mmax + 1, 2, 1)

    def test_m0_imag_zero(self) -> None:
        """m=0 imaginary position should be zero."""
        mmax = 3
        mask = make_m0_imag_mask(mmax)
        assert mask[0, 0, 0, 1, 0] == 0.0

    def test_m0_real_one(self) -> None:
        """m=0 real position should be one."""
        mmax = 3
        mask = make_m0_imag_mask(mmax)
        assert mask[0, 0, 0, 0, 0] == 1.0

    def test_other_positions_one(self) -> None:
        """All other positions should be one."""
        mmax = 3
        mask = make_m0_imag_mask(mmax)

        for m in range(mmax + 1):
            for ri in range(2):
                if m == 0 and ri == 1:
                    continue  # Skip m=0 imaginary
                assert mask[0, 0, m, ri, 0] == 1.0, (
                    f"Position m={m}, ri={ri} should be 1.0"
                )

    def test_validation_errors(self) -> None:
        """Should raise ValueError for invalid mmax."""
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            make_m0_imag_mask(-1)


# =============================================================================
# Test EquivariantRMSNormSHGrid
# =============================================================================


class TestEquivariantRMSNormSHGrid:
    """Comprehensive tests for EquivariantRMSNormSHGrid."""

    def test_output_shape(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Output shape should match input shape.

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
        channels = 32
        batch_size = 50

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_invalid_positions_zero(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Invalid (l, m) positions should remain zero.

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
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        mask = make_grid_mask(lmax, mmax).to(device=device)
        for l_idx in range(lmax + 1):
            for m_idx in range(mmax + 1):
                if not mask[l_idx, m_idx]:
                    torch.testing.assert_close(
                        out[:, l_idx, m_idx, :, :],
                        torch.zeros_like(out[:, l_idx, m_idx, :, :]),
                        rtol=0,
                        atol=0,
                        msg=f"Invalid position (l={l_idx}, m={m_idx}) should be zero",
                    )

    def test_m0_imaginary_zero(
        self, lmax_mmax: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """m=0 imaginary component should remain zero.

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
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        # m=0 imaginary should be zero for all l
        m0_imag = out[:, :, 0, 1, :]
        torch.testing.assert_close(
            m0_imag,
            torch.zeros_like(m0_imag),
            rtol=0,
            atol=0,
            msg="m=0 imaginary should be zero",
        )

    @pytest.mark.parametrize("subtract_mean", [True, False])
    def test_subtract_mean(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax: tuple[int, int],
        subtract_mean: bool,
    ) -> None:
        """l=0 should have zero mean when subtract_mean=True.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        subtract_mean : bool
            Whether to subtract mean from l=0 features.
        """
        lmax, mmax = lmax_mmax
        channels = 32
        batch_size = 100

        norm = EquivariantRMSNormSHGrid(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            subtract_mean=subtract_mean,
            affine=False,
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )
        # Add a large offset to l=0 to test mean subtraction
        x[:, 0, 0, 0, :] += 10.0

        out = norm(x)

        # l=0, m=0, real component should have near-zero mean per sample
        l0_out = out[:, 0, 0, 0, :]  # [batch, channels]
        l0_mean = l0_out.mean(dim=-1)  # [batch]

        if subtract_mean:
            rtol, atol = get_rtol_atol(dtype, scale=10.0)
            torch.testing.assert_close(
                l0_mean,
                torch.zeros_like(l0_mean),
                rtol=rtol,
                atol=atol,
                msg="l=0 should be centered when subtract_mean=True",
            )
        else:
            # otherwise just make sure everything is finite
            assert torch.isfinite(l0_out).all()

    def test_backward_pass(
        self, dtype: torch.dtype, device: torch.device, lmax_mmax: tuple[int, int]
    ) -> None:
        """Gradients should flow to input and parameters.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        """
        lmax, mmax = lmax_mmax
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=True
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

        if norm.affine_weight is not None:
            assert norm.affine_weight.grad is not None, (
                "affine_weight gradients not computed"
            )
            assert torch.isfinite(norm.affine_weight.grad).all(), (
                "affine_weight gradients contain non-finite values"
            )

    @pytest.mark.parametrize("affine", [True, False])
    @pytest.mark.parametrize("subtract_mean", [True, False])
    @pytest.mark.parametrize(
        "alpha_val,beta_val,gamma_val",
        [
            (0.1, 0.2, 0.3),  # Small rotation
            (math.pi / 4, math.pi / 3, math.pi / 6),  # Medium rotation
            (math.pi, math.pi / 2, 0.0),  # Large rotation
            (0.0, math.pi, 0.0),  # Inversion through y-axis
            (2 * math.pi / 3, math.pi / 4, math.pi / 3),  # Arbitrary rotation
        ],
        ids=["small", "medium", "large", "y-inversion", "arbitrary"],
    )
    def test_equivariance_preserved(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax: tuple[int, int],
        affine: bool,
        subtract_mean: bool,
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
    ) -> None:
        """Normalization should commute with SO(3) rotation.

        Since the norm is a scalar (invariant under rotation), the normalization
        operation should commute with rotation: norm(rotate(x)) == rotate(norm(x)).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        affine : bool
            Whether to use affine transformation.
        subtract_mean : bool
            Whether to subtract mean from l=0 features.
        alpha_val : float
            First Euler angle (radians).
        beta_val : float
            Second Euler angle (radians).
        gamma_val : float
            Third Euler angle (radians).
        """
        lmax, mmax = lmax_mmax
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            affine=affine,
            subtract_mean=subtract_mean,
        ).to(device=device, dtype=dtype)

        # Create valid input
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        x[:, :, 0, 1, :] = 0.0  # Zero m=0 imaginary

        # Create Euler angle tensors
        alpha = torch.full((batch_size,), alpha_val, device=device, dtype=dtype)
        beta = torch.full((batch_size,), beta_val, device=device, dtype=dtype)
        gamma = torch.full((batch_size,), gamma_val, device=device, dtype=dtype)

        with torch.no_grad():
            # Method 1: Rotate input, then apply layer
            x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
            y1 = norm(x_rotated)

            # Method 2: Apply layer, then rotate output
            y = norm(x)
            y2 = rotate_grid_coefficients(y, (alpha, beta, gamma))

        # Rescale tolerance based on dtype
        # Note: Normalization layers have higher numerical errors under SO(3) rotations
        # compared to linear layers due to the normalization operation
        match dtype:
            case torch.float32:
                scaling = 1e4
            case torch.float16:
                scaling = 1e4
            case torch.bfloat16:
                scaling = 1e4
            case torch.float64:
                scaling = 1e7
            case _:
                scaling = 1.0
        rtol, atol = get_rtol_atol(dtype, scaling)

        torch.testing.assert_close(
            y1,
            y2,
            rtol=rtol,
            atol=atol,
            msg=f"Equivariance violated: max diff = {(y1 - y2).abs().max():.2e}",
        )

    def test_torch_compile_nograd(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax: tuple[int, int],
        compile_config: tuple[str, str],
    ) -> None:
        """Forward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        compile_config : tuple[str, str]
            Tuple of (backend, mode) for torch.compile.
        """
        lmax, mmax = lmax_mmax
        compile_backend, compile_mode = compile_config
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )
        norm.eval()

        if compile_backend == "cudagraphs":
            compiled_norm = torch.compile(norm, backend=compile_backend)
        else:
            compiled_norm = torch.compile(
                norm, mode=compile_mode, backend=compile_backend
            )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            ref_out = norm(x)
            out = compiled_norm(x)

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(ref_out, out, rtol=rtol, atol=atol)

    def test_torch_compile_withgrad(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax: tuple[int, int],
        compile_config: tuple[str, str],
    ) -> None:
        """Backward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        compile_config : tuple[str, str]
            Tuple of (backend, mode) for torch.compile.
        """
        lmax, mmax = lmax_mmax
        compile_backend, compile_mode = compile_config
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        if compile_backend == "cudagraphs":
            compiled_norm = torch.compile(norm, backend=compile_backend)
        else:
            compiled_norm = torch.compile(
                norm, mode=compile_mode, backend=compile_backend
            )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = compiled_norm(x)
        loss = ((torch.randn_like(out) - out) ** 2.0).mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self, dtype: torch.dtype, device: torch.device) -> None:
        """Each batch element should be processed independently.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )
        norm.eval()

        x = torch.randn(2, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype)

        with torch.no_grad():
            y_batch = norm(x)
            y0 = norm(x[0:1])
            y1 = norm(x[1:2])

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(
            y_batch[0],
            y0[0],
            rtol=rtol,
            atol=atol,
            msg="Batch processing should match individual processing for sample 0",
        )
        torch.testing.assert_close(
            y_batch[1],
            y1[0],
            rtol=rtol,
            atol=atol,
            msg="Batch processing should match individual processing for sample 1",
        )

    def test_batch_size_one(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with batch size of 1.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 1

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_single_channel(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with single channel.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 1
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_lmax0_mmax0(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with lmax=0, mmax=0 (scalar only).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 0, 0
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_no_affine(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with affine=False.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=False
        ).to(device=device, dtype=dtype)

        assert norm.affine_weight is None
        assert norm.affine_bias is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_affine_weight_shape(self) -> None:
        """Test affine weight shapes."""
        lmax, mmax = 4, 2
        channels = 16

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels)
        assert norm.affine_weight.shape == (lmax + 1, channels)
        assert norm.affine_bias.shape == (channels,)

    def test_no_balance(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with std_balance_degrees=False.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, std_balance_degrees=False
        ).to(device=device, dtype=dtype)

        assert norm.balance_degree_weight is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_balance_vs_no_balance_different(
        self, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Outputs should differ with and without degree balancing.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm_balanced = EquivariantRMSNormSHGrid(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            std_balance_degrees=True,
            affine=False,
        ).to(device=device, dtype=dtype)

        norm_unbalanced = EquivariantRMSNormSHGrid(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            std_balance_degrees=False,
            affine=False,
        ).to(device=device, dtype=dtype)

        torch.manual_seed(42)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            out_balanced = norm_balanced(x)
            out_unbalanced = norm_unbalanced(x)

        # Outputs should be different (unless the input happens to have
        # balanced energy, which is unlikely)
        diff = (out_balanced - out_unbalanced).abs().max()
        assert diff > 1e-6, "Balanced and unbalanced outputs should differ"

    def test_deterministic_output(self) -> None:
        """Same input should produce same output."""
        torch.manual_seed(42)
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantRMSNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels)

        torch.manual_seed(123)
        x1 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, channels)

        torch.manual_seed(123)
        x2 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, channels)

        with torch.no_grad():
            y1 = norm(x1)
            y2 = norm(x2)

        torch.testing.assert_close(y1, y2, msg="Forward pass should be deterministic")

    def test_extra_repr(self) -> None:
        """Test string representation."""
        norm = EquivariantRMSNormSHGrid(lmax=4, mmax=2, num_channels=64)
        repr_str = repr(norm)
        assert "lmax=4" in repr_str
        assert "mmax=2" in repr_str
        assert "num_channels=64" in repr_str

    def test_invalid_lmax(self) -> None:
        """lmax must be non-negative."""
        with pytest.raises(ValueError, match="lmax must be non-negative"):
            EquivariantRMSNormSHGrid(lmax=-1, mmax=0, num_channels=16)

    def test_invalid_mmax_negative(self) -> None:
        """mmax must be non-negative."""
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            EquivariantRMSNormSHGrid(lmax=2, mmax=-1, num_channels=16)

    def test_invalid_mmax_gt_lmax(self) -> None:
        """mmax must be <= lmax."""
        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            EquivariantRMSNormSHGrid(lmax=2, mmax=3, num_channels=16)

    def test_invalid_channels(self) -> None:
        """num_channels must be positive."""
        with pytest.raises(ValueError, match="num_channels must be positive"):
            EquivariantRMSNormSHGrid(lmax=2, mmax=2, num_channels=0)

    def test_invalid_input_shape(self) -> None:
        """Should raise error if input shape doesn't match."""
        norm = EquivariantRMSNormSHGrid(lmax=4, mmax=2, num_channels=16)
        x = torch.randn(10, 3, 3, 2, 16)  # Wrong lmax

        with pytest.raises(ValueError, match="Expected input shape"):
            norm(x)


# =============================================================================
# Test EquivariantLayerNormSHGrid
# =============================================================================


class TestEquivariantLayerNormSHGrid:
    """Comprehensive tests for EquivariantLayerNormSHGrid."""

    def test_output_shape(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Output shape should match input shape.

        Parameters
        ----------
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 32
        batch_size = 50

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_invalid_positions_zero(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Invalid (l, m) positions should remain zero.

        Parameters
        ----------
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        mask = make_grid_mask(lmax, mmax).to(device=device)
        for l_idx in range(lmax + 1):
            for m_idx in range(mmax + 1):
                if not mask[l_idx, m_idx]:
                    torch.testing.assert_close(
                        out[:, l_idx, m_idx, :, :],
                        torch.zeros_like(out[:, l_idx, m_idx, :, :]),
                        rtol=0,
                        atol=0,
                        msg=f"Invalid position (l={l_idx}, m={m_idx}) should be zero",
                    )

    def test_m0_imaginary_zero(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """m=0 imaginary component should remain zero.

        Parameters
        ----------
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        # m=0 imaginary should be zero for all l
        m0_imag = out[:, :, 0, 1, :]
        torch.testing.assert_close(
            m0_imag,
            torch.zeros_like(m0_imag),
            rtol=0,
            atol=0,
            msg="m=0 imaginary should be zero",
        )

    def test_l0_uses_layernorm(self, dtype: torch.dtype, device: torch.device) -> None:
        """l=0 should be processed with LayerNorm (zero mean, unit variance).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 32
        batch_size = 100

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=False
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        # l=0, m=0, real component should have zero mean and unit variance
        l0_out = out[:, 0, 0, 0, :]  # [batch, channels]

        # Check mean is near zero (per sample)
        l0_mean = l0_out.mean(dim=-1)
        rtol, atol = get_rtol_atol(dtype, scale=10.0)
        torch.testing.assert_close(
            l0_mean,
            torch.zeros_like(l0_mean),
            rtol=rtol,
            atol=atol,
            msg="l=0 should have zero mean after LayerNorm",
        )

    def test_backward_pass(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax_layernorm_sh: tuple[int, int],
    ) -> None:
        """Gradients should flow to input and parameters.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=True
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    @pytest.mark.parametrize(
        "alpha_val,beta_val,gamma_val",
        [
            (0.1, 0.2, 0.3),  # Small rotation
            (math.pi / 4, math.pi / 3, math.pi / 6),  # Medium rotation
            (math.pi, math.pi / 2, 0.0),  # Large rotation
            (0.0, math.pi, 0.0),  # Inversion through y-axis
            (2 * math.pi / 3, math.pi / 4, math.pi / 3),  # Arbitrary rotation
        ],
        ids=["small", "medium", "large", "y-inversion", "arbitrary"],
    )
    def test_equivariance_preserved(
        self,
        dtype: torch.dtype,
        device: torch.device,
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
    ) -> None:
        """Normalization should commute with SO(3) rotation.

        Parameters
        ----------
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
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)

        # Create valid input
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        x[:, :, 0, 1, :] = 0.0  # Zero m=0 imaginary

        # Create Euler angle tensors
        alpha = torch.full((batch_size,), alpha_val, device=device, dtype=dtype)
        beta = torch.full((batch_size,), beta_val, device=device, dtype=dtype)
        gamma = torch.full((batch_size,), gamma_val, device=device, dtype=dtype)

        with torch.no_grad():
            # Method 1: Rotate input, then apply layer
            x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
            y1 = norm(x_rotated)

            # Method 2: Apply layer, then rotate output
            y = norm(x)
            y2 = rotate_grid_coefficients(y, (alpha, beta, gamma))

        # Rescale tolerance based on dtype
        # Note: Normalization layers have higher numerical errors under SO(3) rotations
        # compared to linear layers due to the normalization operation
        match dtype:
            case torch.float32:
                scaling = 1e4
            case torch.float16:
                scaling = 1e4
            case torch.bfloat16:
                scaling = 1e4
            case torch.float64:
                scaling = 1e7
            case _:
                scaling = 1.0
        rtol, atol = get_rtol_atol(dtype, scaling)

        torch.testing.assert_close(
            y1,
            y2,
            rtol=rtol,
            atol=atol,
            msg=f"Equivariance violated: max diff = {(y1 - y2).abs().max():.2e}",
        )

    def test_torch_compile_nograd(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax_layernorm_sh: tuple[int, int],
        compile_config: tuple[str, str],
    ) -> None:
        """Forward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        compile_config : tuple[str, str]
            Tuple of (backend, mode) for torch.compile.
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        compile_backend, compile_mode = compile_config
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)
        norm.eval()

        if compile_backend == "cudagraphs":
            compiled_norm = torch.compile(norm, backend=compile_backend)
        else:
            compiled_norm = torch.compile(
                norm, mode=compile_mode, backend=compile_backend
            )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            ref_out = norm(x)
            out = compiled_norm(x)

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(ref_out, out, rtol=rtol, atol=atol)

    def test_torch_compile_withgrad(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax_layernorm_sh: tuple[int, int],
        compile_config: tuple[str, str],
    ) -> None:
        """Backward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        compile_config : tuple[str, str]
            Tuple of (backend, mode) for torch.compile.
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        compile_backend, compile_mode = compile_config
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)

        if compile_backend == "cudagraphs":
            compiled_norm = torch.compile(norm, backend=compile_backend)
        else:
            compiled_norm = torch.compile(
                norm, mode=compile_mode, backend=compile_backend
            )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = compiled_norm(x)
        loss = ((torch.randn_like(out) - out) ** 2.0).mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self, dtype: torch.dtype, device: torch.device) -> None:
        """Each batch element should be processed independently.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)
        norm.eval()

        x = torch.randn(2, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype)

        with torch.no_grad():
            y_batch = norm(x)
            y0 = norm(x[0:1])
            y1 = norm(x[1:2])

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(
            y_batch[0],
            y0[0],
            rtol=rtol,
            atol=atol,
            msg="Batch processing should match individual processing for sample 0",
        )
        torch.testing.assert_close(
            y_batch[1],
            y1[0],
            rtol=rtol,
            atol=atol,
            msg="Batch processing should match individual processing for sample 1",
        )

    def test_batch_size_one(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with batch size of 1.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 1

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_single_channel(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with single channel.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 1
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_no_affine(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with affine=False.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=False
        ).to(device=device, dtype=dtype)

        assert norm.affine_weight is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_affine_weight_shape(self) -> None:
        """Test affine weight shapes."""
        lmax, mmax = 4, 2
        channels = 16

        norm = EquivariantLayerNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels)
        assert norm.affine_weight.shape == (lmax, channels)

    def test_no_balance(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with std_balance_degrees=False.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, std_balance_degrees=False
        ).to(device=device, dtype=dtype)

        assert norm.balance_degree_weight is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_balance_vs_no_balance_different(
        self, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Outputs should differ with and without degree balancing.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm_balanced = EquivariantLayerNormSHGrid(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            std_balance_degrees=True,
            affine=False,
        ).to(device=device, dtype=dtype)

        norm_unbalanced = EquivariantLayerNormSHGrid(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            std_balance_degrees=False,
            affine=False,
        ).to(device=device, dtype=dtype)

        torch.manual_seed(42)
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            out_balanced = norm_balanced(x)
            out_unbalanced = norm_unbalanced(x)

        # Outputs should be different
        diff = (out_balanced - out_unbalanced).abs().max()
        assert diff > 1e-6, "Balanced and unbalanced outputs should differ"

    def test_deterministic_output(self) -> None:
        """Same input should produce same output."""
        torch.manual_seed(42)
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormSHGrid(lmax=lmax, mmax=mmax, num_channels=channels)

        torch.manual_seed(123)
        x1 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, channels)

        torch.manual_seed(123)
        x2 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, channels)

        with torch.no_grad():
            y1 = norm(x1)
            y2 = norm(x2)

        torch.testing.assert_close(y1, y2, msg="Forward pass should be deterministic")

    def test_extra_repr(self) -> None:
        """Test string representation."""
        norm = EquivariantLayerNormSHGrid(lmax=4, mmax=2, num_channels=64)
        repr_str = repr(norm)
        assert "lmax=4" in repr_str
        assert "mmax=2" in repr_str
        assert "num_channels=64" in repr_str

    def test_invalid_lmax(self) -> None:
        """lmax must be >= 1."""
        with pytest.raises(ValueError, match="lmax must be >= 1"):
            EquivariantLayerNormSHGrid(lmax=0, mmax=0, num_channels=16)

    def test_invalid_mmax_negative(self) -> None:
        """mmax must be non-negative."""
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            EquivariantLayerNormSHGrid(lmax=2, mmax=-1, num_channels=16)

    def test_invalid_mmax_gt_lmax(self) -> None:
        """mmax must be <= lmax."""
        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            EquivariantLayerNormSHGrid(lmax=2, mmax=3, num_channels=16)

    def test_invalid_channels(self) -> None:
        """num_channels must be positive."""
        with pytest.raises(ValueError, match="num_channels must be positive"):
            EquivariantLayerNormSHGrid(lmax=2, mmax=2, num_channels=0)

    def test_invalid_input_shape(self) -> None:
        """Should raise error if input shape doesn't match."""
        norm = EquivariantLayerNormSHGrid(lmax=4, mmax=2, num_channels=16)
        x = torch.randn(10, 3, 3, 2, 16)  # Wrong lmax

        with pytest.raises(ValueError, match="Expected input shape"):
            norm(x)


# =============================================================================
# Test EquivariantLayerNormGrid
# =============================================================================


class TestEquivariantLayerNormGrid:
    """Comprehensive tests for EquivariantLayerNormGrid."""

    def test_output_shape(
        self, lmax_mmax_small: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Output shape should match input shape.

        Parameters
        ----------
        lmax_mmax_small : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax_small
        channels = 32
        batch_size = 50

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_invalid_positions_zero(
        self, lmax_mmax_small: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """Invalid (l, m) positions should remain zero.

        Parameters
        ----------
        lmax_mmax_small : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax_small
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        mask = make_grid_mask(lmax, mmax).to(device=device)
        for l_idx in range(lmax + 1):
            for m_idx in range(mmax + 1):
                if not mask[l_idx, m_idx]:
                    torch.testing.assert_close(
                        out[:, l_idx, m_idx, :, :],
                        torch.zeros_like(out[:, l_idx, m_idx, :, :]),
                        rtol=0,
                        atol=0,
                        msg=f"Invalid position (l={l_idx}, m={m_idx}) should be zero",
                    )

    def test_m0_imaginary_zero(
        self, lmax_mmax_small: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> None:
        """m=0 imaginary component should remain zero.

        Parameters
        ----------
        lmax_mmax_small : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = lmax_mmax_small
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        m0_imag = out[:, :, 0, 1, :]
        torch.testing.assert_close(
            m0_imag,
            torch.zeros_like(m0_imag),
            rtol=0,
            atol=0,
            msg="m=0 imaginary should be zero",
        )

    def test_backward_pass(self, dtype: torch.dtype, device: torch.device) -> None:
        """Gradients should flow to input and parameters.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 3, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradients not computed"
        assert torch.isfinite(x.grad).all(), "Input gradients contain non-finite values"

    @pytest.mark.parametrize(
        "alpha_val,beta_val,gamma_val",
        [
            (0.1, 0.2, 0.3),  # Small rotation
            (math.pi / 4, math.pi / 3, math.pi / 6),  # Medium rotation
            (math.pi, math.pi / 2, 0.0),  # Large rotation
            (0.0, math.pi, 0.0),  # Inversion through y-axis
            (2 * math.pi / 3, math.pi / 4, math.pi / 3),  # Arbitrary rotation
        ],
        ids=["small", "medium", "large", "y-inversion", "arbitrary"],
    )
    def test_equivariance_preserved(
        self,
        dtype: torch.dtype,
        device: torch.device,
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
    ) -> None:
        """Normalization should commute with SO(3) rotation.

        Parameters
        ----------
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
        lmax, mmax = 3, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        # Create valid input
        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )
        mask = make_grid_mask(lmax, mmax).to(device=device, dtype=dtype)
        x = x * mask[None, :, :, None, None]
        x[:, :, 0, 1, :] = 0.0  # Zero m=0 imaginary

        # Create Euler angle tensors
        alpha = torch.full((batch_size,), alpha_val, device=device, dtype=dtype)
        beta = torch.full((batch_size,), beta_val, device=device, dtype=dtype)
        gamma = torch.full((batch_size,), gamma_val, device=device, dtype=dtype)

        with torch.no_grad():
            # Method 1: Rotate input, then apply layer
            x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
            y1 = norm(x_rotated)

            # Method 2: Apply layer, then rotate output
            y = norm(x)
            y2 = rotate_grid_coefficients(y, (alpha, beta, gamma))

        # Rescale tolerance based on dtype
        # Note: Normalization layers have higher numerical errors under SO(3) rotations
        # compared to linear layers due to the normalization operation
        match dtype:
            case torch.float32:
                scaling = 1e4
            case torch.float16:
                scaling = 1e4
            case torch.bfloat16:
                scaling = 1e4
            case torch.float64:
                scaling = 1e7
            case _:
                scaling = 1.0
        rtol, atol = get_rtol_atol(dtype, scaling)

        torch.testing.assert_close(
            y1,
            y2,
            rtol=rtol,
            atol=atol,
            msg=f"Equivariance violated: max diff = {(y1 - y2).abs().max():.2e}",
        )

    def test_torch_compile_nograd(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax_small: tuple[int, int],
        compile_config: tuple[str, str],
    ) -> None:
        """Forward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax_small : tuple[int, int]
            Tuple of (lmax, mmax) values.
        compile_config : tuple[str, str]
            Tuple of (backend, mode) for torch.compile.
        """
        lmax, mmax = lmax_mmax_small
        compile_backend, compile_mode = compile_config
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )
        norm.eval()

        if compile_backend == "cudagraphs":
            compiled_norm = torch.compile(norm, backend=compile_backend)
        else:
            compiled_norm = torch.compile(
                norm, mode=compile_mode, backend=compile_backend
            )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        with torch.no_grad():
            ref_out = norm(x)
            out = compiled_norm(x)

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(ref_out, out, rtol=rtol, atol=atol)

    def test_torch_compile_withgrad(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax_small: tuple[int, int],
        compile_config: tuple[str, str],
    ) -> None:
        """Backward pass should work with torch.compile.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        lmax_mmax_small : tuple[int, int]
            Tuple of (lmax, mmax) values.
        compile_config : tuple[str, str]
            Tuple of (backend, mode) for torch.compile.
        """
        lmax, mmax = lmax_mmax_small
        compile_backend, compile_mode = compile_config
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        if compile_backend == "cudagraphs":
            compiled_norm = torch.compile(norm, backend=compile_backend)
        else:
            compiled_norm = torch.compile(
                norm, mode=compile_mode, backend=compile_backend
            )

        x = torch.randn(
            batch_size,
            lmax + 1,
            mmax + 1,
            2,
            channels,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        out = compiled_norm(x)
        loss = ((torch.randn_like(out) - out) ** 2.0).mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self, dtype: torch.dtype, device: torch.device) -> None:
        """Each batch element should be processed independently.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 3, 2
        channels = 16

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )
        norm.eval()

        x = torch.randn(2, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype)

        with torch.no_grad():
            y_batch = norm(x)
            y0 = norm(x[0:1])
            y1 = norm(x[1:2])

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(
            y_batch[0],
            y0[0],
            rtol=rtol,
            atol=atol,
            msg="Batch processing should match individual processing for sample 0",
        )
        torch.testing.assert_close(
            y_batch[1],
            y1[0],
            rtol=rtol,
            atol=atol,
            msg="Batch processing should match individual processing for sample 1",
        )

    def test_batch_size_one(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with batch size of 1.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 1

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_single_channel(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with single channel.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 1
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_lmax0_mmax0(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with lmax=0, mmax=0 (scalar only).

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 0, 0
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_no_affine(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with affine=False.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=False
        ).to(device=device, dtype=dtype)

        assert norm.affine_weight is None
        assert norm.affine_bias is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_affine_weight_shape(self) -> None:
        """Test affine weight shapes."""
        lmax, mmax = 4, 2
        channels = 16

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels)
        assert norm.affine_weight.shape == (lmax + 1, channels)
        assert norm.affine_bias.shape == (channels,)

    def test_subtract_mean(self, dtype: torch.dtype, device: torch.device) -> None:
        """Test with subtract_mean=True/False.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(
            lmax=lmax, mmax=mmax, num_channels=channels, subtract_mean=False
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_deterministic_output(self) -> None:
        """Same input should produce same output."""
        torch.manual_seed(42)
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = EquivariantLayerNormGrid(lmax=lmax, mmax=mmax, num_channels=channels)

        torch.manual_seed(123)
        x1 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, channels)

        torch.manual_seed(123)
        x2 = torch.randn(batch_size, lmax + 1, mmax + 1, 2, channels)

        with torch.no_grad():
            y1 = norm(x1)
            y2 = norm(x2)

        torch.testing.assert_close(y1, y2, msg="Forward pass should be deterministic")

    def test_extra_repr(self) -> None:
        """Test string representation."""
        norm = EquivariantLayerNormGrid(lmax=4, mmax=2, num_channels=64)
        repr_str = repr(norm)
        assert "lmax=4" in repr_str
        assert "mmax=2" in repr_str
        assert "num_channels=64" in repr_str

    def test_invalid_lmax(self) -> None:
        """lmax must be non-negative."""
        with pytest.raises(ValueError, match="lmax must be non-negative"):
            EquivariantLayerNormGrid(lmax=-1, mmax=0, num_channels=16)

    def test_invalid_mmax_negative(self) -> None:
        """mmax must be non-negative."""
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            EquivariantLayerNormGrid(lmax=2, mmax=-1, num_channels=16)

    def test_invalid_mmax_gt_lmax(self) -> None:
        """mmax must be <= lmax."""
        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            EquivariantLayerNormGrid(lmax=2, mmax=3, num_channels=16)

    def test_invalid_channels(self) -> None:
        """num_channels must be positive."""
        with pytest.raises(ValueError, match="num_channels must be positive"):
            EquivariantLayerNormGrid(lmax=2, mmax=2, num_channels=0)

    def test_invalid_input_shape(self) -> None:
        """Should raise error if input shape doesn't match."""
        norm = EquivariantLayerNormGrid(lmax=4, mmax=2, num_channels=16)
        x = torch.randn(10, 3, 3, 2, 16)  # Wrong lmax

        with pytest.raises(ValueError, match="Expected input shape"):
            norm(x)
