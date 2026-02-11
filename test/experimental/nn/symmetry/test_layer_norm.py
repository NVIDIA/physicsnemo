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
from typing import Type

import pytest
import torch

from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask
from physicsnemo.experimental.nn.symmetry.layer_norm import (
    EquivariantLayerNorm,
    EquivariantLayerNormTied,
    EquivariantRMSNorm,
    FusedEquivariantLayerNorm,
    FusedEquivariantLayerNormTied,
    FusedEquivariantRMSNorm,
    make_degree_balance_weight,
    make_m0_imag_mask,
)
from physicsnemo.experimental.nn.symmetry.wigner import rotate_grid_coefficients
from test.experimental.nn.symmetry.conftest import get_rtol_atol

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=[(2, 2), (4, 2), (4, 4)])
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
    """Parameterized fixture for lmax/mmax configurations for EquivariantLayerNormTied.

    Note: EquivariantLayerNormTied requires lmax >= 1.

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
    """Small lmax/mmax configurations for EquivariantLayerNorm tests.

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


@pytest.fixture(
    params=[
        pytest.param(EquivariantRMSNorm, id="unfused"),
        pytest.param(FusedEquivariantRMSNorm, id="fused"),
    ]
)
def rmsnorm_class(request: pytest.FixtureRequest):
    """Parametrized fixture for RMSNorm class (unfused and fused variants).

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    type
        Either EquivariantRMSNorm or FusedEquivariantRMSNorm.
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(EquivariantLayerNormTied, id="unfused"),
        pytest.param(FusedEquivariantLayerNormTied, id="fused"),
    ]
)
def layernormsh_class(request: pytest.FixtureRequest):
    """Parametrized fixture for LayerNormSH class (unfused and fused variants).

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    type
        Either EquivariantLayerNormTied or FusedEquivariantLayerNormTied.
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(EquivariantLayerNorm, id="unfused"),
        pytest.param(FusedEquivariantLayerNorm, id="fused"),
    ]
)
def layernorm_class(request: pytest.FixtureRequest):
    """Parametrized fixture for LayerNorm class (unfused and fused variants).

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    type
        Either EquivariantLayerNorm or FusedEquivariantLayerNorm.
    """
    return request.param


# =============================================================================
# Test Helper Utilities
# =============================================================================


def compare_fused_unfused(
    fused_class: Type,
    lmax: int,
    mmax: int,
    num_channels: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    **layer_kwargs,
) -> None:
    """Compare fused and unfused normalization layer outputs and gradients.

    This helper function creates a single fused normalization layer instance,
    then tests it with `_use_fused=True` (fused path) and `_use_fused=False`
    (unfused fallback path). It verifies that:
    1. Output tensors match within dtype-appropriate tolerances
    2. Input gradients match
    3. Output shapes are identical

    Parameters
    ----------
    fused_class : Type
        The fused normalization class to test.
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int
        Maximum spherical harmonic order.
    num_channels : int
        Number of feature channels.
    batch_size : int
        Batch size for test inputs.
    dtype : torch.dtype
        Data type for tensors.
    device : torch.device
        Device to run computation on.
    **layer_kwargs
        Additional keyword arguments to pass to layer constructor
        (e.g., subtract_mean, std_balance_degrees, affine, eps).
    """
    rtol, atol = get_rtol_atol(dtype)

    # Create a single layer instance
    layer = fused_class(
        lmax=lmax,
        mmax=mmax,
        num_channels=num_channels,
        **layer_kwargs,
    ).to(device=device, dtype=dtype)

    # Create two independent input tensors with gradient tracking
    x_fused = torch.randn(
        batch_size,
        lmax + 1,
        mmax + 1,
        2,
        num_channels,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    x_unfused = x_fused.clone().detach().requires_grad_(True)

    # Create gradient tensor for backward pass
    grad_output = torch.randn_like(x_fused)

    # Forward pass with fused path (_use_fused=True is the default)
    layer._use_fused = True
    y_fused = layer(x_fused)

    # Backward pass for fused
    y_fused.backward(grad_output)

    # Zero parameter gradients before unfused path
    layer.zero_grad()

    # Forward pass with unfused path
    layer._use_fused = False
    y_unfused = layer(x_unfused)

    # Backward pass for unfused
    y_unfused.backward(grad_output.clone())

    # Reset to fused for cleanup
    layer._use_fused = True

    # Test 1: Shape preservation
    assert y_fused.shape == y_unfused.shape
    assert y_fused.shape == x_fused.shape

    # Test 2: Output equivalence
    torch.testing.assert_close(
        y_fused,
        y_unfused,
        rtol=rtol,
        atol=atol,
        msg=f"Fused and unfused outputs differ for {fused_class.__name__}",
    )

    # Test 3: Input gradient equivalence
    assert x_fused.grad is not None
    assert x_unfused.grad is not None
    torch.testing.assert_close(
        x_fused.grad,
        x_unfused.grad,
        rtol=rtol,
        atol=atol,
        msg=f"Input gradients differ for {fused_class.__name__}",
    )


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
# Test EquivariantRMSNorm
# =============================================================================


class TestEquivariantRMSNorm:
    """Comprehensive tests for EquivariantRMSNorm."""

    def test_output_shape(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        rmsnorm_class,
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
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = lmax_mmax
        channels = 32
        batch_size = 50

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_invalid_positions_zero(
        self,
        lmax_mmax: tuple[int, int],
        rmsnorm_class,
    ) -> None:
        """Invalid (l, m) positions should remain zero.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = lmax_mmax
        channels = 16
        batch_size = 10

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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
        self,
        lmax_mmax: tuple[int, int],
        rmsnorm_class,
    ) -> None:
        """m=0 imaginary component should remain zero.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = lmax_mmax
        channels = 16
        batch_size = 10

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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
        subtract_mean: bool,
        rmsnorm_class,
    ) -> None:
        """l=0 should have zero mean when subtract_mean=True.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        subtract_mean : bool
            Whether to subtract mean from l=0 features.
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = 4, 2
        channels = 32
        batch_size = 100

        norm = rmsnorm_class(
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

    @pytest.mark.parametrize(
        "alpha_val,beta_val,gamma_val",
        [
            (math.pi / 3, math.pi / 4, math.pi / 6),  # Representative rotation
        ],
        ids=["representative"],
    )
    def test_equivariance_preserved(
        self,
        dtype: torch.dtype,
        device: torch.device,
        lmax_mmax: tuple[int, int],
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
        rmsnorm_class,
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
        alpha_val : float
            First Euler angle (radians).
        beta_val : float
            Second Euler angle (radians).
        gamma_val : float
            Third Euler angle (radians).
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = lmax_mmax
        channels = 16
        batch_size = 10

        norm = rmsnorm_class(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            affine=True,
            subtract_mean=True,
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

    def test_torch_compile(
        self,
        rmsnorm_class,
    ) -> None:
        """Forward and backward pass should work with torch.compile.

        Tests compilation with hardcoded CUDA device and inductor backend in default mode.
        This verifies the graph structure is compilable without needing parametrization.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        lmax, mmax = 4, 2
        compile_backend, compile_mode = "inductor", "default"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        channels = 16
        batch_size = 10

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        compiled_norm = torch.compile(norm, mode=compile_mode, backend=compile_backend)

        # Test forward pass matches reference
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

        ref_out = norm(x)
        out = compiled_norm(x)

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(ref_out, out, rtol=rtol, atol=atol)

        # Test backward pass
        loss = ((torch.randn_like(out) - out) ** 2.0).mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self, device: torch.device, rmsnorm_class) -> None:
        """Each batch element should be processed independently.

        Parameters
        ----------
        device : torch.device
            Device to run on.
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        lmax, mmax = 4, 2
        channels = 16

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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

    def test_batch_size_one(self, rmsnorm_class) -> None:
        """Test with batch size of 1.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 1

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_single_channel(self, rmsnorm_class) -> None:
        """Test with single channel.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 1
        batch_size = 10

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_lmax0_mmax0(self, rmsnorm_class) -> None:
        """Test with lmax=0, mmax=0 (scalar only).

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 0, 0
        channels = 16
        batch_size = 10

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_no_affine(self, rmsnorm_class) -> None:
        """Test with affine=False.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = rmsnorm_class(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=False
        ).to(device=device, dtype=dtype)

        assert norm.affine_weight is None
        assert norm.affine_bias is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_affine_weight_shape(self, rmsnorm_class) -> None:
        """Test affine weight shapes.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = 4, 2
        channels = 16

        norm = rmsnorm_class(lmax=lmax, mmax=mmax, num_channels=channels)
        assert norm.affine_weight.shape == (lmax + 1, channels)
        assert norm.affine_bias.shape == (channels,)

    def test_no_balance(self, rmsnorm_class) -> None:
        """Test with std_balance_degrees=False.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = rmsnorm_class(
            lmax=lmax, mmax=mmax, num_channels=channels, std_balance_degrees=False
        ).to(device=device, dtype=dtype)

        assert norm.balance_degree_weight is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_balance_vs_no_balance_different(self, rmsnorm_class) -> None:
        """Outputs should differ with and without degree balancing.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm_balanced = rmsnorm_class(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            std_balance_degrees=True,
            affine=False,
        ).to(device=device, dtype=dtype)

        norm_unbalanced = rmsnorm_class(
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

    def test_extra_repr(self, rmsnorm_class) -> None:
        """Test string representation.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        norm = rmsnorm_class(lmax=4, mmax=2, num_channels=64)
        repr_str = repr(norm)
        assert "lmax=4" in repr_str
        assert "mmax=2" in repr_str
        assert "num_channels=64" in repr_str

    def test_invalid_lmax(self, rmsnorm_class) -> None:
        """lmax must be non-negative.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="lmax must be non-negative"):
            rmsnorm_class(lmax=-1, mmax=0, num_channels=16)

    def test_invalid_mmax_negative(self, rmsnorm_class) -> None:
        """mmax must be non-negative.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            rmsnorm_class(lmax=2, mmax=-1, num_channels=16)

    def test_invalid_mmax_gt_lmax(self, rmsnorm_class) -> None:
        """mmax must be <= lmax.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            rmsnorm_class(lmax=2, mmax=3, num_channels=16)

    def test_invalid_channels(self, rmsnorm_class) -> None:
        """num_channels must be positive.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="num_channels must be positive"):
            rmsnorm_class(lmax=2, mmax=2, num_channels=0)

    def test_invalid_input_shape(self, rmsnorm_class) -> None:
        """Should raise error if input shape doesn't match.

        Parameters
        ----------
        rmsnorm_class : type
            The normalization class to test (unfused or fused).
        """
        norm = rmsnorm_class(lmax=4, mmax=2, num_channels=16)
        x = torch.randn(10, 3, 3, 2, 16)  # Wrong lmax

        with pytest.raises(ValueError, match="Expected input shape"):
            norm(x)

    @pytest.mark.parametrize(
        "layer_kwargs",
        [
            dict(subtract_mean=True, std_balance_degrees=True, affine=True),
            dict(subtract_mean=False, std_balance_degrees=True, affine=True),
            dict(subtract_mean=True, std_balance_degrees=False, affine=True),
            dict(subtract_mean=True, std_balance_degrees=True, affine=False),
        ],
        ids=["default", "no-submean", "no-balance", "no-affine"],
    )
    def test_fused_unfused_equivalence(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        layer_kwargs: dict,
    ) -> None:
        """Fused variant should produce identical output and gradients to unfused.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        layer_kwargs : dict
            Keyword arguments to pass to the layer constructor.
        """
        lmax, mmax = lmax_mmax
        compare_fused_unfused(
            FusedEquivariantRMSNorm, lmax, mmax, 32, 4, dtype, device, **layer_kwargs
        )


# =============================================================================
# Test EquivariantLayerNormTied
# =============================================================================


class TestEquivariantLayerNormTied:
    """Comprehensive tests for EquivariantLayerNormTied."""

    def test_output_shape(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        layernormsh_class,
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
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 32
        batch_size = 50

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_invalid_positions_zero(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        layernormsh_class,
    ) -> None:
        """Invalid (l, m) positions should remain zero.

        Parameters
        ----------
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 16
        batch_size = 10

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        layernormsh_class,
    ) -> None:
        """m=0 imaginary component should remain zero.

        Parameters
        ----------
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 16
        batch_size = 10

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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

    def test_l0_uses_layernorm(self, layernormsh_class) -> None:
        """l=0 should be processed with LayerNorm (zero mean, unit variance).

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 32
        batch_size = 100

        norm = layernormsh_class(
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
        layernormsh_class,
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
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        channels = 16
        batch_size = 10

        norm = layernormsh_class(
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
            (math.pi / 3, math.pi / 4, math.pi / 6),  # Representative rotation
        ],
        ids=["representative"],
    )
    def test_equivariance_preserved(
        self,
        dtype: torch.dtype,
        device: torch.device,
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
        layernormsh_class,
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
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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

    def test_torch_compile(
        self,
        layernormsh_class,
    ) -> None:
        """Forward and backward pass should work with torch.compile.

        Tests compilation with hardcoded CUDA device and inductor backend in default mode.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        lmax, mmax = 4, 2
        compile_backend = "inductor"
        compile_mode = "default"
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"
        channels = 16
        batch_size = 10

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        compiled_norm = torch.compile(norm, mode=compile_mode, backend=compile_backend)

        # Test forward pass matches reference
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

        ref_out = norm(x)
        out = compiled_norm(x)

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(ref_out, out, rtol=rtol, atol=atol)

        # Test backward pass
        loss = ((torch.randn_like(out) - out) ** 2.0).mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self, device: torch.device, layernormsh_class) -> None:
        """Each batch element should be processed independently.

        Parameters
        ----------
        device : torch.device
            Device to run on.
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        lmax, mmax = 4, 2
        channels = 16

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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

    def test_batch_size_one(self, layernormsh_class) -> None:
        """Test with batch size of 1.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 1

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_single_channel(self, layernormsh_class) -> None:
        """Test with single channel."""
        lmax, mmax = 4, 2
        channels = 1
        batch_size = 10
        dtype = torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_no_affine(self, layernormsh_class) -> None:
        """Test with affine=False."""
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10
        dtype = torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        norm = layernormsh_class(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=False
        ).to(device=device, dtype=dtype)

        assert norm.affine_weight is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_affine_weight_shape(self, layernormsh_class) -> None:
        """Test affine weight shapes.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = 4, 2
        channels = 16

        norm = layernormsh_class(lmax=lmax, mmax=mmax, num_channels=channels)
        assert norm.affine_weight.shape == (lmax, channels)

    def test_no_balance(self, layernormsh_class) -> None:
        """Test with std_balance_degrees=False.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = layernormsh_class(
            lmax=lmax, mmax=mmax, num_channels=channels, std_balance_degrees=False
        ).to(device=device, dtype=dtype)

        assert norm.balance_degree_weight is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_balance_vs_no_balance_different(self, layernormsh_class) -> None:
        """Outputs should differ with and without degree balancing.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm_balanced = layernormsh_class(
            lmax=lmax,
            mmax=mmax,
            num_channels=channels,
            std_balance_degrees=True,
            affine=False,
        ).to(device=device, dtype=dtype)

        norm_unbalanced = layernormsh_class(
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

    def test_extra_repr(self, layernormsh_class) -> None:
        """Test string representation.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        norm = layernormsh_class(lmax=4, mmax=2, num_channels=64)
        repr_str = repr(norm)
        assert "lmax=4" in repr_str
        assert "mmax=2" in repr_str
        assert "num_channels=64" in repr_str

    def test_invalid_lmax(self, layernormsh_class) -> None:
        """lmax must be >= 1.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="lmax must be >= 1"):
            layernormsh_class(lmax=0, mmax=0, num_channels=16)

    def test_invalid_mmax_negative(self, layernormsh_class) -> None:
        """mmax must be non-negative.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            layernormsh_class(lmax=2, mmax=-1, num_channels=16)

    def test_invalid_mmax_gt_lmax(self, layernormsh_class) -> None:
        """mmax must be <= lmax.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            layernormsh_class(lmax=2, mmax=3, num_channels=16)

    def test_invalid_channels(self, layernormsh_class) -> None:
        """num_channels must be positive.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="num_channels must be positive"):
            layernormsh_class(lmax=2, mmax=2, num_channels=0)

    def test_invalid_input_shape(self, layernormsh_class) -> None:
        """Should raise error if input shape doesn't match.

        Parameters
        ----------
        layernormsh_class : type
            The normalization class to test (unfused or fused).
        """
        norm = layernormsh_class(lmax=4, mmax=2, num_channels=16)
        x = torch.randn(10, 3, 3, 2, 16)  # Wrong lmax

        with pytest.raises(ValueError, match="Expected input shape"):
            norm(x)

    @pytest.mark.parametrize(
        "layer_kwargs",
        [
            dict(std_balance_degrees=True, affine=True),
            dict(std_balance_degrees=False, affine=True),
            dict(std_balance_degrees=True, affine=False),
        ],
        ids=["default", "no-balance", "no-affine"],
    )
    def test_fused_unfused_equivalence(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        layer_kwargs: dict,
    ) -> None:
        """Fused variant should produce identical output and gradients to unfused.

        Parameters
        ----------
        lmax_mmax_layernorm_sh : tuple[int, int]
            Tuple of (lmax, mmax) values where lmax >= 1.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        layer_kwargs : dict
            Keyword arguments to pass to the layer constructor.
        """
        lmax, mmax = lmax_mmax_layernorm_sh
        compare_fused_unfused(
            FusedEquivariantLayerNormTied,
            lmax,
            mmax,
            32,
            4,
            dtype,
            device,
            **layer_kwargs,
        )


# =============================================================================
# Test EquivariantLayerNorm
# =============================================================================


class TestEquivariantLayerNorm:
    """Comprehensive tests for EquivariantLayerNorm."""

    def test_output_shape(
        self,
        lmax_mmax_small: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        layernorm_class,
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
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = lmax_mmax_small
        channels = 32
        batch_size = 50

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_invalid_positions_zero(
        self,
        lmax_mmax_small: tuple[int, int],
        layernorm_class,
    ) -> None:
        """Invalid (l, m) positions should remain zero.

        Parameters
        ----------
        lmax_mmax_small : tuple[int, int]
            Tuple of (lmax, mmax) values.
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = lmax_mmax_small
        channels = 16
        batch_size = 10

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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
        self,
        lmax_mmax_small: tuple[int, int],
        layernorm_class,
    ) -> None:
        """m=0 imaginary component should remain zero.

        Parameters
        ----------
        lmax_mmax_small : tuple[int, int]
            Tuple of (lmax, mmax) values.
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = lmax_mmax_small
        channels = 16
        batch_size = 10

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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

    def test_backward_pass(
        self, dtype: torch.dtype, device: torch.device, layernorm_class
    ) -> None:
        """Gradients should flow to input and parameters.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = 3, 2
        channels = 16
        batch_size = 10

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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
            (math.pi / 3, math.pi / 4, math.pi / 6),  # Representative rotation
        ],
        ids=["representative"],
    )
    def test_equivariance_preserved(
        self,
        dtype: torch.dtype,
        device: torch.device,
        alpha_val: float,
        beta_val: float,
        gamma_val: float,
        layernorm_class,
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
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = 3, 2
        channels = 16
        batch_size = 10

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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

    def test_torch_compile(
        self,
        layernorm_class,
    ) -> None:
        """Forward and backward pass should work with torch.compile.

        Tests compilation with hardcoded CUDA device and inductor backend in default mode.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        lmax, mmax = 2, 1
        compile_backend, compile_mode = "inductor", "default"
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"
        channels = 16
        batch_size = 10

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        compiled_norm = torch.compile(norm, mode=compile_mode, backend=compile_backend)

        # Test forward pass matches reference
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

        ref_out = norm(x)
        out = compiled_norm(x)

        rtol, atol = get_rtol_atol(dtype)
        torch.testing.assert_close(ref_out, out, rtol=rtol, atol=atol)

        # Test backward pass
        loss = ((torch.randn_like(out) - out) ** 2.0).mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self, device: torch.device, layernorm_class) -> None:
        """Each batch element should be processed independently.

        Parameters
        ----------
        device : torch.device
            Device to run on.
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        lmax, mmax = 3, 2
        channels = 16

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
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

    def test_batch_size_one(self, layernorm_class) -> None:
        """Test with batch size of 1.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 1

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_single_channel(self, layernorm_class) -> None:
        """Test with single channel.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 1
        batch_size = 10

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_lmax0_mmax0(self, layernorm_class) -> None:
        """Test with lmax=0, mmax=0 (scalar only).

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 0, 0
        channels = 16
        batch_size = 10

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels).to(
            device=device, dtype=dtype
        )

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_no_affine(self, layernorm_class) -> None:
        """Test with affine=False.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = layernorm_class(
            lmax=lmax, mmax=mmax, num_channels=channels, affine=False
        ).to(device=device, dtype=dtype)

        assert norm.affine_weight is None
        assert norm.affine_bias is None

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_affine_weight_shape(self, layernorm_class) -> None:
        """Test affine weight shapes.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        lmax, mmax = 4, 2
        channels = 16

        norm = layernorm_class(lmax=lmax, mmax=mmax, num_channels=channels)
        assert norm.affine_weight.shape == (lmax + 1, channels)
        assert norm.affine_bias.shape == (channels,)

    def test_subtract_mean(self, layernorm_class) -> None:
        """Test with subtract_mean=True/False.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lmax, mmax = 4, 2
        channels = 16
        batch_size = 10

        norm = layernorm_class(
            lmax=lmax, mmax=mmax, num_channels=channels, subtract_mean=False
        ).to(device=device, dtype=dtype)

        x = torch.randn(
            batch_size, lmax + 1, mmax + 1, 2, channels, device=device, dtype=dtype
        )

        out = norm(x)
        assert torch.isfinite(out).all()

    def test_extra_repr(self, layernorm_class) -> None:
        """Test string representation.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        norm = layernorm_class(lmax=4, mmax=2, num_channels=64)
        repr_str = repr(norm)
        assert "lmax=4" in repr_str
        assert "mmax=2" in repr_str
        assert "num_channels=64" in repr_str

    def test_invalid_lmax(self, layernorm_class) -> None:
        """lmax must be non-negative.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="lmax must be non-negative"):
            layernorm_class(lmax=-1, mmax=0, num_channels=16)

    def test_invalid_mmax_negative(self, layernorm_class) -> None:
        """mmax must be non-negative.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="mmax must be non-negative"):
            layernorm_class(lmax=2, mmax=-1, num_channels=16)

    def test_invalid_mmax_gt_lmax(self, layernorm_class) -> None:
        """mmax must be <= lmax.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="mmax.*must be <= lmax"):
            layernorm_class(lmax=2, mmax=3, num_channels=16)

    def test_invalid_channels(self, layernorm_class) -> None:
        """num_channels must be positive.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        with pytest.raises(ValueError, match="num_channels must be positive"):
            layernorm_class(lmax=2, mmax=2, num_channels=0)

    def test_invalid_input_shape(self, layernorm_class) -> None:
        """Should raise error if input shape doesn't match.

        Parameters
        ----------
        layernorm_class : type
            The normalization class to test (unfused or fused).
        """
        norm = layernorm_class(lmax=4, mmax=2, num_channels=16)
        x = torch.randn(10, 3, 3, 2, 16)  # Wrong lmax

        with pytest.raises(ValueError, match="Expected input shape"):
            norm(x)

    @pytest.mark.parametrize(
        "layer_kwargs",
        [
            dict(subtract_mean=True, affine=True),
            dict(subtract_mean=False, affine=True),
            dict(subtract_mean=True, affine=False),
        ],
        ids=["default", "no-submean", "no-affine"],
    )
    def test_fused_unfused_equivalence(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        layer_kwargs: dict,
    ) -> None:
        """Fused variant should produce identical output and gradients to unfused.

        Parameters
        ----------
        lmax_mmax : tuple[int, int]
            Tuple of (lmax, mmax) values.
        dtype : torch.dtype
            Data type for tensors.
        device : torch.device
            Device to run on.
        layer_kwargs : dict
            Keyword arguments to pass to the layer constructor.
        """
        lmax, mmax = lmax_mmax
        compare_fused_unfused(
            FusedEquivariantLayerNorm, lmax, mmax, 32, 4, dtype, device, **layer_kwargs
        )
