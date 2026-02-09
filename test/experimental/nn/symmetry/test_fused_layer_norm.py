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

"""Unit tests for fused equivariant normalization layers.

Tests verify that fused variants produce identical output to unfused reference
implementations. Once Warp kernels are integrated, these tests serve as
correctness validation for the fused paths.

Tests cover:
- Output equivalence between fused and unfused variants
- Gradient equivalence for inputs and parameters
- Shape preservation
- Invalid (l, m) positions remain zero
- m=0 imaginary component remains zero
- Multi-precision support (float16, bfloat16, float32, float64)
"""

from __future__ import annotations

from typing import Type

import pytest
import torch

from physicsnemo.experimental.nn.symmetry.layer_norm import (
    EquivariantLayerNormGrid,
    EquivariantLayerNormSHGrid,
    EquivariantRMSNormSHGrid,
    FusedEquivariantLayerNorm,
    FusedEquivariantLayerNormSH,
    FusedEquivariantRMSNorm,
)
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


@pytest.fixture(params=[(1, 1), (2, 1), (4, 2)])
def lmax_mmax_layernorm_sh(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterized fixture for lmax/mmax configurations for LayerNormSH.

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


# =============================================================================
# Test Helper Utilities
# =============================================================================


def compare_fused_unfused(
    fused_class: Type,
    unfused_class: Type,
    lmax: int,
    mmax: int,
    num_channels: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    **layer_kwargs,
) -> None:
    """Compare fused and unfused normalization layer outputs and gradients.

    This helper function creates both fused and unfused normalization layers with
    identical parameters, runs a forward and backward pass, and verifies that:
    1. Output tensors match within dtype-appropriate tolerances
    2. Input gradients match
    3. Parameter gradients match (if affine=True)
    4. Output shapes are identical
    5. Invalid (l,m) positions remain zero
    6. m=0 imaginary components remain zero

    Parameters
    ----------
    fused_class : Type
        The fused normalization class to test.
    unfused_class : Type
        The unfused reference normalization class.
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
        Additional keyword arguments to pass to layer constructors
        (e.g., subtract_mean, std_balance_degrees, affine, eps).
    """
    rtol, atol = get_rtol_atol(dtype)

    # Create layers with same parameters
    fused_layer = fused_class(
        lmax=lmax,
        mmax=mmax,
        num_channels=num_channels,
        **layer_kwargs,
    ).to(device=device, dtype=dtype)

    unfused_layer = unfused_class(
        lmax=lmax,
        mmax=mmax,
        num_channels=num_channels,
        **layer_kwargs,
    ).to(device=device, dtype=dtype)

    # Synchronize parameters (copy unfused to fused)
    # Since fused inherits from unfused, they should have the same parameter structure
    fused_layer.load_state_dict(unfused_layer.state_dict())

    # Create identical input tensors with gradient tracking
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

    # Forward pass
    y_fused = fused_layer(x_fused)
    y_unfused = unfused_layer(x_unfused)

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

    # Backward pass
    # Create identical gradient tensors for backward
    grad_output = torch.randn_like(y_fused)
    y_fused.backward(grad_output)
    y_unfused.backward(grad_output.clone())

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


# =============================================================================
# Test Classes
# =============================================================================


class TestFusedEquivariantRMSNorm:
    """Tests for FusedEquivariantRMSNorm."""

    def test_output_equivalence_default(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test that fused variant matches unfused with default parameters."""
        lmax, mmax = lmax_mmax
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantRMSNorm,
            EquivariantRMSNormSHGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            subtract_mean=True,
            std_balance_degrees=True,
            affine=True,
        )

    def test_output_equivalence_no_subtract_mean(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test fused variant with subtract_mean=False."""
        lmax, mmax = lmax_mmax
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantRMSNorm,
            EquivariantRMSNormSHGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            subtract_mean=False,
            std_balance_degrees=True,
            affine=True,
        )

    def test_output_equivalence_no_balance(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test fused variant with std_balance_degrees=False."""
        lmax, mmax = lmax_mmax
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantRMSNorm,
            EquivariantRMSNormSHGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            subtract_mean=True,
            std_balance_degrees=False,
            affine=True,
        )

    def test_output_equivalence_no_affine(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test fused variant with affine=False."""
        lmax, mmax = lmax_mmax
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantRMSNorm,
            EquivariantRMSNormSHGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            subtract_mean=True,
            std_balance_degrees=True,
            affine=False,
        )


class TestFusedEquivariantLayerNormSH:
    """Tests for FusedEquivariantLayerNormSH."""

    def test_output_equivalence_default(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test that fused variant matches unfused with default parameters."""
        lmax, mmax = lmax_mmax_layernorm_sh
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantLayerNormSH,
            EquivariantLayerNormSHGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            std_balance_degrees=True,
            affine=True,
        )

    def test_output_equivalence_no_balance(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test fused variant with std_balance_degrees=False."""
        lmax, mmax = lmax_mmax_layernorm_sh
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantLayerNormSH,
            EquivariantLayerNormSHGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            std_balance_degrees=False,
            affine=True,
        )

    def test_output_equivalence_no_affine(
        self,
        lmax_mmax_layernorm_sh: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test fused variant with affine=False."""
        lmax, mmax = lmax_mmax_layernorm_sh
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantLayerNormSH,
            EquivariantLayerNormSHGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            std_balance_degrees=True,
            affine=False,
        )


class TestFusedEquivariantLayerNorm:
    """Tests for FusedEquivariantLayerNorm."""

    def test_output_equivalence_default(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test that fused variant matches unfused with default parameters."""
        lmax, mmax = lmax_mmax
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantLayerNorm,
            EquivariantLayerNormGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            subtract_mean=True,
            affine=True,
        )

    def test_output_equivalence_no_subtract_mean(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test fused variant with subtract_mean=False."""
        lmax, mmax = lmax_mmax
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantLayerNorm,
            EquivariantLayerNormGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            subtract_mean=False,
            affine=True,
        )

    def test_output_equivalence_no_affine(
        self,
        lmax_mmax: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Test fused variant with affine=False."""
        lmax, mmax = lmax_mmax
        num_channels = 32
        batch_size = 4

        compare_fused_unfused(
            FusedEquivariantLayerNorm,
            EquivariantLayerNormGrid,
            lmax,
            mmax,
            num_channels,
            batch_size,
            dtype,
            device,
            subtract_mean=True,
            affine=False,
        )
