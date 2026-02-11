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

r"""Equivariant normalization layers for spherical harmonic features in grid layout.

This module provides normalization layers that preserve SO(3) equivariance when applied
to spherical harmonic representations. These layers are essential for stabilizing
training in equivariant neural networks while maintaining the rotational symmetry
required for physics applications.

The implementations operate on the grid layout tensor format:
``(batch, lmax+1, mmax+1, 2, channels)`` where the dimensions represent:

- ``batch`` - Batch/sample dimension
- ``lmax+1`` - Spherical harmonic degree (0 to lmax)
- ``mmax+1`` - Spherical harmonic order (0 to mmax, non-negative only)
- ``2`` - Real and imaginary components
- ``channels`` - Feature channels

Key Equivariance Constraints
----------------------------
1. **No mean subtraction for l>0**: Subtracting mean from non-scalar components would
   break rotational equivariance, as vectors/tensors must remain centered at origin.

2. **Scalar (l=0) special handling**: The l=0 component is invariant under rotation,
   allowing standard normalization operations (mean subtraction, scaling).

3. **Degree-wise affine weights**: Learnable parameters shaped ``(lmax+1, channels)``
   allow the model to learn relative importance of different angular frequencies.

4. **Degree balancing**: Optional weighting prevents higher degrees (which have more
   m components) from dominating the norm calculation.

Classes
-------
EquivariantRMSNorm
    RMS normalization with global scaling and optional mean subtraction for l=0.
EquivariantLayerNormTied
    LayerNorm for l=0, shared global scaling for l>0 with degree balancing.
EquivariantLayerNorm
    Per-degree normalization with independent scaling for each l.

Functions
---------
make_degree_balance_weight
    Create weights to balance contribution from each spherical harmonic degree.
make_m0_imag_mask
    Create mask that zeros out m=0 imaginary component.

See Also
--------
physicsnemo.experimental.nn.symmetry.grid.make_grid_mask :
    Creates validity mask for (l, m) positions.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import nn, Tensor

from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask
from physicsnemo.nn import Module

# Warp import for fused kernels
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.experimental.nn.symmetry.fused_norm_kernels import (
    fused_layernorm,
    fused_layernormsh_lgt0,
    fused_rmsnorm,
)

__all__ = [
    "EquivariantLayerNorm",
    "EquivariantLayerNormTied",
    "EquivariantRMSNorm",
    "FusedEquivariantLayerNorm",
    "FusedEquivariantLayerNormTied",
    "FusedEquivariantRMSNorm",
    "make_degree_balance_weight",
    "make_m0_imag_mask",
]


# =============================================================================
# Helper Utilities
# =============================================================================


def make_degree_balance_weight(
    lmax: int, mmax: int
) -> Float[Tensor, "lmax_p1 mmax_p1"]:
    r"""Create weights to balance contribution from each spherical harmonic degree.

    When computing norms across all spherical harmonic components, higher degrees
    have more m components and would otherwise dominate the norm calculation.
    This function creates weights that normalize the contribution from each degree,
    accounting for the actual number of valid m positions given the mmax constraint.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be non-negative.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.

    Returns
    -------
    Float[Tensor, "lmax_p1 mmax_p1"]
        Weight tensor of shape ``(lmax+1, mmax+1)`` where the weight for
        valid position ``(l, m)`` is ``1 / (num_valid_m_for_l * (lmax + 1))``
        and invalid positions (m > l) have weight 0.

    Raises
    ------
    ValueError
        If lmax < 0, mmax < 0, or mmax > lmax.

    Notes
    -----
    The weights are designed such that:

    1. Each degree l contributes equally when summed over all its valid m components
    2. The total weight sums to 1.0 when summed over all valid (l, m) positions

    For a given degree l, the number of valid m positions is ``min(l, mmax) + 1``.
    The formula for weight at position (l, m) where m <= l and m <= mmax is:

    .. math::

        w_{l,m} = \frac{1}{(\min(l, m_{max}) + 1) \cdot (L_{max} + 1)}

    Examples
    --------
    >>> weights = make_degree_balance_weight(lmax=2, mmax=2)
    >>> weights.shape
    torch.Size([3, 3])
    >>> # l=0 has 1 valid m, weight = 1/(1*3) = 0.333 at m=0
    >>> # l=1 has 2 valid m, weight = 1/(2*3) = 0.167 at m=0,1
    >>> # l=2 has 3 valid m, weight = 1/(3*3) = 0.111 at m=0,1,2
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if mmax < 0:
        raise ValueError(f"mmax must be non-negative, got {mmax}")
    if mmax > lmax:
        raise ValueError(f"mmax ({mmax}) must be <= lmax ({lmax})")

    # Create validity mask
    validity_mask = make_grid_mask(lmax, mmax).float()  # [lmax+1, mmax+1]

    # For each degree l, count valid m positions: min(l, mmax) + 1
    l_values = torch.arange(lmax + 1).float()  # [lmax+1]
    valid_m_counts = (
        torch.minimum(l_values, torch.tensor(mmax, dtype=torch.float32)) + 1
    )

    # Weight per (l, m) position: 1 / (valid_m_count * num_degrees)
    weights = 1.0 / (valid_m_counts[:, None] * (lmax + 1))  # [lmax+1, 1]
    weights = weights.expand(-1, mmax + 1).clone()  # [lmax+1, mmax+1]

    # Apply validity mask
    weights = weights * validity_mask

    return weights


def make_m0_imag_mask(mmax: int) -> Float[Tensor, "1 1 mmax_p1 2 1"]:
    r"""Create mask that zeros out m=0 imaginary component.

    For spherical harmonics, the m=0 component is purely real. This mask
    ensures the imaginary part of m=0 remains zero after operations.

    Parameters
    ----------
    mmax : int
        Maximum spherical harmonic order. Must be non-negative.

    Returns
    -------
    Float[Tensor, "1 1 mmax_p1 2 1"]
        Mask tensor of shape ``(1, 1, mmax+1, 2, 1)`` for broadcasting over
        grid layout tensors. The mask is 1.0 everywhere except at
        ``[:, :, 0, 1, :]`` (m=0, imaginary) which is 0.0.

    Raises
    ------
    ValueError
        If mmax < 0.

    Notes
    -----
    The m=0 spherical harmonic :math:`Y_l^0` is real-valued by definition.
    In the grid layout, the imaginary component at m=0 must always be zero.
    This mask can be multiplied with feature tensors to enforce this constraint.

    Examples
    --------
    >>> mask = make_m0_imag_mask(mmax=2)
    >>> mask.shape
    torch.Size([1, 1, 3, 2, 1])
    >>> mask[0, 0, 0, 0, 0]  # m=0, real: 1.0
    tensor(1.)
    >>> mask[0, 0, 0, 1, 0]  # m=0, imaginary: 0.0
    tensor(0.)
    >>> mask[0, 0, 1, 1, 0]  # m=1, imaginary: 1.0
    tensor(1.)
    """
    if mmax < 0:
        raise ValueError(f"mmax must be non-negative, got {mmax}")

    mask = torch.ones(1, 1, mmax + 1, 2, 1)
    mask[:, :, 0, 1, :] = 0.0

    return mask


# =============================================================================
# Equivariant Normalization Classes
# =============================================================================


class _EquivariantNormBase(Module):
    r"""Base class for equivariant normalization layers.

    This private base class provides common functionality for all equivariant
    normalization implementations, including parameter validation, buffer
    registration, and input/output processing.

    Note: This is a private class (not exported in ``__all__``).
    """

    @staticmethod
    def _validate_params(
        lmax: int, mmax: int, num_channels: int, min_lmax: int = 0
    ) -> None:
        r"""Validate common parameters for equivariant normalization.

        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        mmax : int
            Maximum spherical harmonic order.
        num_channels : int
            Number of feature channels.
        min_lmax : int, optional
            Minimum allowed value for lmax. Default is 0.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        if lmax < min_lmax:
            if min_lmax == 0:
                raise ValueError(f"lmax must be non-negative, got {lmax}")
            else:
                raise ValueError(
                    f"lmax must be >= {min_lmax} for EquivariantLayerNormTied "
                    f"(need l>0 components), got {lmax}"
                )
        if mmax < 0:
            raise ValueError(f"mmax must be non-negative, got {mmax}")
        if mmax > lmax:
            raise ValueError(f"mmax ({mmax}) must be <= lmax ({lmax})")
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")

    def __init__(
        self, lmax: int, mmax: int, num_channels: int, eps: float = 1e-5
    ) -> None:
        r"""Initialize base normalization layer.

        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        mmax : int
            Maximum spherical harmonic order.
        num_channels : int
            Number of feature channels.
        eps : float, optional
            Small constant for numerical stability. Default is 1e-5.
        """
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels
        self.eps = eps

        # Register validity mask buffer
        validity_mask = make_grid_mask(lmax, mmax).float()
        self.register_buffer("validity_mask", validity_mask, persistent=True)

        # Register m0 imaginary mask buffer
        m0_imag_mask = make_m0_imag_mask(mmax)
        self.register_buffer("m0_imag_mask", m0_imag_mask, persistent=True)

        # Precompute combined mask for efficient input/output processing
        # validity_mask: [lmax+1, mmax+1] -> [1, lmax+1, mmax+1, 1, 1]
        # m0_imag_mask: [1, 1, mmax+1, 2, 1]
        # combined_mask: [1, lmax+1, mmax+1, 2, 1]
        combined_mask = validity_mask[None, :, :, None, None] * m0_imag_mask
        self.register_buffer("combined_mask", combined_mask, persistent=False)

        # 3D grid mask for Warp kernels: [lmax+1, mmax+1, 2]
        # Squeezed from combined_mask [1, lmax+1, mmax+1, 2, 1]
        grid_mask_3d = combined_mask[0, :, :, :, 0].contiguous()
        self.register_buffer("grid_mask_3d", grid_mask_3d, persistent=False)

    def _register_subtract_mean_buffers(self, subtract_mean: bool) -> None:
        r"""Register buffers for mean subtraction control (eliminates runtime branching).

        Parameters
        ----------
        subtract_mean : bool
            Whether mean subtraction is enabled.
        """
        # Subtract mean scale: 1.0 if subtract_mean enabled, 0.0 otherwise
        subtract_mean_scale = torch.tensor(1.0 if subtract_mean else 0.0)
        self.register_buffer(
            "subtract_mean_scale", subtract_mean_scale, persistent=False
        )

        # L0 subtract mean mask: 1.0 at (l=0, m=0, real), 0.0 elsewhere
        # Shape: [1, lmax+1, mmax+1, 2, 1]
        l0_subtract_mean_mask = torch.zeros(1, self.lmax + 1, self.mmax + 1, 2, 1)
        l0_subtract_mean_mask[:, 0, 0, 0, :] = 1.0
        self.register_buffer(
            "l0_subtract_mean_mask", l0_subtract_mean_mask, persistent=False
        )

        # Precompute scaled l0 mask: l0_subtract_mean_mask * subtract_mean_scale
        # This eliminates one multiplication per forward pass
        scaled_l0_mask = l0_subtract_mean_mask * subtract_mean_scale
        self.register_buffer("scaled_l0_mask", scaled_l0_mask, persistent=False)

    def _get_compute_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        r"""Determine computation dtype for numerical stability.

        Parameters
        ----------
        input_dtype : torch.dtype
            Input tensor dtype.

        Returns
        -------
        torch.dtype
            Dtype to use for computation (float32 for float16/bfloat16,
            otherwise same as input).
        """
        if input_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return input_dtype

    def _prepare_input(self, x: Tensor, compute_dtype: torch.dtype) -> Tensor:
        r"""Prepare input tensor for normalization computation.

        Parameters
        ----------
        x : Tensor
            Input tensor in grid layout.
        compute_dtype : torch.dtype
            Target dtype for computation.

        Returns
        -------
        Tensor
            Prepared tensor with validity and m0 masks applied.
        """
        # Cast to compute dtype and apply combined mask in single operation
        return x.to(compute_dtype) * self.combined_mask.to(compute_dtype)

    def _finalize_output(
        self, x: Tensor, compute_dtype: torch.dtype, input_dtype: torch.dtype
    ) -> Tensor:
        r"""Finalize output tensor after normalization.

        Parameters
        ----------
        x : Tensor
            Output tensor in compute dtype.
        compute_dtype : torch.dtype
            Current dtype of tensor.
        input_dtype : torch.dtype
            Target output dtype.

        Returns
        -------
        Tensor
            Finalized tensor with validity and m0 masks applied, cast to input dtype.
        """
        # Apply combined mask and cast to input dtype
        return (x * self.combined_mask.to(compute_dtype)).to(input_dtype)

    def _validate_input_shape(self, x: Tensor) -> None:
        r"""Validate input tensor shape.

        Parameters
        ----------
        x : Tensor
            Input tensor to validate.

        Raises
        ------
        ValueError
            If input shape does not match expected grid layout.
        """
        expected_shape = (self.lmax + 1, self.mmax + 1, 2, self.num_channels)
        if x.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected input shape (batch, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}, {expected_shape[3]}), "
                f"got {x.shape}"
            )


class EquivariantRMSNorm(_EquivariantNormBase):
    r"""RMS normalization for spherical harmonic features in grid layout.

    This layer applies Root Mean Square (RMS) normalization with optional
    mean subtraction for the l=0 component and degree balancing for fair contribution
    from all spherical harmonic degrees. It is the most efficient of the three
    normalization variants.

    The normalization computes a single global scaling factor from the RMS of
    all valid coefficients, then applies degree-wise affine transformation.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be non-negative.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    num_channels : int
        Number of feature channels. Must be positive.
    subtract_mean : bool, optional
        Whether to subtract mean from l=0 component. Default is True.
    std_balance_degrees : bool, optional
        Whether to balance degree contributions to the norm. Default is True.
    affine : bool, optional
        Whether to apply learnable affine parameters. Default is True.
    eps : float, optional
        Small constant for numerical stability in division. Default is 1e-5.

    Attributes
    ----------
    affine_weight : torch.nn.Parameter or None
        Learnable scale parameter of shape ``(lmax+1, num_channels)``.
        Initialized to 1.0. None if ``affine=False``.
    affine_bias : torch.nn.Parameter or None
        Learnable bias parameter of shape ``(num_channels,)`` applied
        to l=0 only. Initialized to 0.0. None if ``affine=False`` or
        ``subtract_mean=False``.
    validity_mask : torch.Tensor
        Boolean mask of shape ``(lmax+1, mmax+1)`` indicating valid (l, m)
        positions where m <= l.
    m0_imag_mask : torch.Tensor
        Mask of shape ``(1, 1, mmax+1, 2, 1)`` that zeros m=0 imaginary.
    balance_degree_weight : torch.Tensor or None
        Weights of shape ``(lmax+1, mmax+1)`` for balanced norm computation.
        None if ``std_balance_degrees=False``.

    Forward
    -------
    x : Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
        Input features in grid layout.

    Outputs
    -------
    Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
        Normalized features with same shape as input.

    Notes
    -----
    **Normalization Algorithm:**

    1. If ``subtract_mean=True``: Subtract channel-wise mean from l=0 component
    2. Compute squared values: :math:`x^2`
    3. If ``std_balance_degrees=True``: Weight by degree balance weights
    4. Compute global RMS: :math:`\text{rms} = \sqrt{\text{mean}(x^2) + \epsilon}`
    5. Scale features: :math:`x_{\text{norm}} = x / \text{rms}`
    6. If ``affine=True``: Apply degree-wise scaling and l=0 bias

    **Equivariance Preservation:**

    - Mean subtraction only on l=0 (rotation-invariant component)
    - Global scaling is a scalar operation, commutes with rotation
    - Degree-wise affine preserves equivariance (same scale for all m in degree l)

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry.layer_norm import EquivariantRMSNorm
    >>> norm = EquivariantRMSNorm(lmax=4, mmax=2, num_channels=64)
    >>> x = torch.randn(100, 5, 3, 2, 64)  # [batch, lmax+1, mmax+1, 2, channels]
    >>> y = norm(x)
    >>> y.shape
    torch.Size([100, 5, 3, 2, 64])

    See Also
    --------
    EquivariantLayerNormTied : LayerNorm variant with separate l=0 handling.
    EquivariantLayerNorm : Per-degree normalization variant.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_channels: int,
        subtract_mean: bool = True,
        std_balance_degrees: bool = True,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        # Validate parameters
        self._validate_params(lmax, mmax, num_channels, min_lmax=0)

        # Initialize base class
        super().__init__(lmax, mmax, num_channels, eps)

        # Store class-specific attributes
        self.subtract_mean = subtract_mean
        self.std_balance_degrees = std_balance_degrees
        self.affine = affine

        # Register mean subtraction control buffers
        self._register_subtract_mean_buffers(subtract_mean)

        # Affine weight: Parameter if affine=True, internal buffer if False
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
        else:
            # Internal buffer for branch-free computation (not exposed as affine_weight)
            self.register_buffer(
                "_affine_weight_buffer",
                torch.ones(lmax + 1, num_channels),
                persistent=False,
            )
            # Public attribute remains None for API compatibility
            self.register_parameter("affine_weight", None)

        # Bias handling: similar approach
        if subtract_mean:
            if affine:
                self.affine_bias = nn.Parameter(torch.zeros(num_channels))
                bias_scale = torch.tensor(1.0)
            else:
                # Internal buffer when subtract_mean but no affine
                self.register_buffer(
                    "_affine_bias_buffer", torch.zeros(num_channels), persistent=False
                )
                self.register_parameter("affine_bias", None)
                bias_scale = torch.tensor(1.0)
        else:
            # When subtract_mean=False, bias is not used
            self.register_buffer(
                "_affine_bias_buffer", torch.zeros(num_channels), persistent=False
            )
            self.register_parameter("affine_bias", None)
            bias_scale = torch.tensor(0.0)
        self.register_buffer("bias_scale", bias_scale, persistent=False)

        # Precompute scaled bias mask: l0_subtract_mean_mask * bias_scale
        # This eliminates one multiplication per forward pass
        scaled_bias_mask = self.l0_subtract_mean_mask * bias_scale
        self.register_buffer("scaled_bias_mask", scaled_bias_mask, persistent=False)

        # Degree balance weights: always computed, but form depends on std_balance_degrees
        if std_balance_degrees:
            balance_weight = make_degree_balance_weight(lmax, mmax)
            self.register_buffer(
                "balance_degree_weight", balance_weight, persistent=True
            )
        else:
            # Uniform weights for branch-free computation (internal buffer)
            validity_mask = make_grid_mask(lmax, mmax).float()
            total_valid = validity_mask.sum() * 2  # *2 for real and imag
            uniform_weight = validity_mask / total_valid
            # Store as internal buffer, public attribute is None
            self.register_buffer(
                "_balance_degree_weight_buffer", uniform_weight, persistent=False
            )
            self.register_buffer("balance_degree_weight", None, persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, x: Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
    ) -> Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]:
        # Validate input shape (skip during torch.compile)
        if not torch.compiler.is_compiling():
            self._validate_input_shape(x)

        # Get compute dtype and prepare input
        input_dtype = x.dtype
        compute_dtype = self._get_compute_dtype(input_dtype)
        x = self._prepare_input(x, compute_dtype)

        # Mean subtraction: subtract mean from l=0 (controlled by subtract_mean_scale and mask)
        # Always compute l0 mean, but only subtract when subtract_mean_scale=1.0
        l0_mean = x[:, 0:1, 0:1, 0:1, :].mean(
            dim=-1, keepdim=True
        )  # [batch, 1, 1, 1, 1]
        # Subtract mean only at l=0, m=0, real - controlled by precomputed scaled_l0_mask
        x = x - l0_mean * self.scaled_l0_mask.to(compute_dtype)

        # Compute norm with degree balancing (always use balanced path)
        x_squared = x.pow(2)  # [batch, lmax+1, mmax+1, 2, channels]

        # Get the appropriate balance weight (public or internal buffer)
        balance_weight = (
            self.balance_degree_weight
            if self.std_balance_degrees
            else self._balance_degree_weight_buffer
        )

        # Weight by degree balance weights (uniform when std_balance_degrees=False)
        # balance_weight: [lmax+1, mmax+1]
        # x_squared: [batch, lmax+1, mmax+1, 2, channels]
        # Using einsum: "blmrc, lm -> brc" (sum over l, m dimensions)
        feature_norm = torch.einsum(
            "blmrc, lm -> brc",
            x_squared,
            balance_weight.to(compute_dtype),
        )  # [batch, 2, channels]
        # Average over real/imag dimension
        feature_norm = feature_norm.mean(dim=1, keepdim=True)  # [batch, 1, channels]

        # Compute RMS and scale
        # Average over channels to get single scale factor
        feature_norm = feature_norm.mean(dim=-1, keepdim=True)  # [batch, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)  # [batch, 1, 1]

        # Apply scaling
        # feature_norm shape: [batch, 1, 1] -> broadcast to [batch, lmax+1, mmax+1, 2, channels]
        x = x * feature_norm[:, :, :, None, None]

        # Apply affine transformation (always computed using internal buffers)
        # Get the appropriate affine weight (Parameter or internal buffer)
        affine_weight = (
            self.affine_weight if self.affine else self._affine_weight_buffer
        )
        # affine_weight: [lmax+1, channels] -> [1, lmax+1, 1, 1, channels]
        weight = affine_weight[None, :, None, None, :].to(compute_dtype)
        x = x * weight

        # Apply bias to l=0 (controlled by bias_scale and l0_subtract_mean_mask)
        # Get the appropriate bias (Parameter if affine=True AND subtract_mean=True, else buffer)
        affine_bias = (
            self.affine_bias
            if self.affine and self.affine_bias is not None
            else self._affine_bias_buffer
        )
        # affine_bias: [channels] -> [1, 1, 1, 1, channels]
        bias = affine_bias[None, None, None, None, :].to(compute_dtype)
        # Only add to l=0, m=0, real component - controlled by precomputed scaled_bias_mask
        x = x + bias * self.scaled_bias_mask.to(compute_dtype)

        # Finalize output
        return self._finalize_output(x, compute_dtype, input_dtype)

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"lmax={self.lmax}, mmax={self.mmax}, num_channels={self.num_channels}, "
            f"subtract_mean={self.subtract_mean}, std_balance_degrees={self.std_balance_degrees}, "
            f"affine={self.affine}, eps={self.eps}"
        )


class EquivariantLayerNormTied(_EquivariantNormBase):
    r"""Layer normalization for spherical harmonic features in grid layout.

    This layer applies standard LayerNorm to the l=0 (scalar) component and
    a shared global scaling to all l>0 components with optional degree balancing.
    This design provides full LayerNorm semantics for scalars while preserving
    equivariance for higher-order terms.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be >= 1.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    num_channels : int
        Number of feature channels. Must be positive.
    std_balance_degrees : bool, optional
        Whether to balance degree contributions to the norm for l>0.
        Default is True.
    affine : bool, optional
        Whether to apply learnable affine parameters. Default is True.
    eps : float, optional
        Small constant for numerical stability. Default is 1e-5.

    Attributes
    ----------
    norm_l0 : torch.nn.LayerNorm
        Standard LayerNorm applied to l=0 component.
    affine_weight : torch.nn.Parameter or None
        Learnable scale for l>0, shape ``(lmax, num_channels)``.
        Note: only lmax entries (not lmax+1) since l=0 uses LayerNorm.
    validity_mask : torch.Tensor
        Boolean mask for valid (l, m) positions.
    m0_imag_mask : torch.Tensor
        Mask that zeros m=0 imaginary component.
    balance_degree_weight : torch.Tensor or None
        Weights for l>0 degree balancing. None if ``std_balance_degrees=False``.

    Forward
    -------
    x : Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
        Input features in grid layout.

    Outputs
    -------
    Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
        Normalized features with same shape as input.

    Notes
    -----
    **Normalization Algorithm:**

    For l=0:
        - Apply standard ``nn.LayerNorm`` (mean subtraction + scaling + affine)

    For l>0:
        1. Compute squared values for all l>0 components
        2. If ``std_balance_degrees=True``: Weight by degree balance weights
        3. Compute shared RMS across all l>0 components
        4. Scale all l>0 by inverse RMS
        5. If ``affine=True``: Apply degree-wise learned scaling

    **Key Difference from RMSNorm:**

    - l=0 gets full LayerNorm treatment (mean subtraction included)
    - l>0 components share a single normalization factor (not mixed with l=0)
    - Separate affine parameters for l=0 (in LayerNorm) and l>0 (learned weight)

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry.layer_norm import EquivariantLayerNormTied
    >>> norm = EquivariantLayerNormTied(lmax=4, mmax=2, num_channels=64)
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> y = norm(x)
    >>> y.shape
    torch.Size([100, 5, 3, 2, 64])

    See Also
    --------
    EquivariantRMSNorm : Simpler RMS variant with global scaling.
    EquivariantLayerNorm : Per-degree normalization variant.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_channels: int,
        std_balance_degrees: bool = True,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        # Validate parameters (requires lmax >= 1)
        self._validate_params(lmax, mmax, num_channels, min_lmax=1)

        # Initialize base class
        super().__init__(lmax, mmax, num_channels, eps)

        # Store class-specific attributes
        self.std_balance_degrees = std_balance_degrees
        self.affine = affine

        # LayerNorm for l=0
        self.norm_l0 = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

        # Affine weight for l>0: Parameter if affine=True, internal buffer if False
        # Note: only lmax entries (not lmax+1) since l=0 uses LayerNorm
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax, num_channels))
        else:
            self.register_buffer(
                "_affine_weight_buffer",
                torch.ones(lmax, num_channels),
                persistent=False,
            )
            self.register_parameter("affine_weight", None)

        # Degree balance weights for l>0 only
        if std_balance_degrees:
            # Create weights for l=1 to lmax
            balance_weight = make_degree_balance_weight(lmax, mmax)
            # Zero out l=0 row and renormalize
            balance_weight_lgt0 = balance_weight.clone()
            balance_weight_lgt0[0, :] = 0.0
            # Renormalize so weights sum to 1
            total = balance_weight_lgt0.sum()
            if total > 0:
                balance_weight_lgt0 = balance_weight_lgt0 / total
            self.register_buffer(
                "balance_degree_weight", balance_weight_lgt0, persistent=True
            )
        else:
            # Uniform weights for l>0 (branch-free computation)
            validity_mask = make_grid_mask(lmax, mmax).float()
            # Zero out l=0 row
            validity_lgt0 = validity_mask.clone()
            validity_lgt0[0, :] = 0.0
            # Total valid positions for l>0
            total_valid = validity_lgt0.sum() * 2  # *2 for real/imag
            uniform_weight = validity_lgt0 / total_valid
            self.register_buffer(
                "_balance_degree_weight_buffer", uniform_weight, persistent=False
            )
            self.register_buffer("balance_degree_weight", None, persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, x: Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
    ) -> Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]:
        # Validate input shape (skip during torch.compile)
        if not torch.compiler.is_compiling():
            self._validate_input_shape(x)

        # Get compute dtype and prepare input
        input_dtype = x.dtype
        compute_dtype = self._get_compute_dtype(input_dtype)
        x = self._prepare_input(x, compute_dtype)

        batch_size = x.shape[0]

        # Process l=0 with standard LayerNorm
        # l=0, m=0, real component: [batch, channels]
        l0_feature = x[:, 0, 0, 0, :]  # [batch, channels]
        l0_normed = self.norm_l0(l0_feature.to(input_dtype)).to(
            compute_dtype
        )  # [batch, channels]

        # Process l>0 with shared scaling (lmax >= 1 guaranteed by __init__ validation)
        # Extract l>0 components
        x_lgt0 = x[:, 1:, :, :, :]  # [batch, lmax, mmax+1, 2, channels]

        # Validity mask for l>0
        validity_lgt0 = self.validity_mask[1:, :].to(compute_dtype)  # [lmax, mmax+1]

        # Compute norm
        x_squared = x_lgt0.pow(2)

        # Get the appropriate balance weight for l>0 (public or internal buffer)
        balance_weight = (
            self.balance_degree_weight
            if self.std_balance_degrees
            else self._balance_degree_weight_buffer
        )
        # Extract l>0 part (already normalized to sum to 1 in __init__)
        balance_lgt0 = balance_weight[1:, :].to(compute_dtype)  # [lmax, mmax+1]

        # Weighted sum: [batch, lmax, mmax+1, 2, channels] -> [batch, 2, channels]
        feature_norm = torch.einsum(
            "blmrc, lm -> brc",
            x_squared,
            balance_lgt0,
        )
        # Average over real/imag
        feature_norm = feature_norm.mean(dim=1, keepdim=True)  # [batch, 1, channels]

        # Average over channels for single scale factor
        feature_norm = feature_norm.mean(dim=-1, keepdim=True)  # [batch, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        # Scale l>0
        x_lgt0_scaled = x_lgt0 * feature_norm[:, :, :, None, None]

        # Apply affine weight for l>0 (always computed using internal buffers)
        affine_weight = (
            self.affine_weight if self.affine else self._affine_weight_buffer
        )
        # affine_weight: [lmax, channels] -> [1, lmax, 1, 1, channels]
        weight = affine_weight[None, :, None, None, :]
        x_lgt0_scaled = x_lgt0_scaled * weight

        # Assemble output
        output = torch.zeros_like(x)
        output[:, 0, 0, 0, :] = l0_normed
        output[:, 1:, :, :, :] = x_lgt0_scaled

        # Finalize output
        return self._finalize_output(output, compute_dtype, input_dtype)

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"lmax={self.lmax}, mmax={self.mmax}, num_channels={self.num_channels}, "
            f"std_balance_degrees={self.std_balance_degrees}, "
            f"affine={self.affine}, eps={self.eps}"
        )


class EquivariantLayerNorm(_EquivariantNormBase):
    r"""Per-degree layer normalization for spherical harmonic features.

    This layer normalizes each spherical harmonic degree independently,
    with special handling for l=0 (mean subtraction allowed) vs l>0
    (scaling only, no mean subtraction).

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be non-negative.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    num_channels : int
        Number of feature channels. Must be positive.
    subtract_mean : bool, optional
        Whether to subtract mean from l=0 component. Default is True.
    affine : bool, optional
        Whether to apply learnable affine parameters. Default is True.
    eps : float, optional
        Small constant for numerical stability. Default is 1e-5.

    Attributes
    ----------
    affine_weight : torch.nn.Parameter or None
        Learnable scale of shape ``(lmax+1, num_channels)``.
    affine_bias : torch.nn.Parameter or None
        Learnable bias of shape ``(num_channels,)`` for l=0 only.
    validity_mask : torch.Tensor
        Boolean mask for valid (l, m) positions.
    m0_imag_mask : torch.Tensor
        Mask that zeros m=0 imaginary component.

    Forward
    -------
    x : Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
        Input features in grid layout.

    Outputs
    -------
    Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
        Normalized features with same shape as input.

    Notes
    -----
    **Normalization Algorithm:**

    For each degree l:
        1. Extract all m components for degree l
        2. If l=0 and ``subtract_mean=True``: Subtract channel-wise mean
        3. Compute RMS over all m components for this degree
        4. Scale by inverse RMS
        5. If ``affine=True``: Apply learned scale (and bias for l=0)

    **Key Difference from Other Variants:**

    - Each degree is normalized independently (not globally)
    - No degree balancing needed (each degree treated separately)
    - More parameters but potentially more expressive

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry.layer_norm import EquivariantLayerNorm
    >>> norm = EquivariantLayerNorm(lmax=4, mmax=2, num_channels=64)
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> y = norm(x)
    >>> y.shape
    torch.Size([100, 5, 3, 2, 64])

    See Also
    --------
    EquivariantRMSNorm : Global RMS normalization variant.
    EquivariantLayerNormTied : Hybrid LayerNorm/global scaling variant.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_channels: int,
        subtract_mean: bool = True,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        # Validate parameters
        self._validate_params(lmax, mmax, num_channels, min_lmax=0)

        # Initialize base class
        super().__init__(lmax, mmax, num_channels, eps)

        # Store class-specific attributes
        self.subtract_mean = subtract_mean
        self.affine = affine

        # Register mean subtraction control buffers
        self._register_subtract_mean_buffers(subtract_mean)

        # Affine weight: Parameter if affine=True, internal buffer if False
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
        else:
            self.register_buffer(
                "_affine_weight_buffer",
                torch.ones(lmax + 1, num_channels),
                persistent=False,
            )
            self.register_parameter("affine_weight", None)

        # Bias handling: similar approach to RMSNorm
        if subtract_mean:
            if affine:
                self.affine_bias = nn.Parameter(torch.zeros(num_channels))
                bias_scale = torch.tensor(1.0)
            else:
                self.register_buffer(
                    "_affine_bias_buffer", torch.zeros(num_channels), persistent=False
                )
                self.register_parameter("affine_bias", None)
                bias_scale = torch.tensor(1.0)
        else:
            self.register_buffer(
                "_affine_bias_buffer", torch.zeros(num_channels), persistent=False
            )
            self.register_parameter("affine_bias", None)
            bias_scale = torch.tensor(0.0)
        self.register_buffer("bias_scale", bias_scale, persistent=False)

        # Precompute valid m counts for each l
        # valid_m_counts[l] = min(l, mmax) + 1
        valid_m_counts = torch.zeros(lmax + 1)
        for l in range(lmax + 1):
            valid_m_counts[l] = min(l, mmax) + 1
        self.register_buffer("valid_m_counts", valid_m_counts, persistent=True)

        # Create per-degree normalization weights for vectorized computation
        # Shape: [lmax+1, mmax+1]
        # For each position (l, m) where m <= l and m <= mmax:
        # - For l=0: weight = 1.0 (only real component counts, single element)
        # - For l>0: weight = 1.0 / (num_valid_m * 2 - 1)
        #   where num_valid_m = min(l, mmax) + 1
        #   The denominator accounts for: num_valid_m real components +
        #   (num_valid_m - 1) imaginary components (m=0 imag is always 0)
        per_degree_norm_weight = torch.zeros(lmax + 1, mmax + 1)
        for l in range(lmax + 1):
            num_valid_m = min(l, mmax) + 1
            if l == 0:
                # l=0: only 1 real component
                per_degree_norm_weight[0, 0] = 1.0
            else:
                # l>0: num_valid_m * 2 - 1 components (m=0 has no imaginary)
                denom = num_valid_m * 2 - 1
                for m in range(num_valid_m):
                    per_degree_norm_weight[l, m] = 1.0 / denom
        self.register_buffer(
            "per_degree_norm_weight", per_degree_norm_weight, persistent=True
        )

        # Create l0_only_mask for bias application
        # Shape: [1, lmax+1, 1, 1, 1]
        # Value 1.0 at l=0, 0.0 elsewhere
        l0_only_mask = torch.zeros(1, lmax + 1, 1, 1, 1)
        l0_only_mask[0, 0, 0, 0, 0] = 1.0
        self.register_buffer("l0_only_mask", l0_only_mask, persistent=True)

        # Precompute scaled bias mask: l0_only_mask * bias_scale
        # This eliminates one multiplication per forward pass
        # Note: EquivariantLayerNorm uses l0_only_mask instead of l0_subtract_mean_mask
        scaled_bias_mask = l0_only_mask * self.bias_scale
        self.register_buffer("scaled_bias_mask", scaled_bias_mask, persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, x: Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
    ) -> Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]:
        # Validate input shape (skip during torch.compile)
        if not torch.compiler.is_compiling():
            self._validate_input_shape(x)

        # Get compute dtype and prepare input
        input_dtype = x.dtype
        compute_dtype = self._get_compute_dtype(input_dtype)
        x = self._prepare_input(x, compute_dtype)

        # Mean subtraction for l=0 (branch-free using subtract_mean_scale and l0_subtract_mean_mask)
        # Compute mean of l=0, m=0, real component
        l0_mean = x[:, 0:1, 0:1, 0:1, :].mean(
            dim=-1, keepdim=True
        )  # [batch, 1, 1, 1, 1]
        # Subtract mean only at l=0, m=0, real - controlled by precomputed scaled_l0_mask
        x = x - l0_mean * self.scaled_l0_mask.to(compute_dtype)

        # Compute per-degree squared values
        x_squared = x.pow(2)  # [batch, lmax+1, mmax+1, 2, channels]

        # Compute per-degree norm using einsum
        # per_degree_norm_weight: [lmax+1, mmax+1]
        # x_squared: [batch, lmax+1, mmax+1, 2, channels]
        # Result: [batch, lmax+1, 2, channels]
        feature_norm = torch.einsum(
            "blmrc, lm -> blrc",
            x_squared,
            self.per_degree_norm_weight.to(compute_dtype),
        )

        # Sum over real/imag dimension
        feature_norm = feature_norm.sum(
            dim=2, keepdim=True
        )  # [batch, lmax+1, 1, channels]

        # Average over channels and compute inverse RMS
        feature_norm = feature_norm.mean(dim=-1, keepdim=True)  # [batch, lmax+1, 1, 1]
        inv_rms = (feature_norm + self.eps).pow(-0.5)  # [batch, lmax+1, 1, 1]

        # Apply scaling: broadcast [batch, lmax+1, 1, 1] to [batch, lmax+1, mmax+1, 2, channels]
        x = x * inv_rms[:, :, :, None, :]

        # Apply affine weight: [lmax+1, channels] -> [1, lmax+1, 1, 1, channels]
        affine_weight = (
            self.affine_weight if self.affine else self._affine_weight_buffer
        )
        x = x * affine_weight[None, :, None, None, :]

        # Apply bias to l=0 only: [channels] -> [1, 1, 1, 1, channels] * scaled_bias_mask
        # Use Parameter if affine=True AND subtract_mean=True, else use buffer
        affine_bias = (
            self.affine_bias
            if self.affine and self.affine_bias is not None
            else self._affine_bias_buffer
        )
        bias = affine_bias[None, None, None, None, :]  # [1, 1, 1, 1, channels]
        x = x + bias * self.scaled_bias_mask.to(compute_dtype)

        # Finalize output
        return self._finalize_output(x, compute_dtype, input_dtype)

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"lmax={self.lmax}, mmax={self.mmax}, num_channels={self.num_channels}, "
            f"subtract_mean={self.subtract_mean}, affine={self.affine}, eps={self.eps}"
        )


# =============================================================================
# Fused Equivariant Normalization Classes (Warp GPU Kernels)
# =============================================================================


class FusedEquivariantRMSNorm(EquivariantRMSNorm):
    r"""Fused RMS normalization for spherical harmonic features using Warp GPU kernels.

    This class is a performance-optimized variant of :class:`EquivariantRMSNorm`
    that uses custom Warp GPU kernels for accelerated computation. When Warp is not
    available or when running on CPU, it falls back to the standard PyTorch implementation
    by delegating to the parent class.

    The fused implementation provides identical mathematical behavior to the unfused
    version but with reduced memory bandwidth and kernel launch overhead through
    operation fusion on the GPU.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be non-negative.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    num_channels : int
        Number of feature channels. Must be positive.
    subtract_mean : bool, optional
        Whether to subtract mean from l=0 component. Default is True.
    std_balance_degrees : bool, optional
        Whether to balance degree contributions to the norm. Default is True.
    affine : bool, optional
        Whether to apply learnable affine parameters. Default is True.
    eps : float, optional
        Small constant for numerical stability in division. Default is 1e-5.

    Attributes
    ----------
    _use_fused : bool
        Class attribute indicating whether fused kernels are active.
        Currently False (will be True once Warp kernels are implemented).

    Notes
    -----
    **Current Implementation:**

    This class currently inherits all functionality from :class:`EquivariantRMSNorm`.
    In future steps, the forward pass will be replaced with Warp GPU kernel calls that
    fuse the normalization operations for improved performance.

    **Inheritance Strategy:**

    To avoid code duplication during the scaffolding phase, this class directly inherits
    from the unfused implementation. The forward() method will be overridden in later
    refactor steps to use Warp kernels when available.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry import FusedEquivariantRMSNorm
    >>> norm = FusedEquivariantRMSNorm(lmax=4, mmax=2, num_channels=64)
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> y = norm(x)  # Currently uses PyTorch implementation
    >>> y.shape
    torch.Size([100, 5, 3, 2, 64])

    See Also
    --------
    EquivariantRMSNorm : Unfused PyTorch reference implementation.
    FusedEquivariantLayerNormTied : Fused LayerNorm variant for l=0 + global scaling for l>0.
    FusedEquivariantLayerNorm : Fused per-degree normalization variant.
    """

    _use_fused: bool = True

    def forward(
        self, x: Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
    ) -> Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]:
        # Validate input shape (skip during torch.compile)
        if not torch.compiler.is_compiling():
            self._validate_input_shape(x)

        # Fall back to PyTorch on CPU or when requested
        if not x.is_cuda or not self._use_fused:
            return super().forward(x)

        # Get compute dtype and cast if needed
        # Note: Warp kernels only support float32, so we cast float64 to float32
        input_dtype = x.dtype
        compute_dtype = self._get_compute_dtype(input_dtype)
        if compute_dtype == torch.float64:
            compute_dtype = torch.float32
        if input_dtype != compute_dtype:
            x = x.to(compute_dtype)

        # Get the appropriate parameters
        affine_weight = (
            self.affine_weight if self.affine else self._affine_weight_buffer
        )
        affine_bias = (
            self.affine_bias
            if self.affine and self.affine_bias is not None
            else self._affine_bias_buffer
        )
        balance_weight = (
            self.balance_degree_weight
            if self.std_balance_degrees
            else self._balance_degree_weight_buffer
        )

        # Determine has_bias flag
        has_bias = self.subtract_mean and self.bias_scale.item() > 0.0

        # Call custom op (handles both forward and backward via autograd)
        output = fused_rmsnorm(
            x,
            affine_weight.to(compute_dtype),
            affine_bias.to(compute_dtype),
            balance_weight.to(compute_dtype),
            self.grid_mask_3d.to(compute_dtype),
            self.lmax,
            self.mmax,
            self.num_channels,
            self.eps,
            self.subtract_mean,
            has_bias,
        )

        # Cast back if needed
        if compute_dtype != input_dtype:
            output = output.to(input_dtype)

        return output


class FusedEquivariantLayerNormTied(EquivariantLayerNormTied):
    r"""Fused layer normalization for spherical harmonic features using Warp GPU kernels.

    This class is a performance-optimized variant of :class:`EquivariantLayerNormTied`
    that uses custom Warp GPU kernels for accelerated computation. When Warp is not
    available or when running on CPU, it falls back to the standard PyTorch implementation
    by delegating to the parent class.

    The fused implementation provides identical mathematical behavior to the unfused
    version but with reduced memory bandwidth and kernel launch overhead through
    operation fusion on the GPU.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be >= 1.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    num_channels : int
        Number of feature channels. Must be positive.
    std_balance_degrees : bool, optional
        Whether to balance degree contributions to the norm for l>0.
        Default is True.
    affine : bool, optional
        Whether to apply learnable affine parameters. Default is True.
    eps : float, optional
        Small constant for numerical stability. Default is 1e-5.

    Attributes
    ----------
    _use_fused : bool
        Class attribute indicating whether fused kernels are active.
        Currently False (will be True once Warp kernels are implemented).

    Notes
    -----
    **Current Implementation:**

    This class currently inherits all functionality from :class:`EquivariantLayerNormTied`.
    In future steps, the forward pass will be replaced with Warp GPU kernel calls that
    fuse the normalization operations for improved performance.

    **Inheritance Strategy:**

    To avoid code duplication during the scaffolding phase, this class directly inherits
    from the unfused implementation. The forward() method will be overridden in later
    refactor steps to use Warp kernels when available.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry import FusedEquivariantLayerNormTied
    >>> norm = FusedEquivariantLayerNormTied(lmax=4, mmax=2, num_channels=64)
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> y = norm(x)  # Currently uses PyTorch implementation
    >>> y.shape
    torch.Size([100, 5, 3, 2, 64])

    See Also
    --------
    EquivariantLayerNormTied : Unfused PyTorch reference implementation.
    FusedEquivariantRMSNorm : Fused RMS normalization variant.
    FusedEquivariantLayerNorm : Fused per-degree normalization variant.
    """

    _use_fused: bool = True

    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_channels: int,
        std_balance_degrees: bool = True,
        affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        """Initialize FusedEquivariantLayerNormTied with additional buffers for Warp kernels."""
        super().__init__(lmax, mmax, num_channels, std_balance_degrees, affine, eps)

        # Register l>0 grid mask for Warp kernels: slice from grid_mask_3d
        # grid_mask_3d: [lmax+1, mmax+1, 2] -> grid_mask_3d_lgt0: [lmax, mmax+1, 2]
        grid_mask_3d_lgt0 = self.grid_mask_3d[1:, :, :].contiguous()
        self.register_buffer("grid_mask_3d_lgt0", grid_mask_3d_lgt0, persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, x: Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
    ) -> Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]:
        # Validate input shape (skip during torch.compile)
        if not torch.compiler.is_compiling():
            self._validate_input_shape(x)

        # Fall back to PyTorch on CPU or explicitly requested
        if not x.is_cuda or not self._use_fused:
            return super().forward(x)

        # Get compute dtype and prepare input
        input_dtype = x.dtype
        compute_dtype = self._get_compute_dtype(input_dtype)
        # Warp kernels only support float32
        if compute_dtype == torch.float64:
            compute_dtype = torch.float32
        if input_dtype != compute_dtype:
            x = x.to(compute_dtype)

        batch_size = x.shape[0]

        # Process l=0 with standard LayerNorm
        l0_feature = x[:, 0, 0, 0, :]  # [batch, channels]
        l0_normed = self.norm_l0(l0_feature.to(input_dtype)).to(compute_dtype)

        # Process l>0 with custom op (handles forward + backward via autograd)
        x_lgt0 = x[:, 1:, :, :, :]  # [batch, lmax, mmax+1, 2, channels]

        # Get parameters
        balance_weight = (
            self.balance_degree_weight
            if self.std_balance_degrees
            else self._balance_degree_weight_buffer
        )
        balance_weight_lgt0 = balance_weight[1:, :].to(compute_dtype)
        affine_weight = (
            self.affine_weight if self.affine else self._affine_weight_buffer
        )

        output_lgt0 = fused_layernormsh_lgt0(
            x_lgt0.contiguous(),
            affine_weight.to(compute_dtype),
            balance_weight_lgt0,
            self.grid_mask_3d_lgt0.to(compute_dtype),
            self.lmax,
            self.mmax,
            self.num_channels,
            self.eps,
        )

        # Assemble output
        output = torch.zeros_like(x)
        output[:, 0, 0, 0, :] = l0_normed
        output[:, 1:, :, :, :] = output_lgt0

        # Finalize output
        return self._finalize_output(output, compute_dtype, input_dtype)


class FusedEquivariantLayerNorm(EquivariantLayerNorm):
    r"""Fused per-degree layer normalization for spherical harmonic features using Warp GPU kernels.

    This class is a performance-optimized variant of :class:`EquivariantLayerNorm`
    that uses custom Warp GPU kernels for accelerated computation. When Warp is not
    available or when running on CPU, it falls back to the standard PyTorch implementation
    by delegating to the parent class.

    The fused implementation provides identical mathematical behavior to the unfused
    version but with reduced memory bandwidth and kernel launch overhead through
    operation fusion on the GPU.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be non-negative.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    num_channels : int
        Number of feature channels. Must be positive.
    subtract_mean : bool, optional
        Whether to subtract mean from l=0 component. Default is True.
    affine : bool, optional
        Whether to apply learnable affine parameters. Default is True.
    eps : float, optional
        Small constant for numerical stability. Default is 1e-5.

    Attributes
    ----------
    _use_fused : bool
        Class attribute indicating whether fused kernels are active.
        Currently False (will be True once Warp kernels are implemented).

    Notes
    -----
    **Current Implementation:**

    This class currently inherits all functionality from :class:`EquivariantLayerNorm`.
    In future steps, the forward pass will be replaced with Warp GPU kernel calls that
    fuse the normalization operations for improved performance.

    **Inheritance Strategy:**

    To avoid code duplication during the scaffolding phase, this class directly inherits
    from the unfused implementation. The forward() method will be overridden in later
    refactor steps to use Warp kernels when available.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry import FusedEquivariantLayerNorm
    >>> norm = FusedEquivariantLayerNorm(lmax=4, mmax=2, num_channels=64)
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> y = norm(x)  # Currently uses PyTorch implementation
    >>> y.shape
    torch.Size([100, 5, 3, 2, 64])

    See Also
    --------
    EquivariantLayerNorm : Unfused PyTorch reference implementation.
    FusedEquivariantRMSNorm : Fused RMS normalization variant.
    FusedEquivariantLayerNormTied : Fused LayerNorm variant for l=0 + global scaling for l>0.
    """

    _use_fused: bool = True  # Warp kernels are now integrated

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, x: Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]
    ) -> Float[Tensor, "batch lmax_p1 mmax_p1 2 channels"]:
        if not torch.compiler.is_compiling():
            self._validate_input_shape(x)

        if not x.is_cuda or not self._use_fused:
            return super().forward(x)

        input_dtype = x.dtype
        compute_dtype = self._get_compute_dtype(input_dtype)
        if compute_dtype == torch.float64:
            compute_dtype = torch.float32
        if input_dtype != compute_dtype:
            x = x.to(compute_dtype)

        affine_weight = (
            self.affine_weight if self.affine else self._affine_weight_buffer
        )
        affine_bias = (
            self.affine_bias
            if self.affine and self.affine_bias is not None
            else self._affine_bias_buffer
        )
        has_bias = self.subtract_mean and self.bias_scale.item() > 0.0

        output = fused_layernorm(
            x,
            affine_weight.to(compute_dtype),
            affine_bias.to(compute_dtype),
            self.per_degree_norm_weight.to(compute_dtype),
            self.grid_mask_3d.to(compute_dtype),
            self.lmax,
            self.mmax,
            self.num_channels,
            self.eps,
            self.subtract_mean,
            has_bias,
        )

        if compute_dtype != input_dtype:
            output = output.to(input_dtype)

        return output
