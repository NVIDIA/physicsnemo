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

"""SO(3) equivariant linear layer using grid layout.

This module provides an SO(3) equivariant linear layer that operates on
spherical harmonic representations with degree-wise weight sharing.

The key insight is that rotations only mix coefficients within the same
degree l (they mix m values but not l values). Therefore, applying separate
linear transformations per degree l preserves SO(3) equivariance.

The grid layout uses explicit dimensions for degree (l), order (m), and
real/imaginary components, enabling efficient vectorized operations via
a single einsum call with masking for invalid positions.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import nn

from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask
from physicsnemo.nn import Module

__all__ = [
    "SO3LinearGrid",
]


class SO3LinearGrid(Module):
    r"""SO(3) equivariant linear layer using grid layout.

    Applies separate linear transformations per spherical harmonic degree l,
    operating on coefficients arranged in an explicit grid layout.
    This preserves SO(3) equivariance because rotations only mix coefficients
    within the same l (they mix m values but not l values).

    The grid layout uses fixed-size tensors with masking to handle invalid
    (l, m) positions where m > l, enabling efficient vectorized operations.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels per coefficient.
    out_channels : int
        Number of output feature channels per coefficient.
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int
        Maximum spherical harmonic order. Must be <= lmax.
    bias : bool, optional
        If True, adds bias only to the (l=0, m=0, real) component.
        Default: True.

    Attributes
    ----------
    weight : nn.Parameter
        Weight matrix for each degree l. Shape: ``[lmax+1, out_channels, in_channels]``.
    bias : nn.Parameter or None
        Bias for the scalar (l=0, m=0, real) component. Shape: ``[out_channels]``.
    mask : torch.Tensor
        Float mask of shape ``[lmax+1, mmax+1]`` where 1.0 indicates valid
        (l, m) positions (i.e., m <= l), 0.0 otherwise.

    Notes
    -----
    Input tensor layout: ``[batch, lmax+1, mmax+1, 2, in_channels]``
        - batch: number of samples
        - lmax+1: degrees from l=0 to l=lmax
        - mmax+1: orders from m=0 to m=mmax
        - 2: real (index 0) and imaginary (index 1) components
        - in_channels: feature channels

    Output tensor layout: ``[batch, lmax+1, mmax+1, 2, out_channels]``

    Each degree l has its own weight matrix of shape ``[out_channels, in_channels]``.
    All m-values and real/imaginary components for a given l share the same weights.

    The bias is only added to the (l=0, m=0, real) position, which is the
    SO(3) scalar invariant. This preserves equivariance since higher-order
    spherical harmonics transform non-trivially under rotations.

    Positions where m > l are invalid and masked to zero. This is handled
    automatically via the validity mask.

    Examples
    --------
    >>> linear = SO3LinearGrid(
    ...     in_channels=64,
    ...     out_channels=128,
    ...     lmax=4,
    ...     mmax=2,
    ... )
    >>> # Input: [batch=10, lmax+1=5, mmax+1=3, 2, in_channels=64]
    >>> x = torch.randn(10, 5, 3, 2, 64)
    >>> out = linear(x)
    >>> out.shape
    torch.Size([10, 5, 3, 2, 128])

    See Also
    --------
    SO3ConvolutionBlock : SO(3) node-wise transformation layer using SO3LinearGrid.

    Forward
    -------
    x : Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"]
        Input tensor with spherical harmonic features in grid layout.
        Shape: ``[batch, lmax+1, mmax+1, 2, in_channels]``.

    Outputs
    -------
    Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 out_channels"]
        Transformed features with shape ``[batch, lmax+1, mmax+1, 2, out_channels]``.
        The transformation applies degree-wise linear transforms where each
        degree l has its own weight matrix. Positions where m > l are masked
        to zero after the transformation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lmax: int,
        mmax: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Parameter validation
        if lmax < 0:
            raise ValueError(f"lmax must be non-negative, got {lmax}")
        if mmax < 0:
            raise ValueError(f"mmax must be non-negative, got {mmax}")
        if mmax > lmax:
            raise ValueError(f"mmax ({mmax}) must be <= lmax ({lmax})")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lmax = lmax
        self.mmax = mmax

        # Weight matrix for each l: shape [lmax+1, out_channels, in_channels]
        # Using [out, in] ordering to match nn.Linear convention for einsum
        self.weight = nn.Parameter(torch.empty(lmax + 1, out_channels, in_channels))

        # Bias only for (l=0, m=0, real) - the scalar invariant
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        self.reset_parameters()

        # Create and register the validity mask for (l, m) positions
        # mask[l, m] = 1.0 if m <= l (valid position), 0.0 otherwise
        mask = make_grid_mask(lmax, mmax).float()
        self.register_buffer("mask", mask, persistent=False)

    def reset_parameters(self) -> None:
        """Initialize parameters using uniform distribution.

        Uses uniform(-1/sqrt(in_channels), 1/sqrt(in_channels)).
        """
        bound = 1 / math.sqrt(self.in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(
        self,
        x: Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"],
    ) -> Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 out_channels"]:
        # Apply linear transformation via einsum
        # x: [batch, lmax+1, mmax+1, 2, in_channels]
        # weight: [lmax+1, out_channels, in_channels]
        # out: [batch, lmax+1, mmax+1, 2, out_channels]
        # einsum contracts over in_channels, uses l to select weights,
        # and broadcasts over m and r (real/imag)
        out = torch.einsum("blmrc,loc->blmro", x, self.weight)

        # Add bias only to (l=0, m=0, real) - the scalar invariant
        if self.bias is not None:
            out[:, 0, 0, 0, :] = out[:, 0, 0, 0, :] + self.bias

        # Apply mask for invalid (l, m) positions where m > l
        # mask: [lmax+1, mmax+1] -> broadcast to [1, lmax+1, mmax+1, 1, 1]
        mask: torch.Tensor = self.mask.to(dtype=x.dtype)  # type: ignore[assignment]
        out = out * mask[None, :, :, None, None]

        return out

    def extra_repr(self) -> str:
        """Return a string with extra representation info."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"lmax={self.lmax}, "
            f"mmax={self.mmax}, "
            f"bias={self.bias is not None}"
        )
