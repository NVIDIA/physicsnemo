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

r"""SO(3) block-wise transformation layer using grid layout.

This module provides a feed-forward block for features in the
spectral (spherical harmonic) domain, using the unified grid layout.

The architecture applies a sequence of SO(3)-equivariant linear transformations
with gated activations, enabling expressive block-wise transformations while
preserving rotational equivariance.

Classes
-------
SO3ConvolutionBlock
    SO(3) block-wise transformation using SO3Linear -> GateActivation -> SO3Linear.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import nn

from physicsnemo.experimental.nn.symmetry.activation import GateActivation
from physicsnemo.experimental.nn.symmetry.so3_linear import SO3LinearGrid
from physicsnemo.nn import Module
from physicsnemo.models.mlp import FullyConnected

__all__ = [
    "SO3ConvolutionBlock",
]


class SO3ConvolutionBlock(Module):
    r"""SO(3) block-wise transformation layer using grid layout.

    This module applies block-wise transformations in the spectral
    domain using SO(3) linear layers with gated activations. It serves as a
    feed-forward block in equivariant architectures.

    The architecture is: SO3Linear -> GateActivation -> SO3Linear

    Gates are computed from the scalar (l=0, m=0, real) component of the input
    via a small MLP, then used to modulate higher-order features.

    Parameters
    ----------
    in_channels : int
        Number of input/output feature channels.
    hidden_channels : int
        Number of hidden channels in the intermediate representation.
    lmax : int
        Maximum spherical harmonic degree. Must be >= 1.
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    num_hidden_layers : int
        Number of hidden layers in then fully connected scalar
        transformation.
    **fc_kwargs
        Extra arguments are unpacked into the creation of
        :class:FullyConnected

    Attributes
    ----------
    num_gates : int
        Number of gate channels: lmax * hidden_channels.
    scalar_mlp : nn.Sequential
        MLP computing gates from scalar input features.
    so3_linear_1 : SO3LinearGrid
        First SO(3) equivariant linear layer.
    act : GateActivation
        Gated activation layer.
    so3_linear_2 : SO3LinearGrid
        Second SO(3) equivariant linear layer.

    Notes
    -----
    Input tensor layout: ``[batch, lmax+1, mmax+1, 2, in_channels]``
        - batch: number of elements
        - lmax+1: degrees from l=0 to l=lmax
        - mmax+1: orders from m=0 to m=mmax
        - 2: real (index 0) and imaginary (index 1) components
        - in_channels: feature channels

    Output tensor layout: ``[batch, lmax+1, mmax+1, 2, in_channels]``
        - Same shape as input (residual-friendly)

    The gate computation extracts scalar features from position (l=0, m=0, real),
    passes them through a Linear -> SiLU network, and embeds the resulting gates
    into the hidden representation for the GateActivation to consume.

    Examples
    --------
    >>> block = SO3ConvolutionBlock(
    ...     in_channels=64,
    ...     hidden_channels=128,
    ...     lmax=4,
    ...     mmax=2,
    ... )
    >>> # Input: [num_elements, lmax+1, mmax+1, 2, in_channels]
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> out = block(x)
    >>> out.shape
    torch.Size([100, 5, 3, 2, 64])

    See Also
    --------
    SO3LinearGrid : SO(3) equivariant linear layer.
    GateActivation : Gated activation for equivariant features.
    physicsnemo.models.mlp.FullyConnected : MLP Block used for scalar transforms.

    Forward
    -------
    x : Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"]
        Input tensor with shape ``[batch, lmax+1, mmax+1, 2, in_channels]``.

    Outputs
    -------
    Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"]
        Transformed tensor with same shape as input. The transformation applies
        SO3Linear -> GateActivation -> SO3Linear, where gates are computed from
        scalar (l=0, m=0, real) features via a small MLP.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        num_hidden_layers: int = 3,
        **fc_kwargs,
    ) -> None:
        fc_kwargs.setdefault("activation_fn", "silu")
        super().__init__()

        # Validate parameters
        if lmax < 1:
            raise ValueError(f"lmax must be >= 1 for SO3ConvolutionBlock, got {lmax}")
        if mmax < 0:
            raise ValueError(f"mmax must be non-negative, got {mmax}")
        if mmax > lmax:
            raise ValueError(f"mmax ({mmax}) must be <= lmax ({lmax})")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        # Number of gates needed: one per l>0 degree, times hidden_channels
        self.num_gates = lmax * hidden_channels

        # Scalar MLP to compute gates from l=0 scalar features
        # Input: in_channels (from x[:, 0, 0, 0, :])
        # Output: lmax * hidden_channels (gates for degrees l=1..lmax)
        self.scalar_mlp = FullyConnected(
            in_features=in_channels,
            layer_size=hidden_channels,
            out_features=self.num_gates,
            num_layers=num_hidden_layers,
            **fc_kwargs,
        )

        # SO3 linear layers
        self.so3_linear_1 = SO3LinearGrid(
            in_channels=in_channels,
            out_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            bias=True,
        )

        # Gated activation expects input with embedded gates
        self.act = GateActivation(
            lmax=lmax,
            mmax=mmax,
            channels=hidden_channels,
        )

        self.so3_linear_2 = SO3LinearGrid(
            in_channels=hidden_channels,
            out_channels=in_channels,
            lmax=lmax,
            mmax=mmax,
            bias=True,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"],
    ) -> Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"]:
        batch = x.shape[0]

        # Extract scalar features (l=0, m=0, real component)
        scalars = x[:, 0, 0, 0, :]  # [batch, in_channels]

        # Compute gates from scalar features
        gates = self.scalar_mlp(scalars)  # [batch, lmax * hidden_channels]

        # Apply first SO3 linear
        h = self.so3_linear_1(x)  # [batch, lmax+1, mmax+1, 2, hidden_channels]

        # Embed gates into hidden representation
        # Create tensor with space for gates
        total_channels = self.hidden_channels + self.num_gates
        h_with_gates = torch.zeros(
            batch,
            self.lmax + 1,
            self.mmax + 1,
            2,
            total_channels,
            dtype=h.dtype,
            device=h.device,
        )

        # Copy hidden features
        h_with_gates[..., : self.hidden_channels] = h

        # Embed gates at (l=0, m=0, real=0) position
        h_with_gates[:, 0, 0, 0, self.hidden_channels :] = gates

        # Apply gated activation (consumes the embedded gates)
        h = self.act(h_with_gates)  # [batch, lmax+1, mmax+1, 2, hidden_channels]

        # Apply second SO3 linear
        out = self.so3_linear_2(h)  # [batch, lmax+1, mmax+1, 2, in_channels]

        return out

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"lmax={self.lmax}, "
            f"mmax={self.mmax}"
        )
