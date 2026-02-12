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

r"""Activation functions for SO(2)/SO(3) equivariant neural networks.

This module provides gated activations for equivariant features in grid layout.
The key insight is that scalar features (l=0) can have arbitrary pointwise
activations applied, while higher-order features (l>0) must be scaled by
invariant gates to preserve equivariance.

This implementation is thematically similar and takes inspirations from
the ``fairchem`` (MIT Licensed) repository, deviating in how data is
laid out and thus how computation is performed.

Classes
-------
GateActivation
    Gated activation applying a nonlinearity to l=0 and learned gating to l>0.
"""

from __future__ import annotations
from typing import Callable

import torch
from jaxtyping import Float
from torch import nn, Tensor

from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask
from physicsnemo.nn import get_activation, Module

__all__ = [
    "GateActivation",
]


class GateActivation(Module):
    r"""Gated activation for grid-layout equivariant features.

    Applies different activations based on spherical harmonic degree:
    - l=0 (scalars): Nonlinear activation
    - l>0 (vectors/tensors): Multiplication by learned gates passed through sigmoid

    This preserves SO(2) equivariance because:
    - l=0 features are invariant under rotation, so any pointwise activation preserves this
    - l>0 features are scaled by invariant scalars (gates), which commutes with rotation

    The gates are extracted from the input tensor itself, embedded in the channel
    dimension at the (l=0, m=0, real=0) position. This is the typical output format
    from ``SO2Convolution`` with ``produce_gates=True``, which produces both the
    main feature channels and additional gate channels in a single tensor.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree. Must be >= 1 (need at least one
        gated degree for this activation to be meaningful).
    mmax : int
        Maximum spherical harmonic order. Must satisfy 0 <= mmax <= lmax.
    channels : int
        Number of output feature channels (not including gates).

    Attributes
    ----------
    l0_mask : torch.Tensor
        Mask selecting l=0 positions. Shape: [lmax+1, 1, 1, 1].
    gate_indices : torch.Tensor
        Index mapping from l to gate index. Shape: [lmax+1].
    validity_mask : torch.Tensor
        Mask for valid (l, m) positions. Shape: [lmax+1, mmax+1].
    m0_imag_mask : torch.Tensor
        Mask zeroing m=0 imaginary component. Shape: [1, 1, mmax+1, 2, 1].

    Examples
    --------
    >>> # Typical usage with SO2Convolution
    >>> # conv = SO2Convolution(in_channels=64, out_channels=64, lmax=4, mmax=2,
    >>> #                       produce_gates=True)
    >>> act = GateActivation(lmax=4, mmax=2, channels=64)
    >>>
    >>> # Input has embedded gates: 64 feature channels + 4*64 gate channels = 320
    >>> x = torch.randn(100, 5, 3, 2, 320)  # [batch, lmax+1, mmax+1, 2, channels_with_gates]
    >>> out = act(x)  # Output: [100, 5, 3, 2, 64] (gates consumed)
    >>> out.shape
    torch.Size([100, 5, 3, 2, 64])

    Notes
    -----
    Input tensor layout: [batch, lmax+1, mmax+1, 2, channels + gate_channels]
        - batch: number of samples/edges
        - lmax+1: degrees from l=0 to l=lmax
        - mmax+1: orders from m=0 to m=mmax
        - 2: real (index 0) and imaginary (index 1) components
        - channels + gate_channels: first ``channels`` are features,
          remaining ``lmax * channels`` are gates

    The gates are extracted from the (l=0, m=0, real=0) position in the input
    tensor. The first ``channels`` values at this position are the l=0 features,
    and the remaining ``lmax * channels`` values are the gates for l>0 degrees.

    Output tensor layout: [batch, lmax+1, mmax+1, 2, channels]
        - Same structure as input but without the gate channels
        - Gate channels are consumed during the activation, not passed through

    See Also
    --------
    SO2Convolution - SO2-preserving linear transformation layer.

    Forward
    -------
    x : Float[Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels_with_gates"]
        Input features with shape [batch, lmax+1, mmax+1, 2, channels + gate_channels].
        The dimension of size 2 contains real (index 0) and imaginary
        (index 1) components. The last dimension contains both the feature
        channels (first ``channels`` values) and gate channels (remaining
        ``lmax * channels`` values).

    Outputs
    -------
    Float[Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]
        Activated features with shape [batch, lmax+1, mmax+1, 2, channels].
        - l=0 positions have a nonlinearity applied
        - l>0 positions are scaled by sigmoid(gates)
        - Invalid (l,m) positions (where m > l) are zero
        - m=0 imaginary components are zero
        - Gate channels are consumed (not in output)
    """

    def __init__(
        self, lmax: int, mmax: int, channels: int, activation: str | Callable = "silu"
    ) -> None:
        super().__init__()

        if lmax < 1:
            raise ValueError(f"lmax must be >= 1 for gated activation, got {lmax}")
        if mmax < 0:
            raise ValueError(f"mmax must be non-negative, got {mmax}")
        if mmax > lmax:
            raise ValueError(f"mmax ({mmax}) must be <= lmax ({lmax})")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.lmax = lmax
        self.mmax = mmax
        self.channels = channels

        # Mask for l=0 positions: 1 at l=0, 0 elsewhere
        # Shape: [lmax+1, 1, 1, 1] for broadcasting over [batch, lmax+1, mmax+1, 2, channels]
        l0_mask = torch.zeros(lmax + 1, 1, 1, 1)
        l0_mask[0] = 1.0
        self.register_buffer("l0_mask", l0_mask, persistent=True)

        # Index mapping: l -> gate index
        # l=0 maps to 0 (unused, will be masked out)
        # l=1 maps to 0, l=2 maps to 1, ..., l=lmax maps to lmax-1
        # Using clamp to handle l=0 safely
        gate_indices = torch.arange(lmax + 1).clamp(min=1) - 1
        self.register_buffer("gate_indices", gate_indices, persistent=True)

        # Validity mask for (l, m) positions where m <= l
        # Shape: [lmax+1, mmax+1]
        validity_mask = make_grid_mask(lmax, mmax).float()
        self.register_buffer("validity_mask", validity_mask, persistent=True)

        # Mask to zero out m=0 imaginary component
        # Shape: [1, 1, mmax+1, 2, 1] for broadcasting
        m0_imag_mask = torch.ones(1, 1, mmax + 1, 2, 1)
        m0_imag_mask[:, :, 0, 1, :] = 0.0
        self.register_buffer("m0_imag_mask", m0_imag_mask, persistent=True)

        # Activation functions
        if isinstance(activation, Callable):
            self.scalar_act = activation
        elif isinstance(activation, str):
            self.scalar_act = get_activation(activation)
        else:
            raise RuntimeError(
                f"GateActivation supports Callable or str specification of activation functions. Got {activation}."
            )
        self.gate_act = nn.Sigmoid()

    @property
    def gate_channels(self) -> int:
        """Number of gate channels (lmax * channels)."""
        return self.lmax * self.channels

    @property
    def total_in_channels(self) -> int:
        """Total input channels including embedded gates."""
        return self.channels + self.gate_channels

    def forward(
        self,
        x: Float[Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels_with_gates"],
    ) -> Float[Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]:
        # Validate input channel dimension (skip during torch.compile for performance)
        if not torch.compiler.is_compiling():
            expected_in_channels = self.total_in_channels
            if x.shape[-1] != expected_in_channels:
                raise ValueError(
                    f"Expected input with {expected_in_channels} channels "
                    f"(channels={self.channels} + gate_channels={self.gate_channels}), "
                    f"got {x.shape[-1]}"
                )

        batch = x.shape[0]

        # Equivalent to a split, but easier to read
        # Extract gates from embedded channels at (l=0, m=0, real=0)
        gates = x[:, 0, 0, 0, self.channels :]  # [batch, lmax * channels]

        # Extract main features (i.e. not the gate channels)
        x_features = x[..., : self.channels]  # [batch, lmax+1, mmax+1, 2, channels]

        # Zero out m=0 imaginary component before processing
        x_features = x_features * self.m0_imag_mask

        # Process gates: [batch, lmax * channels] -> [batch, lmax, channels]
        gates = gates.view(batch, self.lmax, self.channels)
        gates = self.gate_act(gates)

        # Expand gates to match grid layout
        # gate_indices maps each l to its gate index (l-1 for l>0, 0 for l=0)
        # Result shape: [batch, lmax+1, channels]
        expanded_gates = gates[:, self.gate_indices, :]

        # Add dimensions for broadcasting over m and real/imag
        # Shape: [batch, lmax+1, 1, 1, channels]
        expanded_gates = expanded_gates[:, :, None, None, :]

        # Compute l>0 mask (complement of l0_mask)
        lgt0_mask = 1.0 - self.l0_mask

        # Apply activations using masks (no branching for torch.compile compatibility)
        # l=0: nonlinear activation (scalar features)
        # l>0: gate multiplication (preserves equivariance)
        output = (
            self.scalar_act(x_features) * self.l0_mask  # l=0 contribution
            + x_features * expanded_gates * lgt0_mask  # l>0 contribution
        )

        # Apply validity mask to zero invalid (l, m) positions
        output = output * self.validity_mask[None, :, :, None, None]

        # Zero out m=0 imaginary component
        output = output * self.m0_imag_mask

        return output

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return f"lmax={self.lmax}, mmax={self.mmax}, channels={self.channels}"
