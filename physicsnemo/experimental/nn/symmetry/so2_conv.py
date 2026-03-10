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

r"""SO(2) equivariant convolution using regular tensor layout with masking.

The expected use case of this layer is to perform an equivariant convolution
on some graph. The data layout is expected to be quite rigid: we use tensor
dimensions to encode order and degree in a way that allows for a straightforward
`einsum` and mask to be used to compute real/complex operations in a single
operation, as opposed to unrolling the loop to operate on specific +/-m pairs
per parity rules. See the ``forward`` pass documentation to see what the
expected shapes and dimensions are.

This implementation is thematically similar and takes inspirations from
the ``fairchem`` (MIT Licensed) repository, deviating in how data is
laid out and thus how computation is performed.

This layout enables efficient GPU parallelization since:
1. All (l, m) positions have the same shape
2. Invalid positions (m > l) are zeroed via masking
3. The real/complex multiplication can be done with a single einsum call

Classes
-------
SO2Convolution
    SO(2) equivariant convolution layer.

Notes
-----
The grid layout trades memory efficiency (some positions are always zero)
for computational efficiency (vectorized operations, no Python loops).

The complex multiplication structure:

.. math::

    (x_r + i x_i)(W_r + i W_i) = (x_r W_r - x_i W_i) + i(x_r W_i + x_i W_r)

is implemented using a combined weight tensor that encodes the 2x2 block structure::

    | W_r  -W_i |   | x_r |   | out_r |
    | W_i   W_r | × | x_i | = | out_i |

This allows a single einsum operation to perform the complex multiplication.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import nn

from physicsnemo.nn import Module
from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask

__all__ = [
    "SO2Convolution",
]


def _build_radial_mlp(channels_list: list[int]) -> nn.Sequential:
    """Build an MLP with LayerNorm and SiLU activation between layers.

    This matches the RadialMLP architecture from the reference eSCN implementation.
    The structure is: Linear -> LayerNorm -> SiLU -> Linear -> LayerNorm -> SiLU -> ... -> Linear

    Parameters
    ----------
    channels_list : list[int]
        List of channel sizes. First element is input size, last is output size.
        Intermediate elements define hidden layer sizes. Must have at least 2 elements.

    Returns
    -------
    nn.Sequential
        The MLP as a sequential module.

    Examples
    --------
    >>> mlp = _build_radial_mlp([64, 128, 256])
    >>> # Creates: Linear(64->128) -> LayerNorm(128) -> SiLU -> Linear(128->256)
    >>> x = torch.randn(100, 64)
    >>> out = mlp(x)
    >>> out.shape
    torch.Size([100, 256])
    """
    if len(channels_list) < 2:
        raise ValueError(
            f"channels_list must have at least 2 elements, got {len(channels_list)}"
        )

    modules: list[nn.Module] = []
    for i in range(len(channels_list) - 1):
        in_ch = channels_list[i]
        out_ch = channels_list[i + 1]

        modules.append(nn.Linear(in_ch, out_ch, bias=True))

        # Add LayerNorm + SiLU for all but the last layer
        if i < len(channels_list) - 2:
            modules.append(nn.LayerNorm(out_ch))
            modules.append(nn.SiLU())

    return nn.Sequential(*modules)


class SO2Convolution(Module):
    r"""SO(2) equivariant convolution using regular, padded tensor layout.

    This layer performs SO(2) equivariant convolution on spherical harmonic
    coefficients arranged in a regular grid layout. The grid layout uses
    fixed-size tensors with masking to handle invalid (l, m) positions
    where m > l.

    The key advantage is that all m-orders are processed simultaneously via
    a single vectorized einsum operation, eliminating Python loops and
    intermediate tensors in the forward pass.

    The ``edge_channels`` mechanism also facilitates, as the name suggests,
    the mixing edge information such as distances to help break degeneracies
    in the output by modulating what information passes through l, m filters.

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
    edge_channels : int, optional
        Number of edge feature channels for input modulation. When provided,
        an MLP stack is used to compute per-coefficient scaling factors from
        edge features. The input is scaled element-wise before the actual
        SO2 linear transform. Default: None
        (use internal/shared weights without edge modulation).
    produce_gates : bool, optional
        If True, produce additional gate channels embedded in the output tensor.
        The gate channels are computed as ``lmax * out_channels`` additional
        channels that are only non-zero at the (l=0, m=0, real) position.
        These gate channels can be used for gating non-linearities to ensure
        that they are equivariance preserving. Defaults to ``False``, which
        means the output of this layer does not include the extra channels.

    Attributes
    ----------
    W_r : nn.Parameter
        Real part of complex weights.
        Shape: ``[mmax+1, in_channels, out_channels + gate_channels]``.
    W_i : nn.Parameter
        Imaginary part of complex weights.
        Shape: ``[mmax+1, in_channels, out_channels + gate_channels]``.
    W_complex : torch.Tensor
        Combined weight tensor encoding complex multiplication structure.
        Shape: ``[mmax+1, 2, 2, in_channels, out_channels + gate_channels]``.
    mask : torch.Tensor
        Float mask of shape ``[lmax+1, mmax+1]`` where 1.0 indicates valid
        (l, m) positions (i.e., m <= l), 0.0 otherwise.
    gate_channels : int
        Number of gate channels (0 if ``produce_gates=False``, else ``lmax * out_channels``).
    total_out_channels : int
        Total output channels including gate channels (``out_channels + gate_channels``).

    Notes
    -----
    Input tensor layout: ``[batch, lmax+1, mmax+1, 2, in_channels]``
        - batch: number of edges/samples
        - lmax+1: degrees from l=0 to l=lmax
        - mmax+1: orders from m=0 to m=mmax
        - 2: real (index 0) and imaginary (index 1) components
        - in_channels: feature channels

    Output tensor layout: ``[batch, lmax+1, mmax+1, 2, out_channels + gate_channels]``

    For m=0, the imaginary component is always zero (by SO(2) symmetry).
    This is enforced via explicit zeroing after the forward pass.

    When ``produce_gates=True``, the gate channels are embedded directly in the
    output tensor. A gate mask ensures that gate channel values are only non-zero
    at the (l=0, m=0, real) position, making them SO(2) invariant scalars suitable
    for gating higher-order features. The gate channels occupy indices
    ``[out_channels:]`` in the last dimension of the output tensor.

    The complex multiplication is implemented using a combined weight tensor::

        | W_r  -W_i |   | x_r |   | out_r |
        | W_i   W_r | × | x_i | = | out_i |

    This allows a single einsum to compute: ``out = einsum('blmrc,mRrco->blmRo', x, W)``

    Examples
    --------
    Basic usage without gates:

    >>> conv = SO2Convolution(
    ...     in_channels=64,
    ...     out_channels=64,
    ...     lmax=4,
    ...     mmax=2,
    ... )
    >>> # Input: [batch=100, lmax+1=5, mmax+1=3, 2, channels=64]
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> out = conv(x)
    >>> out.shape
    torch.Size([100, 5, 3, 2, 64])

    With embedded gate channels:

    >>> conv_with_gates = SO2Convolution(
    ...     in_channels=64,
    ...     out_channels=64,
    ...     lmax=4,
    ...     mmax=2,
    ...     produce_gates=True,
    ... )
    >>> x = torch.randn(100, 5, 3, 2, 64)
    >>> out = conv_with_gates(x)
    >>> # Output has out_channels + lmax * out_channels = 64 + 4*64 = 320 channels
    >>> out.shape
    torch.Size([100, 5, 3, 2, 320])
    >>> # Gate channels are at indices [64:]
    >>> gates = out[:, 0, 0, 0, 64:]  # Shape: [100, 256]

    See Also
    --------
    GateActivation - Module for applying equivariance preserving non-linearities.

    Forward
    -------
    x : Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"]
        Input tensor with shape ``[batch, lmax+1, mmax+1, 2, in_channels]``.
        The dimension of size 2 contains real (index 0) and imaginary
        (index 1) components.
    x_edge : Float[torch.Tensor, "batch edge_channels"], optional
        Edge features for input modulation. Required if ``edge_channels``
        was specified during initialization. Default: None.

    Outputs
    -------
    Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 out_channels_with_gates"]
        Output tensor with shape ``[batch, lmax+1, mmax+1, 2, out_channels + gate_channels]``.
        When ``produce_gates=False``, ``gate_channels=0`` and the output has
        ``out_channels`` features. When ``produce_gates=True``, the output has
        ``out_channels + lmax * out_channels`` features, where gate channels
        (indices ``[out_channels:]``) are non-zero only at position (l=0, m=0, real).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lmax: int,
        mmax: int,
        edge_channels: int | None = None,
        produce_gates: bool = False,
    ) -> None:
        super().__init__()

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
        self.internal_weights = edge_channels is None
        self.produce_gates = produce_gates

        # compute number of gate channels
        self.num_gate_channels = lmax * out_channels if produce_gates else 0

        # Complex weights: W = W_r + i*W_i
        # Shape: [mmax+1, in_channels, out_channels + gate_channels]
        # Each m-order has its own weight matrix
        total_out = self.total_out_channels
        self.W_r = nn.Parameter(torch.empty(mmax + 1, in_channels, total_out))
        self.W_i = nn.Parameter(torch.empty(mmax + 1, in_channels, total_out))

        # Initialize weights with proper scaling for complex structure
        self._reset_parameters()

        # Create and register the validity mask for (l, m) positions
        # mask[l, m] = 1.0 if m <= l (valid position), 0.0 otherwise
        # Convert boolean mask to float for multiplication
        mask = make_grid_mask(lmax, mmax).float()
        self.register_buffer("mask", mask, persistent=True)

        # Mask to zero out m=0 imaginary component
        # Shape: [1, 1, mmax+1, 2, 1] for broadcasting
        m0_imag_mask = torch.ones(1, 1, mmax + 1, 2, 1)
        m0_imag_mask[:, :, 0, 1, :] = 0.0  # Zero out m=0 imaginary
        self.register_buffer("m0_imag_mask", m0_imag_mask, persistent=True)

        # Gate mask: zeros gate channels except at (l=0, m=0, real=0)
        # Only create if produce_gates=True
        if self.produce_gates:
            # Shape: [1, lmax+1, mmax+1, 2, out_channels + gate_channels]
            gate_mask = torch.ones(1, lmax + 1, mmax + 1, 2, total_out)
            gate_mask[..., out_channels:] = 0.0  # Zero all gate channels
            gate_mask[:, 0, 0, 0, out_channels:] = 1.0  # Restore at (l=0, m=0, real)
            self.register_buffer("gate_mask", gate_mask, persistent=True)

        # Optional radial function for edge-dependent input modulation
        self.rad_func: nn.Module | None = None
        if not self.internal_weights:
            assert edge_channels is not None
            # Output one modulation scalar per input coefficient
            # Shape: (lmax+1) * (mmax+1) * 2 * in_channels
            rad_output_size = (self.lmax + 1) * (self.mmax + 1) * 2 * self.in_channels
            # MLP: Linear -> LayerNorm -> SiLU -> Linear
            self.rad_func = _build_radial_mlp(
                [edge_channels, edge_channels, rad_output_size]
            )

    @property
    def total_out_channels(self) -> int:
        """Total output channels including gate channels."""
        return self.out_channels + self.num_gate_channels

    def _reset_parameters(self) -> None:
        """Initialize weights with proper scaling for complex structure.

        Uses Kaiming uniform initialization scaled by 1/sqrt(2) to account
        for the complex multiplication which combines real and imaginary parts.
        """
        # Standard deviation for Kaiming init
        fan_in = self.in_channels
        std = math.sqrt(2) / math.sqrt(fan_in)

        nn.init.uniform_(self.W_r, -std, std)
        nn.init.uniform_(self.W_i, -std, std)

    @property
    def W_complex(self) -> torch.Tensor:
        """Build combined weight tensor encoding complex multiplication.

        Returns tensor of shape ``[mmax+1, 2, 2, in_channels, out_channels + gate_channels]``
        where indices are ``[m, out_ri, in_ri, in_ch, out_ch]``.

        The structure encodes the 2x2 block matrix for complex multiplication::

            | W_r  -W_i |   | x_r |   | out_r |
            | W_i   W_r | × | x_i | = | out_i |

        Specifically:
            - ``W[:, 0, 0]`` = W_r   (real_out <- real_in)
            - ``W[:, 0, 1]`` = -W_i  (real_out <- imag_in, negative for complex mult)
            - ``W[:, 1, 0]`` = W_i   (imag_out <- real_in)
            - ``W[:, 1, 1]`` = W_r   (imag_out <- imag_in)

        Returns
        -------
        torch.Tensor
            Combined weight tensor of shape
            ``[mmax+1, 2, 2, in_channels, out_channels + gate_channels]``.
        """
        total_out = self.total_out_channels
        W = torch.zeros(
            self.mmax + 1,
            2,
            2,
            self.in_channels,
            total_out,
            dtype=self.W_r.dtype,
            device=self.W_r.device,
        )
        W[:, 0, 0] = self.W_r  # real_out <- real_in
        W[:, 0, 1] = -self.W_i  # real_out <- imag_in (negative)
        W[:, 1, 0] = self.W_i  # imag_out <- real_in
        W[:, 1, 1] = self.W_r  # imag_out <- imag_in
        return W

    def forward(
        self,
        x: Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"],
        x_edge: Float[torch.Tensor, "batch edge_channels"] | None = None,
    ) -> Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 out_channels_with_gates"]:
        # Validate inputs
        if not self.internal_weights and x_edge is None:
            raise ValueError(
                "x_edge is required when edge_channels was specified at init"
            )

        # Validate input shape (skip during torch.compile for performance)
        if not torch.compiler.is_compiling():
            expected_shape = (self.lmax + 1, self.mmax + 1, 2, self.in_channels)
            actual_shape = tuple(x.shape[1:])  # Skip batch dimension
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Expected input shape [batch, {self.lmax + 1}, {self.mmax + 1}, 2, {self.in_channels}], "
                    f"got shape {list(x.shape)}"
                )

        # Apply edge modulation to INPUT (matching reference implementation)
        x = self._apply_edge_modulation(x, x_edge)

        # Apply convolution with standard (unmodulated) weights
        out = torch.einsum("blmrc,mRrco->blmRo", x, self.W_complex)

        # TODO: merge masks into a single op
        # Apply mask for invalid (l, m) positions where m > l
        mask: torch.Tensor = self.mask  # type: ignore[assignment]
        out = out * mask[None, :, :, None, None]

        # Zero out imaginary component for m=0 using multiplicative mask
        m0_imag_mask: torch.Tensor = self.m0_imag_mask  # type: ignore[assignment]
        out = out * m0_imag_mask

        # Apply gate mask if producing gates
        if self.produce_gates:
            gate_mask: torch.Tensor = self.gate_mask  # type: ignore[assignment]
            out = out * gate_mask

        return out

    def _apply_edge_modulation(
        self,
        x: Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"],
        x_edge: Float[torch.Tensor, "batch edge_channels"] | None,
    ) -> Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"]:
        """Apply per-edge, per-coefficient scaling to input features.

        Apply a scalar modulation factor for each input coefficient
        and rescale the inputs before applying the main linear transform.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, lmax+1, mmax+1, 2, in_channels].
        x_edge : torch.Tensor or None
            Edge features of shape [batch, edge_channels]. If None or if
            internal_weights=True, returns x unchanged.

        Returns
        -------
        torch.Tensor
            Modulated input tensor of same shape as x.
        """
        if self.internal_weights or x_edge is None:
            return x

        assert self.rad_func is not None

        # Get modulation factors from RadialMLP
        # Shape: [batch, (lmax+1) * (mmax+1) * 2 * in_channels]
        mod = self.rad_func(x_edge)

        # Reshape to match input layout
        # Shape: [batch, lmax+1, mmax+1, 2, in_channels]
        mod = mod.view(x.shape[0], self.lmax + 1, self.mmax + 1, 2, self.in_channels)

        # Element-wise multiplication (per-edge, per-coefficient scaling)
        return x * mod

    def extra_repr(self) -> str:
        """Return a string representation of the layer's parameters."""
        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"lmax={self.lmax}, mmax={self.mmax}"
        )
        if not self.internal_weights:
            s += ", edge_modulated=True"
        if self.produce_gates:
            s += f", produce_gates=True, gate_channels={self.num_gate_channels}"
        return s
