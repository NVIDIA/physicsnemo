# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""SO(2) equivariant convolution using regular tensor layout with masking.

The expected use case of this layer is to perform an equivariant convolution
on some graph. The data layout is expected to be quite rigid: we use tensor
dimensions to encode order and degree in a way that allows for a straightforward
`einsum` and mask to be used to compute real/complex operations in a single
operation, as opposed to unrolling the loop to operate on specific +/-m pairs
per parity rules. See the ``forward`` pass documentation to see what the
expected shapes and dimensions are.


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

from physicsnemo.experimental.nn.symmetry.grid import make_grid_mask

__all__ = [
    "SO2Convolution",
]


class SO2Convolution(nn.Module):
    """SO(2) equivariant convolution using regular, padded tensor layout.

    This layer performs SO(2) equivariant convolution on spherical harmonic
    coefficients arranged in a regular grid layout. The grid layout uses
    fixed-size tensors with masking to handle invalid (l, m) positions
    where m > l.

    The key advantage is that all m-orders are processed simultaneously via
    a single vectorized einsum operation, eliminating Python loops and
    intermediate tensors in the forward pass.

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
        Edge feature channels for weight modulation. If None, uses internal
        (shared) weights without edge-dependent modulation. Default: None.
    extra_m0_output_channels : int, optional
        Additional output channels for m=0 only (used for gating scalars).
        Default: None (no extra channels).

    Attributes
    ----------
    W_r : nn.Parameter
        Real part of complex weights. Shape: ``[mmax+1, in_channels, out_channels]``.
    W_i : nn.Parameter
        Imaginary part of complex weights. Shape: ``[mmax+1, in_channels, out_channels]``.
    W_complex : torch.Tensor
        Combined weight tensor encoding complex multiplication structure.
        Shape: ``[mmax+1, 2, 2, in_channels, out_channels]``.
    mask : torch.Tensor
        Float mask of shape ``[lmax+1, mmax+1]`` where 1.0 indicates valid
        (l, m) positions (i.e., m <= l), 0.0 otherwise.

    Notes
    -----
    Input tensor layout: ``[batch, lmax+1, mmax+1, 2, in_channels]``
        - batch: number of edges/samples
        - lmax+1: degrees from l=0 to l=lmax
        - mmax+1: orders from m=0 to m=mmax
        - 2: real (index 0) and imaginary (index 1) components
        - in_channels: feature channels

    Output tensor layout: ``[batch, lmax+1, mmax+1, 2, out_channels]``

    For m=0, the imaginary component is always zero (by SO(2) symmetry).
    This is enforced via explicit zeroing after the forward pass.

    The complex multiplication is implemented using a combined weight tensor::

        | W_r  -W_i |   | x_r |   | out_r |
        | W_i   W_r | × | x_i | = | out_i |

    This allows a single einsum to compute: ``out = einsum('blmrc,mRrco->blmRo', x, W)``

    The weight initialization scales by ``1/sqrt(2)`` to maintain proper
    variance with the complex multiplication structure.

    Examples
    --------
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
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lmax: int,
        mmax: int,
        edge_channels: int | None = None,
        extra_m0_output_channels: int | None = None,
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
        self.extra_m0_output_channels = extra_m0_output_channels

        # Complex weights: W = W_r + i*W_i
        # Shape: [mmax+1, in_channels, out_channels]
        # Each m-order has its own weight matrix
        self.W_r = nn.Parameter(torch.empty(mmax + 1, in_channels, out_channels))
        self.W_i = nn.Parameter(torch.empty(mmax + 1, in_channels, out_channels))

        # Initialize weights with proper scaling for complex structure
        self._reset_parameters()

        # Create and register the validity mask for (l, m) positions
        # mask[l, m] = 1.0 if m <= l (valid position), 0.0 otherwise
        # Convert boolean mask to float for multiplication
        mask = make_grid_mask(lmax, mmax).float()
        self.register_buffer("mask", mask, persistent=False)

        # Mask to zero out m=0 imaginary component
        # Shape: [1, 1, mmax+1, 2, 1] for broadcasting
        m0_imag_mask = torch.ones(1, 1, mmax + 1, 2, 1)
        m0_imag_mask[:, :, 0, 1, :] = 0.0  # Zero out m=0 imaginary
        self.register_buffer("m0_imag_mask", m0_imag_mask, persistent=False)

        # Optional radial function for edge-dependent weight modulation
        self.rad_func: nn.Module | None = None
        if not self.internal_weights:
            assert edge_channels is not None
            # Number of weight parameters to modulate per edge
            # We modulate W_r and W_i separately, each of shape [mmax+1, in_ch, out_ch]
            # For simplicity, we output a scale factor per m-order
            num_modulation_params = (mmax + 1) * 2  # scale for W_r and W_i per m
            self.rad_func = nn.Sequential(
                nn.Linear(edge_channels, edge_channels),
                nn.SiLU(),
                nn.Linear(edge_channels, num_modulation_params),
            )

        # Optional extra output channels for m=0 (gating)
        self.fc_m0_extra: nn.Linear | None = None
        if extra_m0_output_channels is not None:
            # Linear layer from m=0 features to extra outputs
            # Input: [batch, lmax+1, in_channels] -> [batch, extra_m0_output_channels]
            m0_input_size = (lmax + 1) * in_channels
            self.fc_m0_extra = nn.Linear(m0_input_size, extra_m0_output_channels)

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

        Returns tensor of shape ``[mmax+1, 2, 2, in_channels, out_channels]``
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
            Combined weight tensor of shape ``[mmax+1, 2, 2, in_channels, out_channels]``.
        """
        W = torch.zeros(
            self.mmax + 1,
            2,
            2,
            self.in_channels,
            self.out_channels,
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
    ) -> (
        Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 out_channels"]
        | tuple[
            Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 out_channels"],
            Float[torch.Tensor, "batch extra_m0"],
        ]
    ):
        """Apply SO(2) convolution to grid-layout spherical harmonic coefficients.

        Parameters
        ----------
        x : Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"]
            Input tensor with shape ``[batch, lmax+1, mmax+1, 2, in_channels]``.
            The dimension of size 2 contains real (index 0) and imaginary
            (index 1) components.
        x_edge : Float[torch.Tensor, "batch edge_channels"], optional
            Edge features for weight modulation. Required if ``edge_channels``
            was specified during initialization. Default: None.

        Returns
        -------
        out : Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 out_channels"]
            Output tensor with same layout as input but ``out_channels`` features.
        extra_m0 : Float[torch.Tensor, "batch extra_m0"], optional
            Extra m=0 output channels (for gating). Only returned if
            ``extra_m0_output_channels`` was specified during initialization.

        Raises
        ------
        ValueError
            If ``x_edge`` is required but not provided.
        """
        # Validate inputs
        if not self.internal_weights and x_edge is None:
            raise ValueError(
                "x_edge is required when edge_channels was specified at init"
            )

        # Get combined complex weight tensor (possibly modulated by edge features)
        # W shape: [mmax+1, 2, 2, in_channels, out_channels]
        # where indices are [m, out_ri, in_ri, in_ch, out_ch]
        W = self._get_complex_weights(x_edge)

        # The einsum contracts over:
        # - r (in_ri): the real/imag input index
        # - c (in_channels): the input channel dimension
        # This implements the complex multiplication:
        # out_r = x_r*W_r - x_i*W_i  (via W[:, 0, 0]=W_r, W[:, 0, 1]=-W_i)
        # out_i = x_r*W_i + x_i*W_r  (via W[:, 1, 0]=W_i, W[:, 1, 1]=W_r)
        out = torch.einsum("blmrc,mRrco->blmRo", x, W)

        # Apply mask for invalid (l, m) positions where m > l
        # mask shape: [lmax+1, mmax+1] -> broadcast to [1, lmax+1, mmax+1, 1, 1]
        mask: torch.Tensor = self.mask  # type: ignore[assignment]
        out = out * mask[None, :, :, None, None]

        # Zero out imaginary component for m=0 using multiplicative mask
        m0_imag_mask: torch.Tensor = self.m0_imag_mask  # type: ignore[assignment]
        out = out * m0_imag_mask

        # Extract extra m=0 features if requested
        if self.extra_m0_output_channels is not None:
            extra_m0 = self._compute_extra_m0(x)
            return out, extra_m0
        else:
            return out

    def _get_complex_weights(
        self,
        x_edge: Float[torch.Tensor, "batch edge_channels"] | None,
    ) -> Float[torch.Tensor, "mmax_plus_1 2 2 in_channels out_channels"]:
        """Get combined complex weight tensor, optionally modulated by edge features.

        Parameters
        ----------
        x_edge : Float[torch.Tensor, "batch edge_channels"], optional
            Edge features for weight modulation.

        Returns
        -------
        torch.Tensor
            Combined weight tensor of shape ``[mmax+1, 2, 2, in_channels, out_channels]``
            encoding the complex multiplication structure.
        """
        if self.internal_weights or x_edge is None:
            return self.W_complex

        # Apply radial function to get modulation factors
        # mod shape: [batch, (mmax+1)*2]
        assert self.rad_func is not None
        mod = self.rad_func(x_edge)

        # Split into real and imaginary modulation factors
        # Each of shape [batch, mmax+1]
        mod_r, mod_i = mod.chunk(2, dim=-1)

        # For edge-dependent weights, we scale the base weights
        # This creates per-edge weights: W_r_edge[b, m, c, o] = W_r[m, c, o] * mod_r[b, m]
        # Note: For simplicity, we apply scalar modulation per m-order
        # More sophisticated approaches could learn full weight matrices per edge

        # Average modulation across batch for use with the combined einsum
        # FUSION_TARGET: edge_modulated_weights - this pattern needs custom kernel
        mod_r_avg = mod_r.mean(dim=0)  # [mmax+1]
        mod_i_avg = mod_i.mean(dim=0)  # [mmax+1]

        # Create modulated W_r and W_i
        # Reshape modulation for broadcasting: [mmax+1, 1, 1]
        mod_r_avg = mod_r_avg[:, None, None]
        mod_i_avg = mod_i_avg[:, None, None]

        W_r_mod = self.W_r * mod_r_avg
        W_i_mod = self.W_i * mod_i_avg

        # Build combined weight tensor with modulated weights
        W = torch.zeros(
            self.mmax + 1,
            2,
            2,
            self.in_channels,
            self.out_channels,
            dtype=self.W_r.dtype,
            device=self.W_r.device,
        )
        W[:, 0, 0] = W_r_mod  # real_out <- real_in
        W[:, 0, 1] = -W_i_mod  # real_out <- imag_in (negative)
        W[:, 1, 0] = W_i_mod  # imag_out <- real_in
        W[:, 1, 1] = W_r_mod  # imag_out <- imag_in
        return W

    def _compute_extra_m0(
        self,
        x: Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 in_channels"],
    ) -> Float[torch.Tensor, "batch extra_m0"]:
        """Compute extra m=0 output channels for gating.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Extra m=0 outputs of shape ``[batch, extra_m0_output_channels]``.
        """
        assert self.fc_m0_extra is not None

        # Extract m=0 real components (imaginary is always 0 for m=0)
        # Shape: [batch, lmax+1, in_channels]
        x_m0_real = x[:, :, 0, 0, :]

        # Flatten and apply linear layer
        batch_size = x.shape[0]
        x_m0_flat = x_m0_real.reshape(batch_size, -1)

        return self.fc_m0_extra(x_m0_flat)

    def extra_repr(self) -> str:
        """Return a string representation of the layer's parameters."""
        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"lmax={self.lmax}, mmax={self.mmax}"
        )
        if not self.internal_weights:
            s += ", edge_modulated=True"
        if self.extra_m0_output_channels is not None:
            s += f", extra_m0_output_channels={self.extra_m0_output_channels}"
        return s
