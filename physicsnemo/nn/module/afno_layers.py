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

r"""Adaptive Fourier Neural Operator (AFNO) layers.

This module contains reusable AFNO building blocks that can be used
in various AFNO-based architectures.
"""

from typing import List, Literal, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

import physicsnemo.nn.module.fft as fft
from physicsnemo.core.module import Module
from physicsnemo.nn.module.mlp_layers import Mlp

Tensor = torch.Tensor


class AFNOMlp(Module):
    r"""Fully-connected Multi-layer perception used inside AFNO.

    Parameters
    ----------
    in_features : int
        Input feature size.
    latent_features : int
        Latent feature size.
    out_features : int
        Output feature size.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function.
    drop : float, optional, default=0.0
        Drop out rate.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(*, D_{in})` where :math:`D_{in}` is
        ``in_features``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(*, D_{out})` where :math:`D_{out}` is
        ``out_features``.

    Examples
    --------
    >>> import torch
    >>> mlp = AFNOMlp(in_features=64, latent_features=128, out_features=64)
    >>> x = torch.randn(4, 32, 32, 64)
    >>> output = mlp(x)
    >>> output.shape
    torch.Size([4, 32, 32, 64])
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        activation_fn: nn.Module = nn.GELU(),
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, latent_features)
        self.act = activation_fn
        self.fc2 = nn.Linear(latent_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Float[Tensor, "*dims D_in"]) -> Float[Tensor, "*dims D_out"]:
        r"""Forward pass of the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2DLayer(Module):
    r"""AFNO spectral convolution layer.

    This layer performs spectral mixing using block-diagonal weight matrices
    in the Fourier domain with soft shrinkage for sparsity.

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality.
    num_blocks : int, optional, default=8
        Number of blocks used in the block diagonal weight matrix.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold (softshrink) of spectral features.
    hard_thresholding_fraction : float, optional, default=1
        Threshold for limiting number of modes used, in range ``[0, 1]``.
    hidden_size_factor : int, optional, default=1
        Factor to increase spectral features by after weight multiplication.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, H, W, C)` where :math:`B` is batch size,
        :math:`H, W` are spatial dimensions, and :math:`C` is ``hidden_size``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, H, W, C)`.

    Examples
    --------
    >>> import torch
    >>> layer = AFNO2DLayer(hidden_size=64, num_blocks=8)
    >>> x = torch.randn(4, 32, 32, 64)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([4, 32, 32, 64])
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        if not (hidden_size % num_blocks == 0):
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(self, x: Float[Tensor, "B H W C"]) -> Float[Tensor, "B H W C"]:
        r"""Forward pass of the AFNO spectral layer."""
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        # Apply 2D FFT in the spatial dimensions
        x = fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_real, x_imag = fft.real(x), fft.imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(
                torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_real[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[0],
                )
                - torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_imag[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[1],
                )
                + self.b1[0]
            )
        )

        o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(
                torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_imag[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[0],
                )
                + torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_real[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[1],
                )
                + self.b1[1]
            )
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 0
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 1
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = fft.view_as_complex(x)
        # TODO(akamenev): replace the following branching with
        # a one-liner, something like x.reshape(..., -1).squeeze(-1),
        # but this currently fails during ONNX export.
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        # Using ONNX friendly FFT functions
        x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class AFNOPatchEmbed(Module):
    r"""Patch embedding layer for AFNO.

    Converts 2D patches into a 1D vector sequence for input to AFNO.
    This differs from :class:`~physicsnemo.nn.module.utils.patch_embed.PatchEmbed2D`
    as it flattens the output to a sequence format.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions as ``[height, width]``.
    in_channels : int
        Number of input channels.
    patch_size : List[int], optional, default=[16, 16]
        Size of image patches as ``[patch_height, patch_width]``.
    embed_dim : int, optional, default=256
        Embedded channel size.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` where :math:`B` is batch
        size, :math:`C_{in}` is the number of input channels, and :math:`H, W` are
        spatial dimensions matching ``inp_shape``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, D)` where :math:`N` is the number of
        patches and :math:`D` is ``embed_dim``.

    Examples
    --------
    >>> import torch
    >>> patch_embed = AFNOPatchEmbed(
    ...     inp_shape=[32, 32], in_channels=3, patch_size=[8, 8], embed_dim=64
    ... )
    >>> x = torch.randn(4, 3, 32, 32)
    >>> output = patch_embed(x)
    >>> output.shape
    torch.Size([4, 16, 64])
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
    ):
        super().__init__()
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        num_patches = (inp_shape[1] // patch_size[1]) * (inp_shape[0] // patch_size[0])
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B N D"]:
        r"""Forward pass of patch embedding."""
        # Input validation: single check for shape (B, C, H, W)
        if not torch.compiler.is_compiling():
            expected_c = self.proj.in_channels
            expected_h, expected_w = self.inp_shape[0], self.inp_shape[1]
            if (
                x.ndim != 4
                or x.shape[1] != expected_c
                or x.shape[2] != expected_h
                or x.shape[3] != expected_w
            ):
                raise ValueError(
                    f"Expected input shape (B, {expected_c}, {expected_h}, {expected_w}), "
                    f"got {tuple(x.shape)}"
                )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# Alias for backward compatibility
PatchEmbed = AFNOPatchEmbed


class ScaleShiftMlp(Module):
    r"""MLP used to compute the scale and shift parameters of the ModAFNO block.

    Parameters
    ----------
    in_features : int
        Input feature size.
    out_features : int
        Output feature size.
    hidden_features : int, optional
        Hidden feature size. Defaults to ``2 * out_features``.
    hidden_layers : int, optional, default=0
        Number of hidden layers.
    activation_fn : Type[nn.Module], optional, default=nn.GELU
        Activation function class.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})`.

    Outputs
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of (scale, shift) tensors, each of shape :math:`(B, D_{out})`.
        Scale is offset by 1, i.e., ``(1 + scale, shift)``.

    Examples
    --------
    >>> import torch
    >>> mlp = ScaleShiftMlp(in_features=64, out_features=128)
    >>> x = torch.randn(4, 64)
    >>> scale, shift = mlp(x)
    >>> scale.shape, shift.shape
    (torch.Size([4, 128]), torch.Size([4, 128]))

    See Also
    --------
    :class:`~physicsnemo.nn.module.mlp_layers.Mlp` :
        The MLP used internally to produce the concatenated (scale, shift) vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Union[int, None] = None,
        hidden_layers: int = 0,
        activation_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if hidden_features is None:
            hidden_features = out_features * 2
        # Build hidden dims: one layer by default, plus hidden_layers extra
        hidden_dims = [hidden_features] * (hidden_layers + 1)
        self.net = Mlp(
            in_features=in_features,
            hidden_features=hidden_dims,
            out_features=out_features * 2,
            act_layer=activation_fn,
            drop=0.0,
            final_dropout=False,
        )

    def forward(
        self, x: Float[Tensor, "B D_in"]
    ) -> tuple[Float[Tensor, "B D_out"], Float[Tensor, "B D_out"]]:
        r"""Forward pass computing scale and shift parameters."""
        (scale, shift) = torch.chunk(self.net(x), 2, dim=1)
        return (1 + scale, shift)


class ModAFNOMlp(AFNOMlp):
    r"""Modulated MLP used inside ModAFNO.

    Extends :class:`AFNOMlp` with scale-shift modulation based on a conditioning
    embedding.

    Parameters
    ----------
    in_features : int
        Input feature size.
    latent_features : int
        Latent feature size.
    out_features : int
        Output feature size.
    mod_features : int
        Modulation embedding feature size.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function.
    drop : float, optional, default=0.0
        Drop out rate.
    scale_shift_kwargs : dict, optional
        Options to the MLP that computes the scale-shift parameters.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(*, D_{in})`.
    mod_embed : torch.Tensor
        Modulation embedding of shape :math:`(B, D_{mod})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(*, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> mlp = ModAFNOMlp(
    ...     in_features=64, latent_features=128, out_features=64, mod_features=32
    ... )
    >>> x = torch.randn(4, 16, 16, 64)
    >>> mod_embed = torch.randn(4, 32)
    >>> output = mlp(x, mod_embed)
    >>> output.shape
    torch.Size([4, 16, 16, 64])
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        mod_features: int,
        activation_fn: nn.Module = nn.GELU(),
        drop: float = 0.0,
        scale_shift_kwargs: Union[dict, None] = None,
    ):
        super().__init__(
            in_features=in_features,
            latent_features=latent_features,
            out_features=out_features,
            activation_fn=activation_fn,
            drop=drop,
        )
        if scale_shift_kwargs is None:
            scale_shift_kwargs = {}
        self.scale_shift = ScaleShiftMlp(
            mod_features, latent_features, **scale_shift_kwargs
        )

    def forward(
        self,
        x: Float[Tensor, "*dims D_in"],
        mod_embed: Float[Tensor, "B D_mod"],
    ) -> Float[Tensor, "*dims D_out"]:
        r"""Forward pass with modulation."""
        # Compute scale and shift from modulation embedding
        (scale, shift) = self.scale_shift(mod_embed)

        scale_shift_shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[1],)
        scale = scale.view(*scale_shift_shape)
        shift = shift.view(*scale_shift_shape)

        # Apply modulated MLP
        x = self.fc1(x)
        x = x * scale + shift
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ModAFNO2DLayer(AFNO2DLayer):
    r"""Modulated AFNO spectral convolution layer.

    Extends :class:`AFNO2DLayer` with scale-shift modulation in the spectral domain.

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality.
    mod_features : int
        Number of modulation features.
    num_blocks : int, optional, default=8
        Number of blocks used in the block diagonal weight matrix.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold (softshrink) of spectral features.
    hard_thresholding_fraction : float, optional, default=1
        Threshold for limiting number of modes used, in range ``[0, 1]``.
    hidden_size_factor : int, optional, default=1
        Factor to increase spectral features by after weight multiplication.
    scale_shift_kwargs : dict, optional
        Options to the MLP that computes the scale-shift parameters.
    scale_shift_mode : Literal["complex", "real"], optional, default="complex"
        If ``"complex"``, compute the scale-shift operation using complex
        operations. If ``"real"``, use real operations.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, H, W, C)`.
    mod_embed : torch.Tensor
        Modulation embedding of shape :math:`(B, D_{mod})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, H, W, C)`.

    Examples
    --------
    >>> import torch
    >>> layer = ModAFNO2DLayer(hidden_size=64, mod_features=32, num_blocks=8)
    >>> x = torch.randn(4, 16, 16, 64)
    >>> mod_embed = torch.randn(4, 32)
    >>> output = layer(x, mod_embed)
    >>> output.shape
    torch.Size([4, 16, 16, 64])
    """

    def __init__(
        self,
        hidden_size: int,
        mod_features: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1,
        hidden_size_factor: int = 1,
        scale_shift_kwargs: Union[dict, None] = None,
        scale_shift_mode: Literal["complex", "real"] = "complex",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            hidden_size_factor=hidden_size_factor,
        )

        if scale_shift_mode not in ("complex", "real"):
            raise ValueError("scale_shift_mode must be 'real' or 'complex'")
        self.scale_shift_mode = scale_shift_mode
        self.channel_mul = 1 if scale_shift_mode == "real" else 2
        if scale_shift_kwargs is None:
            scale_shift_kwargs = {}
        self.scale_shift = ScaleShiftMlp(
            mod_features,
            self.num_blocks
            * self.block_size
            * self.hidden_size_factor
            * self.channel_mul,
            **scale_shift_kwargs,
        )

    def forward(
        self,
        x: Float[Tensor, "B H W C"],
        mod_embed: Float[Tensor, "B D_mod"],
    ) -> Float[Tensor, "B H W C"]:
        r"""Forward pass with modulation."""
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        # Apply 2D FFT in the spatial dimensions
        x = fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_real, x_imag = fft.real(x), fft.imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        o1_shape = (
            B,
            H,
            W // 2 + 1,
            self.num_blocks,
            self.block_size * self.hidden_size_factor,
        )
        scale_shift_shape = (B, self.channel_mul, 1, o1_shape[3], o1_shape[4])

        o1_real = torch.zeros(o1_shape, device=x.device)
        o1_imag = torch.zeros(o1_shape, device=x.device)
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = min(H, W) // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_re = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                x_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                x_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_im = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                x_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                x_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w1[1],
            )
            + self.b1[1]
        )

        # scale-shift operation
        (scale, shift) = self.scale_shift(mod_embed)
        scale = scale.view(*scale_shift_shape)
        shift = shift.view(*scale_shift_shape)
        if self.scale_shift_mode == "real":
            o1_re = o1_re * scale + shift
            o1_im = o1_im * scale + shift
        elif self.scale_shift_mode == "complex":
            (scale_re, scale_im) = torch.chunk(scale, 2, dim=1)
            (shift_re, shift_im) = torch.chunk(shift, 2, dim=1)
            (o1_re, o1_im) = (
                o1_re * scale_re - o1_im * scale_im + shift_re,
                o1_im * scale_re + o1_re * scale_im + shift_im,
            )

        o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(o1_re)
        )

        o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(o1_im)
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 0
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 1
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = fft.view_as_complex(x)
        # TODO(akamenev): replace the following branching with
        # a one-liner, something like x.reshape(..., -1).squeeze(-1),
        # but this currently fails during ONNX export.
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        # Using ONNX friendly FFT functions
        x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias
