# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from dataclasses import dataclass
from functools import partial
from typing import List, Literal, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import physicsnemo  # noqa: F401 for docs
import physicsnemo.models.layers.fft as fft
from physicsnemo.models.afno.afno import AFNO2DLayer, AFNOMlp, PatchEmbed
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

from .modembed import ModEmbedNet

Tensor = torch.Tensor


class ScaleShiftMlp(nn.Module):
    """MLP used to compute the scale and shift parameters of the ModAFNO block

    Parameters
    ----------
    in_features : int
        Input feature size
    out_features : int
        Output feature size
    hidden_features : int, optional
        Hidden feature size, defaults to 2 * out_features
    hidden_layers : int, optional
        Number of hidden layers, defaults to 0
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
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

        sequence = [nn.Linear(in_features, hidden_features), activation_fn()]
        for _ in range(hidden_layers):
            sequence += [nn.Linear(hidden_features, hidden_features), activation_fn()]
        sequence.append(nn.Linear(hidden_features, out_features * 2))
        self.net = nn.Sequential(*sequence)

    def forward(self, x: Tensor):
        (scale, shift) = torch.chunk(self.net(x), 2, dim=1)
        return (1 + scale, shift)


class ModAFNOMlp(AFNOMlp):
    """Modulated MLP used inside ModAFNO

    Parameters
    ----------
    in_features : int
        Input feature size
    latent_features : int
        Latent feature size
    out_features : int
        Output feature size
    activation_fn :  nn.Module, optional
        Activation function, by default nn.GELU
    drop : float, optional
        Drop out rate, by default 0.0
    scale_shift_kwargs : dict, optional
        Options to the MLP that computes the scale-shift parameters
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

    def forward(self, x: Tensor, mod_embed: Tensor) -> Tensor:
        (scale, shift) = self.scale_shift(mod_embed)

        scale_shift_shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[1],)
        scale = scale.view(*scale_shift_shape)
        shift = shift.view(*scale_shift_shape)

        x = self.fc1(x)
        x = x * scale + shift
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ModAFNO2DLayer(AFNO2DLayer):
    """AFNO spectral convolution layer

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality
    mod_features : int
        Number of modulation features
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    hidden_size_factor : int, optional
        Factor to increase spectral features by after weight multiplication, by default 1
    scale_shift_kwargs : dict, optional
        Options to the MLP that computes the scale-shift parameters
    scale_shift_mode: ["complex", "real"]
        If 'complex' (default), compute the scale-shift operation using complex
        operations. If 'real', use real operations.
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

    def forward(self, x: Tensor, mod_embed: Tensor) -> Tensor:
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape
        # Using ONNX friendly FFT functions
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

        o1_real[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(o1_re)

        o1_imag[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(o1_im)

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


class Block(nn.Module):
    """AFNO block, spectral convolution and MLP

    Parameters
    ----------
    embed_dim : int
        Embedded feature dimensionality
    mod_dim : int
        Modululation input dimensionality
    num_blocks : int, optional
        Number of blocks used in the block diagonal weight matrix, by default 8
    mlp_ratio : float, optional
        Ratio of MLP latent variable size to input feature size, by default 4.0
    drop : float, optional
        Drop out rate in MLP, by default 0.0
    activation_fn: nn.Module, optional
        Activation function used in MLP, by default nn.GELU
    norm_layer : nn.Module, optional
        Normalization function, by default nn.LayerNorm
    double_skip : bool, optional
        Residual, by default True
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1
    modulate_filter: bool, optional
        Whether to compute the modulation for the FFT filter
    modulate_mlp: bool, optional
        Whether to compute the modulation for the MLP
    scale_shift_mode: ["complex", "real"]
        If 'complex' (default), compute the scale-shift operation using complex
        operations. If 'real', use real operations.
    """

    def __init__(
        self,
        embed_dim: int,
        mod_dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        activation_fn: nn.Module = nn.GELU(),
        norm_layer: nn.Module = nn.LayerNorm,
        double_skip: bool = True,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        modulate_filter: bool = True,
        modulate_mlp: bool = True,
        scale_shift_mode: Literal["complex", "real"] = "real",
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        if modulate_filter:
            self.filter = ModAFNO2DLayer(
                embed_dim,
                mod_dim,
                num_blocks,
                sparsity_threshold,
                hard_thresholding_fraction,
                scale_shift_mode=scale_shift_mode,
            )
            self.apply_filter = lambda x, mod_embed: self.filter(x, mod_embed)
        else:
            self.filter = AFNO2DLayer(
                embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
            )
            self.apply_filter = lambda x, mod_embed: self.filter(x)

        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        if modulate_mlp:
            self.mlp = ModAFNOMlp(
                in_features=embed_dim,
                latent_features=mlp_latent_dim,
                out_features=embed_dim,
                mod_features=mod_dim,
                activation_fn=activation_fn,
                drop=drop,
            )
            self.apply_mlp = lambda x, mod_embed: self.mlp(x, mod_embed)
        else:
            self.mlp = AFNOMlp(
                in_features=embed_dim,
                latent_features=mlp_latent_dim,
                out_features=embed_dim,
                activation_fn=activation_fn,
                drop=drop,
            )
            self.apply_mlp = lambda x, mod_embed: self.mlp(x)
        self.double_skip = double_skip
        self.modulate_filter = modulate_filter
        self.modulate_mlp = modulate_mlp

    def forward(self, x: Tensor, mod_embed: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.apply_filter(x, mod_embed)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.apply_mlp(x, mod_embed)
        x = x + residual
        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "ModAFNO"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class ModAFNO(Module):
    """Modulated Adaptive Fourier neural operator (ModAFNO) model.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int, optional
        Number of input channels
    out_channels: int, optional
        Number of output channels
    embed_model: dict, optional
        Dictionary of arguments to pass to the `ModEmbedNet` embedding model
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    mod_dim : int
        Modululation input dimensionality
    modulate_filter: bool, optional
        Whether to compute the modulation for the FFT filter, by default True
    modulate_mlp: bool, optional
        Whether to compute the modulation for the MLP, by default True
    scale_shift_mode: ["complex", "real"]
        If 'complex' (default), compute the scale-shift operation using complex
        operations. If 'real', use real operations.
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1

    The default settings correspond to the implementation in the paper cited below.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.models.afno import ModAFNO
    >>> model = ModAFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32) #(N, C, H, W)
    >>> time = torch.full((32, 1), 0.5)
    >>> output = model(input, time)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    Note
    ----
    Reference: Leinonen et al. "Modulated Adaptive Fourier Neural Operators
    for Temporal Interpolation of Weather Forecasts." arXiv preprint arXiv:TODO (2024).
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int = 155,
        out_channels: int = 73,
        embed_model: Union[dict, None] = None,
        patch_size: List[int] = [2, 2],
        embed_dim: int = 512,
        mod_dim: int = 64,
        modulate_filter: bool = True,
        modulate_mlp: bool = True,
        scale_shift_mode: Literal["complex", "real"] = "complex",
        depth: int = 12,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        num_blocks: int = 1,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        super().__init__(meta=MetaData())
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        if not (
            inp_shape[0] % patch_size[0] == 0 and inp_shape[1] % patch_size[1] == 0
        ):
            raise ValueError(
                f"input shape {inp_shape} should be divisible by patch_size {patch_size}"
            )

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.modulate_filter = modulate_filter
        self.modulate_mlp = modulate_mlp
        self.scale_shift_mode = scale_shift_mode
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            inp_shape=inp_shape,
            in_channels=self.in_chans,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    mod_dim=mod_dim,
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    modulate_filter=modulate_filter,
                    modulate_mlp=modulate_mlp,
                    scale_shift_mode=scale_shift_mode,
                )
                for i in range(depth)
            ]
        )

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        self.mod_additive_proj = nn.Linear(mod_dim, embed_dim)
        if not (modulate_mlp or modulate_filter):
            self.mod_embed_net = nn.Identity()
        else:
            embed_model = {} if embed_model is None else embed_model
            self.mod_embed_net = ModEmbedNet(**embed_model)

    def _init_weights(self, m: nn.Module):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor, mod: Tensor) -> Tensor:
        """Forward pass of core ModAFNO"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        mod_embed = self.mod_embed_net(mod)
        mod_additive = self.mod_additive_proj(mod_embed).unsqueeze(dim=(1))
        x = x + mod_additive

        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x, mod_embed=mod_embed)

        return x

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        """The full ModAFNO model logic."""
        x = self.forward_features(x, mod)
        x = self.head(x)

        # Correct tensor shape back into [B, C, H, W]
        # [b h w (p1 p2 c_out)]
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        # [b h w p1 p2 c_out]
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        # [b c_out, h, p1, w, p2]
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])
        # [b c_out, (h*p1), (w*p2)]
        return out
