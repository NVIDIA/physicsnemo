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

from dataclasses import dataclass
from functools import partial
from typing import List

import torch
import torch.nn as nn
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

# Import AFNO layers from physicsnemo.nn
from physicsnemo.nn import AFNO2DLayer, AFNOMlp, AFNOPatchEmbed

Tensor = torch.Tensor

# Backward compatibility alias
PatchEmbed = AFNOPatchEmbed


class Block(Module):
    r"""AFNO block consisting of spectral convolution and MLP.

    Parameters
    ----------
    embed_dim : int
        Embedded feature dimensionality.
    num_blocks : int, optional, default=8
        Number of blocks used in the block diagonal weight matrix.
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP latent variable size to input feature size.
    drop : float, optional, default=0.0
        Drop out rate in MLP.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function used in MLP.
    norm_layer : nn.Module, optional, default=nn.LayerNorm
        Normalization function.
    double_skip : bool, optional, default=True
        Whether to use double skip connections.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold (softshrink) of spectral features.
    hard_thresholding_fraction : float, optional, default=1.0
        Threshold for limiting number of modes used, in range ``[0, 1]``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, H, W, C)` where :math:`B` is batch size,
        :math:`H, W` are spatial dimensions, and :math:`C` is ``embed_dim``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, H, W, C)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.afno.afno import Block
    >>> block = Block(embed_dim=64, num_blocks=8)
    >>> x = torch.randn(2, 8, 8, 64)  # (B, H, W, C)
    >>> out = block(x)
    >>> out.shape
    torch.Size([2, 8, 8, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        activation_fn: nn.Module = nn.GELU(),
        norm_layer: nn.Module = nn.LayerNorm,
        double_skip: bool = True,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = AFNO2DLayer(
            embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        self.mlp = AFNOMlp(
            in_features=embed_dim,
            latent_features=mlp_latent_dim,
            out_features=embed_dim,
            activation_fn=activation_fn,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x: Float[Tensor, "B H W C"]) -> Float[Tensor, "B H W C"]:
        r"""Forward pass of the AFNO block."""
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


@dataclass
class MetaData(ModelMetaData):
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


class AFNO(Module):
    r"""Adaptive Fourier neural operator (AFNO) model.

    AFNO is a model that is designed for 2D images only. It combines patch
    embedding with spectral convolution blocks in the Fourier domain.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions as ``[height, width]``.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    patch_size : List[int], optional, default=[16, 16]
        Size of image patches as ``[patch_height, patch_width]``.
    embed_dim : int, optional, default=256
        Embedded channel size.
    depth : int, optional, default=4
        Number of AFNO layers.
    mlp_ratio : float, optional, default=4.0
        Ratio of layer MLP latent variable size to input feature size.
    drop_rate : float, optional, default=0.0
        Drop out rate in layer MLPs.
    num_blocks : int, optional, default=16
        Number of blocks in the block-diag frequency weight matrices.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold (softshrink) of spectral features.
    hard_thresholding_fraction : float, optional, default=1.0
        Threshold for limiting number of modes used, in range ``[0, 1]``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` where :math:`B` is batch
        size, :math:`C_{in}` is the number of input channels, and :math:`H, W` are
        spatial dimensions matching ``inp_shape``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)` where :math:`C_{out}` is
        ``out_channels``.

    Examples
    --------
    >>> import torch
    >>> import physicsnemo
    >>> model = physicsnemo.models.afno.AFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32)  # (N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    See Also
    --------
    :class:`~physicsnemo.models.afno.distributed.DistributedAFNO` :
        Distributed (model-parallel) AFNO for multi-GPU training.
    `Adaptive Fourier Neural Operator (AFNO) <https://arxiv.org/abs/2111.13587>`_ :
        Original AFNO paper.
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        out_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        num_blocks: int = 16,
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
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = AFNOPatchEmbed(
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
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
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

    def _init_weights(self, m: nn.Module) -> None:
        r"""Initialize model weights.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward_features(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B H W D"]:
        r"""Forward pass of core AFNO feature extraction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C_{in}, H, W)`.

        Returns
        -------
        torch.Tensor
            Features of shape :math:`(B, h, w, D)` where :math:`h, w` are patch
            grid dimensions and :math:`D` is ``embed_dim``.
        """
        B = x.shape[0]

        # Embed patches and add positional encoding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Reshape to 2D grid and apply blocks
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x: Float[Tensor, "B C_in H W"]) -> Float[Tensor, "B C_out H W"]:
        r"""Forward pass of the AFNO model."""
        # Input validation: single check against expected shape (B, in_chans, H, W)
        if not torch.compiler.is_compiling():
            expected = (
                self.in_chans,
                self.inp_shape[0],
                self.inp_shape[1],
            )
            if x.ndim != 4 or (x.shape[1], x.shape[2], x.shape[3]) != expected:
                raise ValueError(
                    f"Expected input shape (B, {expected[0]}, {expected[1]}, {expected[2]}), "
                    f"got {tuple(x.shape)}"
                )

        # Extract features through AFNO blocks
        x = self._forward_features(x)

        # Project to output channels
        x = self.head(x)

        # Reshape tensor back into [B, C, H, W]
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])

        return out
