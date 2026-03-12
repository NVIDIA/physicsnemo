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

r"""Distributed Adaptive Fourier Neural Operator (AFNO) model.

This module provides distributed implementations of the AFNO architecture
for model-parallel training across multiple GPUs.
"""

import logging
from functools import partial
from typing import Tuple, Union

import torch
import torch.distributed as dist
import torch.fft
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.distributed.mappings import (
    copy_to_parallel_region,
    gather_from_parallel_region,
    scatter_to_parallel_region,
)
from physicsnemo.distributed.utils import compute_split_shapes
from physicsnemo.models.afno.distributed.layers import (
    DistributedAFNO2D,
    DistributedMLP,
    DistributedPatchEmbed,
    DropPath,
    _trunc_normal_,
)
from physicsnemo.nn.module.layer_norm import get_layer_norm_class

# LayerNorm class (TE or PyTorch) for default norm_layer and _init_weights
_LayerNormClass = get_layer_norm_class()

logger = logging.getLogger(__name__)


class DistributedBlock(physicsnemo.Module):
    r"""Distributed AFNO transformer block.

    This block combines distributed AFNO spectral convolution with distributed MLP
    layers for model-parallel training.

    Parameters
    ----------
    h : int
        Height of the feature map.
    w : int
        Width of the feature map.
    dim : int
        Feature dimensionality (embedding dimension).
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP hidden features to input features.
    drop : float, optional, default=0.0
        Dropout rate.
    drop_path : float, optional, default=0.0
        Stochastic depth rate.
    act_layer : nn.Module, optional, default=nn.GELU
        Activation layer class.
    norm_layer : callable, optional
        Normalization layer factory (e.g. ``partial(LayerNorm, eps=1e-6)``).
        Defaults to the same mechanism as :func:`~physicsnemo.nn.module.layer_norm.get_layer_norm_class`
        (Transformer Engine if available, else PyTorch).
    double_skip : bool, optional, default=True
        Whether to use double skip connections.
    num_blocks : int, optional, default=8
        Number of blocks in the block diagonal weight matrix.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold for soft shrinkage.
    hard_thresholding_fraction : float, optional, default=1.0
        Fraction of modes to keep in hard thresholding.
    input_is_matmul_parallel : bool, optional, default=False
        Whether input is already sharded across model parallel group.
    output_is_matmul_parallel : bool, optional, default=False
        Whether output should be sharded across model parallel group.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C, H, W)`.

    Examples
    --------
    Requires a distributed environment with model parallel group initialized.

    >>> import torch  # doctest: +SKIP
    >>> from physicsnemo.models.afno.distributed.afno import DistributedBlock  # doctest: +SKIP
    >>> from physicsnemo.distributed.manager import DistributedManager  # doctest: +SKIP
    >>> DistributedManager.initialize()  # doctest: +SKIP
    >>> block = DistributedBlock(h=4, w=4, dim=256, num_blocks=8)  # doctest: +SKIP
    >>> x = torch.randn(2, 256, 4, 4)  # doctest: +SKIP
    >>> out = block(x)  # doctest: +SKIP
    >>> out.shape  # doctest: +SKIP
    torch.Size([2, 256, 4, 4])
    """

    def __init__(
        self,
        h: int,
        w: int,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: Union[type, None] = None,
        double_skip: bool = True,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        input_is_matmul_parallel: bool = False,
        output_is_matmul_parallel: bool = False,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(_LayerNormClass, eps=1e-6)

        # model parallelism
        # matmul parallelism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # norm layer
        self.norm1 = norm_layer((h, w))

        # filter
        self.filter = DistributedAFNO2D(
            dim,
            num_blocks,
            sparsity_threshold,
            hard_thresholding_fraction,
            input_is_matmul_parallel=True,
            output_is_matmul_parallel=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm2 = norm_layer((h, w))

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DistributedMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            input_is_matmul_parallel=True,
            output_is_matmul_parallel=True,
        )
        self.double_skip = double_skip

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        r"""Forward pass of the distributed block."""
        # Scatter input across model parallel group if needed
        if not self.input_is_matmul_parallel:
            scatter_shapes = compute_split_shapes(
                x.shape[1], DistributedManager().group_size("model_parallel")
            )
            x = scatter_to_parallel_region(x, dim=1, group="model_parallel")

        # Apply spectral convolution with skip connection
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        # Apply MLP with skip connection
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual

        # Gather output if not model parallel
        if not self.output_is_matmul_parallel:
            x = gather_from_parallel_region(
                x, dim=1, shapes=scatter_shapes, group="model_parallel"
            )

        return x


class DistributedAFNONet(physicsnemo.Module):
    r"""Internal distributed AFNO network implementation.

    This class contains the core distributed AFNO architecture with patch embedding,
    transformer blocks, and output head.

    Parameters
    ----------
    inp_shape : Tuple[int, int], optional, default=(720, 1440)
        Input image dimensions as ``(height, width)``.
    patch_size : Tuple[int, int], optional, default=(16, 16)
        Patch size as ``(patch_height, patch_width)``.
    in_chans : int, optional, default=2
        Number of input channels.
    out_chans : int, optional, default=2
        Number of output channels.
    embed_dim : int, optional, default=768
        Embedding dimension.
    depth : int, optional, default=12
        Number of transformer blocks.
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP hidden features to embedding dimension.
    drop_rate : float, optional, default=0.0
        Dropout rate.
    drop_path_rate : float, optional, default=0.0
        Stochastic depth rate.
    num_blocks : int, optional, default=16
        Number of blocks in the block diagonal weight matrix.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold for soft shrinkage.
    hard_thresholding_fraction : float, optional, default=1.0
        Fraction of modes to keep in hard thresholding.
    input_is_matmul_parallel : bool, optional, default=False
        Whether input is already sharded across model parallel group.
    output_is_matmul_parallel : bool, optional, default=False
        Whether output should be sharded across model parallel group.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)`.

    Examples
    --------
    Requires a distributed environment with model parallel group initialized.

    >>> import torch  # doctest: +SKIP
    >>> from physicsnemo.models.afno.distributed.afno import DistributedAFNONet  # doctest: +SKIP
    >>> from physicsnemo.distributed.manager import DistributedManager  # doctest: +SKIP
    >>> DistributedManager.initialize()  # doctest: +SKIP
    >>> net = DistributedAFNONet(inp_shape=(64, 64), in_chans=2, out_chans=2, depth=2)  # doctest: +SKIP
    >>> x = torch.randn(2, 2, 64, 64)  # doctest: +SKIP
    >>> out = net(x)  # doctest: +SKIP
    >>> out.shape  # doctest: +SKIP
    torch.Size([2, 2, 64, 64])
    """

    def __init__(
        self,
        inp_shape: Tuple[int, int] = (720, 1440),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 2,
        out_chans: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        input_is_matmul_parallel: bool = False,
        output_is_matmul_parallel: bool = False,
    ):
        super().__init__()

        # comm sizes
        matmul_comm_size = DistributedManager().group_size("model_parallel")

        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        norm_layer = partial(_LayerNormClass, eps=1e-6)

        self.patch_embed = DistributedPatchEmbed(
            inp_shape=inp_shape,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            input_is_matmul_parallel=self.input_is_matmul_parallel,
            output_is_matmul_parallel=True,
        )
        num_patches = self.patch_embed.num_patches

        # original: x = B, H*W, C
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # new: x = B, C, H*W
        self.embed_dim_local = self.embed_dim // matmul_comm_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim_local, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]

        # add blocks
        blks = []
        for i in range(0, depth):
            input_is_matmul_parallel = True  # if i > 0 else False
            output_is_matmul_parallel = True if i < (depth - 1) else False
            blks.append(
                DistributedBlock(
                    h=self.h,
                    w=self.w,
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                    input_is_matmul_parallel=input_is_matmul_parallel,
                    output_is_matmul_parallel=output_is_matmul_parallel,
                )
            )
        self.blocks = nn.ModuleList(blks)

        # head
        if self.output_is_matmul_parallel:
            self.out_chans_local = (
                self.out_chans + matmul_comm_size - 1
            ) // matmul_comm_size
        else:
            self.out_chans_local = self.out_chans
        self.head = nn.Conv2d(
            self.embed_dim,
            self.out_chans_local * self.patch_size[0] * self.patch_size[1],
            1,
            bias=False,
        )
        self.synchronized_head = False

        # init weights
        _trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        r"""Initialize weights for linear, conv, and layer norm layers.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, _LayerNormClass):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def _no_weight_decay(self) -> set:
        r"""Return set of parameters that should not have weight decay.

        Returns
        -------
        set
            Set of parameter names to exclude from weight decay.
        """
        return {"pos_embed", "cls_token"}

    def _forward_features(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C H W"]:
        r"""Extract features through patch embedding and transformer blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C_{in}, H, W)`.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape :math:`(B, C_{embed}, H_{patch}, W_{patch})`.
        """
        B = x.shape[0]

        # Apply patch embedding and add positional embedding
        x = self.patch_embed(x)  # (B, C_local, num_patches)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Reshape to spatial format for transformer blocks
        x = x.reshape(B, self.embed_dim_local, self.h, self.w)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x: Float[Tensor, "B C_in H W"]) -> Float[Tensor, "B C_out H W"]:
        r"""Forward pass of the distributed AFNO network."""
        # Extract features through transformer blocks
        x = self._forward_features(x)

        # Handle distributed head computation
        if self.output_is_matmul_parallel:
            x = copy_to_parallel_region(x, group="model_parallel")
        else:
            if not self.synchronized_head:
                # Synchronize head parameters across model parallel group
                for param in self.head.parameters():
                    dist.broadcast(
                        param, 0, group=DistributedManager().group("model_parallel")
                    )
                self.synchronized_head = True

        # Apply output head
        x = self.head(x)  # (B, out_chans * patch_h * patch_w, h, w)

        # Unpatchify: rearrange patches back to image format
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(
            b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1])
        )

        return x


class DistributedAFNO(physicsnemo.Module):
    r"""Distributed Adaptive Fourier Neural Operator (AFNO) model.

    This model implements a distributed version of AFNO for model-parallel
    training across multiple GPUs. AFNO is designed for 2D images only.

    See :class:`~physicsnemo.models.afno.AFNO` for the non-distributed version.

    .. note::
        This model requires the model parallel group to be initialized via
        :class:`~physicsnemo.distributed.DistributedManager` before instantiation.
        Set the ``MODEL_PARALLEL_SIZE`` environment variable to configure the
        number of GPUs for model parallelism.

    Parameters
    ----------
    inp_shape : Tuple[int, int]
        Input image dimensions as ``(height, width)``.
    in_channels : int
        Number of input channels.
    out_channels : int, optional
        Number of output channels. Defaults to ``in_channels`` if not specified.
    patch_size : int, optional, default=16
        Size of image patches (applied to both height and width).
    embed_dim : int, optional, default=256
        Embedding dimension.
    depth : int, optional, default=4
        Number of AFNO transformer layers.
    num_blocks : int, optional, default=4
        Number of blocks in the frequency weight matrices.
    channel_parallel_inputs : bool, optional, default=False
        Whether inputs are already sharded along the channel dimension.
    channel_parallel_outputs : bool, optional, default=False
        Whether outputs should be sharded along the channel dimension.

    Forward
    -------
    in_vars : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` where :math:`B` is batch
        size, :math:`C_{in}` is the number of input channels, and :math:`H, W`
        are spatial dimensions matching ``inp_shape``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)` where :math:`C_{out}`
        is the number of output channels.

    Examples
    --------
    Requires a distributed environment with model parallel group initialized.

    >>> import torch  # doctest: +SKIP
    >>> from physicsnemo.models.afno.distributed import DistributedAFNO  # doctest: +SKIP
    >>> from physicsnemo.distributed.manager import DistributedManager  # doctest: +SKIP
    >>> DistributedManager.initialize()  # doctest: +SKIP
    >>> model = DistributedAFNO(inp_shape=(64, 64), in_channels=2)  # doctest: +SKIP
    >>> x = torch.randn(4, 2, 64, 64)  # doctest: +SKIP
    >>> output = model(x)  # doctest: +SKIP
    >>> output.shape  # doctest: +SKIP
    torch.Size([4, 2, 64, 64])

    See Also
    --------
    :class:`~physicsnemo.models.afno.AFNO` :
        Non-distributed AFNO model.
    `Adaptive Fourier Neural Operator (AFNO) <https://arxiv.org/abs/2111.13587>`_ :
        Original AFNO paper.
    """

    def __init__(
        self,
        inp_shape: Tuple[int, int],
        in_channels: int,
        out_channels: Union[int, None] = None,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_blocks: int = 4,
        channel_parallel_inputs: bool = False,
        channel_parallel_outputs: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        if DistributedManager().group("model_parallel") is None:
            raise RuntimeError(
                "Distributed AFNO needs to have model parallel group created first. "
                "Check the MODEL_PARALLEL_SIZE environment variable"
            )

        comm_size = DistributedManager().group_size("model_parallel")
        if channel_parallel_inputs:
            if not (in_channels % comm_size == 0):
                raise ValueError(
                    "Error, in_channels needs to be divisible by model_parallel size"
                )

        self.inp_shape = inp_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._impl = DistributedAFNONet(
            inp_shape=inp_shape,
            patch_size=(patch_size, patch_size),
            in_chans=in_channels,
            out_chans=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
            input_is_matmul_parallel=False,
            output_is_matmul_parallel=False,
        )

    def forward(
        self, in_vars: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H W"]:
        r"""Forward pass of the distributed AFNO model."""
        # Input validation: single check against expected shape (B, in_channels, H, W)
        if not torch.compiler.is_compiling():
            expected = (
                self.in_channels,
                self.inp_shape[0],
                self.inp_shape[1],
            )
            if (
                in_vars.ndim != 4
                or (
                    in_vars.shape[1],
                    in_vars.shape[2],
                    in_vars.shape[3],
                )
                != expected
            ):
                raise ValueError(
                    f"Expected input shape (B, {expected[0]}, {expected[1]}, {expected[2]}), "
                    f"got {tuple(in_vars.shape)}"
                )

        return self._impl(in_vars)
