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

from typing import Tuple, Union, Optional, Any
import torch
import torch.nn as nn
import numpy as np
import math

from timm.models.vision_transformer import Attention

try:
    from transformer_engine.pytorch import MultiheadAttention
except ImportError:
    print("Transformer Engine not found, use 'timm' Attention.")

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ImportError:
    print("Apex not found, using torch LayerNorm.")
    from torch.nn import LayerNorm

from physicsnemo.models.utils import PatchEmbed2D, PatchRecovery2D
from physicsnemo.models.diffusion import PositionalEmbedding, Linear
from physicsnemo.models.layers import Mlp
from dataclasses import dataclass
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "DiT"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = False
    torch_fx: bool = False
    bf16: bool = False
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = False


class DiTBlock(nn.Module):
    """
    A Diffusion Transformer (DiT) block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_backbone: str = "timm",
        mlp_ratio: float = 4.0,
        **block_kwargs: Any,
    ):
        """
        Initializes the DiTBlock.

        Parameters
        -----------
        hidden_size (int): The dimensionality of the input and output.
        num_heads (int): The number of attention heads.
        attention_backbone (str): The attention implementation ('timm' or 'transformer_engine').
        mlp_ratio (float): The ratio for the MLP's hidden dimension.
        **block_kwargs (Any): Additional keyword arguments for the attention layer.
        """
        super().__init__()
        self.pre_attention_norm = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        if attention_backbone == "transformer_engine":
            self.attention = MultiheadAttention(
                hidden_size=hidden_size, num_attention_heads=num_heads, **block_kwargs
            )
        else:
            self.attention = Attention(
                dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
            )

        self.pre_mlp_norm = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.modulation = lambda x, scale, shift: x * (
            1 + scale.unsqueeze(1)
        ) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the DiTBlock.

        Forward
        -------
        x (torch.Tensor): Input tensor of shape (Batch, Sequence_Length, Hidden_Size).
        c (torch.Tensor): Conditioning tensor of shape (Batch, Hidden_Size).

        Outputs
        -------
        torch.Tensor: Output tensor of shape (Batch, Sequence_Length, Hidden_Size).
        """
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.adaptive_modulation(c).chunk(6, dim=1)

        # Attention block
        modulated_attention_input = self.modulation(
            self.pre_attention_norm(x), attention_scale, attention_shift
        )
        attention_output = self.attention(modulated_attention_input)
        x = x + attention_gate.unsqueeze(1) * attention_output

        # Feed-forward block
        modulated_mlp_input = self.modulation(
            self.pre_mlp_norm(x), mlp_scale, mlp_shift
        )
        mlp_output = self.linear(modulated_mlp_input)
        x = x + mlp_gate.unsqueeze(1) * mlp_output

        return x


class ProjLayer(nn.Module):
    """
    The penultimate layer of the DiT model, which projects the transformer output
    to a final embedding space.
    """

    def __init__(self, hidden_size: int, emb_channels: int):
        """
        Initializes the ProjLayer.

        Parameters
        -----------
        hidden_size (int): The dimensionality of the input from the transformer blocks.
        emb_channels (int): The number of embedding channels for final projection.
        """
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.output_projection = nn.Linear(hidden_size, emb_channels, bias=True)
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.modulation = lambda x, scale, shift: x * (
            1 + scale.unsqueeze(1)
        ) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the ProjLayer.

        Forward
        -------
        x (torch.Tensor): Input tensor of shape (Batch, Sequence_Length, Hidden_Size).
        c (torch.Tensor): Conditioning tensor of shape (Batch, Hidden_Size).

        Outputs
        -------
        torch.Tensor: Output tensor of shape (Batch, Sequence_Length, Embed_Size).
        """
        final_shift, final_scale = self.adaptive_modulation(c).chunk(2, dim=1)
        modulated_output = self.modulation(
            self.final_layer_norm(x), final_scale, final_shift
        )
        projected_output = self.output_projection(modulated_output)
        return projected_output


class DiT(Module):
    """
    The Diffusion Transformer (DiT) model.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = (32, 32),
        patch_size: Union[int, Tuple[int, int]] = (2, 2),
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_backbone: str = "transformer_engine",
        condition_dim: Optional[int] = None,
        pos_embedding_dim: int = 1,
    ):
        """
        Initializes the DiT model.

        Parameters
        -----------
        input_size (Union[int, Tuple[int, int]], optional): Height and width of the input images. Defaults to (32, 32).
        patch_size (Union[int, Tuple[int, int]], optional): The size of each image patch along height and width. Defaults to (2,2).
        in_channels (int, optional): The number of input channels. Defaults to 4.
        out_channels (Union[None, int], optional): The number of output channels. If None, it is `in_channels`. Defaults to None.
        hidden_size (int, optional): The dimensionality of the transformer embeddings. Defaults to 256.
        depth (int, optional): The number of transformer blocks. Defaults to 6.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
        mlp_ratio (float, optional): The ratio of the MLP hidden dimension to the embedding dimension. Defaults to 4.0.
        attention_backbone (str, optional): If 'timm' uses Attention from timm. If 'transformer_engine', uses MultiheadAttention from transformer_engine. Defaults to 'transformer_engine'.
        condition_dim (int, optional): Dimensionality of conditioning. If None, the model is unconditional. Defaults to None.
        embedding_type (str, optional): The type of positional embedding ('sin-cos' or 'learnable'). Defaults to 'sin-cos'.
        pos_embedding_dim (int, optional): The dimensionality of the positional embedding. Defaults to 1.

        Notes
        -----
        Reference: Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers.
        In Proceedings of the IEEE/CVF international conference on computer vision (pp. 4195-4205).

        Example
        --------
        >>> model = DiT(
        ...     input_size=(32,64),
        ...     patch_size=4,
        ...     in_channels=3,
        ...     out_channels=3,
        ...     condition_dim=8,
        ... )
        >>> x = torch.randn(2, 3, 32, 64)     # [B, C, H, W]
        >>> t = torch.randint(0, 1000, (2,))  # [B]
        >>> condition = torch.randn(2, 8)    # [B, d]
        >>> output = model(x, t, condition)
        >>> output.size()
        torch.Size([2, 3, 32, 64])
        """
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        if out_channels:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.num_heads = num_heads
        self.condition_dim = condition_dim

        self.x_embedder = PatchEmbed2D(
            input_size,
            self.patch_size,
            in_channels + pos_embedding_dim,
            hidden_size,
        )
        self.t_embedder = PositionalEmbedding(hidden_size)
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        self.cond_embedder = (
            Linear(
                in_features=condition_dim,
                out_features=hidden_size,
                bias=False,
                **init_zero,
            )
            if condition_dim
            else None
        )
        self.h_patches = input_size[0] // self.patch_size[0]
        self.w_patches = input_size[1] // self.patch_size[1]
        self.num_patches = self.h_patches * self.w_patches

        # Learnable positional embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(pos_embedding_dim, input_size[0], input_size[1]),
            requires_grad=True,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size, num_heads, attention_backbone, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = ProjLayer(
            hidden_size, patch_size[0] * patch_size[1] * self.out_channels
        )
        self.patch_recovery = PatchRecovery2D(
            input_size,
            self.patch_size,
            patch_size[0] * patch_size[1] * self.out_channels,
            self.out_channels,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass of the DiT model.

        Forward
        -------
        x (torch.Tensor): (N, C, H, W) tensor of spatial inputs.
        t (torch.Tensor): (N,) tensor of diffusion timesteps.
        condition (Optional[torch.Tensor]): (N, d) tensor of conditions.

        Outputs
        -------
        torch.Tensor: The output tensor of shape (N, out_channels, H, W).
        """
        b, ch, h, w = x.shape
        x = torch.cat([x, self.pos_embed.repeat(b, 1, 1, 1)], dim=1)
        x_emb = self.x_embedder(x)
        # (N, D, H//patch[0], W//patch[1])
        x = x_emb.flatten(2).transpose(1, 2)
        # (N, T, D) T = H//patch[0] * W//patch[1]
        t = self.t_embedder(t)  # (N, D)

        # Handle conditioning
        if self.cond_embedder is not None:
            if condition is None:
                # Fallback to using only timestep embedding if condition is not provided
                c = t
            else:
                condition_embedding = self.cond_embedder(condition)  # (N, D)
                c = t + condition_embedding  # (N, D)
        else:
            c = t  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, D')
        x = x.reshape(x.shape[0], x.shape[-1], self.h_patches, self.w_patches)
        # (N, D', H//patch[0], W//patch[1])
        x = self.patch_recovery(x)  # (N, out_channels, H, W)
        return x
