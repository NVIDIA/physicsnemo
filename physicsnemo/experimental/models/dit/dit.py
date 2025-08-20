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

from typing import Tuple, Union, Optional, Literal
import torch
import torch.nn as nn

from physicsnemo.models.utils import PatchEmbed2D, PatchRecovery2D
from physicsnemo.models.diffusion import PositionalEmbedding, Linear
from dataclasses import dataclass
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
from physicsnemo.experimental.models.dit import DiTBlock, ProjLayer

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


class DiT(Module):
    """
    Warning
    -----------
    This model is experimental and there may be changes in the future.
    
    The Diffusion Transformer (DiT) model.

    Parameters
    -----------
    input_size (Union[int, Tuple[int, int]]):
        Height and width of the input images.
    in_channels (int):
        The number of input channels..
    patch_size (Union[int, Tuple[int, int]], optional):
        The size of each image patch along height and width. Defaults to (8,8).
    out_channels (Union[None, int], optional):
        The number of output channels. If None, it is `in_channels`. Defaults to None,
        which means the output will have the same number of channels as the input.
    hidden_size (int, optional):
        The dimensionality of the transformer embeddings. Defaults to 384.
    depth (int, optional):
        The number of transformer blocks. Defaults to 12.
    num_heads (int, optional):
        The number of attention heads. Defaults to 8.
    mlp_ratio (float, optional):
        The ratio of the MLP hidden dimension to the embedding dimension. Defaults to 4.0.
    attention_backbone (str, optional):
        If 'timm' uses Attention from timm. If 'transformer_engine', uses MultiheadAttention from transformer_engine. Defaults to 'transformer_engine'.
    layernorm_backbone (str, optional):
        If 'apex', uses FusedLayerNorm from apex. If 'torch', uses LayerNorm from torch.nn. Defaults to 'apex'.
    condition_dim (int, optional):
        Dimensionality of conditioning. If None, the model is unconditional. Defaults to None.
    embedding_type (str, optional):
        The type of positional embedding ('sin-cos' or 'learnable'). Defaults to 'sin-cos'.
    pos_embedding_dim (int, optional):
        The dimensionality of the positional embedding. Defaults to 1.

    Forward
    -------
    x (torch.Tensor):
        (N, C, H, W) tensor of spatial inputs.
    t (torch.Tensor):
        (N,) tensor of diffusion timesteps.
    condition (Optional[torch.Tensor]):
        (N, d) tensor of conditions.

    Outputs
    -------
    torch.Tensor: The output tensor of shape (N, out_channels, H, W).
    
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

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        in_channels: int,
        patch_size: Union[int, Tuple[int, int]] = (8, 8),
        out_channels: Optional[int] = None,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_backbone: Literal["timm", "transformer_engine"] = "transformer_engine",
        layernorm_backbone: Literal["apex", "torch"] = "torch",
        condition_dim: Optional[int] = None,
        pos_embedding_dim: int = 1,
    ):
        """
        Initializes the DiT model.
        """
        super().__init__(meta=MetaData())
        self.input_size = input_size if isinstance(input_size, (tuple, list)) else (input_size, input_size)
        self.in_channels = in_channels
        if out_channels:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.num_heads = num_heads
        self.condition_dim = condition_dim

        self.x_embedder = PatchEmbed2D(
            self.input_size,
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
        self.h_patches = self.input_size[0] // self.patch_size[0]
        self.w_patches = self.input_size[1] // self.patch_size[1]
        self.num_patches = self.h_patches * self.w_patches

        # Learnable positional embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(pos_embedding_dim, self.input_size[0], self.input_size[1]),
            requires_grad=True,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    attention_backbone,
                    layernorm_backbone,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.proj_layer = ProjLayer(
            hidden_size,
            self.patch_size[0] * self.patch_size[1] * self.out_channels,
            layernorm_backbone,
        )
        self.patch_recovery = PatchRecovery2D(
            self.input_size,
            self.patch_size,
            self.patch_size[0] * self.patch_size[1] * self.out_channels,
            self.out_channels,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass of the DiT model.
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
            if condition is not None:
                raise ValueError("Conditioning is provided but condition_dim is None.")
            c = t  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.proj_layer(x, c)  # (N, T, D')
        x = x.reshape(x.shape[0], x.shape[-1], self.h_patches, self.w_patches)
        # (N, D', H//patch[0], W//patch[1])
        x = self.patch_recovery(x)  # (N, out_channels, H, W)
        return x
