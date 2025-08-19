# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

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


from typing import Tuple, Union
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention
from transformer_engine.pytorch import MultiheadAttention

from physicsnemo.models.layers import Mlp
from dataclasses import dataclass
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


# ------------------------------
#      Metadata Definition
# ------------------------------


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


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(
    embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Conditions                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds Condition vectors into latent vector representations.
    """

    def __init__(self, condition_dim: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.condition_dim = condition_dim
        self.out_features = hidden_size

        self.projection = nn.Linear(condition_dim, hidden_size, bias=True)

        # If using dropout for classifier-free guidance, create a learnable unconditional embedding
        if self.dropout_prob > 0:
            self.unconditional_embedding = nn.Parameter(torch.randn(1, hidden_size))

    def forward(
        self, condition: torch.Tensor, train: bool, force_drop: bool = False
    ) -> torch.Tensor:
        """
        Args:
            condition (torch.Tensor): A (N, condition_dim) tensor of latent vectors.
            train (bool): Whether the model is in training mode.
            force_drop (bool): Whether to force dropping the conditioning.
        """
        # Project the latent vectors to the hidden size
        embeddings = self.projection(condition)

        # Apply dropout for classifier-free guidance
        if self.dropout_prob > 0 and (train or force_drop):
            mask = (
                torch.rand(condition.shape[0], device=condition.device)
                < self.dropout_prob
            )
            if force_drop:
                mask = torch.ones_like(mask)
            uncond_embeddings = self.unconditional_embedding.expand(
                embeddings.shape[0], -1
            )
            embeddings = torch.where(mask[:, None], uncond_embeddings, embeddings)

        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self, hidden_size, num_heads, attention_backbone, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attention_backbone == "transformer_engine":
            self.attn = MultiheadAttention(
                hidden_size=hidden_size, num_attention_heads=num_heads, **block_kwargs
            )
        else:
            self.attn = Attention(
                dim=hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
            )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(Module):
    """
    Diffusion Transformer (DiT) model.

    Parameters
    ----------
    input_size : Union[int, Tuple[int, int]], optional
        Height and width of the input images, by default (32, 32).
    patch_size : int, optional
        The size of each image patch, by default 2.
    in_channels : int, optional
        The number of input channels, by default 4.
    out_channels : Union[None, int], optional
        The number of output channels. If None, it is in_channels, by default None.
    hidden_size : int, optional
        The dimensionality of the transformer embeddings, by default 1152.
    depth : int, optional
        The number of transformer blocks, by default 28.
    num_heads : int, optional
        The number of attention heads, by default 16.
    mlp_ratio : float, optional
        The ratio of the MLP hidden dimension to the embedding dimension, by default 4.0.
    attention_backbone : str, optional
        If 'transformer_engine', uses MultiheadAttention from transformer_engine, by default 'timm'.
    condition_dropout_prob : float, optional
        The dropout probability for classifier-free guidance, by default 0.1.
    condition_dim : int, optional
        Dimensionality of conditioning. If None, the model is unconditional, by default None.
    embedding_type : str, optional
        The type of positional embedding to use. 'sin-cos' for fixed sinusoidal embeddings,
        'learnable' for learnable positional embeddings, by default 'sin-cos'.

    Example
    -------
    >>> model = physicsnemo.experimental.models.dit.DiT(
    ... input_size=(32,64),
    ... patch_size=4,
    ... in_channels=3,
    ... out_channels=3,
    ... condition_dim=8,
    ... )
    >>> x = torch.randn(2, 3, 32, 64)     # [B, C, H, W]
    >>> t = torch.randint(0, 1000, (2,))  # [B]
    >>> condition = torch.randin(2, 8)    # [B, d]
    >>> output = model(x, t, condition)
    >>> output.size()
    torch.Size([2, 3, 32, 64])

    Note
    ----
    Reference: Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers.
    In Proceedings of the IEEE/CVF international conference on computer vision (pp. 4195-4205).
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = (32, 32),
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: Union[None, int] = None,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attention_backbone: str = "timm",
        condition_dropout_prob=0.1,
        condition_dim=None,
        embedding_type="sin-cos",
    ):
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        if out_channels:
            self.out_channels = out_channels
        else:
            self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.condition_dim = condition_dim
        self.embedding_type = embedding_type

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.condition_dim is not None:
            self.cond_embedder = ConditionEmbedder(
                condition_dim, hidden_size, condition_dropout_prob
            )
        else:
            self.cond_embedder = None
        num_patches = self.x_embedder.num_patches

        # Will use either fixed sin-cos embedding or learnable positional embedding:
        if self.embedding_type == "learnable":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, hidden_size), requires_grad=True
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, hidden_size), requires_grad=False
            )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size, num_heads, attention_backbone, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize pos_embed by either sin-cos or normal init:
        if self.embedding_type == "learnable":
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.x_embedder.grid_size[0],
                self.x_embedder.grid_size[1],
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize Condition embedding table if it exists:
        if self.cond_embedder is not None:
            nn.init.normal_(self.cond_embedder.projection.weight, std=0.02)
            nn.init.constant_(self.cond_embedder.projection.bias, 0)
            if hasattr(self.cond_embedder, "unconditional_embedding"):
                nn.init.normal_(self.cond_embedder.unconditional_embedding, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h, w = self.x_embedder.grid_size
        assert (
            h * w == x.shape[1]
        ), "The number of patches does not match the grid size."

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, condition=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        condition: (N, d) tensor of conditions, optional. If None, the model is unconditional.
        """
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)

        # Handle conditioning
        if self.cond_embedder is not None:
            if condition is None:
                c = t
            else:
                condition_embedding = self.cond_embedder(
                    condition, self.training
                )  # (N, D)
                c = t + condition_embedding  # (N, D)
        else:
            c = t  # (N, D)

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    in_channels = 4

    test_configs = [
        {
            "name": "Base: Unconditional, 32x64, sin-cos pos_embed",
            "params": {
                "input_size": (32, 64),
                "condition_dim": None,
                "embedding_type": "sin-cos",
                "attention_backbone": False,
            },
        },
        {
            "name": "Conditional, 32x64, learnable pos_embed",
            "params": {
                "input_size": (32, 64),
                "condition_dim": 10,
                "embedding_type": "positional",
                "attention_backbone": False,
            },
        },
        {
            "name": "Conditional, 32x64, TransformerEngine Attention",
            "params": {
                "input_size": (32, 64),
                "condition_dim": 10,
                "embedding_type": "sin-cos",
                "attention_backbone": True,
            },
        },
        {
            "name": "Unconditional, 64x64, specified out_channels",
            "params": {
                "input_size": (64, 64),
                "out_channels": 8,
                "condition_dim": None,
                "embedding_type": "sin-cos",
                "attention_backbone": False,
            },
        },
        {
            "name": "Conditional, 64x64, no sigma learning",
            "params": {
                "input_size": (64, 64),
                "condition_dim": 10,
                "embedding_type": "sin-cos",
                "attention_backbone": False,
            },
        },
    ]

    for config in test_configs:
        print(f"--- Testing: {config['name']} ---")
        params = config["params"]
        model = DiT(in_channels=in_channels, **params).to(device)

        H, W = (
            params["input_size"]
            if isinstance(params["input_size"], tuple)
            else (params["input_size"], params["input_size"])
        )
        x = torch.randn(batch_size, in_channels, H, W).to(device)
        t = torch.randint(0, 1000, (batch_size,), device=device)

        condition = None
        if params.get("condition_dim") is not None:
            condition = torch.randn(batch_size, params["condition_dim"]).to(device)

        output = model(x, t, condition)

        # Determine expected output channels
        if params.get("out_channels"):
            expected_out_channels = params["out_channels"]
        else:
            expected_out_channels = in_channels

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, expected_out_channels, H, W)
        print("Test PASSED.\n")
