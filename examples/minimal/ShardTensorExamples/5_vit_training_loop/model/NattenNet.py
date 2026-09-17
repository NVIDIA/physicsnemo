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

import torch
import torch.nn as nn
from einops import rearrange

from physicsnemo.nn.functional.natten import na2d, na3d

from .MLP import MLP


class NeighborhoodAttention(nn.Module):
    """Multi-head neighborhood attention over channels-last spatial inputs.

    Each token attends to a local window of size ``kernel_size`` along every
    spatial axis. Constraints: only odd kernel sizes are supported, and
    dilation is fixed at 1 (the configuration supported by ShardTensor's
    halo-based sharded neighborhood attention). Under domain parallelism,
    each rank's local patch-grid extent along the sharded axis must be
    >= kernel_size.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        kernel_size: Size of the neighborhood attention window (odd)
        qkv_bias: Whether to use bias in QKV projections
    """

    def __init__(
        self, dim: int, num_heads: int, kernel_size: int, qkv_bias: bool = True
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head neighborhood self-attention.

        Args:
            x: Input tensor of shape (B, *spatial, C)

        Returns:
            Attention output of shape (B, *spatial, C)
        """
        spatial_shape = x.shape[1:-1]
        C = x.shape[-1]

        # Project to Q, K, V, keeping the spatial structure intact
        qkv = self.qkv(x).reshape(*x.shape[:-1], 3, self.num_heads, self.head_dim)
        # Each of q, k, v: B, *spatial, num_heads, head_dim
        q, k, v = qkv.unbind(dim=-3)

        # Select the 2D or 3D neighborhood attention from the spatial rank
        if q.ndim == 5:
            x = na2d(q, k, v, kernel_size=self.kernel_size, dilation=1)
        elif q.ndim == 6:
            x = na3d(q, k, v, kernel_size=self.kernel_size, dilation=1)
        else:
            raise ValueError(f"Expected 2 or 3 spatial dims, got {len(spatial_shape)}")

        # Merge heads and project back
        x = x.reshape(*x.shape[:-2], C)
        x = self.proj(x)

        return x


class NattenBlock(nn.Module):
    """Transformer block with neighborhood attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, num_heads=num_heads, kernel_size=kernel_size, qkv_bias=qkv_bias
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # The MLP is linear over the last dim, so it works on (B, *spatial, C)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply neighborhood attention block with residual connections.

        Args:
            x: Input tensor of shape (B, *spatial, C)

        Returns:
            Transformed tensor of shape (B, *spatial, C)
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class HybridNattenViT(nn.Module):
    """Hybrid Vision Transformer with conv patch embedding and neighborhood attention.

    Unlike the global-attention ViT, tokens keep their spatial layout
    (B, *spatial_patches, embed_dim) throughout the blocks, since neighborhood
    attention needs the spatial structure. Dimensionality (2D vs 3D) is
    selected from len(img_size).

    Args:
        img_size: Input image size
        patch_size: Size of patches for tokenization
        in_channels: Number of input channels
        num_classes: Number of classes for classification
        embed_dim: Embedding dimension (same for all layers)
        num_heads: Number of attention heads for each stage
        depth: Number of neighborhood attention layers
        mlp_ratio: MLP ratios for each layer
        kernel_size: Size of the neighborhood attention window (odd)
    """

    def __init__(
        self,
        img_size: int = [512, 512],
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 6,
        depth: int = 16,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        for i in img_size:
            assert i % patch_size == 0, (
                f"Image size {i} must be divisible by patch size {patch_size}"
            )

        # Use the image size to select the convolution dimensionality:
        if len(img_size) == 2:
            conv_layer = nn.Conv2d
        elif len(img_size) == 3:
            conv_layer = nn.Conv3d
        else:
            raise ValueError(f"img_size must have 2 or 3 dimensions, got {img_size}")

        # Single convolution that acts as both tokenizer and linear embedding
        self.patch_embed = conv_layer(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings over the full patch grid
        patch_grid = tuple(i // patch_size for i in img_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, *patch_grid, embed_dim))

        # Build neighborhood attention stages (all operating on same resolution)
        self.stages = nn.ModuleList(
            [
                NattenBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        # Classification head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through all stages.

        Args:
            x: Input tensor of shape (B, C, *spatial)

        Returns:
            Pooled features of shape (B, embed_dim)
        """
        # Patch embedding, then move channels last to keep the spatial layout
        x = self.patch_embed(x)
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
        else:
            x = rearrange(x, "b c d h w -> b d h w c")

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply neighborhood attention stages
        for stage in self.stages:
            x = stage(x)

        # Return the mean over all spatial dimensions
        return x.mean(dim=tuple(range(1, x.ndim - 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass for classification.

        Args:
            x: Input tensor of shape (B, C, *spatial)

        Returns:
            Classification logits of shape (B, num_classes)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x
