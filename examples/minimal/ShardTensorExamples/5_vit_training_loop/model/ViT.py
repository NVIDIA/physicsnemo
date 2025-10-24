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

import torch
import torch.nn as nn

from .PatchEmbed2d import PatchEmbedding2d
from .PatchEmbed3d import PatchEmbedding3d
from .TransformerBlock import TransformerBlock


class HybridViT(nn.Module):
    """
    Hybrid Vision Transformer with conv patch embedding and multiple transformer layers.

    Args:
        img_size: Input image size
        patch_size: Size of patches for tokenization
        in_channels: Number of input channels
        num_classes: Number of classes for classification
        embed_dim: Embedding dimension (same for all layers)
        num_heads: Number of attention heads for each stage
        depth: Number of transformer layers
        mlp_ratio: MLP ratios for each layer
        qkv_bias: Whether to use bias in QKV projections
    """

    def __init__(
        self,
        img_size: int = [256, 256],
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 6,
        depth: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        # Use the image size to select the padding:
        if len(img_size) == 2:
            self.patch_embed = PatchEmbedding2d(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
            )
        elif len(img_size) == 3:
            self.patch_embed = PatchEmbedding3d(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
            )

        # Positional embeddings (for patches + CLS token)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )

        # Build transformer stages (all operating on same resolution)
        self.stages = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
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
            x: Input tensor of shape (B, C, H, W)

        Returns:
            CLS token features of shape (B, embed_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # B, N, C

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply transformer stages
        for stage in self.stages:
            x = stage(x)

        # Return the mean of all tokens
        return x.mean(dim=(1,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass for classification.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Classification logits of shape (B, num_classes)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x
