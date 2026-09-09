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


class ConvBlock(nn.Module):
    """Residual convolution block with two 3x3 convolutions and GroupNorm.

    All convolutions use kernel size 3, stride 1, padding 1, dilation 1,
    and groups 1. GroupNorm is channel-wise, so it is safe on spatially
    sharded data.

    Args:
        channels: Number of input and output channels
        conv_layer: Convolution class to use (nn.Conv2d or nn.Conv3d)
        num_groups: Number of groups for GroupNorm
    """

    def __init__(
        self,
        channels: int,
        conv_layer: type[nn.Module],
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.conv1 = conv_layer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.act1 = nn.GELU()
        self.conv2 = conv_layer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolution block.

        Args:
            x: Input tensor of shape (B, C, *spatial)

        Returns:
            Transformed tensor of shape (B, C, *spatial)
        """
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # Residual connection around the block, followed by a final activation
        x = self.act2(x + residual)
        return x


class HybridConvNet(nn.Module):
    """Residual convolutional network for large-image classification.

    All convolutions in the residual stages use an odd kernel (3), stride 1,
    padding 1, dilation 1, and groups 1 -- this is the configuration envelope
    supported by ShardTensor's halo-based sharded convolution (even kernels
    and strides have restrictions). The single patch-reduction convolution in
    the stem uses the supported even-kernel exception: kernel_size == stride
    with padding == 0.

    NOTE: THIS IS A TOY MODEL TO DEMONSTRATE DOMAIN PARALLELISM.

    Dimensionality (2D vs 3D) is selected from len(img_size).

    Args:
        img_size: Input image size
        in_channels: Number of input channels
        num_classes: Number of classes for classification
        base_channels: Number of channels in every residual stage
        depth: Number of residual blocks
    """

    def __init__(
        self,
        img_size: int = [512, 512],
        in_channels: int = 3,
        num_classes: int = 1000,
        base_channels: int = 192,
        depth: int = 16,
    ) -> None:
        super().__init__()

        # Use the image size to select the convolution dimensionality:
        if len(img_size) == 2:
            conv_layer = nn.Conv2d
        elif len(img_size) == 3:
            conv_layer = nn.Conv3d
        else:
            raise ValueError(f"img_size must have 2 or 3 dimensions, got {img_size}")

        # Single patch-reduction convolution.  An even kernel is allowed here
        # because kernel_size == stride and padding == 0, which is the
        # explicitly supported patch-reduction form for sharded convolutions.
        self.patch_embed = conv_layer(
            in_channels, base_channels, kernel_size=4, stride=4
        )

        # Build residual convolution stages (all operating on same resolution)
        self.stages = nn.ModuleList(
            [
                ConvBlock(channels=base_channels, conv_layer=conv_layer)
                for _ in range(depth)
            ]
        )

        # Classification head
        self.head = (
            nn.Linear(base_channels, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through all stages.

        Args:
            x: Input tensor of shape (B, C, *spatial)

        Returns:
            Pooled features of shape (B, base_channels)
        """
        # Patch-reduction stem
        x = self.patch_embed(x)  # B, base_channels, *spatial / 4

        # Apply residual convolution stages
        for stage in self.stages:
            x = stage(x)

        # Return the mean over all spatial dimensions
        return x.mean(dim=tuple(range(2, x.ndim)))

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
