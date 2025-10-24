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
from torch import nn


class CubeEmbedding(nn.Module):
    """
    3D Image Cube Embedding
    Args:
        img_size (tuple[int]): Image size [T, Lat, Lon].
        patch_size (tuple[int]): Patch token size [T, Lat, Lon].
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: torch.nn.LayerNorm
    """

    def __init__(
        self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class ConvBlock(nn.Module):
    """
    Conv2d block
    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        num_groups (int): Number of groups to separate the channels into for group normalization.
        num_residuals (int, optinal): Number of Conv2d operator. Default: 2
        upsample (int, optinal): 1: Upsample, 0: Conv, -1: Downsample. Default: 0
    """

    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2, upsample=0):
        super().__init__()
        if upsample == 1:
            self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif upsample == -1:
            self.conv = nn.Conv2d(
                in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1
            )
        elif upsample == 0:
            self.conv = nn.Conv2d(
                in_chans, out_chans, kernel_size=(3, 3), stride=1, padding=1
            )

        blk = []
        for i in range(num_residuals):
            blk.append(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
            )
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)
        x_skip = x
        x = self.b(x)
        return x + x_skip
