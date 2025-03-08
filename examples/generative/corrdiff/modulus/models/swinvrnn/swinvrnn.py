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

import torch
from torch import nn

from ..layers import (
    ConvBlock,
    CubeEmbedding,
    SwinTransformer,
)
from ..meta import ModelMetaData
from ..module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "SwinRNN"
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


class SwinRNN(Module):
    """
    Implementation of SwinRNN https://arxiv.org/abs/2205.13158
    Args:
        img_size (Sequence[int], optional): Image size [T, Lat, Lon].
        patch_size (Sequence[int], optional): Patch token size [T, Lat, Lon].
        in_chans (int, optional): number of input channels.
        out_chans (int, optional): number of output channels.
        embed_dim (int, optional): number of embed channels.
        num_groups (Sequence[int] | int, optional): number of groups to separate the channels into.
        num_heads (int, optional): Number of attention heads.
        window_size (int | tuple[int], optional): Local window size.
    """

    def __init__(
        self,
        img_size=(2, 721, 1440),
        patch_size=(2, 4, 4),
        in_chans=70,
        out_chans=70,
        embed_dim=1536,
        num_groups=32,
        num_heads=8,
        window_size=7,
    ):
        super().__init__(meta=MetaData())
        input_resolution = img_size[1:]
        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.swin_block1 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.down1 = ConvBlock(embed_dim, embed_dim, num_groups, upsample=-1)
        self.down1x = ConvBlock(in_chans, in_chans, in_chans, upsample=-1)
        self.lin_proj1 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder1 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.swin_block2 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.down2 = ConvBlock(embed_dim, embed_dim, num_groups, upsample=-1)
        self.down2x = ConvBlock(in_chans, in_chans, in_chans, upsample=-1)
        self.lin_proj2 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder2 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.swin_block3 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.down3 = ConvBlock(embed_dim, embed_dim, num_groups, upsample=-1)
        self.down3x = ConvBlock(in_chans, in_chans, in_chans, upsample=-1)
        self.lin_proj3 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder3 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.swin_block4 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.lin_proj4 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder4 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        self.up3x = ConvBlock(embed_dim, embed_dim, num_groups, upsample=1)
        self.up2x = ConvBlock(embed_dim * 2, embed_dim, num_groups, upsample=1)
        self.up1x = ConvBlock(embed_dim * 2, embed_dim, num_groups, upsample=1)
        self.pred = ConvBlock(embed_dim * 2, out_chans, out_chans, upsample=0)

        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_chans
        self.img_size = img_size
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        B, Cin, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        xT = x[:, :, -1, :, :]
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        h1 = self.swin_block1(x)
        x = self.down1(h1)
        h2 = self.swin_block2(x)
        x = self.down2(h2)
        h3 = self.swin_block3(x)
        x = self.down3(h3)
        h4 = self.swin_block4(x)
        B, Cin, H, W = xT.shape
        h1_d = torch.cat(
            [xT.reshape(B, Cin, -1), h1.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h1_d = self.lin_proj1(h1_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h1_d = self.swin_decoder1(h1_d)
        h1 = h1 + h1_d
        x2T = self.down1x(xT)
        B, Cin, H, W = x2T.shape
        h2_d = torch.cat(
            [x2T.reshape(B, Cin, -1), h2.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h2_d = self.lin_proj2(h2_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h2_d = self.swin_decoder2(h2_d)
        h2 = h2 + h2_d
        x3T = self.down2x(x2T)
        B, Cin, H, W = x3T.shape
        h3_d = torch.cat(
            [x3T.reshape(B, Cin, -1), h3.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h3_d = self.lin_proj3(h3_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h3_d = self.swin_decoder3(h3_d)
        h3 = h3 + h3_d
        x4T = self.down3x(x3T)
        B, Cin, H, W = x4T.shape
        h4_d = torch.cat(
            [x4T.reshape(B, Cin, -1), h4.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h4_d = self.lin_proj4(h4_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h4_d = self.swin_decoder4(h4_d)
        h4 = h4 + h4_d
        h4_up = self.up3x(h4)
        h3_up = self.up2x(torch.cat([h3, h4_up], dim=1))
        h2_up = self.up1x(torch.cat([h2, h3_up], dim=1))
        h1_up = self.pred(torch.cat([h1, h2_up], dim=1))
        x_h1 = xT + h1_up
        return x_h1
