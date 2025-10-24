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

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..layers import (
    DecoderLayer,
    EncoderLayer,
    FuserLayer,
)
from ..meta import ModelMetaData
from ..module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "Fengwu"
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


class Fengwu(Module):
    """
    FengWu PyTorch impl of: `FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead`
    - https://arxiv.org/pdf/2304.02948.pdf

    Args:
        img_size: Image size(Lat, Lon). Default: (721,1440)
        pressure_level: Number of pressure_level. Default: 37
        embed_dim (int): Patch embedding dimension. Default: 192
        patch_size (tuple[int]): Patch token size. Default: (4,4)
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        pressure_level=37,
        embed_dim=192,
        patch_size=(4, 4),
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ):
        super().__init__(meta=MetaData())
        drop_path = np.linspace(0, 0.2, 8).tolist()
        drop_path_fuser = [0.2] * 6
        resolution_down1 = (
            math.ceil(img_size[0] / patch_size[0]),
            math.ceil(img_size[1] / patch_size[1]),
        )
        resolution_down2 = (
            math.ceil(resolution_down1[0] / 2),
            math.ceil(resolution_down1[1] / 2),
        )
        resolution = (resolution_down1, resolution_down2)
        self.encoder_surface = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=4,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_z = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_r = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_u = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_v = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_t = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.fuser = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=(6, resolution[1][0], resolution[1][1]),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path_fuser,
        )

        self.decoder_surface = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=4,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_z = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_r = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_u = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_v = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_t = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

    def prepare_input(self, surface, z, r, u, v, t):
        """Prepares the input to the model in the required shape.
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            z (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            r (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            u (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            v (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            t (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
        """
        return torch.concat([surface, z, r, u, v, t], dim=1)

    def forward(self, x):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            z (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            r (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            u (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            v (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            t (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
        """
        surface = x[:, :4, :, :]
        z = x[:, 4:41, :, :]
        r = x[:, 41:78, :, :]
        u = x[:, 78:115, :, :]
        v = x[:, 115:152, :, :]
        t = x[:, 152:189, :, :]
        surface, skip_surface = self.encoder_surface(surface)
        z, skip_z = self.encoder_z(z)
        r, skip_r = self.encoder_r(r)
        u, skip_u = self.encoder_u(u)
        v, skip_v = self.encoder_v(v)
        t, skip_t = self.encoder_t(t)

        x = torch.concat(
            [
                surface.unsqueeze(1),
                z.unsqueeze(1),
                r.unsqueeze(1),
                u.unsqueeze(1),
                v.unsqueeze(1),
                t.unsqueeze(1),
            ],
            dim=1,
        )
        B, PL, L_SIZE, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.fuser(x)

        x = x.reshape(B, PL, L_SIZE, C)
        surface, z, r, u, v, t = (
            x[:, 0, :, :],
            x[:, 1, :, :],
            x[:, 2, :, :],
            x[:, 3, :, :],
            x[:, 4, :, :],
            x[:, 5, :, :],
        )

        surface = self.decoder_surface(surface, skip_surface)
        z = self.decoder_z(z, skip_z)
        r = self.decoder_r(r, skip_r)
        u = self.decoder_u(u, skip_u)
        v = self.decoder_v(v, skip_v)
        t = self.decoder_t(t, skip_t)
        return surface, z, r, u, v, t
