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

import math
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import (
    DecoderLayer,
    EncoderLayer,
    FuserLayer,
)


@dataclass
class MetaData(ModelMetaData):
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
    r"""
    FengWu weather forecasting model.

    This implementation follows `FengWu: Pushing the Skillful Global Medium-range
    Weather Forecast beyond 10 Days Lead <https://arxiv.org/pdf/2304.02948.pdf>`_.

    Parameters
    ----------
    img_size : tuple[int, int], optional, default=(721, 1440)
        Spatial resolution :math:`(H, W)` of all input and output fields.
    pressure_level : int, optional, default=37
        Number of pressure levels :math:`L`.
    embed_dim : int, optional, default=192
        Embedding channel size used in encoder/decoder/fuser blocks.
    patch_size : tuple[int, int], optional, default=(4, 4)
        Patch size :math:`(p_h, p_w)` used by the hierarchical encoder/decoder.
    num_heads : tuple[int, int, int, int], optional, default=(6, 12, 12, 6)
        Number of attention heads used at each stage.
    window_size : tuple[int, int, int], optional, default=(2, 6, 12)
        Window size used by the transformer blocks.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` with
        :math:`C_{in} = 4 + 5L`.

    Outputs
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple ``(surface, z, r, u, v, t)`` where:

        - ``surface`` has shape :math:`(B, 4, H, W)`.
        - ``z, r, u, v, t`` each have shape :math:`(B, L, H, W)`.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (721, 1440),
        pressure_level: int = 37,
        embed_dim: int = 192,
        patch_size: tuple[int, int] = (4, 4),
        num_heads: tuple[int, int, int, int] = (6, 12, 12, 6),
        window_size: tuple[int, int, int] = (2, 6, 12),
    ) -> None:
        super().__init__(meta=MetaData())
        self.img_size = tuple(img_size)
        self.pressure_level = pressure_level
        self.patch_size = tuple(patch_size)
        self.embed_dim = embed_dim
        self.surface_channels = 4
        self.in_channels = self.surface_channels + 5 * self.pressure_level

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

    def prepare_input(
        self,
        surface: Float[torch.Tensor, "batch c_surface lat lon"],
        z: Float[torch.Tensor, "batch c_pressure lat lon"],
        r: Float[torch.Tensor, "batch c_pressure lat lon"],
        u: Float[torch.Tensor, "batch c_pressure lat lon"],
        v: Float[torch.Tensor, "batch c_pressure lat lon"],
        t: Float[torch.Tensor, "batch c_pressure lat lon"],
    ) -> Float[torch.Tensor, "batch channels lat lon"]:
        r"""
        Prepare input fields by concatenating all variables along channels.

        Parameters
        ----------
        surface : torch.Tensor
            Surface tensor of shape :math:`(B, 4, H, W)`.
        z : torch.Tensor
            Geopotential tensor of shape :math:`(B, L, H, W)`.
        r : torch.Tensor
            Relative humidity tensor of shape :math:`(B, L, H, W)`.
        u : torch.Tensor
            U-wind tensor of shape :math:`(B, L, H, W)`.
        v : torch.Tensor
            V-wind tensor of shape :math:`(B, L, H, W)`.
        t : torch.Tensor
            Temperature tensor of shape :math:`(B, L, H, W)`.

        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape :math:`(B, 4 + 5L, H, W)`.
        """
        if not torch.compiler.is_compiling():
            if surface.ndim != 4:
                raise ValueError(
                    f"Expected 'surface' to be a 4D tensor, got {surface.ndim}D tensor with shape {tuple(surface.shape)}"
                )
            if surface.shape[1] != self.surface_channels:
                raise ValueError(
                    f"Expected 'surface' to have {self.surface_channels} channels, got tensor with shape {tuple(surface.shape)}"
                )
            if surface.shape[2:] != self.img_size:
                raise ValueError(
                    f"Expected 'surface' spatial shape {self.img_size}, got tensor with shape {tuple(surface.shape)}"
                )

            batch_size = surface.shape[0]
            expected_spatial = surface.shape[2:]
            for name, tensor in (
                ("z", z),
                ("r", r),
                ("u", u),
                ("v", v),
                ("t", t),
            ):
                if tensor.ndim != 4:
                    raise ValueError(
                        f"Expected '{name}' to be a 4D tensor, got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
                    )
                if tensor.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected '{name}' batch size {batch_size}, got tensor with shape {tuple(tensor.shape)}"
                    )
                if tensor.shape[1] != self.pressure_level:
                    raise ValueError(
                        f"Expected '{name}' to have {self.pressure_level} channels, got tensor with shape {tuple(tensor.shape)}"
                    )
                if tensor.shape[2:] != expected_spatial:
                    raise ValueError(
                        f"Expected '{name}' spatial shape {expected_spatial}, got tensor with shape {tuple(tensor.shape)}"
                    )

        return torch.concat([surface, z, r, u, v, t], dim=1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels lat lon"],
    ) -> tuple[
        Float[torch.Tensor, "batch c_surface lat lon"],
        Float[torch.Tensor, "batch c_pressure lat lon"],
        Float[torch.Tensor, "batch c_pressure lat lon"],
        Float[torch.Tensor, "batch c_pressure lat lon"],
        Float[torch.Tensor, "batch c_pressure lat lon"],
        Float[torch.Tensor, "batch c_pressure lat lon"],
    ]:
        r"""
        Run Fengwu forward prediction.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated input tensor of shape :math:`(B, 4 + 5L, H, W)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Output tuple ``(surface, z, r, u, v, t)`` where ``surface`` has
            shape :math:`(B, 4, H, W)` and the other outputs have shape
            :math:`(B, L, H, W)`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 'x' to be a 4D tensor, got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected 'x' to have {self.in_channels} channels, got tensor with shape {tuple(x.shape)}"
                )
            if x.shape[2:] != self.img_size:
                raise ValueError(
                    f"Expected 'x' spatial shape {self.img_size}, got tensor with shape {tuple(x.shape)}"
                )

        pressure_level = self.pressure_level
        start = self.surface_channels
        surface = x[:, :start, :, :]
        z = x[:, start : start + pressure_level, :, :]
        start += pressure_level
        r = x[:, start : start + pressure_level, :, :]
        start += pressure_level
        u = x[:, start : start + pressure_level, :, :]
        start += pressure_level
        v = x[:, start : start + pressure_level, :, :]
        start += pressure_level
        t = x[:, start : start + pressure_level, :, :]
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
        batch_size, pressure_levels, latent_size, channels = x.shape
        x = x.reshape(batch_size, -1, channels)
        x = self.fuser(x)

        x = x.reshape(batch_size, pressure_levels, latent_size, channels)
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
