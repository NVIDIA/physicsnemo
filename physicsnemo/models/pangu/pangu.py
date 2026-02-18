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
from physicsnemo.nn import DownSample3D, FuserLayer, UpSample3D
from physicsnemo.nn.module.utils import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
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


class Pangu(Module):
    r"""
    Pangu weather forecasting model.

    This implementation follows `Pangu-Weather: A 3D High-Resolution Model for
    Fast and Accurate Global Weather Forecast
    <https://arxiv.org/abs/2211.02556>`_.

    Parameters
    ----------
    img_size : tuple[int, int], optional, default=(721, 1440)
        Spatial resolution :math:`(H, W)` of the latitude-longitude grid.
    patch_size : tuple[int, int, int], optional, default=(2, 4, 4)
        Patch size :math:`(p_l, p_h, p_w)` for pressure-level and spatial axes.
    embed_dim : int, optional, default=192
        Embedding channel size used throughout the transformer hierarchy.
    num_heads : tuple[int, int, int, int], optional, default=(6, 12, 12, 6)
        Number of attention heads used at each stage.
    window_size : tuple[int, int, int], optional, default=(2, 6, 12)
        Window size used by the transformer blocks.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, 72, H, W)` where channels are arranged
        as ``surface(7) + upper_air(5*13)``.

    Outputs
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple ``(surface, upper_air)`` where ``surface`` has shape
        :math:`(B, 4, H, W)` and ``upper_air`` has shape
        :math:`(B, 5, 13, H, W)`.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (721, 1440),
        patch_size: tuple[int, int, int] = (2, 4, 4),
        embed_dim: int = 192,
        num_heads: tuple[int, int, int, int] = (6, 12, 12, 6),
        window_size: tuple[int, int, int] = (2, 6, 12),
    ) -> None:
        super().__init__(meta=MetaData())
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.embed_dim = embed_dim
        self.surface_channels = 4
        self.surface_mask_channels = 3
        self.upper_air_channels = 5
        self.upper_air_levels = 13
        self.surface_input_channels = self.surface_channels + self.surface_mask_channels
        self.in_channels = (
            self.surface_input_channels
            + self.upper_air_channels * self.upper_air_levels
        )

        drop_path = np.linspace(0, 0.2, 8).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size[1:],
            in_chans=4 + 3,  # add
            embed_dim=embed_dim,
        )
        self.patchembed3d = PatchEmbed3D(
            img_size=(13, img_size[0], img_size[1]),
            patch_size=patch_size,
            in_chans=5,
            embed_dim=embed_dim,
        )
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.layer1 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )

        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )
        self.downsample = DownSample3D(
            in_dim=embed_dim,
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
        )
        self.layer2 = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.layer3 = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.upsample = UpSample3D(
            embed_dim * 2, embed_dim, patched_inp_shape_downsample, patched_inp_shape
        )
        self.layer4 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D(
            img_size, patch_size[1:], 2 * embed_dim, 4
        )
        self.patchrecovery3d = PatchRecovery3D(
            (13, img_size[0], img_size[1]), patch_size, 2 * embed_dim, 5
        )

    def prepare_input(
        self,
        surface: Float[torch.Tensor, "batch c_surface lat lon"],
        surface_mask: Float[torch.Tensor, "c_mask lat lon"]
        | Float[torch.Tensor, "batch c_mask lat lon"],
        upper_air: Float[torch.Tensor, "batch c_upper levels lat lon"],
    ) -> Float[torch.Tensor, "batch channels lat lon"]:
        r"""
        Prepare input by combining surface, static masks, and upper-air fields.

        Parameters
        ----------
        surface : torch.Tensor
            Surface tensor of shape :math:`(B, 4, H, W)`.
        surface_mask : torch.Tensor
            Static mask tensor of shape :math:`(3, H, W)` or
            :math:`(B, 3, H, W)`.
        upper_air : torch.Tensor
            Upper-air tensor of shape :math:`(B, 5, 13, H, W)`.

        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape :math:`(B, 72, H, W)`.
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

            if surface_mask.ndim not in (3, 4):
                raise ValueError(
                    f"Expected 'surface_mask' to be a 3D or 4D tensor, got {surface_mask.ndim}D tensor with shape {tuple(surface_mask.shape)}"
                )
            if surface_mask.ndim == 3:
                if surface_mask.shape[0] != self.surface_mask_channels:
                    raise ValueError(
                        f"Expected 'surface_mask' to have {self.surface_mask_channels} channels, got tensor with shape {tuple(surface_mask.shape)}"
                    )
                if surface_mask.shape[1:] != self.img_size:
                    raise ValueError(
                        f"Expected 'surface_mask' spatial shape {self.img_size}, got tensor with shape {tuple(surface_mask.shape)}"
                    )
            else:
                if surface_mask.shape[0] != surface.shape[0]:
                    raise ValueError(
                        f"Expected 'surface_mask' batch size {surface.shape[0]}, got tensor with shape {tuple(surface_mask.shape)}"
                    )
                if surface_mask.shape[1] != self.surface_mask_channels:
                    raise ValueError(
                        f"Expected 'surface_mask' to have {self.surface_mask_channels} channels, got tensor with shape {tuple(surface_mask.shape)}"
                    )
                if surface_mask.shape[2:] != self.img_size:
                    raise ValueError(
                        f"Expected 'surface_mask' spatial shape {self.img_size}, got tensor with shape {tuple(surface_mask.shape)}"
                    )

            if upper_air.ndim != 5:
                raise ValueError(
                    f"Expected 'upper_air' to be a 5D tensor, got {upper_air.ndim}D tensor with shape {tuple(upper_air.shape)}"
                )
            if upper_air.shape[0] != surface.shape[0]:
                raise ValueError(
                    f"Expected 'upper_air' batch size {surface.shape[0]}, got tensor with shape {tuple(upper_air.shape)}"
                )
            if upper_air.shape[1] != self.upper_air_channels:
                raise ValueError(
                    f"Expected 'upper_air' to have {self.upper_air_channels} channels, got tensor with shape {tuple(upper_air.shape)}"
                )
            if upper_air.shape[2] != self.upper_air_levels:
                raise ValueError(
                    f"Expected 'upper_air' to have {self.upper_air_levels} pressure levels, got tensor with shape {tuple(upper_air.shape)}"
                )
            if upper_air.shape[3:] != self.img_size:
                raise ValueError(
                    f"Expected 'upper_air' spatial shape {self.img_size}, got tensor with shape {tuple(upper_air.shape)}"
                )

        upper_air = upper_air.reshape(
            upper_air.shape[0], -1, upper_air.shape[3], upper_air.shape[4]
        )
        if surface_mask.ndim == 3:
            surface_mask = surface_mask.unsqueeze(0).repeat(surface.shape[0], 1, 1, 1)
        return torch.concat([surface, surface_mask, upper_air], dim=1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels lat lon"],
    ) -> tuple[
        Float[torch.Tensor, "batch c_surface lat lon"],
        Float[torch.Tensor, "batch c_upper levels lat lon"],
    ]:
        r"""
        Run Pangu forward prediction.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated input tensor of shape :math:`(B, 72, H, W)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output tuple ``(surface, upper_air)`` with shapes
            :math:`(B, 4, H, W)` and :math:`(B, 5, 13, H, W)`.
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

        surface = x[:, : self.surface_input_channels, :, :]
        upper_air = x[:, self.surface_input_channels :, :, :].reshape(
            x.shape[0],
            self.upper_air_channels,
            self.upper_air_levels,
            x.shape[2],
            x.shape[3],
        )
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air
