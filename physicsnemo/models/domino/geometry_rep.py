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
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from physicsnemo.models.unet import UNet
from physicsnemo.nn import BQWarp, Mlp, fourier_encode, get_activation

# from .encodings import fourier_encode


def scale_sdf(sdf: torch.Tensor, scaling_factor: float = 0.04) -> torch.Tensor:
    """
    Scale a signed distance function (SDF) to emphasize surface regions.

    This function applies a non-linear scaling to the SDF values that compresses
    the range while preserving the sign, effectively giving more weight to points
    near surfaces where abs(SDF) is small.

    Args:
        sdf: Tensor containing signed distance function values

    Returns:
        Tensor with scaled SDF values in range [-1, 1]
    """
    return sdf / (scaling_factor + torch.abs(sdf))


class GeoConvOut(nn.Module):
    """
    Geometry layer to project STL geometry data onto regular grids.
    """

    def __init__(
        self,
        input_features: int,
        neighbors_in_radius: int,
        model_parameters,
        grid_resolution=None,
    ):
        """
        Initialize the GeoConvOut layer.

        Args:
            input_features: Number of input feature dimensions
            neighbors_in_radius: Number of neighbors in radius
            model_parameters: Configuration parameters for the model
            grid_resolution: Resolution of the output grid [nx, ny, nz]
        """
        super().__init__()
        if grid_resolution is None:
            grid_resolution = [256, 96, 64]
        base_neurons = model_parameters.base_neurons
        self.fourier_features = model_parameters.fourier_features
        self.num_modes = model_parameters.num_modes

        if self.fourier_features:
            input_features_calculated = (
                input_features * (1 + 2 * self.num_modes) * neighbors_in_radius
            )
        else:
            input_features_calculated = input_features * neighbors_in_radius

        self.mlp = Mlp(
            in_features=input_features_calculated,
            hidden_features=[base_neurons, base_neurons // 2],
            out_features=model_parameters.base_neurons_in,
            act_layer=get_activation(model_parameters.activation),
            drop=0.0,
        )

        self.grid_resolution = grid_resolution

        self.activation = get_activation(model_parameters.activation)

        self.neighbors_in_radius = neighbors_in_radius

        if self.fourier_features:
            self.register_buffer(
                "freqs", torch.exp(torch.linspace(0, math.pi, self.num_modes))
            )

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
        radius: float = 0.025,
        neighbors_in_radius: int = 10,
    ) -> torch.Tensor:
        """
        Process and project geometric features onto a 3D grid.

        Args:
            x: Input tensor containing coordinates of the neighboring points
               (batch_size, nx*ny*nz, n_points, 3)
            grid: Input tensor represented as a grid of shape
                (batch_size, nx, ny, nz, 3)

        Returns:
            Processed geometry features of shape (batch_size, base_neurons_in, nx, ny, nz)
        """

        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )
        grid = grid.reshape(1, nx * ny * nz, 3, 1)

        x = rearrange(
            x, "b x y z -> b x (y z)", x=nx * ny * nz, y=self.neighbors_in_radius, z=3
        )
        if self.fourier_features:
            facets = torch.cat((x, fourier_encode(x, self.freqs)), axis=-1)
        else:
            facets = x

        x = F.tanh(self.mlp(facets))

        x = rearrange(x, "b (x y z) c -> b c x y z", x=nx, y=ny, z=nz)

        return x


class GeoProcessor(nn.Module):
    """Geometry processing layer using CNNs"""

    def __init__(self, input_filters: int, output_filters: int, model_parameters):
        """
        Initialize the GeoProcessor network.

        Args:
            input_filters: Number of input channels
            model_parameters: Configuration parameters for the model
        """
        super().__init__()
        base_filters = model_parameters.base_filters
        self.conv1 = nn.Conv3d(
            input_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv3d(
            base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv3 = nn.Conv3d(
            2 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv3_1 = nn.Conv3d(
            4 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv3d(
            4 * base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv5 = nn.Conv3d(
            4 * base_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv6 = nn.Conv3d(
            2 * base_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv7 = nn.Conv3d(
            2 * input_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv8 = nn.Conv3d(
            input_filters, output_filters, kernel_size=3, padding="same"
        )
        self.avg_pool = torch.nn.AvgPool3d((2, 2, 2))
        self.max_pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.activation = get_activation(model_parameters.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process geometry information through the 3D CNN network.

        The network follows an encoder-decoder architecture with skip connections:
        1. Downsampling path (encoder) with three levels of max pooling
        2. Processing loop in the bottleneck
        3. Upsampling path (decoder) with skip connections from the encoder

        Args:
            x: Input tensor containing grid-represented geometry of shape
               (batch_size, input_filters, nx, ny, nz)

        Returns:
            Processed geometry features of shape (batch_size, 1, nx, ny, nz)
        """
        # Encoder
        x0 = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x1 = x
        x = self.conv2(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x2 = x
        x = self.conv3(x)
        x = self.activation(x)
        x = self.max_pool(x)

        # Processor loop
        x = self.activation(self.conv3_1(x))

        # Decoder
        x = self.conv4(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = torch.cat((x, x2), dim=1)

        x = self.conv5(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = torch.cat((x, x1), dim=1)

        x = self.conv6(x)
        x = self.activation(x)
        x = self.upsample(x)
        x = torch.cat((x, x0), dim=1)

        x = self.activation(self.conv7(x))
        x = self.conv8(x)

        return x


class GeometryRep(nn.Module):
    """
    Geometry representation module that processes STL geometry data.

    This module constructs a multiscale representation of geometry by:
    1. Computing multi-scale geometry encoding for local and global context
    2. Processing signed distance field (SDF) data for surface information

    The combined encoding enables the model to reason about both local and global
    geometric properties.
    """

    def __init__(
        self,
        input_features: int,
        radii: Sequence[float],
        neighbors_in_radius,
        hops=1,
        sdf_scaling_factor: Sequence[float] = [0.04],
        model_parameters=None,
        # activation_conv: nn.Module,
        # activation_processor: nn.Module,
    ):
        """
        Initialize the GeometryRep module.

        Args:
            input_features: Number of input feature dimensions
            model_parameters: Configuration parameters for the model
        """
        super().__init__()
        geometry_rep = model_parameters.geometry_rep
        self.geo_encoding_type = model_parameters.geometry_encoding_type
        self.cross_attention = geometry_rep.geo_processor.cross_attention
        self.self_attention = geometry_rep.geo_processor.self_attention
        self.activation_conv = get_activation(geometry_rep.geo_conv.activation)
        self.activation_processor = geometry_rep.geo_processor.activation
        self.sdf_scaling_factor = sdf_scaling_factor

        self.bq_warp = nn.ModuleList()
        self.geo_processors = nn.ModuleList()
        for j in range(len(radii)):
            self.bq_warp.append(
                BQWarp(
                    radius=radii[j],
                    neighbors_in_radius=neighbors_in_radius[j],
                )
            )
            if geometry_rep.geo_processor.processor_type == "unet":
                h = geometry_rep.geo_processor.base_filters
                if self.self_attention:
                    normalization_in_unet = "layernorm"
                else:
                    normalization_in_unet = None
                self.geo_processors.append(
                    UNet(
                        in_channels=geometry_rep.geo_conv.base_neurons_in,
                        out_channels=geometry_rep.geo_conv.base_neurons_out,
                        model_depth=3,
                        feature_map_channels=[
                            h,
                            2 * h,
                            4 * h,
                        ],
                        num_conv_blocks=1,
                        kernel_size=3,
                        stride=1,
                        conv_activation=self.activation_processor,
                        padding=1,
                        padding_mode="zeros",
                        pooling_type="MaxPool3d",
                        pool_size=2,
                        normalization=normalization_in_unet,
                        use_attn_gate=self.self_attention,
                        attn_decoder_feature_maps=[4 * h, 2 * h],
                        attn_feature_map_channels=[2 * h, h],
                        attn_intermediate_channels=4 * h,
                        gradient_checkpointing=True,
                    )
                )
            elif geometry_rep.geo_processor.processor_type == "conv":
                self.geo_processors.append(
                    nn.Sequential(
                        GeoProcessor(
                            input_filters=geometry_rep.geo_conv.base_neurons_in,
                            output_filters=geometry_rep.geo_conv.base_neurons_out,
                            model_parameters=geometry_rep.geo_processor,
                        ),
                    )
                )
            else:
                raise ValueError("Invalid prompt. Specify unet or conv ...")

        self.geo_conv_out = nn.ModuleList()
        self.geo_processor_out = nn.ModuleList()
        for u in range(len(radii)):
            self.geo_conv_out.append(
                GeoConvOut(
                    input_features=input_features,
                    neighbors_in_radius=neighbors_in_radius[u],
                    model_parameters=geometry_rep.geo_conv,
                    grid_resolution=model_parameters.interp_res,
                )
            )
            self.geo_processor_out.append(
                nn.Conv3d(
                    geometry_rep.geo_conv.base_neurons_out,
                    1,
                    kernel_size=3,
                    padding="same",
                )
            )

        if geometry_rep.geo_processor.processor_type == "unet":
            h = geometry_rep.geo_processor.base_filters
            if self.self_attention:
                normalization_in_unet = "layernorm"
            else:
                normalization_in_unet = None

            self.geo_processor_sdf = UNet(
                in_channels=5 + len(self.sdf_scaling_factor),
                out_channels=geometry_rep.geo_conv.base_neurons_out,
                model_depth=3,
                feature_map_channels=[
                    h,
                    2 * h,
                    4 * h,
                ],
                num_conv_blocks=1,
                kernel_size=3,
                stride=1,
                conv_activation=self.activation_processor,
                padding=1,
                padding_mode="zeros",
                pooling_type="MaxPool3d",
                pool_size=2,
                normalization=normalization_in_unet,
                use_attn_gate=self.self_attention,
                attn_decoder_feature_maps=[4 * h, 2 * h],
                attn_feature_map_channels=[2 * h, h],
                attn_intermediate_channels=4 * h,
                gradient_checkpointing=True,
            )
        elif geometry_rep.geo_processor.processor_type == "conv":
            self.geo_processor_sdf = nn.Sequential(
                GeoProcessor(
                    input_filters=5 + len(self.sdf_scaling_factor),
                    output_filters=geometry_rep.geo_conv.base_neurons_out,
                    model_parameters=geometry_rep.geo_processor,
                ),
            )
        else:
            raise ValueError("Invalid prompt. Specify unet or conv ...")
        self.radii = radii
        self.neighbors_in_radius = neighbors_in_radius
        self.hops = hops

        self.geo_processor_sdf_out = nn.Conv3d(
            geometry_rep.geo_conv.base_neurons_out, 1, kernel_size=3, padding="same"
        )

        if self.cross_attention:
            self.combined_unet = UNet(
                in_channels=1 + len(radii),
                out_channels=1 + len(radii),
                model_depth=3,
                feature_map_channels=[
                    h,
                    2 * h,
                    4 * h,
                ],
                num_conv_blocks=1,
                kernel_size=3,
                stride=1,
                conv_activation=self.activation_processor,
                padding=1,
                padding_mode="zeros",
                pooling_type="MaxPool3d",
                pool_size=2,
                normalization="layernorm",
                use_attn_gate=True,
                attn_decoder_feature_maps=[4 * h, 2 * h],
                attn_feature_map_channels=[2 * h, h],
                attn_intermediate_channels=4 * h,
                gradient_checkpointing=True,
            )

    def forward(
        self, x: torch.Tensor, p_grid: torch.Tensor, sdf: torch.Tensor
    ) -> torch.Tensor:
        """
        Process geometry data to create a comprehensive representation.

        This method combines short-range, long-range, and SDF-based geometry
        encodings to create a rich representation of the geometry.

        Args:
            x: Input tensor containing geometric point data
            p_grid: Grid points for sampling
            sdf: Signed distance field tensor

        Returns:
            Comprehensive geometry encoding that concatenates short-range,
            SDF-based, and long-range features
        """
        if self.geo_encoding_type == "both" or self.geo_encoding_type == "stl":
            # Calculate multi-scale geoemtry dependency
            x_encoding = []
            for j in range(len(self.radii)):
                mapping, k_short = self.bq_warp[j](x, p_grid)
                x_encoding_inter = self.geo_conv_out[j](k_short, p_grid)
                # Propagate information in the geometry enclosed BBox
                for _ in range(self.hops):
                    dx = self.geo_processors[j](x_encoding_inter) / self.hops
                    x_encoding_inter = x_encoding_inter + dx
                x_encoding_inter = self.geo_processor_out[j](x_encoding_inter)
                x_encoding.append(x_encoding_inter)
            x_encoding = torch.cat(x_encoding, dim=1)

        if self.geo_encoding_type == "both" or self.geo_encoding_type == "sdf":
            # Expand SDF
            sdf = torch.unsqueeze(sdf, 1)
            # Binary sdf
            binary_sdf = torch.where(sdf >= 0, 0.0, 1.0)
            # Gradients of SDF
            sdf_x, sdf_y, sdf_z = torch.gradient(sdf, dim=[2, 3, 4])

            scaled_sdf = []
            # Scaled sdf to emphasize near surface
            for s in range(len(self.sdf_scaling_factor)):
                s_sdf = scale_sdf(sdf, self.sdf_scaling_factor[s])
                scaled_sdf.append(s_sdf)

            scaled_sdf = torch.cat(scaled_sdf, dim=1)

            # Process SDF and its computed features
            sdf = torch.cat((sdf, scaled_sdf, binary_sdf, sdf_x, sdf_y, sdf_z), 1)

            sdf_encoding = self.geo_processor_sdf(sdf)
            sdf_encoding = self.geo_processor_sdf_out(sdf_encoding)

        if self.geo_encoding_type == "both":
            # Geometry encoding comprised of short-range, long-range and SDF features
            encoding_g = torch.cat((x_encoding, sdf_encoding), 1)
        elif self.geo_encoding_type == "sdf":
            encoding_g = sdf_encoding
        elif self.geo_encoding_type == "stl":
            encoding_g = x_encoding

        if self.cross_attention:
            encoding_g = self.combined_unet(encoding_g)

        return encoding_g
