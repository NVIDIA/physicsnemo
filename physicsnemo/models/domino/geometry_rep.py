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

r"""
DoMINO Geometry Representation Modules.

This module contains geometry representation layers for the DoMINO model architecture,
including SDF scaling, geometry convolution, and geometry processing modules.
"""

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.models.unet import UNet
from physicsnemo.nn import BQWarp, Mlp, fourier_encode, get_activation


def scale_sdf(
    sdf: Float[torch.Tensor, "..."],
    scaling_factor: float = 0.04,
) -> Float[torch.Tensor, "..."]:
    r"""
    Scale a signed distance function (SDF) to emphasize surface regions.

    This function applies a non-linear scaling to the SDF values that compresses
    the range while preserving the sign, effectively giving more weight to points
    near surfaces where :math:`|\text{SDF}|` is small.

    Parameters
    ----------
    sdf : torch.Tensor
        Tensor containing signed distance function values.
    scaling_factor : float, optional, default=0.04
        Controls the steepness of the scaling. Smaller values emphasize
        regions closer to the surface.

    Returns
    -------
    torch.Tensor
        Tensor with scaled SDF values in range :math:`[-1, 1]`.

    Note
    ----
    The scaling formula is: :math:`\text{scaled} = \frac{\text{sdf}}{s + |\text{sdf}|}`
    where :math:`s` is the scaling factor.
    """
    return sdf / (scaling_factor + torch.abs(sdf))


class GeoConvOut(Module):
    r"""
    Geometry layer to project STL geometry data onto regular grids.

    This module processes ball-query outputs through an MLP to produce
    a grid-based geometry representation.

    Parameters
    ----------
    input_features : int
        Number of input feature dimensions (typically 3 for x, y, z coordinates).
    neighbors_in_radius : int
        Number of neighbors to consider within the ball query radius.
    model_parameters : Any
        Configuration parameters containing:
        - ``base_neurons``: Number of neurons in hidden layers
        - ``fourier_features``: Whether to use Fourier feature encoding
        - ``num_modes``: Number of Fourier modes if using Fourier features
        - ``base_neurons_in``: Output feature dimension
        - ``activation``: Activation function name
    grid_resolution : list[int], optional, default=[256, 96, 64]
        Resolution of the output grid as :math:`[N_x, N_y, N_z]`.

    Forward
    -------
    x : torch.Tensor
        Input tensor containing coordinates of neighboring points
        of shape :math:`(B, N_x \cdot N_y \cdot N_z, K, 3)` where :math:`K`
        is ``neighbors_in_radius``.
    grid : torch.Tensor
        Input tensor represented as a grid of shape :math:`(B, N_x, N_y, N_z, 3)`.

    Outputs
    -------
    torch.Tensor
        Processed geometry features of shape :math:`(B, D_{out}, N_x, N_y, N_z)`
        where :math:`D_{out}` is ``base_neurons_in``.

    See Also
    --------
    :class:`~physicsnemo.nn.Mlp` : MLP module used for feature processing.
    :class:`GeometryRep` : Uses this module for multi-scale geometry encoding.
    """

    def __init__(
        self,
        input_features: int,
        neighbors_in_radius: int,
        model_parameters: Any,
        grid_resolution: list[int] | None = None,
    ):
        super().__init__(meta=None)
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
        x: Float[torch.Tensor, "batch grid_points neighbors 3"],
        grid: Float[torch.Tensor, "batch nx ny nz 3"],
        radius: float = 0.025,
        neighbors_in_radius: int = 10,
    ) -> Float[torch.Tensor, "batch out_features nx ny nz"]:
        r"""
        Process and project geometric features onto a 3D grid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing coordinates of the neighboring points
            of shape :math:`(B, N_x \cdot N_y \cdot N_z, K, 3)`.
        grid : torch.Tensor
            Input tensor represented as a grid of shape :math:`(B, N_x, N_y, N_z, 3)`.
        radius : float, optional, default=0.025
            Ball query radius (unused, kept for API compatibility).
        neighbors_in_radius : int, optional, default=10
            Number of neighbors (unused, kept for API compatibility).

        Returns
        -------
        torch.Tensor
            Processed geometry features of shape :math:`(B, D_{out}, N_x, N_y, N_z)`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 4 or x.shape[-1] != 3:
                raise ValueError(
                    f"Expected x of shape (B, N, K, 3), got shape {tuple(x.shape)}"
                )
            if grid.ndim != 5 or grid.shape[-1] != 3:
                raise ValueError(
                    f"Expected grid of shape (B, Nx, Ny, Nz, 3), "
                    f"got shape {tuple(grid.shape)}"
                )

        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )
        grid = grid.reshape(1, nx * ny * nz, 3, 1)

        # Rearrange input to flatten spatial and neighbor dimensions
        x = rearrange(
            x, "b x y z -> b x (y z)", x=nx * ny * nz, y=self.neighbors_in_radius, z=3
        )

        # Apply Fourier feature encoding if enabled
        if self.fourier_features:
            facets = torch.cat((x, fourier_encode(x, self.freqs)), axis=-1)
        else:
            facets = x

        # Process through MLP with tanh activation
        x = F.tanh(self.mlp(facets))

        # Reshape to 3D grid format
        x = rearrange(x, "b (x y z) c -> b c x y z", x=nx, y=ny, z=nz)

        return x


class GeoProcessor(Module):
    r"""
    Geometry processing layer using 3D CNNs.

    This module implements an encoder-decoder architecture with skip connections
    for processing 3D geometry representations. It follows a U-Net-like structure
    with three levels of downsampling and upsampling.

    Parameters
    ----------
    input_filters : int
        Number of input channels.
    output_filters : int
        Number of output channels.
    model_parameters : Any
        Configuration parameters containing:
        - ``base_filters``: Base number of filters in CNN layers
        - ``activation``: Activation function name

    Forward
    -------
    x : torch.Tensor
        Input tensor containing grid-represented geometry of shape
        :math:`(B, C_{in}, N_x, N_y, N_z)`.

    Outputs
    -------
    torch.Tensor
        Processed geometry features of shape :math:`(B, C_{out}, N_x, N_y, N_z)`.

    See Also
    --------
    :class:`~physicsnemo.models.unet.UNet` : Alternative processor using UNet architecture.
    """

    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        model_parameters: Any,
    ):
        super().__init__(meta=None)
        base_filters = model_parameters.base_filters

        # Encoder layers
        self.conv1 = nn.Conv3d(
            input_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv3d(
            base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv3 = nn.Conv3d(
            2 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )

        # Bottleneck
        self.conv3_1 = nn.Conv3d(
            4 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )

        # Decoder layers
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

        # Pooling and upsampling
        self.avg_pool = torch.nn.AvgPool3d((2, 2, 2))
        self.max_pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.activation = get_activation(model_parameters.activation)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels nx ny nz"],
    ) -> Float[torch.Tensor, "batch out_channels nx ny nz"]:
        r"""
        Process geometry information through the 3D CNN network.

        The network follows an encoder-decoder architecture with skip connections:
        1. Downsampling path (encoder) with three levels of max pooling
        2. Processing loop in the bottleneck
        3. Upsampling path (decoder) with skip connections from the encoder

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing grid-represented geometry of shape
            :math:`(B, C_{in}, N_x, N_y, N_z)`.

        Returns
        -------
        torch.Tensor
            Processed geometry features of shape :math:`(B, C_{out}, N_x, N_y, N_z)`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(
                    f"Expected x to be 5D (B, C, Nx, Ny, Nz), "
                    f"got {x.ndim}D with shape {tuple(x.shape)}"
                )

        # Encoder path
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

        # Bottleneck
        x = self.activation(self.conv3_1(x))

        # Decoder path with skip connections
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


class GeometryRep(Module):
    r"""
    Geometry representation module that processes STL geometry data.

    This module constructs a multiscale representation of geometry by:

    1. Computing multi-scale geometry encoding for local and global context
       using ball queries at different radii
    2. Processing signed distance field (SDF) data for surface information
    3. Optionally combining encodings using cross-attention

    The combined encoding enables the model to reason about both local and global
    geometric properties.

    Parameters
    ----------
    input_features : int
        Number of input feature dimensions (typically 3 for x, y, z coordinates).
    radii : Sequence[float]
        List of radii for multi-scale ball queries.
    neighbors_in_radius : Sequence[int]
        List of neighbor counts for each radius scale.
    hops : int, optional, default=1
        Number of message passing hops in geometry processors.
    sdf_scaling_factor : Sequence[float], optional, default=[0.04]
        Scaling factors for SDF encoding to emphasize near-surface regions.
    model_parameters : Any, optional, default=None
        Configuration parameters for geometry representation, containing
        nested config for ``geometry_rep.geo_conv`` and ``geometry_rep.geo_processor``.

    Forward
    -------
    x : torch.Tensor
        Input tensor containing geometric point data of shape :math:`(B, N_{geo}, 3)`.
    p_grid : torch.Tensor
        Grid points for sampling of shape :math:`(B, N_x, N_y, N_z, 3)`.
    sdf : torch.Tensor
        Signed distance field tensor of shape :math:`(B, N_x, N_y, N_z)`.

    Outputs
    -------
    torch.Tensor
        Comprehensive geometry encoding of shape :math:`(B, C, N_x, N_y, N_z)`
        that concatenates multi-scale STL-based and SDF-based features.
        The number of channels :math:`C` depends on the ``geo_encoding_type``.

    Example
    -------
    >>> import torch
    >>> # GeometryRep requires model_parameters configuration
    >>> # See DoMINO model for typical usage

    See Also
    --------
    :class:`GeoConvOut` : Used for projecting geometry onto grids.
    :class:`GeoProcessor` : Used for processing geometry features.
    :class:`~physicsnemo.models.unet.UNet` : Alternative processor architecture.
    """

    def __init__(
        self,
        input_features: int,
        radii: Sequence[float],
        neighbors_in_radius: Sequence[int],
        hops: int = 1,
        sdf_scaling_factor: Sequence[float] = [0.04],
        model_parameters: Any = None,
    ):
        super().__init__(meta=None)
        geometry_rep = model_parameters.geometry_rep
        self.geo_encoding_type = model_parameters.geometry_encoding_type
        self.cross_attention = geometry_rep.geo_processor.cross_attention
        self.self_attention = geometry_rep.geo_processor.self_attention
        self.activation_conv = get_activation(geometry_rep.geo_conv.activation)
        self.activation_processor = geometry_rep.geo_processor.activation
        self.sdf_scaling_factor = sdf_scaling_factor

        # Build ball query warp and geometry processors for each scale
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
                raise ValueError(
                    f"Invalid processor_type: {geometry_rep.geo_processor.processor_type}. "
                    f"Specify 'unet' or 'conv'."
                )

        # Build geometry convolution output layers
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

        # Build SDF processor
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
            raise ValueError(
                f"Invalid processor_type: {geometry_rep.geo_processor.processor_type}. "
                f"Specify 'unet' or 'conv'."
            )

        self.radii = radii
        self.neighbors_in_radius = neighbors_in_radius
        self.hops = hops

        self.geo_processor_sdf_out = nn.Conv3d(
            geometry_rep.geo_conv.base_neurons_out, 1, kernel_size=3, padding="same"
        )

        # Optional cross-attention for combining encodings
        if self.cross_attention:
            h = geometry_rep.geo_processor.base_filters
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
        self,
        x: Float[torch.Tensor, "batch num_geo 3"],
        p_grid: Float[torch.Tensor, "batch nx ny nz 3"],
        sdf: Float[torch.Tensor, "batch nx ny nz"],
    ) -> Float[torch.Tensor, "batch channels nx ny nz"]:
        r"""
        Process geometry data to create a comprehensive representation.

        This method combines short-range, long-range, and SDF-based geometry
        encodings to create a rich representation of the geometry.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing geometric point data of shape :math:`(B, N_{geo}, 3)`.
        p_grid : torch.Tensor
            Grid points for sampling of shape :math:`(B, N_x, N_y, N_z, 3)`.
        sdf : torch.Tensor
            Signed distance field tensor of shape :math:`(B, N_x, N_y, N_z)`.

        Returns
        -------
        torch.Tensor
            Comprehensive geometry encoding of shape :math:`(B, C, N_x, N_y, N_z)`
            that concatenates STL-based and/or SDF-based features depending on
            ``geo_encoding_type``.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 3 or x.shape[-1] != 3:
                raise ValueError(
                    f"Expected x of shape (B, N, 3), got shape {tuple(x.shape)}"
                )
            if p_grid.ndim != 5 or p_grid.shape[-1] != 3:
                raise ValueError(
                    f"Expected p_grid of shape (B, Nx, Ny, Nz, 3), "
                    f"got shape {tuple(p_grid.shape)}"
                )
            if sdf.ndim != 4:
                raise ValueError(
                    f"Expected sdf of shape (B, Nx, Ny, Nz), "
                    f"got {sdf.ndim}D with shape {tuple(sdf.shape)}"
                )

        # Compute multi-scale STL-based geometry encoding
        if self.geo_encoding_type == "both" or self.geo_encoding_type == "stl":
            x_encoding = []
            for j in range(len(self.radii)):
                # Ball query to find neighbors
                mapping, k_short = self.bq_warp[j](x, p_grid)

                # Project neighbors onto grid
                x_encoding_inter = self.geo_conv_out[j](k_short, p_grid)

                # Propagate information through geometry processor
                for _ in range(self.hops):
                    dx = self.geo_processors[j](x_encoding_inter) / self.hops
                    x_encoding_inter = x_encoding_inter + dx

                # Final projection to single channel
                x_encoding_inter = self.geo_processor_out[j](x_encoding_inter)
                x_encoding.append(x_encoding_inter)

            x_encoding = torch.cat(x_encoding, dim=1)

        # Compute SDF-based geometry encoding
        if self.geo_encoding_type == "both" or self.geo_encoding_type == "sdf":
            # Expand SDF to add channel dimension
            sdf = torch.unsqueeze(sdf, 1)

            # Compute binary SDF (inside/outside indicator)
            binary_sdf = torch.where(sdf >= 0, 0.0, 1.0)

            # Compute SDF gradients
            sdf_x, sdf_y, sdf_z = torch.gradient(sdf, dim=[2, 3, 4])

            # Apply multiple scaling factors to emphasize near-surface regions
            scaled_sdf = []
            for s in range(len(self.sdf_scaling_factor)):
                s_sdf = scale_sdf(sdf, self.sdf_scaling_factor[s])
                scaled_sdf.append(s_sdf)
            scaled_sdf = torch.cat(scaled_sdf, dim=1)

            # Concatenate all SDF features
            sdf = torch.cat((sdf, scaled_sdf, binary_sdf, sdf_x, sdf_y, sdf_z), 1)

            # Process SDF features
            sdf_encoding = self.geo_processor_sdf(sdf)
            sdf_encoding = self.geo_processor_sdf_out(sdf_encoding)

        # Combine encodings based on encoding type
        if self.geo_encoding_type == "both":
            encoding_g = torch.cat((x_encoding, sdf_encoding), 1)
        elif self.geo_encoding_type == "sdf":
            encoding_g = sdf_encoding
        elif self.geo_encoding_type == "stl":
            encoding_g = x_encoding
        else:
            raise ValueError(
                f"Invalid geo_encoding_type: {self.geo_encoding_type}. "
                f"Must be 'both', 'stl', or 'sdf'."
            )

        # Apply cross-attention if enabled
        if self.cross_attention:
            encoding_g = self.combined_unet(encoding_g)

        return encoding_g
