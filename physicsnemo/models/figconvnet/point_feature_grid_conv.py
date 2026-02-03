# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

r"""Grid feature convolution operations for FIGConvNet.

This module provides 2D convolution operations for factorized grid features,
which are the core building blocks of the FIGConvNet architecture.

The main classes are:

- :class:`GridFeatureConv2d`: 2D convolution on factorized grid features
- :class:`GridFeatureConv2dBlock`: Residual block with two convolutions
- :class:`GridFeatureTransform`: Apply arbitrary transforms to grid features
- :class:`GridFeatureMemoryFormatConverter`: Convert between memory formats
"""

# ruff: noqa: S101
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from physicsnemo.models.figconvnet.geometries import (
    GridFeatures,
    GridFeaturesMemoryFormat,
)


class GridFeaturePadToMatch(nn.Module):
    r"""Pad or crop grid features to match a reference resolution.

    This module adjusts the spatial dimensions of grid features to match
    a reference grid, either by padding with zeros or cropping.

    Parameters
    ----------
    None

    Forward
    -------
    ref_grid : GridFeatures
        Reference grid defining target resolution.
    x_grid : GridFeatures
        Grid to be padded/cropped.

    Outputs
    -------
    GridFeatures
        Grid features with spatial dimensions matching reference.

    Note
    ----
    Both grids must have the same memory format (cannot be ``b_x_y_z_c``).
    """

    def forward(self, ref_grid: GridFeatures, x_grid: GridFeatures) -> GridFeatures:
        r"""Pad or crop grid to match reference.

        Parameters
        ----------
        ref_grid : GridFeatures
            Reference grid defining target dimensions.
        x_grid : GridFeatures
            Grid to be modified.

        Returns
        -------
        GridFeatures
            Modified grid matching reference dimensions.
        """
        if ref_grid.memory_format != x_grid.memory_format:
            raise ValueError(
                f"Memory format mismatch: ref_grid has {ref_grid.memory_format}, "
                f"x_grid has {x_grid.memory_format}"
            )
        if x_grid.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            raise ValueError(
                "b_x_y_z_c memory format is not supported for GridFeaturePadToMatch"
            )

        if x_grid.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            # Handle 3D grid format
            x = x_grid.batch_features
            height, width, depth = x_grid.resolution
            ref_height, ref_width, ref_depth = ref_grid.resolution

            # Compute padding/cropping needed
            pad_height = ref_height - height
            pad_width = ref_width - width
            pad_depth = ref_depth - depth

            # Crop if any dimension is negative
            if pad_height < 0 or pad_width < 0 or pad_depth < 0:
                x = x[:, :, :ref_height, :ref_width, :ref_depth]

            # Pad if any dimension is positive
            if pad_height > 0 or pad_width > 0 or pad_depth > 0:
                x = F.pad(x, (0, pad_depth, 0, pad_width, 0, pad_height), "constant", 0)

            return GridFeatures(
                vertices=ref_grid.vertices,
                features=x,
                memory_format=x_grid.memory_format,
                grid_shape=ref_grid.grid_shape,
                num_channels=x_grid.num_channels,
            )

        else:
            # Handle 2D factorized formats
            x = x_grid.features
            ref = ref_grid.features
            height, width = x.shape[2], x.shape[3]
            ref_height, ref_width = ref.shape[2], ref.shape[3]

            pad_height = ref_height - height
            pad_width = ref_width - width

            # Crop if needed
            if pad_height < 0 or pad_width < 0:
                x = x[:, :, :ref_height, :ref_width]

            # Pad if needed
            if pad_height > 0 or pad_width > 0:
                x = F.pad(x, (0, pad_width, 0, pad_height), "constant", 0)

            return GridFeatures(
                vertices=ref_grid.vertices,
                features=x,
                memory_format=x_grid.memory_format,
                grid_shape=ref_grid.grid_shape,
                num_channels=x_grid.num_channels,
            )


class GridFeatureConv2d(nn.Module):
    r"""2D convolution for factorized grid features.

    This module applies 2D convolution to factorized grid representations.
    It is equivalent to a 3D convolution with kernel size :math:`K \times K
    \times (2D + 1)`, where :math:`K` is the kernel size and :math:`D` is the
    size of the compressed (low resolution) dimension.

    This makes the convolution effectively global along the compressed dimension
    while being local along the high-resolution dimensions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    compressed_spatial_dim : int, optional, default=1
        Size of the compressed spatial dimension. This is multiplied with
        the channel dimension in the factorized representation.
    stride : int, optional, default=1
        Stride of the convolution.
    up_stride : int, optional, default=None
        If provided, uses transposed convolution for upsampling with this stride.
    padding : int, optional, default=None
        Padding for the convolution. Defaults to ``(kernel_size - 1) // 2``.
    output_padding : int, optional, default=None
        Output padding for transposed convolution.
    bias : bool, optional, default=True
        Whether to include a bias term.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features in a factorized memory format.

    Outputs
    -------
    GridFeatures
        Convolved grid features.

    Note
    ----
    The input must be in a factorized memory format (``b_zc_x_y``, ``b_xc_y_z``,
    or ``b_yc_x_z``), not in the standard 3D formats.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.figconvnet.geometries import GridFeatures, GridFeaturesMemoryFormat
    >>> conv = GridFeatureConv2d(in_channels=32, out_channels=64, kernel_size=3, compressed_spatial_dim=2)
    >>> # Input features in factorized format (B, compressed_dim * C, H, W)
    >>> vertices = torch.randn(2, 128, 128, 2, 3)
    >>> features = torch.randn(2, 64, 128, 128)  # 2 * 32 = 64 channels
    >>> gf = GridFeatures(vertices, features, GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2), 32)
    >>> out = conv(gf)
    >>> out.num_channels
    64
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dim: int = 1,
        stride: Optional[int] = 1,
        up_stride: Optional[int] = None,
        padding: Optional[int] = None,
        output_padding: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()

        if up_stride is None:
            # Standard convolution for downsampling or same-resolution
            self.conv = nn.Conv2d(
                in_channels=in_channels * compressed_spatial_dim,
                out_channels=out_channels * compressed_spatial_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if padding is not None else (kernel_size - 1) // 2,
                bias=bias,
            )
        else:
            # Transposed convolution for upsampling
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels * compressed_spatial_dim,
                out_channels=out_channels * compressed_spatial_dim,
                kernel_size=kernel_size,
                stride=up_stride,
                output_padding=output_padding if output_padding is not None else 0,
                bias=bias,
            )
        self.out_channels = out_channels

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        r"""Apply convolution to grid features.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features in factorized format.

        Returns
        -------
        GridFeatures
            Convolved grid features.
        """
        # Verify input is in a factorized format
        if (
            grid_features.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c
            or grid_features.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z
        ):
            raise ValueError(
                f"GridFeatureConv2d requires factorized memory format, "
                f"got {grid_features.memory_format}"
            )

        # Apply 2D convolution
        plane_view = grid_features.features
        plane_view = self.conv(plane_view)

        # Construct output GridFeatures
        out_grid_features = GridFeatures.from_conv_output(
            plane_view,
            grid_features.vertices,
            grid_features.memory_format,
            grid_features.grid_shape,
            self.out_channels,
        )
        return out_grid_features


class LayerNorm2d(nn.LayerNorm):
    r"""LayerNorm for 2D feature maps.

    Applies layer normalization to 2D feature maps by permuting to channels-last
    format, applying normalization, and permuting back.

    Parameters
    ----------
    normalized_shape : int or tuple
        Input shape from an expected input of size
        :math:`(*, normalized\_shape[0], normalized\_shape[1], ...)`.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Normalized tensor of shape :math:`(B, C, H, W)`.
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply layer normalization to 2D features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        # Permute to channels-last: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # Apply layer norm
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # Permute back: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        return x


class GridFeatureTransform(nn.Module):
    r"""Apply a transform to grid feature tensors.

    Wraps an arbitrary transform (e.g., normalization, activation) to operate
    on the feature tensors within GridFeatures objects.

    Parameters
    ----------
    transform : nn.Module
        Transform module to apply to the feature tensor.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features.

    Outputs
    -------
    GridFeatures
        Transformed grid features.

    Examples
    --------
    >>> import torch.nn as nn
    >>> transform = GridFeatureTransform(nn.GELU())
    """

    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.feature_transform = transform

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        r"""Apply transform to grid features.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features.

        Returns
        -------
        GridFeatures
            Transformed grid features.
        """
        if grid_features.memory_format == GridFeaturesMemoryFormat.b_x_y_z_c:
            raise ValueError(
                "b_x_y_z_c memory format is not supported for GridFeatureTransform"
            )

        if grid_features.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            batch_view = grid_features.batch_features
            if batch_view.ndim != 5:
                raise ValueError(f"Expected 5D batch_features, got {batch_view.ndim}D")
        else:
            batch_view = grid_features.features
            if batch_view.ndim == 3:
                raise ValueError(
                    "Unexpected 3D features tensor in GridFeatureTransform"
                )

        # Apply the transform
        batch_view = self.feature_transform(batch_view)

        # Construct output GridFeatures
        out_grid_features = GridFeatures.from_conv_output(
            batch_view,
            grid_features.vertices,
            grid_features.memory_format,
            grid_features.grid_shape,
            grid_features.num_channels,
        )
        return out_grid_features


class GridFeatureConv2dBlock(nn.Module):
    r"""Residual block with two GridFeatureConv2d layers.

    A standard residual block architecture consisting of:
    1. First convolution (with optional stride/upsampling)
    2. Normalization and activation
    3. Second convolution (same resolution)
    4. Normalization
    5. Skip connection (with optional projection)
    6. Final activation

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    compressed_spatial_dim : int, optional, default=1
        Size of compressed spatial dimension.
    stride : int, optional, default=1
        Downsampling stride for first convolution.
    up_stride : int, optional, default=None
        Upsampling stride for transposed convolution.
    apply_nonlinear_at_end : bool, optional, default=True
        Whether to apply activation after the skip connection.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features.

    Outputs
    -------
    GridFeatures
        Processed grid features.

    Note
    ----
    The skip connection is automatically adjusted via strided convolution
    or identity based on stride and channel dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dim: int = 1,
        stride: Optional[int] = 1,
        up_stride: Optional[int] = None,
        apply_nonlinear_at_end: bool = True,
    ):
        super().__init__()

        # First convolution: handles stride/upsampling
        self.conv1 = GridFeatureConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size if up_stride is None else up_stride,
            stride=stride,
            up_stride=up_stride,
            compressed_spatial_dim=compressed_spatial_dim,
        )

        # Second convolution: maintains resolution
        self.conv2 = GridFeatureConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            compressed_spatial_dim=compressed_spatial_dim,
            up_stride=None,
        )

        # Normalization layers
        self.norm1 = GridFeatureTransform(
            LayerNorm2d(out_channels * compressed_spatial_dim)
        )
        self.norm2 = GridFeatureTransform(
            LayerNorm2d(out_channels * compressed_spatial_dim)
        )
        self.apply_nonlinear_at_end = apply_nonlinear_at_end

        # Configure skip connection based on stride and channels
        if up_stride is None:
            if stride == 1 and in_channels == out_channels:
                # Identity shortcut
                self.shortcut = nn.Identity()
            elif stride == 1:
                # 1x1 conv for channel projection only
                self.shortcut = GridFeatureConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    compressed_spatial_dim=compressed_spatial_dim,
                )
            elif stride > 1:
                # Strided conv for downsampling
                self.shortcut = GridFeatureConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=stride,
                    stride=stride,
                    compressed_spatial_dim=compressed_spatial_dim,
                )
        else:
            # Transposed conv for upsampling
            self.shortcut = GridFeatureConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=up_stride,
                up_stride=up_stride,
                compressed_spatial_dim=compressed_spatial_dim,
            )

        self.pad_to_match = GridFeaturePadToMatch()
        self.nonlinear = GridFeatureTransform(nn.GELU())

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        r"""Apply residual block to grid features.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features.

        Returns
        -------
        GridFeatures
            Processed grid features.
        """
        # Main path: conv -> norm -> activation -> conv -> norm
        out = self.conv1(grid_features)
        out = self.nonlinear(self.norm1(out))
        out = self.norm2(self.conv2(out))

        # Shortcut path: project and pad to match
        shortcut = self.shortcut(grid_features)
        shortcut = self.pad_to_match(out, shortcut)

        # Residual connection
        out = out + shortcut

        # Optional final activation
        if self.apply_nonlinear_at_end:
            out = self.nonlinear(out)

        return out


class GridFeatureMemoryFormatConverter(nn.Module):
    r"""Convert grid features between memory formats.

    This module converts GridFeatures from their current memory format to
    a target format, enabling efficient transitions between factorized 2D
    representations and standard 3D representations.

    Parameters
    ----------
    memory_format : GridFeaturesMemoryFormat
        Target memory format.

    Forward
    -------
    grid_features : GridFeatures
        Input grid features in any memory format.

    Outputs
    -------
    GridFeatures
        Grid features converted to target format.

    Examples
    --------
    >>> from physicsnemo.models.figconvnet.geometries import GridFeaturesMemoryFormat
    >>> converter = GridFeatureMemoryFormatConverter(GridFeaturesMemoryFormat.b_x_y_z_c)
    """

    def __init__(self, memory_format: GridFeaturesMemoryFormat) -> None:
        super().__init__()
        self.memory_format = memory_format

    def __repr__(self):
        """Return string representation."""
        return f"GridFeatureMemoryFormatConverter(memory_format={self.memory_format})"

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        r"""Convert grid features to target memory format.

        Parameters
        ----------
        grid_features : GridFeatures
            Input grid features.

        Returns
        -------
        GridFeatures
            Grid features in target memory format.
        """
        return grid_features.to(memory_format=self.memory_format)
