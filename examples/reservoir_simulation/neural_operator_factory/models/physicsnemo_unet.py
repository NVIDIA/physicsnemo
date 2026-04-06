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

"""
PhysicsNemo U-Net wrappers for 2D and 3D data.

This module provides wrappers around PhysicsNemo's 3D U-Net for:
- 2D spatial data (H × W) - used in DeepONet branch network
- 3D spatiotemporal data (H × W × T) - used in U-FNO and standalone models

PhysicsNemo's UNet is natively 3D, so the 2D wrapper adds/removes a dummy
temporal dimension to enable 2D processing.
"""

from typing import List, Optional

import torch.nn as nn
from torch import Tensor

from physicsnemo.models.unet import UNet as PhysicsNemoUNet

# ==============================================================================
# PhysicsNemo UNet Wrappers (2D and 3D)
# ==============================================================================


class PhysicsNemoUNet2D(nn.Module):
    """Wrapper to use PhysicsNemo's 3D UNet for 2D spatial data.

    This wrapper adds a dummy temporal dimension (T=1) to use PhysicsNemo's
    3D UNet architecture for 2D spatial processing (H × W).

    Use this for models that process spatial slices independently,
    such as DeepONet's branch network.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Convolution kernel size
    model_depth : int
        Depth of the U-Net (number of downsampling levels)
    feature_map_channels : list
        Number of channels at each depth level
    **kwargs
        Additional arguments passed to PhysicsNemo's UNet

    Example
    -------
    >>> unet = PhysicsNemoUNet2D(in_channels=64, out_channels=64)
    >>> x = torch.randn(4, 64, 104, 200)  # (B, C, H, W)
    >>> y = unet(x)  # (B, 64, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        model_depth: int = 3,
        feature_map_channels: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()

        if feature_map_channels is None:
            feature_map_channels = [in_channels] * model_depth

        self.unet = PhysicsNemoUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kwargs.get("stride", 1),
            model_depth=model_depth,
            feature_map_channels=feature_map_channels,
            num_conv_blocks=kwargs.get("num_conv_blocks", 1),
            conv_activation=kwargs.get("conv_activation", "leaky_relu"),
            conv_transpose_activation=kwargs.get(
                "conv_transpose_activation", "leaky_relu"
            ),
            padding=kwargs.get("padding", kernel_size // 2),
            padding_mode=kwargs.get("padding_mode", "zeros"),
            pooling_type=kwargs.get("pooling_type", "MaxPool3d"),
            pool_size=kwargs.get("pool_size", 2),
            normalization=kwargs.get("normalization", "batchnorm"),
            normalization_args=kwargs.get("normalization_args", None),
            use_attn_gate=kwargs.get("use_attn_gate", False),
            gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, H, W)

        Returns
        -------
        Tensor
            Output of shape (B, C, H, W)
        """
        # Add dummy temporal dimension: (B, C, H, W) -> (B, C, H, W, 1)
        x = x.unsqueeze(-1)

        # Forward through 3D UNet
        x = self.unet(x)

        # Remove temporal dimension: (B, C, H, W, 1) -> (B, C, H, W)
        x = x.squeeze(-1)

        return x


class PhysicsNemoUNet3D(nn.Module):
    """Wrapper for PhysicsNemo's 3D UNet for spatiotemporal data.

    This is a thin wrapper around PhysicsNemo's native 3D UNet for
    processing spatiotemporal data (H × W × T).

    Use this for models that process full spatiotemporal volumes,
    such as U-FNO or standalone U-Net models.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Convolution kernel size
    model_depth : int
        Depth of the U-Net (number of downsampling levels)
    feature_map_channels : list
        Number of channels at each depth level
    **kwargs
        Additional arguments passed to PhysicsNemo's UNet

    Example
    -------
    >>> unet = PhysicsNemoUNet3D(in_channels=12, out_channels=1)
    >>> x = torch.randn(4, 12, 104, 200, 24)  # (B, C, H, W, T)
    >>> y = unet(x)  # (B, 1, H, W, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        model_depth: int = 3,
        feature_map_channels: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()

        if feature_map_channels is None:
            feature_map_channels = [in_channels] * model_depth

        self.unet = PhysicsNemoUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kwargs.get("stride", 1),
            model_depth=model_depth,
            feature_map_channels=feature_map_channels,
            num_conv_blocks=kwargs.get("num_conv_blocks", 1),
            conv_activation=kwargs.get("conv_activation", "leaky_relu"),
            conv_transpose_activation=kwargs.get(
                "conv_transpose_activation", "leaky_relu"
            ),
            padding=kwargs.get("padding", kernel_size // 2),
            padding_mode=kwargs.get("padding_mode", "zeros"),
            pooling_type=kwargs.get("pooling_type", "MaxPool3d"),
            pool_size=kwargs.get("pool_size", 2),
            normalization=kwargs.get("normalization", "batchnorm"),
            normalization_args=kwargs.get("normalization_args", None),
            use_attn_gate=kwargs.get("use_attn_gate", False),
            gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, H, W, T)

        Returns
        -------
        Tensor
            Output of shape (B, C, H, W, T)
        """
        return self.unet(x)


# ==============================================================================
# Standalone U-Net for baseline comparisons
# ==============================================================================


class StandaloneUNet(nn.Module):
    """Standalone U-Net wrapper for spatiotemporal prediction (no Fourier layers).

    This provides a pure U-Net architecture without any spectral convolution layers,
    useful for comparison and baseline experiments.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    unet_type : str
        Type of UNet: "custom" (UNet3D) or "physicsnemo" (PhysicsNemo's UNet)
    **unet_kwargs
        Additional arguments passed to the UNet constructor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        unet_type: str = "custom",
        **unet_kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_type = unet_type.lower()

        # Build U-Net
        if self.unet_type == "custom":
            raise ValueError(
                "Custom UNet3D is only designed for use within U-FNO (where input_channels == output_channels). "
                "For standalone U-Net models, please use unet_type='physicsnemo' instead."
            )
        elif self.unet_type == "physicsnemo":
            self.unet = PhysicsNemoUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=unet_kwargs.get("kernel_size", 3),
                stride=unet_kwargs.get("stride", 1),
                model_depth=unet_kwargs.get("model_depth", 3),
                feature_map_channels=unet_kwargs.get(
                    "feature_map_channels", [36, 36, 36]
                ),
                num_conv_blocks=unet_kwargs.get("num_conv_blocks", 1),
                conv_activation=unet_kwargs.get("conv_activation", "relu"),
                conv_transpose_activation=unet_kwargs.get(
                    "conv_transpose_activation", "relu"
                ),
                padding=unet_kwargs.get("padding", 1),
                padding_mode=unet_kwargs.get("padding_mode", "zeros"),
                pooling_type=unet_kwargs.get("pooling_type", "MaxPool3d"),
                pool_size=unet_kwargs.get("pool_size", 2),
                normalization=unet_kwargs.get("normalization", "batchnorm"),
                normalization_args=unet_kwargs.get("normalization_args", None),
                use_attn_gate=unet_kwargs.get("use_attn_gate", False),
                gradient_checkpointing=unet_kwargs.get("gradient_checkpointing", False),
            )
        else:
            raise ValueError(
                f"Unknown unet_type: {self.unet_type}. Use 'custom' or 'physicsnemo'"
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, height, width, time, channels)

        Returns
        -------
        Tensor
            Output tensor of shape (batch, height, width, time)
        """
        # Input: (B, H, W, T, C)
        # Permute to (B, C, H, W, T) for 3D convolution
        x = x.permute(0, 4, 1, 2, 3)

        # U-Net forward pass
        x = self.unet(x)

        # Permute back: (B, out_channels, H, W, T) -> (B, H, W, T, out_channels)
        x = x.permute(0, 2, 3, 4, 1)

        # Squeeze out channel dimension: (B, H, W, T, 1) -> (B, H, W, T)
        return x.squeeze(-1)

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
