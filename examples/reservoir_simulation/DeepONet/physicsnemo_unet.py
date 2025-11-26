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

"""
PhysicsNemo U-Net wrapper for spatiotemporal prediction.

This module provides a pure U-Net architecture (either custom or PhysicsNemo's implementation)
without any spectral convolution layers, useful for baseline comparisons and ablation studies.
"""

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.models.unet import UNet as PhysicsNemoUNet
from unet3d import UNetModule3D


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
