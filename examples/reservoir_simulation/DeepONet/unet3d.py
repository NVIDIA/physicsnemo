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
3D U-Net Module for spatiotemporal data.

This module implements a 3D U-Net architecture for multi-scale feature extraction
from spatiotemporal data (Height × Width × Time). It can be used as a standalone
architecture or as a component in hybrid models like U-FNO.

The U-Net has:
- 3 downsampling layers (conv with stride=2)
- 3 upsampling layers (transposed conv)
- Skip connections at each level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UNet2D(nn.Module):
    """2D U-Net for spatial data (H × W)."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Encoder
        self.conv1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2_1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.conv3 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv3_1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        # Decoder
        self.deconv2 = self._deconv_block(input_channels, output_channels)
        self.deconv1 = self._deconv_block(input_channels * 2, output_channels)
        self.deconv0 = self._deconv_block(input_channels * 2, output_channels)

        # Output
        self.output_layer = self._output_block(
            input_channels * 2,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Module:
        """2D convolutional block."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def _deconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """2D transposed convolutional block."""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _output_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Module:
        """Output layer."""
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Input: (batch, channels, H, W)"""
        # Validate dimensions (must be divisible by 8)
        h, w = x.shape[2], x.shape[3]
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(
                f"Input dimensions ({h}, {w}) must be divisible by 8. Got shape: {x.shape}"
            )

        # Encoder
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        # Decoder with skip connections
        out_deconv2 = self.deconv2(out_conv3)
        if out_deconv2.shape[2:] != out_conv2.shape[2:]:
            out_deconv2 = F.interpolate(
                out_deconv2,
                size=out_conv2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)

        out_deconv1 = self.deconv1(concat2)
        if out_deconv1.shape[2:] != out_conv1.shape[2:]:
            out_deconv1 = F.interpolate(
                out_deconv1,
                size=out_conv1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)

        out_deconv0 = self.deconv0(concat1)
        if out_deconv0.shape[2:] != x.shape[2:]:
            out_deconv0 = F.interpolate(
                out_deconv0, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        concat0 = torch.cat((x, out_deconv0), dim=1)

        out = self.output_layer(concat0)

        return out

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNet3D(nn.Module):
    """3D U-Net for spatiotemporal data (H × W × T)."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Encoder
        self.conv1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2_1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.conv3 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv3_1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        # Decoder
        self.deconv2 = self._deconv_block(input_channels, output_channels)
        self.deconv1 = self._deconv_block(input_channels * 2, output_channels)
        self.deconv0 = self._deconv_block(input_channels * 2, output_channels)

        # Output
        self.output_layer = self._output_block(
            input_channels * 2,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Module:
        """3D convolutional block."""
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def _deconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """3D transposed convolutional block."""
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _output_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Module:
        """Output layer."""
        return nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Input: (batch, channels, H, W, T)"""
        # Validate dimensions (must be divisible by 8)
        h, w, t = x.shape[2], x.shape[3], x.shape[4]
        if h % 8 != 0 or w % 8 != 0 or t % 8 != 0:
            raise ValueError(
                f"Input dimensions ({h}, {w}, {t}) must be divisible by 8. Got shape: {x.shape}"
            )

        # Encoder
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        # Decoder with skip connections
        out_deconv2 = self.deconv2(out_conv3)
        if out_deconv2.shape[2:] != out_conv2.shape[2:]:
            out_deconv2 = F.interpolate(
                out_deconv2,
                size=out_conv2.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)

        out_deconv1 = self.deconv1(concat2)
        if out_deconv1.shape[2:] != out_conv1.shape[2:]:
            out_deconv1 = F.interpolate(
                out_deconv1,
                size=out_conv1.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)

        out_deconv0 = self.deconv0(concat1)
        if out_deconv0.shape[2:] != x.shape[2:]:
            out_deconv0 = F.interpolate(
                out_deconv0, size=x.shape[2:], mode="trilinear", align_corners=False
            )
        concat0 = torch.cat((x, out_deconv0), dim=1)

        out = self.output_layer(concat0)

        return out

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNet4D(nn.Module):
    """4D U-Net for higher-dimensional spatiotemporal data (H × W × T × D)."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Encoder
        self.conv1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2_1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.conv3 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv3_1 = self._conv_block(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        # Decoder
        self.deconv2 = self._deconv_block(input_channels, output_channels)
        self.deconv1 = self._deconv_block(input_channels * 2, output_channels)
        self.deconv0 = self._deconv_block(input_channels * 2, output_channels)

        # Output
        self.output_layer = self._output_block(
            input_channels * 2,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Module:
        """4D convolutional block."""
        return nn.Sequential(
            nn.Conv4d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm4d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def _deconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """4D transposed convolutional block."""
        return nn.Sequential(
            nn.ConvTranspose4d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _output_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Module:
        """Output layer."""
        return nn.Conv4d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Input: (batch, channels, H, W, T, D)"""
        # Validate dimensions (must be divisible by 8)
        dims = x.shape[2:]
        if any(d % 8 != 0 for d in dims):
            raise ValueError(
                f"Input dimensions {dims} must be divisible by 8. Got shape: {x.shape}"
            )

        # Encoder
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        # Decoder with skip connections
        out_deconv2 = self.deconv2(out_conv3)
        if out_deconv2.shape[2:] != out_conv2.shape[2:]:
            out_deconv2 = F.interpolate(
                out_deconv2, size=out_conv2.shape[2:], mode="nearest"
            )
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)

        out_deconv1 = self.deconv1(concat2)
        if out_deconv1.shape[2:] != out_conv1.shape[2:]:
            out_deconv1 = F.interpolate(
                out_deconv1, size=out_conv1.shape[2:], mode="nearest"
            )
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)

        out_deconv0 = self.deconv0(concat1)
        if out_deconv0.shape[2:] != x.shape[2:]:
            out_deconv0 = F.interpolate(out_deconv0, size=x.shape[2:], mode="nearest")
        concat0 = torch.cat((x, out_deconv0), dim=1)

        out = self.output_layer(concat0)

        return out

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Aliases for backward compatibility with U-FNO
UNetModule2D = UNet2D
UNetModule3D = UNet3D
UNetModule4D = UNet4D
