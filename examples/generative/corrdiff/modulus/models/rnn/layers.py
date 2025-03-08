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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _get_same_padding(x: int, k: int, s: int) -> int:
    """Function to compute "same" padding. Inspired from:
    https://github.com/huggingface/pytorch-image-models/blob/0.5.x/timm/models/layers/padding.py
    """
    return max(s * math.ceil(x / s) - s - x + k, 0)


class _ConvLayer(nn.Module):
    """Generalized Convolution Block

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    dimension : int
        Dimensionality of the input, 1, 2, 3, or 4
    kernel_size : int
        Kernel size for the convolution
    stride : int
        Stride for the convolution, by default 1
    activation_fn : nn.Module, optional
        Activation function to use, by default nn.Identity()
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,  # TODO check if there are ways to infer this
        kernel_size: int,
        stride: int = 1,
        activation_fn: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dimension = dimension
        self.activation_fn = activation_fn

        if self.dimension == 1:
            self.conv = nn.Conv1d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif self.dimension == 2:
            self.conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif self.dimension == 3:
            self.conv = nn.Conv3d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        else:
            raise ValueError("Only 1D, 2D and 3D dimensions are supported")

        self.reset_parameters()

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def reset_parameters(self) -> None:
        """Initialization for network parameters"""
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        input_length = len(x.size()) - 2  # exclude channel and batch dims
        if input_length != self.dimension:
            raise ValueError("Input dimension not compatible")

        if input_length == 1:
            iw = x.size()[-1:][0]
            pad_w = _get_same_padding(iw, self.kernel_size, self.stride)
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2], mode="constant", value=0.0)
        elif input_length == 2:
            ih, iw = x.size()[-2:]
            pad_h, pad_w = _get_same_padding(
                ih, self.kernel_size, self.stride
            ), _get_same_padding(iw, self.kernel_size, self.stride)
            x = F.pad(
                x,
                [pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2],
                mode="constant",
                value=0.0,
            )
        else:
            _id, ih, iw = x.size()[-3:]
            pad_d, pad_h, pad_w = (
                _get_same_padding(_id, self.kernel_size, self.stride),
                _get_same_padding(ih, self.kernel_size, self.stride),
                _get_same_padding(iw, self.kernel_size, self.stride),
            )
            x = F.pad(
                x,
                [
                    pad_d // 2,
                    pad_d - pad_d // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                    pad_w // 2,
                    pad_w - pad_w // 2,
                ],
                mode="constant",
                value=0.0,
            )

        x = self.conv(x)

        if self.activation_fn is not nn.Identity():
            x = self.exec_activation_fn(x)

        return x


class _TransposeConvLayer(nn.Module):
    """Generalized Transposed Convolution Block

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    dimension : int
        Dimensionality of the input, 1, 2, 3, or 4
    kernel_size : int
        Kernel size for the convolution
    stride : int
        Stride for the convolution, by default 1
    activation_fn : nn.Module, optional
        Activation function to use, by default nn.Identity()
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int,
        stride: int = 1,
        activation_fn=nn.Identity(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dimension = dimension
        self.activation_fn = activation_fn

        if dimension == 1:
            self.trans_conv = nn.ConvTranspose1d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif dimension == 2:
            self.trans_conv = nn.ConvTranspose2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        elif dimension == 3:
            self.trans_conv = nn.ConvTranspose3d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                bias=True,
            )
        else:
            raise ValueError("Only 1D, 2D and 3D dimensions are supported")

        self.reset_parameters()

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def reset_parameters(self) -> None:
        """Initialization for network parameters"""
        nn.init.constant_(self.trans_conv.bias, 0)
        nn.init.xavier_uniform_(self.trans_conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        orig_x = x
        input_length = len(orig_x.size()) - 2  # exclude channel and batch dims
        if input_length != self.dimension:
            raise ValueError("Input dimension not compatible")

        x = self.trans_conv(x)

        if input_length == 1:
            iw = orig_x.size()[-1:][0]
            pad_w = _get_same_padding(iw, self.kernel_size, self.stride)
            x = x[
                :,
                :,
                pad_w // 2 : x.size(-1) - (pad_w - pad_w // 2),
            ]
        elif input_length == 2:
            ih, iw = orig_x.size()[-2:]
            pad_h, pad_w = _get_same_padding(
                ih,
                self.kernel_size,
                self.stride,
            ), _get_same_padding(iw, self.kernel_size, self.stride)
            x = x[
                :,
                :,
                pad_h // 2 : x.size(-2) - (pad_h - pad_h // 2),
                pad_w // 2 : x.size(-1) - (pad_w - pad_w // 2),
            ]
        else:
            _id, ih, iw = orig_x.size()[-3:]
            pad_d, pad_h, pad_w = (
                _get_same_padding(_id, self.kernel_size, self.stride),
                _get_same_padding(ih, self.kernel_size, self.stride),
                _get_same_padding(iw, self.kernel_size, self.stride),
            )
            x = x[
                :,
                :,
                pad_d // 2 : x.size(-3) - (pad_d - pad_d // 2),
                pad_h // 2 : x.size(-2) - (pad_h - pad_h // 2),
                pad_w // 2 : x.size(-1) - (pad_w - pad_w // 2),
            ]

        if self.activation_fn is not nn.Identity():
            x = self.exec_activation_fn(x)

        return x


class _ConvGRULayer(nn.Module):
    """Convolutional GRU layer

    Parameters
    ----------
    in_features : int
        Input features/channels
    hidden_size : int
        Hidden layer features/channels
    dimension : int
        Spatial dimension of the input
    activation_fn : nn.Module, optional
        Activation Function to use, by default nn.ReLU()
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        dimension: int,
        activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.conv_1 = _ConvLayer(
            in_channels=in_features + hidden_size,
            out_channels=2 * hidden_size,
            kernel_size=3,
            stride=1,
            dimension=dimension,
        )
        self.conv_2 = _ConvLayer(
            in_channels=in_features + hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            dimension=dimension,
        )

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        concat = torch.cat((x, hidden), dim=1)
        conv_concat = self.conv_1(concat)
        conv_r, conv_z = torch.split(conv_concat, self.hidden_size, 1)

        reset_gate = torch.special.expit(conv_r)
        update_gate = torch.special.expit(conv_z)
        concat = torch.cat((x, torch.mul(hidden, reset_gate)), dim=1)
        n = self.exec_activation_fn(self.conv_2(concat))
        h_next = torch.mul((1 - update_gate), n) + torch.mul(update_gate, hidden)

        return h_next


class _ConvResidualBlock(nn.Module):
    """Convolutional ResNet Block

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    dimension : int
        Dimensionality of the input
    stride : int
        Stride of the convolutions, by default 1
    gated : bool, optional
        Residual Gate, by default False
    layer_normalization : bool, optional
        Layer Normalization, by default False
    begin_activation_fn : bool, optional
        Whether to use activation function in the beginning, by default True
    activation_fn : nn.Module, optional
        Activation function to use, by default nn.ReLU()

    Raises
    ------
    ValueError
        Stride not supported
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        stride: int = 1,
        gated: bool = False,
        layer_normalization: bool = False,
        begin_activation_fn: bool = True,
        activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dimension = dimension
        self.gated = gated
        self.layer_normalization = layer_normalization
        self.begin_activation_fn = begin_activation_fn
        self.activation_fn = activation_fn

        if self.stride == 1:
            self.conv_1 = _ConvLayer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stride,
                dimension=self.dimension,
            )
        elif self.stride == 2:
            self.conv_1 = _ConvLayer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=self.stride,
                dimension=self.dimension,
            )
        else:
            raise ValueError("stride > 2 is not supported")

        if not self.gated:
            self.conv_2 = _ConvLayer(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                dimension=self.dimension,
            )
        else:
            self.conv_2 = _ConvLayer(
                in_channels=self.out_channels,
                out_channels=2 * self.out_channels,
                kernel_size=3,
                stride=1,
                dimension=self.dimension,
            )

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        """Executes activation function on the input"""
        return self.activation_fn(x)

    def forward(self, x: Tensor) -> Tensor:
        orig_x = x

        if self.begin_activation_fn:
            if self.layer_normalization:
                layer_norm = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
                x = layer_norm(x)
            x = self.exec_activation_fn(x)

        # first convolutional layer
        x = self.conv_1(x)

        # add layer normalization
        if self.layer_normalization:
            layer_norm = nn.LayerNorm(x.size()[1:], elementwise_affine=False)
            x = layer_norm(x)

        # second activation
        x = self.exec_activation_fn(x)
        # second convolutional layer
        x = self.conv_2(x)
        if self.gated:
            x_1, x_2 = torch.split(x, x.size(1) // 2, 1)
            x = x_1 * torch.special.expit(x_2)

        # possibly reshape skip connection
        if orig_x.size(-1) > x.size(-1):  # Check if widths are same)
            if len(orig_x.size()) - 2 == 1:
                iw = orig_x.size()[-1:][0]
                pad_w = _get_same_padding(iw, 2, 2)
                pool = torch.nn.AvgPool1d(
                    2, 2, padding=pad_w // 2, count_include_pad=False
                )
            elif len(orig_x.size()) - 2 == 2:
                ih, iw = orig_x.size()[-2:]
                pad_h, pad_w = _get_same_padding(
                    ih,
                    2,
                    2,
                ), _get_same_padding(iw, 2, 2)
                pool = torch.nn.AvgPool2d(
                    2, 2, padding=(pad_h // 2, pad_w // 2), count_include_pad=False
                )
            elif len(orig_x.size()) - 2 == 3:
                _id, ih, iw = orig_x.size()[-3:]
                pad_d, pad_h, pad_w = (
                    _get_same_padding(_id, 2, 2),
                    _get_same_padding(ih, 2, 2),
                    _get_same_padding(iw, 2, 2),
                )
                pool = torch.nn.AvgPool3d(
                    2,
                    2,
                    padding=(pad_d // 2, pad_h // 2, pad_w // 2),
                    count_include_pad=False,
                )
            else:
                raise ValueError("Only 1D, 2D and 3D dimensions are supported")
            orig_x = pool(orig_x)

        # possibly change the channels for skip connection
        in_channels = int(orig_x.size(1))
        if self.out_channels > in_channels:
            orig_x = F.pad(
                orig_x,
                (len(orig_x.size()) - 2) * (0, 0)
                + (self.out_channels - self.in_channels, 0),
            )
        elif self.out_channels < in_channels:
            pass

        return orig_x + x
