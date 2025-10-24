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

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
from physicsnemo.models.layers import get_activation
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
from physicsnemo.models.rnn.layers import (
    _ConvGRULayer,
    _ConvLayer,
    _ConvResidualBlock,
    _TransposeConvLayer,
)


@dataclass
class MetaData(ModelMetaData):
    name: str = "Seq2SeqRNN"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    torch_fx: bool = True
    # Inference
    onnx: bool = False
    onnx_runtime: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class Seq2SeqRNN(Module):
    """A RNN model with encoder/decoder for 2d/3d problems. Given input 0 to t-1,
    predicts signal t to t + nr_tsteps

    Parameters
    ----------
    input_channels : int
        Number of channels in the input
    dimension : int, optional
        Spatial dimension of the input. Only 2d and 3d are supported, by default 2
    nr_latent_channels : int, optional
        Channels for encoding/decoding, by default 512
    nr_residual_blocks : int, optional
        Number of residual blocks, by default 2
    activation_fn : str, optional
        Activation function to use, by default "relu"
    nr_downsamples : int, optional
        Number of downsamples, by default 2
    nr_tsteps : int, optional
        Time steps to predict, by default 32

    Example
    -------
    >>> model = physicsnemo.models.rnn.Seq2SeqRNN(
    ... input_channels=6,
    ... dimension=2,
    ... nr_latent_channels=32,
    ... activation_fn="relu",
    ... nr_downsamples=2,
    ... nr_tsteps=16,
    ... )
    >>> input = invar = torch.randn(4, 6, 16, 16, 16) # [N, C, T, H, W]
    >>> output = model(input)
    >>> output.size()
    torch.Size([4, 6, 16, 16, 16])
    """

    def __init__(
        self,
        input_channels: int,
        dimension: int = 2,
        nr_latent_channels: int = 512,
        nr_residual_blocks: int = 2,
        activation_fn: str = "relu",
        nr_downsamples: int = 2,
        nr_tsteps: int = 32,
    ) -> None:
        super().__init__(meta=MetaData())

        self.nr_tsteps = nr_tsteps
        self.nr_residual_blocks = nr_residual_blocks
        self.nr_downsamples = nr_downsamples
        self.encoder_layers = nn.ModuleList()
        channels_out = nr_latent_channels
        activation_fn = get_activation(activation_fn)

        # check valid dimensions
        if dimension not in [2, 3]:
            raise ValueError("Only 2D and 3D spatial dimensions are supported")

        for i in range(nr_downsamples):
            for j in range(nr_residual_blocks):
                stride = 1
                if i == 0 and j == 0:
                    channels_in = input_channels
                else:
                    channels_in = channels_out
                if (j == nr_residual_blocks - 1) and (i < nr_downsamples - 1):
                    channels_out = channels_out * 2
                    stride = 2
                self.encoder_layers.append(
                    _ConvResidualBlock(
                        in_channels=channels_in,
                        out_channels=channels_out,
                        stride=stride,
                        dimension=dimension,
                        gated=True,
                        layer_normalization=False,
                        begin_activation_fn=not ((i == 0) and (j == 0)),
                        activation_fn=activation_fn,
                    )
                )

        self.rnn_layer = _ConvGRULayer(
            in_features=channels_out, hidden_size=channels_out, dimension=dimension
        )

        self.conv_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(nr_downsamples):
            self.upsampling_layers = nn.ModuleList()
            channels_in = channels_out
            channels_out = channels_out // 2
            self.upsampling_layers.append(
                _TransposeConvLayer(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size=4,
                    stride=2,
                    dimension=dimension,
                )
            )
            for j in range(nr_residual_blocks):
                self.upsampling_layers.append(
                    _ConvResidualBlock(
                        in_channels=channels_out,
                        out_channels=channels_out,
                        stride=1,
                        dimension=dimension,
                        gated=True,
                        layer_normalization=False,
                        begin_activation_fn=not ((i == 0) and (j == 0)),
                        activation_fn=activation_fn,
                    )
                )
            self.conv_layers.append(
                _ConvLayer(
                    in_channels=channels_in,
                    out_channels=nr_latent_channels,
                    kernel_size=1,
                    stride=1,
                    dimension=dimension,
                )
            )
            self.decoder_layers.append(self.upsampling_layers)

        if dimension == 2:
            self.final_conv = nn.Conv2d(
                nr_latent_channels, input_channels, (1, 1), (1, 1), padding="valid"
            )
        else:
            # dimension is 3
            self.final_conv = nn.Conv3d(
                nr_latent_channels,
                input_channels,
                (1, 1, 1),
                (1, 1, 1),
                padding="valid",
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Parameters
        ----------
        x : Tensor
            Expects a tensor of size [N, C, T, H, W] for 2D or [N, C, T, D, H, W] for 3D
            Where, N is the batch size, C is the number of channels, T is the number of
            input timesteps and D, H, W are spatial dimensions. Currently, this
            requires input time steps to be same as predicted time steps.
        Returns
        -------
        Tensor
            Size [N, C, T, H, W] for 2D or [N, C, T, D, H, W] for 3D.
            Where, T is the number of timesteps being predicted.
        """
        # Encoding step
        encoded_inputs = []
        for t in range(self.nr_tsteps):
            x_in = x[:, :, t, ...]
            for layer in self.encoder_layers:
                x_in = layer(x_in)
            encoded_inputs.append(x_in)

        # RNN step
        # encode
        for t in range(x.size(2)):  # time dimension of the input signal
            if t == 0:
                h = torch.zeros(list(x_in.size())).to(x.device)
            x_in_rnn = encoded_inputs[t]
            h = self.rnn_layer(x_in_rnn, h)

        # decode
        rnn_output = []
        for t in range(self.nr_tsteps):
            if t == 0:
                x_in_rnn = encoded_inputs[-1]
            h = self.rnn_layer(x_in_rnn, h)
            x_in_rnn = h
            rnn_output.append(h)

        decoded_output = []
        for t in range(self.nr_tsteps):
            x_out = rnn_output[t]
            # Decoding step
            latent_context_grid = []
            for conv_layer, decoder in zip(self.conv_layers, self.decoder_layers):
                latent_context_grid.append(conv_layer(x_out))
                upsampling_layers = decoder
                for upsampling_layer in upsampling_layers:
                    x_out = upsampling_layer(x_out)

            # Add a convolution here to make the channel dimensions same as output
            # Only last latent context grid is used, but mult-resolution is available
            out = self.final_conv(latent_context_grid[-1])
            decoded_output.append(out)

        decoded_output = torch.stack(decoded_output, dim=2)
        return decoded_output
