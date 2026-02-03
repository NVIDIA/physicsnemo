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

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import (
    ConvGRULayer,
    ConvLayer,
    ConvResidualBlock,
    TransposeConvLayer,
    get_activation,
)


@dataclass
class MetaData(ModelMetaData):
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


class One2ManyRNN(Module):
    r"""
    A RNN model with encoder/decoder for 2D/3D problems that provides predictions
    based on single initial condition.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input.
    dimension : int, optional, default=2
        Spatial dimension of the input. Only 2D and 3D are supported.
    nr_latent_channels : int, optional, default=512
        Channels for encoding/decoding.
    nr_residual_blocks : int, optional, default=2
        Number of residual blocks.
    activation_fn : str, optional, default="relu"
        Activation function to use.
    nr_downsamples : int, optional, default=2
        Number of downsamples.
    nr_tsteps : int, optional, default=32
        Time steps to predict.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(N, C, 1, H, W)` for 2D or
        :math:`(N, C, 1, D, H, W)` for 3D, where :math:`N` is the batch size,
        :math:`C` is the number of channels, ``1`` is the number of input
        timesteps, and :math:`D, H, W` are spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, C, T, H, W)` for 2D or
        :math:`(N, C, T, D, H, W)` for 3D, where :math:`T` is the number of
        timesteps being predicted.

    Examples
    --------
    >>> import torch
    >>> import physicsnemo
    >>> model = physicsnemo.models.rnn.One2ManyRNN(
    ...     input_channels=6,
    ...     dimension=2,
    ...     nr_latent_channels=32,
    ...     activation_fn="relu",
    ...     nr_downsamples=2,
    ...     nr_tsteps=16,
    ... )
    >>> input_tensor = torch.randn(4, 6, 1, 16, 16)  # [N, C, T, H, W]
    >>> output = model(input_tensor)
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
                    ConvResidualBlock(
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

        self.rnn_layer = ConvGRULayer(
            in_features=channels_out, hidden_size=channels_out, dimension=dimension
        )

        self.conv_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(nr_downsamples):
            self.upsampling_layers = nn.ModuleList()
            channels_in = channels_out
            channels_out = channels_out // 2
            self.upsampling_layers.append(
                TransposeConvLayer(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size=4,
                    stride=2,
                    dimension=dimension,
                )
            )
            for j in range(nr_residual_blocks):
                self.upsampling_layers.append(
                    ConvResidualBlock(
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
                ConvLayer(
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

    def forward(
        self,
        x: Float[Tensor, "batch channels 1 ..."],  # noqa: F722
    ) -> Float[Tensor, "batch channels tsteps ..."]:  # noqa: F722
        r"""
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Expects a tensor of shape :math:`(N, C, 1, H, W)` for 2D or
            :math:`(N, C, 1, D, H, W)` for 3D. Where :math:`N` is the batch size,
            :math:`C` is the number of channels, ``1`` is the number of input
            timesteps, and :math:`D, H, W` are spatial dimensions.

        Returns
        -------
        torch.Tensor
            Size :math:`(N, C, T, H, W)` for 2D or :math:`(N, C, T, D, H, W)` for 3D,
            where :math:`T` is the number of timesteps being predicted.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            # Check number of dimensions
            expected_ndim = 5 if self.encoder_layers[0].dimension == 2 else 6
            if x.ndim != expected_ndim:
                raise ValueError(
                    f"Expected {expected_ndim}D input tensor, "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

            # Check time dimension is 1
            if x.shape[2] != 1:
                raise ValueError(
                    f"Expected input with 1 timestep (dimension 2), "
                    f"got {x.shape[2]} timesteps in tensor with shape {tuple(x.shape)}"
                )

        # Encoding step - encode the single input timestep
        encoded_inputs = []
        for t in range(1):
            x_in = x[:, :, t, ...]  # (B, C, *spatial)
            # Pass through encoder layers
            for layer in self.encoder_layers:
                x_in = layer(x_in)
            encoded_inputs.append(x_in)

        # RNN step - autoregressively generate future timesteps
        rnn_output = []
        for t in range(self.nr_tsteps):
            if t == 0:
                # Initialize hidden state to zeros
                h = torch.zeros(list(x_in.size())).to(
                    x.device
                )  # (B, C_latent, *spatial)
                x_in_rnn = encoded_inputs[0]
            # Update hidden state
            h = self.rnn_layer(x_in_rnn, h)  # (B, C_latent, *spatial)
            x_in_rnn = h
            rnn_output.append(h)

        # Decoding step - decode each hidden state to output
        decoded_output = []
        for t in range(self.nr_tsteps):
            x_out = rnn_output[t]  # (B, C_latent, *spatial)

            # Multi-resolution decoding with skip connections
            latent_context_grid = []
            for conv_layer, decoder in zip(self.conv_layers, self.decoder_layers):
                latent_context_grid.append(conv_layer(x_out))
                upsampling_layers = decoder
                # Progressively upsample
                for upsampling_layer in upsampling_layers:
                    x_out = upsampling_layer(x_out)

            # Final convolution to match output channels
            # Only last latent context grid is used, but multi-resolution is available
            out = self.final_conv(latent_context_grid[-1])  # (B, C, *spatial)
            decoded_output.append(out)

        # Stack outputs along time dimension
        decoded_output = torch.stack(decoded_output, dim=2)  # (B, C, T, *spatial)
        return decoded_output
