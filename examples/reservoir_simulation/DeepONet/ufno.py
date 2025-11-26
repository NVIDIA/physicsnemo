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
U-FNO: Enhanced Fourier Neural Operator with U-Net or Conv skip connections.

Reference:
    Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., & Benson, S. M. (2022).
    U-FNO--An enhanced Fourier neural operator-based deep-learning model for multiphase flow.
    Advances in Water Resources, 104180.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from physicsnemo.models.module import Module
from physicsnemo.models.layers import (
    SpectralConv3d,
    ConvNdKernel1Layer,
    get_activation,
    Conv3dFCLayer,
)
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.unet import UNet as PhysicsNemoUNet

from unet3d import UNetModule3D
from physicsnemo_unet import StandaloneUNet


class UFNO(Module):
    """U-FNO/Conv-FNO: Fourier Neural Operator enhanced with U-Net or Conv modules.

    Architecture consists of:
    - Lifting network to project input to latent space
    - num_fno_layers standard Fourier layers (Spectral Conv + 1x1 Conv)
    - num_unet_layers enhanced layers (Spectral Conv + 1x1 Conv + U-Net)
    - num_conv_layers conv-enhanced layers (Spectral Conv + 1x1 Conv + 3D Conv)
    - Decoder network to project latent space to output

    Note:
    - U-FNO: num_unet_layers > 0, num_conv_layers = 0
    - Conv-FNO: num_unet_layers = 0, num_conv_layers > 0
    - Standard FNO: num_unet_layers = 0, num_conv_layers = 0

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    width : int
        Latent channel dimension
    modes1, modes2, modes3 : int
        Number of Fourier modes in each dimension
    num_fno_layers : int
        Number of standard Fourier layers
    num_unet_layers : int
        Number of U-Net enhanced layers
    num_conv_layers : int
        Number of Conv enhanced layers (Conv-FNO)
    conv_kernel_size : int
        Kernel size for Conv layers (Conv-FNO)
    unet_kernel_size : int
        Kernel size for U-Net convolutions
    unet_dropout : float
        Dropout rate in U-Net
    unet_type : str
        Type of UNet: "custom" (your UNet3D) or "physicsnemo" (PhysicsNemo's UNet)
    activation_fn : str
        Activation function name
    lifting_type : str
        Type of lifting layer: "mlp" or "conv"
    lifting_layers : int
        Number of layers in lifting network
    lifting_width : int
        Hidden width factor for multi-layer lifting
    decoder_type : str
        Type of decoder: "mlp" or "conv"
    decoder_layers : int
        Number of hidden layers in decoder
    decoder_width : int
        Hidden layer size in decoder
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 36,
        modes1: int = 10,
        modes2: int = 10,
        modes3: int = 10,
        num_fno_layers: int = 3,
        num_unet_layers: int = 3,
        num_conv_layers: int = 0,
        conv_kernel_size: int = 3,
        unet_kernel_size: int = 3,
        unet_dropout: float = 0.0,
        unet_type: str = "custom",
        activation_fn: str = "relu",
        lifting_type: str = "mlp",
        lifting_layers: int = 1,
        lifting_width: int = 2,
        decoder_type: str = "mlp",
        decoder_layers: int = 1,
        decoder_width: int = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.num_fno_layers = num_fno_layers
        self.num_unet_layers = num_unet_layers
        self.num_conv_layers = num_conv_layers
        self.total_layers = num_fno_layers + num_unet_layers + num_conv_layers
        self.lifting_type = lifting_type.lower()
        self.decoder_type = decoder_type.lower()
        self.unet_type = unet_type.lower()
        self.activation_fn_name = activation_fn
        self.activation_fn = get_activation(activation_fn)
        self.conv_kernel_size = conv_kernel_size

        # Build lifting network
        self.lift_network = self._build_lifting_network(
            in_channels, width, lifting_layers, lifting_width, lifting_type
        )

        # Build Fourier layers
        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()

        for _ in range(self.total_layers):
            self.spectral_convs.append(
                SpectralConv3d(self.width, self.width, modes1, modes2, modes3)
            )
            self.conv_1x1s.append(ConvNdKernel1Layer(self.width, self.width))

        # Build U-Net modules
        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            if self.unet_type == "custom":
                # Use custom UNet3D
                self.unet_modules.append(
                    UNetModule3D(
                        self.width,
                        self.width,
                        kernel_size=unet_kernel_size,
                        dropout_rate=unet_dropout,
                    )
                )
            elif self.unet_type == "physicsnemo":
                # Use PhysicsNemo's UNet - configured to be as close as possible to custom UNet3D
                self.unet_modules.append(
                    PhysicsNemoUNet(
                        in_channels=self.width,
                        out_channels=self.width,
                        kernel_size=unet_kernel_size,
                        stride=1,
                        model_depth=3,  # 3 downsampling levels like custom UNet3D
                        feature_map_channels=[self.width]
                        * 3,  # model_depth * num_conv_blocks = 3 * 1
                        num_conv_blocks=1,  # Simpler blocks
                        conv_activation="relu",  # Closest to LeakyReLU
                        conv_transpose_activation="relu",
                        padding=1,
                        padding_mode="zeros",
                        pooling_type="MaxPool3d",
                        pool_size=2,
                        normalization="batchnorm",  # Match custom UNet3D's BatchNorm
                        normalization_args=None,
                        use_attn_gate=False,  # No attention gates
                        gradient_checkpointing=False,  # Disable for fair comparison
                    )
                )
            else:
                raise ValueError(
                    f"Unknown unet_type: {self.unet_type}. Use 'custom' or 'physicsnemo'"
                )

        # Build Conv modules (for Conv-FNO)
        self.conv_modules = nn.ModuleList()
        for _ in range(num_conv_layers):
            # Simple 3D convolution with activation
            padding = (conv_kernel_size - 1) // 2
            self.conv_modules.append(
                nn.Sequential(
                    nn.Conv3d(
                        self.width,
                        self.width,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm3d(self.width),
                    self.activation_fn,
                )
            )

        # Build decoder network
        self.decoder = self._build_decoder_network(
            width, out_channels, decoder_layers, decoder_width, decoder_type
        )

    def _build_lifting_network(
        self,
        in_channels: int,
        width: int,
        num_layers: int,
        hidden_width_factor: int,
        lift_type: str,
    ) -> nn.Module:
        """Build lifting network to project input to latent space."""
        if lift_type == "mlp":
            if num_layers == 1:
                return nn.Linear(in_channels, width)
            else:
                return FullyConnected(
                    in_features=in_channels,
                    layer_size=width // hidden_width_factor,
                    out_features=width,
                    num_layers=num_layers,
                    activation_fn=self.activation_fn_name,
                )
        elif lift_type == "conv":
            if num_layers == 1:
                return Conv3dFCLayer(in_channels, width)
            else:
                layers_list = []
                hidden_width = width // hidden_width_factor
                layers_list.append(Conv3dFCLayer(in_channels, hidden_width))
                layers_list.append(self.activation_fn)
                for _ in range(num_layers - 2):
                    layers_list.append(Conv3dFCLayer(hidden_width, hidden_width))
                    layers_list.append(self.activation_fn)
                layers_list.append(Conv3dFCLayer(hidden_width, width))
                return nn.Sequential(*layers_list)
        else:
            raise ValueError(f"Unknown lifting_type: {lift_type}. Use 'mlp' or 'conv'")

    def _build_decoder_network(
        self,
        width: int,
        out_channels: int,
        num_layers: int,
        hidden_width: int,
        decoder_type: str,
    ) -> nn.Module:
        """Build decoder network to project latent space to output."""
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            else:
                return FullyConnected(
                    in_features=width,
                    layer_size=hidden_width,
                    out_features=out_channels,
                    num_layers=num_layers,
                    activation_fn=self.activation_fn_name,
                )
        elif decoder_type == "conv":
            if num_layers == 0:
                return Conv3dFCLayer(width, out_channels)
            else:
                layers_list = []
                for _ in range(num_layers):
                    layers_list.append(Conv3dFCLayer(width, hidden_width))
                    layers_list.append(self.activation_fn)
                    width = hidden_width
                layers_list.append(Conv3dFCLayer(hidden_width, out_channels))
                return nn.Sequential(*layers_list)
        else:
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. Use 'mlp' or 'conv'"
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through U-FNO."""
        # Lifting
        if self.lifting_type == "mlp":
            x = self.lift_network(x)
            x = x.permute(0, 4, 1, 2, 3)
        else:  # conv
            # Conv lifting: need channel-first input
            # (batch, H, W, T, in_channels) -> (batch, in_channels, H, W, T)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.lift_network(x)

        # Standard Fourier layers
        for layer_idx in range(self.num_fno_layers):
            x1 = self.spectral_convs[layer_idx](x)
            x2 = self.conv_1x1s[layer_idx](x)
            x = x1 + x2
            x = self.activation_fn(x)

        # Fourier + U-Net layers
        for unet_idx in range(self.num_unet_layers):
            layer_idx = self.num_fno_layers + unet_idx
            x1 = self.spectral_convs[layer_idx](x)
            x2 = self.conv_1x1s[layer_idx](x)
            x3 = self.unet_modules[unet_idx](x)
            x = x1 + x2 + x3
            x = self.activation_fn(x)

        # Fourier + Conv layers (Conv-FNO)
        for conv_idx in range(self.num_conv_layers):
            layer_idx = self.num_fno_layers + self.num_unet_layers + conv_idx
            x1 = self.spectral_convs[layer_idx](x)
            x2 = self.conv_1x1s[layer_idx](x)
            x3 = self.conv_modules[conv_idx](x)
            x = x1 + x2 + x3
            x = self.activation_fn(x)

        # Decoder: project back to output space
        if self.decoder_type == "mlp":
            # MLP decoder: need channel-last for pointwise operations
            # (batch, width, H, W, T) -> (batch, H, W, T, width)
            x = x.permute(0, 2, 3, 4, 1)
            x = self.decoder(x)  # (batch, H, W, T, out_channels)
        else:  # conv
            # Conv decoder: already in channel-first format
            x = self.decoder(x)  # (batch, out_channels, H, W, T)
            # Convert to channel-last for consistency
            x = x.permute(0, 2, 3, 4, 1)  # (batch, H, W, T, out_channels)

        return x

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UFNONet(nn.Module):
    """Wrapper for UFNO that handles padding/de-padding."""

    def __init__(
        self,
        modes1: int,
        modes2: int,
        modes3: int,
        width: int,
        in_channels: int = 12,
        out_channels: int = 1,
        num_fno_layers: int = 2,
        num_unet_layers: int = 2,
        padding: int = 8,
        **kwargs,
    ):
        super(UFNONet, self).__init__()

        # Ensure padding is divisible by 8 for U-Net compatibility
        if padding % 8 != 0:
            self.padding = ((padding + 7) // 8) * 8
            print(
                f"Warning: Padding adjusted from {padding} to {self.padding} for U-Net compatibility"
            )
        else:
            self.padding = padding

        self.ufno = UFNO(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            num_fno_layers=num_fno_layers,
            num_unet_layers=num_unet_layers,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with padding/de-padding."""
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        # Apply padding
        x = F.pad(x, (0, 0, 0, self.padding, 0, self.padding), "replicate")
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, self.padding), "constant", 0)

        x = self.ufno(x)

        # Remove padding
        x = x.view(
            batchsize,
            size_x + self.padding,
            size_y + self.padding,
            size_z + self.padding,
            -1,
        )[..., : -self.padding, : -self.padding, : -self.padding, :]

        # Squeeze out channel dimension: (B, H, W, T, 1) -> (B, H, W, T)
        return x.squeeze(-1)

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return self.ufno.count_params()
