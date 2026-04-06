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
U-FNO: Enhanced Fourier Neural Operator with U-Net or Conv skip connections.

Reference:
    Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., & Benson, S. M. (2022).
    U-FNO--An enhanced Fourier neural operator-based deep-learning model for multiphase flow.
    Advances in Water Resources, 104180.
"""

import torch
import torch.nn as nn
from physicsnemo.models.module import Module
from torch import Tensor
from utils.padding import (
    compute_right_pad_to_multiple,
    compute_right_pad_to_multiple_per_dim,
    pad_spatial_right,
)

from models.physicsnemo_unet import PhysicsNemoUNet3D
from models.unet import UNet3D
from physicsnemo.models.layers import (
    Conv3dFCLayer,
    ConvNdFCLayer,
    ConvNdKernel1Layer,
    SpectralConv3d,
    SpectralConv4d,
    get_activation,
)
from physicsnemo.models.mlp import FullyConnected


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
    decoder_activation_fn : str, optional
        Activation for decoder layers (None uses activation_fn, last layer always linear)
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
        decoder_activation_fn: str = None,  # None means use activation_fn
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
        self.decoder_activation_fn_name = (
            decoder_activation_fn if decoder_activation_fn else activation_fn
        )
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
                    UNet3D(
                        self.width,
                        self.width,
                        kernel_size=unet_kernel_size,
                        dropout_rate=unet_dropout,
                    )
                )
            elif self.unet_type == "physicsnemo":
                # Use PhysicsNemo's UNet wrapper (from physicsnemo_unet.py)
                self.unet_modules.append(
                    PhysicsNemoUNet3D(
                        in_channels=self.width,
                        out_channels=self.width,
                        kernel_size=unet_kernel_size,
                        model_depth=3,  # 3 downsampling levels like custom UNet3D
                        feature_map_channels=[self.width] * 3,
                        conv_activation="leaky_relu",
                        normalization="batchnorm",
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
                    get_activation(self.activation_fn_name),
                )
            )

        # Build decoder network
        self.decoder = self._build_decoder_network(
            width,
            out_channels,
            decoder_layers,
            decoder_width,
            decoder_type,
            self.decoder_activation_fn_name,
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
                layers_list.append(get_activation(self.activation_fn_name))
                for _ in range(num_layers - 2):
                    layers_list.append(Conv3dFCLayer(hidden_width, hidden_width))
                    layers_list.append(get_activation(self.activation_fn_name))
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
        activation_fn: str,
    ) -> nn.Module:
        """Build decoder network to project latent space to output.

        Parameters
        ----------
        width : int
            Input width from FNO layers
        out_channels : int
            Output channels
        num_layers : int
            Number of hidden layers (0 = direct projection)
        hidden_width : int
            Hidden layer width
        decoder_type : str
            'mlp' for fully connected, 'conv' for 1x1 convolutions
        activation_fn : str
            Activation function name (last layer always linear)

        Returns
        -------
        nn.Module
            Decoder network
        """
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            else:
                return FullyConnected(
                    in_features=width,
                    layer_size=hidden_width,
                    out_features=out_channels,
                    num_layers=num_layers,
                    activation_fn=activation_fn,
                )
        elif decoder_type == "conv":
            if num_layers == 0:
                return Conv3dFCLayer(width, out_channels)
            else:
                layers_list = []
                in_ch = width
                for _ in range(num_layers):
                    layers_list.append(Conv3dFCLayer(in_ch, hidden_width))
                    layers_list.append(get_activation(activation_fn))
                    in_ch = hidden_width
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

        self.time_modes = modes3

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

    def forward(
        self,
        x: Tensor,
        target_times: Tensor = None,
    ) -> Tensor:
        """Forward pass with padding/de-padding.

        Parameters
        ----------
        x : Tensor
            Input ``(B, H, W, T_in, C)``.
        target_times : Tensor, optional
            Explicit target time coordinates ``(K,)`` or ``(K, 1)``.
            When provided and K != T_in, the time axis is padded so the
            FNO operates on at least L+K timesteps, and the output is
            cropped to the last K timesteps.

        Returns
        -------
        Tensor  ``(B, H, W, T_out)`` where T_out = K if target_times given,
                else T_in.
        """
        h, w, t_in = x.shape[1], x.shape[2], x.shape[3]

        K = target_times.shape[0] if target_times is not None else None

        if K is not None and K != t_in:
            desired_t = t_in + K
            min_t = max(desired_t, 2 * self.time_modes)
            extra = min_t - t_in
            x = pad_spatial_right(
                x,
                spatial_ndim=3,
                right_pad=(0, 0, extra),
                mode="replicate",
            )
            t_padded = x.shape[3]
        else:
            K = None
            t_padded = t_in

        pad_h, pad_w, pad_t = compute_right_pad_to_multiple(
            (h, w, t_padded), multiple=8, min_right_pad=self.padding
        )
        x = pad_spatial_right(
            x, spatial_ndim=3, right_pad=(pad_h, pad_w, pad_t), mode="replicate"
        )

        x = self.ufno(x)

        if K is not None:
            x = x[:, :h, :w, t_in : t_in + K, :]
        else:
            x = x[:, :h, :w, :t_in, :]

        return x.squeeze(-1)

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return self.ufno.count_params()


# =============================================================================
# 4D FNO CLASSES (3D spatial + time)
# =============================================================================
# Note: U-Net and Conv skip connections are NOT available for 4D problems
# because PyTorch does not provide native nn.Conv4d. These classes use only
# officially supported PhysicsNemo layers: SpectralConv4d, ConvNdKernel1Layer,
# and ConvNdFCLayer.
# =============================================================================


class FNO4D(Module):
    """4D Fourier Neural Operator for volumetric (3D space + time) problems.

    Input: (B, X, Y, Z, T, C)
    Output: (B, X, Y, Z, T, out_channels)

    Architecture:
    - Lifting network (ConvNdFCLayer)
    - num_fno_layers Fourier layers (SpectralConv4d + ConvNdKernel1Layer)
    - Decoder network (ConvNdFCLayer)

    Note: Only pure FNO mode is supported for 4D. U-Net and Conv skip
    connections are not available because PyTorch has no native Conv4d.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    width : int
        Latent channel dimension
    modes1, modes2, modes3, modes4 : int
        Number of Fourier modes in each dimension (X, Y, Z, T)
    num_fno_layers : int
        Number of Fourier layers
    activation_fn : str
        Activation function name
    lifting_layers : int
        Number of layers in lifting network
    decoder_layers : int
        Number of hidden layers in decoder
    decoder_width : int
        Hidden layer size in decoder
    coord_features : bool
        Whether to add coordinate features (x, y, z, t)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 6,
        modes4: int = 6,
        num_fno_layers: int = 4,
        activation_fn: str = "gelu",
        lifting_layers: int = 2,
        decoder_layers: int = 1,
        decoder_width: int = 128,
        coord_features: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.num_fno_layers = num_fno_layers
        self.coord_features = coord_features
        self.activation_fn_name = activation_fn
        self.activation_fn = get_activation(activation_fn)

        # Coordinate features add 4 channels (x, y, z, t)
        lift_in_channels = in_channels + 4 if coord_features else in_channels

        # Lifting network using ConvNdFCLayer (supports arbitrary dimensions)
        self.lift_network = self._build_lifting_network(
            lift_in_channels, width, lifting_layers
        )

        # Fourier layers: SpectralConv4d + ConvNdKernel1Layer
        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()

        for _ in range(num_fno_layers):
            self.spectral_convs.append(
                SpectralConv4d(self.width, self.width, modes1, modes2, modes3, modes4)
            )
            self.conv_1x1s.append(ConvNdKernel1Layer(self.width, self.width))

        # Decoder network using ConvNdFCLayer
        self.decoder = self._build_decoder_network(
            width, out_channels, decoder_layers, decoder_width
        )

    def _build_lifting_network(
        self, in_channels: int, width: int, num_layers: int
    ) -> nn.Module:
        """Build lifting network using ConvNdFCLayer."""
        if num_layers == 1:
            return ConvNdFCLayer(in_channels, width)
        else:
            layers_list = []
            hidden_width = width // 2
            layers_list.append(ConvNdFCLayer(in_channels, hidden_width))
            layers_list.append(get_activation(self.activation_fn_name))
            for _ in range(num_layers - 2):
                layers_list.append(ConvNdFCLayer(hidden_width, hidden_width))
                layers_list.append(get_activation(self.activation_fn_name))
            layers_list.append(ConvNdFCLayer(hidden_width, width))
            return nn.Sequential(*layers_list)

    def _build_decoder_network(
        self, width: int, out_channels: int, num_layers: int, hidden_width: int
    ) -> nn.Module:
        """Build decoder network using ConvNdFCLayer."""
        if num_layers == 0:
            return ConvNdFCLayer(width, out_channels)
        else:
            layers_list = []
            in_ch = width
            for _ in range(num_layers):
                layers_list.append(ConvNdFCLayer(in_ch, hidden_width))
                layers_list.append(get_activation(self.activation_fn_name))
                in_ch = hidden_width
            layers_list.append(ConvNdFCLayer(hidden_width, out_channels))
            return nn.Sequential(*layers_list)

    def _create_meshgrid(self, shape: list, device: torch.device) -> Tensor:
        """Create 4D coordinate meshgrid (x, y, z, t) normalized to [0, 1]."""
        bsize = shape[0]
        size_x, size_y, size_z, size_t = shape[2], shape[3], shape[4], shape[5]

        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_t = torch.linspace(0, 1, size_t, dtype=torch.float32, device=device)

        grid_x, grid_y, grid_z, grid_t = torch.meshgrid(
            grid_x, grid_y, grid_z, grid_t, indexing="ij"
        )

        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)
        grid_t = grid_t.unsqueeze(0).unsqueeze(0).expand(bsize, 1, -1, -1, -1, -1)

        return torch.cat((grid_x, grid_y, grid_z, grid_t), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through FNO4D.

        Input: (B, X, Y, Z, T, C)
        Output: (B, X, Y, Z, T, out_channels)
        """
        # Convert to channel-first: (B, C, X, Y, Z, T)
        x = x.permute(0, 5, 1, 2, 3, 4)

        # Add coordinate features
        if self.coord_features:
            coord_feat = self._create_meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Lifting
        x = self.lift_network(x)

        # Fourier layers
        for layer_idx in range(self.num_fno_layers):
            x1 = self.spectral_convs[layer_idx](x)
            x2 = self.conv_1x1s[layer_idx](x)
            if layer_idx < self.num_fno_layers - 1:
                x = self.activation_fn(x1 + x2)
            else:
                x = x1 + x2

        # Decoder
        x = self.decoder(x)

        # Convert to channel-last: (B, X, Y, Z, T, out_channels)
        x = x.permute(0, 2, 3, 4, 5, 1)
        return x

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FNO4DNet(nn.Module):
    """Wrapper for FNO4D that handles padding/de-padding.

    Input: (B, X, Y, Z, T, C)
    Output: (B, X, Y, Z, T)

    Parameters
    ----------
    modes1, modes2, modes3, modes4 : int
        Number of Fourier modes in each dimension
    width : int
        Latent channel dimension
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    num_fno_layers : int
        Number of Fourier layers
    padding : int or list
        Padding for each dimension (X, Y, Z, T)
    **kwargs
        Additional arguments passed to FNO4D
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        modes3: int,
        modes4: int,
        width: int,
        in_channels: int = 11,
        out_channels: int = 1,
        num_fno_layers: int = 4,
        padding: int = 8,
        **kwargs,
    ):
        super(FNO4DNet, self).__init__()

        # Store padding for each dimension (X, Y, Z, T)
        if isinstance(padding, int):
            self.padding = [padding, padding, padding, padding]
        else:
            self.padding = list(padding) + [0] * (4 - len(padding))
            self.padding = self.padding[:4]

        self.time_modes = modes4

        self.fno4d = FNO4D(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            modes4=modes4,
            width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            num_fno_layers=num_fno_layers,
            **kwargs,
        )

    def forward(
        self,
        x: Tensor,
        target_times: Tensor = None,
    ) -> Tensor:
        """Forward pass with padding/de-padding.

        Parameters
        ----------
        x : Tensor
            Input ``(B, X, Y, Z, T_in, C)``.
        target_times : Tensor, optional
            Explicit target time coordinates ``(K,)`` or ``(K, 1)``.
            When provided and K != T_in, the time axis is padded so the
            FNO operates on at least L+K timesteps, and the output is
            cropped to the last K timesteps.

        Returns
        -------
        Tensor  ``(B, X, Y, Z, T_out)`` where T_out = K if target_times given,
                else T_in.
        """
        x0, y0, z0, t_in = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        K = target_times.shape[0] if target_times is not None else None

        if K is not None and K != t_in:
            desired_t = t_in + K
            min_t = max(desired_t, 2 * self.time_modes)
            extra = min_t - t_in
            x = pad_spatial_right(
                x,
                spatial_ndim=4,
                right_pad=(0, 0, 0, extra),
                mode="replicate",
            )
            t_padded = x.shape[4]
        else:
            K = None
            t_padded = t_in

        pad_x, pad_y, pad_z, pad_t = compute_right_pad_to_multiple_per_dim(
            (x0, y0, z0, t_padded), multiple=8, min_right_pad=self.padding
        )
        x = pad_spatial_right(
            x,
            spatial_ndim=4,
            right_pad=(pad_x, pad_y, pad_z, pad_t),
            mode="replicate",
        )

        x = self.fno4d(x)

        if K is not None:
            x = x[:, :x0, :y0, :z0, t_in : t_in + K, :]
        else:
            x = x[:, :x0, :y0, :z0, :t_in, :]

        return x.squeeze(-1)

    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return self.fno4d.count_params()
