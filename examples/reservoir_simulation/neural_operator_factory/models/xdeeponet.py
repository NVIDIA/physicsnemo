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
DeepONet Variants for 2D and 3D spatial problems.

Supported variants:
    - deeponet: Basic DeepONet
    - u_deeponet: DeepONet with U-Net branch
    - fourier_deeponet: DeepONet with Fourier layers
    - conv_deeponet: DeepONet with convolutional layers
    - hybrid_deeponet: Combination of Fourier + U-Net + Conv
    - mionet: Multiple-input operator network (2 branches)
    - fourier_mionet: MIONet with Fourier layers
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from physicsnemo.models.module import Module
from torch import Tensor
from utils.padding import compute_right_pad_to_multiple, pad_spatial_right

from models.physicsnemo_unet import PhysicsNemoUNet2D, PhysicsNemoUNet3D
from models.unet import UNet2D, UNet3D
from physicsnemo.models.layers import (
    Conv2dFCLayer,
    Conv3dFCLayer,
    SpectralConv2d,
    SpectralConv3d,
    get_activation,
)
from physicsnemo.models.mlp import FullyConnected

# =============================================================================
# Branch Config Normalization
# =============================================================================


def _normalize_branch_config(config: dict) -> dict:
    """Normalize branch config to the nested encoder/layers format.

    Supports two formats:

    **New format** (nested)::

        branch1:
          encoder:
            type: linear       # "linear" (LazyLinear lift) or "mlp"
            hidden_width: 64   # MLP-only settings
            num_layers: 2
            activation_fn: tanh
          layers:
            num_fourier_layers: 3
            num_unet_layers: 1
            num_conv_layers: 0
            modes1: 12
            ...
          internal_resolution: null

    **Old format** (flat, auto-converted for backward compat)::

        branch1:
          encoder: spatial     # or "mlp"
          num_fourier_layers: 3
          hidden_width: 64
          ...

    Returns a dict in the new format.
    """
    if "encoder" not in config:
        return config

    enc = config["encoder"]

    if not isinstance(enc, str):
        return config

    enc_type_str = str(enc).lower()
    cfg = dict(config)
    cfg.pop("encoder")

    encoder_keys = {"hidden_width", "num_layers"}
    layer_keys = {
        "num_fourier_layers",
        "num_unet_layers",
        "num_conv_layers",
        "modes1",
        "modes2",
        "modes3",
        "kernel_size",
        "dropout",
        "unet_impl",
    }

    activation = cfg.pop("activation_fn", "sin")
    internal_res = cfg.pop("internal_resolution", None)
    in_channels = cfg.pop("in_channels", None)

    encoder_dict = {
        "type": "mlp" if enc_type_str == "mlp" else "linear",
        "activation_fn": activation,
    }
    for k in encoder_keys:
        if k in cfg:
            encoder_dict[k] = cfg.pop(k)

    layers_dict = {"activation_fn": activation}
    for k in layer_keys:
        if k in cfg:
            layers_dict[k] = cfg.pop(k)

    result = {"encoder": encoder_dict, "layers": layers_dict}
    if internal_res is not None:
        result["internal_resolution"] = internal_res
    if in_channels is not None:
        result["in_channels"] = in_channels

    return result


def _build_conv_encoder(width: int, enc_config: dict) -> nn.Module:
    """Build a multi-layer pointwise encoder to replace the default LazyLinear lift.

    Operates in channels-last format ``(B, *spatial, C)`` — matching the
    SpatialBranch lift interface.  Each layer is a ``Linear`` with activation,
    equivalent to a 1x1 convolution applied independently at every spatial point.

    Parameters
    ----------
    width : int
        Output width (latent dimension).
    enc_config : dict
        Encoder config with optional ``num_layers``, ``hidden_width``,
        ``activation_fn``.
    """
    num_layers = enc_config.get("num_layers", 1)
    activation_fn = enc_config.get("activation_fn", "relu")
    act = get_activation(activation_fn)

    if num_layers <= 1:
        return nn.LazyLinear(width)

    hidden_width = enc_config.get("hidden_width", width // 2)
    layers_list = [nn.LazyLinear(hidden_width), act]
    for _ in range(num_layers - 2):
        layers_list.extend([nn.Linear(hidden_width, hidden_width), act])
    layers_list.append(nn.Linear(hidden_width, width))
    return nn.Sequential(*layers_list)


# =============================================================================
# Shared Components
# =============================================================================


class TrunkNet(nn.Module):
    """
    MLP trunk network for encoding query coordinates.

    Input: (T, in_features) - T query points with in_features dimensions
    Output: (T, out_features) - encoded representations

    Parameters
    ----------
    output_activation : bool
        If True (default), apply activation to the output layer.
        If False, the output layer is linear (no activation).
        The original TNO paper uses False (linear trunk output)
        to avoid squashing the Hadamard product's dynamic range.
    """

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 64,
        hidden_width: int = 128,
        num_layers: int = 6,
        activation_fn: str = "sin",
        output_activation: bool = True,
    ):
        super().__init__()

        self._output_activation = output_activation

        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)

        self.layers = nn.ModuleList()
        self.layers.append(self._make_linear(in_features, hidden_width))
        for _ in range(num_layers - 1):
            self.layers.append(self._make_linear(hidden_width, hidden_width))

        self.output_layer = self._make_linear(hidden_width, out_features)

    def _make_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        init.xavier_normal_(layer.weight)
        init.zeros_(layer.bias)
        return layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)
        if self._output_activation:
            x = self.activation_fn(x)
        return x


class MLPBranch(nn.Module):
    """
    MLP branch network for scalar/vector inputs.

    Input: (B, in_features) - batch of scalar inputs
    Output: (B, out_features) - encoded representations

    Note: in_features is auto-discovered using LazyLinear on first forward pass.
    """

    def __init__(
        self,
        out_features: int,
        hidden_width: int = 64,
        num_layers: int = 3,
        activation_fn: str = "relu",
    ):
        super().__init__()

        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)

        self.layers = nn.ModuleList()
        # First layer uses LazyLinear to auto-discover input size
        self.layers.append(nn.LazyLinear(hidden_width))
        for _ in range(num_layers - 2):
            self.layers.append(self._make_linear(hidden_width, hidden_width))

        self.output_layer = self._make_linear(hidden_width, out_features)

    def _make_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        init.xavier_normal_(layer.weight)
        init.zeros_(layer.bias)
        return layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return self.activation_fn(self.output_layer(x))


# =============================================================================
# 2D Components
# =============================================================================


class SpatialBranch(nn.Module):
    """
    2D spatial branch network with Fourier, U-Net, and/or Conv layers.

    Input: (B, H, W, C) - batch of 2D spatial fields
    Output: (B, H, W, width) - encoded spatial representations

    When *internal_resolution* is set, the feature maps are pooled to
    that fixed size before processing and upsampled back afterwards.
    This enables resolution-agnostic training and inference.
    """

    def __init__(
        self,
        in_channels: int,
        width: int,
        num_fourier_layers: int = 0,
        num_unet_layers: int = 0,
        num_conv_layers: int = 0,
        modes1: int = 12,
        modes2: int = 12,
        kernel_size: int = 3,
        dropout: float = 0.0,
        unet_impl: str = "custom",
        activation_fn: str = "gelu",
        internal_resolution: Optional[list] = None,
    ):
        super().__init__()

        self.num_fourier_layers = num_fourier_layers
        self.num_unet_layers = num_unet_layers
        self.num_conv_layers = num_conv_layers
        self.use_fourier_base = num_fourier_layers > 0
        self.internal_resolution = (
            tuple(internal_resolution) if internal_resolution else None
        )

        total_layers = num_fourier_layers + num_unet_layers + num_conv_layers
        if total_layers == 0:
            raise ValueError("SpatialBranch requires at least one layer type")

        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)

        if self.internal_resolution is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(self.internal_resolution)

        # Lifting layer
        self.lift = nn.LazyLinear(width)

        # Spectral convolutions (Fourier layers)
        num_fourier_components = (
            total_layers if self.use_fourier_base else num_fourier_layers
        )
        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()
        for _ in range(num_fourier_components):
            self.spectral_convs.append(SpectralConv2d(width, width, modes1, modes2))
            self.conv_1x1s.append(nn.Conv2d(width, width, kernel_size=1))

        # U-Net modules
        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            if unet_impl == "custom":
                self.unet_modules.append(UNet2D(width, width, kernel_size, dropout))
            else:
                self.unet_modules.append(PhysicsNemoUNet2D(width, width, kernel_size))

        # Convolutional modules
        self.conv_modules = nn.ModuleList()
        padding = (kernel_size - 1) // 2
        for _ in range(num_conv_layers):
            self.conv_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        width,
                        width,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(width),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        # Lift to width dimension
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, width) -> (B, width, H, W)

        # Adaptive pool to internal resolution if configured
        original_size = x.shape[2:]
        if self.internal_resolution is not None:
            x = self.adaptive_pool(x)

        # Fourier layers
        for i in range(self.num_fourier_layers):
            x = self.activation_fn(self.spectral_convs[i](x) + self.conv_1x1s[i](x))

        # Hybrid or standalone layers
        if self.use_fourier_base:
            for i in range(self.num_unet_layers):
                j = self.num_fourier_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x)
                    + self.conv_1x1s[j](x)
                    + self.unet_modules[i](x)
                )
            for i in range(self.num_conv_layers):
                j = self.num_fourier_layers + self.num_unet_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x)
                    + self.conv_1x1s[j](x)
                    + self.conv_modules[i](x)
                )
        else:
            for unet in self.unet_modules:
                x = self.activation_fn(unet(x))
            for conv in self.conv_modules:
                x = self.activation_fn(conv(x))

        # Upsample back to original resolution
        if self.internal_resolution is not None and x.shape[2:] != original_size:
            x = F.interpolate(
                x, size=original_size, mode="bilinear", align_corners=True
            )

        return x.permute(0, 2, 3, 1)  # (B, width, H, W) -> (B, H, W, width)


class DeepONet(Module):
    """
    2D DeepONet for operator learning.

    Input:
        - x_branch1: (B, H, W, C) for spatial or (B, in_features) for MLP
        - x_time: (T,) or (T, in_features) query coordinates
        - x_branch2: optional second branch input for MIONet
    Output: (B, H, W, T) for spatial or (B, T) for MLP
    """

    VALID_VARIANTS = [
        "deeponet",
        "u_deeponet",
        "fourier_deeponet",
        "conv_deeponet",
        "hybrid_deeponet",
        "mionet",
        "fourier_mionet",
        "tno",
    ]

    def __init__(
        self,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
    ):
        super().__init__()

        self.variant = variant.lower()
        self.width = width
        self.decoder_type = decoder_type.lower()
        self.decoder_activation_fn = decoder_activation_fn

        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Valid: {self.VALID_VARIANTS}"
            )

        branch1_config = branch1_config or {}
        trunk_config = trunk_config or {}

        # Build networks
        self.branch1 = self._build_branch(branch1_config, width)

        self.has_branch2 = branch2_config is not None
        if self.has_branch2:
            self.branch2 = self._build_branch(branch2_config, width)

        self.trunk = TrunkNet(
            in_features=trunk_config.get("in_features", 1),
            out_features=width,
            hidden_width=trunk_config.get("hidden_width", 128),
            num_layers=trunk_config.get("num_layers", 6),
            activation_fn=trunk_config.get("activation_fn", "sin"),
            output_activation=trunk_config.get("output_activation", True),
        )

        if decoder_type == "temporal_projection":
            self._temporal_projection = True
            self.decoder = self._build_decoder(
                width,
                width,
                decoder_layers,
                decoder_width,
                "mlp",
                decoder_activation_fn,
            )
            self.temporal_head = None
        else:
            self._temporal_projection = False
            self.decoder = self._build_decoder(
                width,
                1,
                decoder_layers,
                decoder_width,
                decoder_type,
                decoder_activation_fn,
            )

    def set_output_window(self, K: int):
        """Set the temporal projection head for K output timesteps."""
        if self._temporal_projection:
            device = next(self.parameters()).device
            self.temporal_head = nn.Linear(self.width, K).to(device)

    def _build_branch(self, config: dict, width: int) -> nn.Module:
        config = _normalize_branch_config(config)
        enc = config.get("encoder", {})
        layers = config.get("layers", {})

        enc_type = enc.get("type", "linear")
        enc_activation = enc.get("activation_fn", "sin")

        has_layers = (
            layers.get("num_fourier_layers", 0)
            + layers.get("num_unet_layers", 0)
            + layers.get("num_conv_layers", 0)
        ) > 0

        if enc_type == "mlp" and not has_layers:
            return MLPBranch(
                out_features=width,
                hidden_width=enc.get("hidden_width", 64),
                num_layers=enc.get("num_layers", 3),
                activation_fn=enc_activation,
            )

        layer_activation = layers.get("activation_fn", enc_activation)
        branch = SpatialBranch(
            in_channels=config.get("in_channels", 12),
            width=width,
            num_fourier_layers=layers.get("num_fourier_layers", 0),
            num_unet_layers=layers.get("num_unet_layers", 0),
            num_conv_layers=layers.get("num_conv_layers", 0),
            modes1=layers.get("modes1", 12),
            modes2=layers.get("modes2", 12),
            kernel_size=layers.get("kernel_size", 3),
            dropout=layers.get("dropout", 0.0),
            unet_impl=layers.get("unet_impl", "custom"),
            activation_fn=layer_activation,
            internal_resolution=config.get("internal_resolution", None),
        )
        if enc_type == "conv":
            branch.lift = _build_conv_encoder(width, enc)
        return branch

    def _build_decoder(
        self,
        width: int,
        out_channels: int,
        num_layers: int,
        hidden_width: int,
        decoder_type: str,
        activation_fn: str,
    ) -> nn.Module:
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            return FullyConnected(
                width, hidden_width, out_channels, num_layers, activation_fn
            )

        elif decoder_type == "conv":
            if num_layers == 0:
                return Conv2dFCLayer(width, out_channels)

            layers = []
            in_ch = width
            for _ in range(num_layers):
                layers.extend(
                    [Conv2dFCLayer(in_ch, hidden_width), get_activation(activation_fn)]
                )
                in_ch = hidden_width
            layers.append(Conv2dFCLayer(hidden_width, out_channels))
            return nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self, x_branch1: Tensor, x_time: Tensor, x_branch2: Tensor = None
    ) -> Tensor:
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)

        b1_out = self.branch1(x_branch1)

        if self.has_branch2:
            if x_branch2 is None:
                raise ValueError("x_branch2 required for mionet/tno variants")
            b2_out = self.branch2(x_branch2)

        trunk_out = self.trunk(x_time)

        # Combine branch and trunk
        if b1_out.dim() == 4:  # Spatial branch
            if self._temporal_projection:
                # Single trunk query → (B, H, W, width) combined
                trunk_single = trunk_out[0:1]  # (1, width)
                trunk_exp = trunk_single.unsqueeze(1).unsqueeze(2)  # (1, 1, 1, w)
                combined = b1_out * trunk_exp
                if self.has_branch2:
                    if b2_out.dim() == 4:
                        combined = combined * b2_out
                    else:
                        combined = combined * b2_out.unsqueeze(1).unsqueeze(2)
                combined = self.decoder(combined)  # (B, H, W, width)
                if self.temporal_head is not None:
                    combined = self.temporal_head(combined)  # (B, H, W, K)
                return combined

            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            if self.has_branch2:
                if b2_out.dim() == 4:
                    b2_out = b2_out.unsqueeze(1)
                else:
                    b2_out = b2_out.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                combined = b1_out * b2_out * trunk_out
            else:
                combined = b1_out * trunk_out

            if self.decoder_type == "mlp":
                return self.decoder(combined).squeeze(-1).permute(0, 2, 3, 1)

            B, T, H, W, C = combined.shape
            combined = combined.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
            return self.decoder(combined).reshape(B, T, H, W).permute(0, 2, 3, 1)

        else:  # MLP branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0)

            if self.has_branch2:
                combined = b1_out * b2_out.unsqueeze(1) * trunk_out
            else:
                combined = b1_out * trunk_out

            return self.decoder(combined).squeeze(-1)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepONetWrapper(nn.Module):
    """
    2D DeepONet wrapper with automatic padding and input extraction.

    Input: (B, H, W, T, C) - batch of spatiotemporal fields
    Output: (B, H, W, T) - predicted output field
    """

    def __init__(
        self,
        padding: int = 8,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
    ):
        super().__init__()

        self.padding = ((padding + 7) // 8) * 8 if padding % 8 != 0 else padding
        self.variant = variant

        trunk_config = trunk_config or {}
        self.trunk_input = trunk_config.get("input_type", "time").lower()

        if self.trunk_input not in ["time", "grid"]:
            raise ValueError("trunk input_type must be 'time' or 'grid'")

        if self.trunk_input == "grid":
            trunk_config["in_features"] = 3  # (x, y, t)
        else:
            trunk_config["in_features"] = trunk_config.get("in_features", 1)

        self.model = DeepONet(
            variant=variant,
            width=width,
            branch1_config=branch1_config,
            branch2_config=branch2_config,
            trunk_config=trunk_config,
            decoder_type=decoder_type,
            decoder_width=decoder_width,
            decoder_layers=decoder_layers,
            decoder_activation_fn=decoder_activation_fn,
        )
        self._temporal_projection = self.model._temporal_projection

    def set_output_window(self, K: int):
        """Delegate to the inner DeepONet model."""
        self.model.set_output_window(K)

    def forward(
        self,
        x: Tensor,
        x_branch2: Tensor = None,
        target_times: Tensor = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input ``(B, H, W, T_in, C)``.
        x_branch2 : Tensor, optional
            Secondary branch input (MIONet variants).
        target_times : Tensor, optional
            Explicit trunk query coordinates ``(K,)`` or ``(K, 1)``.
            When provided the trunk evaluates at these K points instead of
            extracting time values from ``x``.  This enables autoregressive
            temporal bundling where K != T_in.

        Returns
        -------
        Tensor  ``(B, H, W, T_out)`` where T_out = K if target_times given,
                else T_in.
        """
        H, W = x.shape[1], x.shape[2]

        pad_h, pad_w = compute_right_pad_to_multiple(
            (H, W), multiple=8, min_right_pad=self.padding
        )
        x = pad_spatial_right(
            x, spatial_ndim=2, right_pad=(pad_h, pad_w), mode="replicate"
        )

        if x_branch2 is not None and x_branch2.dim() > 2:
            x_branch2 = pad_spatial_right(
                x_branch2,
                spatial_ndim=2,
                right_pad=(pad_h, pad_w),
                mode="replicate",
            )

        x_spatial = x.permute(0, 4, 1, 2, 3)[..., 0].permute(0, 2, 3, 1)

        if target_times is not None:
            if self.trunk_input == "grid":
                t_vals = (
                    target_times
                    if target_times.dim() == 1
                    else target_times.squeeze(-1)
                )
                spatial = x[0, 0, 0, 0, -3:-1]  # (2,) = grid_x, grid_y
                spatial_exp = spatial.unsqueeze(0).expand(t_vals.shape[0], -1)
                x_trunk = torch.cat(
                    [spatial_exp, t_vals.unsqueeze(-1)], dim=-1
                )  # (K, 3)
            else:
                x_trunk = (
                    target_times
                    if target_times.dim() == 2
                    else target_times.unsqueeze(-1)
                )
        elif self.trunk_input == "grid":
            x_trunk = x[0, 0, 0, :, -3:]
        else:
            x_trunk = x[0, 0, 0, :, -1].unsqueeze(-1)

        return self.model(x_spatial, x_trunk, x_branch2)[:, :H, :W, :]

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return self.model.count_params()


# =============================================================================
# 3D Components
# =============================================================================


class SpatialBranch3D(nn.Module):
    """
    3D spatial branch network with Fourier, U-Net, and/or Conv layers.

    Input: (B, X, Y, Z, C) - batch of 3D spatial fields
    Output: (B, X, Y, Z, width) - encoded spatial representations
    """

    def __init__(
        self,
        in_channels: int,
        width: int,
        num_fourier_layers: int = 0,
        num_unet_layers: int = 0,
        num_conv_layers: int = 0,
        modes1: int = 10,
        modes2: int = 10,
        modes3: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.0,
        unet_impl: str = "custom",
        activation_fn: str = "gelu",
        internal_resolution: Optional[list] = None,
    ):
        super().__init__()

        self.num_fourier_layers = num_fourier_layers
        self.num_unet_layers = num_unet_layers
        self.num_conv_layers = num_conv_layers
        self.use_fourier_base = num_fourier_layers > 0
        self.internal_resolution = (
            tuple(internal_resolution) if internal_resolution else None
        )

        total_layers = num_fourier_layers + num_unet_layers + num_conv_layers
        if total_layers == 0:
            raise ValueError("SpatialBranch3D requires at least one layer type")

        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)

        if self.internal_resolution is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool3d(self.internal_resolution)

        # Lifting layer
        self.lift = nn.LazyLinear(width)

        # Spectral convolutions (Fourier layers)
        num_fourier_components = (
            total_layers if self.use_fourier_base else num_fourier_layers
        )
        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()
        for _ in range(num_fourier_components):
            self.spectral_convs.append(
                SpectralConv3d(width, width, modes1, modes2, modes3)
            )
            self.conv_1x1s.append(nn.Conv3d(width, width, kernel_size=1))

        # U-Net modules
        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            if unet_impl == "custom":
                self.unet_modules.append(UNet3D(width, width, kernel_size, dropout))
            else:
                self.unet_modules.append(PhysicsNemoUNet3D(width, width, kernel_size))

        # Convolutional modules
        self.conv_modules = nn.ModuleList()
        padding = (kernel_size - 1) // 2
        for _ in range(num_conv_layers):
            self.conv_modules.append(
                nn.Sequential(
                    nn.Conv3d(
                        width,
                        width,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm3d(width),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)

        original_size = x.shape[2:]
        if self.internal_resolution is not None:
            x = self.adaptive_pool(x)

        for i in range(self.num_fourier_layers):
            x = self.activation_fn(self.spectral_convs[i](x) + self.conv_1x1s[i](x))

        if self.use_fourier_base:
            for i in range(self.num_unet_layers):
                j = self.num_fourier_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x)
                    + self.conv_1x1s[j](x)
                    + self.unet_modules[i](x)
                )
            for i in range(self.num_conv_layers):
                j = self.num_fourier_layers + self.num_unet_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x)
                    + self.conv_1x1s[j](x)
                    + self.conv_modules[i](x)
                )
        else:
            for unet in self.unet_modules:
                x = self.activation_fn(unet(x))
            for conv in self.conv_modules:
                x = self.activation_fn(conv(x))

        if self.internal_resolution is not None and x.shape[2:] != original_size:
            x = F.interpolate(
                x, size=original_size, mode="trilinear", align_corners=True
            )

        return x.permute(0, 2, 3, 4, 1)


class DeepONet3D(Module):
    """
    3D DeepONet for operator learning on volumetric data.

    Input:
        - x_branch1: (B, X, Y, Z, C) for spatial or (B, in_features) for MLP
        - x_time: (T,) or (T, in_features) query coordinates
        - x_branch2: optional second branch input for MIONet
    Output: (B, X, Y, Z, T) for spatial or (B, T) for MLP
    """

    VALID_VARIANTS = [
        "deeponet",
        "u_deeponet",
        "fourier_deeponet",
        "conv_deeponet",
        "hybrid_deeponet",
        "mionet",
        "fourier_mionet",
        "tno",
    ]

    def __init__(
        self,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
    ):
        super().__init__()

        self.variant = variant.lower()
        self.width = width
        self.decoder_type = decoder_type.lower()
        self.decoder_activation_fn = decoder_activation_fn

        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Valid: {self.VALID_VARIANTS}"
            )

        branch1_config = branch1_config or {}
        trunk_config = trunk_config or {}

        # Build networks
        self.branch1 = self._build_branch(branch1_config, width)

        self.has_branch2 = branch2_config is not None
        if self.has_branch2:
            self.branch2 = self._build_branch(branch2_config, width)

        self.trunk = TrunkNet(
            in_features=trunk_config.get("in_features", 1),
            out_features=width,
            hidden_width=trunk_config.get("hidden_width", 128),
            num_layers=trunk_config.get("num_layers", 6),
            activation_fn=trunk_config.get("activation_fn", "sin"),
            output_activation=trunk_config.get("output_activation", True),
        )

        if decoder_type == "temporal_projection":
            self._temporal_projection = True
            self.decoder = self._build_decoder(
                width,
                width,
                decoder_layers,
                decoder_width,
                "mlp",
                decoder_activation_fn,
            )
            self.temporal_head = None
        else:
            self._temporal_projection = False
            self.decoder = self._build_decoder(
                width,
                1,
                decoder_layers,
                decoder_width,
                decoder_type,
                decoder_activation_fn,
            )

    def set_output_window(self, K: int):
        """Set the temporal projection head for K output timesteps."""
        if self._temporal_projection:
            device = next(self.parameters()).device
            self.temporal_head = nn.Linear(self.width, K).to(device)

    def _build_branch(self, config: dict, width: int) -> nn.Module:
        config = _normalize_branch_config(config)
        enc = config.get("encoder", {})
        layers = config.get("layers", {})

        enc_type = enc.get("type", "linear")
        enc_activation = enc.get("activation_fn", "sin")

        has_layers = (
            layers.get("num_fourier_layers", 0)
            + layers.get("num_unet_layers", 0)
            + layers.get("num_conv_layers", 0)
        ) > 0

        if enc_type == "mlp" and not has_layers:
            return MLPBranch(
                out_features=width,
                hidden_width=enc.get("hidden_width", 64),
                num_layers=enc.get("num_layers", 3),
                activation_fn=enc_activation,
            )

        layer_activation = layers.get("activation_fn", enc_activation)
        branch = SpatialBranch3D(
            in_channels=config.get("in_channels", 11),
            width=width,
            num_fourier_layers=layers.get("num_fourier_layers", 0),
            num_unet_layers=layers.get("num_unet_layers", 0),
            num_conv_layers=layers.get("num_conv_layers", 0),
            modes1=layers.get("modes1", 10),
            modes2=layers.get("modes2", 10),
            modes3=layers.get("modes3", 8),
            kernel_size=layers.get("kernel_size", 3),
            dropout=layers.get("dropout", 0.0),
            unet_impl=layers.get("unet_impl", "custom"),
            activation_fn=layer_activation,
            internal_resolution=config.get("internal_resolution", None),
        )
        if enc_type == "conv":
            branch.lift = _build_conv_encoder(width, enc)
        return branch

    def _build_decoder(
        self,
        width: int,
        out_channels: int,
        num_layers: int,
        hidden_width: int,
        decoder_type: str,
        activation_fn: str,
    ) -> nn.Module:
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            return FullyConnected(
                width, hidden_width, out_channels, num_layers, activation_fn
            )

        elif decoder_type == "conv":
            if num_layers == 0:
                return Conv3dFCLayer(width, out_channels)

            layers = []
            in_ch = width
            for _ in range(num_layers):
                layers.extend(
                    [Conv3dFCLayer(in_ch, hidden_width), get_activation(activation_fn)]
                )
                in_ch = hidden_width
            layers.append(Conv3dFCLayer(hidden_width, out_channels))
            return nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self, x_branch1: Tensor, x_time: Tensor, x_branch2: Tensor = None
    ) -> Tensor:
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)

        b1_out = self.branch1(x_branch1)

        if self.has_branch2:
            if x_branch2 is None:
                raise ValueError("x_branch2 required for mionet/tno variants")
            b2_out = self.branch2(x_branch2)

        trunk_out = self.trunk(x_time)

        # Combine branch and trunk
        if b1_out.dim() == 5:  # Spatial branch
            if self._temporal_projection:
                trunk_single = trunk_out[0:1]
                trunk_exp = trunk_single.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                combined = b1_out * trunk_exp
                if self.has_branch2:
                    if b2_out.dim() == 5:
                        combined = combined * b2_out
                    else:
                        combined = combined * b2_out.unsqueeze(1).unsqueeze(
                            2
                        ).unsqueeze(3)
                combined = self.decoder(combined)
                if self.temporal_head is not None:
                    combined = self.temporal_head(combined)
                return combined

            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

            if self.has_branch2:
                if b2_out.dim() == 5:
                    b2_out = b2_out.unsqueeze(1)
                else:
                    b2_out = b2_out.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                combined = b1_out * b2_out * trunk_out
            else:
                combined = b1_out * trunk_out

            if self.decoder_type == "mlp":
                return self.decoder(combined).squeeze(-1).permute(0, 2, 3, 4, 1)

            B, T, X, Y, Z, C = combined.shape
            combined = combined.permute(0, 1, 5, 2, 3, 4).reshape(B * T, C, X, Y, Z)
            return self.decoder(combined).reshape(B, T, X, Y, Z).permute(0, 2, 3, 4, 1)

        else:  # MLP branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0)

            if self.has_branch2:
                combined = b1_out * b2_out.unsqueeze(1) * trunk_out
            else:
                combined = b1_out * trunk_out

            return self.decoder(combined).squeeze(-1)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepONet3DWrapper(nn.Module):
    """
    3D DeepONet wrapper with automatic padding and input extraction.

    Input: (B, X, Y, Z, T, C) - batch of 4D spatiotemporal fields
    Output: (B, X, Y, Z, T) - predicted output field
    """

    def __init__(
        self,
        padding: int = 8,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
    ):
        super().__init__()

        self.padding = ((padding + 7) // 8) * 8 if padding % 8 != 0 else padding
        self.variant = variant

        trunk_config = trunk_config or {}
        self.trunk_input = trunk_config.get("input_type", "time").lower()

        if self.trunk_input not in ["time", "grid"]:
            raise ValueError("trunk input_type must be 'time' or 'grid'")

        if self.trunk_input == "grid":
            trunk_config["in_features"] = 4  # (x, y, z, t)
        else:
            trunk_config["in_features"] = trunk_config.get("in_features", 1)

        self.model = DeepONet3D(
            variant=variant,
            width=width,
            branch1_config=branch1_config,
            branch2_config=branch2_config,
            trunk_config=trunk_config,
            decoder_type=decoder_type,
            decoder_width=decoder_width,
            decoder_layers=decoder_layers,
            decoder_activation_fn=decoder_activation_fn,
        )
        self._temporal_projection = self.model._temporal_projection

    def set_output_window(self, K: int):
        """Delegate to the inner DeepONet3D model."""
        self.model.set_output_window(K)

    def forward(
        self,
        x: Tensor,
        x_branch2: Tensor = None,
        target_times: Tensor = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input ``(B, X, Y, Z, T_in, C)``.
        x_branch2 : Tensor, optional
            Secondary branch input (MIONet variants).
        target_times : Tensor, optional
            Explicit trunk query coordinates ``(K,)`` or ``(K, 1)``.
            When provided the trunk evaluates at these K points instead of
            extracting time values from ``x``.  This enables autoregressive
            temporal bundling where K != T_in.

        Returns
        -------
        Tensor  ``(B, X, Y, Z, T_out)`` where T_out = K if target_times given,
                else T_in.
        """
        X, Y, Z = x.shape[1], x.shape[2], x.shape[3]

        pad_x, pad_y, pad_z = compute_right_pad_to_multiple(
            (X, Y, Z), multiple=8, min_right_pad=self.padding
        )
        x = pad_spatial_right(
            x, spatial_ndim=3, right_pad=(pad_x, pad_y, pad_z), mode="replicate"
        )

        if x_branch2 is not None and x_branch2.dim() > 2:
            x_branch2 = pad_spatial_right(
                x_branch2,
                spatial_ndim=3,
                right_pad=(pad_x, pad_y, pad_z),
                mode="replicate",
            )

        x_spatial = x[:, :, :, :, 0, :]

        if target_times is not None:
            if self.trunk_input == "grid":
                t_vals = (
                    target_times
                    if target_times.dim() == 1
                    else target_times.squeeze(-1)
                )
                spatial = x[0, 0, 0, 0, 0, -4:-1]  # (3,) = grid_x, grid_y, grid_z
                spatial_exp = spatial.unsqueeze(0).expand(t_vals.shape[0], -1)
                x_trunk = torch.cat(
                    [spatial_exp, t_vals.unsqueeze(-1)], dim=-1
                )  # (K, 4)
            else:
                x_trunk = (
                    target_times
                    if target_times.dim() == 2
                    else target_times.unsqueeze(-1)
                )
        elif self.trunk_input == "grid":
            x_trunk = x[0, 0, 0, 0, :, -4:]
        else:
            x_trunk = x[0, 0, 0, 0, :, -1].unsqueeze(-1)

        return self.model(x_spatial, x_trunk, x_branch2)[:, :X, :Y, :Z, :]

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return self.model.count_params()
