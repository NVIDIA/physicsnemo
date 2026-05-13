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

"""Branch and trunk building blocks used by the xDeepONet family.

Provides four sub-networks:

- :class:`TrunkNet` — MLP trunk that encodes query coordinates (time or grid).
- :class:`MLPBranch` — fully-connected branch for scalar/vector inputs
  (e.g. the scalar branch in MIONet).
- :class:`SpatialBranch` — 2D spatial encoder composable from Fourier, UNet,
  and Conv layers.
- :class:`SpatialBranch3D` — 3D counterpart of ``SpatialBranch``.

UNet sub-modules inside the spatial branches use
:class:`physicsnemo.models.unet.UNet` (3D).  A small adapter
:class:`_UNet2DFromUNet3D` is provided locally for the 2D variant: it wraps
the 3D UNet with a singleton time dimension so the same library model covers
both spatial dimensionalities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from jaxtyping import Float
from torch import Tensor

from physicsnemo.models.unet import UNet as _PhysicsNeMoUNet
from physicsnemo.nn import SpectralConv2d, SpectralConv3d, get_activation

# ---------------------------------------------------------------------------
# UNet adapters (wrap the library's 3D UNet for reuse inside spatial branches)
# ---------------------------------------------------------------------------


class _UNet2DFromUNet3D(nn.Module):
    r"""Adapter using :class:`physicsnemo.models.unet.UNet` for 2D inputs.

    The library UNet is 3D only.  To reuse it for 2D, this adapter adds a
    short tiled time axis of length :math:`2^{\text{model\_depth}}` (long
    enough to survive the UNet's ``model_depth`` pooling stages), runs the
    3D UNet, and averages the result back to 2D.  Channel-first layout
    :math:`(B, C, H, W)` is preserved on input and output.

    .. important::

        Selecting ``num_unet_layers > 0`` in a 2D
        :class:`~physicsnemo.experimental.models.xdeeponet.SpatialBranch`
        (i.e. when this 2D adapter is used) makes the UNet branch operate
        on a tiled :math:`2^{\text{model\_depth}}`-deep volume.  With the
        default ``model_depth=3`` this is an **8x** memory and compute
        cost relative to a native 2D UNet of the same width and depth.
        This overhead is a property of the upstream library UNet being
        3D-only, not of this branch.  When ``num_unet_layers == 0`` the
        branch is bypassed and there is no overhead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        model_depth: int = 3,
        feature_map_channels: list[int] | None = None,
    ):
        super().__init__()
        if feature_map_channels is None:
            feature_map_channels = [in_channels] * model_depth
        self._t_tile = 2**model_depth
        self.unet = _PhysicsNeMoUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            model_depth=model_depth,
            feature_map_channels=feature_map_channels,
            num_conv_blocks=1,
            conv_activation="leaky_relu",
            conv_transpose_activation="leaky_relu",
            padding=kernel_size // 2,
            pooling_type="MaxPool3d",
            normalization="batchnorm",
            gradient_checkpointing=False,
        )

    def forward(
        self,
        x: Float[Tensor, "batch channels h w"],
    ) -> Float[Tensor, "batch out_channels h w"]:
        """Forward through the 3D UNet via a tiled time axis."""
        x = x.unsqueeze(-1).repeat(1, 1, 1, 1, self._t_tile)
        x = self.unet(x)
        return x.mean(dim=-1)


class _UNet3DFromUNet3D(nn.Module):
    r"""Thin wrapper exposing :class:`physicsnemo.models.unet.UNet`.

    Exposes the library 3D UNet with a fixed default configuration suitable
    for skip-connection reuse inside :class:`SpatialBranch3D`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        model_depth: int = 3,
        feature_map_channels: list[int] | None = None,
    ):
        super().__init__()
        if feature_map_channels is None:
            feature_map_channels = [in_channels] * model_depth
        self.unet = _PhysicsNeMoUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            model_depth=model_depth,
            feature_map_channels=feature_map_channels,
            num_conv_blocks=1,
            conv_activation="leaky_relu",
            conv_transpose_activation="leaky_relu",
            padding=kernel_size // 2,
            pooling_type="MaxPool3d",
            normalization="batchnorm",
            gradient_checkpointing=False,
        )

    def forward(
        self,
        x: Float[Tensor, "batch channels x y z"],
    ) -> Float[Tensor, "batch out_channels x y z"]:
        """Forward pass through the library 3D UNet."""
        return self.unet(x)


# ---------------------------------------------------------------------------
# Trunk and MLP branch
# ---------------------------------------------------------------------------


class TrunkNet(nn.Module):
    r"""MLP trunk network encoding query coordinates.

    Parameters
    ----------
    in_features : int
        Dimensionality of each query point (``1`` for time-only, ``3`` for 2D
        grid coordinates, ``4`` for 3D grid coordinates).
    out_features : int
        Output width (matches the DeepONet latent size).
    hidden_width : int
        Hidden layer width.
    num_layers : int
        Number of hidden layers.
    activation_fn : str
        Activation function name (``"sin"``, ``"tanh"``, ``"relu"``, etc.).
    output_activation : bool
        When ``True`` (default) the final layer is followed by the activation.
        Set ``False`` for linear output (e.g. the TNO configuration).

    Forward
    -------
    x : torch.Tensor
        Query coordinates of shape :math:`(T, D_{in})` where
        :math:`D_{in}` equals ``in_features``.

    Outputs
    -------
    torch.Tensor
        Encoded coordinates of shape :math:`(T, D_{out})` where
        :math:`D_{out}` equals ``out_features``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import TrunkNet
    >>> trunk = TrunkNet(in_features=1, out_features=64, hidden_width=64, num_layers=4)
    >>> t = torch.linspace(0, 1, 10).unsqueeze(-1)   # (10, 1)
    >>> phi = trunk(t)                                # (10, 64)
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

    def forward(
        self,
        x: Float[Tensor, "time in_features"],
    ) -> Float[Tensor, "time out_features"]:
        """Forward pass of the trunk network."""
        if not torch.compiler.is_compiling():
            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2D input (T, in_features), got {x.ndim}D "
                    f"tensor with shape {tuple(x.shape)}"
                )
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)
        if self._output_activation:
            x = self.activation_fn(x)
        return x


class MLPBranch(nn.Module):
    r"""Fully-connected branch for scalar/vector inputs.

    Used for the scalar branch in MIONet-style architectures.  Input features
    are auto-discovered via :class:`torch.nn.LazyLinear` on the first forward.

    Parameters
    ----------
    out_features : int
        Output width (matches the DeepONet latent size).
    hidden_width : int
        Hidden layer width.
    num_layers : int
        Number of fully-connected layers (including output). Must be ``>= 2``.
    activation_fn : str
        Activation function name.

    Forward
    -------
    x : torch.Tensor
        Scalar input of shape :math:`(B, D_{in})` where :math:`D_{in}` is
        auto-discovered on the first forward pass.

    Outputs
    -------
    torch.Tensor
        Encoded features of shape :math:`(B, D_{out})` where
        :math:`D_{out}` equals ``out_features``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import MLPBranch
    >>> branch = MLPBranch(out_features=64, hidden_width=64, num_layers=3)
    >>> x = torch.randn(2, 128)
    >>> out = branch(x)                               # (2, 64)
    """

    def __init__(
        self,
        out_features: int,
        hidden_width: int = 64,
        num_layers: int = 3,
        activation_fn: str = "relu",
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError(
                f"MLPBranch requires num_layers >= 2 (input + output), "
                f"got num_layers={num_layers}"
            )

        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)

        self.layers = nn.ModuleList()
        self.layers.append(nn.LazyLinear(hidden_width))
        for _ in range(num_layers - 2):
            self.layers.append(self._make_linear(hidden_width, hidden_width))

        self.output_layer = self._make_linear(hidden_width, out_features)

    def _make_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        init.xavier_normal_(layer.weight)
        init.zeros_(layer.bias)
        return layer

    def forward(
        self,
        x: Float[Tensor, "batch in_features"],
    ) -> Float[Tensor, "batch out_features"]:
        """Forward pass of the MLP branch."""
        if not torch.compiler.is_compiling():
            if x.ndim != 2:
                raise ValueError(
                    f"Expected 2D input (B, in_features), got {x.ndim}D "
                    f"tensor with shape {tuple(x.shape)}"
                )
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return self.activation_fn(self.output_layer(x))


# ---------------------------------------------------------------------------
# 2D spatial branch
# ---------------------------------------------------------------------------


class SpatialBranch(nn.Module):
    r"""2D spatial branch composable from Fourier, UNet, and Conv layers.

    The branch can be configured to use any combination of spectral, UNet,
    and plain convolutional layers.  When Fourier layers are present (the
    "base" mode) UNet/Conv layers are added alongside the spectral path
    (hybrid residual).  When no Fourier layers are present UNet/Conv act
    as independent sequential layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels (used only for documentation; the lift is
        :class:`torch.nn.LazyLinear`).
    width : int
        Latent/output width.
    num_fourier_layers : int
        Number of spectral layers.
    num_unet_layers : int
        Number of UNet layers (uses :class:`physicsnemo.models.unet.UNet`).
    num_conv_layers : int
        Number of Conv+BN layers.
    modes1, modes2 : int
        Fourier modes along H, W.
    kernel_size : int
        Kernel size for UNet and Conv layers.
    dropout : float
        Unused; kept for config compatibility.
    activation_fn : str
        Activation function name.
    internal_resolution : list, optional
        If set, inputs are adaptively pooled to this resolution before
        processing and upsampled back, decoupling model size from grid size.

    Forward
    -------
    x : torch.Tensor
        Channels-last input of shape :math:`(B, H, W, C)`.

    Outputs
    -------
    torch.Tensor
        Channels-last output of shape :math:`(B, H, W, D)` where
        :math:`D` equals ``width``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import SpatialBranch
    >>> branch = SpatialBranch(
    ...     in_channels=5, width=64, num_unet_layers=1, kernel_size=3
    ... )
    >>> x = torch.randn(2, 32, 32, 5)   # (B, H, W, C)
    >>> out = branch(x)                  # (2, 32, 32, 64)
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
        dropout: float = 0.0,  # noqa: ARG002 - kept for config compatibility
        activation_fn: str = "gelu",
        internal_resolution: list | None = None,
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

        self.lift = nn.LazyLinear(width)

        num_fourier_components = (
            total_layers if self.use_fourier_base else num_fourier_layers
        )
        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()
        for _ in range(num_fourier_components):
            self.spectral_convs.append(SpectralConv2d(width, width, modes1, modes2))
            self.conv_1x1s.append(nn.Conv2d(width, width, kernel_size=1))

        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            self.unet_modules.append(
                _UNet2DFromUNet3D(width, width, kernel_size=kernel_size)
            )

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

    def forward(
        self,
        x: Float[Tensor, "batch height width channels"],
    ) -> Float[Tensor, "batch height width out_channels"]:
        """Forward pass of the 2D spatial branch."""
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 4D input (B, H, W, C), got {x.ndim}D "
                    f"tensor with shape {tuple(x.shape)}"
                )
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)

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
                x, size=original_size, mode="bilinear", align_corners=True
            )

        return x.permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# 3D spatial branch
# ---------------------------------------------------------------------------


class SpatialBranch3D(nn.Module):
    r"""3D spatial branch composable from Fourier, UNet, and Conv layers.

    See :class:`SpatialBranch` for parameter semantics.  The 3D variant adds
    ``modes3`` for the third spectral axis.

    Forward
    -------
    x : torch.Tensor
        Channels-last input of shape :math:`(B, X, Y, Z, C)`.

    Outputs
    -------
    torch.Tensor
        Channels-last output of shape :math:`(B, X, Y, Z, D)` where
        :math:`D` equals ``width``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import SpatialBranch3D
    >>> branch = SpatialBranch3D(
    ...     in_channels=5, width=64, num_unet_layers=1, kernel_size=3
    ... )
    >>> x = torch.randn(1, 16, 16, 16, 5)   # (B, X, Y, Z, C)
    >>> out = branch(x)                      # (1, 16, 16, 16, 64)
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
        dropout: float = 0.0,  # noqa: ARG002 - kept for config compatibility
        activation_fn: str = "gelu",
        internal_resolution: list | None = None,
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

        self.lift = nn.LazyLinear(width)

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

        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            self.unet_modules.append(
                _UNet3DFromUNet3D(width, width, kernel_size=kernel_size)
            )

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

    def forward(
        self,
        x: Float[Tensor, "batch x y z channels"],
    ) -> Float[Tensor, "batch x y z out_channels"]:
        """Forward pass of the 3D spatial branch."""
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(
                    f"Expected 5D input (B, X, Y, Z, C), got {x.ndim}D "
                    f"tensor with shape {tuple(x.shape)}"
                )
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


__all__ = [
    "TrunkNet",
    "MLPBranch",
    "SpatialBranch",
    "SpatialBranch3D",
]
