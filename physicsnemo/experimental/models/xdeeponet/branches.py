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

"""Spatial branch building block used by the xDeepONet family.

Provides a single dimension-generic spatial encoder:

- :class:`SpatialBranch` — composable from Fourier, UNet, and Conv layers,
  parameterized by ``dimension`` (``2`` or ``3``) to operate on either
  :math:`(B, H, W, C)` or :math:`(B, X, Y, Z, C)` inputs.  Per-dimension
  primitives are dispatched through the module-level :data:`_DIM_LAYERS`
  lookup table.

The MLP trunk and the optional MLP branch are built directly from
:class:`physicsnemo.models.mlp.FullyConnected` by the helpers in
``deeponet.py`` (``_build_trunk_mlp`` and ``_build_mlp_branch``).

UNet sub-modules inside the spatial branch use
:class:`physicsnemo.models.unet.UNet` (3D).  A small adapter
:class:`_UNet2DFromUNet3D` is provided locally for the 2D path: it wraps
the 3D UNet with a singleton time dimension so the same library model covers
both spatial dimensionalities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.unet import UNet as _PhysicsNeMoUNet
from physicsnemo.nn import SpectralConv2d, SpectralConv3d, get_activation

# Per-dimension layer lookup table used by :class:`SpatialBranch` to dispatch
# spectral / conv / pooling / UNet primitives without code duplication.  The
# UNet adapter entries are populated lazily below (after the adapter classes
# are defined) so this module remains importable in any order.
_DIM_LAYERS: dict[int, dict] = {
    2: {
        "SpectralConv": SpectralConv2d,
        "Conv": nn.Conv2d,
        "BatchNorm": nn.BatchNorm2d,
        "AdaptiveAvgPool": nn.AdaptiveAvgPool2d,
        "interp_mode": "bilinear",
        "default_modes": (12, 12),
    },
    3: {
        "SpectralConv": SpectralConv3d,
        "Conv": nn.Conv3d,
        "BatchNorm": nn.BatchNorm3d,
        "AdaptiveAvgPool": nn.AdaptiveAvgPool3d,
        "interp_mode": "trilinear",
        "default_modes": (10, 10, 8),
    },
}


def _channel_first_permute(dimension: int) -> tuple[int, ...]:
    """Permutation that moves the channels axis from the last position
    (``(B, *spatial, C)``) to immediately after the batch dim
    (``(B, C, *spatial)``)."""
    return (0, dimension + 1, *range(1, dimension + 1))


def _channel_last_permute(dimension: int) -> tuple[int, ...]:
    """Inverse of :func:`_channel_first_permute`."""
    return (0, *range(2, dimension + 2), 1)


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
    for skip-connection reuse inside :class:`SpatialBranch` (``dimension=3``).
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


# Populate the UNet adapter entries now that the adapter classes are
# defined; keeps the lookup table self-contained for callers below.
_DIM_LAYERS[2]["UNetAdapter"] = _UNet2DFromUNet3D
_DIM_LAYERS[3]["UNetAdapter"] = _UNet3DFromUNet3D


# ---------------------------------------------------------------------------
# Spatial branch (dimension-generic)
# ---------------------------------------------------------------------------


@dataclass
class _SpatialBranchMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`SpatialBranch`."""


class SpatialBranch(Module):
    r"""Dimension-generic spatial branch composable from Fourier, UNet, and
    Conv layers.

    Operates on 2D :math:`(B, H, W, C)` or 3D :math:`(B, X, Y, Z, C)` inputs
    selected via the ``dimension`` constructor argument; the spectral /
    convolutional / pooling / UNet sub-modules are dispatched through the
    module-level :data:`_DIM_LAYERS` lookup table so no per-dimension
    subclasses are needed.  The branch can be configured to use any
    combination of spectral, UNet, and plain convolutional layers.  When
    Fourier layers are present (the "base" mode) UNet/Conv layers are
    added alongside the spectral path (hybrid residual).  When no Fourier
    layers are present UNet/Conv act as independent sequential layers.

    Parameters
    ----------
    dimension : int, optional
        Spatial dimensionality of the inputs.  Must be ``2`` (default) or
        ``3``.
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
        Fourier modes along the first two spatial axes.
    modes3 : int, optional
        Fourier modes along the third spatial axis.  Required when
        ``dimension == 3``; ignored when ``dimension == 2``.
    kernel_size : int
        Kernel size for UNet and Conv layers.
    activation_fn : str
        Activation function name.
    internal_resolution : list, optional
        If set, inputs are adaptively pooled to this resolution before
        processing and upsampled back, decoupling model size from grid size.
    coord_features : bool, optional
        When ``True``, concatenates ``dimension`` channels containing
        the per-axis normalized coordinates (each spanning :math:`[0, 1]`)
        to the input before the lift.  Useful for operator-learning
        architectures that don't carry coordinates through a trunk MLP
        (e.g. the xFNO family) and instead inject them as extra channels.
        Default ``False``.
    lift_layers : int, optional
        Number of layers in the lifting network (default ``1``, a single
        :class:`torch.nn.LazyLinear`).  When ``> 1`` the lift becomes a
        multi-layer pointwise MLP equivalent to a stack of 1x1 (1x1x1)
        convolutions.
    lift_hidden_width : int, optional
        Hidden width inside the multi-layer lift.  Only consulted when
        ``lift_layers > 1``; defaults to ``width // 2``.

    Attributes
    ----------
    modes_per_dim : tuple[int, ...]
        The Fourier mode counts the branch was built with, in spatial-axis
        order.  Length matches ``dimension``.

    Forward
    -------
    x : torch.Tensor
        Channels-last input of shape :math:`(B, H, W, C)` for
        ``dimension=2`` or :math:`(B, X, Y, Z, C)` for ``dimension=3``.

    Outputs
    -------
    torch.Tensor
        Channels-last output with the same spatial layout as the input and
        the channels dimension replaced by ``width``.

    Examples
    --------
    2D:

    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import SpatialBranch
    >>> branch = SpatialBranch(
    ...     dimension=2, in_channels=5, width=64, num_unet_layers=1, kernel_size=3
    ... )
    >>> x = torch.randn(2, 32, 32, 5)   # (B, H, W, C)
    >>> out = branch(x)                 # (2, 32, 32, 64)

    3D:

    >>> branch = SpatialBranch(
    ...     dimension=3, in_channels=5, width=64, num_unet_layers=1, kernel_size=3
    ... )
    >>> x = torch.randn(1, 16, 16, 16, 5)   # (B, X, Y, Z, C)
    >>> out = branch(x)                      # (1, 16, 16, 16, 64)

    With coordinate features (xFNO-style trunkless operator):

    >>> branch = SpatialBranch(
    ...     dimension=3, in_channels=5, width=64,
    ...     num_fourier_layers=4, modes1=12, modes2=12, modes3=8,
    ...     coord_features=True, lift_layers=2,
    ... )
    >>> x = torch.randn(1, 16, 16, 16, 5)   # (B, X, Y, Z, C)
    >>> out = branch(x)                      # (1, 16, 16, 16, 64)
    """

    def __init__(
        self,
        dimension: int = 2,
        in_channels: int = 12,
        width: int = 64,
        num_fourier_layers: int = 0,
        num_unet_layers: int = 0,
        num_conv_layers: int = 0,
        modes1: int = 12,
        modes2: int = 12,
        modes3: int | None = None,
        kernel_size: int = 3,
        activation_fn: str = "gelu",
        internal_resolution: list | None = None,
        coord_features: bool = False,
        lift_layers: int = 1,
        lift_hidden_width: int | None = None,
    ):
        super().__init__(meta=_SpatialBranchMetaData())

        if dimension not in _DIM_LAYERS:
            raise ValueError(
                f"SpatialBranch only supports dimension=2 or dimension=3, "
                f"got dimension={dimension!r}."
            )
        layers = _DIM_LAYERS[dimension]
        self.dimension = dimension

        if dimension == 3 and modes3 is None:
            modes3 = layers["default_modes"][2]
        modes_for_spec = (
            (modes1, modes2) if dimension == 2 else (modes1, modes2, modes3)
        )
        # Public attribute so downstream code (e.g.
        # :class:`DeepONet`'s time-axis-extend) can introspect the
        # branch's mode configuration.
        self.modes_per_dim: tuple[int, ...] = tuple(modes_for_spec)

        self.num_fourier_layers = num_fourier_layers
        self.num_unet_layers = num_unet_layers
        self.num_conv_layers = num_conv_layers
        self.use_fourier_base = num_fourier_layers > 0
        self.internal_resolution = (
            tuple(internal_resolution) if internal_resolution else None
        )
        self.coord_features = coord_features

        total_layers = num_fourier_layers + num_unet_layers + num_conv_layers
        if total_layers == 0:
            raise ValueError("SpatialBranch requires at least one layer type")

        if lift_layers < 1:
            raise ValueError(f"lift_layers must be >= 1, got {lift_layers}.")

        self.activation_fn = get_activation(activation_fn)

        if self.internal_resolution is not None:
            self.adaptive_pool = layers["AdaptiveAvgPool"](self.internal_resolution)

        # Lifting network: single LazyLinear by default, or a multi-layer
        # pointwise MLP when ``lift_layers > 1`` (equivalent to a stack of
        # 1x1 / 1x1x1 convolutions applied channels-last).
        if lift_layers == 1:
            self.lift: nn.Module = nn.LazyLinear(width)
        else:
            hidden = lift_hidden_width if lift_hidden_width is not None else width // 2
            stack: list[nn.Module] = [
                nn.LazyLinear(hidden),
                get_activation(activation_fn),
            ]
            for _ in range(lift_layers - 2):
                stack.extend([nn.Linear(hidden, hidden), get_activation(activation_fn)])
            stack.append(nn.Linear(hidden, width))
            self.lift = nn.Sequential(*stack)

        num_fourier_components = (
            total_layers if self.use_fourier_base else num_fourier_layers
        )
        SpectralConv = layers["SpectralConv"]
        Conv = layers["Conv"]
        BatchNorm = layers["BatchNorm"]
        UNetAdapter = layers["UNetAdapter"]

        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()
        for _ in range(num_fourier_components):
            self.spectral_convs.append(SpectralConv(width, width, *modes_for_spec))
            self.conv_1x1s.append(Conv(width, width, kernel_size=1))

        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            self.unet_modules.append(UNetAdapter(width, width, kernel_size=kernel_size))

        self.conv_modules = nn.ModuleList()
        padding = (kernel_size - 1) // 2
        for _ in range(num_conv_layers):
            self.conv_modules.append(
                nn.Sequential(
                    Conv(
                        width,
                        width,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    BatchNorm(width),
                )
            )

        # Cached so the forward path is dimension-agnostic.
        self._channel_first_permute = _channel_first_permute(dimension)
        self._channel_last_permute = _channel_last_permute(dimension)
        self._interp_mode = layers["interp_mode"]

    def _build_coord_features(self, x: Tensor) -> Tensor:
        """Build a channels-last coordinate-feature tensor matching ``x``.

        Returns a tensor of shape ``(B, *spatial, dimension)`` whose
        ``dimension`` trailing channels are the per-axis normalized
        coordinates in :math:`[0, 1]`.
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[1:-1]
        grids = [
            torch.linspace(0.0, 1.0, s, dtype=x.dtype, device=x.device)
            for s in spatial_shape
        ]
        mesh = torch.meshgrid(*grids, indexing="ij")
        coord = torch.stack(mesh, dim=-1)  # (*spatial, dimension)
        coord = coord.unsqueeze(0).expand(batch_size, *spatial_shape, self.dimension)
        return coord

    def forward(
        self,
        x: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        """Forward pass of the spatial branch (2D or 3D, selected at init)."""
        if not torch.compiler.is_compiling():
            expected_ndim = self.dimension + 2  # batch + spatial dims + channels
            if x.ndim != expected_ndim:
                raise ValueError(
                    f"Expected {expected_ndim}D input "
                    f"(B, {'H, W' if self.dimension == 2 else 'X, Y, Z'}, C), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}."
                )
        if self.coord_features:
            x = torch.cat([x, self._build_coord_features(x)], dim=-1)
        x = self.lift(x)
        x = x.permute(*self._channel_first_permute)

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
                x, size=original_size, mode=self._interp_mode, align_corners=True
            )

        return x.permute(*self._channel_last_permute)


__all__ = [
    "SpatialBranch",
]
