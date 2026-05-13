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

"""Convenience wrappers for xDeepONet.

These wrappers add two ergonomic features on top of the core
:class:`~physicsnemo.experimental.models.xdeeponet.deeponet.DeepONet` and
:class:`~physicsnemo.experimental.models.xdeeponet.deeponet.DeepONet3D`:

1. **Automatic spatial padding** to align the input to a multiple (default
   8), which makes the Fourier, UNet, and Conv sub-branches compatible
   across arbitrary grid sizes.  Outputs are cropped back to the original
   spatial shape before return.
2. **Automatic trunk input extraction** from the full spatiotemporal input
   tensor.  Given ``(B, H, W, T, C)`` (2D) or ``(B, X, Y, Z, T, C)`` (3D)
   and a ``target_times`` kwarg (optional), the wrapper assembles the
   trunk query coordinates according to the ``trunk.input_type`` setting
   (``"time"`` or ``"grid"``).

These wrappers are the recommended public entry points for xDeepONet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.xdeeponet.deeponet import (
    DeepONet,
    DeepONet3D,
    _DecoderTypeStr,
    _VariantStr,
)
from physicsnemo.experimental.models.xdeeponet._padding import (
    compute_right_pad_to_multiple,
    pad_spatial_right,
)


@dataclass
class _DeepONetWrapperMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`DeepONetWrapper`."""


@dataclass
class _DeepONet3DWrapperMetaData(ModelMetaData):
    """PhysicsNeMo model metadata for :class:`DeepONet3DWrapper`."""


class DeepONetWrapper(Module):
    r"""2D xDeepONet wrapper with automatic padding and input extraction.

    Extracts the spatial channels and trunk coordinates from a packed 5D
    input tensor, pads spatial dimensions to a multiple of 8, runs the
    core :class:`~physicsnemo.experimental.models.xdeeponet.deeponet.DeepONet`,
    and unpads.

    Parameters
    ----------
    padding : int, optional
        Minimum right-side padding; the wrapper rounds up to the next
        multiple of 8.
    variant : Literal["deeponet", "u_deeponet", "fourier_deeponet", "conv_deeponet", "hybrid_deeponet", "mionet", "fourier_mionet", "tno"], optional
        xDeepONet variant (see
        :attr:`~physicsnemo.experimental.models.xdeeponet.deeponet.DeepONet.VALID_VARIANTS`).
        Mixed-case strings are accepted at runtime and lowercased.
    width : int, optional
        Latent width.
    branch1_config : dict, optional
        Primary branch configuration (see core class docstring).
    branch2_config : dict, optional
        Secondary branch configuration for MIONet/TNO variants.
    trunk_config : dict, optional
        Trunk configuration.  May specify ``input_type`` as ``"time"``
        (uses the last input channel as the time coordinate) or
        ``"grid"`` (uses the last three channels
        ``(grid_x, grid_y, grid_t)``).
    decoder_type : Literal["mlp", "conv", "temporal_projection"], optional
        One of ``"mlp"``, ``"conv"``, or ``"temporal_projection"``;
        mixed-case strings are accepted and lowercased.
    decoder_width : int, optional
        Decoder hidden width.
    decoder_layers : int, optional
        Decoder layer count.
    decoder_activation_fn : str, optional
        Activation function name for the decoder.
    output_window : int, optional
        Output window for the ``"temporal_projection"`` decoder (forwarded
        to :class:`DeepONet`).

    Forward
    -------
    x : torch.Tensor
        Packed input of shape :math:`(B, H, W, T, C)` where the last
        channel axis holds features plus time/grid coordinates.
    x_branch2 : torch.Tensor, optional
        Secondary branch input for MIONet/TNO variants.
    target_times : torch.Tensor, optional
        Explicit trunk query coordinates of shape :math:`(K,)` or
        :math:`(K, 1)`.  When provided the trunk evaluates at these
        :math:`K` points instead of the times extracted from ``x``,
        enabling autoregressive temporal bundling where :math:`K \neq T`.

    Outputs
    -------
    torch.Tensor
        Operator output of shape :math:`(B, H, W, T_{out})` where
        :math:`T_{out} = K` when ``target_times`` is given and
        :math:`T_{out} = T` otherwise.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import DeepONetWrapper
    >>> model = DeepONetWrapper(
    ...     variant="u_deeponet",
    ...     width=64,
    ...     branch1_config={
    ...         "encoder": {"type": "linear", "activation_fn": "tanh"},
    ...         "layers": {"num_unet_layers": 1, "kernel_size": 3},
    ...     },
    ...     trunk_config={"input_type": "time", "hidden_width": 64, "num_layers": 4},
    ... )
    >>> x = torch.randn(2, 32, 32, 3, 5)   # (B, H, W, T, C)
    >>> out = model(x)                     # (2, 32, 32, 3)
    """

    def __init__(
        self,
        padding: int = 8,
        variant: _VariantStr = "u_deeponet",
        width: int = 64,
        branch1_config: dict[str, Any] | None = None,
        branch2_config: dict[str, Any] | None = None,
        trunk_config: dict[str, Any] | None = None,
        decoder_type: _DecoderTypeStr = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
        output_window: int | None = None,
    ):
        super().__init__(meta=_DeepONetWrapperMetaData())

        self.padding = ((padding + 7) // 8) * 8 if padding % 8 != 0 else padding
        self.variant = variant

        trunk_config = dict(trunk_config or {})
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
            output_window=output_window,
        )
    def set_output_window(self, K: int):
        """Delegate to the inner :class:`DeepONet` model."""
        self.model.set_output_window(K)

    def forward(
        self,
        x: Float[Tensor, "batch height width time channels"],
        x_branch2: Float[Tensor, "..."] | None = None,
        target_times: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "batch height width time_out"]:
        """Forward pass through the 2D wrapper.

        See class docstring for input/output shapes.  ``x_branch2`` and
        ``target_times`` accept multiple ranks (see Forward section); their
        strict shapes are validated at the top of this method under the
        :func:`torch.compiler.is_compiling` guard, so the jaxtyping
        annotation uses the unconstrained ``"..."`` shape for those.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 5:
                raise ValueError(
                    f"Expected 5D input (B, H, W, T, C), got {x.ndim}D "
                    f"tensor with shape {tuple(x.shape)}"
                )
            if target_times is not None and target_times.ndim not in (1, 2):
                raise ValueError(
                    f"Expected target_times to be 1D (K,) or 2D (K, 1), "
                    f"got {target_times.ndim}D tensor with shape "
                    f"{tuple(target_times.shape)}"
                )

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

        x_spatial = x[:, :, :, 0, :]

        if target_times is not None:
            if self.trunk_input == "grid":
                t_vals = (
                    target_times
                    if target_times.dim() == 1
                    else target_times.squeeze(-1)
                )
                spatial = x[0, 0, 0, 0, -3:-1]
                spatial_exp = spatial.unsqueeze(0).expand(t_vals.shape[0], -1)
                x_trunk = torch.cat([spatial_exp, t_vals.unsqueeze(-1)], dim=-1)
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


class DeepONet3DWrapper(Module):
    r"""3D xDeepONet wrapper with automatic padding and input extraction.

    See :class:`DeepONetWrapper` for parameter semantics.  The 3D trunk
    ``input_type="grid"`` uses the last four input channels
    ``(grid_x, grid_y, grid_z, grid_t)``.

    Forward
    -------
    x : torch.Tensor
        Packed input of shape :math:`(B, X, Y, Z, T, C)`.
    x_branch2 : torch.Tensor, optional
        Secondary branch input for MIONet/TNO variants.
    target_times : torch.Tensor, optional
        Explicit trunk query coordinates of shape :math:`(K,)` or
        :math:`(K, 1)`.

    Outputs
    -------
    torch.Tensor
        Operator output of shape :math:`(B, X, Y, Z, T_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.xdeeponet import DeepONet3DWrapper
    >>> model = DeepONet3DWrapper(
    ...     variant="tno",
    ...     width=64,
    ...     branch1_config={
    ...         "encoder": {"type": "linear", "activation_fn": "tanh"},
    ...         "layers": {"num_unet_layers": 1, "kernel_size": 3},
    ...     },
    ...     branch2_config={
    ...         "encoder": {"type": "linear", "activation_fn": "tanh"},
    ...         "layers": {"num_unet_layers": 1, "kernel_size": 3},
    ...     },
    ...     trunk_config={"input_type": "time", "hidden_width": 64, "num_layers": 4},
    ... )
    >>> x = torch.randn(1, 16, 16, 16, 2, 5)   # (B, X, Y, Z, T, C)
    >>> prev = torch.randn(1, 16, 16, 16, 1)   # previous solution (TNO branch2)
    >>> out = model(x, x_branch2=prev)          # (1, 16, 16, 16, 2)
    """

    def __init__(
        self,
        padding: int = 8,
        variant: _VariantStr = "u_deeponet",
        width: int = 64,
        branch1_config: dict[str, Any] | None = None,
        branch2_config: dict[str, Any] | None = None,
        trunk_config: dict[str, Any] | None = None,
        decoder_type: _DecoderTypeStr = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu",
        output_window: int | None = None,
    ):
        super().__init__(meta=_DeepONet3DWrapperMetaData())

        self.padding = ((padding + 7) // 8) * 8 if padding % 8 != 0 else padding
        self.variant = variant

        trunk_config = dict(trunk_config or {})
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
            output_window=output_window,
        )
    def set_output_window(self, K: int):
        """Delegate to the inner :class:`DeepONet3D` model."""
        self.model.set_output_window(K)

    def forward(
        self,
        x: Float[Tensor, "batch X Y Z time channels"],
        x_branch2: Float[Tensor, "..."] | None = None,
        target_times: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "batch X Y Z time_out"]:
        """Forward pass through the 3D wrapper.

        See class docstring for input/output shapes.  ``x_branch2`` and
        ``target_times`` accept multiple ranks; their strict shapes are
        validated at the top of this method under the
        :func:`torch.compiler.is_compiling` guard.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 6:
                raise ValueError(
                    f"Expected 6D input (B, X, Y, Z, T, C), got {x.ndim}D "
                    f"tensor with shape {tuple(x.shape)}"
                )
            if target_times is not None and target_times.ndim not in (1, 2):
                raise ValueError(
                    f"Expected target_times to be 1D (K,) or 2D (K, 1), "
                    f"got {target_times.ndim}D tensor with shape "
                    f"{tuple(target_times.shape)}"
                )

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
                spatial = x[0, 0, 0, 0, 0, -4:-1]
                spatial_exp = spatial.unsqueeze(0).expand(t_vals.shape[0], -1)
                x_trunk = torch.cat([spatial_exp, t_vals.unsqueeze(-1)], dim=-1)
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


__all__ = ["DeepONetWrapper", "DeepONet3DWrapper"]
