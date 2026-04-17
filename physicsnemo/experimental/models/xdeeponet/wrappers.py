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

from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.experimental.models.xdeeponet.deeponet import DeepONet, DeepONet3D
from physicsnemo.experimental.models.xdeeponet.padding import (
    compute_right_pad_to_multiple,
    pad_spatial_right,
)


class DeepONetWrapper(nn.Module):
    """2D xDeepONet wrapper with automatic padding and input extraction.

    Input
    -----
    ``x`` : Tensor of shape ``(B, H, W, T, C)``.

    Output
    ------
    Tensor of shape ``(B, H, W, T_out)`` where ``T_out == T`` unless
    ``target_times`` is provided (then ``T_out == len(target_times)``).

    Parameters
    ----------
    padding : int
        Minimum right-side padding; the wrapper rounds up to the next
        multiple of 8.  Default is 8.
    variant : str
        xDeepONet variant (see
        :attr:`~physicsnemo.experimental.models.xdeeponet.deeponet.DeepONet.VALID_VARIANTS`).
    width : int
        Latent width.
    branch1_config, branch2_config, trunk_config : dict, optional
        Sub-network configurations (see core class docstrings).  The trunk
        config may additionally specify ``input_type`` as ``"time"`` or
        ``"grid"``: ``"time"`` uses the last input channel as the time
        coordinate; ``"grid"`` uses the last three channels
        ``(grid_x, grid_y, grid_t)``.
    decoder_type : {"mlp", "conv", "temporal_projection"}
        See :class:`~physicsnemo.experimental.models.xdeeponet.deeponet.DeepONet`.
    decoder_width, decoder_layers : int
        Decoder hidden width and layer count.
    decoder_activation_fn : str
        Activation function name for the decoder.
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
        )
        self._temporal_projection = self.model._temporal_projection

    def set_output_window(self, K: int):
        """Delegate to the inner :class:`DeepONet` model."""
        self.model.set_output_window(K)

    def forward(
        self,
        x: Tensor,
        x_branch2: Tensor = None,
        target_times: Tensor = None,
    ) -> Tensor:
        """Forward pass through the 2D wrapper.

        Parameters
        ----------
        x : Tensor
            Input ``(B, H, W, T_in, C)``.
        x_branch2 : Tensor, optional
            Secondary branch input (MIONet/TNO variants).
        target_times : Tensor, optional
            Explicit trunk query coordinates ``(K,)`` or ``(K, 1)``.  When
            provided the trunk evaluates at these K points instead of
            extracting time values from ``x``, enabling autoregressive
            temporal bundling where ``K != T_in``.

        Returns
        -------
        Tensor
            ``(B, H, W, T_out)`` where ``T_out = K`` if ``target_times`` is
            given, else ``T_in``.
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


class DeepONet3DWrapper(nn.Module):
    """3D xDeepONet wrapper with automatic padding and input extraction.

    Input
    -----
    ``x`` : Tensor of shape ``(B, X, Y, Z, T, C)``.

    Output
    ------
    Tensor of shape ``(B, X, Y, Z, T_out)`` where ``T_out == T`` unless
    ``target_times`` is provided.

    See :class:`DeepONetWrapper` for parameter semantics.  The 3D trunk
    ``input_type="grid"`` uses the last four input channels
    ``(grid_x, grid_y, grid_z, grid_t)``.
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
        )
        self._temporal_projection = self.model._temporal_projection

    def set_output_window(self, K: int):
        """Delegate to the inner :class:`DeepONet3D` model."""
        self.model.set_output_window(K)

    def forward(
        self,
        x: Tensor,
        x_branch2: Tensor = None,
        target_times: Tensor = None,
    ) -> Tensor:
        """Forward pass through the 3D wrapper.

        Parameters
        ----------
        x : Tensor
            Input ``(B, X, Y, Z, T_in, C)``.
        x_branch2 : Tensor, optional
            Secondary branch input (MIONet/TNO variants).
        target_times : Tensor, optional
            Explicit trunk query coordinates ``(K,)`` or ``(K, 1)``.

        Returns
        -------
        Tensor
            ``(B, X, Y, Z, T_out)`` where ``T_out = K`` if ``target_times``
            is given, else ``T_in``.
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
