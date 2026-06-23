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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from physicsnemo.core import ModelMetaData, Module


def nested_to(
    x: torch.Tensor | Mapping | list | tuple | Any, **kwargs
) -> torch.Tensor | dict | list | Any:
    """Move tensors inside a nested structure to a device / dtype."""
    if isinstance(x, Mapping):
        return {k: nested_to(v, **kwargs) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [nested_to(v, **kwargs) for v in x]
    if not isinstance(x, torch.Tensor):
        return x
    return x.to(**kwargs)


class FGNDiT(Module, register=True):
    r"""DiT-based backbone for FGN (arXiv:2506.10772 §2.3).

    Patchifies the 721×1440 input before the transformer, giving 16-64×
    memory reduction over full-resolution convolutions.  The latent noise
    vector ``z ~ N(0,I)^latent_dim`` conditions every layer via AdaLN-Zero
    (passed as ``condition`` to DiT), mirroring the paper's global conditional
    layer-norm.  A zero dummy timestep satisfies DiT's ``t`` argument; its
    positional embedding becomes a learned constant bias.

    Parameters
    ----------
    state_channels : int
        Number of prognostic channels C.
    history_frames : int, optional, default=2
        Number of past frames T concatenated as input.
    background_channels : int, optional, default=0
        Slowly-varying background channels (e.g. SST).
    invariant_channels : int, optional, default=0
        Static invariant channels (e.g. orography, land-sea mask).
    latent_dim : int, optional, default=32
        Dimension of z (paper §2.3 uses 32).
    input_height, input_width : int
        Spatial dimensions of the input grid (default 721×1440 for 0.25° ERA5).
    patch_size : int or (int, int), optional, default=(4, 4)
        Spatial patch size.  (4,4) → 181×360 = 65k tokens (16× compression).
    hidden_size : int, optional, default=384
        Transformer hidden dimension.
    depth : int, optional, default=12
        Number of transformer layers.
    num_heads : int, optional, default=8
        Number of attention heads.
    """

    def __init__(
        self,
        state_channels: int,
        history_frames: int = 2,
        background_channels: int = 0,
        invariant_channels: int = 0,
        latent_dim: int = 32,
        input_height: int = 721,
        input_width: int = 1440,
        patch_size: int | tuple[int, int] = (4, 4),
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 8,
    ):
        from physicsnemo.models.dit import DiT

        super().__init__(meta=ModelMetaData())
        self.state_channels = state_channels
        self.history_frames = history_frames
        self.background_channels = background_channels
        self.invariant_channels = invariant_channels
        self.latent_dim = latent_dim

        in_channels = (
            history_frames * state_channels + background_channels + invariant_channels
        )
        ps = (
            tuple(patch_size)
            if isinstance(patch_size, (list, tuple))
            else (patch_size, patch_size)
        )
        # PatchEmbed2D allocates pos_embed with floor(H/ps) but pads at runtime to
        # ceil(H/ps), causing a token-count mismatch for non-divisible grids (e.g.
        # ERA5 721×1440 with ps=4).  Pre-pad here so DiT receives an already-divisible
        # input and never triggers its internal padding path.
        pad_h = (-input_height) % ps[0]
        pad_w = (-input_width) % ps[1]
        self._pad_top = pad_h // 2
        self._pad_bottom = pad_h - self._pad_top
        self._pad_left = pad_w // 2
        self._pad_right = pad_w - self._pad_left
        self._crop_h = input_height
        self._crop_w = input_width
        self.backbone = DiT(
            input_size=(input_height + pad_h, input_width + pad_w),
            in_channels=in_channels,
            out_channels=state_channels,
            patch_size=ps,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            condition_dim=latent_dim,
            conditioning_embedder="dit",
        )

    def forward(
        self,
        history: torch.Tensor,
        latent: torch.Tensor,
        background: torch.Tensor | None = None,
        invariants: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 5:
            raise ValueError("history must have shape [B, T, C, H, W]")
        batch, frames, channels, height, width = history.shape
        if frames != self.history_frames or channels != self.state_channels:
            raise ValueError("history shape does not match model configuration")

        pieces = [history.reshape(batch, frames * channels, height, width)]
        if background is not None:
            pieces.append(background)
        if invariants is not None:
            pieces.append(invariants)
        x = torch.cat(pieces, dim=1)

        if self._pad_top or self._pad_bottom or self._pad_left or self._pad_right:
            x = F.pad(
                x, (self._pad_left, self._pad_right, self._pad_top, self._pad_bottom)
            )

        # t=0 dummy: timestep embedding becomes a learned constant bias;
        # all stochasticity comes from latent z via AdaLN-Zero.
        t = torch.zeros(batch, device=x.device, dtype=torch.float32)
        out = self.backbone(x, t, condition=latent)
        return out[
            ...,
            self._pad_top : self._pad_top + self._crop_h,
            self._pad_left : self._pad_left + self._crop_w,
        ]


def build_model(
    cfg,
    state_channels: int,
    background_channels: int,
    invariant_channels: int,
) -> FGNDiT:
    if cfg.model.background_channels not in ("auto", background_channels):
        raise ValueError("config model.background_channels disagrees with dataset")
    if cfg.model.invariant_channels not in ("auto", invariant_channels):
        raise ValueError("config model.invariant_channels disagrees with dataset")

    ps = list(cfg.model.patch_size) if hasattr(cfg.model, "patch_size") else [4, 4]
    return FGNDiT(
        state_channels=state_channels,
        history_frames=int(cfg.model.history_frames),
        background_channels=background_channels,
        invariant_channels=invariant_channels,
        latent_dim=int(cfg.model.latent_dim),
        patch_size=ps,
        hidden_size=int(cfg.model.hidden_size),
        depth=int(cfg.model.depth),
        num_heads=int(cfg.model.num_heads),
    )
