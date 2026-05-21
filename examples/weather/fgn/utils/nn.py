# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from physicsnemo.core import ModelMetaData, Module


def nested_to(
    x: torch.Tensor | Mapping | list | tuple | Any, **kwargs
) -> torch.Tensor | dict | list | Any:
    """Move tensors inside a nested structure to a device / dtype.

    Mirrors ``examples/weather/stormcast/utils/nn.nested_to`` so the two
    recipes share the same container-handling convention.
    """
    if isinstance(x, Mapping):
        return {k: nested_to(v, **kwargs) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [nested_to(v, **kwargs) for v in x]
    if not isinstance(x, torch.Tensor):
        return x
    return x.to(**kwargs)


class _ConditionalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, groups: int):
        super().__init__()
        norm_groups_in = min(groups, in_channels)
        norm_groups_out = min(groups, out_channels)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups_out, out_channels)
        self.cond = nn.Linear(cond_dim, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + residual


class FGNUNet(Module, register=True):
    r"""Latent-conditioned U-Net backbone for Functional Generative Networks (FGN).

    A shallow encoder-decoder U-Net whose decoder features are modulated by a
    per-step latent noise vector ``z``, implementing the stochastic generator
    :math:`G_\theta(x_{t-T:t}, z_t)` from arXiv:2506.10772 §2.1.

    Parameters
    ----------
    state_channels : int
        Number of prognostic channels :math:`C` (output channels = input state
        channels per frame).
    history_frames : int, optional, default=2
        Number of past frames :math:`T` concatenated as input.
    background_channels : int, optional, default=0
        Number of slowly-varying background channels (e.g. SST) appended to
        the encoder input.
    invariant_channels : int, optional, default=0
        Number of static invariant channels (e.g. land-sea mask, orography)
        appended to the encoder input.
    latent_dim : int, optional, default=16
        Dimensionality :math:`d_z` of the latent noise vector ``z``.
    hidden_channels : int, optional, default=32
        Base width :math:`H` of the U-Net. Channel counts at successive
        encoder levels are :math:`H`, :math:`2H`.
    group_norm_groups : int, optional, default=8
        Number of groups in all ``GroupNorm`` layers.

    Forward
    -------
    history : torch.Tensor
        Past state frames of shape :math:`(B, T, C, H_{in}, W_{in})`.
    latent : torch.Tensor
        Noise sample of shape :math:`(B, d_z)`.
    background : torch.Tensor, optional
        Background field of shape :math:`(B, C_{bg}, H_{in}, W_{in})`.
    invariants : torch.Tensor, optional
        Static invariants of shape :math:`(B, C_{inv}, H_{in}, W_{in})`.

    Outputs
    -------
    torch.Tensor
        Predicted next state of shape :math:`(B, C, H_{in}, W_{in})`.

    Examples
    --------
    >>> import torch
    >>> model = FGNUNet(state_channels=4, history_frames=2, latent_dim=8, hidden_channels=16)
    >>> history = torch.randn(2, 2, 4, 32, 48)
    >>> latent = torch.randn(2, 8)
    >>> out = model(history=history, latent=latent)
    >>> out.shape
    torch.Size([2, 4, 32, 48])
    """

    def __init__(
        self,
        state_channels: int,
        history_frames: int = 2,
        background_channels: int = 0,
        invariant_channels: int = 0,
        latent_dim: int = 16,
        hidden_channels: int = 32,
        group_norm_groups: int = 8,
    ):
        super().__init__(meta=ModelMetaData())
        self.state_channels = state_channels
        self.history_frames = history_frames
        self.background_channels = background_channels
        self.invariant_channels = invariant_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.group_norm_groups = group_norm_groups

        input_channels = history_frames * state_channels
        input_channels += background_channels + invariant_channels

        cond_dim = hidden_channels * 4
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.stem = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.down1 = _ConditionalResidualBlock(
            hidden_channels, hidden_channels, cond_dim, group_norm_groups
        )
        self.down2 = _ConditionalResidualBlock(
            hidden_channels, hidden_channels * 2, cond_dim, group_norm_groups
        )
        self.bottleneck = _ConditionalResidualBlock(
            hidden_channels * 2, hidden_channels * 2, cond_dim, group_norm_groups
        )
        self.up1 = _ConditionalResidualBlock(
            hidden_channels * 4, hidden_channels * 2, cond_dim, group_norm_groups
        )
        self.up2 = _ConditionalResidualBlock(
            hidden_channels * 3, hidden_channels, cond_dim, group_norm_groups
        )
        self.head = nn.Conv2d(hidden_channels, state_channels, kernel_size=1)

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

        cond = self.latent_mlp(latent)

        stem = self.stem(x)
        skip1 = self.down1(stem, cond)
        x = F.avg_pool2d(skip1, kernel_size=2)
        skip2 = self.down2(x, cond)
        x = F.avg_pool2d(skip2, kernel_size=2)
        x = self.bottleneck(x, cond)

        x = F.interpolate(
            x, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up1(torch.cat([x, skip2], dim=1), cond)
        x = F.interpolate(
            x, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.up2(torch.cat([x, skip1], dim=1), cond)
        return self.head(x)


def build_model(
    cfg,
    state_channels: int,
    background_channels: int,
    invariant_channels: int,
) -> FGNUNet:
    if cfg.model.background_channels not in ("auto", background_channels):
        raise ValueError("config model.background_channels disagrees with dataset")
    if cfg.model.invariant_channels not in ("auto", invariant_channels):
        raise ValueError("config model.invariant_channels disagrees with dataset")

    return FGNUNet(
        state_channels=state_channels,
        history_frames=int(cfg.model.history_frames),
        background_channels=background_channels,
        invariant_channels=invariant_channels,
        latent_dim=int(cfg.model.latent_dim),
        hidden_channels=int(cfg.model.hidden_channels),
        group_norm_groups=int(cfg.model.group_norm_groups),
    )
