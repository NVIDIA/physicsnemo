# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HEALPix tokenization and detokenization layers for HealDA."""

from typing import Optional

import earth2grid
import earth2grid.healpix
import einops
import torch
import torch.nn as nn

from physicsnemo.experimental.models.dit.layers import (
    DetokenizerModuleBase,
    TokenizerModuleBase,
)

from .embedding import CalendarEmbedding


class HPXPatchTokenizer(TokenizerModuleBase):
    r"""
    HEALPix patch tokenizer for DiT integration.

    Folds 12 HEALPix faces into batch, applies conv, unfolds, adds global pos_embed + calendar_embed.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_size : int
        Number of output embedding channels.
    level_fine : int
        HEALPix resolution level of input data.
    level_coarse : int
        HEALPix resolution level after patch embedding (model level).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, T, N_{pix})` where :math:`N_{pix} = 12 \\times 4^{level\\_fine}`.
    second_of_day : torch.Tensor, optional
        Second of day for calendar embedding.
    day_of_year : torch.Tensor, optional
        Day of year for calendar embedding.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, D)` where :math:`L = T \\times 12 \\times patches\\_per\\_face`.
    """

    pixel_order = earth2grid.healpix.HEALPIX_PAD_XY

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        level_fine: int,
        level_coarse: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.level_fine = level_fine
        self.level_coarse = level_coarse
        self.nside = 2**level_fine
        self.patch_size = 2 ** (level_fine - level_coarse)

        # Patch embedding conv
        self.conv = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

        # Global positional embedding across all 12 faces
        npix_coarse = 12 * 4**level_coarse
        self.pos_embed = nn.Parameter(torch.randn(npix_coarse, hidden_size))

        # Calendar embedding (HEALPix-specific: incorporates longitude)
        grid = earth2grid.healpix.Grid(level=level_coarse, pixel_order=self.pixel_order)
        lon = torch.as_tensor(grid.lon)
        if hidden_size % 4 != 0:
            raise ValueError(f"hidden_size must be divisible by 4, got {hidden_size}")
        self.calendar_embed = CalendarEmbedding(lon, hidden_size // 4).float()

    def initialize_weights(self) -> None:
        w = self.conv.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.conv.bias, 0)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        second_of_day: Optional[torch.Tensor] = None,
        day_of_year: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, t, npix = x.shape

        # 1. Fold faces into batch: (B, C, T, 12*nside²) -> (B*T*12, C, nside, nside)
        x = einops.rearrange(x, "b c t (f x y) -> (b t f) c x y", f=12, x=self.nside, y=self.nside)

        # 2. Conv: (B*T*12, C, nside, nside) -> (B*T*12, D, nside_c, nside_c)
        x = self.conv(x)

        # 3. Flatten + unfold: (B*T*12, D, n_c, n_c) -> (B, T*12*n_c*n_c, D)
        x = einops.rearrange(x, "(b t f) d x y -> b (t f x y) d", b=b, t=t, f=12)

        # 4. Add global positional embedding
        x = x + self.pos_embed

        # 5. Add calendar embedding
        if second_of_day is not None and day_of_year is not None:
            calendar_emb = self.calendar_embed(second_of_day=second_of_day, day_of_year=day_of_year)
            calendar_emb = einops.rearrange(calendar_emb, "b d t x -> b (t x) d")
            x = x + calendar_emb

        return x


class HPXPatchDetokenizer(DetokenizerModuleBase):
    r"""
    HEALPix patch detokenizer for DiT integration.

    Applies final AdaLN modulation and conv transpose to upsample patches.

    Parameters
    ----------
    hidden_size : int
        Input embedding dimension.
    out_channels : int
        Number of output channels.
    level_coarse : int
        HEALPix resolution level of input patches.
    level_fine : int
        HEALPix resolution level of output data.
    time_length : int, optional, default=1
        Number of time steps.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)`.
    c : torch.Tensor
        Conditioning tensor of shape :math:`(B, D)`. Pass zeros for VIT mode.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, T, N_{pix})`.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        out_channels: int,
        level_coarse: int,
        level_fine: int,
        time_length: int = 1,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.level_coarse = level_coarse
        self.level_fine = level_fine
        self.time_length = time_length
        self.nside_coarse = 2**level_coarse
        self.patch_size = 2 ** (level_fine - level_coarse)

        # AdaLN: c -> (shift, scale)
        modulation_input_dim = hidden_size if condition_dim is None else condition_dim
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(modulation_input_dim, 2 * hidden_size),
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Conv transpose for upsampling
        self.conv_t = nn.ConvTranspose2d(
            hidden_size, out_channels,
            kernel_size=self.patch_size, stride=self.patch_size,
        )

    def initialize_weights(self) -> None:
        nn.init.constant_(self.adaptive_modulation[-1].weight, 0)
        nn.init.constant_(self.adaptive_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = self.time_length
        n = self.nside_coarse

        # 1. Unflatten: (B, L, D) -> (B, T, 12*n*n, D)
        x = einops.rearrange(x, "b (t f x y) d -> b t (f x y) d", t=t, f=12, x=n, y=n)

        # 2. AdaLN: norm(x) * (1 + scale) + shift
        shift, scale = self.adaptive_modulation(c).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]

        # 3. Fold faces: (B, T, 12*n*n, D) -> (B*T*12, D, n, n)
        x = einops.rearrange(x, "b t (f x y) d -> (b t f) d x y", f=12, x=n, y=n)

        # 4. Conv transpose
        x = self.conv_t(x)

        # 5. Unfold: (B*T*12, C, nside_fine, nside_fine) -> (B, C, T, npix)
        x = einops.rearrange(x, "(b t f) c x y -> b c t (f x y)", f=12, b=b, t=t)
        return x
