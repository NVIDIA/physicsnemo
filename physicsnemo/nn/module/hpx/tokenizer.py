# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

import importlib
import math
from typing import Optional

import einops
import torch
import torch.nn as nn

from physicsnemo.core.version_check import check_version_spec

HEALPIXPAD_AVAILABLE = check_version_spec("earth2grid", "0.1.0", hard_fail=False)

if HEALPIXPAD_AVAILABLE:
    hpx_grid = importlib.import_module("earth2grid.healpix").Grid
    HEALPIX_PAD_XY = importlib.import_module("earth2grid.healpix").HEALPIX_PAD_XY
else:
    HEALPIX_PAD_XY = None

    def hpx_grid(*args, **kwargs):
        """Dummy symbol for missing earth2grid backend."""
        raise ImportError(
            (
                "earth2grid is not installed, cannot use it as a backend for HEALPix padding.\n"
                "Install earth2grid from https://github.com/NVlabs/earth2grid.git to enable the accelerated path.\n"
                "pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz"
            )
        )


class HEALPixPatchTokenizer(nn.Module):
    r"""
    ViT-style tokenizer for HEALPix data.

    Tokenizes each HEALPix face into a patch sequence with a learnable positional
    embedding and a calendar embedding.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_size : int
        Number of output embedding channels (token dimension).
    level_fine : int
        HEALPix resolution level of input data.
    level_coarse : int
        HEALPix resolution level after patch embedding (model level).

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, T, N_{pix})` where
        :math:`N_{pix} = 12 \\times 4^{\\mathrm{level}_{fine}}`. Must have
        HEALPIX_PAD_XY pixel order.
    second_of_day : torch.Tensor
        Second-of-day tensor of shape :math:`(B, T)` for calendar embedding.
    day_of_year : torch.Tensor
        Day-of-year tensor of shape :math:`(B, T)` for calendar embedding.

    Outputs
    -------
    torch.Tensor
        Token tensor of shape :math:`(B, L, D)` where
        :math:`L = T \\times 12 \\times 4^{\\mathrm{level}_{coarse}}` and
        :math:`D=\\mathrm{hidden\\_size}`. In HEALPIX_PAD_XY pixel order.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        level_fine: int,
        level_coarse: int,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.level_fine = level_fine
        self.level_coarse = level_coarse
        self.nside = 2**level_fine
        self.nside_coarse = 2**level_coarse
        self.patch_size = 2 ** (level_fine - level_coarse)

        self.conv = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Global positional embedding
        npix_coarse = 12 * 4**level_coarse
        self.pos_embed = nn.Parameter(torch.randn(npix_coarse, hidden_size))

        # Calendar embedding
        grid = hpx_grid(level=level_coarse, pixel_order=HEALPIX_PAD_XY)
        lon = torch.as_tensor(grid.lon)
        if hidden_size % 4 != 0:
            raise ValueError(f"hidden_size must be divisible by 4, got {hidden_size}")
        self.calendar_embed = CalendarEmbedding(lon, hidden_size // 4).float()

    def initialize_weights(self) -> None:
        pass

    def forward(
        self,
        x: torch.Tensor,
        second_of_day: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> torch.Tensor:
        b, c, t, npix = x.shape

        # Fold faces into batch for per-face convolution.
        x = einops.rearrange(
            x,
            "b c t (f x y) -> (b t f) c x y",
            f=12,
            x=self.nside,
            y=self.nside,
        )

        x = self.conv(x)
        x = einops.rearrange(
            x,
            "(b t f) c x y -> b t (f x y) c",
            b=b,
            t=t,
            f=12,
        )

        calendar_emb = self.calendar_embed(
            second_of_day=second_of_day, day_of_year=day_of_year
        )  # (b, c, t, x)
        calendar_emb = einops.rearrange(calendar_emb, "b c t x -> b t x c")

        x = x + calendar_emb + self.pos_embed

        # Unfold into a token sequence.
        x = einops.rearrange(
            x,
            "b t (f x y) c -> b (t f x y) c",
            t=t,
            f=12,
            x=self.nside_coarse,
            y=self.nside_coarse,
        )

        return x


class HEALPixPatchDetokenizer(nn.Module):
    r"""
    HEALPix patch detokenizer for DiT integration.

    Upsamples HEALPix patch tokens back to the full-resolution grid using a transpose convolution.

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
    condition_dim : int, optional, default=None
        Conditioning dimension for AdaLN modulation. If None, uses ``hidden_size``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, D)` where
        :math:`L = T \\times 12 \\times 4^{\\mathrm{level}_{coarse}}`. Must have
        HEALPIX_PAD_XY pixel order.
    c : torch.Tensor
        Conditioning tensor of shape :math:`(B, D_c)` where :math:`D_c` is
        ``condition_dim`` if provided, otherwise ``hidden_size``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, T, N_{pix})` where
        :math:`N_{pix} = 12 \\times 4^{\\mathrm{level}_{fine}}`.
        In HEALPIX_PAD_XY pixel order.
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
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.level_coarse = level_coarse
        self.level_fine = level_fine
        self.time_length = time_length
        self.nside_coarse = 2**level_coarse
        self.patch_size = 2 ** (level_fine - level_coarse)

        modulation_input_dim = hidden_size if condition_dim is None else condition_dim
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(modulation_input_dim, 2 * hidden_size),
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.conv_t = nn.ConvTranspose2d(
            hidden_size,
            out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def initialize_weights(self) -> None:
        nn.init.constant_(self.adaptive_modulation[-1].weight, 0)
        nn.init.constant_(self.adaptive_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = self.time_length
        n = self.nside_coarse

        x = einops.rearrange(x, "b (t f x y) d -> b t (f x y) d", t=t, f=12, x=n, y=n)

        shift, scale = self.adaptive_modulation(c).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]

        x = einops.rearrange(x, "b t (f x y) d -> (b t f) d x y", f=12, x=n, y=n)

        x = self.conv_t(x)

        x = einops.rearrange(x, "(b t f) c x y -> b c t (f x y)", f=12, b=b, t=t)
        return x


class FrequencyEmbedding(nn.Module):
    r"""
    Periodic embedding using sinusoidal features. Useful for inputs defined on the circle [0, 2π).

    Parameters
    ----------
    num_channels : int
        Number of frequency bands to use.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, T, X)`.

    Outputs
    -------
    torch.Tensor
        Embedded tensor of shape :math:`(B, 2C, T, X)` where
        :math:`C = \\mathrm{num\\_channels}`.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer(
            "freqs", torch.arange(1, num_channels + 1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs[None, :, None, None]
        x = x[:, None, :, :]
        x = x * (2 * math.pi * freqs).to(x.dtype)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CalendarEmbedding(nn.Module):
    r"""
    Calendar embedding using day-of-year and local solar time. Assumes 365.25 day years.

    Parameters
    ----------
    lon : torch.Tensor
        Longitude values in degrees of shape :math:`(X,)`.
    embed_channels : int
        Number of frequency channels for each component.
    include_legacy_bug : bool, optional, default=False
        If True, uses the legacy local-time formula (``hour - lon``).

    Forward
    -------
    day_of_year : torch.Tensor
        Day-of-year tensor of shape :math:`(B, T)`.
    second_of_day : torch.Tensor
        Second-of-day tensor of shape :math:`(B, T)`.

    Outputs
    -------
    torch.Tensor
        Calendar embedding of shape :math:`(B, 4C, T, X)` where
        :math:`C = \\mathrm{embed\\_channels}`.
    """

    def __init__(
        self,
        lon: torch.Tensor,
        embed_channels: int,
        include_legacy_bug: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("lon", lon, persistent=False)
        self.embed_channels = embed_channels
        self.embed_second = FrequencyEmbedding(embed_channels)
        self.embed_day = FrequencyEmbedding(embed_channels)
        self.out_channels = embed_channels * 4
        self.include_legacy_bug = include_legacy_bug

    def forward(
        self,
        day_of_year: torch.Tensor,
        second_of_day: torch.Tensor,
    ) -> torch.Tensor:
        if second_of_day.shape != day_of_year.shape:
            raise ValueError()

        if self.include_legacy_bug:
            local_time = (second_of_day.unsqueeze(2) - self.lon * 86400 // 360) % 86400
        else:
            local_time = (second_of_day.unsqueeze(2) + self.lon * 86400 // 360) % 86400

        a = self.embed_second(local_time / 86400)
        doy = day_of_year.unsqueeze(2)
        b = self.embed_day((doy / 365.25) % 1)
        a, b = torch.broadcast_tensors(a, b)
        return torch.concat([a, b], dim=1)
