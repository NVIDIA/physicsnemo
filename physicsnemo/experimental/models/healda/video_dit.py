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
"""Video DiT: a HEALPix field-sequence diffusion transformer over ``(B, C, T, X)``.

Follows the DiT template -- tokenize, condition, transformer blocks, detokenize --
on an explicit time axis. Reuses the existing HEALPix patch tokenizer /
detokenizer (which already fold the time axis and add a calendar embedding) and
composes :class:`.video_dit_block.VideoDiTBlock` (spatial attention + optional
factorized temporal attention + optional observation cross-attention). The flat
tokenizer sequence is reshaped to ``(B, T, X, hidden_size)`` for the blocks and
back for the detokenizer.

This is a diffusion model: ``noise_labels`` drive an EDM conditioning embedder
feeding the blocks' adaLN-Zero modulation.
"""

from typing import Any, Dict, Optional

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.nn import ConditioningEmbedderType, get_conditioning_embedder
from physicsnemo.nn.module.hpx.tokenizer import (
    HEALPixPatchDetokenizer,
    HEALPixPatchTokenizer,
)

from .obs_packing import ObsCrossAttention
from .video_dit_block import VideoDiTBlock


class VideoDiT(nn.Module):
    r"""HEALPix field-sequence diffusion transformer with temporal + obs attention.

    Parameters
    ----------
    in_channels : int
        Number of input field channels.
    out_channels : int
        Number of output field channels.
    level_fine : int
        HEALPix resolution level of the input/output grid.
    level_coarse : int
        HEALPix model (token) resolution level after patch embedding.
    time_length : int
        Number of time steps :math:`T`.
    hidden_size : int
        Transformer token dimension.
    num_heads : int
        Number of spatial-attention heads.
    num_layers : int
        Number of :class:`.video_dit_block.VideoDiTBlock` blocks.
    emb_channels : int, optional, default=None
        Conditioning-embedding dimension. Defaults to ``4 * hidden_size``.
    noise_channels : int, optional, default=None
        Noise positional-embedding dimension. Defaults to ``hidden_size``.
    condition_dim : int, optional, default=0
        Class-label condition dimension (0 = noise-only).
    temporal_attention : bool, optional, default=False
        Enable factorized temporal attention in every block.
    obs_cross_attention : bool, optional, default=False
        Enable observation cross-attention in every block (requires
        ``obs_token_dim`` and an ``obs`` input to :meth:`forward`).
    obs_token_dim : int, optional, default=None
        Observation-token feature dimension.
    attention_backend : str, optional, default="timm"
        Spatial-attention backend for the blocks.
    mlp_ratio : float, optional, default=4.0
        Block MLP hidden-dim multiplier.
    drop_path : float, optional, default=0.0
        Maximum stochastic-depth rate, scheduled linearly across blocks.
    temporal_kwargs : dict, optional, default=None
        Extra keyword arguments for the temporal-attention layers.
    block_kwargs : dict, optional, default=None
        Extra keyword arguments forwarded to every block.

    Forward
    -------
    x : torch.Tensor
        Field sequence of shape :math:`(B, C, T, N_{pix})`.
    noise_labels : torch.Tensor
        Diffusion noise levels of shape :math:`(B,)`.
    second_of_day : torch.Tensor
        Second-of-day of shape :math:`(B, T)` for the calendar embedding.
    day_of_year : torch.Tensor
        Day-of-year of shape :math:`(B, T)` for the calendar embedding.
    condition : torch.Tensor, optional
        Class-label condition of shape :math:`(B, \text{condition\_dim})`.
    obs : ObsCrossAttention, optional
        Packed observation tokens + ragged packing for the obs cross-attention.
    is_causal : bool, optional, default=False
        Causal masking for temporal attention.

    Outputs
    -------
    torch.Tensor
        Field sequence of shape :math:`(B, C_{out}, T, N_{pix})`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        level_fine: int,
        level_coarse: int,
        time_length: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        *,
        emb_channels: Optional[int] = None,
        noise_channels: Optional[int] = None,
        condition_dim: int = 0,
        temporal_attention: bool = False,
        obs_cross_attention: bool = False,
        obs_token_dim: Optional[int] = None,
        attention_backend: str = "timm",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        block_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.time_length = time_length
        self.npix_coarse = 12 * 4**level_coarse

        if emb_channels is None:
            emb_channels = 4 * hidden_size
        if noise_channels is None:
            noise_channels = hidden_size
        self.conditioning_embedder = get_conditioning_embedder(
            ConditioningEmbedderType.EDM,
            emb_channels=emb_channels,
            noise_channels=noise_channels,
            condition_dim=condition_dim,
        )
        cond_dim = self.conditioning_embedder.output_dim

        self.tokenizer = HEALPixPatchTokenizer(
            in_channels=in_channels,
            hidden_size=hidden_size,
            level_fine=level_fine,
            level_coarse=level_coarse,
        )
        self.detokenizer = HEALPixPatchDetokenizer(
            hidden_size=hidden_size,
            out_channels=out_channels,
            level_coarse=level_coarse,
            level_fine=level_fine,
            time_length=time_length,
            condition_dim=cond_dim,
        )

        drop_path_schedule = [
            drop_path * i / max(1, num_layers - 1) for i in range(num_layers)
        ]
        self.blocks = nn.ModuleList(
            [
                VideoDiTBlock(
                    hidden_size,
                    num_heads,
                    emb_channels=cond_dim,
                    attention_backend=attention_backend,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_schedule[i],
                    temporal_attention=temporal_attention,
                    temporal_kwargs=temporal_kwargs,
                    obs_cross_attention=obs_cross_attention,
                    obs_token_dim=obs_token_dim,
                    **(block_kwargs or {}),
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels time npix"],
        noise_labels: Float[torch.Tensor, " batch"],
        second_of_day: Float[torch.Tensor, "batch time"],
        day_of_year: Float[torch.Tensor, "batch time"],
        condition: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
        obs: Optional[ObsCrossAttention] = None,
        is_causal: bool = False,
    ) -> Float[torch.Tensor, "batch out_channels time npix"]:
        # (B, C, T, npix) -> (B, T * npix_coarse, hidden) -> (B, T, npix_coarse, hidden)
        tokens = self.tokenizer(x, second_of_day, day_of_year)
        h = einops.rearrange(tokens, "b (t x) d -> b t x d", t=self.time_length)

        emb = self.conditioning_embedder(noise_labels, condition=condition)
        for block in self.blocks:
            h = block(h, emb, obs=obs, is_causal=is_causal)

        h = einops.rearrange(h, "b t x d -> b (t x) d")
        return self.detokenizer(h, emb)
