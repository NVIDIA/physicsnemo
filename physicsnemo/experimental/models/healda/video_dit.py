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
"""Video DiT: a field-sequence diffusion transformer over ``(B, C, T, X)``.

Follows the DiT template (tokenize, condition, transformer blocks, detokenize) on
an explicit time axis, composing :class:`.video_dit_block.VideoDiTBlock`. The
backbone is grid-agnostic: the grid is consumed only by the pluggable tokenizer /
detokenizer, which produce / consume a flat ``(B, T * X', D)`` token sequence the
model reshapes to ``(B, T, X', D)`` for the blocks.

It composes (rather than subclasses) :class:`physicsnemo.models.dit.DiT`, whose
2D-spatial ``input_size`` / ``patch_size`` and ``TokenizerModuleBase`` contract do
not fit the HEALPix pluggable-tokenizer case, but still inherits
:class:`physicsnemo.Module`. Observations enter as a prebuilt
:class:`.obs_packing.ObsCrossAttention` bundle consumed inside each block.
"""

from typing import Any, Dict, Optional

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import ConditioningEmbedderType, get_conditioning_embedder

from .obs_packing import ObsCrossAttention
from .video_dit_block import VideoDiTBlock


class VideoDiT(Module):
    r"""Grid-agnostic field-sequence diffusion transformer with temporal + obs attention.

    Parameters
    ----------
    tokenizer : torch.nn.Module
        Maps :math:`(B, C, T, X)` to a flat token sequence :math:`(B, T X', D)`
        (defines the grid); e.g. ``HEALPixPatchTokenizer``.
    detokenizer : torch.nn.Module
        Maps tokens :math:`(B, T X', D)` and the conditioning embedding back to
        :math:`(B, C_{out}, T, X)`.
    time_length : int
        Number of time steps :math:`T`, used to reshape the flat tokens to
        :math:`(B, T, X', D)` for the blocks.
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
        Enable observation cross-attention in every block (requires ``obs_kwargs``
        with ``obs_token_dim`` and an ``obs`` input to :meth:`forward`).
    obs_kwargs : dict, optional, default=None
        Obs cross-attention config forwarded to each block (see
        :class:`.video_dit_block.VideoDiTBlock`); needs ``obs_token_dim``.
    is_causal : bool, optional, default=False
        Causal masking for temporal attention, fixed at construction.
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
        Field sequence of shape :math:`(B, C, T, X)`.
    noise_labels : torch.Tensor
        Diffusion noise levels of shape :math:`(B,)`.
    condition : torch.Tensor, optional
        Class-label condition of shape :math:`(B, \text{condition\_dim})`.
    obs : ObsCrossAttention, optional
        Packed observation tokens + ragged packing for the obs cross-attention.
    tokenizer_kwargs : dict, optional
        Extra keyword arguments forwarded to the tokenizer's forward (e.g.
        ``second_of_day`` / ``day_of_year`` for the HEALPix tokenizer).

    Outputs
    -------
    torch.Tensor
        Field sequence of shape :math:`(B, C_{out}, T, X)`.
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        detokenizer: nn.Module,
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
        obs_kwargs: Optional[Dict[str, Any]] = None,
        is_causal: bool = False,
        attention_backend: str = "timm",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        block_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(meta=ModelMetaData())
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.time_length = time_length

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

        drop_path_schedule = [
            drop_path * i / max(1, num_layers - 1) for i in range(num_layers)
        ]
        self.blocks = nn.ModuleList(
            [
                VideoDiTBlock(
                    hidden_size,
                    num_heads,
                    condition_embed_dim=cond_dim,
                    attention_backend=attention_backend,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_schedule[i],
                    temporal_attention=temporal_attention,
                    temporal_kwargs=temporal_kwargs,
                    obs_cross_attention=obs_cross_attention,
                    obs_kwargs=obs_kwargs,
                    is_causal=is_causal,
                    **(block_kwargs or {}),
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels time space"],
        noise_labels: Float[torch.Tensor, " batch"],
        condition: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
        obs: Optional[ObsCrossAttention] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[torch.Tensor, "batch out_channels time space"]:
        # (B, C, T, X) -> (B, T * X', hidden) -> (B, T, X', hidden)
        tokens = self.tokenizer(x, **(tokenizer_kwargs or {}))
        h = einops.rearrange(tokens, "b (t x) d -> b t x d", t=self.time_length)

        emb = self.conditioning_embedder(noise_labels, condition=condition)
        for block in self.blocks:
            h = block(h, emb, obs=obs)

        h = einops.rearrange(h, "b t x d -> b (t x) d")
        return self.detokenizer(h, emb)
