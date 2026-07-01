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
"""Video + observation data-assimilation model composing :class:`VideoDiT`."""

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import torch
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.healda.obs_context import ObsContext
from physicsnemo.experimental.models.healda.obs_tokenizer import ObsTokenizerFiLM
from physicsnemo.experimental.models.healda.pixel_cross_attention import (
    PixelCrossAttention,
)
from physicsnemo.experimental.models.healda.video_dit import VideoDiT
from physicsnemo.nn.module.hpx.tokenizer import (
    HEALPixPatchDetokenizer,
    HEALPixPatchTokenizer,
)


@dataclass
class HealDAv2MetaData(ModelMetaData):
    """Metadata for HealDAv2 model."""

    name: str = "HealDAv2"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    bf16: bool = True
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = False


class HealDAv2(Module):
    r"""Observation-conditioned video diffusion model for weather data assimilation on
    the HEALPix sphere.

    ``HealDAv2`` maps a set of sparse, irregularly located observations (plus
    static fields and calendar features) to a gridded field sequence -- a short
    video window of :math:`T` frames over a HEALPix grid.

    The model is conditioned on a diffusion noise level (``noise_labels``)
    through the adaLN modulation, so it can be trained as a diffusion model or as
    a deterministic regression model, where the ``noise_labels`` are fixed to zero.

    Data flows through four stages:

    1. Static conditioning fields are patch-tokenized from the fine ingest grid
       (``level_in``) down to the backbone grid (``level_model``) by
       :class:`physicsnemo.nn.module.hpx.tokenizer.HEALPixPatchTokenizer`.
    2. A :class:`.video_dit.VideoDiT` backbone processes the token sequence with
       spatial attention, factorized temporal attention, and adaLN-Zero
       conditioning built from the EDM noise embedding and the calendar
       (second-of-day / day-of-year) features.
    3. Observations are embedded per observation by
       :class:`.obs_tokenizer.ObsTokenizerFiLM` and assimilated inside every block
       by :class:`.pixel_cross_attention.PixelCrossAttention`: each grid pixel
       attends only to the observations that land on it (ragged, local
       cross-attention).
    4. :class:`physicsnemo.nn.module.hpx.tokenizer.HEALPixPatchDetokenizer` maps
       the backbone tokens back to the fine grid, producing the ``out_channels``
       output fields.

    The grid enters only at the boundaries: the patch tokenizer /
    detokenizer (stages 1 and 4) and, upstream in the dataloader, the assignment
    of a flat pixel index to each observation (which builds the ragged packing
    carried on ``obs``). Everything in between is grid-agnostic -- the backbone
    operates on a flat ``(B, T, X, C)`` token sequence and the observation
    cross-attention only needs each observation tagged with the pixel it belongs
    to. Adapting to a different grid therefore means swapping the tokenizer /
    detokenizer pair and the observation pixel-assignment step.

    Because spatial and temporal attention are factorized (each is independent
    along the axis the other mixes over), the model supports context parallelism:
    a group of :math:`N` GPUs reshards activations between time- and space-sharded
    layouts around each attention (see :mod:`.sharding`). :math:`N` must divide
    both the time and space extents, so the default ``time_length = 8`` caps it at
    8-way. Enable via :meth:`set_context_parallel`.

    Parameters
    ----------
    in_channels : int, optional, default=2
        Number of static conditioning channels (e.g. orography and land fraction).
    out_channels : int, optional, default=74
        Number of decoder output channels.
    hidden_size : int, optional, default=1536
        Transformer token dimension.
    num_layers : int, optional, default=32
        Number of :class:`.video_dit_block.VideoDiTBlock` blocks.
    num_heads : int, optional, default=16
        Number of spatial- and temporal-attention heads.
    mlp_ratio : float, optional, default=4.0
        Block MLP hidden-dim multiplier.
    level_in : int, optional, default=6
        HEALPix ingest resolution level (``npix = 12 * 4**level_in``).
    level_model : int, optional, default=5
        HEALPix backbone resolution level after patch embedding.
    time_length : int, optional, default=8
        Number of frames per video window.
    emb_channels : int, optional, default=128
        EDM conditioning-embedding dimension feeding every adaLN and the detokenizer.
    noise_channels : int, optional, default=128
        EDM noise positional-embedding dimension.
    condition_dim : int, optional, default=0
        Class-label condition dimension (0 = noise-only conditioning).
    temporal_attention : bool, optional, default=True
        Enable factorized temporal attention in every block.
    is_causal : bool, optional, default=True
        Causal masking for temporal attention, fixed at construction.
    linear_attention : bool, optional, default=True
        Use the softmax-free (linear) temporal attention variant.
    rope_base : int, optional, default=100
        Base frequency for the temporal-attention rotary position embedding.
    max_seq_len : int, optional, default=100
        Maximum sequence length for the temporal rotary-embedding cache.
    temporal_causal_window : int, optional, default=None
        Sliding causal lookback for temporal attention. ``None`` is unbounded.
    drop_path : float, optional, default=0.1
        Stochastic-depth rate applied to every block past the warmup blocks.
    drop_path_zero_first_n_blocks : int, optional, default=4
        Number of leading blocks forced to drop-path rate 0 to stabilize early
        training (the first blocks are prone to gradient spikes otherwise).
    qk_norm_type : Literal["RMSNorm", "LayerNorm"], optional, default="RMSNorm"
        Spatial-attention QK normalization type. ``None`` disables it.
    qk_norm_affine : bool, optional, default=False
        Whether spatial QK normalization layers use learnable affine parameters.
    attention_backend : str, optional, default="timm"
        Spatial-attention backend for the blocks.
    layernorm_backend : str, optional, default="torch"
        LayerNorm backend for the blocks' adaLN-Zero pre-norms.
    obs_token_dim : int, optional, default=32
        Observation token width produced by the FiLM tokenizer.
    obs_meta_dim : int, optional, default=50
        Dimension of per-observation float metadata features.
    obs_type_embed_dim : int, optional, default=4
        Dimension of the observation-type embedding.
    channel_embed_dim : int, optional, default=16
        Dimension of the channel embedding.
    platform_embed_dim : int, optional, default=8
        Dimension of the platform embedding. ``0`` disables platform embedding.
    obs_film_hidden_dim : int, optional, default=64
        Hidden dimension of the FiLM conditioning MLP.
    use_fused_obs_mlp : bool, optional, default=True
        Prefer the fused Triton FiLM backend when CUDA and triton are available.
    pixel_attn_n_q_heads : int, optional, default=64
        Number of query heads in the per-block observation cross-attention.
    pixel_attn_n_kv_heads : int, optional, default=2
        Number of key/value heads in the observation cross-attention.
    pixel_attn_head_dim : int, optional, default=32
        Per-head dimension of the observation cross-attention.
    pixel_attn_use_proj_bias : bool, optional, default=True
        Whether the cross-attention q/v/out projections use a bias.

    Forward
    -------
    x : torch.Tensor
        Static conditioning fields of shape :math:`(B, C_{in}, T, N_{pix})` with
        :math:`N_{pix} = 12 \times 4^{\mathrm{level\_in}}`.
    noise_labels : torch.Tensor
        EDM noise levels of shape :math:`(B,)`.
    second_of_day : torch.Tensor
        Second-of-day tensor of shape :math:`(B, T)` for the calendar embedding.
    day_of_year : torch.Tensor
        Day-of-year tensor of shape :math:`(B, T)` for the calendar embedding.
    obs : ObsContext
        Per-observation arrays (``values``, ``float_metadata``, ``obs_type``,
        ``channel``, ``platform``) plus the per-pixel ragged packing
        (``cu_seqlens_k``, ``max_seqlen_k``, optional ``group_map``), with pixel
        prefix sums over :math:`B \cdot T \cdot X'` and
        :math:`X' = 12 \times 4^{\mathrm{level\_model}}`. The tokenizer fills
        ``tokens`` internally; observations are sorted by flat pixel index.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, T, N_{pix})`.

    Notes
    -----
    Observations are passed as an :class:`.obs_context.ObsContext` whose tokens
    must be packed per pixel (observations sorted by flat pixel index); build the
    packing with :mod:`.pixel_cross_attention`. The FiLM tokenizer and pixel
    cross-attention run fused Triton kernels on CUDA.

    Only the default ``attention_backend="timm"`` currently supports the
    ``qk_norm_type="RMSNorm"`` with ``qk_norm_affine=False`` this model
    relies on for stable training.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 74,
        hidden_size: int = 1536,
        num_layers: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        level_in: int = 6,
        level_model: int = 5,
        time_length: int = 8,
        emb_channels: int = 128,
        noise_channels: int = 128,
        condition_dim: int = 0,
        temporal_attention: bool = True,
        is_causal: bool = True,
        linear_attention: bool = True,
        rope_base: int = 100,
        max_seq_len: int = 100,
        temporal_causal_window: Optional[int] = None,
        drop_path: float = 0.1,
        drop_path_zero_first_n_blocks: int = 4,
        qk_norm_type: Literal["RMSNorm", "LayerNorm"] | None = "RMSNorm",
        qk_norm_affine: bool = False,
        attention_backend: str = "timm",
        layernorm_backend: str = "torch",
        obs_token_dim: int = 32,
        obs_meta_dim: int = 50,
        obs_type_embed_dim: int = 4,
        channel_embed_dim: int = 16,
        platform_embed_dim: int = 8,
        obs_film_hidden_dim: int = 64,
        use_fused_obs_mlp: bool = True,
        pixel_attn_n_q_heads: int = 64,
        pixel_attn_n_kv_heads: int = 2,
        pixel_attn_head_dim: int = 32,
        pixel_attn_use_proj_bias: bool = True,
    ):
        super().__init__(meta=HealDAv2MetaData())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.level_in = level_in
        self.level_model = level_model
        self.time_length = time_length
        self.npix = 12 * 4**level_in

        self.obs_tokenizer = ObsTokenizerFiLM(
            meta_dim=obs_meta_dim,
            out_dim=obs_token_dim,
            obs_type_embed_dim=obs_type_embed_dim,
            channel_embed_dim=channel_embed_dim,
            platform_embed_dim=platform_embed_dim,
            hidden_dim=obs_film_hidden_dim,
            use_fused_mlp=use_fused_obs_mlp,
        )

        cross_attention = partial(
            PixelCrossAttention,
            hidden_size=hidden_size,
            token_dim=obs_token_dim,
            n_q_heads=pixel_attn_n_q_heads,
            n_kv_heads=pixel_attn_n_kv_heads,
            d_head=pixel_attn_head_dim,
            use_proj_bias=pixel_attn_use_proj_bias,
        )

        attn_kwargs = {"qk_norm_type": qk_norm_type} if qk_norm_type else {}
        if qk_norm_type:
            attn_kwargs["qk_norm_affine"] = qk_norm_affine

        temporal_kwargs = {
            "linear_attention": linear_attention,
            "rope_base": rope_base,
            "max_seq_len": max_seq_len,
            "causal_window": temporal_causal_window,
        }

        n_zero = min(drop_path_zero_first_n_blocks, num_layers)
        drop_path_rates = [0.0] * n_zero + [drop_path] * (num_layers - n_zero)

        tokenizer = HEALPixPatchTokenizer(
            in_channels=in_channels,
            hidden_size=hidden_size,
            level_fine=level_in,
            level_coarse=level_model,
            separate_time_axis=True,
        )
        detokenizer = HEALPixPatchDetokenizer(
            hidden_size=hidden_size,
            out_channels=out_channels,
            level_coarse=level_model,
            level_fine=level_in,
            time_length=time_length,
            condition_dim=emb_channels,
        )
        self.dit = VideoDiT(
            tokenizer,
            detokenizer,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            emb_channels=emb_channels,
            noise_channels=noise_channels,
            condition_dim=condition_dim,
            temporal_attention=temporal_attention,
            temporal_kwargs=temporal_kwargs,
            cross_attention=cross_attention,
            is_causal=is_causal,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
            mlp_ratio=mlp_ratio,
            drop_path_rates=drop_path_rates,
            conditioning_embedder="edm",
            attn_kwargs=attn_kwargs,
        )

    def set_context_parallel(self, mode: Optional[str], target=None) -> None:
        r"""Enable or disable context-parallel resharding on the backbone.

        Off by default; call once after the process group / device mesh is
        available (e.g. from the distributed training setup) to shard each block's
        attention across ``target``.

        Parameters
        ----------
        mode : str or None
            ``None`` (no resharding), ``"all_to_all"`` (manual collective over a
            ``ProcessGroup``), or ``"shardtensor"`` (``ShardTensor.redistribute``
            over a 1D mesh).
        target : ProcessGroup or DeviceMesh, optional, default=None
            The process group (``all_to_all``) or device mesh (``shardtensor``).
        """
        self.dit.set_context_parallel(mode, target)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels time npix"],
        noise_labels: Float[torch.Tensor, " batch"],
        second_of_day: Float[torch.Tensor, "batch time"],
        day_of_year: Float[torch.Tensor, "batch time"],
        obs: ObsContext,
    ) -> Float[torch.Tensor, "batch out_channels time npix"]:
        tokens = self.obs_tokenizer(
            obs.values,
            obs.float_metadata,
            obs.obs_type,
            obs.channel,
            obs.platform,
        )
        cross_attention_context = dataclasses.replace(obs, tokens=tokens)
        return self.dit(
            x,
            noise_labels,
            cross_attention_context=cross_attention_context,
            tokenizer_kwargs={
                "second_of_day": second_of_day,
                "day_of_year": day_of_year,
            },
        )
