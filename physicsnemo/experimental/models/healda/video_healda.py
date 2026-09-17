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
"""Video + observation data-assimilation model composing :class:`~physicsnemo.experimental.models.healda.video_dit.VideoDiT`."""

from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import torch
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.healda.attention_layers import (
    PixelCrossAttention,
)
from physicsnemo.experimental.models.healda.obs_context import ObsContext
from physicsnemo.experimental.models.healda.obs_tokenizer import ObsTokenizerFiLM
from physicsnemo.experimental.models.healda.video_dit import VideoDiT
from physicsnemo.nn.module.hpx.tokenizer import (
    HEALPixPatchDetokenizer,
    HEALPixPatchTokenizer,
)


@dataclass
class VideoHealDAMetaData(ModelMetaData):
    r"""Metadata for :class:`VideoHealDA` (see :class:`~physicsnemo.core.meta.ModelMetaData`)."""

    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class VideoHealDA(Module):
    r"""Video transformer model for data assimilation of point-cloud like observations on the
    HEALPix grid.

    ``VideoHealDA`` maps a set of sparse, irregularly located observations (plus
    static fields and calendar features) to a gridded field sequence -- a short
    video window of :math:`T` frames over a HEALPix grid.

    It uses the DiT architecture (adaLN-Zero conditioning on ``t`` and
    ``class_labels``), but is currently only trained as a regression model
    with ``t`` fixed to zero.

    Data flows through four stages:

    1. Static conditioning fields are patch-tokenized from the fine ingest grid
       (``level_in``) down to the backbone grid (``level_model``) by
       :class:`~physicsnemo.nn.module.hpx.tokenizer.HEALPixPatchTokenizer`.
    2. A :class:`~physicsnemo.experimental.models.healda.video_dit.VideoDiT` backbone processes the token sequence with
       spatial attention, factorized temporal attention, and adaLN-Zero
       conditioning built from ``t``, ``class_labels``, and the calendar
       (second-of-day / day-of-year) features.
    3. Observations are embedded per observation by
       :class:`~physicsnemo.experimental.models.healda.obs_tokenizer.ObsTokenizerFiLM` and assimilated inside every block
       by :class:`~physicsnemo.experimental.models.healda.attention_layers.PixelCrossAttention`: each grid pixel
       attends only to the observations that land on it (ragged, local
       cross-attention).
    4. :class:`~physicsnemo.nn.module.hpx.tokenizer.HEALPixPatchDetokenizer` maps
       the backbone tokens back to the fine grid, producing the ``out_channels``
       output fields.

    The grid enters only at the boundaries: the patch tokenizer /
    detokenizer (stages 1 and 4) and, upstream in the dataloader, the assignment
    of a flat pixel index to each observation (which builds the ragged packing
    carried on ``obs_ctx``). Everything in between is grid-agnostic -- the backbone
    operates on a token sequence of shape :math:`(B, T, X, C)` and the
    observation cross-attention only needs each observation tagged with the pixel it belongs
    to. Adapting to a different grid therefore means swapping the tokenizer /
    detokenizer pair and the observation pixel-assignment step.

    Because spatial and temporal attention are factorized (each is independent
    along the axis the other mixes over), the model supports context parallelism:
    a group of :math:`N` GPUs reshards activations between time- and space-sharded
    layouts around each attention (see :mod:`~physicsnemo.experimental.models.healda.sharding`). :math:`N` must divide
    both the time and space extents, so the default ``time_length = 8`` caps it at
    8-way. Enable via :meth:`~physicsnemo.experimental.models.healda.video_healda.VideoHealDA.set_context_parallel`.

    Parameters
    ----------
    in_channels : int, optional, default=2
        Number of static conditioning channels (e.g. orography and land fraction).
    out_channels : int, optional, default=74
        Number of decoder output channels.
    hidden_size : int, optional, default=1536
        Transformer token dimension.
    num_layers : int, optional, default=32
        Number of :class:`~physicsnemo.experimental.models.healda.video_dit.VideoDiTBlock` blocks.
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
    condition_embed_dim : int, optional, default=128
        Width of the conditioning vector the adaLN modulations consume.
    noise_channels : int, optional, default=128
        Intermediate width of the noise-level embedding inside the
        conditioning embedder, before it is combined with ``class_labels``
        and projected to ``condition_embed_dim``.
    condition_dim : int, optional, default=0
        Width of the raw ``class_labels`` input the conditioning embedder
        accepts (0 = noise-only conditioning, no class-label input).
    temporal_attention : bool, optional, default=True
        Enable factorized temporal attention in every block.
    is_causal : bool, optional, default=True
        Causal masking for temporal attention, fixed at construction.
    linear_attention : bool, optional, default=True
        Use the softmax-free (linear) temporal attention variant.
    use_rope : bool, optional, default=True
        Apply rotary position embeddings on the temporal-attention time axis.
    rope_base : int, optional, default=100
        Base frequency for the temporal-attention rotary position embedding.
    max_seq_len : int, optional, default=100
        Maximum sequence length for the temporal rotary-embedding cache.
    temporal_causal_window : int, optional, default=None
        Sliding causal lookback for temporal attention. ``None`` is unbounded.
    drop_path : float, optional, default=0.1
        Stochastic-depth rate applied to every block past the warmup blocks.
    drop_path_zero_first_n_blocks : int, optional, default=4
        Number of leading blocks forced to drop-path rate 0.
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
    t : torch.Tensor
        Noise/timestep-conditioning input of shape :math:`(B,)`; fixed to zero
        for the current regression training setup (see Notes).
    second_of_day : torch.Tensor
        Second-of-day tensor of shape :math:`(B, T)` for the calendar embedding.
    day_of_year : torch.Tensor
        Day-of-year tensor of shape :math:`(B, T)` for the calendar embedding.
    obs_ctx : :class:`~physicsnemo.experimental.models.healda.obs_context.ObsContext`
        Raw observations plus ragged packing, with pixel prefix sums over
        :math:`B \cdot T \cdot X'` and :math:`X' = 12 \times 4^{\mathrm{level\_model}}`.
        Tokenized internally, then assimilated by the observation cross-attention.
    class_labels : torch.Tensor, optional, default=None
        Class-label condition vector of shape :math:`(B, \text{condition\_dim})`.
        ``None`` when ``condition_dim=0`` (noise-only conditioning).

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, T, N_{pix})`.

    Notes
    -----
    Build an :class:`~physicsnemo.experimental.models.healda.obs_context.ObsContext` with
    :func:`~physicsnemo.experimental.models.healda.obs_context.prepare_obs_context` to handle
    the obs preprocessing (sorting and packing). The FiLM tokenizer and pixel cross-attention run
    fused Triton kernels on CUDA. For repeated multi-GPU training, set
    ``HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR`` to a writable directory to cache
    Triton autotune tile configs across runs and cut first-hit stalls from
    variable obs counts (unset = off).

    Only the default ``attention_backend="timm"`` currently supports the
    ``qk_norm_type="RMSNorm"`` with ``qk_norm_affine=False`` this model
    relies on for stable training.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.healda import VideoHealDA, prepare_obs_context
    >>> model = VideoHealDA(
    ...     in_channels=2,
    ...     out_channels=3,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     num_heads=2,
    ...     level_in=2,
    ...     level_model=1,
    ...     time_length=2,
    ...     condition_embed_dim=32,
    ...     noise_channels=32,
    ...     obs_token_dim=16,
    ...     obs_meta_dim=8,
    ...     pixel_attn_n_q_heads=32,
    ...     pixel_attn_n_kv_heads=2,
    ...     pixel_attn_head_dim=16,
    ... )
    >>> b, t, npix = 1, 2, 12 * 4**2
    >>> npix_model = 12 * 4**1
    >>> nobs = 3
    >>> obs = torch.tensor([1.0, 2.0, 3.0])
    >>> float_metadata = torch.randn(nobs, 8)
    >>> ids = torch.zeros(nobs, dtype=torch.long)
    >>> flat = torch.tensor([5, 0, 5], dtype=torch.int32)
    >>> obs_ctx = prepare_obs_context(
    ...     obs=obs,
    ...     float_metadata=float_metadata,
    ...     obs_type=ids,
    ...     channel=ids,
    ...     platform=ids,
    ...     flat_idx=flat,
    ...     total_pixels=b * t * npix_model,
    ... )
    >>> out = model(
    ...     torch.randn(b, 2, t, npix),
    ...     torch.zeros(b),
    ...     torch.rand(b, t) * 86400.0,
    ...     torch.rand(b, t) * 365.0,
    ...     obs_ctx,
    ... )
    >>> out.shape
    torch.Size([1, 3, 2, 192])
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
        condition_embed_dim: int = 128,
        noise_channels: int = 128,
        condition_dim: int = 0,
        temporal_attention: bool = True,
        is_causal: bool = True,
        linear_attention: bool = True,
        use_rope: bool = True,
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
        super().__init__(meta=VideoHealDAMetaData())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.level_in = level_in
        self.level_model = level_model
        self.time_length = time_length
        self.condition_dim = condition_dim
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
            condition_dim=condition_embed_dim,
        )
        self.dit = VideoDiT(
            tokenizer,
            detokenizer,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            emb_channels=condition_embed_dim,
            noise_channels=noise_channels,
            condition_dim=condition_dim,
            temporal_attention=temporal_attention,
            use_rope=use_rope,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
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
        t: Float[torch.Tensor, " batch"],
        second_of_day: Float[torch.Tensor, "batch time"],
        day_of_year: Float[torch.Tensor, "batch time"],
        obs_ctx: ObsContext,
        class_labels: Optional[Float[torch.Tensor, "batch condition_dim"]] = None,
    ) -> Float[torch.Tensor, "batch out_channels time npix"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 4D input (B, C, T, X), got {x.ndim}D tensor with shape "
                    f"{tuple(x.shape)}"
                )
            b, c, time_len, x_dim = x.shape
            if c != self.in_channels:
                raise ValueError(
                    f"Expected {self.in_channels} input channels, got {c} channels"
                )
            if time_len != self.time_length:
                raise ValueError(
                    f"Expected time_length {self.time_length}, got {time_len} frames"
                )
            if x_dim != self.npix:
                raise ValueError(f"Expected npix {self.npix}, got {x_dim} pixels")
            if t.ndim != 1 or t.shape[0] != b:
                raise ValueError(
                    f"Expected t of shape ({b},), got tensor with shape "
                    f"{tuple(t.shape)}"
                )
            if second_of_day.shape != (b, time_len) or day_of_year.shape != (
                b,
                time_len,
            ):
                raise ValueError(
                    f"Expected calendar tensors of shape ({b}, {time_len}), got "
                    f"second_of_day {tuple(second_of_day.shape)} and day_of_year "
                    f"{tuple(day_of_year.shape)}"
                )
            npix_model = 12 * 4**self.level_model
            expected_cu_len = b * time_len * npix_model + 1
            if obs_ctx.cu_seqlens_k.numel() != expected_cu_len:
                raise ValueError(
                    f"Expected cu_seqlens_k length {expected_cu_len} "
                    f"(B*T*npix_model+1), got {obs_ctx.cu_seqlens_k.numel()}"
                )
            if self.condition_dim > 0:
                if class_labels is None:
                    raise ValueError(
                        f"condition_dim={self.condition_dim} but class_labels was not "
                        "provided"
                    )
                if class_labels.shape != (b, self.condition_dim):
                    raise ValueError(
                        f"Expected class_labels of shape ({b}, {self.condition_dim}), "
                        f"got tensor with shape {tuple(class_labels.shape)}"
                    )
            elif class_labels is not None:
                raise ValueError(
                    "class_labels was provided but condition_dim=0 (noise-only "
                    "conditioning)"
                )

        tokens = self.obs_tokenizer(
            obs_ctx.obs,
            obs_ctx.float_metadata,
            obs_ctx.obs_type,
            obs_ctx.channel,
            obs_ctx.platform,
        )
        cross_attn_kwargs = {
            "tokens": tokens,
            "cu_seqlens_k": obs_ctx.cu_seqlens_k,
            "max_seqlen_k": obs_ctx.max_seqlen_k,
            "group_map": obs_ctx.group_map,
        }
        return self.dit(
            x,
            t,
            condition=class_labels,
            cross_attn_kwargs=cross_attn_kwargs,
            tokenizer_kwargs={
                "second_of_day": second_of_day,
                "day_of_year": day_of_year,
            },
        )
