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

"""FiLM-conditioned observation tokenizer with an optional fused Triton backend.

:class:`ObsTokenizerFiLM` maps each scalar observation to an ``out_dim`` token,
modulated by per-observation metadata; see its docstring for the FiLM math and
conditioning layout. Triton kernels, ``torch.library.custom_op`` registration,
and launch-config presets live in :mod:`~physicsnemo.experimental.models.healda.kernels.obs_tokenizer_film`, imported
lazily by :func:`_fused_film_tokenizer_triton`.
"""

from typing import Optional

import torch
from jaxtyping import Float, Int

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")


GLOBAL_MAX_CHANNELS = 1024
GLOBAL_MAX_PLATFORM = 1024


def _default_film_hidden_dim(out_dim: int) -> int:
    return out_dim * 2 if out_dim <= 64 else out_dim


# ═══════════════════════════════════════════════════════════════════════════
# Triton public entry point
# ═══════════════════════════════════════════════════════════════════════════


def _fused_film_tokenizer_triton(
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: Optional[torch.Tensor],
    embed_table: "torch.nn.Embedding",
    channel_embedding: "torch.nn.Embedding",
    platform_embedding: Optional["torch.nn.Embedding"],
    linear1: "torch.nn.Linear",
    layer_norm: "torch.nn.LayerNorm",
    linear2: "torch.nn.Linear",
    eps: float = 1e-5,
    force_fp32: bool = False,
) -> torch.Tensor:
    """Fused Triton backend for :class:`ObsTokenizerFiLM` (see it for the FiLM
    math), computing the whole tokenizer in a single kernel. This means neither the
    forward or backward materializes large intermediate tensors in HBM.

    No intermediate activations are saved for the backward pass, which
    recomputes the forward from the conditioning vector.

    The embedding/``linear``/``layer_norm`` arguments are the modules owned by
    :class:`ObsTokenizerFiLM`, consumed here as raw tensors. ``linear1.out_features``
    must be a power of two and ``linear2.out_features`` must equal ``2 * out_dim``;
    ``platform`` is required when ``platform_embedding`` is provided.
    """
    meta_dim = float_meta.shape[1]

    obs_embed_dim = embed_table.weight.shape[1]
    chan_embed_dim = channel_embedding.weight.shape[1]

    if platform_embedding is None:
        platform_embed_dim = 0
        # Dummy weight: custom_op requires a tensor arg, but the kernel
        # never dereferences it (PLATFORM_EMBED_DIM=0 guards all loads).
        platform_embed_w = torch.empty((1, 1), device=obs.device, dtype=obs.dtype)
        if platform is None:
            # Likewise, platform pointer is passed but never loaded
            # when PLATFORM_EMBED_DIM=0.
            platform = obs_type_id
    else:
        platform_embed_dim = platform_embedding.weight.shape[1]
        platform_embed_w = platform_embedding.weight
        if platform is None:
            raise ValueError("platform required when platform_embedding is provided")

    hidden_dim = linear1.out_features
    if hidden_dim <= 0 or (hidden_dim & (hidden_dim - 1)) != 0:
        raise ValueError(f"hidden_dim must be power of 2, got {hidden_dim}")

    out_dim = linear2.out_features // 2
    # Triton kernels expect column-major logical access for the dense weights.
    w1_t = linear1.weight.t().contiguous()
    w2_t = linear2.weight.t().contiguous()

    from .kernels import obs_tokenizer_film as kernels

    return kernels.fused_film_fwd(
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_table.weight,
        channel_embedding.weight,
        platform_embed_w,
        w1_t,
        linear1.bias,
        layer_norm.weight,
        layer_norm.bias,
        w2_t,
        linear2.bias,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FiLM tokenizer module
# ═══════════════════════════════════════════════════════════════════════════


class ObsTokenizerFiLM(torch.nn.Module):
    r"""FiLM-style observation tokenizer: map each scalar observation to a token.

    Each observation is a single scalar measurement plus metadata describing it
    (location/time features, which instrument channel, which platform). This
    module turns that into an ``out_dim``-vector token, one per observation.

    FiLM (Feature-wise Linear Modulation) keeps the raw measurement as the signal
    and lets the metadata *modulate* it with a per-feature scale and shift::

        conditioning = cat(metadata, obs_type_emb, channel_emb[, platform_emb])
        alpha, beta  = cond_mlp(conditioning).chunk(2)   # 2 * out_dim -> two out_dim vectors
        token        = alpha * obs + beta                # broadcast scalar obs over out_dim

    Embedding tables:
      - ``embed_table``: observation *type* embedding (which kind of obs).
      - ``channel_embedding``: instrument *channel* embedding.
      - ``platform_embedding`` (optional): satellite/platform embedding.
    All are looked up per observation and concatenated into the conditioning.

    When the inputs are on CUDA and triton is available, the whole tokenizer
    (embedding gather + conditioning + 2-layer MLP + FiLM) runs in a single fused
    Triton kernel.

    Parameters
    ----------
    meta_dim : int
        Dimension of float metadata features.
    out_dim : int
        Output token dimension.
    n_embed : int, optional, default=1024
        Size of the observation-type embedding table.
    obs_type_embed_dim : int, optional, default=4
        Dimension of observation-type embeddings.
    channel_embed_dim : int, optional
        Dimension of channel embeddings. Defaults to ``obs_type_embed_dim``.
    platform_embed_dim : int, optional
        Dimension of platform embeddings. ``0``/``None`` disables platform
        embedding.
    use_fused_mlp : bool, optional, default=True
        Prefer the fused Triton backend when CUDA + triton are available.
    hidden_dim : int, optional
        Hidden dimension of the conditioning MLP. Defaults to a heuristic on
        ``out_dim``; must be a power of 2 for the fused kernel.

    Forward
    -------
    obs : torch.Tensor
        Observation values with shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor
        Float metadata with shape :math:`(N_{obs}, M_{float})`.
    obs_type : torch.Tensor
        Observation-type ids with shape :math:`(N_{obs},)`.
    channel_ids : torch.Tensor
        Channel ids with shape :math:`(N_{obs},)`.
    platform_ids : torch.Tensor, optional
        Platform ids with shape :math:`(N_{obs},)`. Required when platform
        embedding is enabled.

    Outputs
    -------
    torch.Tensor
        Tokenized observation features of shape :math:`(N_{obs}, D_{out})`.
    """

    def __init__(
        self,
        meta_dim: int,
        out_dim: int,
        n_embed: int = 1024,
        obs_type_embed_dim: int = 4,
        channel_embed_dim: int | None = None,
        platform_embed_dim: int | None = None,
        use_fused_mlp: bool = True,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.meta_dim = meta_dim
        self.out_dim = out_dim
        if channel_embed_dim is None:
            channel_embed_dim = obs_type_embed_dim
        if platform_embed_dim is None:
            platform_embed_dim = 0
        self.use_platform_embedding = platform_embed_dim > 0
        self.obs_type_embed_dim = obs_type_embed_dim
        self.channel_embed_dim = channel_embed_dim
        self.platform_embed_dim = platform_embed_dim
        self.embed_table = torch.nn.Embedding(n_embed, obs_type_embed_dim)
        self.channel_embedding = torch.nn.Embedding(
            GLOBAL_MAX_CHANNELS, channel_embed_dim
        )
        self.platform_embedding = (
            torch.nn.Embedding(GLOBAL_MAX_PLATFORM, platform_embed_dim)
            if self.use_platform_embedding
            else None
        )
        self.use_fused_mlp = use_fused_mlp
        if hidden_dim is None:
            hidden_dim = _default_film_hidden_dim(out_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim

        cond_dim = (
            meta_dim + obs_type_embed_dim + channel_embed_dim + platform_embed_dim
        )

        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2 * out_dim),
        )

    # --- Pure-PyTorch reference path ------------

    def _build_conditioning(
        self,
        float_metadata: torch.Tensor,
        obs_type: torch.Tensor,
        channel_ids: torch.Tensor,
        platform_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        embed_vec = self.embed_table(obs_type)
        chan_emb = self.channel_embedding(channel_ids)
        conditioning_parts = [float_metadata, embed_vec, chan_emb]
        if self.use_platform_embedding:
            if platform_ids is None:
                raise ValueError("platform embedding requires platform ids")
            conditioning_parts.append(self.platform_embedding(platform_ids))
        return torch.cat(conditioning_parts, dim=-1)

    def forward(
        self,
        obs: Float[torch.Tensor, " nobs"],
        float_metadata: Float[torch.Tensor, "nobs meta_dim"],
        obs_type: Int[torch.Tensor, " nobs"],
        channel_ids: Int[torch.Tensor, " nobs"],
        platform_ids: Int[torch.Tensor, " nobs"] | None = None,
    ) -> Float[torch.Tensor, "nobs out_dim"]:
        if not torch.compiler.is_compiling():
            if obs.ndim != 1:
                raise ValueError(
                    f"Expected obs of shape (nobs,), got {obs.ndim}D tensor with shape "
                    f"{tuple(obs.shape)}"
                )
            nobs = obs.shape[0]
            if float_metadata.ndim != 2 or float_metadata.shape[0] != nobs:
                raise ValueError(
                    f"Expected float_metadata of shape ({nobs}, meta_dim), got tensor "
                    f"with shape {tuple(float_metadata.shape)}"
                )
            if float_metadata.shape[1] != self.meta_dim:
                raise ValueError(
                    f"Expected float_metadata with meta_dim {self.meta_dim}, got "
                    f"{float_metadata.shape[1]}"
                )
            for name, tensor in (
                ("obs_type", obs_type),
                ("channel_ids", channel_ids),
            ):
                if tensor.ndim != 1 or tensor.shape[0] != nobs:
                    raise ValueError(
                        f"Expected {name} of shape ({nobs},) matching obs, got tensor "
                        f"with shape {tuple(tensor.shape)}"
                    )
            if self.use_platform_embedding:
                if platform_ids is None:
                    raise ValueError(
                        "platform_ids required when platform embedding is enabled"
                    )
                if platform_ids.ndim != 1 or platform_ids.shape[0] != nobs:
                    raise ValueError(
                        f"Expected platform_ids of shape ({nobs},) matching obs, got "
                        f"tensor with shape {tuple(platform_ids.shape)}"
                    )

        if self.use_fused_mlp and triton.available and obs.is_cuda:
            return _fused_film_tokenizer_triton(
                obs,
                float_metadata,
                obs_type,
                channel_ids,
                platform_ids if self.use_platform_embedding else None,
                self.embed_table,
                self.channel_embedding,
                self.platform_embedding,
                self.cond_mlp[0],
                self.cond_mlp[1],
                self.cond_mlp[3],
                eps=self.cond_mlp[1].eps,
            )

        conditioning = self._build_conditioning(
            float_metadata, obs_type, channel_ids, platform_ids
        )
        ab = self.cond_mlp(conditioning)
        alpha, beta = ab.chunk(2, dim=-1)
        return alpha * obs.unsqueeze(-1) + beta
