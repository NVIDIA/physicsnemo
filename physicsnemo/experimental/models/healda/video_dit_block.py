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
"""Video / observation DiT block for ``(b, t, x, c)`` field sequences.

A 4D analog of :class:`physicsnemo.nn.DiTBlock`. It keeps the DiT-style template
-- adaLN-Zero conditioning, a pluggable spatial-attention backend, and a gated
MLP -- and adds two optional gated sub-layers:

* **temporal attention** across the time axis (factorized video attention), with
  an optional all-to-all / ShardTensor time<->space reshard for context
  parallelism (see :mod:`.sharding`);
* **observation cross-attention**, where each pixel latent attends to its packed
  observation tokens (see :class:`.pixel_cross_attention.PixelCrossAttention`).

Each sub-layer is a residual update gated by its own adaLN-Zero modulation, so
the block reduces exactly to the spatial DiT block when temporal and obs
attention are disabled.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.nn.module.dit_layers import get_attention, get_layer_norm
from physicsnemo.nn.module.drop import DropPath
from physicsnemo.nn.module.mlp_layers import Mlp

from .obs_packing import ObsCrossAttention
from .pixel_cross_attention import PixelCrossAttention
from .sharding import (
    shard_t,
    shard_t_shardtensor,
    shard_x,
    shard_x_shardtensor,
)
from .temporal_attention import TemporalAttention


def _broadcast(param: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a per-sample ``[b, c]`` modulation vector to broadcast over ``ndim``."""
    shape = (param.shape[0],) + (1,) * (ndim - 2) + (param.shape[1],)
    return param.view(shape)


class AdaLayerNormZero(nn.Module):
    """adaLN-Zero modulation that broadcasts over arbitrary tensor rank.

    Emits ``n_blocks`` ``(shift, scale, gate)`` triples from the conditioning
    embedding. The first triple's shift/scale are applied to the (affine-free)
    layer-normed hidden states; its gate and any further triples are returned for
    the caller to apply to later residual branches.
    """

    def __init__(self, embedding_dim: int, condition_embed_dim: int, n_blocks: int = 2):
        super().__init__()
        self.n_blocks = n_blocks
        self.silu = nn.SiLU()
        self.linear = nn.Linear(condition_embed_dim, 3 * n_blocks * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: Float[torch.Tensor, "*leading hidden_size"],
        emb: Float[torch.Tensor, "batch condition_embed_dim"],
    ) -> Tuple[torch.Tensor, ...]:
        chunks = self.linear(self.silu(emb)).chunk(3 * self.n_blocks, dim=1)
        shift, scale, gate = chunks[0], chunks[1], chunks[2]
        x = self.norm(x) * (1 + _broadcast(scale, x.ndim)) + _broadcast(shift, x.ndim)
        outputs = [x, _broadcast(gate, x.ndim)]
        outputs.extend(_broadcast(extra, x.ndim) for extra in chunks[3:])
        return tuple(outputs)

    def initialize_weights(self) -> None:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class VideoDiTBlock(nn.Module):
    r"""DiT block over ``(b, t, x, c)`` with optional temporal and obs attention.

    Parameters
    ----------
    hidden_size : int
        Token / channel dimension ``c``.
    num_heads : int
        Number of spatial-attention heads.
    condition_embed_dim : int
        Conditioning-embedding dimension feeding the adaLN modulations.
    attention_backend : str, optional, default="timm"
        Spatial-attention backend, passed to
        :func:`physicsnemo.nn.module.dit_layers.get_attention` (e.g. ``"timm"``,
        ``"transformer_engine"``). Applied per frame.
    mlp_ratio : float, optional, default=4.0
        MLP hidden-dim multiplier.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        LayerNorm backend for the MLP pre-norm.
    drop_path : float, optional, default=0.0
        Stochastic-depth rate applied to every residual branch.
    temporal_attention : bool, optional, default=False
        Add a gated temporal-attention sub-layer.
    temporal_kwargs : dict, optional
        Extra keyword arguments for :class:`.temporal_attention.TemporalAttention`.
    obs_cross_attention : bool, optional, default=False
        Add a gated observation cross-attention sub-layer.
    obs_token_dim : int, optional
        Observation-token feature dim (required when ``obs_cross_attention``).
    obs_q_heads, obs_kv_heads, obs_q_head_dim : int, optional
        Grouped-query head configuration for
        :class:`.pixel_cross_attention.PixelCrossAttention`.
    attn_kwargs : Any
        Forwarded to the spatial-attention backend constructor.

    Forward
    -------
    hidden_states : torch.Tensor
        ``(b, t, x, c)`` field-sequence latents (t-sharded under context
        parallelism: each rank holds all ``x`` for its time slice).
    emb : torch.Tensor
        ``(b, emb_channels)`` conditioning embedding.
    obs : ObsCrossAttention, optional
        Packed observation tokens + ragged packing metadata (single bundle) for
        the observation cross-attention.
    attn_kwargs : dict, optional
        Forwarded to the spatial-attention backend forward.
    is_causal : bool, optional, default=False
        Causal masking for temporal attention.

    Outputs
    -------
    torch.Tensor
        ``(b, t, x, c)`` updated latents in the same (t-sharded) layout.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        condition_embed_dim: int,
        attention_backend: str = "timm",
        mlp_ratio: float = 4.0,
        layernorm_backend: str = "torch",
        norm_eps: float = 1e-6,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        final_mlp_dropout: bool = True,
        drop_path: float = 0.0,
        temporal_attention: bool = False,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        obs_cross_attention: bool = False,
        obs_token_dim: Optional[int] = None,
        obs_q_heads: Optional[int] = None,
        obs_kv_heads: int = 1,
        obs_q_head_dim: Optional[int] = None,
        **attn_kwargs: Any,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Spatial self-attention (per frame) + MLP, both adaLN-Zero gated. This
        # path mirrors physicsnemo.nn.DiTBlock (same modulation, MLP, and dropout
        # wiring); temporal and obs attention are additional gated sub-layers.
        self.spatial_norm = AdaLayerNormZero(
            hidden_size, condition_embed_dim, n_blocks=2
        )
        self.spatial_attn = get_attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend=attention_backend,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            **attn_kwargs,
        )
        self.mlp_norm = get_layer_norm(
            hidden_size, layernorm_backend, elementwise_affine=False, eps=norm_eps
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=mlp_drop_rate,
            final_dropout=final_mlp_dropout,
        )

        # Optional temporal attention.
        self.temporal_attn = None
        self.temporal_norm = None
        if temporal_attention:
            self.temporal_norm = AdaLayerNormZero(
                hidden_size, condition_embed_dim, n_blocks=1
            )
            self.temporal_attn = TemporalAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                **(temporal_kwargs or {}),
            )

        # Optional observation cross-attention.
        self.obs_attn = None
        self.obs_norm = None
        if obs_cross_attention:
            if obs_token_dim is None:
                raise ValueError(
                    "obs_token_dim is required when obs_cross_attention=True"
                )
            if obs_q_heads is None:
                if hidden_size % obs_token_dim != 0:
                    raise ValueError(
                        f"hidden_size={hidden_size} must be divisible by "
                        f"obs_token_dim={obs_token_dim}"
                    )
                obs_q_heads = hidden_size // obs_token_dim
                obs_q_head_dim = obs_token_dim
            self.obs_norm = AdaLayerNormZero(
                hidden_size, condition_embed_dim, n_blocks=1
            )
            self.obs_attn = PixelCrossAttention(
                token_dim=obs_token_dim,
                input_dim=hidden_size,
                output_dim=hidden_size,
                n_q_heads=obs_q_heads,
                n_kv_heads=obs_kv_heads,
                d_head=obs_q_head_dim,
                use_proj_bias=True,
            )

        self.drop_path = DropPath(drop_path)

        # Context-parallel reshard config (set via set_context_parallel).
        self._reshard_mode: Optional[str] = None
        self._reshard_target = None

    def initialize_weights(self) -> None:
        """Zero-init every adaLN modulation (adaLN-Zero)."""
        self.spatial_norm.initialize_weights()
        if self.temporal_norm is not None:
            self.temporal_norm.initialize_weights()
        if self.obs_norm is not None:
            self.obs_norm.initialize_weights()

    def set_context_parallel(self, mode: Optional[str], target=None) -> None:
        """Configure the temporal time<->space reshard.

        Parameters
        ----------
        mode : {None, "all_to_all", "shardtensor"}
            ``None`` disables resharding (single device / no context parallelism).
            ``"all_to_all"`` uses the manual collective with a ``ProcessGroup``;
            ``"shardtensor"`` uses ``ShardTensor.redistribute`` with a 1D mesh.
        target : ProcessGroup or DeviceMesh, optional
            The process group (all_to_all) or device mesh (shardtensor).
        """
        if mode not in (None, "all_to_all", "shardtensor"):
            raise ValueError(f"unknown reshard mode {mode!r}")
        self._reshard_mode = mode
        self._reshard_target = target

    def _to_space_sharded(self, x: torch.Tensor) -> torch.Tensor:
        if self._reshard_mode == "all_to_all":
            return shard_x(x, self._reshard_target)
        if self._reshard_mode == "shardtensor":
            return shard_x_shardtensor(x, self._reshard_target)
        return x

    def _to_time_sharded(self, x: torch.Tensor) -> torch.Tensor:
        if self._reshard_mode == "all_to_all":
            return shard_t(x, self._reshard_target)
        if self._reshard_mode == "shardtensor":
            return shard_t_shardtensor(x, self._reshard_target)
        return x

    def _apply_obs_attention(self, hidden_states, emb, obs) -> torch.Tensor:
        b, t, npix, c = hidden_states.shape
        total_pixels = obs.cu_seqlens_k.shape[0] - 1
        if total_pixels != b * t * npix:
            raise ValueError(
                f"obs packing total_pixels={total_pixels} != b*t*npix={b * t * npix}"
            )
        x_bt = hidden_states.reshape(b * t, npix, c)
        emb_bt = emb[:, None, :].expand(b, t, -1).reshape(b * t, emb.shape[-1])
        normed, gate = self.obs_norm(x_bt, emb_bt)
        out = self.obs_attn(
            normed,
            obs.tokens,
            total_pixels,
            obs.cu_seqlens_k,
            obs.max_seqlen_k,
            group_map=obs.group_map,
        ).view_as(x_bt)
        x_bt = torch.addcmul(x_bt, self.drop_path(gate), out)
        return x_bt.view_as(hidden_states)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch time space hidden_size"],
        emb: Float[torch.Tensor, "batch condition_embed_dim"],
        obs: Optional[ObsCrossAttention] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        is_causal: bool = False,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        b, t, x, c = hidden_states.shape

        # Spatial self-attention (per frame), adaLN-Zero gated.
        normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.spatial_norm(
            hidden_states, emb
        )
        attn_in = normed.reshape(b * t, x, c)
        attn_out = self.spatial_attn(attn_in, **(attn_kwargs or {})).reshape(b, t, x, c)
        hidden_states = torch.addcmul(hidden_states, self.drop_path(gate_msa), attn_out)

        # Observation cross-attention (per frame), adaLN-Zero gated.
        if self.obs_attn is not None:
            if obs is None:
                raise ValueError(
                    "obs_cross_attention=True requires an ObsCrossAttention input."
                )
            hidden_states = self._apply_obs_attention(hidden_states, emb, obs)

        # Temporal attention across time, adaLN-Zero gated, with t<->x reshard.
        if self.temporal_attn is not None:
            hidden_states = self._to_space_sharded(hidden_states)
            normed, gate = self.temporal_norm(hidden_states, emb)
            temporal_out = self.temporal_attn(normed, is_causal)
            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate), temporal_out
            )
            hidden_states = self._to_time_sharded(hidden_states)

        # Gated MLP.
        mlp_in = self.mlp_norm(hidden_states) * (1 + scale_mlp) + shift_mlp
        hidden_states = torch.addcmul(
            hidden_states, self.drop_path(gate_mlp), self.mlp(mlp_in)
        )
        return hidden_states
