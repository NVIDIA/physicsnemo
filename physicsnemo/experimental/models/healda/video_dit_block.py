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
"""Video / observation DiT block over ``(b, t, x, c)`` field sequences.

A 4D extension of :class:`physicsnemo.nn.DiTBlock`: it subclasses the production
block (reusing its spatial attention, gated MLP, pre-norms, adaLN-Zero
modulation, and drop-path) and adds two optional gated sub-layers -- factorized
temporal attention (with the time<->space reshard of :mod:`.sharding`) and
observation cross-attention (:class:`.pixel_cross_attention.PixelCrossAttention`).
With both disabled it reduces exactly to the per-frame spatial DiT block.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.nn.module.dit_layers import DiTBlock, get_layer_norm

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
    shape = (param.shape[0],) + (1,) * (ndim - 2) + (param.shape[1],)
    return param.view(shape)


class VideoDiTBlock(DiTBlock):
    r"""DiT block over ``(b, t, x, c)`` with optional temporal and obs attention.

    Subclasses :class:`physicsnemo.nn.DiTBlock` and reuses its spatial
    self-attention, gated MLP, affine-free pre-norms, and 6-chunk adaLN-Zero
    modulation. Spatial attention runs per frame (time folded into batch); the
    optional temporal and observation sub-layers each add their own gated
    adaLN-Zero residual branch.

    Parameters
    ----------
    hidden_size : int
        Token / channel dimension :math:`C`.
    num_heads : int
        Number of spatial- and temporal-attention heads.
    condition_embed_dim : int
        Dimension of the conditioning embedding feeding the adaLN modulations.
    attention_backend : str, optional, default="timm"
        Spatial-attention backend forwarded to :class:`physicsnemo.nn.DiTBlock`.
    layernorm_backend : str, optional, default="torch"
        LayerNorm backend for all pre-norms.
    mlp_ratio : float, optional, default=4.0
        MLP hidden-dim multiplier.
    norm_eps : float, optional, default=1e-6
        Epsilon for the affine-free layer norms.
    attn_drop_rate : float, optional, default=0.0
        Spatial-attention dropout rate.
    proj_drop_rate : float, optional, default=0.0
        Spatial-attention output-projection dropout rate.
    mlp_drop_rate : float, optional, default=0.0
        Dropout rate inside the MLP.
    final_mlp_dropout : bool, optional, default=True
        Whether to apply the final MLP dropout.
    drop_path : float, optional, default=0.0
        Stochastic-depth rate applied to every residual branch.
    temporal_attention : bool, optional, default=False
        Add a gated temporal-attention sub-layer.
    temporal_kwargs : Dict[str, Any], optional, default=None
        Extra arguments for :class:`.temporal_attention.TemporalAttention`.
    obs_cross_attention : bool, optional, default=False
        Add a gated observation cross-attention sub-layer.
    obs_kwargs : Dict[str, Any], optional, default=None
        Obs cross-attention config (required when ``obs_cross_attention``); keys
        ``obs_token_dim`` (required), ``obs_q_heads``, ``obs_kv_heads``,
        ``obs_q_head_dim``.
    is_causal : bool, optional, default=False
        Causal masking for temporal attention, fixed at construction.
    attn_kwargs : Dict[str, Any], optional, default=None
        Extra arguments for the spatial-attention backend constructor.

    Forward
    -------
    hidden_states : torch.Tensor
        Field-sequence latents of shape :math:`(B, T, X, C)` (t-sharded under
        context parallelism).
    c : torch.Tensor
        Conditioning embedding of shape :math:`(B, D_c)`.
    obs : ObsCrossAttention, optional
        Packed observation tokens + ragged packing for the obs cross-attention.
    attn_kwargs : Dict[str, Any], optional
        Forwarded to the spatial-attention backend forward.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(B, T, X, C)` in the same layout.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        condition_embed_dim: int,
        attention_backend: str = "timm",
        layernorm_backend: str = "torch",
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-6,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        final_mlp_dropout: bool = True,
        drop_path: float = 0.0,
        temporal_attention: bool = False,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        obs_cross_attention: bool = False,
        obs_kwargs: Optional[Dict[str, Any]] = None,
        is_causal: bool = False,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            hidden_size,
            num_heads,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
            mlp_ratio=mlp_ratio,
            norm_eps=norm_eps,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            mlp_drop_rate=mlp_drop_rate,
            final_mlp_dropout=final_mlp_dropout,
            drop_path=drop_path,
            condition_embed_dim=condition_embed_dim,
            **(attn_kwargs or {}),
        )
        self.hidden_size = hidden_size
        self._is_causal = is_causal

        # Optional temporal attention: own SiLU+Linear adaLN-Zero modulation
        # (shift, scale, gate) + affine-free norm, applied as a gated residual.
        self.temporal_attn = None
        self.temporal_norm = None
        self.temporal_modulation = None
        if temporal_attention:
            self.temporal_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(condition_embed_dim, 3 * hidden_size)
            )
            self.temporal_norm = get_layer_norm(
                hidden_size, layernorm_backend, elementwise_affine=False, eps=norm_eps
            )
            self.temporal_attn = TemporalAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                **(temporal_kwargs or {}),
            )

        # Optional observation cross-attention: own modulation + affine-free norm.
        self.obs_attn = None
        self.obs_norm = None
        self.obs_modulation = None
        if obs_cross_attention:
            obs_cfg = dict(obs_kwargs or {})
            obs_token_dim = obs_cfg.pop("obs_token_dim", None)
            obs_q_heads = obs_cfg.pop("obs_q_heads", None)
            obs_kv_heads = obs_cfg.pop("obs_kv_heads", 1)
            obs_q_head_dim = obs_cfg.pop("obs_q_head_dim", None)
            if obs_token_dim is None:
                raise ValueError(
                    "obs_kwargs['obs_token_dim'] is required when "
                    "obs_cross_attention=True"
                )
            # Default to one query head per token-dim slice (head_dim == token_dim).
            if obs_q_heads is None:
                if hidden_size % obs_token_dim != 0:
                    raise ValueError(
                        f"hidden_size={hidden_size} must be divisible by "
                        f"obs_token_dim={obs_token_dim}"
                    )
                obs_q_heads = hidden_size // obs_token_dim
                obs_q_head_dim = obs_token_dim
            self.obs_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(condition_embed_dim, 3 * hidden_size)
            )
            self.obs_norm = get_layer_norm(
                hidden_size, layernorm_backend, elementwise_affine=False, eps=norm_eps
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

        # Context-parallel reshard config (set via set_context_parallel).
        self._reshard_mode: Optional[str] = None
        self._reshard_target = None

    def initialize_weights(self) -> None:
        r"""Zero-init every adaLN modulation (adaLN-Zero).

        Returns
        -------
        None
            Delegates to :meth:`physicsnemo.nn.DiTBlock.initialize_weights` for
            the spatial / MLP modulation and additionally zeros the temporal and
            obs modulation linears.
        """
        super().initialize_weights()
        for modulation in (self.temporal_modulation, self.obs_modulation):
            if modulation is not None:
                nn.init.zeros_(modulation[-1].weight)
                nn.init.zeros_(modulation[-1].bias)

    def set_context_parallel(self, mode: Optional[str], target=None) -> None:
        r"""Configure the temporal time<->space reshard.

        Parameters
        ----------
        mode : str or None
            One of ``None`` (no resharding), ``"all_to_all"`` (manual collective
            over a ``ProcessGroup``), or ``"shardtensor"``
            (``ShardTensor.redistribute`` over a 1D mesh).
        target : ProcessGroup or DeviceMesh, optional, default=None
            The process group (``all_to_all``) or device mesh (``shardtensor``).

        Returns
        -------
        None
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

    def _modulate(self, modulation, norm, x, c):
        shift, scale, gate = modulation(c).chunk(3, dim=1)
        normed = norm(x) * (1 + _broadcast(scale, x.ndim)) + _broadcast(shift, x.ndim)
        return normed, _broadcast(gate, x.ndim)

    def _apply_obs_attention(
        self, hidden_states: torch.Tensor, c: torch.Tensor, obs: ObsCrossAttention
    ) -> torch.Tensor:
        b, t, npix, ch = hidden_states.shape
        total_pixels = obs.cu_seqlens_k.shape[0] - 1
        if total_pixels != b * t * npix:
            raise ValueError(
                f"obs packing total_pixels={total_pixels} != b*t*npix={b * t * npix}"
            )
        # Fold time into batch and broadcast the conditioning per frame.
        x_bt = hidden_states.reshape(b * t, npix, ch)  # (B*T, X, C)
        c_bt = c[:, None, :].expand(b, t, -1).reshape(b * t, c.shape[-1])
        normed, gate = self._modulate(self.obs_modulation, self.obs_norm, x_bt, c_bt)
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
        c: Float[torch.Tensor, "batch condition_embed_dim"],
        obs: Optional[ObsCrossAttention] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        b, t, x, ch = hidden_states.shape

        # Spatial self-attention per frame, reusing DiTBlock's 6-chunk adaLN-Zero
        # modulation. The (B, C) modulation is broadcast to 4D here (not via
        # DiTBlock.modulation, which only unsqueezes to 3D).
        (
            attn_shift,
            attn_scale,
            attn_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.adaptive_modulation(c).chunk(6, dim=1)
        normed = self.pre_attention_norm(hidden_states)
        normed = normed * (1 + _broadcast(attn_scale, normed.ndim)) + _broadcast(
            attn_shift, normed.ndim
        )
        attn_out = self.attention(
            normed.reshape(b * t, x, ch), **(attn_kwargs or {})
        ).reshape(b, t, x, ch)
        hidden_states = torch.addcmul(
            hidden_states,
            self.drop_path(_broadcast(attn_gate, hidden_states.ndim)),
            attn_out,
        )

        # Observation cross-attention per frame.
        if self.obs_attn is not None:
            if obs is None:
                raise ValueError(
                    "obs_cross_attention=True requires an ObsCrossAttention input."
                )
            hidden_states = self._apply_obs_attention(hidden_states, c, obs)

        # Temporal attention across time, with t<->x reshard around it.
        if self.temporal_attn is not None:
            hidden_states = self._to_space_sharded(hidden_states)
            normed, gate = self._modulate(
                self.temporal_modulation, self.temporal_norm, hidden_states, c
            )
            temporal_out = self.temporal_attn(normed, self._is_causal)
            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate), temporal_out
            )
            hidden_states = self._to_time_sharded(hidden_states)

        # Gated MLP (DiTBlock's pre_mlp_norm + self.linear).
        mlp_in = self.pre_mlp_norm(hidden_states)
        mlp_in = mlp_in * (1 + _broadcast(mlp_scale, mlp_in.ndim)) + _broadcast(
            mlp_shift, mlp_in.ndim
        )
        hidden_states = torch.addcmul(
            hidden_states,
            self.drop_path(_broadcast(mlp_gate, hidden_states.ndim)),
            self.linear(mlp_in),
        )
        return hidden_states
