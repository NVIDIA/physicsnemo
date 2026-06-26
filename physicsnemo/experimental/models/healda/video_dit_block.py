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
"""DiT block over ``(b, t, x, c)`` with optional temporal and cross-attention."""

from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.nn.module.dit_layers import get_attention, get_layer_norm
from physicsnemo.nn.module.drop import DropPath
from physicsnemo.nn.module.mlp_layers import Mlp

from .adaln import AdaLayerNormZero
from .sharding import (
    shard_t,
    shard_t_shardtensor,
    shard_x,
    shard_x_shardtensor,
)
from .temporal_attention import TemporalAttention


class VideoDiTBlock(nn.Module):
    r"""A DiT block over :math:`(B, T, X, C)` with optional temporal and cross-attention.

    Spatial attention runs per frame (time folded into batch); the optional
    temporal and cross-attention sub-layers each add a gated residual branch.

    Parameters
    ----------
    hidden_size : int
        Token / channel dimension :math:`C`.
    num_heads : int
        Number of spatial- and temporal-attention heads.
    condition_embed_dim : int
        Dimension of the conditioning embedding feeding the adaLN modulations.
    attention_backend : Literal["timm", "transformer_engine", "natten2d", "natten2d_rope"] or Module, optional, default="timm"
        Spatial-attention backend name or a pre-instantiated attention module.
    layernorm_backend : Literal["apex", "torch"], optional, default="torch"
        LayerNorm backend for all adaLN-Zero pre-norms.
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
    cross_attention : Callable[..., Module], optional, default=None
        Factory building this block's cross-attention module
        (:class:`~physicsnemo.experimental.models.healda.cross_attention.CrossAttentionModuleBase`).
        When set, adds a gated cross-attention sub-layer consuming the opaque
        ``cross_attention_context`` passed to :meth:`forward`.
    is_causal : bool, optional, default=False
        Causal masking for temporal attention, fixed at construction.
    adaln_zero_init : bool, optional, default=True
        Forwarded to every :class:`.adaln.AdaLayerNormZero` ``zero_init``.
    attn_kwargs : Dict[str, Any], optional, default=None
        Extra arguments for the spatial-attention backend constructor.

    Notes
    -----
    A single ``norm1``
    :class:`~physicsnemo.experimental.models.healda.adaln.AdaLayerNormZero`
    (``n_blocks=2``) drives both spatial attention and the MLP, matching the
    DiT/diffusers layout; the MLP pre-norm is a separate parameter-free
    LayerNorm. The optional temporal and cross-attention sub-layers each own a
    one-block ``AdaLayerNormZero``.

    Forward
    -------
    hidden_states : torch.Tensor
        Latents of shape :math:`(B, T, X, C)` (t-sharded under context
        parallelism).
    c : torch.Tensor
        Conditioning embedding of shape :math:`(B, D_c)`.
    cross_attention_context : Any, optional
        Opaque per-call context forwarded to the injected ``cross_attention`` module.
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
        attention_backend: Union[str, Module] = "timm",
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
        cross_attention: Optional[Callable[..., Module]] = None,
        is_causal: bool = False,
        adaln_zero_init: bool = True,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._is_causal = is_causal

        # Spatial self-attention backend (name -> built here, or injected module),
        # named ``attention`` to match DiTBlock for checkpoint translatability.
        if isinstance(attention_backend, Module):
            self.attention = attention_backend
        else:
            attn_kwargs_final = dict(attn_kwargs or {})
            if attention_backend in ("natten2d", "natten2d_rope"):
                attn_kwargs_final.setdefault("norm_layer", layernorm_backend)
            self.attention = get_attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                attention_backend=attention_backend,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=proj_drop_rate,
                **attn_kwargs_final,
            )
        # One adaLN-Zero (n_blocks=2) drives both spatial attention and the MLP:
        # its modulation emits 6 chunks (attn shift/scale/gate + mlp
        # shift/scale/gate), matching the DiT/diffusers layout.
        self.norm1 = AdaLayerNormZero(
            hidden_size,
            condition_embed_dim,
            n_blocks=2,
            zero_init=adaln_zero_init,
            layernorm_backend=layernorm_backend,
            norm_eps=norm_eps,
        )

        self.linear = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=mlp_drop_rate,
            final_dropout=final_mlp_dropout,
        )
        # MLP pre-norm: a parameter-free LayerNorm modulated by norm1's MLP
        # shift/scale (not its own adaLN-Zero).
        self.mlp_norm = get_layer_norm(
            hidden_size,
            layernorm_backend,
            elementwise_affine=False,
            eps=norm_eps,
        )

        # Optional gated temporal-attention sub-layer.
        self.temporal_attention = None
        self.temporal_attn_norm = None
        if temporal_attention:
            self.temporal_attention = TemporalAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                **(temporal_kwargs or {}),
            )
            self.temporal_attn_norm = AdaLayerNormZero(
                hidden_size,
                condition_embed_dim,
                zero_init=adaln_zero_init,
                layernorm_backend=layernorm_backend,
                norm_eps=norm_eps,
            )

        self.cross_attention = cross_attention() if cross_attention is not None else None
        self.cross_attn_norm = None
        if self.cross_attention is not None:
            self.cross_attn_norm = AdaLayerNormZero(
                hidden_size,
                condition_embed_dim,
                zero_init=adaln_zero_init,
                layernorm_backend=layernorm_backend,
                norm_eps=norm_eps,
            )

        self.drop_path = DropPath(drop_path)

        # Context-parallel reshard config (set via set_context_parallel).
        self._reshard_mode: Optional[str] = None
        self._reshard_target = None

    def initialize_weights(self) -> None:
        r"""Zero-init every adaLN-Zero modulation (when their ``zero_init`` is set).

        Returns
        -------
        None
            Delegates to each :class:`.adaln.AdaLayerNormZero`.
        """
        for adaln in (
            self.norm1,
            self.temporal_attn_norm,
            self.cross_attn_norm,
        ):
            if adaln is not None:
                adaln.initialize_weights()

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

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch time space hidden_size"],
        c: Float[torch.Tensor, "batch condition_embed_dim"],
        cross_attention_context: Optional[Any] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        b, t, x, ch = hidden_states.shape

        normed, attn_gate, mlp_shift, mlp_scale, mlp_gate = self.norm1(
            hidden_states, c
        )

        # Spatial self-attention per frame (time folded into batch).
        attn_out = self.attention(
            normed.reshape(b * t, x, ch), **(attn_kwargs or {})
        ).reshape(b, t, x, ch)
        hidden_states = torch.addcmul(
            hidden_states, self.drop_path(attn_gate), attn_out
        )

        # Cross-attention to the opaque injected context.
        if self.cross_attention is not None:
            if cross_attention_context is None:
                raise ValueError(
                    "cross_attention was provided at construction but no "
                    "cross_attention_context was passed to forward."
                )
            normed, gate = self.cross_attn_norm(hidden_states, c)
            cross_out = self.cross_attention(normed, cross_attention_context)
            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate), cross_out
            )

        # Temporal attention across time, with t<->x reshard around it.
        if self.temporal_attention is not None:
            hidden_states = self._to_space_sharded(hidden_states)
            normed, gate = self.temporal_attn_norm(hidden_states, c)
            temporal_out = self.temporal_attention(normed, self._is_causal)
            hidden_states = torch.addcmul(
                hidden_states, self.drop_path(gate), temporal_out
            )
            hidden_states = self._to_time_sharded(hidden_states)

        # Feed-forward block, modulated by norm1's MLP shift/scale/gate.
        mlp_in = self.mlp_norm(hidden_states) * (1 + mlp_scale) + mlp_shift
        hidden_states = torch.addcmul(
            hidden_states, self.drop_path(mlp_gate), self.linear(mlp_in)
        )
        return hidden_states
