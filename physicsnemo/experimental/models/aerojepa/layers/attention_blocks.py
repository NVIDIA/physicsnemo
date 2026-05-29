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

"""Attention building blocks for AeroJEPA.

Three reusable layers: ``ResidualMLP`` (pre-norm residual MLP with optional
AdaLN/AdaLN-Zero conditioning), ``LocalPointTransformerBlock`` (local
self-attention over a per-point k-NN graph with relative positional bias),
and ``LocalTokenCrossAttentionBlock`` (cross-attention from queries to a
per-query k-NN of context tokens). All three optionally accept a
conditioning tensor that produces shift/scale/gate parameters in the style
of DiT's AdaLN-Zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .token_utils import chunked_knn_indices, gather_rows


def _make_conditioning_mlp(cond_dim: int, out_dim: int) -> nn.Sequential:
    hidden_dim = max(int(cond_dim), int(out_dim))
    mlp = nn.Sequential(
        nn.Linear(int(cond_dim), hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, int(out_dim)),
    )
    last = mlp[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return mlp


def _reshape_condition(cond: torch.Tensor) -> torch.Tensor:
    if cond.ndim == 1:
        return cond.unsqueeze(0)
    if cond.ndim != 2:
        raise ValueError(
            "conditioning tensor must have shape [D], [1, D], or [N, D], "
            f"got {tuple(cond.shape)}"
        )
    return cond


def _apply_neighbor_mask(
    logits: torch.Tensor, neighbor_mask: torch.Tensor | None
) -> torch.Tensor:
    if neighbor_mask is None:
        return torch.softmax(logits, dim=-1)
    mask = neighbor_mask.unsqueeze(1)
    masked_logits = logits.masked_fill(~mask, -1e9)
    attn = torch.softmax(masked_logits, dim=-1)
    attn = attn * mask.to(dtype=attn.dtype)
    return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class ResidualMLP(nn.Module):
    r"""Pre-norm residual MLP with optional AdaLN/AdaLN-Zero conditioning.

    Applies ``LayerNorm`` → ``Linear → GELU → Dropout → Linear → Dropout``
    and adds the result back to the input. When ``conditioning_dim`` is
    set, a small zero-initialized MLP turns ``cond`` into ``(shift, scale,
    gate)`` that modulate the pre-MLP and post-MLP signals in the AdaLN /
    AdaLN-Zero style.

    Parameters
    ----------
    dim : int
        Feature dimension.
    mlp_ratio : int
        Hidden dimension is ``max(1, mlp_ratio) * dim``.
    dropout : float
        Dropout probability used inside the MLP.
    conditioning_dim : int, optional
        Size of the conditioning vector. ``None`` disables conditioning.
    adaln_zero : bool, optional
        If ``True``, the residual is gated by ``gate`` (AdaLN-Zero); if
        ``False``, it is gated by ``1 + gate``. Default ``False``.

    Shape
    -----
    - Input ``x``: ``(N, dim)`` or ``(B, N, dim)``.
    - Conditioning ``cond`` (if used): ``(dim,)``, ``(1, D_cond)`` or
      ``(N, D_cond)``.
    - Output: same shape as ``x``.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        dropout: float,
        conditioning_dim: int | None = None,
        adaln_zero: bool = False,
    ):
        super().__init__()
        hidden = max(1, int(mlp_ratio)) * int(dim)
        self.norm = nn.LayerNorm(int(dim))
        self.conditioning = (
            None
            if conditioning_dim is None
            else _make_conditioning_mlp(int(conditioning_dim), 3 * int(dim))
        )
        self.adaln_zero = bool(adaln_zero)
        self.net = nn.Sequential(
            nn.Linear(int(dim), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm(x)
        gate = None
        if self.conditioning is not None:
            if cond is None:
                raise ValueError(
                    "conditioning input must be provided for conditioned ResidualMLP."
                )
            shift, scale, gate = self.conditioning(
                _reshape_condition(cond)
            ).chunk(3, dim=-1)
            h = h * (1.0 + scale) + shift
        out = self.net(h)
        if gate is not None:
            out = out * (gate if self.adaln_zero else (1.0 + gate))
        return x + out


class LocalPointTransformerBlock(nn.Module):
    r"""Local self-attention block over a per-point k-NN graph.

    For each point, attends to its ``neighbor_k`` nearest neighbors (chosen
    by :func:`physicsnemo.experimental.models.aerojepa.layers.token_utils.chunked_knn_indices`)
    with a learned relative-position bias and per-head attention scores.
    Followed by a :class:`ResidualMLP`. Optional AdaLN/AdaLN-Zero
    conditioning modulates both the attention sublayer and the
    feed-forward sublayer.

    When the input has at most one point, the attention sublayer is
    skipped and only the FFN is applied (still receiving ``cond`` if
    provided).

    Parameters
    ----------
    dim : int
        Feature dimension. Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    neighbor_k : int
        Number of nearest neighbors used per query point (post-dilation).
    dilation : int
        Stride applied to the top-``k * dilation`` neighbor indices before
        truncation. Lets the block attend at coarser receptive fields
        without re-running the search. Clamped to at least 1.
    mlp_ratio : int
        Hidden multiplier for the inner ``ResidualMLP``.
    dropout : float
        Dropout used after the output projection and inside the FFN.
    knn_chunk_size : int
        Chunk size passed to ``chunked_knn_indices``.
    conditioning_dim : int, optional
        Size of the conditioning vector. ``None`` disables conditioning
        on both sublayers.
    adaln_zero : bool, optional
        Forwarded to the FFN and used to gate the attention output the
        same way (``gate`` vs ``1 + gate``). Default ``False``.

    Shape
    -----
    - ``features``: ``(N, dim)``.
    - ``coords``: ``(N, 3)``.
    - ``cond`` (if used): ``(D_cond,)``, ``(1, D_cond)`` or
      ``(N, D_cond)``.
    - ``batch_ids`` (optional): ``(N,)`` ``int64``; when provided, neighbors
      from different batches are masked out of attention.
    - Output: ``(N, dim)``.

    Raises
    ------
    ValueError
        If ``dim`` is not divisible by ``num_heads``, or if conditioning
        is requested but ``cond`` is not provided.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        neighbor_k: int,
        dilation: int,
        mlp_ratio: int,
        dropout: float,
        knn_chunk_size: int,
        conditioning_dim: int | None = None,
        adaln_zero: bool = False,
    ):
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.neighbor_k = int(neighbor_k)
        self.dilation = int(max(1, dilation))
        self.knn_chunk_size = int(knn_chunk_size)
        self.norm = nn.LayerNorm(self.dim)
        self.adaln_zero = bool(adaln_zero)
        self.conditioning = (
            None
            if conditioning_dim is None
            else _make_conditioning_mlp(int(conditioning_dim), 3 * self.dim)
        )
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.attn_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_heads),
        )
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(float(dropout))
        self.ffn = ResidualMLP(
            dim=self.dim,
            mlp_ratio=int(mlp_ratio),
            dropout=float(dropout),
            conditioning_dim=conditioning_dim,
            adaln_zero=adaln_zero,
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        cond: torch.Tensor | None = None,
        batch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(features.shape[0]) <= 1:
            return self.ffn(features, cond=cond)

        residual = features
        h = self.norm(features)
        gate = None
        if self.conditioning is not None:
            if cond is None:
                raise ValueError(
                    "conditioning input must be provided for conditioned "
                    "LocalPointTransformerBlock."
                )
            shift, scale, gate = self.conditioning(
                _reshape_condition(cond)
            ).chunk(3, dim=-1)
            h = h * (1.0 + scale) + shift
        idx = chunked_knn_indices(
            query_coords=coords,
            key_coords=coords,
            k=min(self.neighbor_k, int(coords.shape[0])),
            chunk_size=self.knn_chunk_size,
            dilation=self.dilation,
        )
        neighbor_mask = None
        if batch_ids is not None:
            gathered_batch_ids = gather_rows(batch_ids.unsqueeze(-1), idx).squeeze(-1)
            neighbor_mask = gathered_batch_ids == batch_ids.unsqueeze(1)
        q = self.q_proj(h).reshape(int(h.shape[0]), self.num_heads, self.head_dim)
        k = gather_rows(self.k_proj(h), idx).reshape(
            int(h.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        v = gather_rows(self.v_proj(h), idx).reshape(
            int(h.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        rel = gather_rows(coords, idx) - coords.unsqueeze(1)
        rel_bias = self.pos_proj(rel).reshape(
            int(h.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        attn_in = (q.unsqueeze(1) - k + rel_bias).reshape(
            int(h.shape[0]), idx.shape[1], self.dim
        )
        logits = self.attn_proj(attn_in).transpose(1, 2) / max(
            self.head_dim**0.5, 1.0
        )
        attn = _apply_neighbor_mask(logits, neighbor_mask)
        value = (v + rel_bias).permute(0, 2, 1, 3)
        out = (attn.unsqueeze(-1) * value).sum(dim=2).reshape(
            int(h.shape[0]), self.dim
        )
        out = self.dropout(self.out_proj(out))
        if gate is not None:
            out = out * (gate if self.adaln_zero else (1.0 + gate))
        out = residual + out
        return self.ffn(out, cond=cond)


class LocalTokenCrossAttentionBlock(nn.Module):
    r"""Local cross-attention from query tokens to a per-query k-NN of context.

    Each query attends to its ``neighbor_k`` nearest context tokens (by
    Euclidean distance in coordinate space) with a learned relative-position
    bias and per-head attention scores. Followed by a :class:`ResidualMLP`.

    When conditioning is enabled, a single MLP produces a 5-way chunked
    output ``(q_shift, q_scale, kv_shift, kv_scale, gate)``. The query side
    is modulated by ``(q_shift, q_scale)`` from ``cond``; the key/value
    side is modulated by ``(kv_shift, kv_scale)`` from
    ``context_cond if context_cond is not None else cond``.

    When either input has zero tokens the block is a no-op (returns
    ``query_features`` unchanged).

    Parameters
    ----------
    dim : int
        Feature dimension shared by queries and context. Must be divisible
        by ``num_heads``.
    num_heads : int
        Number of attention heads.
    neighbor_k : int
        Number of nearest context tokens used per query.
    mlp_ratio : int
        Hidden multiplier for the inner ``ResidualMLP``.
    dropout : float
        Dropout used after the output projection and inside the FFN.
    knn_chunk_size : int
        Chunk size passed to ``chunked_knn_indices``.
    conditioning_dim : int, optional
        Size of the conditioning vector. ``None`` disables conditioning.
    adaln_zero : bool, optional
        Forwarded to the FFN and used to gate the attention output the
        same way. Default ``False``.

    Shape
    -----
    - ``query_features``: ``(Nq, dim)``; ``query_coords``: ``(Nq, 3)``.
    - ``context_features``: ``(Nc, dim)``; ``context_coords``: ``(Nc, 3)``.
    - ``cond`` / ``context_cond`` (if used): ``(D_cond,)`` or
      ``(N, D_cond)``.
    - ``query_batch_ids`` / ``context_batch_ids`` (optional): ``(Nq,)`` /
      ``(Nc,)`` ``int64``; when both are provided, neighbors from
      different batches are masked out of attention.
    - Output: ``(Nq, dim)``.

    Raises
    ------
    ValueError
        If ``dim`` is not divisible by ``num_heads``, or if conditioning
        is requested but ``cond`` is not provided.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        neighbor_k: int,
        mlp_ratio: int,
        dropout: float,
        knn_chunk_size: int,
        conditioning_dim: int | None = None,
        adaln_zero: bool = False,
    ):
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.neighbor_k = int(neighbor_k)
        self.knn_chunk_size = int(knn_chunk_size)
        self.norm_q = nn.LayerNorm(self.dim)
        self.adaln_zero = bool(adaln_zero)
        self.norm_kv = nn.LayerNorm(self.dim)
        self.conditioning = (
            None
            if conditioning_dim is None
            else _make_conditioning_mlp(int(conditioning_dim), 5 * self.dim)
        )
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.attn_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_heads),
        )
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(float(dropout))
        self.ffn = ResidualMLP(
            dim=self.dim,
            mlp_ratio=int(mlp_ratio),
            dropout=float(dropout),
            conditioning_dim=conditioning_dim,
            adaln_zero=adaln_zero,
        )

    def forward(
        self,
        query_features: torch.Tensor,
        query_coords: torch.Tensor,
        context_features: torch.Tensor,
        context_coords: torch.Tensor,
        cond: torch.Tensor | None = None,
        context_cond: torch.Tensor | None = None,
        query_batch_ids: torch.Tensor | None = None,
        context_batch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(query_features.shape[0]) == 0 or int(context_features.shape[0]) == 0:
            return query_features

        residual = query_features
        q_in = self.norm_q(query_features)
        kv_in = self.norm_kv(context_features)
        gate = None
        if self.conditioning is not None:
            if cond is None:
                raise ValueError(
                    "conditioning input must be provided for conditioned "
                    "LocalTokenCrossAttentionBlock."
                )
            q_shift, q_scale, _, _, gate = self.conditioning(
                _reshape_condition(cond)
            ).chunk(5, dim=-1)
            q_in = q_in * (1.0 + q_scale) + q_shift
            kv_source = cond if context_cond is None else context_cond
            kv_shift, kv_scale = self.conditioning(
                _reshape_condition(kv_source)
            ).chunk(5, dim=-1)[2:4]
            kv_in = kv_in * (1.0 + kv_scale) + kv_shift
        idx = chunked_knn_indices(
            query_coords=query_coords,
            key_coords=context_coords,
            k=min(self.neighbor_k, int(context_coords.shape[0])),
            chunk_size=self.knn_chunk_size,
        )
        neighbor_mask = None
        if query_batch_ids is not None and context_batch_ids is not None:
            gathered_batch_ids = gather_rows(
                context_batch_ids.unsqueeze(-1), idx
            ).squeeze(-1)
            neighbor_mask = gathered_batch_ids == query_batch_ids.unsqueeze(1)
        q = self.q_proj(q_in).reshape(
            int(q_in.shape[0]), self.num_heads, self.head_dim
        )
        k = gather_rows(self.k_proj(kv_in), idx).reshape(
            int(q_in.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        v = gather_rows(self.v_proj(kv_in), idx).reshape(
            int(q_in.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        rel = gather_rows(context_coords, idx) - query_coords.unsqueeze(1)
        rel_bias = self.pos_proj(rel).reshape(
            int(q_in.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        attn_in = (q.unsqueeze(1) - k + rel_bias).reshape(
            int(q_in.shape[0]), idx.shape[1], self.dim
        )
        logits = self.attn_proj(attn_in).transpose(1, 2) / max(
            self.head_dim**0.5, 1.0
        )
        attn = _apply_neighbor_mask(logits, neighbor_mask)
        value = (v + rel_bias).permute(0, 2, 1, 3)
        out = (attn.unsqueeze(-1) * value).sum(dim=2).reshape(
            int(q_in.shape[0]), self.dim
        )
        out = self.dropout(self.out_proj(out))
        if gate is not None:
            out = out * (gate if self.adaln_zero else (1.0 + gate))
        out = residual + out
        return self.ffn(out, cond=cond)
