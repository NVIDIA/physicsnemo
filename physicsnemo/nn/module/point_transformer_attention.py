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

r"""Point-Transformer local vector-attention blocks.

This module provides local (k-NN) *vector* attention over point clouds in
the style of Point Transformer (Zhao et al., 2021), generalized with the
grouped (per-head) weighting of Point Transformer v2 and optional
DiT-style AdaLN / AdaLN-Zero conditioning.

Unlike scaled dot-product attention, the per-neighbour score is produced by
a small MLP applied to :math:`q - k + \delta` (a learned comparison rather
than an inner product), restricted to each query's :math:`k` nearest
neighbours. A relative-position MLP produces :math:`\delta` and biases both
the scores and the aggregated values.

Three layers are exposed:

- :class:`ResidualMLP` -- pre-norm residual MLP with optional AdaLN/AdaLN-Zero
  conditioning, used as the feed-forward sublayer.
- :class:`LocalPointTransformerBlock` -- local self-attention over a
  per-point k-NN graph.
- :class:`LocalTokenCrossAttentionBlock` -- local cross-attention from
  query tokens to a per-query k-NN of context tokens.

All three are tensor-in/tensor-out (operating on flat :math:`(N, D)`
features plus :math:`(N, 3)` coordinates) and carry no model-specific data
structures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.nn.functional import knn

from .layer_norm import LayerNorm


def _gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    r"""Gather rows of ``x`` by an arbitrary-shape index tensor.

    Parameters
    ----------
    x : torch.Tensor
        Source tensor of shape :math:`(N, F)`.
    idx : torch.Tensor
        Index tensor of any shape holding integer indices into the first
        axis of ``x``.

    Returns
    -------
    torch.Tensor
        Tensor whose entries are ``x[idx[...]]``, of shape ``idx.shape``
        followed by :math:`F`.
    """
    flat = idx.reshape(-1)
    gathered = x.index_select(0, flat)
    return gathered.reshape(*idx.shape, x.shape[-1])


def _dilated_knn(
    *,
    query_coords: torch.Tensor,
    key_coords: torch.Tensor,
    k: int,
    dilation: int,
) -> torch.Tensor:
    r"""Distance-sorted k-NN indices with optional dilation.

    Returns, for each query, the indices into ``key_coords`` of its nearest
    neighbours (ascending distance). With ``dilation > 1`` the search widens
    to the :math:`k \cdot \mathrm{dilation}` nearest neighbours and then
    keeps every ``dilation``-th, giving a coarser receptive field at the same
    neighbour count -- a strided subsample of the sorted neighbour list.

    Parameters
    ----------
    query_coords : torch.Tensor
        Query positions of shape :math:`(N_q, D)`.
    key_coords : torch.Tensor
        Key positions of shape :math:`(N_k, D)`.
    k : int
        Number of neighbours retained per query (post-dilation).
    dilation : int
        Stride applied to the top-:math:`k \cdot \mathrm{dilation}` sorted
        indices before keeping the first :math:`k`.

    Returns
    -------
    torch.Tensor
        Index tensor of shape :math:`(N_q, k_{eff})`, dtype ``int64``, where
        :math:`k_{eff} = \max(1, \min(k \cdot \mathrm{dilation}, N_k) //
        \mathrm{dilation})`.
    """
    n_keys = int(key_coords.shape[0])
    k_wide = min(int(k) * int(dilation), n_keys)
    # ``knn`` arg order is (points = search-from / keys, queries = search-for);
    # returned indices index into ``points`` (i.e. ``key_coords``). Cast both
    # to float so dtype matches (knn requires it) and the distance ordering is
    # computed in fp32 regardless of the input dtype.
    idx, _ = knn(
        points=key_coords.float(),
        queries=query_coords.float(),
        k=k_wide,
    )
    if dilation > 1:
        idx = idx[:, :: int(dilation)]
    out_k = max(1, k_wide // int(dilation))
    return idx[:, :out_k].long()


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


class ResidualMLP(Module):
    r"""Pre-norm residual MLP with optional AdaLN/AdaLN-Zero conditioning.

    Applies ``LayerNorm`` -> ``Linear -> GELU -> Dropout -> Linear ->
    Dropout`` and adds the result back to the input. When
    ``conditioning_dim`` is set, a small zero-initialized MLP turns ``cond``
    into ``(shift, scale, gate)`` that modulate the pre-MLP and post-MLP
    signals in the AdaLN / AdaLN-Zero style.

    Parameters
    ----------
    dim : int
        Feature dimension :math:`D`.
    mlp_ratio : int
        Hidden dimension is :math:`\max(1, \mathrm{mlp\_ratio}) \cdot D`.
    dropout : float
        Dropout probability used inside the MLP.
    conditioning_dim : int, optional
        Size of the conditioning vector. ``None`` disables conditioning.
    adaln_zero : bool, optional
        If ``True``, the residual is gated by ``gate`` (AdaLN-Zero); if
        ``False``, it is gated by :math:`1 + \mathrm{gate}`. Default
        ``False``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(N, D)` or :math:`(B, N, D)`.
    cond : torch.Tensor, optional
        Conditioning of shape :math:`(D_{cond},)`, :math:`(1, D_{cond})` or
        :math:`(N, D_{cond})`. Required when ``conditioning_dim`` is set.

    Outputs
    -------
    torch.Tensor
        Output tensor with the same shape as ``x``.
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
        self.dim = int(dim)
        hidden = max(1, int(mlp_ratio)) * int(dim)
        self.norm = LayerNorm(int(dim))
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
        self,
        x: Float[torch.Tensor, "*dims dim"],
        cond: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "*dims dim"]:
        if not torch.compiler.is_compiling():
            if x.shape[-1] != self.dim:
                raise ValueError(
                    f"Expected x with last dim {self.dim}, got tensor of "
                    f"shape {tuple(x.shape)}"
                )
        h = self.norm(x)
        gate = None
        if self.conditioning is not None:
            if cond is None:
                raise ValueError(
                    "conditioning input must be provided for conditioned ResidualMLP."
                )
            shift, scale, gate = self.conditioning(_reshape_condition(cond)).chunk(
                3, dim=-1
            )
            h = h * (1.0 + scale) + shift
        out = self.net(h)
        if gate is not None:
            out = out * (gate if self.adaln_zero else (1.0 + gate))
        return x + out


class LocalPointTransformerBlock(Module):
    r"""Local self-attention block over a per-point k-NN graph.

    For each point, attends to its ``neighbor_k`` nearest neighbors with a
    learned relative-position bias and per-head (grouped) vector-attention
    scores -- the score for each neighbour is produced by an MLP applied to
    :math:`q - k + \delta` (Point Transformer), not a dot product. Followed
    by a :class:`ResidualMLP`. Optional AdaLN/AdaLN-Zero conditioning
    modulates both the attention sublayer and the feed-forward sublayer.

    When the input has at most one point, the attention sublayer is skipped
    and only the FFN is applied (still receiving ``cond`` if provided).

    Parameters
    ----------
    dim : int
        Feature dimension :math:`D`. Must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads (groups for the grouped vector attention).
    neighbor_k : int
        Number of nearest neighbors used per query point (post-dilation).
    dilation : int
        Stride applied to the top-:math:`k \cdot \mathrm{dilation}` neighbor
        indices before truncation. Lets the block attend at coarser
        receptive fields without re-running the search. Clamped to at least
        1.
    mlp_ratio : int
        Hidden multiplier for the inner ``ResidualMLP``.
    dropout : float
        Dropout used after the output projection and inside the FFN.
    knn_chunk_size : int
        Unused. Retained for backwards-compatible construction; the k-NN is
        delegated to :func:`physicsnemo.nn.functional.knn`, which selects and
        chunks its own backend.
    conditioning_dim : int, optional
        Size of the conditioning vector. ``None`` disables conditioning on
        both sublayers.
    adaln_zero : bool, optional
        Forwarded to the FFN and used to gate the attention output the same
        way (``gate`` vs :math:`1 + \mathrm{gate}`). Default ``False``.

    Forward
    -------
    features : torch.Tensor
        Per-point features of shape :math:`(N, D)`.
    coords : torch.Tensor
        Per-point coordinates of shape :math:`(N, 3)`.
    cond : torch.Tensor, optional
        Conditioning of shape :math:`(D_{cond},)`, :math:`(1, D_{cond})` or
        :math:`(N, D_{cond})`. Required when ``conditioning_dim`` is set.
    batch_ids : torch.Tensor, optional
        Integer tensor of shape :math:`(N,)`; when provided, neighbors from
        different batches are masked out of attention.

    Outputs
    -------
    torch.Tensor
        Updated per-point features of shape :math:`(N, D)`.

    Notes
    -----
    Raises ``ValueError`` if ``dim`` is not divisible by ``num_heads``, or if
    conditioning is requested but ``cond`` is not provided.
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
        self.norm = LayerNorm(self.dim)
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
        features: Float[torch.Tensor, "n dim"],
        coords: Float[torch.Tensor, "n 3"],
        cond: torch.Tensor | None = None,
        batch_ids: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "n dim"]:
        if not torch.compiler.is_compiling():
            if features.ndim != 2 or features.shape[1] != self.dim:
                raise ValueError(
                    f"Expected features of shape (N, {self.dim}), got tensor of "
                    f"shape {tuple(features.shape)}"
                )
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(
                    f"Expected coords of shape (N, 3), got tensor of shape "
                    f"{tuple(coords.shape)}"
                )
            if coords.shape[0] != features.shape[0]:
                raise ValueError(
                    "features and coords must share the point count N, got "
                    f"{int(features.shape[0])} and {int(coords.shape[0])}"
                )

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
            shift, scale, gate = self.conditioning(_reshape_condition(cond)).chunk(
                3, dim=-1
            )
            h = h * (1.0 + scale) + shift
        idx = _dilated_knn(
            query_coords=coords,
            key_coords=coords,
            k=min(self.neighbor_k, int(coords.shape[0])),
            dilation=self.dilation,
        )
        neighbor_mask = None
        if batch_ids is not None:
            gathered_batch_ids = _gather_rows(batch_ids.unsqueeze(-1), idx).squeeze(-1)
            neighbor_mask = gathered_batch_ids == batch_ids.unsqueeze(1)
        q = self.q_proj(h).reshape(int(h.shape[0]), self.num_heads, self.head_dim)
        k = _gather_rows(self.k_proj(h), idx).reshape(
            int(h.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        v = _gather_rows(self.v_proj(h), idx).reshape(
            int(h.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        rel = _gather_rows(coords, idx) - coords.unsqueeze(1)
        rel_bias = self.pos_proj(rel).reshape(
            int(h.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        attn_in = (q.unsqueeze(1) - k + rel_bias).reshape(
            int(h.shape[0]), idx.shape[1], self.dim
        )
        logits = self.attn_proj(attn_in).transpose(1, 2)
        attn = _apply_neighbor_mask(logits, neighbor_mask)
        value = (v + rel_bias).permute(0, 2, 1, 3)
        out = (attn.unsqueeze(-1) * value).sum(dim=2).reshape(int(h.shape[0]), self.dim)
        out = self.dropout(self.out_proj(out))
        if gate is not None:
            out = out * (gate if self.adaln_zero else (1.0 + gate))
        out = residual + out
        return self.ffn(out, cond=cond)


class LocalTokenCrossAttentionBlock(Module):
    r"""Local cross-attention from query tokens to a per-query k-NN of context.

    Each query attends to its ``neighbor_k`` nearest context tokens (by
    Euclidean distance in coordinate space) with a learned relative-position
    bias and per-head (grouped) vector-attention scores. Followed by a
    :class:`ResidualMLP`.

    When conditioning is enabled, a single MLP produces a 5-way chunked
    output ``(q_shift, q_scale, kv_shift, kv_scale, gate)``. The query side
    is modulated by ``(q_shift, q_scale)`` from ``cond``; the key/value side
    is modulated by ``(kv_shift, kv_scale)`` from ``context_cond if
    context_cond is not None else cond``.

    When either input has zero tokens the block is a no-op (returns
    ``query_features`` unchanged).

    Parameters
    ----------
    dim : int
        Feature dimension :math:`D` shared by queries and context. Must be
        divisible by ``num_heads``.
    num_heads : int
        Number of attention heads (groups for the grouped vector attention).
    neighbor_k : int
        Number of nearest context tokens used per query.
    mlp_ratio : int
        Hidden multiplier for the inner ``ResidualMLP``.
    dropout : float
        Dropout used after the output projection and inside the FFN.
    knn_chunk_size : int
        Unused. Retained for backwards-compatible construction; the k-NN is
        delegated to :func:`physicsnemo.nn.functional.knn`, which selects and
        chunks its own backend.
    conditioning_dim : int, optional
        Size of the conditioning vector. ``None`` disables conditioning.
    adaln_zero : bool, optional
        Forwarded to the FFN and used to gate the attention output the same
        way. Default ``False``.

    Forward
    -------
    query_features : torch.Tensor
        Query features of shape :math:`(N_q, D)`.
    query_coords : torch.Tensor
        Query coordinates of shape :math:`(N_q, 3)`.
    context_features : torch.Tensor
        Context features of shape :math:`(N_c, D)`.
    context_coords : torch.Tensor
        Context coordinates of shape :math:`(N_c, 3)`.
    cond : torch.Tensor, optional
        Query-side conditioning of shape :math:`(D_{cond},)` or
        :math:`(N_q, D_{cond})`. Required when ``conditioning_dim`` is set.
    context_cond : torch.Tensor, optional
        Key/value-side conditioning; defaults to ``cond`` when ``None``.
    query_batch_ids : torch.Tensor, optional
        Integer tensor of shape :math:`(N_q,)`.
    context_batch_ids : torch.Tensor, optional
        Integer tensor of shape :math:`(N_c,)`; when both batch-id tensors
        are provided, neighbors from different batches are masked out.

    Outputs
    -------
    torch.Tensor
        Updated query features of shape :math:`(N_q, D)`.

    Notes
    -----
    Raises ``ValueError`` if ``dim`` is not divisible by ``num_heads``, or if
    conditioning is requested but ``cond`` is not provided.
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
        self.norm_q = LayerNorm(self.dim)
        self.adaln_zero = bool(adaln_zero)
        self.norm_kv = LayerNorm(self.dim)
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
        query_features: Float[torch.Tensor, "nq dim"],
        query_coords: Float[torch.Tensor, "nq 3"],
        context_features: Float[torch.Tensor, "nc dim"],
        context_coords: Float[torch.Tensor, "nc 3"],
        cond: torch.Tensor | None = None,
        context_cond: torch.Tensor | None = None,
        query_batch_ids: torch.Tensor | None = None,
        context_batch_ids: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "nq dim"]:
        if not torch.compiler.is_compiling():
            if query_features.ndim != 2 or query_features.shape[1] != self.dim:
                raise ValueError(
                    f"Expected query_features of shape (Nq, {self.dim}), got "
                    f"tensor of shape {tuple(query_features.shape)}"
                )
            if context_features.ndim != 2 or context_features.shape[1] != self.dim:
                raise ValueError(
                    f"Expected context_features of shape (Nc, {self.dim}), got "
                    f"tensor of shape {tuple(context_features.shape)}"
                )
            if query_coords.ndim != 2 or query_coords.shape[1] != 3:
                raise ValueError(
                    f"Expected query_coords of shape (Nq, 3), got tensor of "
                    f"shape {tuple(query_coords.shape)}"
                )
            if context_coords.ndim != 2 or context_coords.shape[1] != 3:
                raise ValueError(
                    f"Expected context_coords of shape (Nc, 3), got tensor of "
                    f"shape {tuple(context_coords.shape)}"
                )
            if query_coords.shape[0] != query_features.shape[0]:
                raise ValueError(
                    "query_features and query_coords must share Nq, got "
                    f"{int(query_features.shape[0])} and {int(query_coords.shape[0])}"
                )
            if context_coords.shape[0] != context_features.shape[0]:
                raise ValueError(
                    "context_features and context_coords must share Nc, got "
                    f"{int(context_features.shape[0])} and "
                    f"{int(context_coords.shape[0])}"
                )

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
            kv_shift, kv_scale = self.conditioning(_reshape_condition(kv_source)).chunk(
                5, dim=-1
            )[2:4]
            kv_in = kv_in * (1.0 + kv_scale) + kv_shift
        idx = _dilated_knn(
            query_coords=query_coords,
            key_coords=context_coords,
            k=min(self.neighbor_k, int(context_coords.shape[0])),
            dilation=1,
        )
        neighbor_mask = None
        if query_batch_ids is not None and context_batch_ids is not None:
            gathered_batch_ids = _gather_rows(
                context_batch_ids.unsqueeze(-1), idx
            ).squeeze(-1)
            neighbor_mask = gathered_batch_ids == query_batch_ids.unsqueeze(1)
        q = self.q_proj(q_in).reshape(int(q_in.shape[0]), self.num_heads, self.head_dim)
        k = _gather_rows(self.k_proj(kv_in), idx).reshape(
            int(q_in.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        v = _gather_rows(self.v_proj(kv_in), idx).reshape(
            int(q_in.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        rel = _gather_rows(context_coords, idx) - query_coords.unsqueeze(1)
        rel_bias = self.pos_proj(rel).reshape(
            int(q_in.shape[0]), idx.shape[1], self.num_heads, self.head_dim
        )
        attn_in = (q.unsqueeze(1) - k + rel_bias).reshape(
            int(q_in.shape[0]), idx.shape[1], self.dim
        )
        logits = self.attn_proj(attn_in).transpose(1, 2)
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


__all__ = [
    "ResidualMLP",
    "LocalPointTransformerBlock",
    "LocalTokenCrossAttentionBlock",
]
