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

r"""Attention building blocks for the Cross_Unet PV-power model.

The mask branch from upstream ``FullAttention`` is removed because every
``Cross_Unet`` instantiation uses ``mask_flag=False``; modules such as the
upstream ``space_attn`` / ``dim_sender`` / ``dim_receiver`` / ``router`` that are
allocated but never read in the forward pass are also dropped here for
clarity.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
from einops import rearrange


class FullAttention(nn.Module):
    r"""Scaled dot-product attention without masking.

    Parameters
    ----------
    attention_dropout : float, optional, default=0.0
        Dropout probability applied to the attention weights.
    """

    def __init__(self, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, e = queries.shape
        scale = 1.0 / sqrt(e)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        return out.contiguous()


class AttentionLayer(nn.Module):
    r"""Multi-head attention layer wrapping :class:`FullAttention`.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the input and output tokens.
    n_heads : int
        Number of attention heads.
    attention_dropout : float, optional, default=0.0
        Dropout applied within the attention scores.
    d_keys : int or None, optional, default=None
        Per-head key dimension. Defaults to ``d_model // n_heads``.
    d_values : int or None, optional, default=None
        Per-head value dimension. Defaults to ``d_model // n_heads``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_dropout: float = 0.0,
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(attention_dropout=attention_dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        b, lq, _ = queries.shape
        _, lk, _ = keys.shape
        h = self.n_heads
        q = self.query_projection(queries).view(b, lq, h, -1)
        k = self.key_projection(keys).view(b, lk, h, -1)
        v = self.value_projection(values).view(b, lk, h, -1)
        out = self.inner_attention(q, k, v)
        out = out.view(b, lq, -1)
        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    r"""Two-stage attention block: time attention + correlation channel mixing.

    The block first applies multi-head self-attention along the patch axis of
    each channel independently (the *time* stage). It then mixes channels with
    a precomputed correlation matrix (the *channel* stage). The result is
    fused with a residual connection, layer-normalised, and refined by a
    feed-forward MLP.

    Parameters
    ----------
    n_vars : int
        Total number of channels :math:`C` flowing through the block.
    d_model : int
        Per-token hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int or None, optional, default=None
        Feed-forward inner width. Defaults to ``4 * d_model``.
    dropout : float, optional, default=0.1
        Dropout probability for attention and MLP residuals.
    swap_corr_axis : bool, optional, default=False
        If True, multiply the channel-mixing correlation matrix on the right
        of the channel-flattened activations (``[B, L, C] @ [B, C, C]``)
        rather than the left.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        swap_corr_axis: bool = False,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            d_model=d_model, n_heads=n_heads, attention_dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.swap_corr_axis = swap_corr_axis
        self.n_vars = n_vars

    def _apply_correlation(
        self, x_flat: torch.Tensor, corr: torch.Tensor
    ) -> torch.Tensor:
        # ``x_flat``: (B, n_vars, seg_num * d_model); ``corr``: (B, n_vars, n_vars).
        if self.swap_corr_axis:
            return torch.bmm(x_flat.permute(0, 2, 1), corr).permute(0, 2, 1)
        return torch.bmm(corr, x_flat)

    def forward(
        self, x: torch.Tensor, corr: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # ``x``: (B, n_vars, seg_num, d_model).
        b, n_vars, seg_num, d_model = x.shape

        # Time-axis attention applied independently per channel.
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc = self.time_attention(time_in, time_in, time_in)
        time_reco = time_enc.view(b, n_vars, seg_num, d_model)

        # Channel-axis mixing using the supplied correlation matrix.
        x_flat = rearrange(x, "b ts_d seg_num d_model -> b ts_d (seg_num d_model)")
        corr_matrix = self._apply_correlation(x_flat, corr).view(
            b, n_vars, seg_num, d_model
        )
        attn_trace = [x_flat.detach(), corr.detach(), corr_matrix.detach()]

        # Residual fusion + post-norm + feed-forward block.
        dim_in = self.norm1(x + self.dropout(time_reco + corr_matrix))
        dim_in = self.norm2(dim_in + self.dropout(self.mlp(dim_in)))
        return dim_in, attn_trace


class ParallelTwoStageAttentionLayer(nn.Module):
    r"""Parallel two-stage attention: simultaneous time and channel attention.

    Differs from :class:`TwoStageAttentionLayer` in that it replaces the
    correlation-based channel mixer with a second self-attention pass over the
    channel axis. The two stages are then fused by addition.

    Parameters
    ----------
    n_vars : int
        Total number of channels :math:`C`.
    d_model : int
        Per-token hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int or None, optional, default=None
        Feed-forward inner width. Defaults to ``4 * d_model``.
    dropout : float, optional, default=0.1
        Dropout probability.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            d_model=d_model, n_heads=n_heads, attention_dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.n_vars = n_vars

    def forward(
        self, x: torch.Tensor, corr: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # ``x``: (B, n_vars, seg_num, d_model); ``corr`` accepted but unused.
        del corr
        b, n_vars, seg_num, d_model = x.shape

        # Time-axis attention.
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc = self.time_attention(time_in, time_in, time_in)
        time_reco = time_enc.view(b, n_vars, seg_num, d_model)

        # Channel-axis attention reuses the same head weights.
        x_perm = x.permute(0, 2, 1, 3).contiguous().view(b * seg_num, n_vars, d_model)
        space_enc = self.time_attention(x_perm, x_perm, x_perm)
        space_reco = space_enc.view(b, seg_num, n_vars, d_model).permute(0, 2, 1, 3)
        attn_trace = [time_reco.detach(), space_reco.detach()]

        # Residual fusion + post-norm + feed-forward block.
        dim_in = self.norm1(x + self.dropout(time_reco + space_reco))
        dim_in = self.norm2(dim_in + self.dropout(self.mlp(dim_in)))
        return dim_in, attn_trace
