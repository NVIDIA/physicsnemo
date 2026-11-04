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

r"""Attention layers of :class:`~physicsnemo.experimental.models.argent.ArGEnT`.

The attention differs from a standard transformer in three ways:

1. **Continuous-coordinate RoPE.** Each token is rotated by its real-valued
   spatial coordinates rather than an integer position. The head dimension is
   split across spatial axes (``axis_slices``) and each slice is rotated by
   its own coordinate.
2. **Linear attention.** The output is :math:`Q\,(K^\top V) / N` with no
   softmax, so the cost is linear in the number of geometry tokens.
3. **Per-layer RoPE decay.** Layer :math:`d` multiplies the coordinates by
   :math:`\text{max\_freq}^2 / 2^d` before the RoPE frequency schedule. This
   is the factor :math:`\lambda` of Eq. (C.2) in `Chen et al., 2026
   <https://arxiv.org/abs/2602.11626>`_. The square of ``max_freq`` matches
   the configuration used for the paper's results.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Shaped

from physicsnemo.experimental.nn import build_rope_cos_sin_1d_continuous
from physicsnemo.nn import Mlp, apply_rotary_pos_emb


def _split_last_dim(x: torch.Tensor, splits: list[int]) -> list[torch.Tensor]:
    r"""Split the last dimension of ``x`` into chunks of the given sizes.

    Channels beyond ``sum(splits)`` are returned as one extra trailing chunk.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    splits : list of int
        Chunk sizes along the last dimension.

    Returns
    -------
    list of torch.Tensor
        Chunks of ``x``.
    """
    chunks: list[torch.Tensor] = []
    start = 0
    for s in splits:
        chunks.append(x[..., start : start + s])
        start += s
    if start < x.shape[-1]:
        chunks.append(x[..., start:])
    return chunks


class _RotaryLinearCrossAttentionND(nn.Module):
    r"""Multi-head linear cross-attention with N-D continuous-coordinate RoPE.

    Computes :math:`Q\,(K^\top V) / N` per head, where :math:`N` is the number
    of valid key/value tokens. :math:`K` and :math:`V` are layer-normalized
    per head, and RoPE is applied to :math:`Q` and :math:`K` per spatial axis
    on the channel slices given by ``axis_slices``.

    Parameters
    ----------
    dim_inout : int, optional, default=128
        Input and output feature dimension :math:`E`.
    dim_head : int, optional, default=128
        Per-head feature dimension.
    num_heads : int, optional, default=4
        Number of attention heads.
    axis_slices : list of int, optional
        Even channel counts within ``dim_head``, one per spatial axis. Entry
        ``i`` is rotated by coordinate ``i`` of ``pos_q`` and ``pos_kv``.
        Defaults to ``[dim_head]``.
    max_freq : float, optional, default=64.0
        RoPE frequency scale.
    scale : float, optional, default=1.0
        Additional RoPE coordinate multiplier (per-layer decay factor).
        Coordinates are multiplied by ``scale * max_freq``.

    Forward
    -------
    x_q : torch.Tensor
        Query tokens of shape :math:`(B, L_q, E)`.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, L_{kv}, E)`.
    pos_q : torch.Tensor
        Query coordinates of shape :math:`(B, L_q, n_{axes})`.
    pos_kv : torch.Tensor
        Key/value coordinates of shape :math:`(B, L_{kv}, n_{axes})`.
    padding_kv : torch.Tensor, optional
        Boolean or float mask of shape :math:`(B, L_{kv})` or
        :math:`(B, L_{kv}, 1)`. Any nonzero entry marks a valid token. Padded
        tokens are excluded from :math:`K^\top V` and from :math:`N`.

    Outputs
    -------
    torch.Tensor
        Output tokens of shape :math:`(B, L_q, E)`.
    """

    def __init__(
        self,
        dim_inout: int = 128,
        dim_head: int = 128,
        num_heads: int = 4,
        axis_slices: list[int] | None = None,
        max_freq: float = 64.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim_inout = int(dim_inout)
        self.dim_head = int(dim_head)
        self.num_heads = int(num_heads)
        self.embed_dim = self.dim_head * self.num_heads
        if axis_slices is None:
            axis_slices = [self.dim_head]
        if any(s % 2 != 0 for s in axis_slices):
            raise ValueError(
                f"Expected every entry of axis_slices to be even, got {axis_slices}"
            )
        if sum(axis_slices) > self.dim_head:
            raise ValueError(
                f"Expected sum(axis_slices) <= dim_head={self.dim_head}, "
                f"got {sum(axis_slices)}"
            )
        self.axis_slices = list(axis_slices)
        self.n_axes = len(self.axis_slices)
        self.max_freq = float(max_freq)
        self.scale = float(scale)

        # kv_proj emits K and V concatenated along the last dim.
        self.q_proj = nn.Linear(self.dim_inout, self.embed_dim, bias=False)
        self.kv_proj = nn.Linear(self.dim_inout, self.embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.dim_inout, bias=True)

        self.k_norm = nn.LayerNorm(self.dim_head)
        self.v_norm = nn.LayerNorm(self.dim_head)

        self._reset_projections()

    def _reset_projections(self) -> None:
        r"""Initialize each head block of Q, K, V as orthogonal plus identity.

        Each ``dim_head`` block is set to an orthogonal matrix with gain
        ``1/dim_head`` plus ``1/dim_head`` times the identity.
        """
        gain = 1.0 / self.dim_head

        def _init_block(block: torch.Tensor) -> None:
            nn.init.orthogonal_(block, gain=gain)
            m = min(block.shape)
            block[:m, :m] += gain * torch.eye(m, dtype=block.dtype, device=block.device)

        with torch.no_grad():
            for h in range(self.num_heads):
                _init_block(
                    self.q_proj.weight[h * self.dim_head : (h + 1) * self.dim_head]
                )
            for h in range(self.num_heads):
                k_start = h * self.dim_head
                v_start = self.embed_dim + k_start
                _init_block(self.kv_proj.weight[k_start : k_start + self.dim_head])
                _init_block(self.kv_proj.weight[v_start : v_start + self.dim_head])

    def _rope_tables(
        self, coords: Float[torch.Tensor, "batch length"], axis_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""RoPE ``(cos, sin)`` tables of shape :math:`(B, 1, L, \text{axis\_dim})`.

        Coordinates are scaled in fp32 so that reduced-precision inputs do not
        lose the scaling factor's precision. The singleton axis broadcasts the
        tables over heads.
        """
        cos, sin = build_rope_cos_sin_1d_continuous(
            coords.float() * (self.scale * self.max_freq), axis_dim
        )
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def forward(
        self,
        x_q: Float[torch.Tensor, "batch len_q dim"],
        x_kv: Float[torch.Tensor, "batch len_kv dim"],
        pos_q: Float[torch.Tensor, "batch len_q axes"],
        pos_kv: Float[torch.Tensor, "batch len_kv axes"],
        padding_kv: Shaped[torch.Tensor, "batch len_kv ..."] | None = None,
    ) -> Float[torch.Tensor, "batch len_q dim"]:
        B, L_q, E = x_q.shape
        L_kv = x_kv.shape[1]

        ### Input validation
        if not torch.compiler.is_compiling():
            if E != self.dim_inout:
                raise ValueError(
                    f"Expected x_q of shape (B, L_q, {self.dim_inout}) "
                    f"but got tensor of shape {tuple(x_q.shape)}"
                )
            if x_kv.shape[0] != B or x_kv.shape[-1] != self.dim_inout:
                raise ValueError(
                    f"Expected x_kv of shape ({B}, L_kv, {self.dim_inout}) "
                    f"but got tensor of shape {tuple(x_kv.shape)}"
                )
            if pos_q.shape != (B, L_q, self.n_axes):
                raise ValueError(
                    f"Expected pos_q of shape ({B}, {L_q}, {self.n_axes}) "
                    f"but got tensor of shape {tuple(pos_q.shape)}"
                )
            if pos_kv.shape != (B, L_kv, self.n_axes):
                raise ValueError(
                    f"Expected pos_kv of shape ({B}, {L_kv}, {self.n_axes}) "
                    f"but got tensor of shape {tuple(pos_kv.shape)}"
                )
            if padding_kv is not None and padding_kv.shape not in (
                (B, L_kv),
                (B, L_kv, 1),
            ):
                raise ValueError(
                    f"Expected padding_kv of shape ({B}, {L_kv}) or ({B}, {L_kv}, 1) "
                    f"but got tensor of shape {tuple(padding_kv.shape)}"
                )

        q = self.q_proj(x_q)
        k, v = self.kv_proj(x_kv).chunk(2, dim=-1)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        # Per-head LayerNorm on K and V over dim_head.
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Rotate each axis slice of head_dim by the matching coordinate.
        q_chunks = _split_last_dim(q, self.axis_slices)
        k_chunks = _split_last_dim(k, self.axis_slices)
        for ax, axis_dim in enumerate(self.axis_slices):
            cos_q, sin_q = self._rope_tables(pos_q[..., ax], axis_dim)
            cos_k, sin_k = self._rope_tables(pos_kv[..., ax], axis_dim)
            q_chunks[ax] = apply_rotary_pos_emb(q_chunks[ax], cos_q, sin_q)
            k_chunks[ax] = apply_rotary_pos_emb(k_chunks[ax], cos_k, sin_k)
        q = torch.cat(q_chunks, dim=-1)
        k = torch.cat(k_chunks, dim=-1)

        # Zero padded K/V tokens and normalize by the number of valid tokens.
        # masked_fill keeps the dtype of k/v and also clears non-finite values,
        # but in the forward pass only: the backward of k_norm and kv_proj can
        # still produce NaN gradients from non-finite padded inputs.
        if padding_kv is not None:
            valid = padding_kv.reshape(B, 1, L_kv, 1) != 0
            k = k.masked_fill(~valid, 0.0)
            v = v.masked_fill(~valid, 0.0)
            # Integer count. Dividing by it keeps the dtype of the output.
            n_valid = valid.sum(dim=-2, keepdim=True).clamp(min=1)  # (B, 1, 1, 1)
        else:
            n_valid = float(max(L_kv, 1))

        # Linear attention: Q (K^T V) / N, no softmax.
        kv = torch.matmul(k.transpose(-1, -2), v)  # (B, H, dim_head, dim_head)
        out = torch.matmul(q, kv) / n_valid  # (B, H, L_q, dim_head)

        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)


class _GeGELU(nn.Module):
    r"""GELU-gated linear unit (GEGLU, `Shazeer, 2020
    <https://arxiv.org/abs/2002.05202>`_).

    Splits the last dimension in half and returns
    :math:`\text{GELU}(x_1) \odot x_2`.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(\ldots, 2C)`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(\ldots, C)`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gelu = nn.GELU()

    def forward(
        self, x: Float[torch.Tensor, "... two_c"]
    ) -> Float[torch.Tensor, "... c"]:
        c = x.shape[-1] // 2
        return self.gelu(x[..., :c]) * x[..., c:]


class _FeedForward(nn.Module):
    r"""Two-layer feed-forward block with GEGLU activation.

    Parameters
    ----------
    dim : int
        Input and output feature dimension :math:`E`.
    hidden_dim : int
        Hidden dimension after gating. The first linear emits
        ``2 * hidden_dim`` channels.
    dropout : float, optional, default=0.0
        Dropout probability after each linear.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, L, E)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, L, E)`.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            _GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch len dim"]
    ) -> Float[torch.Tensor, "batch len dim"]:
        return self.net(x)


class _MultiLayerCrossAttention(nn.Module):
    r"""Stack of residual cross-attention and feed-forward blocks.

    Layer :math:`d` multiplies the RoPE coordinates by
    :math:`\text{max\_freq}^2 / 2^d` (``scale = max_freq / 2**d`` in
    :class:`_RotaryLinearCrossAttentionND`), so deeper layers see lower RoPE
    frequencies. The square of ``max_freq`` matches the configuration used for
    the results in `Chen et al., 2026 <https://arxiv.org/abs/2602.11626>`_.

    Parameters
    ----------
    dim_inout : int, optional, default=128
        Input and output feature dimension :math:`E`.
    depth : int, optional, default=4
        Number of attention + feed-forward layers.
    dim_head : int, optional, default=128
        Per-head feature dimension.
    num_heads : int, optional, default=4
        Number of attention heads.
    dim_mlp : int, optional, default=256
        Hidden dimension of the feed-forward blocks.
    axis_slices : list of int, optional
        Per-axis RoPE channel split. See :class:`_RotaryLinearCrossAttentionND`.
    dropout : float, optional, default=0.0
        Dropout probability inside the feed-forward blocks.
    max_freq : float, optional, default=64.0
        RoPE frequency scale.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use ReLU feed-forward blocks. Otherwise use GEGLU.

    Forward
    -------
    x_q : torch.Tensor
        Query tokens of shape :math:`(B, L_q, E)`.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, L_{kv}, E)`.
    pos_q : torch.Tensor
        Query coordinates of shape :math:`(B, L_q, n_{axes})`.
    pos_kv : torch.Tensor
        Key/value coordinates of shape :math:`(B, L_{kv}, n_{axes})`.
    padding_kv : torch.Tensor, optional
        Key/value padding mask. See :class:`_RotaryLinearCrossAttentionND`.

    Outputs
    -------
    torch.Tensor
        Output tokens of shape :math:`(B, L_q, E)`.
    """

    def __init__(
        self,
        dim_inout: int = 128,
        depth: int = 4,
        dim_head: int = 128,
        num_heads: int = 4,
        dim_mlp: int = 256,
        axis_slices: list[int] | None = None,
        dropout: float = 0.0,
        max_freq: float = 64.0,
        use_relu_mlp: bool = False,
    ) -> None:
        super().__init__()

        def _ffn() -> nn.Module:
            if use_relu_mlp:
                return Mlp(dim_inout, dim_mlp, dim_inout, act_layer="relu", drop=dropout)
            return _FeedForward(dim_inout, dim_mlp, dropout=dropout)

        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    _RotaryLinearCrossAttentionND(
                        dim_inout=dim_inout,
                        dim_head=dim_head,
                        num_heads=num_heads,
                        axis_slices=axis_slices,
                        max_freq=max_freq,
                        scale=max_freq / (2**d),
                    ),
                    _ffn(),
                ]
            )
            for d in range(depth)
        )

    def forward(
        self,
        x_q: Float[torch.Tensor, "batch len_q dim"],
        x_kv: Float[torch.Tensor, "batch len_kv dim"],
        pos_q: Float[torch.Tensor, "batch len_q axes"],
        pos_kv: Float[torch.Tensor, "batch len_kv axes"],
        padding_kv: Shaped[torch.Tensor, "batch len_kv ..."] | None = None,
    ) -> Float[torch.Tensor, "batch len_q dim"]:
        for attn, ffn in self.layers:
            x_q = attn(x_q, x_kv, pos_q, pos_kv, padding_kv=padding_kv) + x_q
            x_q = ffn(x_q) + x_q
        return x_q
