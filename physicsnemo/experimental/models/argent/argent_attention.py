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
3. **Per-layer RoPE decay.** Layer :math:`d` scales its RoPE frequencies by
   :math:`1 / 2^d`.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Shaped

from physicsnemo.nn.module.rope import apply_rotary_pos_emb


def build_continuous_rope_cos_sin(
    coords: Float[torch.Tensor, "batch length"],
    axis_dim: int,
    max_freq: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build RoPE ``(cos, sin)`` tables for continuous per-token coordinates.

    Unlike :class:`~physicsnemo.nn.module.rope.RotaryEmbedding1DTables`, which
    encodes integer positions, this accepts real-valued coordinates. The
    angle for frequency index :math:`k` is

    .. math::

        \theta_k = \text{scale} \cdot \text{max\_freq} \cdot x \cdot
        10000^{-2k/\text{axis\_dim}},\quad k = 0, \ldots, \text{axis\_dim}/2 - 1

    and the tables use the interleaved layout ``[c0, c0, c1, c1, ...]``
    expected by :func:`~physicsnemo.nn.module.rope.apply_rotary_pos_emb`.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinate along one spatial axis, of shape :math:`(B, L)`.
    axis_dim : int
        Number of channels rotated by this axis. Must be even.
    max_freq : float
        RoPE frequency scale.
    scale : float
        Additional multiplier on the coordinates (per-layer decay factor).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)`` tables of shape :math:`(B, L, \text{axis\_dim})`, in fp32.
    """
    if not torch.compiler.is_compiling() and axis_dim % 2 != 0:
        raise ValueError(f"Expected axis_dim to be even, got {axis_dim}")
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, axis_dim, 2, dtype=torch.float32, device=coords.device)
            / axis_dim
        )
    )  # (axis_dim/2,)
    t = coords.float() * (scale * max_freq)  # (B, L)
    freqs = torch.einsum("bl,d->bld", t, inv_freq)  # (B, L, axis_dim/2)
    freqs = freqs.repeat_interleave(2, dim=-1)  # (B, L, axis_dim), interleaved
    return freqs.cos(), freqs.sin()


def _split_last_dim(x: torch.Tensor, splits: List[int]) -> List[torch.Tensor]:
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
    chunks: List[torch.Tensor] = []
    start = 0
    for s in splits:
        chunks.append(x[..., start : start + s])
        start += s
    if start < x.shape[-1]:
        chunks.append(x[..., start:])
    return chunks


class RotaryLinearCrossAttentionND(nn.Module):
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
        Even channel counts within ``dim_head``, one per spatial axis; entry
        ``i`` is rotated by coordinate ``i`` of ``pos_q``/``pos_kv``. Defaults
        to ``[dim_head]``.
    dropout : float, optional, default=0.0
        Unused.
    max_freq : float, optional, default=64.0
        RoPE frequency scale.
    scale : float, optional, default=1.0
        Additional RoPE coordinate multiplier (per-layer decay factor).

    Forward
    -------
    x_q : torch.Tensor
        Query tokens of shape :math:`(B, L_q, E)`.
    x_kv : torch.Tensor, optional
        Key/value tokens of shape :math:`(B, L_{kv}, E)`. Defaults to ``x_q``.
    pos_q : torch.Tensor, optional
        Query coordinates of shape :math:`(B, L_q, n_{axes})`. RoPE is applied
        only when coordinates are given.
    pos_kv : torch.Tensor, optional
        Key/value coordinates of shape :math:`(B, L_{kv}, n_{axes})`.
    padding_kv : torch.Tensor, optional
        Boolean or float mask of shape :math:`(B, L_{kv})` or
        :math:`(B, L_{kv}, 1)`. Nonzero entries mark valid tokens; padded
        tokens are excluded from :math:`K^\top V` and from :math:`N`.

    Outputs
    -------
    torch.Tensor
        Output tokens of shape :math:`(B, L_q, E)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.argent.argent_attention import RotaryLinearCrossAttentionND
    >>> attn = RotaryLinearCrossAttentionND(dim_inout=64, dim_head=64, num_heads=2,
    ...                                     axis_slices=[32, 32], max_freq=200.0)
    >>> x_q = torch.randn(2, 100, 64)
    >>> x_kv = torch.randn(2, 50, 64)
    >>> pos_q = torch.randn(2, 100, 2)
    >>> pos_kv = torch.randn(2, 50, 2)
    >>> out = attn(x_q, x_kv, pos_q=pos_q, pos_kv=pos_kv)
    >>> out.shape
    torch.Size([2, 100, 64])
    """

    def __init__(
        self,
        dim_inout: int = 128,
        dim_head: int = 128,
        num_heads: int = 4,
        axis_slices: Optional[List[int]] = None,
        dropout: float = 0.0,
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
        self.dropout = float(dropout)

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

    def forward(
        self,
        x_q: Float[torch.Tensor, "batch len_q dim"],
        x_kv: Optional[Float[torch.Tensor, "batch len_kv dim"]] = None,
        pos_q: Optional[Float[torch.Tensor, "batch len_q axes"]] = None,
        pos_kv: Optional[Float[torch.Tensor, "batch len_kv axes"]] = None,
        padding_kv: Optional[Shaped[torch.Tensor, "batch len_kv ..."]] = None,
    ) -> Float[torch.Tensor, "batch len_q dim"]:
        if x_kv is None:
            x_kv = x_q
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
            if pos_q is not None and pos_q.shape != (B, L_q, self.n_axes):
                raise ValueError(
                    f"Expected pos_q of shape ({B}, {L_q}, {self.n_axes}) "
                    f"but got tensor of shape {tuple(pos_q.shape)}"
                )
            if pos_kv is not None and pos_kv.shape != (B, L_kv, self.n_axes):
                raise ValueError(
                    f"Expected pos_kv of shape ({B}, {L_kv}, {self.n_axes}) "
                    f"but got tensor of shape {tuple(pos_kv.shape)}"
                )
            if padding_kv is not None and padding_kv.numel() != B * L_kv:
                raise ValueError(
                    f"Expected padding_kv of shape ({B}, {L_kv}) or ({B}, {L_kv}, 1) "
                    f"but got tensor of shape {tuple(padding_kv.shape)}"
                )

        # Self-attention may give only pos_q; cross-attention needs both.
        if x_kv is x_q:
            use_rope = pos_q is not None
            if use_rope and pos_kv is None:
                pos_kv = pos_q
        else:
            use_rope = pos_q is not None and pos_kv is not None

        q = self.q_proj(x_q)
        k, v = self.kv_proj(x_kv).chunk(2, dim=-1)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        # Per-head LayerNorm on K and V over dim_head.
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Rotate each axis slice of head_dim by the matching coordinate.
        if use_rope:
            q_chunks = _split_last_dim(q, self.axis_slices)
            k_chunks = _split_last_dim(k, self.axis_slices)
            for ax, axis_dim in enumerate(self.axis_slices):
                cos_q, sin_q = build_continuous_rope_cos_sin(
                    pos_q[..., ax], axis_dim, self.max_freq, self.scale
                )  # (B, L_q, axis_dim)
                cos_k, sin_k = build_continuous_rope_cos_sin(
                    pos_kv[..., ax], axis_dim, self.max_freq, self.scale
                )  # (B, L_kv, axis_dim)
                # unsqueeze(1) broadcasts the tables over heads.
                q_chunks[ax] = apply_rotary_pos_emb(
                    q_chunks[ax], cos_q.unsqueeze(1), sin_q.unsqueeze(1)
                )
                k_chunks[ax] = apply_rotary_pos_emb(
                    k_chunks[ax], cos_k.unsqueeze(1), sin_k.unsqueeze(1)
                )
            q = torch.cat(q_chunks, dim=-1)
            k = torch.cat(k_chunks, dim=-1)

        # Zero padded K/V tokens and normalize by the number of valid tokens.
        if padding_kv is not None:
            mask = padding_kv.reshape(B, 1, L_kv, 1).float()
            k = k * mask
            v = v * mask
            n_valid = mask.sum(dim=-2, keepdim=True).clamp(min=1.0)  # (B, 1, 1, 1)
        else:
            n_valid = float(max(L_kv, 1))

        # Linear attention: Q (K^T V) / N, no softmax.
        kv = torch.matmul(k.transpose(-1, -2), v)  # (B, H, dim_head, dim_head)
        out = torch.matmul(q, kv) / n_valid  # (B, H, L_q, dim_head)

        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)


class GeGELU(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[-1] // 2
        return self.gelu(x[..., :c]) * x[..., c:]


class FeedForward(nn.Module):
    r"""Two-layer feed-forward block with GEGLU activation.

    Parameters
    ----------
    dim : int
        Input and output feature dimension :math:`E`.
    hidden_dim : int
        Hidden dimension after gating; the first linear emits ``2 * hidden_dim``.
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
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReLUFeedForward(nn.Module):
    r"""Two-layer feed-forward block with ReLU activation.

    Parameters
    ----------
    dim : int
        Input and output feature dimension :math:`E`.
    hidden_dim : int
        Hidden dimension.
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
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiLayerCrossAttention(nn.Module):
    r"""Stack of residual :class:`RotaryLinearCrossAttentionND` and feed-forward blocks.

    Layer :math:`d` uses RoPE ``scale = max_freq / 2**d``, so deeper layers see
    lower RoPE frequencies.

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
        Per-axis RoPE channel split; see :class:`RotaryLinearCrossAttentionND`.
    dropout : float, optional, default=0.0
        Dropout probability inside the feed-forward blocks.
    max_freq : float, optional, default=64.0
        RoPE frequency scale.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use :class:`ReLUFeedForward`; otherwise :class:`FeedForward`.

    Forward
    -------
    x_q : torch.Tensor
        Query tokens of shape :math:`(B, L_q, E)`.
    x_kv : torch.Tensor, optional
        Key/value tokens of shape :math:`(B, L_{kv}, E)`. Defaults to ``x_q``.
    pos_q : torch.Tensor, optional
        Query coordinates of shape :math:`(B, L_q, n_{axes})`.
    pos_kv : torch.Tensor, optional
        Key/value coordinates of shape :math:`(B, L_{kv}, n_{axes})`.
    padding_kv : torch.Tensor, optional
        Key/value padding mask; see :class:`RotaryLinearCrossAttentionND`.

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
        axis_slices: Optional[List[int]] = None,
        dropout: float = 0.0,
        max_freq: float = 64.0,
        use_relu_mlp: bool = False,
    ) -> None:
        super().__init__()
        ffn_cls = ReLUFeedForward if use_relu_mlp else FeedForward
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    RotaryLinearCrossAttentionND(
                        dim_inout=dim_inout,
                        dim_head=dim_head,
                        num_heads=num_heads,
                        axis_slices=axis_slices,
                        dropout=dropout,
                        max_freq=max_freq,
                        scale=max_freq / (2**d),
                    ),
                    ffn_cls(dim_inout, dim_mlp, dropout=dropout),
                ]
            )
            for d in range(depth)
        )

    def forward(
        self,
        x_q: Float[torch.Tensor, "batch len_q dim"],
        x_kv: Optional[Float[torch.Tensor, "batch len_kv dim"]] = None,
        pos_q: Optional[Float[torch.Tensor, "batch len_q axes"]] = None,
        pos_kv: Optional[Float[torch.Tensor, "batch len_kv axes"]] = None,
        padding_kv: Optional[Shaped[torch.Tensor, "batch len_kv ..."]] = None,
    ) -> Float[torch.Tensor, "batch len_q dim"]:
        for attn, ffn in self.layers:
            x_q = (
                attn(x_q, x_kv=x_kv, pos_q=pos_q, pos_kv=pos_kv, padding_kv=padding_kv)
                + x_q
            )
            x_q = ffn(x_q) + x_q
        return x_q
