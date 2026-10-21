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

r"""Attention primitives used by :class:`~physicsnemo.experimental.models.argent.ArGEnTCross`.

Ported from the reference implementation accompanying "ArGEnT: Arbitrary
Geometry-encoded Transformer for Operator Learning" (Chen et al., 2026). The
port preserves three algorithmic choices that are not swappable for their
standard-transformer equivalents without changing the trained model:

1. **Continuous-coordinate rotary embeddings.** Instead of rotating by integer
   sequence/grid positions, each token carries its own real-valued position
   :math:`(x, y, \ldots)`; ``head_dim`` is sliced across spatial axes
   (``axis_slices``) and each axis-slice is rotated independently.
2. **Linearized cross-attention.** The attention output is
   :math:`Q\,(K^\top V) / N` — no softmax. This is what argent uses to make
   attention over large geometry point clouds tractable, and the training
   dynamics (and therefore the paper's Table 3 numbers) depend on it.
3. **Per-layer RoPE scale decay.** Layer :math:`d \in \{0, \ldots, \text{depth}-1\}`
   uses ``scale = max_freq / 2**d``, halving the effective frequency range at
   deeper layers.

The rotation math reuses
:func:`~physicsnemo.nn.module.rope.apply_rotary_pos_emb` from physicsnemo,
with cos/sin built in the physicsnemo-standard **interleaved** layout
(``[c0, c0, c1, c1, ...]``). This deviates from argent_exp, which uses a
**halves** layout (``[c0, c1, ..., c0, c1, ...]``) and therefore produces a
different — arguably non-coherent — rotation of adjacent channels. See
:func:`build_continuous_rope_cos_sin` for the switch site; a training run
with the interleaved convention is needed to establish new baseline numbers
(paper Table 3 was reported with the halves layout).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from physicsnemo.nn.module.rope import apply_rotary_pos_emb


def build_continuous_rope_cos_sin(
    coords: Float[torch.Tensor, "batch length"],
    axis_dim: int,
    max_freq: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build RoPE ``(cos, sin)`` tables for continuous per-token coordinates.

    Unlike :class:`~physicsnemo.nn.module.rope.RotaryEmbedding1DTables`, which
    encodes integer sequence positions, this helper accepts real-valued
    coordinates and returns cos/sin tables sized to match the tokens. The
    frequency schedule and coordinate scaling mirror argent_exp's
    ``RotaryEmbedding`` module ([MultiHeadAttetnion.py:10-23]):

    .. math::

        \text{inv\_freq}_k = 10000^{-2k/\text{axis\_dim}},\quad
        k = 0, 1, \ldots, \text{axis\_dim}/2 - 1

    The coordinate is multiplied by ``scale * max_freq`` before broadcasting
    against ``inv_freq`` — the ``scale`` factor is passed by the caller so a
    multi-layer stack can shrink it geometrically at deeper layers
    (``scale = max_freq / 2**layer_idx`` in argent).

    Parameters
    ----------
    coords : torch.Tensor
        Per-token coordinate for one spatial axis, of shape
        :math:`(B, L)`.
    axis_dim : int
        Number of channels this axis owns within ``head_dim``. Must be even.
    max_freq : float
        Maximum frequency (the RoPE angular scale, prior to layer decay).
    scale : float
        Multiplicative layer-decay factor applied to ``coords`` before the
        frequency broadcast. Argent's convention is
        ``scale = max_freq / 2**layer_idx``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)`` tables of shape :math:`(B, L, \text{axis\_dim})`, in fp32.
    """
    if not torch.compiler.is_compiling() and axis_dim % 2 != 0:
        raise ValueError(
            f"Expected axis_dim to be even, got {axis_dim}"
        )
    device = coords.device
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim)
    )  # (axis_dim/2,)
    t = coords.float() * (scale * max_freq)  # (B, L)
    freqs = torch.einsum("bl,d->bld", t, inv_freq)  # (B, L, axis_dim/2)
    # Interleaved layout [f0, f0, f1, f1, ...] — physicsnemo convention.
    # Deviates from argent_exp which uses halves layout via
    # `torch.cat((freqs, freqs), dim=-1)` — kept below for reference. See
    # class docstring for the trade-off.
    freqs = freqs.repeat_interleave(2, dim=-1)  # (B, L, axis_dim)
    # freqs = torch.cat((freqs, freqs), dim=-1)  # argent_exp halves layout
    return freqs.cos(), freqs.sin()


def _split_last_dim(x: torch.Tensor, splits: List[int]) -> List[torch.Tensor]:
    r"""Split the last dimension of ``x`` into contiguous chunks of the given sizes.

    Any unclaimed trailing channels (``x.shape[-1] > sum(splits)``) are
    returned as an extra chunk — these pass through the attention without
    rotation. Matches argent_exp's ``split_last_dim``
    ([MultiHeadAttetnion.py:36-47]).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    splits : list of int
        Per-chunk channel counts along the last dimension.

    Returns
    -------
    list of torch.Tensor
        Chunks of ``x``. If ``sum(splits) < x.shape[-1]`` there is one extra
        trailing chunk holding the leftover channels.
    """
    D = x.shape[-1]
    total = sum(splits)
    chunks: List[torch.Tensor] = []
    start = 0
    for s in splits:
        chunks.append(x[..., start : start + s])
        start += s
    if start < D:
        chunks.append(x[..., start:])
    return chunks


class RotaryLinearCrossAttentionND(nn.Module):
    r"""Multi-head linear cross-attention with N-D continuous-coordinate RoPE.

    Faithful port of argent_exp's ``RotaryMHA_ND`` ([MultiHeadAttetnion.py:50-234]).
    The attention is **linear** (:math:`Q\,K^\top V / N`, not softmax) and RoPE
    is applied to real-valued per-token coordinates split per spatial axis by
    ``axis_slices``. See the module docstring of
    :mod:`physicsnemo.experimental.models.argent.argent_attention` for why
    these choices matter for reproducibility.

    Parameters
    ----------
    dim_inout : int, optional, default=128
        Input and output feature dimension.
    dim_head : int, optional, default=128
        Per-head feature dimension. The full multi-head embedding dimension is
        ``dim_head * num_heads``.
    num_heads : int, optional, default=4
        Number of attention heads.
    axis_slices : list of int, optional
        Per-axis channel slice within ``dim_head`` (each entry must be even).
        Position ``i`` in ``axis_slices`` corresponds to spatial axis ``i`` in
        the ``pos_q`` / ``pos_kv`` tensors. If ``None``, defaults to
        ``[dim_head]`` (single axis using the full head).
    dropout : float, optional, default=0.0
        Dropout probability. Currently unused (argent's reference does not
        apply dropout inside the attention op) — kept in the signature for API
        stability.
    max_freq : float, optional, default=64.0
        Base RoPE frequency (prior to layer decay).
    scale : float, optional, default=1.0
        Layer-decay factor applied to coordinates; argent's multi-layer
        wrapper (:class:`MultiLayerCrossAttention`) sets this to
        ``max_freq / 2**layer_idx``.

    Forward
    -------
    x_q : torch.Tensor
        Query tokens of shape :math:`(B, L_q, E)`, with :math:`E =` ``dim_inout``.
    x_kv : torch.Tensor, optional
        Key/value tokens of shape :math:`(B, L_{kv}, E)`. If ``None``, defaults
        to ``x_q`` (self-attention).
    pos_q : torch.Tensor, optional
        Query coordinates of shape :math:`(B, L_q, n_{\text{axes}})`. Required
        for cross-attention; ``None`` disables RoPE.
    pos_kv : torch.Tensor, optional
        KV coordinates of shape :math:`(B, L_{kv}, n_{\text{axes}})`.
    padding_kv : torch.Tensor, optional
        Boolean/float mask over KV tokens, of shape :math:`(B, L_{kv})` or
        :math:`(B, L_{kv}, 1)` — ``True`` (or nonzero) means "keep." Padded
        tokens contribute zero to :math:`K^\top V` and are excluded from the
        normalizer :math:`N`.

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
        for s in axis_slices:
            if s % 2 != 0:
                raise ValueError(
                    f"Each entry of axis_slices must be even, got {axis_slices}"
                )
        if sum(axis_slices) > self.dim_head:
            raise ValueError(
                f"sum(axis_slices)={sum(axis_slices)} exceeds dim_head={self.dim_head}"
            )
        self.axis_slices = list(axis_slices)
        self.n_axes = len(self.axis_slices)
        self.max_freq = float(max_freq)
        self.scale = float(scale)
        self.dropout = float(dropout)

        # Projections: kv_proj emits K and V concatenated along the last dim.
        self.q_proj = nn.Linear(self.dim_inout, self.embed_dim, bias=False)
        self.kv_proj = nn.Linear(self.dim_inout, self.embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.dim_inout, bias=True)

        # Per-head LayerNorm on K and V (matches argent's k_norm/v_norm).
        self.k_norm = nn.LayerNorm(self.dim_head)
        self.v_norm = nn.LayerNorm(self.dim_head)

        self._reset_projections()

    def _reset_projections(self) -> None:
        r"""Apply orthogonal-plus-identity per-head init to Q, K, V projections.

        Mirrors argent_exp's init recipe: for each attention head, initialize
        the head's block of weights with an orthogonal matrix scaled by
        ``1/dim_head``, then add ``1/dim_head`` times the identity on the
        diagonal. This preserves norms at initialization and is what argent
        was trained with.
        """
        init_gain = 1.0 / self.dim_head
        diag_weight = 1.0 / self.dim_head
        with torch.no_grad():
            # Q projection: (embed_dim, dim_inout), per-head block along dim 0.
            for h in range(self.num_heads):
                block = self.q_proj.weight[h * self.dim_head : (h + 1) * self.dim_head, :]
                nn.init.orthogonal_(block, gain=init_gain)
                # Add identity along the smaller dim to keep the shape general.
                m = min(block.shape[0], block.shape[1])
                block[:m, :m] += diag_weight * torch.eye(m, dtype=block.dtype, device=block.device)

            # KV projection: (2*embed_dim, dim_inout). Argent applies the same
            # ortho+identity init to each head block for K (first half) and V
            # (second half). See MultiHeadAttetnion.py:124-135.
            kv_weight = self.kv_proj.weight  # (2*embed_dim, dim_inout)
            for h in range(self.num_heads):
                k_block = kv_weight[h * self.dim_head : (h + 1) * self.dim_head, :]
                nn.init.orthogonal_(k_block, gain=init_gain)
                m = min(k_block.shape[0], k_block.shape[1])
                k_block[:m, :m] += diag_weight * torch.eye(m, dtype=k_block.dtype, device=k_block.device)

                v_start = self.embed_dim + h * self.dim_head
                v_block = kv_weight[v_start : v_start + self.dim_head, :]
                nn.init.orthogonal_(v_block, gain=init_gain)
                m = min(v_block.shape[0], v_block.shape[1])
                v_block[:m, :m] += diag_weight * torch.eye(m, dtype=v_block.dtype, device=v_block.device)

    def forward(
        self,
        x_q: Float[torch.Tensor, "batch len_q dim"],
        x_kv: Optional[Float[torch.Tensor, "batch len_kv dim"]] = None,
        pos_q: Optional[Float[torch.Tensor, "batch len_q axes"]] = None,
        pos_kv: Optional[Float[torch.Tensor, "batch len_kv axes"]] = None,
        padding_kv: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "batch len_q dim"]:
        if x_kv is None:
            x_kv = x_q
        B, L_q, E = x_q.shape
        L_kv = x_kv.shape[1]

        # Input validation (skipped under torch.compile per MOD-005).
        if not torch.compiler.is_compiling():
            if E != self.dim_inout:
                raise ValueError(
                    f"Expected x_q with last dim {self.dim_inout}, got {tuple(x_q.shape)}"
                )
            if x_kv.shape[0] != B or x_kv.shape[-1] != self.dim_inout:
                raise ValueError(
                    f"Expected x_kv shape (B, L_kv, {self.dim_inout}), got {tuple(x_kv.shape)}"
                )
            if pos_q is not None and (
                pos_q.ndim != 3 or pos_q.shape[0] != B or pos_q.shape[1] != L_q or pos_q.shape[2] != self.n_axes
            ):
                raise ValueError(
                    f"Expected pos_q shape (B, L_q, {self.n_axes}), got {tuple(pos_q.shape)}"
                )
            if pos_kv is not None and (
                pos_kv.ndim != 3 or pos_kv.shape[0] != B or pos_kv.shape[1] != L_kv or pos_kv.shape[2] != self.n_axes
            ):
                raise ValueError(
                    f"Expected pos_kv shape (B, L_kv, {self.n_axes}), got {tuple(pos_kv.shape)}"
                )

        # Decide whether to apply RoPE.  Self-attention with pos_q inferred from pos_kv
        # was in argent_exp; here we accept both cross- and self-configurations.
        is_self = x_kv is x_q
        if is_self:
            use_rope = pos_q is not None
            if use_rope and pos_kv is None:
                pos_kv = pos_q
        else:
            use_rope = pos_q is not None and pos_kv is not None

        # Project to Q, K, V and reshape into per-head layout.
        q = self.q_proj(x_q)  # (B, L_q, embed_dim)
        k, v = self.kv_proj(x_kv).chunk(2, dim=-1)  # each (B, L_kv, embed_dim)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        # Per-head LayerNorm on K and V — collapse (batch, heads) into one
        # normalization axis so nn.LayerNorm operates on the (L, dim_head)
        # slice for each (b, h) independently.
        k = rearrange(
            self.k_norm(rearrange(k, "b h l d -> (b h) l d")),
            "(b h) l d -> b h l d",
            b=B,
        )
        v = rearrange(
            self.v_norm(rearrange(v, "b h l d -> (b h) l d")),
            "(b h) l d -> b h l d",
            b=B,
        )

        # Apply per-axis RoPE by slicing head_dim according to axis_slices.
        if use_rope:
            q_chunks = _split_last_dim(q, self.axis_slices)
            k_chunks = _split_last_dim(k, self.axis_slices)
            for ax in range(self.n_axes):
                axis_dim = self.axis_slices[ax]
                cos_q, sin_q = build_continuous_rope_cos_sin(
                    pos_q[..., ax], axis_dim=axis_dim, max_freq=self.max_freq, scale=self.scale
                )  # (B, L_q, axis_dim)
                cos_k, sin_k = build_continuous_rope_cos_sin(
                    pos_kv[..., ax], axis_dim=axis_dim, max_freq=self.max_freq, scale=self.scale
                )  # (B, L_kv, axis_dim)
                # Broadcast over the head dimension.
                q_chunks[ax] = apply_rotary_pos_emb(
                    q_chunks[ax], cos_q.unsqueeze(1), sin_q.unsqueeze(1)
                )
                k_chunks[ax] = apply_rotary_pos_emb(
                    k_chunks[ax], cos_k.unsqueeze(1), sin_k.unsqueeze(1)
                )
            q = torch.cat(q_chunks, dim=-1)
            k = torch.cat(k_chunks, dim=-1)

        # Optional KV padding mask: zero out padded K and V, and use the valid
        # count as the linear-attention normalizer.  Matches argent's
        # MultiHeadAttetnion.py:222-228.
        n_valid = float(L_kv)
        if padding_kv is not None:
            # Normalize mask to shape (B, 1, L_kv, 1) so it broadcasts over heads and dim_head.
            mask = padding_kv.reshape(B, 1, L_kv, 1).float()
            k = k * mask
            v = v * mask
            n_valid = mask.sum(dim=-2, keepdim=True)  # (B, 1, 1, 1) — per-batch scalar

        # Linear attention: out = q @ (k^T v) / n_valid.  See rule 3 in the
        # plan's Faithfulness invariants — do NOT replace with softmax.
        kv = torch.matmul(k.transpose(-1, -2), v)  # (B, H, dim_head, dim_head)
        out = torch.matmul(q, kv)  # (B, H, L_q, dim_head)
        if isinstance(n_valid, torch.Tensor):
            # n_valid has shape (B, 1, 1, 1); avoid divide-by-zero on all-padded cases.
            out = out / n_valid.clamp(min=1.0)
        else:
            out = out / max(n_valid, 1.0)

        # (B, H, L_q, D) -> (B, L_q, H*D)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)


class GeGELU(nn.Module):
    r"""GEGLU activation — GELU-gated linear unit ("A. Shazeer, 2020").

    Splits the last dimension in half; the first half is passed through
    :class:`torch.nn.GELU` and gates the second half by elementwise product.
    Matches argent_exp's ``GeGELU`` ([MultiHeadAttetnion.py:237-245]).

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(\ldots, 2C)`. The last dim must be even.

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

    Matches argent_exp's ``FeedForward`` block ([MultiHeadAttetnion.py:247-259])
    used inside each attention layer.

    Parameters
    ----------
    dim : int
        Input/output feature dimension :math:`E`.
    hidden_dim : int
        Hidden dimension of the block. Internally allocates ``2 * hidden_dim``
        so the gated split leaves ``hidden_dim`` post-activation channels.
    dropout : float, optional, default=0.0
        Dropout probability applied after each linear.

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

    Alternative to :class:`FeedForward`, selected by
    :class:`~physicsnemo.experimental.models.argent.ArGEnTCross` when
    ``use_relu_mlp=True``. Matches argent_exp's ``ReLUFeedForward``
    ([MultiHeadAttetnion.py:262-274]).

    Parameters
    ----------
    dim : int
        Input/output feature dimension.
    hidden_dim : int
        Hidden dimension of the block.
    dropout : float, optional, default=0.0
        Dropout probability applied after each linear.

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
    r"""Stack of :class:`RotaryLinearCrossAttentionND` + :class:`FeedForward` blocks.

    Faithful port of argent_exp's ``MultiHeadMultiLayerAttention`` ([ArGEnT.py:85-128]).

    Composes ``depth`` attention layers, each followed by a feed-forward block,
    both wrapped in residual connections. The RoPE scale for layer index
    :math:`d` is ``max_freq / 2**d`` — this per-layer decay is set at
    ArGEnT.py:107 and changes the model, so keep it.

    Parameters
    ----------
    dim_inout : int, optional, default=128
        Input and output feature dimension.
    depth : int, optional, default=4
        Number of attention + feed-forward layers.
    dim_head : int, optional, default=128
        Per-head feature dimension.
    num_heads : int, optional, default=4
        Number of attention heads.
    dim_mlp : int, optional, default=256
        Hidden dimension for the feed-forward blocks.
    axis_slices : list of int, optional
        See :class:`RotaryLinearCrossAttentionND`.
    dropout : float, optional, default=0.0
        Dropout probability inside the feed-forward blocks. The attention op
        does not use dropout.
    max_freq : float, optional, default=64.0
        Base RoPE frequency; layer ``d`` uses ``scale = max_freq / 2**d``.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use :class:`ReLUFeedForward` instead of :class:`FeedForward`
        (GEGLU) for the feed-forward blocks.

    Forward
    -------
    x_q : torch.Tensor
        Query tokens of shape :math:`(B, L_q, E)`.
    x_kv : torch.Tensor, optional
        KV tokens of shape :math:`(B, L_{kv}, E)`. Defaults to ``x_q``
        (self-attention).
    pos_q : torch.Tensor, optional
        Query coordinates of shape :math:`(B, L_q, n_{\text{axes}})`.
    pos_kv : torch.Tensor, optional
        KV coordinates of shape :math:`(B, L_{kv}, n_{\text{axes}})`.
    padding_kv : torch.Tensor, optional
        KV padding mask; see :class:`RotaryLinearCrossAttentionND`.

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
        self.layers = nn.ModuleList()
        for d in range(depth):
            attn = RotaryLinearCrossAttentionND(
                dim_inout=dim_inout,
                dim_head=dim_head,
                num_heads=num_heads,
                axis_slices=axis_slices,
                dropout=dropout,
                max_freq=max_freq,
                # Per-layer RoPE decay — see Faithfulness invariant #2 in the plan.
                scale=max_freq / (2 ** d),
            )
            ffn: nn.Module
            if use_relu_mlp:
                ffn = ReLUFeedForward(dim_inout, dim_mlp, dropout=dropout)
            else:
                ffn = FeedForward(dim_inout, dim_mlp, dropout=dropout)
            self.layers.append(nn.ModuleList([attn, ffn]))

    def forward(
        self,
        x_q: Float[torch.Tensor, "batch len_q dim"],
        x_kv: Optional[Float[torch.Tensor, "batch len_kv dim"]] = None,
        pos_q: Optional[Float[torch.Tensor, "batch len_q axes"]] = None,
        pos_kv: Optional[Float[torch.Tensor, "batch len_kv axes"]] = None,
        padding_kv: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "batch len_q dim"]:
        for attn, ffn in self.layers:
            x_q = attn(x_q, x_kv=x_kv, pos_q=pos_q, pos_kv=pos_kv, padding_kv=padding_kv) + x_q
            x_q = ffn(x_q) + x_q
        return x_q
