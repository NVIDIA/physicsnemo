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
"""Attention layers for HealDA video DiT blocks.

Provides :class:`TemporalAttention` (temporal self-attention over
:math:`(B, T, X, C)`), :class:`CrossAttentionModuleBase` (pluggable
cross-attention contract), and :class:`PixelCrossAttention` (ragged local
cross-attention from pixels to observation tokens). Triton kernels live in
:mod:`~physicsnemo.experimental.models.healda.kernels.pixel_attention`; packing
utilities live in :mod:`~physicsnemo.experimental.models.healda.obs_context`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.core.version_check import OptionalImport
from physicsnemo.experimental.models.healda.obs_context import PixelGroupMap
from physicsnemo.nn import apply_rotary_pos_emb

triton = OptionalImport("triton")


def mask_causal(
    attn: Float[torch.Tensor, "batch time_q time_k space heads"],
    linear: bool = True,
    window: Optional[int] = None,
) -> Float[torch.Tensor, "batch time_q time_k space heads"]:
    r"""Apply a causal mask to a :math:`(B, T_q, T_k, X, H)` attention tensor.

    Masks out positions where :math:`T_q < T_k` (future frames). Uses zero-fill
    for linear attention and ``-inf``-fill for softmax attention.

    Parameters
    ----------
    attn : torch.Tensor
        Attention logit tensor of shape :math:`(B, T_q, T_k, X, H)`.
    linear : bool, optional, default=True
        If ``True``, fill masked positions with ``0.0``; otherwise fill with
        ``-inf`` (for softmax attention).
    window : int or None, optional, default=None
        When set, additionally restricts each query frame to a lookback of
        ``window`` frames (including itself). ``None`` gives unbounded causal
        attention.

    Returns
    -------
    torch.Tensor
        Masked attention tensor of the same shape :math:`(B, T_q, T_k, X, H)`.
    """
    tq, tk = attn.shape[1], attn.shape[2]
    # Upper-triangular mask: True where t_k > t_q (future frames to mask out).
    mask = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).triu(diagonal=1)
    if window is not None:
        # Lower-triangular mask for frames older than `window` steps.
        too_old = torch.ones(tq, tk, dtype=torch.bool, device=attn.device).tril(
            diagonal=-window
        )
        mask = mask | too_old
    return attn.masked_fill(
        mask.view(1, tq, tk, 1, 1), 0.0 if linear else float("-inf")
    )


class TemporalAttention(torch.nn.Module):
    r"""Temporal self-attention over the time dimension of :math:`(B, T, X, C)` tensors.

    Each spatial location attends independently across the time axis, complementing
    per-frame spatial attention in a factorized video DiT block. Supports rotary
    position embeddings on the time axis, an optional softmax-free (linear)
    attention variant, and causal / sliding-window masking.

    RoPE tables are built once by :class:`~physicsnemo.experimental.models.healda.video_dit.VideoDiT`
    and passed in via ``rope_cos``/``rope_sin``.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension :math:`C`, split evenly across ``num_heads``.
    num_heads : int
        Number of attention heads.
    use_rope : bool, optional, default=True
        Apply rotary position embeddings to queries and keys along the time
        axis. When ``True``, :meth:`forward` requires ``rope_cos``/``rope_sin``.
    is_causal : bool, optional, default=False
        Apply causal masking via :func:`mask_causal`.
    linear_attention : bool, optional, default=True
        If ``True``, skip softmax — attention weights are raw dot-products and
        causal masking uses zero-fill instead of ``-inf``.
    causal_window : int or None, optional, default=None
        When set (and ``is_causal=True``), restricts each frame to attend to
        itself and the previous ``causal_window - 1`` frames only.

    Forward
    -------
    x : torch.Tensor
        Input latents of shape :math:`(B, T, X, C)`.
    rope_cos, rope_sin : torch.Tensor, optional
        1D RoPE tables of shape :math:`(T, C / \text{num\_heads})` from a
        :class:`~physicsnemo.nn.module.rope.RotaryEmbedding1DTables` provider.
        Required when ``use_rope=True``.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(B, T, X, C)`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.healda.attention_layers import TemporalAttention
    >>> layer = TemporalAttention(hidden_size=64, num_heads=4, use_rope=False)
    >>> x = torch.randn(2, 8, 16, 64)
    >>> out = layer(x)
    >>> out.shape
    torch.Size([2, 8, 16, 64])
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        use_rope: bool = True,
        is_causal: bool = False,
        linear_attention: bool = True,
        causal_window: Optional[int] = None,
    ) -> None:
        if causal_window is not None and not is_causal:
            raise ValueError("causal_window was set but is_causal=False")
        super().__init__()
        self.hidden_size = hidden_size
        self._time_parallel_group = None
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.is_causal = is_causal
        self.head_dim = hidden_size // num_heads
        self.linear_attention = linear_attention
        self.causal_window = causal_window

    @torch.compile
    def forward(
        self,
        x: Float[torch.Tensor, "batch time space hidden_size"],
        rope_cos: Optional[Float[torch.Tensor, "time head_dim"]] = None,
        rope_sin: Optional[Float[torch.Tensor, "time head_dim"]] = None,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        r"""Compute temporal self-attention over the time axis.

        Parameters
        ----------
        x : torch.Tensor
            Input latents of shape :math:`(B, T, X, C)`.
        rope_cos, rope_sin : torch.Tensor, optional, default=None
            1D RoPE tables of shape :math:`(T, C / \text{num\_heads})`. Required
            when ``use_rope=True``.

        Returns
        -------
        torch.Tensor
            Output latents of shape :math:`(B, T, X, C)`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 4D input (B, T, X, C), got {x.ndim}D tensor with shape "
                    f"{tuple(x.shape)}"
                )
            if x.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"Expected hidden_size {self.hidden_size}, got "
                    f"{x.shape[-1]} channels"
                )
            if self.use_rope and (rope_cos is None or rope_sin is None):
                raise ValueError(
                    "use_rope=True but rope_cos/rope_sin were not provided"
                )

        # Project to queries, keys, values: (B, T, X, 3*C) -> 3 x (B, T, X, H, C_h)
        qkv = self.qkv(x)
        q, k, v = einops.rearrange(
            qkv,
            "b t x (n heads c) -> n b t x heads c",
            n=3,
            heads=self.num_heads,
        )

        if self.use_rope:
            # apply_rotary_pos_emb broadcasts cos/sin over leading dims; move T
            # next-to-last to align with the (time, head_dim) table layout.
            q = einops.rearrange(q, "b t x h c -> b x h t c")
            k = einops.rearrange(k, "b t x h c -> b x h t c")
            cos = rope_cos.view(1, 1, 1, *rope_cos.shape)
            sin = rope_sin.view(1, 1, 1, *rope_sin.shape)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
            q = einops.rearrange(q, "b x h t c -> b t x h c")
            k = einops.rearrange(k, "b x h t c -> b t x h c")

        attn = torch.einsum(
            "b q x h c, b k x h c -> b q k x h", q, k / math.sqrt(k.shape[-1])
        )

        if self.is_causal:
            attn = mask_causal(
                attn, linear=self.linear_attention, window=self.causal_window
            )
        if not self.linear_attention:
            attn = attn.softmax(2)

        out = einops.einsum(attn, v, "b q k x h, b k x h c -> b q x h c")
        out = einops.rearrange(out, "b t x h c -> b t x (h c)")
        out = self.proj(out)
        return out


# ---------------------------------------------------------------------------
# Cross-attention
#
# :class:`CrossAttentionModuleBase` is the pluggable contract injected into
# :class:`~physicsnemo.experimental.models.healda.video_dit.VideoDiTBlock`.
# :class:`PixelCrossAttention` is the HealDA implementation over packed
# observation tokens.
# ---------------------------------------------------------------------------


class CrossAttentionModuleBase(Module, ABC):
    r"""Abstract base for a cross-attention sub-layer: attends from ``hidden_states``
    to conditioning inputs a subclass defines via ``**cross_attn_kwargs``.

    Forward
    -------
    hidden_states : torch.Tensor
        Latents of shape :math:`(*B, C)`.
    **cross_attn_kwargs : Any
        Subclass-defined conditioning inputs.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(*B, C)`.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "*batch hidden_size"],
        **cross_attn_kwargs: Any,
    ) -> Float[torch.Tensor, "*batch hidden_size"]:
        pass


def _pixel_attention_reference(
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    n_kv_heads: int,
    scale: float,
    B_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Pure-PyTorch equivalent of :func:`~physicsnemo.experimental.models.healda.kernels.pixel_attention.pixel_attention`,
    for small inputs and the no-triton path.

    Same tensor contract and shapes as :func:`~physicsnemo.experimental.models.healda.kernels.pixel_attention.pixel_attention`
    (see its Parameters), minus the kernel-only arguments: ``B_k`` is dropped (softmax
    cancels it) and ``group_map`` does not apply (grouping only changes kernel
    launches, not the result). Loops over pixels and heads in Python, so it is
    far slower than the Triton path and not meant for actual use.

    Returns
    -------
    torch.Tensor
        Attention output of shape :math:`(\text{total\_pixels}, n_q\_heads, d\_head)`.
    """
    n_pixels, n_q_heads, d_head = Q.shape
    q_per_kv = n_q_heads // n_kv_heads
    out = torch.zeros_like(Q)
    for p in range(n_pixels):
        start, end = int(cu_seqlens_k[p]), int(cu_seqlens_k[p + 1])
        if end == start:
            continue
        tok = tokens[start:end]
        K = (tok @ W_k.t()).view(-1, n_kv_heads, d_head)
        V = tok @ W_v.t()
        if B_v is not None:
            V = V + B_v
        V = V.view(-1, n_kv_heads, d_head)
        for h in range(n_q_heads):
            kv = h // q_per_kv
            scores = (K[:, kv] @ Q[p, h]) * scale
            weights = torch.softmax(scores, dim=0)
            out[p, h] = weights @ V[:, kv]
    return out


class PixelCrossAttention(CrossAttentionModuleBase):
    r"""Cross-attention from per-pixel latents to that pixel's own observation tokens.

    A standard q_proj -> attention -> out_proj cross-attention layer using
    grouped-query attention (GQA): ``n_q_heads`` query heads share ``n_kv_heads``
    key/value heads, and the kernel is built for ``n_q_heads >> n_kv_heads``.

    It is specialized for local attention: the number of queries (pixels) is much
    larger than the number of keys/values each one attends to, and every query's
    key/value set is a small, non-overlapping slice of a much larger token pool
    (each observation token is assigned to exactly one pixel). See
    :func:`~physicsnemo.experimental.models.healda.kernels.pixel_attention.pixel_attention` for how the Triton kernel
    exploits this locality.

    A :class:`CrossAttentionModuleBase` taking the observation tokens and
    their ragged packing as keyword arguments: it folds the time axis into
    the batch, runs ragged grouped-query attention from each pixel latent to
    that pixel's token slice, and unfolds the result back to :math:`(B, T, X, C)`.

    ``tokens`` must be pre-packed so that each pixel's tokens form a single
    contiguous slice, laid out in pixel order, with ``cu_seqlens_k`` holding
    the prefix sums that delimit those slices. Build this packing with
    :func:`~physicsnemo.experimental.models.healda.obs_context.prepare_obs_context`.

    Parameters
    ----------
    hidden_size : int
        Residual-stream width; latents enter and leave at this width. The
        internal attention width ``n_q_heads * d_head`` may differ.
    token_dim : int
        Channel dimension of the observation tokens (the key/value source).
    n_q_heads : int
        Number of query heads.
    n_kv_heads : int
        Number of key/value heads (grouped-query attention). Must be 1, 2, or an
        even number, divide ``n_q_heads``, with ``n_q_heads / n_kv_heads >= 16``.
    d_head : int
        Per-head channel dimension.
    use_proj_bias : bool, optional, default=False
        Add bias to the query/value/output projections (the key projection is
        always bias-free).

    Forward
    -------
    hidden_states : torch.Tensor
        Per-pixel latents of shape :math:`(B, T, X, \text{hidden\_size})`.
    tokens : torch.Tensor
        Packed observation tokens of shape :math:`(N_{tok}, \text{token\_dim})`.
    cu_seqlens_k : torch.Tensor
        Prefix sums delimiting each pixel's token slice; length
        :math:`B \cdot T \cdot X + 1`.
    max_seqlen_k : int
        Largest per-pixel token count, for the Triton kernel's tile sizing.
    group_map : :class:`~physicsnemo.experimental.models.healda.obs_context.PixelGroupMap`, optional
        Small-pixel grouping map for the Triton kernel.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(B, T, X, \text{hidden\_size})`.

    Notes
    -----
    The kernel runs one program per pixel, so when many pixels hold only a few
    tokens the fixed per-program overhead can be significant;
    :func:`~physicsnemo.experimental.models.healda.obs_context.prepare_obs_context`'s
    default ``build_group_map=True`` groups them for better throughput.
    On CUDA, set ``HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR`` to a writable directory
    to reuse Triton ``@autotune`` tile configs across repeated runs to reduce startup overhead.
    """

    def __init__(
        self,
        hidden_size: int,
        token_dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        use_proj_bias: bool = False,
    ):
        super().__init__()

        if n_kv_heads < 1 or (n_kv_heads > 2 and n_kv_heads % 2 != 0):
            raise ValueError(
                f"PixelCrossAttention requires n_kv_heads=1,2 or an even number, got {n_kv_heads}"
            )
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )
        q_per_kv = n_q_heads // n_kv_heads
        if q_per_kv < 16:
            raise ValueError(
                f"n_q_heads/n_kv_heads={q_per_kv} < 16, below Triton tl.dot minimum. "
                f"For n_kv_heads={n_kv_heads}, need n_q_heads >= {n_kv_heads * 16}"
            )
        self.attn_dim = n_q_heads * d_head
        self.hidden_size = hidden_size
        self.token_dim = token_dim
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        kv_dim = n_kv_heads * d_head
        self.q_proj = nn.Linear(hidden_size, self.attn_dim, bias=use_proj_bias)
        self.k_proj = nn.Linear(token_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, kv_dim, bias=use_proj_bias)
        self.out_proj = nn.Linear(self.attn_dim, hidden_size, bias=use_proj_bias)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch time space hidden_size"],
        tokens: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        group_map: Optional[PixelGroupMap] = None,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        b, t, x, _ = hidden_states.shape
        total_pixels = b * t * x
        if not torch.compiler.is_compiling():
            if tokens is None:
                raise ValueError(
                    "tokens must be set before PixelCrossAttention forward"
                )
            if hidden_states.ndim != 4:
                raise ValueError(
                    f"Expected hidden_states of shape (B, T, X, C), got {hidden_states.ndim}D "
                    f"tensor with shape {tuple(hidden_states.shape)}"
                )
            if hidden_states.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"Expected hidden_size {self.hidden_size}, got "
                    f"{hidden_states.shape[-1]} channels"
                )
            if cu_seqlens_k.numel() != total_pixels + 1:
                raise ValueError(
                    f"Expected cu_seqlens_k length {total_pixels + 1} (B*T*X+1), got "
                    f"{cu_seqlens_k.numel()}"
                )
            if int(cu_seqlens_k[-1]) != tokens.shape[0]:
                raise ValueError(
                    f"Expected {tokens.shape[0]} packed tokens, but "
                    f"cu_seqlens_k ends at {int(cu_seqlens_k[-1])}"
                )

        # Fold (B, T, X) into the flat pixel axis the ragged kernel expects.
        hidden_flat = hidden_states.reshape(total_pixels, self.hidden_size)

        if tokens.shape[0] == 0:
            # Keep every projection parameter in the graph even when a batch has
            # no observations, so empty groups still produce gradients (prevents issues with DDP).
            token_dummy = tokens.sum() * 0
            q_dummy = self.q_proj.weight.sum() * 0
            if self.q_proj.bias is not None:
                q_dummy = q_dummy + self.q_proj.bias.sum() * 0
            kv_dummy = self.k_proj.weight.sum() * 0 + self.v_proj.weight.sum() * 0
            if self.v_proj.bias is not None:
                kv_dummy = kv_dummy + self.v_proj.bias.sum() * 0
            out = self.out_proj(hidden_flat.new_zeros((total_pixels, self.attn_dim)))
            out = out + token_dummy + q_dummy + kv_dummy
            return out.view(b, t, x, self.hidden_size)

        hidden_flat = self.q_proj(hidden_flat)
        Q = hidden_flat.view(total_pixels, self.n_q_heads, self.d_head).contiguous()

        if triton.available and Q.is_cuda:
            from .kernels.pixel_attention import pixel_attention

            attn_out = pixel_attention(
                Q,
                tokens,
                self.k_proj.weight,
                self.v_proj.weight,
                cu_seqlens_k,
                max_seqlen_k,
                n_kv_heads=self.n_kv_heads,
                scale=self.scale,
                B_k=self.k_proj.bias,
                B_v=self.v_proj.bias,
                group_map=group_map,
            )
        else:
            attn_out = _pixel_attention_reference(
                Q,
                tokens,
                self.k_proj.weight,
                self.v_proj.weight,
                cu_seqlens_k,
                self.n_kv_heads,
                self.scale,
                B_v=self.v_proj.bias,
            )

        # Unfold the per-pixel output back to the (B, T, X, hidden_size) layout.
        out = self.out_proj(attn_out.reshape(total_pixels, self.attn_dim))
        return out.view(b, t, x, self.hidden_size)
