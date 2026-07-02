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
"""Ragged local cross-attention: each pixel attends only to its own slice of
observation tokens.

Triton kernels and dispatch (:func:`~physicsnemo.experimental.models.healda.kernels.pixel_attention.pixel_attention`)
live in :mod:`~physicsnemo.experimental.models.healda.kernels.pixel_attention`; the pure-PyTorch fallback
(:func:`_pixel_attention_reference`) lives here, since it has no triton
dependency. Packing utilities that build the ragged layout
(:func:`~physicsnemo.experimental.models.healda.obs_context.sort_and_pack`, :func:`~physicsnemo.experimental.models.healda.obs_context.counts_to_cu_seqlens`,
:func:`~physicsnemo.experimental.models.healda.obs_context.build_pixel_group_map`) live in :mod:`~physicsnemo.experimental.models.healda.obs_context`.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.experimental.models.healda.cross_attention import (
    CrossAttentionModuleBase,
)
from physicsnemo.experimental.models.healda.obs_context import (
    ObsContext,
    PixelGroupMap,
)

triton = OptionalImport("triton")


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

    A :class:`~physicsnemo.experimental.models.healda.cross_attention.CrossAttentionModuleBase` whose ``context`` is an
    :class:`~physicsnemo.experimental.models.healda.obs_context.ObsContext`: it folds the time axis into the batch, runs
    ragged grouped-query attention from each pixel latent to that pixel's token
    slice, and unfolds the result back to :math:`(B, T, X, C)`.

    The tokens in ``context`` must be pre-packed so that each pixel's tokens form
    a single contiguous slice, laid out in pixel order, with ``cu_seqlens_k``
    holding the prefix sums that delimit those slices. Build this packing with
    :func:`~physicsnemo.experimental.models.healda.obs_context.sort_and_pack` followed by
    :func:`~physicsnemo.experimental.models.healda.obs_context.counts_to_cu_seqlens`.

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
    context : :class:`~physicsnemo.experimental.models.healda.obs_context.ObsContext`
        Packed observation tokens and ragged packing whose ``cu_seqlens_k``
        describes :math:`B \cdot T \cdot X` pixels.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(B, T, X, \text{hidden\_size})`.

    Notes
    -----
    The kernel runs one program per pixel, so when many pixels hold only a few
    tokens the fixed per-program overhead can be significant. For best throughput,
    precompute a :class:`~physicsnemo.experimental.models.healda.obs_context.PixelGroupMap` with
    :func:`~physicsnemo.experimental.models.healda.obs_context.build_pixel_group_map` to build a map grouping pixels
    with less obs together, and carry it on the ``context``.
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

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        tokens: torch.Tensor,
        total_pixels: int,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        group_map: Optional[PixelGroupMap] = None,
    ) -> torch.Tensor:
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
            return out + token_dummy + q_dummy + kv_dummy

        hidden_flat = self.q_proj(hidden_flat)
        Q = hidden_flat.view(total_pixels, self.n_q_heads, self.d_head)
        Q = Q.contiguous()

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

        return self.out_proj(attn_out.reshape(total_pixels, self.attn_dim))

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch time space hidden_size"],
        context: ObsContext,
    ) -> Float[torch.Tensor, "batch time space hidden_size"]:
        b, t, x, _ = hidden_states.shape
        total_pixels = b * t * x
        if not torch.compiler.is_compiling():
            if context.tokens is None:
                raise ValueError(
                    "ObsContext.tokens must be set before PixelCrossAttention forward"
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
            if context.cu_seqlens_k.numel() != total_pixels + 1:
                raise ValueError(
                    f"Expected cu_seqlens_k length {total_pixels + 1} (B*T*X+1), got "
                    f"{context.cu_seqlens_k.numel()}"
                )
            if int(context.cu_seqlens_k[-1]) != context.tokens.shape[0]:
                raise ValueError(
                    f"Expected {context.tokens.shape[0]} packed tokens, but "
                    f"cu_seqlens_k ends at {int(context.cu_seqlens_k[-1])}"
                )
        # Fold (B, T, X) into the flat pixel axis the ragged kernel expects, then
        # unfold the per-pixel output back to the (B, T, X, hidden_size) layout.
        out = self._forward_impl(
            hidden_states,
            context.tokens,
            total_pixels,
            context.cu_seqlens_k,
            context.max_seqlen_k,
            group_map=context.group_map,
        )
        return out.view(b, t, x, self.hidden_size)
