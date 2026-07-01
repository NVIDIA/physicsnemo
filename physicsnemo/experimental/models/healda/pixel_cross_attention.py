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

Triton kernels, ``torch.library.custom_op`` registration, and the autotune
cache live in :mod:`._pixel_attn_kernels`, imported lazily by :func:`pixel_attention`.
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


def pixel_attention(
    Q,
    tokens,
    W_k,
    W_v,
    cu_seqlens_k,
    max_seqlen_k,
    n_kv_heads=1,
    scale=None,
    B_k=None,
    B_v=None,
    force_fp32=False,
    group_map=None,
):
    r"""Triton-backed ragged grouped-query attention; see
    :func:`_pixel_attention_reference` for the equivalent pure-PyTorch computation.

    Operates on a packed ragged layout: ``Q`` holds one query per pixel,
    ``tokens`` concatenates every pixel's observation tokens, and
    ``cu_seqlens_k`` gives the prefix sums that carve ``tokens`` into per-pixel
    slices. For each pixel, only that pixel's token slice is projected to
    keys/values and attended over; the kernel streams over tokens with online
    softmax and never materializes a full attention matrix.

    Parameters
    ----------
    Q : torch.Tensor
        Per-pixel queries, shape :math:`(\text{total\_pixels}, n_q\_heads, d\_head)`.
    tokens : torch.Tensor
        Packed observation tokens, shape :math:`(N_{obs}, \text{token\_dim})`.
    W_k, W_v : torch.Tensor
        Key/value projection weights, shape
        :math:`(n_{kv}\_heads \cdot d\_head, \text{token\_dim})`.
    cu_seqlens_k : torch.Tensor
        Int prefix sums of shape :math:`(\text{total\_pixels} + 1,)` delimiting
        each pixel's token slice.
    max_seqlen_k : int
        Longest per-pixel token slice, used to size the kernel's tiling.
    n_kv_heads : int, optional, default=1
        Number of key/value heads. Must be 1, 2, or an even number, and must
        divide ``n_q_heads`` with ``n_q_heads / n_kv_heads >= 16``.
    scale : float, optional, default=None
        Softmax logit scale. Defaults to :math:`1/\sqrt{d\_head}`.
    B_k : torch.Tensor, optional, default=None
        Ignored: a constant per-query shift to every key logit is cancelled
        exactly by softmax, so it is dropped before reaching the kernel.
    B_v : torch.Tensor, optional, default=None
        Value projection bias, shape :math:`(n_{kv}\_heads \cdot d\_head,)`.
    force_fp32 : bool, optional, default=False
        Accumulate attention math in fp32 regardless of input dtype.
    group_map : PixelGroupMap, optional, default=None
        CSR map packing several small pixels into one kernel program. When
        ``None``, every pixel runs as its own program.

    Returns
    -------
    torch.Tensor
        Attention output, shape :math:`(\text{total\_pixels}, n_q\_heads, d\_head)`.

    Notes
    -----
    For ``n_kv_heads <= 2`` this runs one kernel launch over every pixel. For
    larger ``n_kv_heads`` it loops over sequential two-KV-head phases and
    concatenates the outputs; each phase re-reads every token from HBM to
    project its own key/value slice, so runtime scales ~linearly with
    ``n_kv_heads // 2``.
    """
    from . import _pixel_attn_kernels as kernels

    kernels._ensure_autotune_cache()
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    n_q_heads = Q.shape[1]
    if n_kv_heads < 1 or (n_kv_heads > 2 and n_kv_heads % 2 != 0):
        raise ValueError(
            f"pixel_attention requires n_kv_heads=1,2 or an even number, got {n_kv_heads}"
        )
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
        )
    kv_dim = n_kv_heads * Q.shape[-1]
    token_dim = tokens.shape[1]
    if W_k.shape != (kv_dim, token_dim) or W_v.shape != (kv_dim, token_dim):
        raise ValueError(
            f"Expected W_k/W_v shape {(kv_dim, token_dim)}, "
            f"got W_k={tuple(W_k.shape)}, W_v={tuple(W_v.shape)}"
        )
    if B_v is not None and B_v.shape != (kv_dim,):
        raise ValueError(f"Expected B_v shape {(kv_dim,)}, got B_v={tuple(B_v.shape)}")
    # See docstring: K bias is dropped, softmax cancels it exactly.
    B_k = None

    if group_map is None:
        # Kernel expects empty tensors, not None, for the ungrouped path.
        prog_ptr = torch.empty(0, dtype=torch.int32, device=cu_seqlens_k.device)
        prog_pix = torch.empty(0, dtype=torch.int32, device=cu_seqlens_k.device)
    else:
        prog_ptr = group_map.program_ptr
        prog_pix = group_map.program_pixels

    if n_kv_heads <= 2:
        return kernels._pixel_attention_gqa(
            Q,
            tokens,
            W_k,
            W_v,
            B_k,
            B_v,
            cu_seqlens_k,
            prog_ptr,
            prog_pix,
            max_seqlen_k,
            n_kv_heads,
            scale,
            force_fp32=force_fp32,
        )

    # For larger grouped-query layouts, run the same kernel in two-KV-head
    # phases and concatenate the head blocks back in the original order.
    n_phases = n_kv_heads // 2
    q_per_phase = n_q_heads // n_phases
    d_head = Q.shape[-1]
    kv_rows_per_phase = 2 * d_head
    outs = []
    for p in range(n_phases):
        q_slice = Q[:, p * q_per_phase : (p + 1) * q_per_phase]
        wk_slice = W_k[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        wv_slice = W_v[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        bv_slice = (
            None
            if B_v is None
            else B_v[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        )
        outs.append(
            kernels._pixel_attention_gqa(
                q_slice,
                tokens,
                wk_slice,
                wv_slice,
                None,
                bv_slice,
                cu_seqlens_k,
                prog_ptr,
                prog_pix,
                max_seqlen_k,
                2,
                scale,
                force_fp32=force_fp32,
            )
        )
    return torch.cat(outs, dim=1)


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
    r"""Pure-PyTorch equivalent of :func:`pixel_attention`, for small inputs and
    the no-triton path.

    Same tensor contract and shapes as :func:`pixel_attention` (see its
    Parameters), minus the kernel-only arguments: ``B_k`` is dropped (softmax
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
    :func:`pixel_attention` for how the Triton kernel exploits this locality.

    A :class:`..cross_attention.CrossAttentionModuleBase` whose ``context`` is an
    :class:`..obs_context.ObsContext`: it folds the time axis into the batch, runs
    ragged grouped-query attention from each pixel latent to that pixel's token
    slice, and unfolds the result back to :math:`(B, T, X, C)`.

    The tokens in ``context`` must be pre-packed so that each pixel's tokens form
    a single contiguous slice, laid out in pixel order, with ``cu_seqlens_k``
    holding the prefix sums that delimit those slices. Build this packing with
    :func:`..pixel_attention_utils.sort_and_pack` followed by
    :func:`..pixel_attention_utils.counts_to_cu_seqlens`.

    Parameters
    ----------
    token_dim : int
        Channel dimension of the observation tokens (the key/value source).
    n_q_heads : int
        Number of query heads.
    n_kv_heads : int
        Number of key/value heads (grouped-query attention). Must be 1, 2, or an
        even number, divide ``n_q_heads``, with ``n_q_heads / n_kv_heads >= 16``.
    d_head : int
        Per-head channel dimension.
    input_dim : int, optional, default=None
        Latent input dimension. Defaults to ``n_q_heads * d_head``.
    output_dim : int, optional, default=None
        Output dimension. Defaults to ``n_q_heads * d_head``.
    use_proj_bias : bool, optional, default=False
        Add bias to the query/value/output projections (the key projection is
        always bias-free).

    Forward
    -------
    hidden_states : torch.Tensor
        Per-pixel latents of shape :math:`(B, T, X, C)`.
    context : ObsContext
        Packed observation tokens and ragged packing whose ``cu_seqlens_k``
        describes :math:`B \cdot T \cdot X` pixels.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(B, T, X, C)`.

    Notes
    -----
    The kernel runs one program per pixel, so when many pixels hold only a few
    tokens the fixed per-program overhead can be significant. For best throughput,
    precompute a :class:`..obs_context.PixelGroupMap` with
    :func:`..pixel_attention_utils.build_pixel_group_map` to build a map grouping pixels
    with less obs together, and carry it on the ``context``.
    """

    def __init__(
        self,
        token_dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_head: int,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
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
        self.input_dim = self.attn_dim if input_dim is None else input_dim
        self.output_dim = self.attn_dim if output_dim is None else output_dim
        self.token_dim = token_dim
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        kv_dim = n_kv_heads * d_head
        self.q_proj = nn.Linear(self.input_dim, self.attn_dim, bias=use_proj_bias)
        self.k_proj = nn.Linear(token_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, kv_dim, bias=use_proj_bias)
        self.out_proj = nn.Linear(self.attn_dim, self.output_dim, bias=use_proj_bias)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        tokens: torch.Tensor,
        total_pixels: int,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        group_map: Optional[PixelGroupMap] = None,
    ) -> torch.Tensor:
        hidden_flat = hidden_states.reshape(total_pixels, self.input_dim)

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
        b, t, x, ch = hidden_states.shape
        total_pixels = b * t * x
        # Fold (B, T, X) into the flat pixel axis the ragged kernel expects, then
        # unfold the per-pixel output back to the (B, T, X, C) layout.
        out = self._forward_impl(
            hidden_states,
            context.tokens,
            total_pixels,
            context.cu_seqlens_k,
            context.max_seqlen_k,
            group_map=context.group_map,
        )
        return out.view(b, t, x, ch)
