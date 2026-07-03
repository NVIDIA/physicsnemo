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

"""Triton kernels for pixel/observation cross-attention.

Importing this module requires triton (``tl = triton.language`` runs eagerly
below); callers must guard the import with ``triton.available``. The
pure-PyTorch fallback has no triton dependency and lives in
:mod:`~physicsnemo.experimental.models.healda.attention_layers` instead. Contains:

1. The ``@triton.jit`` grouped-query-attention forward/backward kernels.
2. ``torch.library.custom_op`` wrappers with fake-tensor and autograd
   registration.
3. An opt-in autotune config cache (see ``.. important::`` below).
4. :func:`pixel_attention`, the GQA dispatch entry point
   :class:`~physicsnemo.experimental.models.healda.attention_layers.PixelCrossAttention` calls.
5. The ``@triton.jit`` counting-sort kernel and :func:`counting_sort_and_pack`,
   the entry point :func:`~physicsnemo.experimental.models.healda.obs_context.sort_and_pack` calls.

.. important::
   On CUDA, the first call per shape bucket runs Triton ``@autotune`` to pick the
   best tile config (warps, stages, block sizes). That is not the compiled
   kernel — Triton caches those separately via ``TRITON_CACHE_DIR``. Variable obs
   counts can hit new buckets mid-training and stall all ranks while one rank
   benchmarks; set ``HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR`` to a directory to
   persist the winning config across processes/runs so known buckets skip the
   sweep. Unset (default) means no disk read/write.
"""

import hashlib
import math
import os
from typing import Tuple

import torch
import torch.distributed as dist
from jaxtyping import Int

from physicsnemo.core.version_check import OptionalImport

from . import autotune_cache as tac

triton = OptionalImport("triton")
tl = triton.language

# Base-2 softmax: exp(x) == exp2(x * log2e), log(x) == log2(x) / log2e.
# Folding log2e into the score scale lets the kernels use the faster MUFU
# exp2/log2 hardware instructions. The LSE is stored in the log2 domain so the
# forward (producer) and backward (consumer) agree on the convention.
LOG2E = tl.constexpr(1.4426950408889634)


def _next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


_MAX_SEQLEN_K_BUCKETS = (64, 256, 1024, 4096)
_MAX_SEQLEN_K_BUCKET_OVERFLOW = 8192


def _bucket_max_seqlen_k(max_seqlen_k: int) -> int:
    # Bucket raw sequence lengths so nearby shapes share the same autotune cache.
    for upper_bound in _MAX_SEQLEN_K_BUCKETS:
        if max_seqlen_k <= upper_bound:
            return upper_bound
    return _MAX_SEQLEN_K_BUCKET_OVERFLOW


def _autotune_configs():
    return [
        triton.Config({"TILE_K": tk}, num_warps=nw, num_stages=ns)
        for tk in [32, 64, 128]
        for nw in [1, 2, 4, 8]
        for ns in [1, 2, 4]
    ]


# ─── Grouped-query fused kernels ─────────────────────────────────────
# One program per pixel group, all KV heads processed together.
# Handles n_kv_heads={1,2,4} via constexpr branching with explicit
# per-head accumulators. Tokens loaded ONCE per tile, d_tokens uses
# plain tl.store (no per-head atomic contention).
#
# Wk and Wv passed as separate pointers


@triton.jit
def _gqa_fwd_head(
    tokens_tile,
    wk_h,
    wv_h,
    bv_h,
    q_h,
    kv_mask,
    scale,
    m_h,
    l_h,
    acc_h,
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Head selection already happened in the caller: q_h holds the Q_PER_KV
    # query heads assigned to one KV head, and wk_h/wv_h/bv_h are that KV
    # head's projection parameters.
    k_h = tl.dot(tokens_tile, tl.trans(wk_h)).to(COMPUTE_DTYPE)
    v_h = tl.dot(tokens_tile, tl.trans(wv_h)).to(COMPUTE_DTYPE)
    if USE_V_BIAS:
        v_h = v_h + bv_h[None, :]

    # Fold log2e into the score scale so scores live in the log2 domain and the
    # softmax can use exp2 (faster MUFU instruction) instead of exp.
    scores = tl.dot(q_h.to(COMPUTE_DTYPE), tl.trans(k_h)).to(tl.float32) * (
        scale * LOG2E
    )
    scores = tl.where(kv_mask[None, :], scores, float("-inf"))

    # Online softmax over KV tiles (as in FlashAttention). We keep a running max (m_h), denominator
    # (l_h), and weighted value sum (acc_h) for each query row so we never
    # have to materialize the full attention matrix across all keys.
    m_tile = tl.max(scores, axis=1)
    m_new = tl.maximum(m_h, m_tile)
    # If this tile raises the running max, rescale the previous partial sums
    # into the new log-sum-exp coordinate system before adding this tile.
    corr = tl.exp2(m_h - m_new)
    exp_s = tl.exp2(scores - m_new[:, None])

    l_h = l_h * corr + tl.sum(exp_s, axis=1)
    acc_h = acc_h * corr[:, None] + tl.dot(exp_s.to(COMPUTE_DTYPE), v_h).to(tl.float32)
    m_h = m_new
    return m_h, l_h, acc_h


@triton.jit
def _gqa_bwd_head(
    tokens_tile,
    wk_h,
    wv_h,
    bv_h,
    q_h,
    dout_h,
    D_h,
    lse_h,
    kv_mask,
    scale,
    dq_h,
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOKEN_DIM: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # HYBRID unfuse: keep the cheap K/V recompute + dtokens IN-kernel (read
    # tokens once, project in registers per tile) but DROP the loop-carried
    # [32,32] fp32 weight-grad accumulators (dWk/dWv/dBv) that pinned 255 regs /
    # 28M spills. Instead this returns the per-tile dk/dv, which the caller
    # stores; dWk/dWv/dBv are recovered as dense GEMMs after the kernel. This
    # keeps the fused kernel's *minimal* HBM footprint (no K/V materialization)
    # while removing the spill source -> far less extra traffic than full unfuse.
    k_h = tl.dot(tokens_tile, tl.trans(wk_h)).to(COMPUTE_DTYPE)
    v_h = tl.dot(tokens_tile, tl.trans(wv_h)).to(COMPUTE_DTYPE)
    if USE_V_BIAS:
        v_h = v_h + bv_h[None, :]

    scores = tl.dot(q_h.to(COMPUTE_DTYPE), tl.trans(k_h)).to(tl.float32) * (
        scale * LOG2E
    )
    scores = tl.where(kv_mask[None, :], scores, float("-inf"))
    weights = tl.exp2(scores - lse_h[:, None])

    # D_h = rowsum(dO * O) is the FA "delta"; dout/out are tile-invariant, so the
    # caller computes it ONCE per program and passes it in (not recomputed per tile).
    dv_tile = tl.dot(tl.trans(weights.to(COMPUTE_DTYPE)), dout_h.to(COMPUTE_DTYPE))
    pt = tl.dot(dout_h.to(tl.float32), tl.trans(v_h.to(tl.float32)))
    # ds is the gradient w.r.t. the raw logits q·k (natural score scale).
    ds = weights * (pt - D_h[:, None]) * scale
    dk_tile = tl.dot(tl.trans(ds.to(COMPUTE_DTYPE)), q_h.to(COMPUTE_DTYPE))
    dq_h += tl.dot(ds.to(COMPUTE_DTYPE), k_h.to(COMPUTE_DTYPE)).to(tl.float32)

    dk_cast = dk_tile.to(COMPUTE_DTYPE)
    dv_cast = dv_tile.to(COMPUTE_DTYPE)

    d_tok = tl.dot(dk_cast, wk_h) + tl.dot(dv_cast, wv_h)

    return dq_h, d_tok, dk_cast, dv_cast


# ─── Unified GQA kernel: n_kv_heads={1,2} via constexpr branching ───


@triton.autotune(
    configs=_autotune_configs(),
    key=[
        "Q_PER_KV",
        "N_KV_HEADS",
        "COMPUTE_DTYPE",
        "max_seqlen_k_bucket",
        "n_pix",
        "GROUPED",
    ],
)
@triton.jit
def _pixel_attn_gqa_fwd(
    Q_ptr,
    Tokens_ptr,
    Wk_ptr,
    Wv_ptr,
    Bk_ptr,
    Bv_ptr,
    Out_ptr,
    LSE_ptr,
    cu_seqlens_ptr,
    ProgPtr_ptr,
    ProgPix_ptr,
    scale,
    max_seqlen_k_bucket,
    n_pix,  # autotune-key only: grid size (T1 vs T2 want different configs)
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOKEN_DIM: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GROUPED: tl.constexpr,
):
    # CSR program map: program p handles pixels prog_pix[prog_ptr[p]:prog_ptr[p+1]]
    # (1 = ungrouped, 2 = paired small pixels). Output/LSE stay pixel-id indexed, so the
    # layout is identical to the ungrouped kernel. Weights are loaded ONCE here and
    # shared across the program's pixels -- the amortization that makes pairing win.
    # GROUPED=False (no map given) -> program p IS pixel p; skip the map loads so the
    # default/ungrouped path has no CSR overhead vs the pre-grouping kernel.
    prog = tl.program_id(0)
    if GROUPED:
        start = tl.load(ProgPtr_ptr + prog).to(tl.int64)
        end = tl.load(ProgPtr_ptr + prog + 1).to(tl.int64)
    else:
        start = prog.to(tl.int64)
        end = start + 1

    N_Q: tl.constexpr = N_KV_HEADS * Q_PER_KV
    offs_qh = tl.arange(0, BLOCK_Q)
    qh_mask = offs_qh < Q_PER_KV
    offs_d = tl.arange(0, D_HEAD)
    offs_td = tl.arange(0, TOKEN_DIM)
    # Wk/Wv are stored per KV head, so wk0/wv0 are the parameters for KV head 0.
    wk0 = tl.load(
        Wk_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    wv0 = tl.load(
        Wv_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    if USE_V_BIAS:
        bv0 = tl.load(Bv_ptr + 0 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0).to(
            COMPUTE_DTYPE
        )
    else:
        bv0 = tl.zeros((D_HEAD,), dtype=COMPUTE_DTYPE)
    if N_KV_HEADS >= 2:
        wk1 = tl.load(
            Wk_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        wv1 = tl.load(
            Wv_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        if USE_V_BIAS:
            bv1 = tl.load(
                Bv_ptr + 1 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0
            ).to(COMPUTE_DTYPE)
        else:
            bv1 = tl.zeros((D_HEAD,), dtype=COMPUTE_DTYPE)
    else:
        wk1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        wv1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        bv1 = tl.zeros((D_HEAD,), dtype=COMPUTE_DTYPE)

    # Per-pixel forward is inlined so weights are loaded once per program and then
    # shared by the pixels in the CSR group.
    for i in range(start, end):
        pix = tl.load(ProgPix_ptr + i).to(tl.int64) if GROUPED else i
        kv_start = tl.load(cu_seqlens_ptr + pix).to(tl.int64)
        kv_end = tl.load(cu_seqlens_ptr + pix + 1).to(tl.int64)
        seqlen_k = kv_end - kv_start
        if seqlen_k > 0:
            q_base = pix * N_Q * D_HEAD
            # Queries are laid out as [kv_head_0's Q_PER_KV queries][kv_head_1's
            # Q_PER_KV queries]... . q0 selects the first query-head group.
            q0 = tl.load(
                Q_ptr
                + q_base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            m0 = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
            l0 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
            acc0 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)
            if N_KV_HEADS >= 2:
                q1 = tl.load(
                    Q_ptr
                    + q_base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                m1 = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
                l1 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
                acc1 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)

            for tile_off in range(0, seqlen_k, TILE_K):
                offs_kv = tl.arange(0, TILE_K)
                kv_mask = offs_kv < (seqlen_k - tile_off)
                tok_base = (kv_start + tile_off) * TOKEN_DIM
                tokens_tile = tl.load(
                    Tokens_ptr
                    + tok_base
                    + offs_kv[:, None] * TOKEN_DIM
                    + offs_td[None, :],
                    mask=kv_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                m0, l0, acc0 = _gqa_fwd_head(
                    tokens_tile,
                    wk0,
                    wv0,
                    bv0,
                    q0,
                    kv_mask,
                    scale,
                    m0,
                    l0,
                    acc0,
                    USE_V_BIAS,
                    Q_PER_KV,
                    D_HEAD,
                    TILE_K,
                    COMPUTE_DTYPE,
                )
                if N_KV_HEADS >= 2:
                    m1, l1, acc1 = _gqa_fwd_head(
                        tokens_tile,
                        wk1,
                        wv1,
                        bv1,
                        q1,
                        kv_mask,
                        scale,
                        m1,
                        l1,
                        acc1,
                        USE_V_BIAS,
                        Q_PER_KV,
                        D_HEAD,
                        TILE_K,
                        COMPUTE_DTYPE,
                    )

            out_base = pix * N_Q * D_HEAD
            lse_base = pix * N_Q
            tl.store(
                Out_ptr
                + out_base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                acc0 / l0[:, None],
                mask=qh_mask[:, None],
            )
            # LSE in the log2 domain: m0 is the log2-scaled running max and
            # log2(l0) keeps the denominator in the same domain.
            tl.store(
                LSE_ptr + lse_base + 0 * Q_PER_KV + offs_qh,
                m0 + tl.log2(l0),
                mask=qh_mask,
            )
            if N_KV_HEADS >= 2:
                tl.store(
                    Out_ptr
                    + out_base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    acc1 / l1[:, None],
                    mask=qh_mask[:, None],
                )
                tl.store(
                    LSE_ptr + lse_base + 1 * Q_PER_KV + offs_qh,
                    m1 + tl.log2(l1),
                    mask=qh_mask,
                )


@triton.autotune(
    configs=_autotune_configs(),
    key=[
        "Q_PER_KV",
        "N_KV_HEADS",
        "COMPUTE_DTYPE",
        "max_seqlen_k_bucket",
        "n_pix",
        "GROUPED",
    ],
)
@triton.jit
def _pixel_attn_gqa_bwd(
    Q_ptr,
    Tokens_ptr,
    Wk_ptr,
    Wv_ptr,
    Bk_ptr,
    Bv_ptr,
    Out_ptr,
    LSE_ptr,
    dOut_ptr,
    dQ_ptr,
    dTokens_ptr,
    dKV_ptr,  # combined [dK | dV] rows, stride 2 * KV_DIM
    cu_seqlens_ptr,
    ProgPtr_ptr,
    ProgPix_ptr,
    scale,
    max_seqlen_k_bucket,
    n_pix,  # autotune-key only: grid size (T1 vs T2 want different configs)
    USE_V_BIAS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOKEN_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    TILE_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    GROUPED: tl.constexpr,
):
    # CSR program map (see forward kernel). Weights loaded once per program and
    # shared across its 1-2 pixels; dK/dV/dTokens are written per global token row
    # so the pixel-id indirection never reorders any output. GROUPED=False -> program
    # p is pixel p (no map loads; same cost as the pre-grouping kernel).
    prog = tl.program_id(0)
    if GROUPED:
        start = tl.load(ProgPtr_ptr + prog).to(tl.int64)
        end = tl.load(ProgPtr_ptr + prog + 1).to(tl.int64)
    else:
        start = prog.to(tl.int64)
        end = start + 1

    N_Q: tl.constexpr = N_KV_HEADS * Q_PER_KV
    offs_qh = tl.arange(0, BLOCK_Q)
    qh_mask = offs_qh < Q_PER_KV
    offs_d = tl.arange(0, D_HEAD)
    offs_td = tl.arange(0, TOKEN_DIM)
    wk0 = tl.load(
        Wk_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    wv0 = tl.load(
        Wv_ptr + 0 * D_HEAD * TOKEN_DIM + offs_d[:, None] * TOKEN_DIM + offs_td[None, :]
    ).to(COMPUTE_DTYPE)
    if USE_V_BIAS:
        bv0 = tl.load(Bv_ptr + 0 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0).to(
            COMPUTE_DTYPE
        )
    else:
        bv0 = tl.zeros((D_HEAD,), dtype=COMPUTE_DTYPE)
    if N_KV_HEADS >= 2:
        wk1 = tl.load(
            Wk_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        wv1 = tl.load(
            Wv_ptr
            + 1 * D_HEAD * TOKEN_DIM
            + offs_d[:, None] * TOKEN_DIM
            + offs_td[None, :]
        ).to(COMPUTE_DTYPE)
        if USE_V_BIAS:
            bv1 = tl.load(
                Bv_ptr + 1 * D_HEAD + offs_d, mask=offs_d < D_HEAD, other=0.0
            ).to(COMPUTE_DTYPE)
        else:
            bv1 = tl.zeros((D_HEAD,), dtype=COMPUTE_DTYPE)
    else:
        wk1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        wv1 = tl.zeros((D_HEAD, TOKEN_DIM), dtype=COMPUTE_DTYPE)
        bv1 = tl.zeros((D_HEAD,), dtype=COMPUTE_DTYPE)

    # Per-pixel backward is inlined so weights are amortized across the CSR group.
    for i in range(start, end):
        pix = tl.load(ProgPix_ptr + i).to(tl.int64) if GROUPED else i
        kv_start = tl.load(cu_seqlens_ptr + pix).to(tl.int64)
        kv_end = tl.load(cu_seqlens_ptr + pix + 1).to(tl.int64)
        seqlen_k = kv_end - kv_start
        if seqlen_k > 0:
            base = pix * N_Q * D_HEAD
            lse_b = pix * N_Q
            q0 = tl.load(
                Q_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            dout0 = tl.load(
                dOut_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            out0 = tl.load(
                Out_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                mask=qh_mask[:, None],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            lse0 = tl.load(
                LSE_ptr + lse_b + 0 * Q_PER_KV + offs_qh, mask=qh_mask, other=0.0
            )
            dq0 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)
            D0 = tl.sum(dout0.to(tl.float32) * out0.to(tl.float32), axis=1)

            if N_KV_HEADS >= 2:
                q1 = tl.load(
                    Q_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                dout1 = tl.load(
                    dOut_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                out1 = tl.load(
                    Out_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    mask=qh_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)
                lse1 = tl.load(
                    LSE_ptr + lse_b + 1 * Q_PER_KV + offs_qh, mask=qh_mask, other=0.0
                )
                dq1 = tl.zeros((BLOCK_Q, D_HEAD), dtype=tl.float32)
                D1 = tl.sum(dout1.to(tl.float32) * out1.to(tl.float32), axis=1)

            for tile_off in range(0, seqlen_k, TILE_K):
                offs_kv = tl.arange(0, TILE_K)
                kv_mask = offs_kv < (seqlen_k - tile_off)
                tok_base = (kv_start + tile_off) * TOKEN_DIM
                tokens_tile = tl.load(
                    Tokens_ptr
                    + tok_base
                    + offs_kv[:, None] * TOKEN_DIM
                    + offs_td[None, :],
                    mask=kv_mask[:, None],
                    other=0.0,
                ).to(COMPUTE_DTYPE)

                # dKV holds combined [dK | dV] rows, so the row stride is 2 * KV_DIM.
                kv_row_off = (
                    (kv_start + tile_off) * (2 * KV_DIM)
                    + offs_kv[:, None] * (2 * KV_DIM)
                    + offs_d[None, :]
                )
                dq0, dt0, dk0, dv0 = _gqa_bwd_head(
                    tokens_tile,
                    wk0,
                    wv0,
                    bv0,
                    q0,
                    dout0,
                    D0,
                    lse0,
                    kv_mask,
                    scale,
                    dq0,
                    USE_V_BIAS,
                    Q_PER_KV,
                    D_HEAD,
                    TOKEN_DIM,
                    TILE_K,
                    COMPUTE_DTYPE,
                )
                d_tok_sum = dt0
                tl.store(dKV_ptr + kv_row_off + 0 * D_HEAD, dk0, mask=kv_mask[:, None])
                tl.store(
                    dKV_ptr + kv_row_off + KV_DIM + 0 * D_HEAD,
                    dv0,
                    mask=kv_mask[:, None],
                )
                if N_KV_HEADS >= 2:
                    dq1, dt1, dk1, dv1 = _gqa_bwd_head(
                        tokens_tile,
                        wk1,
                        wv1,
                        bv1,
                        q1,
                        dout1,
                        D1,
                        lse1,
                        kv_mask,
                        scale,
                        dq1,
                        USE_V_BIAS,
                        Q_PER_KV,
                        D_HEAD,
                        TOKEN_DIM,
                        TILE_K,
                        COMPUTE_DTYPE,
                    )
                    d_tok_sum += dt1
                    tl.store(
                        dKV_ptr + kv_row_off + 1 * D_HEAD,
                        dk1,
                        mask=kv_mask[:, None],
                    )
                    tl.store(
                        dKV_ptr + kv_row_off + KV_DIM + 1 * D_HEAD,
                        dv1,
                        mask=kv_mask[:, None],
                    )

                tl.store(
                    dTokens_ptr
                    + tok_base
                    + offs_kv[:, None] * TOKEN_DIM
                    + offs_td[None, :],
                    d_tok_sum,
                    mask=kv_mask[:, None],
                )

            tl.store(
                dQ_ptr
                + base
                + 0 * Q_PER_KV * D_HEAD
                + offs_qh[:, None] * D_HEAD
                + offs_d[None, :],
                dq0,
                mask=qh_mask[:, None],
            )
            if N_KV_HEADS >= 2:
                tl.store(
                    dQ_ptr
                    + base
                    + 1 * Q_PER_KV * D_HEAD
                    + offs_qh[:, None] * D_HEAD
                    + offs_d[None, :],
                    dq1,
                    mask=qh_mask[:, None],
                )


# ─── Custom op registration ──────────────────────────────────────────
# Wrap the Triton launches as torch custom ops so autograd and fake-tensor
# tracing see a single op.


def _gqa_fwd_impl(
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32=False,
):
    n_groups = cu_seqlens_k.shape[0] - 1
    # Empty CSR map => ungrouped: one program per pixel, kernel derives pixel =
    # program_id (GROUPED=False) and skips the per-program map loads.
    grouped = prog_pix.numel() > 0
    n_programs = (prog_ptr.shape[0] - 1) if grouped else n_groups
    n_q_heads = Q.shape[1]
    d_head = Q.shape[2]
    block_q = max(16, _next_power_of_2(q_per_kv))
    max_seqlen_k_bucket = _bucket_max_seqlen_k(int(max_seqlen_k))
    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    # The Triton kernels below use flat pointer math for packed [group, head, d]
    # storage and do not take explicit tensor strides. Multi-phase q/head slices
    # are views with the original group stride, so materialize packed inputs here.
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_k = B_k.contiguous()
    B_v = B_v.contiguous()
    Out = torch.zeros_like(Q)
    LSE = torch.empty(n_groups, n_q_heads, device=Q.device, dtype=torch.float32)

    _pixel_attn_gqa_fwd[(n_programs,)](
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k_bucket,
        n_groups,
        USE_V_BIAS=use_v_bias,
        Q_PER_KV=q_per_kv,
        BLOCK_Q=block_q,
        N_KV_HEADS=n_kv_heads,
        D_HEAD=d_head,
        TOKEN_DIM=token_dim,
        COMPUTE_DTYPE=compute_dtype,
        GROUPED=grouped,
    )
    return Out, LSE


def _gqa_bwd_impl(
    dOut,
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    Out,
    LSE,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32=False,
):
    n_groups = cu_seqlens_k.shape[0] - 1
    grouped = prog_pix.numel() > 0
    n_programs = (prog_ptr.shape[0] - 1) if grouped else n_groups
    d_head = Q.shape[2]
    kv_dim = n_kv_heads * d_head
    block_q = max(16, _next_power_of_2(q_per_kv))
    max_seqlen_k_bucket = _bucket_max_seqlen_k(int(max_seqlen_k))
    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    torch_compute_dtype = torch.float32 if force_fp32 else torch.bfloat16
    # Backward sees the original saved inputs from the custom op; for multi-phase
    # q/head slicing those can be non-contiguous views, which breaks the kernel's
    # flat indexing unless we repack them first.
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_k = B_k.contiguous()
    B_v = B_v.contiguous()
    Out = Out.contiguous()
    LSE = LSE.contiguous()
    dOut = dOut.contiguous()
    dQ = torch.zeros_like(Q)
    d_tokens = torch.zeros_like(tokens)
    # HYBRID: kernel keeps in-kernel K/V recompute + in-kernel dtokens, but emits
    # per-token [dK | dV] rows so the weight grads are recovered with one dense
    # GEMM instead of two. Every token is written by exactly one non-empty pixel.
    dKV = torch.empty(
        tokens.shape[0], 2 * kv_dim, device=Q.device, dtype=torch_compute_dtype
    )

    _pixel_attn_gqa_bwd[(n_programs,)](
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        dOut,
        dQ,
        d_tokens,
        dKV,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k_bucket,
        n_groups,
        USE_V_BIAS=use_v_bias,
        Q_PER_KV=q_per_kv,
        BLOCK_Q=block_q,
        N_KV_HEADS=n_kv_heads,
        D_HEAD=d_head,
        TOKEN_DIM=token_dim,
        KV_DIM=kv_dim,
        COMPUTE_DTYPE=compute_dtype,
        GROUPED=grouped,
    )
    # Recover weight grads as one dense cuBLAS GEMM:
    # dKV rows are [dK | dV], so the result rows split back into [dW_k | dW_v].
    tokens_compute = tokens.to(torch_compute_dtype)
    dW_kv = (dKV.t() @ tokens_compute).to(torch.float32)
    dW_k = dW_kv[:kv_dim].clone()
    dW_v = dW_kv[kv_dim:].clone()
    if use_v_bias:
        # Accumulate in fp32 directly; do NOT materialize an fp32 copy of dV
        # (millions of rows) before reducing -- that HBM pass dominated the bwd.
        dB_v = dKV[:, kv_dim:].sum(dim=0, dtype=torch.float32)
    else:
        dB_v = torch.zeros(kv_dim, device=Q.device, dtype=torch.float32)
    dB_k = torch.zeros_like(B_k)
    dW_k = dW_k if W_k.dtype == dW_k.dtype else dW_k.to(W_k.dtype)
    dW_v = dW_v if W_v.dtype == dW_v.dtype else dW_v.to(W_v.dtype)
    if B_v.dtype != dB_v.dtype:
        dB_v = dB_v.to(B_v.dtype)
    return dQ, d_tokens, dW_k, dW_v, dB_k, dB_v


@torch.library.custom_op("healda::pixel_attn_fwd", mutates_args=())
def pixel_attn_fwd(
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_k: torch.Tensor,
    B_v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    prog_ptr: torch.Tensor,
    prog_pix: torch.Tensor,
    scale: float,
    max_seqlen_k: int,
    q_per_kv: int,
    token_dim: int,
    n_kv_heads: int,
    use_v_bias: bool,
    force_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_impl(
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )


@pixel_attn_fwd.register_fake
def _fake_fwd(
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32,
):
    # Fake registrations mirror output metadata so torch.compile/export can trace
    # through the custom op without running the Triton kernel.
    n_groups, n_q_heads, d_head = Q.shape
    return Q.new_empty((n_groups, n_q_heads, d_head)), Q.new_empty(
        (n_groups, n_q_heads), dtype=torch.float32
    )


@torch.library.custom_op("healda::pixel_attn_bwd", mutates_args=())
def pixel_attn_bwd(
    dOut: torch.Tensor,
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_k: torch.Tensor,
    B_v: torch.Tensor,
    Out: torch.Tensor,
    LSE: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    prog_ptr: torch.Tensor,
    prog_pix: torch.Tensor,
    scale: float,
    max_seqlen_k: int,
    q_per_kv: int,
    token_dim: int,
    n_kv_heads: int,
    use_v_bias: bool,
    force_fp32: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    return _gqa_bwd_impl(
        dOut,
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )


@pixel_attn_bwd.register_fake
def _fake_bwd(
    dOut,
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    Out,
    LSE,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32,
):
    return (
        Q.new_empty(Q.shape),
        tokens.new_empty(tokens.shape),
        W_k.new_empty(W_k.shape),
        W_v.new_empty(W_v.shape),
        B_k.new_empty(B_k.shape),
        B_v.new_empty(B_v.shape),
    )


def _setup_context(ctx, inputs, output):
    (
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    ) = inputs
    Out, LSE = output
    # Save the packed tensors the Triton backward expects rather than rebuilding
    # projections during the autograd callback.
    ctx.save_for_backward(
        Q, tokens, W_k, W_v, B_k, B_v, Out, LSE, cu_seqlens_k, prog_ptr, prog_pix
    )
    ctx.scale = scale
    ctx.max_seqlen_k = max_seqlen_k
    ctx.q_per_kv = q_per_kv
    ctx.token_dim = token_dim
    ctx.n_kv_heads = n_kv_heads
    ctx.use_v_bias = use_v_bias
    ctx.force_fp32 = force_fp32


def _backward(ctx, grad_Out, grad_LSE):
    del grad_LSE
    (
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
    ) = ctx.saved_tensors
    dQ, d_tokens, dW_k, dW_v, dB_k, dB_v = pixel_attn_bwd(
        grad_Out,
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        ctx.scale,
        ctx.max_seqlen_k,
        ctx.q_per_kv,
        ctx.token_dim,
        ctx.n_kv_heads,
        ctx.use_v_bias,
        ctx.force_fp32,
    )
    # One grad slot per fwd input: 6 real grads then None for
    # cu_seqlens_k, prog_ptr, prog_pix, scale, max_seqlen_k, q_per_kv,
    # token_dim, n_kv_heads, use_v_bias, force_fp32.
    return (
        dQ,
        d_tokens,
        dW_k,
        dW_v,
        dB_k,
        dB_v,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


pixel_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


def _pixel_attention_gqa(
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
    force_fp32=False,
):
    n_q_heads = Q.shape[1]
    q_per_kv = n_q_heads // n_kv_heads
    token_dim = tokens.shape[1]
    use_v_bias = B_v is not None
    if B_k is None:
        # The custom op has a fixed tensor schema, so use empty placeholders when
        # a bias is logically absent.
        B_k = W_k.new_empty((0,))
    if B_v is None:
        B_v = W_v.new_empty((0,))
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_k = B_k.contiguous()
    B_v = B_v.contiguous()
    Out, _LSE = pixel_attn_fwd(
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )
    return Out


# ---------------------------------------------------------------------------
# Triton autotune config cache (opt-in startup optimization).
#
# Triton @autotune benchmarks every config the first time each shape-key is
# hit. When HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR points at a directory, the
# winning config for each shape-key is persisted there and reloaded on the next
# process. Tuning stays LAZY on the real first batch; write-through saves each
# newly tuned config. Each (GPU, rank) owns its own JSON in that dir, so there
# are no write races and no rank-0 barrier. The filename embeds GPU model +
# source/Triton-version hash, so a kernel edit or Triton upgrade invalidates it.
#
# This is orthogonal to Triton's own compiled-kernel cache (TRITON_CACHE_DIR):
# it stores only the chosen @autotune config, never generated kernels.
# ---------------------------------------------------------------------------
def _autotuners():
    """Return the ``{name: triton.Autotuner}`` map for the kernel backend."""
    return {
        "pixel_attn_gqa_fwd": _pixel_attn_gqa_fwd,
        "pixel_attn_gqa_bwd": _pixel_attn_gqa_bwd,
    }


_AUTOTUNE_CACHE_READY = False


def _autotune_cache_dir() -> str | None:
    """Directory for persisted autotune configs, or ``None`` when disabled.

    Enabled only when ``HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR`` is set to a
    non-empty path.
    """
    raw = os.environ.get("HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR")
    if raw is None or not raw.strip():
        return None
    return os.path.expanduser(raw.strip())


def _autotune_cache_file(cache_dir: str) -> str:
    with open(__file__, "rb") as f:
        digest = hashlib.sha1(f.read(), usedforsecurity=False).hexdigest()[:8]
    ver = getattr(triton, "__version__", "0")
    # Key by GPU model (a cache dir reused across GPU types never serves wrong-arch
    # configs) and by rank (each rank owns its file -> no write races, no barrier).
    gpu = (
        torch.cuda.get_device_name().replace(" ", "_")
        if torch.cuda.is_available()
        else "cpu"
    )
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    name = f"pixel_cross_attention-{gpu}-{digest}-triton{ver}-rank{rank}.json"
    return os.path.join(cache_dir, name)


def load_autotune_cache(path):
    return tac.load_caches(_autotuners(), path)


def save_autotune_cache(path):
    return tac.save_caches(_autotuners(), path)


def _install_writethrough(tuner, path):
    # Persist this rank's cache whenever the autotuner tunes a new key (i.e. the
    # first time a new shape/grid bucket is hit on the real workload), so the next
    # process loads it instead of re-benchmarking.
    if getattr(tuner, "_healda_writethrough", False):
        return
    tuner._healda_writethrough = True
    run = tuner.run

    def run_and_persist(*args, **kwargs):
        before = len(tuner.cache)
        out = run(*args, **kwargs)
        if len(tuner.cache) > before:
            save_autotune_cache(path)
        return out

    tuner.run = run_and_persist


def _ensure_autotune_cache():
    """Lazily wire up the per-rank autotune cache on the first pixel-attention call
    when ``HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE_DIR`` is set. Idempotent and best-effort."""
    global _AUTOTUNE_CACHE_READY
    if _AUTOTUNE_CACHE_READY:
        return
    _AUTOTUNE_CACHE_READY = True  # set first: best-effort, never retry per-call
    cache_dir = _autotune_cache_dir()
    if cache_dir is None:
        return
    path = _autotune_cache_file(cache_dir)
    load_autotune_cache(path)
    for tuner in _autotuners().values():
        _install_writethrough(tuner, path)


# ---------------------------------------------------------------------------
# Host dispatch: the Triton entry point PixelCrossAttention calls.
# ---------------------------------------------------------------------------


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
    :func:`~physicsnemo.experimental.models.healda.attention_layers._pixel_attention_reference` for the
    equivalent pure-PyTorch computation.

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
    group_map : :class:`~physicsnemo.experimental.models.healda.obs_context.PixelGroupMap`, optional, default=None
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
    _ensure_autotune_cache()
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
        return _pixel_attention_gqa(
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
            _pixel_attention_gqa(
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


# ---------------------------------------------------------------------------
# Counting sort: packs observations into the per-pixel contiguous layout
# pixel_attention consumes. Backend for ..obs_context.sort_and_pack.
# ---------------------------------------------------------------------------


@triton.jit
def _counting_sort_scatter(
    keys_ptr,
    sorted_order_ptr,
    bucket_offsets_ptr,
    N,
    BLOCK: tl.constexpr,
):
    # Counting sort over N items keyed by a bounded integer: each item
    # atomically claims the next free slot in its key's bucket and writes its
    # source index there, producing a key-grouped permutation in one pass.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    key = tl.load(keys_ptr + offs, mask=mask).to(tl.int64)
    pos = tl.atomic_add(bucket_offsets_ptr + key, 1, mask=mask)
    tl.store(sorted_order_ptr + pos.to(tl.int32), offs.to(tl.int32), mask=mask)


def counting_sort_and_pack(
    flat_idx: Int[torch.Tensor, " nobs"], total_pixels: int
) -> Tuple[Int[torch.Tensor, " nobs"], Int[torch.Tensor, " total_pixels"]]:
    r"""Sort observations by flat pixel index with a Triton counting sort (CUDA only).

    For bounded integer keys a counting sort is :math:`O(N)` in a single
    atomic-scatter pass, faster than ``argsort``'s multi-pass radix sort and
    with lower peak memory use. Within-bucket order is non-deterministic, which is
    fine for attention (permutation-invariant over a pixel's tokens).

    Parameters
    ----------
    flat_idx : torch.Tensor
        Int per-observation flat pixel indices of shape :math:`(N_{obs},)`, each
        in :math:`[0, \text{total\_pixels})`.
    total_pixels : int
        Number of pixel buckets (:math:`B \cdot T \cdot X`).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``sorted_order`` (int32 permutation of shape :math:`(N_{obs},)`) and
        ``counts`` (int64 per-pixel counts of shape
        :math:`(\text{total\_pixels},)`).
    """
    n = flat_idx.shape[0]
    device = flat_idx.device
    counts = torch.bincount(flat_idx.long(), minlength=total_pixels)
    bucket_offsets = torch.zeros(total_pixels, dtype=torch.int64, device=device)
    bucket_offsets[1:] = counts[:-1].cumsum(0)
    sorted_order = torch.empty(n, dtype=torch.int32, device=device)
    block = 1024
    grid = ((n + block - 1) // block,)
    _counting_sort_scatter[grid](flat_idx, sorted_order, bucket_offsets, n, BLOCK=block)
    return sorted_order, counts
