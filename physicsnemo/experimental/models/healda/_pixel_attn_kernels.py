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

"""Triton grouped-query kernels for pixel/observation cross-attention.

Private kernel backend for :mod:`..pixel_cross_attention`, imported lazily only
when triton is installed (mirrors the warp ``_warp_impl`` backends).
"""

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")
tl = triton.language

# Base-2 softmax: exp(x) == exp2(x * log2e), log(x) == log2(x) / log2e.
# Folding log2e into the score scale lets the kernels use the faster MUFU
# exp2/log2 hardware instructions. The LSE is stored in the log2 domain so the
# forward (producer) and backward (consumer) agree on the convention.
LOG2E = tl.constexpr(1.4426950408889634)


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
                tl.store(
                    dKV_ptr + kv_row_off + 0 * D_HEAD, dk0, mask=kv_mask[:, None]
                )
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
