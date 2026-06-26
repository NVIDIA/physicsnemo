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

"""Fused Triton kernels for the 2-layer FiLM observation tokenizer.

Private kernel backend for :mod:`..obs_film_tokenizer`, imported lazily only
when triton is installed (mirrors the warp ``_warp_impl`` backends and the
``_pixel_attn_kernels`` module).

The kernels keep the FiLM MLP weights resident in SRAM and replay the forward
computation during backward. They compute::

    cond  = [float_meta, obs_type_emb, channel_emb, platform_emb?]
    h     = SiLU(LayerNorm(Linear1(cond)))
    alpha, beta = split(Linear2(h))
    out   = alpha * obs + beta

Implementation notes
--------------------
All MLP weights fit in SRAM. The forward therefore evaluates ``Linear1`` as a
sum of segment-local matmuls instead of materializing ``cond`` in HBM. The
backward reconstructs the logical conditioning matrix with pointer-gather.
"""

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")
tl = triton.language


# ═══════════════════════════════════════════════════════════════════════════
# Forward kernel
#
# Linear1 is evaluated as a sum of segment-local matmuls instead of first
# materializing the full conditioning vector.
#
# Logical W1 layout:
#   [meta | obs_emb | channel_emb | platform_emb]
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _fused_film_fwd(
    OBS,
    FLOAT_META,
    OBS_TYPE_ID,
    CHANNEL,
    PLATFORM,
    EMBED_TABLE,
    CHAN_EMBED_TABLE,
    PLATFORM_EMBED_TABLE,
    W1,
    B1,
    LN_W,
    LN_B,
    W2,
    B2,
    OUT,
    N,
    META_DIM: tl.constexpr,
    OBS_EMBED_DIM: tl.constexpr,
    CHAN_EMBED_DIM: tl.constexpr,
    PLATFORM_EMBED_DIM: tl.constexpr,
    # Padded sizes (next-pow2, for tl.arange)
    META_PAD: tl.constexpr,
    OBS_EMBED_PAD: tl.constexpr,
    CHAN_EMBED_PAD: tl.constexpr,
    PLATFORM_EMBED_PAD: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUT_DIM: tl.constexpr,
    OUT_PAD: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N

    MLP_OUT: tl.constexpr = 2 * OUT_DIM
    MLP_OUT_PAD: tl.constexpr = 2 * OUT_PAD

    # Broadcasted offset vectors define the register tiles for each logical
    # segment and the flat row-major address grids for W1/W2 submatrices.
    offs_meta = tl.arange(0, META_PAD)
    offs_obs_emb = tl.arange(0, OBS_EMBED_PAD)
    offs_channel_emb = tl.arange(0, CHAN_EMBED_PAD)
    offs_hid = tl.arange(0, HIDDEN)
    offs_mlp = tl.arange(0, MLP_OUT_PAD)

    # Split `Linear1(cond)` into a sum of segment-local matmuls so forward
    # never has to materialize the concatenated conditioning vector.
    w1_meta = tl.load(
        W1 + offs_meta[:, None] * HIDDEN + offs_hid[None, :],
        mask=offs_meta[:, None] < META_DIM,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    w1_obs = tl.load(
        W1 + (META_DIM + offs_obs_emb[:, None]) * HIDDEN + offs_hid[None, :],
        mask=offs_obs_emb[:, None] < OBS_EMBED_DIM,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    w1_channel = tl.load(
        W1
        + (META_DIM + OBS_EMBED_DIM + offs_channel_emb[:, None]) * HIDDEN
        + offs_hid[None, :],
        mask=offs_channel_emb[:, None] < CHAN_EMBED_DIM,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    if PLATFORM_EMBED_DIM > 0:
        offs_platform_emb = tl.arange(0, PLATFORM_EMBED_PAD)
        w1_platform = tl.load(
            W1
            + (META_DIM + OBS_EMBED_DIM + CHAN_EMBED_DIM + offs_platform_emb[:, None])
            * HIDDEN
            + offs_hid[None, :],
            mask=offs_platform_emb[:, None] < PLATFORM_EMBED_DIM,
            other=0.0,
        ).to(COMPUTE_DTYPE)

    b1 = tl.load(B1 + offs_hid).to(tl.float32)
    ln_w = tl.load(LN_W + offs_hid).to(tl.float32)
    ln_b = tl.load(LN_B + offs_hid).to(tl.float32)
    w2 = tl.load(
        W2 + offs_hid[:, None] * MLP_OUT + offs_mlp[None, :],
        mask=offs_mlp[None, :] < MLP_OUT,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    b2 = tl.load(B2 + offs_mlp, mask=offs_mlp < MLP_OUT, other=0.0).to(tl.float32)

    # ── Load per-row inputs ───────────────────────────────────────
    obs_type = tl.load(OBS_TYPE_ID + rows, mask=rmask, other=0)
    channel = tl.load(CHANNEL + rows, mask=rmask, other=0)
    meta = tl.load(
        FLOAT_META + rows[:, None] * META_DIM + offs_meta[None, :],
        mask=rmask[:, None] & (offs_meta[None, :] < META_DIM),
        other=0.0,
    )

    obs_emb = tl.load(
        EMBED_TABLE + obs_type[:, None] * OBS_EMBED_DIM + offs_obs_emb[None, :],
        mask=rmask[:, None] & (offs_obs_emb[None, :] < OBS_EMBED_DIM),
        other=0.0,
    )
    channel_emb = tl.load(
        CHAN_EMBED_TABLE
        + channel[:, None] * CHAN_EMBED_DIM
        + offs_channel_emb[None, :],
        mask=rmask[:, None] & (offs_channel_emb[None, :] < CHAN_EMBED_DIM),
        other=0.0,
    )
    if PLATFORM_EMBED_DIM > 0:
        platform = tl.load(PLATFORM + rows, mask=rmask, other=0)
        platform_emb = tl.load(
            PLATFORM_EMBED_TABLE
            + platform[:, None] * PLATFORM_EMBED_DIM
            + offs_platform_emb[None, :],
            mask=rmask[:, None] & (offs_platform_emb[None, :] < PLATFORM_EMBED_DIM),
            other=0.0,
        )

    # This is exactly `Linear1(cond)`, just written as
    # `b1 + sum_i cond_segment_i @ W1_segment_i`.
    h = b1[None, :]
    h += tl.dot(meta.to(COMPUTE_DTYPE), w1_meta, out_dtype=tl.float32)
    h += tl.dot(obs_emb.to(COMPUTE_DTYPE), w1_obs, out_dtype=tl.float32)
    h += tl.dot(channel_emb.to(COMPUTE_DTYPE), w1_channel, out_dtype=tl.float32)
    if PLATFORM_EMBED_DIM > 0:
        h += tl.dot(platform_emb.to(COMPUTE_DTYPE), w1_platform, out_dtype=tl.float32)

    # ── LayerNorm + SiLU ─────────────────────────────────────────
    _, _, _, act, _ = _fwd_layernorm_silu(
        h,
        ln_w,
        ln_b,
        EPS=EPS,
        HIDDEN=HIDDEN,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
    )

    # ── Linear2 -> split -> FiLM ──────────────────────────────────
    ab = tl.dot(act, w2, out_dtype=tl.float32) + b2[None, :]
    ab = tl.reshape(ab, BLOCK_M, 2, OUT_PAD)
    ab = tl.permute(ab, (0, 2, 1))
    alpha, beta = ab.split()

    obs_val = tl.load(OBS + rows, mask=rmask, other=0.0)
    output = alpha * obs_val[:, None] + beta

    offs_od = tl.arange(0, OUT_PAD)
    tl.store(
        OUT + rows[:, None] * OUT_DIM + offs_od[None, :],
        output,
        mask=rmask[:, None] & (offs_od[None, :] < OUT_DIM),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Backward replay and kernel
#
# Design:
#   1. Rebuild the logical conditioning vector with pointer-gather.
#   2. Replay Linear1 -> LayerNorm -> SiLU -> Linear2.
#   3. Accumulate dense w1/w2/ln gradients in registers across the CTA's
#      tile-strided work, flushing once at the end.
#   4. Flush embedding-table gradients (which are too large to also fit in
#      registers) with atomics since rows can collide on the same embedding
#      ids across CTAs.
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _fwd_layernorm_silu(
    h,
    ln_w,
    ln_b,
    EPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Forward-replay LayerNorm + SiLU from pre-activation h."""
    mean = tl.sum(h, axis=1) / HIDDEN
    cent = h - mean[:, None]
    var = tl.sum(cent * cent, axis=1) / HIDDEN
    rstd = 1.0 / tl.sqrt(var + EPS)
    xhat = cent * rstd[:, None]
    normed = xhat * ln_w[None, :] + ln_b[None, :]
    sig = tl.sigmoid(normed)
    act = (normed * sig).to(COMPUTE_DTYPE)
    return xhat, normed, sig, act, rstd


# ── Backward ──────────────────────────────────────────────────────────────


@triton.jit
def _fused_film_bwd(
    GRAD_OUT,
    OBS,
    FLOAT_META,
    OBS_TYPE_ID,
    CHANNEL,
    PLATFORM,
    EMBED_TABLE,
    CHAN_EMBED_TABLE,
    PLATFORM_EMBED_TABLE,
    W1,
    B1,
    LN_W,
    LN_B,
    W2,
    DW1,
    DB1,
    DLN_W,
    DLN_B,
    DW2,
    DB2,
    GRAD_EMBED_TABLE,
    GRAD_CHAN_EMBED_TABLE,
    GRAD_PLATFORM_EMBED_TABLE,
    N,
    META_DIM: tl.constexpr,
    OBS_EMBED_DIM: tl.constexpr,
    CHAN_EMBED_DIM: tl.constexpr,
    PLATFORM_EMBED_DIM: tl.constexpr,
    COND_DIM: tl.constexpr,
    HIDDEN: tl.constexpr,
    OUT_DIM: tl.constexpr,
    COND_PAD: tl.constexpr,
    OUT_PAD: tl.constexpr,
    OBS_EMBED_PAD: tl.constexpr,
    CHAN_EMBED_PAD: tl.constexpr,
    PLATFORM_EMBED_PAD: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    total_tiles = tl.cdiv(N, BLOCK_M)

    # As in forward, these offsets define the logical tile shapes and the flat
    # address arithmetic for the weight and gradient matrices.
    offs_cond = tl.arange(0, COND_PAD)
    offs_hid = tl.arange(0, HIDDEN)
    offs_od = tl.arange(0, OUT_PAD)
    offs_obs_emb = tl.arange(0, OBS_EMBED_PAD)
    offs_channel_emb = tl.arange(0, CHAN_EMBED_PAD)
    offs_platform_emb = tl.arange(0, PLATFORM_EMBED_PAD)
    MLP_OUT: tl.constexpr = 2 * OUT_DIM
    MLP_OUT_PAD: tl.constexpr = 2 * OUT_PAD
    offs_mlp = tl.arange(0, MLP_OUT_PAD)

    w1 = tl.load(
        W1 + offs_cond[:, None] * HIDDEN + offs_hid[None, :],
        mask=offs_cond[:, None] < COND_DIM,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    b1 = tl.load(B1 + offs_hid).to(tl.float32)
    ln_w = tl.load(LN_W + offs_hid).to(tl.float32)
    ln_b = tl.load(LN_B + offs_hid).to(tl.float32)
    w2 = tl.load(
        W2 + offs_hid[:, None] * MLP_OUT + offs_mlp[None, :],
        mask=offs_mlp[None, :] < MLP_OUT,
        other=0.0,
    ).to(COMPUTE_DTYPE)

    # Only the W1 rows attached to embedding segments are needed to form
    # embedding-table gradients, so cache those slices explicitly.
    w1_obs = tl.load(
        W1 + (META_DIM + offs_obs_emb[:, None]) * HIDDEN + offs_hid[None, :],
        mask=offs_obs_emb[:, None] < OBS_EMBED_DIM,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    w1_channel = tl.load(
        W1
        + (META_DIM + OBS_EMBED_DIM + offs_channel_emb[:, None]) * HIDDEN
        + offs_hid[None, :],
        mask=offs_channel_emb[:, None] < CHAN_EMBED_DIM,
        other=0.0,
    ).to(COMPUTE_DTYPE)
    w1_platform = tl.zeros([PLATFORM_EMBED_PAD, HIDDEN], dtype=COMPUTE_DTYPE)
    if PLATFORM_EMBED_DIM > 0:
        w1_platform = tl.load(
            W1
            + (META_DIM + OBS_EMBED_DIM + CHAN_EMBED_DIM + offs_platform_emb[:, None])
            * HIDDEN
            + offs_hid[None, :],
            mask=offs_platform_emb[:, None] < PLATFORM_EMBED_DIM,
            other=0.0,
        ).to(COMPUTE_DTYPE)

    dw1_acc = tl.zeros([COND_PAD, HIDDEN], dtype=tl.float32)
    db1_acc = tl.zeros([HIDDEN], dtype=tl.float32)
    dln_w_acc = tl.zeros([HIDDEN], dtype=tl.float32)
    dln_b_acc = tl.zeros([HIDDEN], dtype=tl.float32)
    dw2_acc = tl.zeros([HIDDEN, MLP_OUT_PAD], dtype=tl.float32)
    db2_acc = tl.zeros([MLP_OUT_PAD], dtype=tl.float32)

    for tile_idx in range(pid, total_tiles, num_ctas):
        rows = tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < N

        # ── Build [BM, COND_PAD] conditioning via pointer-gather ──
        obs_type = tl.load(OBS_TYPE_ID + rows, mask=rmask, other=0)
        channel = tl.load(CHANNEL + rows, mask=rmask, other=0)
        platform = tl.load(PLATFORM + rows, mask=rmask, other=0)

        # Rebuild the logical conditioning vector column-by-column: every
        # column points at exactly one backing source tensor.
        is_meta = offs_cond[None, :] < META_DIM
        is_obs_emb = (offs_cond[None, :] >= META_DIM) & (
            offs_cond[None, :] < META_DIM + OBS_EMBED_DIM
        )
        is_channel_emb = (offs_cond[None, :] >= META_DIM + OBS_EMBED_DIM) & (
            offs_cond[None, :] < META_DIM + OBS_EMBED_DIM + CHAN_EMBED_DIM
        )

        ptr_meta = FLOAT_META + rows[:, None] * META_DIM + offs_cond[None, :]
        ptr_obs = (
            EMBED_TABLE
            + obs_type[:, None] * OBS_EMBED_DIM
            + (offs_cond[None, :] - META_DIM)
        )
        ptr_channel = (
            CHAN_EMBED_TABLE
            + channel[:, None] * CHAN_EMBED_DIM
            + (offs_cond[None, :] - META_DIM - OBS_EMBED_DIM)
        )
        ptr_platform = (
            PLATFORM_EMBED_TABLE
            + platform[:, None] * PLATFORM_EMBED_DIM
            + (offs_cond[None, :] - META_DIM - OBS_EMBED_DIM - CHAN_EMBED_DIM)
        )

        # Exactly one source pointer is selected per logical conditioning
        # column, matching the layout an explicit concatenation would produce.
        cond_ptr = tl.where(
            is_meta,
            ptr_meta,
            tl.where(
                is_obs_emb,
                ptr_obs,
                tl.where(is_channel_emb, ptr_channel, ptr_platform),
            ),
        )
        cond = tl.load(
            cond_ptr, mask=rmask[:, None] & (offs_cond[None, :] < COND_DIM), other=0.0
        )
        cond_compute = cond.to(COMPUTE_DTYPE)

        grad = tl.load(
            GRAD_OUT + rows[:, None] * OUT_DIM + offs_od[None, :],
            mask=rmask[:, None] & (offs_od[None, :] < OUT_DIM),
            other=0.0,
        ).to(tl.float32)
        obs_val = tl.load(OBS + rows, mask=rmask, other=0.0)

        h = tl.dot(cond_compute, w1, out_dtype=tl.float32) + b1[None, :]
        xhat, normed, sig, act, rstd = _fwd_layernorm_silu(
            h,
            ln_w,
            ln_b,
            EPS=EPS,
            HIDDEN=HIDDEN,
            COMPUTE_DTYPE=COMPUTE_DTYPE,
        )

        # For `out = alpha * obs + beta`, the FiLM Jacobian is immediate:
        # `d_alpha = grad_out * obs`, `d_beta = grad_out`.
        d_alpha = grad * obs_val[:, None]
        d_beta = grad
        d_ab = tl.join(d_alpha, d_beta)
        d_ab = tl.permute(d_ab, (0, 2, 1))
        d_ab = tl.reshape(d_ab, BLOCK_M, MLP_OUT_PAD)

        d_act = tl.dot(d_ab.to(COMPUTE_DTYPE), tl.trans(w2), out_dtype=tl.float32)
        dsilu = sig + normed * sig * (1.0 - sig)
        d_normed = d_act * dsilu

        dxhat = d_normed * ln_w[None, :]
        s1 = tl.sum(dxhat, axis=1)
        s2 = tl.sum(dxhat * xhat, axis=1)
        # Closed-form LayerNorm backward:
        # dh = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)).
        dh = rstd[:, None] * (dxhat - (s1[:, None] + xhat * s2[:, None]) / HIDDEN)

        dw2_acc += tl.dot(tl.trans(act), d_ab.to(COMPUTE_DTYPE), out_dtype=tl.float32)
        db2_acc += tl.sum(d_ab, axis=0)
        dln_w_acc += tl.sum(d_normed * xhat, axis=0)
        dln_b_acc += tl.sum(d_normed, axis=0)

        dh_compute = dh.to(COMPUTE_DTYPE)
        dw1_acc += tl.dot(tl.trans(cond_compute), dh_compute, out_dtype=tl.float32)
        db1_acc += tl.sum(dh, axis=0)

        # Flush embedding gradients to gradient tables shared across all CTAs.
        d_obs_emb = tl.dot(dh_compute, tl.trans(w1_obs), out_dtype=tl.float32)
        d_channel_emb = tl.dot(dh_compute, tl.trans(w1_channel), out_dtype=tl.float32)
        tl.atomic_add(
            GRAD_EMBED_TABLE
            + obs_type[:, None] * OBS_EMBED_DIM
            + offs_obs_emb[None, :],
            d_obs_emb,
            mask=rmask[:, None] & (offs_obs_emb[None, :] < OBS_EMBED_DIM),
        )
        tl.atomic_add(
            GRAD_CHAN_EMBED_TABLE
            + channel[:, None] * CHAN_EMBED_DIM
            + offs_channel_emb[None, :],
            d_channel_emb,
            mask=rmask[:, None] & (offs_channel_emb[None, :] < CHAN_EMBED_DIM),
        )
        if PLATFORM_EMBED_DIM > 0:
            d_platform_emb = tl.dot(
                dh_compute, tl.trans(w1_platform), out_dtype=tl.float32
            )
            tl.atomic_add(
                GRAD_PLATFORM_EMBED_TABLE
                + platform[:, None] * PLATFORM_EMBED_DIM
                + offs_platform_emb[None, :],
                d_platform_emb,
                mask=rmask[:, None] & (offs_platform_emb[None, :] < PLATFORM_EMBED_DIM),
            )

    # Flush register accumulators once per CTA after tile-striding over N.
    tl.atomic_add(
        DW1 + offs_cond[:, None] * HIDDEN + offs_hid[None, :],
        dw1_acc,
        mask=offs_cond[:, None] < COND_DIM,
    )
    tl.atomic_add(DB1 + offs_hid, db1_acc)
    tl.atomic_add(DLN_W + offs_hid, dln_w_acc)
    tl.atomic_add(DLN_B + offs_hid, dln_b_acc)
    tl.atomic_add(
        DW2 + offs_hid[:, None] * MLP_OUT + offs_mlp[None, :],
        dw2_acc,
        mask=offs_mlp[None, :] < MLP_OUT,
    )
    tl.atomic_add(DB2 + offs_mlp, db2_acc, mask=offs_mlp < MLP_OUT)
