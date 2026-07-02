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

Private backend for :mod:`~physicsnemo.experimental.models.healda.obs_tokenizer`,
imported lazily only when triton is installed. Contains:

1. The ``@triton.jit`` forward/backward kernels.
2. ``torch.library.custom_op`` wrappers with fake-tensor and autograd
   registration.
3. The sweep-selected launch config presets (block/warp/stage counts).

The kernels compute::

    cond  = [float_meta, obs_type_emb, channel_emb, platform_emb?]
    h     = SiLU(LayerNorm(Linear1(cond)))
    alpha, beta = split(Linear2(h))
    out   = alpha * obs + beta

Implementation notes
--------------------
All MLP weights fit in SRAM, so the forward evaluates ``Linear1`` as a sum of
segment-local matmuls instead of materializing ``cond`` in HBM. The backward
reconstructs the logical conditioning matrix with pointer-gather and replays
the forward to recompute activations.
"""

from dataclasses import dataclass

import torch

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")
tl = triton.language


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _cond_dim(
    *, meta_dim: int, obs_embed_dim: int, chan_embed_dim: int, platform_embed_dim: int
) -> int:
    return meta_dim + obs_embed_dim + chan_embed_dim + platform_embed_dim


# Presets chosen from sweep on H100. Need optimal performance with dynamic input
# sizes, so hard coded presets for nobs >O(1M).
@dataclass(frozen=True)
class _KernelPreset:
    BLOCK_M: int
    num_warps: int
    num_stages: int

    def as_config_dict(self) -> dict[str, int | None]:
        return {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "num_ctas": 1,
            "maxnreg": None,
            "BLOCK_M": self.BLOCK_M,
        }


_FWD_PRESET = _KernelPreset(BLOCK_M=64, num_warps=4, num_stages=2)

_BWD_PRESET_A = _KernelPreset(BLOCK_M=64, num_warps=8, num_stages=1)
_BWD_PRESET_B = _KernelPreset(BLOCK_M=128, num_warps=8, num_stages=1)
_BWD_PRESET_C = _KernelPreset(BLOCK_M=128, num_warps=8, num_stages=2)


def _select_bwd_preset(
    *,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
) -> _KernelPreset:
    # Backward config selection is driven mostly by the padded conditioning width
    # because the persistent kernel keeps replay state and reduction accumulators
    # live across the tile loop.
    cond_dim = _cond_dim(
        meta_dim=meta_dim,
        obs_embed_dim=obs_embed_dim,
        chan_embed_dim=chan_embed_dim,
        platform_embed_dim=platform_embed_dim,
    )
    cond_pad = _next_pow2(max(cond_dim, 16))
    if cond_pad > 64:
        return _BWD_PRESET_A
    if cond_dim <= 46:
        return _BWD_PRESET_C
    if platform_embed_dim > 0:
        return _BWD_PRESET_B
    return _BWD_PRESET_A


def get_fused_film_launch_configs(
    *,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
) -> dict[str, dict[str, int | None]]:
    """Return the sweep-selected Triton launch config for this FiLM layout."""
    bwd_preset = _select_bwd_preset(
        meta_dim=meta_dim,
        obs_embed_dim=obs_embed_dim,
        chan_embed_dim=chan_embed_dim,
        platform_embed_dim=platform_embed_dim,
    )
    return {
        "fwd": _FWD_PRESET.as_config_dict(),
        "bwd": bwd_preset.as_config_dict(),
    }


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


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers (custom ops for torch.compile)
# These wrappers are split into:
#   - Python launchers that allocate outputs/gradients,
#   - public/private custom ops for ``torch.compile`` compatibility,
#   - autograd glue that saves replay inputs for backward.
# ═══════════════════════════════════════════════════════════════════════════


def _launch_fused_film_fwd(
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> torch.Tensor:
    N = obs.shape[0]
    hidden = w1.shape[1]

    out_dtype = torch.float32 if force_fp32 else torch.bfloat16
    out = torch.empty(N, out_dim, device=obs.device, dtype=out_dtype)
    if N == 0:
        return out

    # Forward handles conditioning as a sum of segment-local loads/dots, so each
    # segment gets its own masked `tl.arange` extent. Using next-pow2 widths
    # keeps those tile shapes Triton-friendly without materializing full `cond`.
    meta_pad = _next_pow2(max(meta_dim, 16))
    obs_embed_pad = _next_pow2(max(obs_embed_dim, 16))
    chan_embed_pad = _next_pow2(max(chan_embed_dim, 16))
    platform_embed_pad = _next_pow2(max(platform_embed_dim, 16))
    out_pad = _next_pow2(max(out_dim, 16))
    grid = ((N + _FWD_PRESET.BLOCK_M - 1) // _FWD_PRESET.BLOCK_M,)
    _fused_film_fwd[grid](
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        out,
        N,
        META_DIM=meta_dim,
        OBS_EMBED_DIM=obs_embed_dim,
        CHAN_EMBED_DIM=chan_embed_dim,
        PLATFORM_EMBED_DIM=platform_embed_dim,
        META_PAD=meta_pad,
        OBS_EMBED_PAD=obs_embed_pad,
        CHAN_EMBED_PAD=chan_embed_pad,
        PLATFORM_EMBED_PAD=platform_embed_pad,
        HIDDEN=hidden,
        OUT_DIM=out_dim,
        OUT_PAD=out_pad,
        EPS=eps,
        BLOCK_M=_FWD_PRESET.BLOCK_M,
        COMPUTE_DTYPE=tl.float32 if force_fp32 else tl.bfloat16,
        num_warps=_FWD_PRESET.num_warps,
        num_stages=_FWD_PRESET.num_stages,
    )
    return out


@torch.library.custom_op("healda::fused_film_fwd", mutates_args=())
def fused_film_fwd(
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> torch.Tensor:
    """Forward custom op that launches the fused FiLM Triton kernel."""
    return _launch_fused_film_fwd(
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    )


@fused_film_fwd.register_fake
def _fake_fused_film_fwd(
    obs,
    float_meta,
    obs_type_id,
    channel,
    platform,
    embed_weight,
    chan_embed_weight,
    platform_embed_weight,
    w1,
    b1,
    ln_w,
    ln_b,
    w2,
    b2,
    eps,
    meta_dim,
    obs_embed_dim,
    chan_embed_dim,
    platform_embed_dim,
    out_dim,
    force_fp32,
):
    N = obs.shape[0]
    return obs.new_empty(
        (N, out_dim), dtype=torch.float32 if force_fp32 else torch.bfloat16
    )


def _launch_fused_film_bwd(
    grad_out: torch.Tensor,
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    N = obs.shape[0]
    cond_dim = meta_dim + obs_embed_dim + chan_embed_dim + platform_embed_dim
    hidden = w1.shape[1]
    mlp_out_dim = 2 * out_dim
    grad_out = grad_out.contiguous()
    grad_embed = torch.zeros_like(embed_weight)
    grad_chan_embed = torch.zeros_like(chan_embed_weight)
    grad_platform_embed = torch.zeros_like(platform_embed_weight)

    out_pad = _next_pow2(max(out_dim, 16))
    obs_embed_pad = _next_pow2(max(obs_embed_dim, 16))
    chan_embed_pad = _next_pow2(max(chan_embed_dim, 16))
    platform_embed_pad = _next_pow2(max(platform_embed_dim, 16))

    dw1 = torch.zeros(cond_dim, hidden, device=obs.device, dtype=torch.float32)
    db1 = torch.zeros(hidden, device=obs.device, dtype=torch.float32)
    dln_w = torch.zeros(hidden, device=obs.device, dtype=torch.float32)
    dln_b = torch.zeros(hidden, device=obs.device, dtype=torch.float32)
    dw2 = torch.zeros(hidden, mlp_out_dim, device=obs.device, dtype=torch.float32)
    db2 = torch.zeros(mlp_out_dim, device=obs.device, dtype=torch.float32)

    if N == 0:
        return (
            grad_embed,
            grad_chan_embed,
            grad_platform_embed,
            dw1,
            db1,
            dln_w,
            dln_b,
            dw2,
            db2,
        )

    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    num_sms = torch.cuda.get_device_properties(obs.device).multi_processor_count

    bwd_preset = _select_bwd_preset(
        meta_dim=meta_dim,
        obs_embed_dim=obs_embed_dim,
        chan_embed_dim=chan_embed_dim,
        platform_embed_dim=platform_embed_dim,
    )
    cond_pad = _next_pow2(max(cond_dim, 16))
    # Persistent launch: cap the grid at the SM count and let each CTA stride
    # over multiple row tiles.
    bwd_grid = (min(num_sms, (N + bwd_preset.BLOCK_M - 1) // bwd_preset.BLOCK_M),)

    _fused_film_bwd[bwd_grid](
        grad_out,
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        dw1,
        db1,
        dln_w,
        dln_b,
        dw2,
        db2,
        grad_embed,
        grad_chan_embed,
        grad_platform_embed,
        N,
        META_DIM=meta_dim,
        OBS_EMBED_DIM=obs_embed_dim,
        CHAN_EMBED_DIM=chan_embed_dim,
        PLATFORM_EMBED_DIM=platform_embed_dim,
        COND_DIM=cond_dim,
        HIDDEN=hidden,
        OUT_DIM=out_dim,
        COND_PAD=cond_pad,
        OUT_PAD=out_pad,
        OBS_EMBED_PAD=obs_embed_pad,
        CHAN_EMBED_PAD=chan_embed_pad,
        PLATFORM_EMBED_PAD=platform_embed_pad,
        EPS=eps,
        BLOCK_M=bwd_preset.BLOCK_M,
        COMPUTE_DTYPE=compute_dtype,
        num_warps=bwd_preset.num_warps,
        num_stages=bwd_preset.num_stages,
    )

    return (
        grad_embed,
        grad_chan_embed,
        grad_platform_embed,
        dw1,
        db1,
        dln_w,
        dln_b,
        dw2,
        db2,
    )


@torch.library.custom_op("healda::fused_film_bwd", mutates_args=())
def fused_film_bwd(
    grad_out: torch.Tensor,
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Backward pass for the fused FiLM tokenizer.

    Rebuilds a padded conditioning matrix with pointer-gather, replays the
    FiLM MLP, and accumulates parameter and embedding gradients with a
    single persistent-CTA Triton kernel.

    Returns (grad_embed, grad_chan_embed, grad_platform_embed,
             dw1, db1, dln_w, dln_b, dw2, db2).
    """
    return _launch_fused_film_bwd(
        grad_out,
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    )


@fused_film_bwd.register_fake
def _fake_fused_film_bwd(
    grad_out,
    obs,
    float_meta,
    obs_type_id,
    channel,
    platform,
    embed_weight,
    chan_embed_weight,
    platform_embed_weight,
    w1,
    b1,
    ln_w,
    ln_b,
    w2,
    b2,
    eps,
    meta_dim,
    obs_embed_dim,
    chan_embed_dim,
    platform_embed_dim,
    out_dim,
    force_fp32,
):
    cond_dim = meta_dim + obs_embed_dim + chan_embed_dim + platform_embed_dim
    hidden = w1.shape[1]
    mlp_out_dim = 2 * out_dim
    return (
        embed_weight.new_empty(embed_weight.shape),
        chan_embed_weight.new_empty(chan_embed_weight.shape),
        platform_embed_weight.new_empty(platform_embed_weight.shape),
        w1.new_empty((cond_dim, hidden), dtype=torch.float32),
        w1.new_empty((hidden,), dtype=torch.float32),
        w1.new_empty((hidden,), dtype=torch.float32),
        w1.new_empty((hidden,), dtype=torch.float32),
        w1.new_empty((hidden, mlp_out_dim), dtype=torch.float32),
        w1.new_empty((mlp_out_dim,), dtype=torch.float32),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Autograd glue
# ═══════════════════════════════════════════════════════════════════════════


def _setup_context(ctx, inputs, output):
    (
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    ) = inputs
    # Save exactly the tensors needed to replay the FiLM computation in backward.
    ctx.save_for_backward(
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
    )
    ctx.eps = eps
    ctx.meta_dim = meta_dim
    ctx.obs_embed_dim = obs_embed_dim
    ctx.chan_embed_dim = chan_embed_dim
    ctx.platform_embed_dim = platform_embed_dim
    ctx.out_dim = out_dim
    ctx.force_fp32 = force_fp32


def _backward(ctx, grad_out):
    (
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
    ) = ctx.saved_tensors

    ge, gce, gpe, dw1, db1, dln_w, dln_b, dw2, db2 = fused_film_bwd(
        grad_out,
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        ctx.eps,
        ctx.meta_dim,
        ctx.obs_embed_dim,
        ctx.chan_embed_dim,
        ctx.platform_embed_dim,
        ctx.out_dim,
        ctx.force_fp32,
    )
    # The inputs are metadata, indices, and scalar observations; gradients are
    # exposed for the learned tables and dense FiLM MLP parameters only.
    return (
        None,
        None,
        None,
        None,
        None,  # obs, float_meta, obs_type_id, channel, platform
        ge,
        gce,
        gpe,  # embed, chan_embed, platform_embed
        dw1.to(w1.dtype),
        db1.to(b1.dtype),
        dln_w.to(ln_w.dtype),
        dln_b.to(ln_b.dtype),
        dw2.to(w2.dtype),
        db2.to(b2.dtype),
        None,  # eps
        None,  # meta_dim
        None,  # obs_embed_dim
        None,  # chan_embed_dim
        None,  # platform_embed_dim
        None,  # out_dim
        None,  # force_fp32
    )


fused_film_fwd.register_autograd(_backward, setup_context=_setup_context)
