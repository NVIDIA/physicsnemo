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

r"""Fused Triton kernel for ISLA's per-layer geometry region.

The geometry region of a slice block (``_blocks._geo_region``) computes four things on
the (B, N, S) point-slice grid: the point-anchor relational invariants; the (5+T)->1
routing-bias Linear on them, with T the per-point vectors (the global directions, then
the token's rank-1 inputs); the point->slice softmax of the pre-logits plus that bias;
and the mix-weighted pooling of the invariants back to the points. Eager PyTorch
writes every intermediate to memory and reads it back: rel (B,N,S,3), dist, rel_hat,
the T + 3 dot products and the concatenated (B,N,S,geo) invariants. torch.compile
fuses the elementwise chain but still materializes the (B,N,S,geo) invariants, in
float32 and again in bfloat16 for the GEMM, in forward and again in backward. The
region is memory-bound, so this traffic is its cost.

The two kernels here compute the same arithmetic on register tiles of ``BLOCK_N``
points by all ``S`` slices, and write no (B,N,S,.) intermediate. The forward reads the
pre-logits once and writes the routing bias, the mix and the (B,N,geo) pooled
invariants once. The backward rebuilds the invariants inside the tile from the
(B,N,3) and (B,S,3) geometry, reads the incoming gradients once and writes the
pre-logit gradient once. Besides those (B,N,S) inputs and outputs, the region touches
O(B N) and O(B S) tensors but for one kind. The per-slice sums over points (the
gradients of the anchors, of their mean normals and of the Linear) are stored as
per-tile partial sums that torch reduces, so the result is deterministic (no
atomics). The two anchor partials are (B, N / BLOCK_N, S, 3) each, with ``BLOCK_N``
of 1 to 4 points (:func:`_bwd_block`), so they grow as N S: at 1,024 slices and
200,000 tokens in float32 each is 2.3 GiB. A program that sums several tiles into
one partial, as ``_fused_basis._basis_bwd_kernel`` does, would bound them; that
changes the backward's summation order and is left to a follow-up.

The kernels round as the eager region does under autocast: the invariants and the
Linear's operands to bfloat16 where autocast runs the Linear and the pooling GEMM in
bfloat16, the pre-logit plus bias sum to the pre-logits' dtype, and the softmax in
float32. The fused forward thus matches the eager forward up to floating-point
reordering. The backward's reductions accumulate in float32 where autocast's GEMM
backward rounds to bfloat16. Only there do the two paths differ by more than
reordering, and the fused gradient is the more precise. Float64 is supported for
exactness tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

### ISLA resolves _geo_kernel to the eager region when HAS_TRITON is False.
from physicsnemo.experimental.models.isla._triton import (
    HAS_TRITON,
    libdevice,
    tl,
    triton,
)
from physicsnemo.experimental.models.isla._blocks import _geo_width

MAX_SLICES = 1024
#: Per-point vectors the kernels are unrolled for: global directions plus token rank-1 vectors.
MAX_DIRECTIONS = 4


if HAS_TRITON:

    @triton.jit
    def _rn(x, ROUND_BF16: tl.constexpr):
        """Round-trip through bfloat16 where autocast would have rounded."""
        if ROUND_BF16:
            return x.to(tl.bfloat16).to(tl.float32)
        else:
            return x

    @triton.jit
    def _div(a, b, F64: tl.constexpr):
        # correctly rounded division, as PyTorch's; Triton's ``/`` on fp32 is the approximate div.full
        if F64:
            return a / b
        else:
            return tl.math.div_rn(a, b)

    @triton.jit
    def _sqrt(x, F64: tl.constexpr):
        if F64:
            return libdevice.sqrt(x)
        else:
            return tl.math.sqrt_rn(x)

    @triton.jit
    def _load3(ptr, offs, mask):
        return (
            tl.load(ptr + offs, mask=mask, other=0.0),
            tl.load(ptr + offs + 1, mask=mask, other=0.0),
            tl.load(ptr + offs + 2, mask=mask, other=0.0),
        )

    @triton.jit
    def _region_tile(
        r_ptr, n_ptr, d_ptr, z_ptr, m_ptr, logit_ptr, w_ptr, b_ptr, pid_b, offs_n, offs_s, mask_n, mask_s, mask, eps,
        r_sb, r_sn, n_sb, n_sn, d_sb, d_sn, d_sk, z_sb, z_ss, m_sb, m_ss, l_sb, l_sn,
        K: tl.constexpr, T: tl.constexpr, IDX_E: tl.constexpr, IDX_F: tl.constexpr, IDX_G: tl.constexpr,
        LIN_BF16: tl.constexpr, LOGIT_BF16: tl.constexpr, F64: tl.constexpr,
    ):
        """The forward tile both kernels compute, from the loads to the mix. Returns the
        inputs, the offsets ``rel``, the distance before and after its clamp,
        ``rel_hat``, the invariants and routing weights as the Linear reads them, the
        routing bias and the point->slice mix. The per-point vector slots past ``T``, and
        their invariants and weights, repeat the first one's."""
        r0, r1, r2 = _load3(r_ptr, pid_b * r_sb + offs_n * r_sn, mask_n)
        n0, n1, n2 = _load3(n_ptr, pid_b * n_sb + offs_n * n_sn, mask_n)
        d0, d1, d2 = _load3(d_ptr, pid_b * d_sb + offs_n * d_sn, mask_n)
        e0, e1, e2 = d0, d1, d2
        f0, f1, f2 = d0, d1, d2
        i0, i1, i2 = d0, d1, d2
        if T >= 2:
            e0, e1, e2 = _load3(d_ptr, pid_b * d_sb + offs_n * d_sn + d_sk, mask_n)
        if T >= 3:
            f0, f1, f2 = _load3(d_ptr, pid_b * d_sb + offs_n * d_sn + 2 * d_sk, mask_n)
        if T >= 4:
            i0, i1, i2 = _load3(d_ptr, pid_b * d_sb + offs_n * d_sn + 3 * d_sk, mask_n)
        z0, z1, z2 = _load3(z_ptr, pid_b * z_sb + offs_s * z_ss, mask_s)
        m0, m1, m2 = _load3(m_ptr, pid_b * m_sb + offs_s * m_ss, mask_s)

        rel0 = r0[:, None] - z0[None, :]
        rel1 = r1[:, None] - z1[None, :]
        rel2 = r2[:, None] - z2[None, :]
        dist_raw = _sqrt(rel0 * rel0 + rel1 * rel1 + rel2 * rel2, F64)
        dist = tl.maximum(dist_raw, eps)
        h0 = _div(rel0, dist, F64)
        h1 = _div(rel1, dist, F64)
        h2 = _div(rel2, dist, F64)
        g0 = dist
        g1 = libdevice.log(dist)
        g2 = h0 * d0[:, None] + h1 * d1[:, None] + h2 * d2[:, None]
        if T >= 2:
            g2b = h0 * e0[:, None] + h1 * e1[:, None] + h2 * e2[:, None]
        if T >= 3:
            g2c = h0 * f0[:, None] + h1 * f1[:, None] + h2 * f2[:, None]
        if T >= 4:
            g2d = h0 * i0[:, None] + h1 * i1[:, None] + h2 * i2[:, None]
        g3 = h0 * n0[:, None] + h1 * n1[:, None] + h2 * n2[:, None]
        g4 = h0 * m0[None, :] + h1 * m1[None, :] + h2 * m2[None, :]
        g5 = (
            n0[:, None] * m0[None, :]
            + n1[:, None] * m1[None, :]
            + n2[:, None] * m2[None, :]
        )

        bb = _rn(tl.load(b_ptr), LIN_BF16)
        q0 = _rn(g0, LIN_BF16)
        q1 = _rn(g1, LIN_BF16)
        q2 = _rn(g2, LIN_BF16)
        q3 = _rn(g3, LIN_BF16)
        q4 = _rn(g4, LIN_BF16)
        q5 = _rn(g5, LIN_BF16)
        q2b = q2
        q2c = q2
        q2d = q2
        if T >= 2:
            q2b = _rn(g2b, LIN_BF16)
        if T >= 3:
            q2c = _rn(g2c, LIN_BF16)
        if T >= 4:
            q2d = _rn(g2d, LIN_BF16)
        w0 = _rn(tl.load(w_ptr + 0), LIN_BF16)
        w1 = _rn(tl.load(w_ptr + 1), LIN_BF16)
        w2 = _rn(tl.load(w_ptr + 2), LIN_BF16)
        w3 = _rn(tl.load(w_ptr + 2 + K), LIN_BF16)
        w4 = _rn(tl.load(w_ptr + 3 + K), LIN_BF16)
        w5 = _rn(tl.load(w_ptr + 4 + K), LIN_BF16)
        w2b = w2
        w2c = w2
        w2d = w2
        bias = q0 * w0 + q1 * w1 + q2 * w2 + q3 * w3 + q4 * w4 + q5 * w5
        if T >= 2:
            w2b = _rn(tl.load(w_ptr + IDX_E), LIN_BF16)
            bias = bias + q2b * w2b
        if T >= 3:
            w2c = _rn(tl.load(w_ptr + IDX_F), LIN_BF16)
            bias = bias + q2c * w2c
        if T >= 4:
            w2d = _rn(tl.load(w_ptr + IDX_G), LIN_BF16)
            bias = bias + q2d * w2d
        bias = _rn(bias + bb, LIN_BF16)

        l_off = pid_b * l_sb + offs_n[:, None] * l_sn + offs_s[None, :]
        logit = tl.load(logit_ptr + l_off, mask=mask, other=0.0).to(bias.dtype)
        x = _rn(logit + bias, LOGIT_BF16)
        x = tl.where(mask_s[None, :], x, float("-inf"))
        mx = tl.max(x, axis=1)
        e = libdevice.exp(x - mx[:, None])
        ssum = tl.sum(e, axis=1)
        mix = _div(e, ssum[:, None], F64)
        return (
            n0, n1, n2, d0, d1, d2, e0, e1, e2, f0, f1, f2, i0, i1, i2, m0, m1, m2, rel0, rel1, rel2, dist_raw, dist,
            h0, h1, h2, q0, q1, q2, q2b, q2c, q2d, q3, q4, q5, w0, w1, w2, w2b, w2c, w2d, w3, w4, w5, bias, mix,
        )

    @triton.jit
    def _geo_fwd_kernel(
        r_ptr,
        n_ptr,
        d_ptr,
        z_ptr,
        m_ptr,
        logit_ptr,
        w_ptr,
        b_ptr,
        bias_out_ptr,
        mix_out_ptr,
        pooled_out_ptr,
        N,
        S,
        eps,
        r_sb,
        r_sn,
        n_sb,
        n_sn,
        d_sb,
        d_sn,
        d_sk,
        z_sb,
        z_ss,
        m_sb,
        m_ss,
        l_sb,
        l_sn,
        bo_sb,
        bo_sn,
        mo_sb,
        mo_sn,
        po_sb,
        po_sn,
        K: tl.constexpr,
        T: tl.constexpr,
        LIN_BF16: tl.constexpr,
        LOGIT_BF16: tl.constexpr,
        F64: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        ### int64 program ids, so every offset below is int64: the (B, N, S) offsets pass
        ### 2**31 at N x S >= 2**31 (4,194,304 points at 512 slices).
        pid_n = tl.program_id(0).to(tl.int64)
        pid_b = tl.program_id(1).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_s = tl.arange(0, BLOCK_S)
        mask_n = offs_n < N
        mask_s = offs_s < S
        mask = mask_n[:, None] & mask_s[None, :]
        # invariant / weight index of the second, third and fourth per-point vectors: a global
        # direction sits right after the first (2 + j); a token vector after n.m (5 + K + j')
        IDX_E: tl.constexpr = 3 if K >= 2 else 6
        IDX_F: tl.constexpr = 4 if K >= 3 else 7
        IDX_G: tl.constexpr = 5 if K >= 4 else 8

        (
            n0, n1, n2, d0, d1, d2, e0, e1, e2, f0, f1, f2, i0, i1, i2, m0, m1, m2, rel0, rel1, rel2, dist_raw, dist,
            h0, h1, h2, q0, q1, q2, q2b, q2c, q2d, q3, q4, q5, w0, w1, w2, w2b, w2c, w2d, w3, w4, w5, bias, mix,
        ) = _region_tile(
            r_ptr, n_ptr, d_ptr, z_ptr, m_ptr, logit_ptr, w_ptr, b_ptr, pid_b, offs_n, offs_s, mask_n, mask_s, mask, eps,
            r_sb, r_sn, n_sb, n_sn, d_sb, d_sn, d_sk, z_sb, z_ss, m_sb, m_ss, l_sb, l_sn,
            K, T, IDX_E, IDX_F, IDX_G, LIN_BF16, LOGIT_BF16, F64,
        )

        tl.store(
            bias_out_ptr + pid_b * bo_sb + offs_n[:, None] * bo_sn + offs_s[None, :],
            bias.to(bias_out_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            mix_out_ptr + pid_b * mo_sb + offs_n[:, None] * mo_sn + offs_s[None, :],
            mix.to(mix_out_ptr.dtype.element_ty),
            mask=mask,
        )

        mq = _rn(mix, LIN_BF16)
        po = pooled_out_ptr + pid_b * po_sb + offs_n * po_sn
        ot = pooled_out_ptr.dtype.element_ty
        tl.store(po + 0, _rn(tl.sum(mq * q0, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 1, _rn(tl.sum(mq * q1, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 2, _rn(tl.sum(mq * q2, axis=1), LIN_BF16).to(ot), mask=mask_n)
        if T >= 2:
            tl.store(po + IDX_E, _rn(tl.sum(mq * q2b, axis=1), LIN_BF16).to(ot), mask=mask_n)
        if T >= 3:
            tl.store(po + IDX_F, _rn(tl.sum(mq * q2c, axis=1), LIN_BF16).to(ot), mask=mask_n)
        if T >= 4:
            tl.store(po + IDX_G, _rn(tl.sum(mq * q2d, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 2 + K, _rn(tl.sum(mq * q3, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 3 + K, _rn(tl.sum(mq * q4, axis=1), LIN_BF16).to(ot), mask=mask_n)
        tl.store(po + 4 + K, _rn(tl.sum(mq * q5, axis=1), LIN_BF16).to(ot), mask=mask_n)

    @triton.jit
    def _geo_bwd_kernel(
        r_ptr,
        n_ptr,
        d_ptr,
        z_ptr,
        m_ptr,
        logit_ptr,
        w_ptr,
        b_ptr,
        gbias_ptr,
        gmix_ptr,
        gpooled_ptr,
        glogit_ptr,
        gr_ptr,
        gn_ptr,
        gd_ptr,
        gz_part_ptr,
        gm_part_ptr,
        gw_part_ptr,
        gb_part_ptr,
        N,
        S,
        eps,
        n_tiles,
        r_sb,
        r_sn,
        n_sb,
        n_sn,
        d_sb,
        d_sn,
        d_sk,
        z_sb,
        z_ss,
        m_sb,
        m_ss,
        l_sb,
        l_sn,
        gb_sb,
        gb_sn,
        gm_sb,
        gm_sn,
        gp_sb,
        gp_sn,
        gl_sb,
        gl_sn,
        gd_sb,
        gd_sn,
        gd_sk,
        K: tl.constexpr,
        T: tl.constexpr,
        NW: tl.constexpr,
        NEED_R: tl.constexpr,
        NEED_N: tl.constexpr,
        NEED_D: tl.constexpr,
        HAS_GBIAS: tl.constexpr,
        LIN_BF16: tl.constexpr,
        LOGIT_BF16: tl.constexpr,
        F64: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """The backward of one tile of ``BLOCK_N`` points by every slice of one case
        (grid ``(n_tiles, B)``). It rebuilds the forward tile (:func:`_region_tile`),
        then runs back through the pooling, the softmax over slices, the routing Linear
        and the invariants. It writes the pre-logits' gradient; the per-point gradients
        of ``r``, ``n_hat`` and the per-point vectors ``d``, each only when its
        ``NEED_*`` flag is set; and per tile the partial sums over its points of the
        gradients of the anchors ``z`` and their mean normals ``m`` (``gz_part``,
        ``gm_part``), of the Linear's weights and of its bias, which the launcher sums.

        The invariants by their column in the Linear's weights and in ``pooled``, with
        ``rel = r - z_s``, ``rel_hat = rel / max(|rel|, eps)`` and the ``K`` global
        directions first among the ``T`` per-point vectors ``d_j``:

        - 0: ``max(|rel|, eps)`` (``g0``); 1: its log (``g1``);
        - 2 to 1 + K: ``rel_hat . d_j`` of the global directions (``g2``, then
          ``g2b``, ``g2c`` and ``g2d`` at ``IDX_E``, ``IDX_F`` and ``IDX_G``);
        - 2 + K: ``rel_hat . n`` (``g3``); 3 + K: ``rel_hat . m_s`` (``g4``);
          4 + K: ``n . m_s`` (``g5``);
        - 5 + K onwards: ``rel_hat . d_j`` of the token's rank-1 inputs (the later
          ones of ``g2b`` to ``g2d``).

        An invariant added to ``_blocks._relational_invariants`` (and
        ``_blocks._geo_width``) goes in the same column into :func:`_region_tile` (its
        value and its weight), both kernels' stores of ``pooled`` and of the weight
        gradient, this kernel's ``gmix`` and ``gg`` terms and its chain rule into
        ``rel_hat``, ``n_hat``, ``m`` or ``d``, and :func:`_check`.
        """
        pid_n = tl.program_id(0).to(tl.int64)  # int64 offsets, as in the forward
        pid_b = tl.program_id(1).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_s = tl.arange(0, BLOCK_S)
        mask_n = offs_n < N
        mask_s = offs_s < S
        mask = mask_n[:, None] & mask_s[None, :]
        # invariant / weight indices of the later per-point vectors, as in the forward
        IDX_E: tl.constexpr = 3 if K >= 2 else 6
        IDX_F: tl.constexpr = 4 if K >= 3 else 7
        IDX_G: tl.constexpr = 5 if K >= 4 else 8

        # ---- the forward tile (_region_tile, as _geo_fwd_kernel)
        (
            n0, n1, n2, d0, d1, d2, e0, e1, e2, f0, f1, f2, i0, i1, i2, m0, m1, m2, rel0, rel1, rel2, dist_raw, dist,
            h0, h1, h2, q0, q1, q2, q2b, q2c, q2d, q3, q4, q5, w0, w1, w2, w2b, w2c, w2d, w3, w4, w5, bias, mix,
        ) = _region_tile(
            r_ptr, n_ptr, d_ptr, z_ptr, m_ptr, logit_ptr, w_ptr, b_ptr, pid_b, offs_n, offs_s, mask_n, mask_s, mask, eps,
            r_sb, r_sn, n_sb, n_sn, d_sb, d_sn, d_sk, z_sb, z_ss, m_sb, m_ss, l_sb, l_sn,
            K, T, IDX_E, IDX_F, IDX_G, LIN_BF16, LOGIT_BF16, F64,
        )

        # ---- incoming gradients (a caller that drops the routing bias sends no gradient for it)
        if HAS_GBIAS:
            gbias_in = tl.load(
                gbias_ptr + pid_b * gb_sb + offs_n[:, None] * gb_sn + offs_s[None, :],
                mask=mask,
                other=0.0,
            ).to(bias.dtype)
        gmix_in = tl.load(
            gmix_ptr + pid_b * gm_sb + offs_n[:, None] * gm_sn + offs_s[None, :],
            mask=mask,
            other=0.0,
        ).to(bias.dtype)
        gp = gpooled_ptr + pid_b * gp_sb + offs_n * gp_sn
        gp0 = tl.load(gp + 0, mask=mask_n, other=0.0).to(bias.dtype)
        gp1 = tl.load(gp + 1, mask=mask_n, other=0.0).to(bias.dtype)
        gp2 = tl.load(gp + 2, mask=mask_n, other=0.0).to(bias.dtype)
        gp3 = tl.load(gp + 2 + K, mask=mask_n, other=0.0).to(bias.dtype)
        gp4 = tl.load(gp + 3 + K, mask=mask_n, other=0.0).to(bias.dtype)
        gp5 = tl.load(gp + 4 + K, mask=mask_n, other=0.0).to(bias.dtype)
        if T >= 2:
            gp2b = tl.load(gp + IDX_E, mask=mask_n, other=0.0).to(bias.dtype)
        if T >= 3:
            gp2c = tl.load(gp + IDX_F, mask=mask_n, other=0.0).to(bias.dtype)
        if T >= 4:
            gp2d = tl.load(gp + IDX_G, mask=mask_n, other=0.0).to(bias.dtype)
        # pooled = sum_s mix_s geo_s  ->  d/dmix_s = sum_g gp_g geo_gs ; d/dgeo_gs = mix_s gp_g
        gmix = (
            gmix_in
            + gp0[:, None] * q0
            + gp1[:, None] * q1
            + gp2[:, None] * q2
            + gp3[:, None] * q3
            + gp4[:, None] * q4
            + gp5[:, None] * q5
        )
        if T >= 2:
            gmix = gmix + gp2b[:, None] * q2b
        if T >= 3:
            gmix = gmix + gp2c[:, None] * q2c
        if T >= 4:
            gmix = gmix + gp2d[:, None] * q2d
        # softmax backward over slices
        dot = tl.sum(gmix * mix, axis=1)
        glogit = mix * (gmix - dot[:, None])
        glogit = tl.where(mask, glogit, 0.0)
        tl.store(
            glogit_ptr + pid_b * gl_sb + offs_n[:, None] * gl_sn + offs_s[None, :],
            glogit.to(glogit_ptr.dtype.element_ty),
            mask=mask,
        )
        gb = glogit  # total gradient on the routing bias
        if HAS_GBIAS:
            gb = glogit + gbias_in
        tile = pid_b * n_tiles + pid_n
        gw = gw_part_ptr + tile * NW
        tl.store(gb_part_ptr + tile, tl.sum(tl.sum(gb, axis=1), axis=0))
        # Linear backward: d/dgeo_g = gb w_g ; d/dw_g = sum gb geo_g ; d/db = sum gb
        gq0 = gb * w0
        gq1 = gb * w1
        gq2 = gb * w2
        gq3 = gb * w3
        gq4 = gb * w4
        gq5 = gb * w5
        gq2b = gq2
        gq2c = gq2
        gq2d = gq2
        if T >= 2:
            gq2b = gb * w2b
        if T >= 3:
            gq2c = gb * w2c
        if T >= 4:
            gq2d = gb * w2d
        tl.store(gw + 0, tl.sum(tl.sum(gb * q0, axis=1), axis=0))
        tl.store(gw + 1, tl.sum(tl.sum(gb * q1, axis=1), axis=0))
        tl.store(gw + 2, tl.sum(tl.sum(gb * q2, axis=1), axis=0))
        if T >= 2:
            tl.store(gw + IDX_E, tl.sum(tl.sum(gb * q2b, axis=1), axis=0))
        if T >= 3:
            tl.store(gw + IDX_F, tl.sum(tl.sum(gb * q2c, axis=1), axis=0))
        if T >= 4:
            tl.store(gw + IDX_G, tl.sum(tl.sum(gb * q2d, axis=1), axis=0))
        tl.store(gw + 2 + K, tl.sum(tl.sum(gb * q3, axis=1), axis=0))
        tl.store(gw + 3 + K, tl.sum(tl.sum(gb * q4, axis=1), axis=0))
        tl.store(gw + 4 + K, tl.sum(tl.sum(gb * q5, axis=1), axis=0))
        gg0 = gq0 + mix * gp0[:, None]
        gg1 = gq1 + mix * gp1[:, None]
        gg2 = gq2 + mix * gp2[:, None]
        gg3 = gq3 + mix * gp3[:, None]
        gg4 = gq4 + mix * gp4[:, None]
        gg5 = gq5 + mix * gp5[:, None]
        if T >= 2:
            gg2b = gq2b + mix * gp2b[:, None]
        if T >= 3:
            gg2c = gq2c + mix * gp2c[:, None]
        if T >= 4:
            gg2d = gq2d + mix * gp2d[:, None]
        # ---- invariants backward
        # v = d(loss)/d(rel_hat)
        v0 = gg2 * d0[:, None] + gg3 * n0[:, None] + gg4 * m0[None, :]
        v1 = gg2 * d1[:, None] + gg3 * n1[:, None] + gg4 * m1[None, :]
        v2 = gg2 * d2[:, None] + gg3 * n2[:, None] + gg4 * m2[None, :]
        if T >= 2:
            v0 = v0 + gg2b * e0[:, None]
            v1 = v1 + gg2b * e1[:, None]
            v2 = v2 + gg2b * e2[:, None]
        if T >= 3:
            v0 = v0 + gg2c * f0[:, None]
            v1 = v1 + gg2c * f1[:, None]
            v2 = v2 + gg2c * f2[:, None]
        if T >= 4:
            v0 = v0 + gg2d * i0[:, None]
            v1 = v1 + gg2d * i1[:, None]
            v2 = v2 + gg2d * i2[:, None]
        vdoth = v0 * h0 + v1 * h1 + v2 * h2
        # rel_hat = rel / dist: numerator -> v/dist ; denominator -> -(v.rel)/dist^2 = -vdoth/dist
        gdist = gg0 + _div(gg1, dist, F64) - _div(vdoth, dist, F64)
        # dist = clamp_min(|rel|, eps): gradient passes where |rel| >= eps; |rel| backward = rel/|rel|
        ok = dist_raw >= eps
        safe = tl.where(dist_raw > 0, dist_raw, 1.0)
        gscale = tl.where(ok, _div(gdist, safe, F64), 0.0)
        grel0 = _div(v0, dist, F64) + gscale * rel0
        grel1 = _div(v1, dist, F64) + gscale * rel1
        grel2 = _div(v2, dist, F64) + gscale * rel2
        grel0 = tl.where(mask, grel0, 0.0)
        grel1 = tl.where(mask, grel1, 0.0)
        grel2 = tl.where(mask, grel2, 0.0)
        ### The per-point gradients (r, n_hat, d) are written only where autograd asks for them.
        if NEED_R:
            gr = gr_ptr + pid_b * r_sb + offs_n * r_sn
            tl.store(gr + 0, tl.sum(grel0, axis=1), mask=mask_n)
            tl.store(gr + 1, tl.sum(grel1, axis=1), mask=mask_n)
            tl.store(gr + 2, tl.sum(grel2, axis=1), mask=mask_n)
        gz_part = gz_part_ptr + tile * S * 3 + offs_s * 3
        gz0 = -tl.sum(grel0, axis=0)
        gz1 = -tl.sum(grel1, axis=0)
        gz2 = -tl.sum(grel2, axis=0)
        # n_hat: g3 = rel_hat.n ; g5 = n.m
        if NEED_N:
            gn = gn_ptr + pid_b * n_sb + offs_n * n_sn
            tl.store(
                gn + 0,
                tl.sum(tl.where(mask, gg3 * h0 + gg5 * m0[None, :], 0.0), axis=1),
                mask=mask_n,
            )
            tl.store(
                gn + 1,
                tl.sum(tl.where(mask, gg3 * h1 + gg5 * m1[None, :], 0.0), axis=1),
                mask=mask_n,
            )
            tl.store(
                gn + 2,
                tl.sum(tl.where(mask, gg3 * h2 + gg5 * m2[None, :], 0.0), axis=1),
                mask=mask_n,
            )
        # m_s: g4 = rel_hat.m ; g5 = n.m
        gm_part = gm_part_ptr + tile * S * 3 + offs_s * 3
        tl.store(
            gm_part + 0,
            tl.sum(tl.where(mask, gg4 * h0 + gg5 * n0[:, None], 0.0), axis=0),
            mask=mask_s,
        )
        tl.store(
            gm_part + 1,
            tl.sum(tl.where(mask, gg4 * h1 + gg5 * n1[:, None], 0.0), axis=0),
            mask=mask_s,
        )
        tl.store(
            gm_part + 2,
            tl.sum(tl.where(mask, gg4 * h2 + gg5 * n2[:, None], 0.0), axis=0),
            mask=mask_s,
        )
        tl.store(gz_part + 0, gz0, mask=mask_s)
        tl.store(gz_part + 1, gz1, mask=mask_s)
        tl.store(gz_part + 2, gz2, mask=mask_s)
        # d: g2 = rel_hat.d
        if NEED_D:
            gd0 = gg2 * h0
            gd1 = gg2 * h1
            gd2 = gg2 * h2
            gd = gd_ptr + pid_b * gd_sb + offs_n * gd_sn  # grad_d is (B, N, T, 3) contiguous
            tl.store(gd + 0, tl.sum(tl.where(mask, gd0, 0.0), axis=1), mask=mask_n)
            tl.store(gd + 1, tl.sum(tl.where(mask, gd1, 0.0), axis=1), mask=mask_n)
            tl.store(gd + 2, tl.sum(tl.where(mask, gd2, 0.0), axis=1), mask=mask_n)
            if T >= 2:
                tl.store(gd + gd_sk + 0, tl.sum(tl.where(mask, gg2b * h0, 0.0), axis=1), mask=mask_n)
                tl.store(gd + gd_sk + 1, tl.sum(tl.where(mask, gg2b * h1, 0.0), axis=1), mask=mask_n)
                tl.store(gd + gd_sk + 2, tl.sum(tl.where(mask, gg2b * h2, 0.0), axis=1), mask=mask_n)
            if T >= 3:
                tl.store(gd + 2 * gd_sk + 0, tl.sum(tl.where(mask, gg2c * h0, 0.0), axis=1), mask=mask_n)
                tl.store(gd + 2 * gd_sk + 1, tl.sum(tl.where(mask, gg2c * h1, 0.0), axis=1), mask=mask_n)
                tl.store(gd + 2 * gd_sk + 2, tl.sum(tl.where(mask, gg2c * h2, 0.0), axis=1), mask=mask_n)
            if T >= 4:
                tl.store(gd + 3 * gd_sk + 0, tl.sum(tl.where(mask, gg2d * h0, 0.0), axis=1), mask=mask_n)
                tl.store(gd + 3 * gd_sk + 1, tl.sum(tl.where(mask, gg2d * h1, 0.0), axis=1), mask=mask_n)
                tl.store(gd + 3 * gd_sk + 2, tl.sum(tl.where(mask, gg2d * h2, 0.0), axis=1), mask=mask_n)


def _check(r, n_hat, d_hat, z_pos, m_s, logits_pre, weight, bias, n_global):
    b, n, _ = r.shape
    s = z_pos.shape[1]
    if not (r.is_cuda and r.dtype in (torch.float32, torch.float64)):
        raise ValueError("fused geo region needs CUDA float32/float64 geometry")
    if s > MAX_SLICES:
        raise ValueError(
            f"fused geo region supports at most {MAX_SLICES} slices, got {s}"
        )
    if logits_pre.shape != (b, n, s) or logits_pre.stride(-1) != 1:
        raise ValueError(
            "logits_pre must be (B, N, S) with a contiguous last dimension"
        )
    if n_hat.shape != (b, n, 3) or n_hat.stride(-1) != 1:
        raise ValueError("r, n_hat must be (B, N, 3) with unit last stride")
    if d_hat.ndim != 4 or d_hat.shape[:2] != (b, n) or d_hat.shape[-1] != 3 or d_hat.stride(-1) != 1:
        raise ValueError("d_hat must be (B, N, K, 3) with unit last stride")
    t = d_hat.shape[2]
    if not 1 <= t <= MAX_DIRECTIONS:
        raise ValueError(
            f"fused geo region supports 1 to {MAX_DIRECTIONS} per-point vectors (global directions plus token vectors), got {t}"
        )
    if not 1 <= n_global <= t:
        raise ValueError(f"n_global must be between 1 and the vector count {t}, got {n_global}")
    if (
        r.stride(-1) != 1
        or m_s.shape != (b, s, 3)
        or z_pos.shape != (b, s, 3)
        or z_pos.stride(-1) != 1
        or m_s.stride(-1) != 1
    ):
        raise ValueError("geometry tensors must have unit last stride")
    if bias.numel() != 1:
        raise ValueError("the routing bias head must end in one scalar bias")
    if weight.shape[-1] != _geo_width(t):
        raise ValueError(f"routing Linear must be {_geo_width(t)}->1 for {t} vectors")


#: Launch overrides: points per forward and per backward program, and the backward's
#: warps. ``None`` picks the defaults below, tuned on Ada-class GPUs (compute capability
#: below 10) and Blackwell-class GPUs (10 and above); re-tune them when the kernel or
#: Triton changes. The tile size and the warp count change only the floating-point
#: summation order (pointwise outputs agree to about 1e-5 relative). On Blackwell at up
#: to 128 slices the backward runs 1.4 to 1.9 times faster on one warp than on four,
#: since its reductions then never cross warps; more slices keep four.
FWD_BLOCK_N: int | None = None
BWD_BLOCK_N: int | None = None
BWD_NUM_WARPS: int | None = None
NUM_WARPS: int = 4


def _blackwell() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
    except Exception:  # pragma: no cover - no device
        return False


def _fwd_block(s, k=1):
    """Points per forward program; halved for K > 1, whose unrolled tiles hold K
    extra (BLOCK_N, S) invariants (the K = 1 defaults spill registers there)."""
    if FWD_BLOCK_N is not None:
        return FWD_BLOCK_N
    if _blackwell():
        base = 4 if s <= 256 else 2 if s <= 512 else 1
    else:
        base = 8 if s <= 256 else 4 if s <= 512 else 2
    return base if k == 1 else max(1, base // 2)


def _bwd_block(s, k=1):
    if BWD_BLOCK_N is not None:
        return BWD_BLOCK_N
    if _blackwell():
        base = 2 if s <= 256 else 1
    else:
        base = 4 if s <= 256 else 2 if s <= 512 else 1
    return base if k == 1 else max(1, base // 2)


def _bwd_warps(s):
    if BWD_NUM_WARPS is not None:
        return BWD_NUM_WARPS
    return 1 if _blackwell() and s <= 128 else NUM_WARPS


@torch.library.custom_op("physicsnemo::isla_geo_region_fwd", mutates_args=())
def _geo_region_fwd(
    logits_pre: torch.Tensor,
    r: torch.Tensor,
    n_hat: torch.Tensor,
    d_hat: torch.Tensor,
    z_pos: torch.Tensor,
    m_s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    lin_bf16: bool,
    n_global: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(r, n_hat, d_hat, z_pos, m_s, logits_pre, weight, bias, n_global)
    b, n, s = logits_pre.shape
    t, k = d_hat.shape[2], n_global
    ng = _geo_width(t)
    out_dt = torch.bfloat16 if lin_bf16 else r.dtype
    bias_out = torch.empty(b, n, s, device=r.device, dtype=out_dt)
    mix_out = torch.empty(b, n, s, device=r.device, dtype=r.dtype)
    pooled = torch.empty(b, n, ng, device=r.device, dtype=out_dt)
    w = weight.reshape(-1).to(r.dtype).contiguous()
    bb = bias.reshape(-1).to(r.dtype).contiguous()
    block_n, block_s = _fwd_block(s, k), triton.next_power_of_2(max(s, 1))
    grid = (triton.cdiv(n, block_n), b)
    _geo_fwd_kernel[grid](
        r,
        n_hat,
        d_hat,
        z_pos,
        m_s,
        logits_pre,
        w,
        bb,
        bias_out,
        mix_out,
        pooled,
        n,
        s,
        float(eps),
        r.stride(0),
        r.stride(1),
        n_hat.stride(0),
        n_hat.stride(1),
        d_hat.stride(0),
        d_hat.stride(1),
        d_hat.stride(2),
        z_pos.stride(0),
        z_pos.stride(1),
        m_s.stride(0),
        m_s.stride(1),
        logits_pre.stride(0),
        logits_pre.stride(1),
        bias_out.stride(0),
        bias_out.stride(1),
        mix_out.stride(0),
        mix_out.stride(1),
        pooled.stride(0),
        pooled.stride(1),
        K=k,
        T=t,
        LIN_BF16=lin_bf16,
        LOGIT_BF16=logits_pre.dtype == torch.bfloat16,
        F64=r.dtype == torch.float64,
        BLOCK_N=block_n,
        BLOCK_S=block_s,
        num_warps=NUM_WARPS,
    )
    return bias_out, mix_out, pooled


@_geo_region_fwd.register_fake
def _(logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias, eps, lin_bf16, n_global):
    b, n, s = logits_pre.shape
    out_dt = torch.bfloat16 if lin_bf16 else r.dtype
    return (
        torch.empty(b, n, s, device=r.device, dtype=out_dt),
        torch.empty(b, n, s, device=r.device, dtype=r.dtype),
        torch.empty(b, n, _geo_width(d_hat.shape[2]), device=r.device, dtype=out_dt),
    )


@torch.library.custom_op("physicsnemo::isla_geo_region_bwd", mutates_args=())
def _geo_region_bwd(
    logits_pre: torch.Tensor,
    r: torch.Tensor,
    n_hat: torch.Tensor,
    d_hat: torch.Tensor,
    z_pos: torch.Tensor,
    m_s: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_bias: torch.Tensor | None,
    grad_mix: torch.Tensor,
    grad_pooled: torch.Tensor,
    eps: float,
    lin_bf16: bool,
    n_global: int,
    need_r: bool = True,
    need_n: bool = True,
    need_d: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    b, n, s = logits_pre.shape
    t, k = d_hat.shape[2], n_global
    nw = weight.numel()
    ### The kernel reads r and n_hat with the strides it writes gr and gn with.
    r, n_hat = r.contiguous(), n_hat.contiguous()
    has_gbias = grad_bias is not None
    grad_bias = grad_bias.contiguous() if has_gbias else grad_mix.new_empty(1, 1, 0)
    grad_mix = grad_mix.contiguous()
    grad_pooled = grad_pooled.contiguous()
    block_n, block_s = _bwd_block(s, k), triton.next_power_of_2(max(s, 1))
    n_tiles = triton.cdiv(n, block_n)
    glogit = torch.empty_like(logits_pre)
    ### A per-point gradient autograd does not ask for is neither computed nor stored: it comes back with no points.
    gr = torch.empty(b, n if need_r else 0, 3, device=r.device, dtype=r.dtype)
    gn = torch.empty(b, n if need_n else 0, 3, device=r.device, dtype=r.dtype)
    gd = torch.empty(b, n if need_d else 0, t, 3, device=r.device, dtype=r.dtype)
    gz_part = torch.empty(b, n_tiles, s, 3, device=r.device, dtype=r.dtype)
    gm_part = torch.empty(b, n_tiles, s, 3, device=r.device, dtype=r.dtype)
    gw_part = torch.empty(b, n_tiles, nw, device=r.device, dtype=r.dtype)
    gb_part = torch.empty(b, n_tiles, device=r.device, dtype=r.dtype)
    w = weight.reshape(-1).to(r.dtype).contiguous()
    bb = bias.reshape(-1).to(r.dtype).contiguous()
    _geo_bwd_kernel[(n_tiles, b)](
        r,
        n_hat,
        d_hat,
        z_pos,
        m_s,
        logits_pre,
        w,
        bb,
        grad_bias,
        grad_mix,
        grad_pooled,
        glogit,
        gr,
        gn,
        gd,
        gz_part,
        gm_part,
        gw_part,
        gb_part,
        n,
        s,
        float(eps),
        n_tiles,
        r.stride(0),
        r.stride(1),
        n_hat.stride(0),
        n_hat.stride(1),
        d_hat.stride(0),
        d_hat.stride(1),
        d_hat.stride(2),
        z_pos.stride(0),
        z_pos.stride(1),
        m_s.stride(0),
        m_s.stride(1),
        logits_pre.stride(0),
        logits_pre.stride(1),
        grad_bias.stride(0),
        grad_bias.stride(1),
        grad_mix.stride(0),
        grad_mix.stride(1),
        grad_pooled.stride(0),
        grad_pooled.stride(1),
        glogit.stride(0),
        glogit.stride(1),
        gd.stride(0),
        gd.stride(1),
        gd.stride(2),
        K=k,
        T=t,
        NW=nw,
        NEED_R=need_r,
        NEED_N=need_n,
        NEED_D=need_d,
        HAS_GBIAS=has_gbias,
        LIN_BF16=lin_bf16,
        LOGIT_BF16=logits_pre.dtype == torch.bfloat16,
        F64=r.dtype == torch.float64,
        BLOCK_N=block_n,
        BLOCK_S=block_s,
        num_warps=_bwd_warps(s),
    )
    gz = gz_part.sum(dim=1)
    gm = gm_part.sum(dim=1)
    gw = gw_part.sum(dim=(0, 1)).reshape(weight.shape).to(weight.dtype)
    gbias = gb_part.sum().reshape(bias.shape).to(bias.dtype)
    return glogit, gr, gn, gd, gz, gm, gw, gbias


@_geo_region_bwd.register_fake
def _(
    logits_pre,
    r,
    n_hat,
    d_hat,
    z_pos,
    m_s,
    weight,
    bias,
    grad_bias,
    grad_mix,
    grad_pooled,
    eps,
    lin_bf16,
    n_global,
    need_r=True,
    need_n=True,
    need_d=True,
):
    b, n, s = logits_pre.shape
    e = lambda *shape, dt=r.dtype: torch.empty(*shape, device=r.device, dtype=dt)  # noqa: E731
    return (
        torch.empty_like(logits_pre),
        e(b, n if need_r else 0, 3),
        e(b, n if need_n else 0, 3),
        e(b, n if need_d else 0, d_hat.shape[2], 3),
        e(b, s, 3),
        e(b, s, 3),
        e(*weight.shape, dt=weight.dtype),
        e(*bias.shape, dt=bias.dtype),
    )


def _setup_context(ctx, inputs, output):
    logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias, eps, lin_bf16, n_global = inputs
    ctx.save_for_backward(logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias)
    ### An output the caller drops (the routing bias in a passive read) sends None, not a zero tensor.
    ctx.set_materialize_grads(False)
    ctx.eps, ctx.lin_bf16, ctx.n_global = eps, lin_bf16, n_global


def _backward(ctx, grad_bias, grad_mix, grad_pooled):
    logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias = ctx.saved_tensors
    if grad_mix is None:
        grad_mix = torch.zeros(logits_pre.shape, device=r.device, dtype=r.dtype)
    if grad_pooled is None:
        grad_pooled = torch.zeros(
            logits_pre.shape[0],
            logits_pre.shape[1],
            _geo_width(d_hat.shape[2]),
            device=r.device,
            dtype=r.dtype,
        )
    ### ISLA never differentiates the positions and normals, and a passive point's vectors carry no gradient.
    need_r, need_n, need_d = ctx.needs_input_grad[1:4]
    glogit, gr, gn, gd, gz, gm, gw, gb = _geo_region_bwd(
        logits_pre,
        r,
        n_hat,
        d_hat,
        z_pos,
        m_s,
        weight,
        bias,
        grad_bias,
        grad_mix,
        grad_pooled,
        ctx.eps,
        ctx.lin_bf16,
        ctx.n_global,
        need_r,
        need_n,
        need_d,
    )
    gr, gn, gd = gr if need_r else None, gn if need_n else None, gd if need_d else None
    return glogit, gr, gn, gd, gz, gm, gw, gb, None, None, None


_geo_region_fwd.register_autograd(_backward, setup_context=_setup_context)


def fused_geo_region(
    lin: torch.nn.Module,
    logits_pre,
    r,
    n_hat,
    d_hat,
    z_pos,
    m_s,
    eps: float,
    n_global: int | None = None,
):
    """Drop-in for ``_blocks._geo_region``: (bias (B,N,S), mix (B,N,S), pooled
    (B,N,geo)) from one fused kernel.

    ``lin`` is the routing-bias head, ``nn.Linear(n_geo, 1)``; a missing bias enters
    as an exact zero. ``d_hat`` (B, N, T, d), d at most 3, holds the per-point unit
    vectors: the ``n_global`` global directions first (K, possibly none; default T,
    every vector a global direction), then the token's rank-1 vectors (P' = T - K).
    T is at most :data:`MAX_DIRECTIONS`, with K = 0 counted as one. The invariants
    come in the eager region's order (the K direction dots right after log distance,
    the token-vector dots after n.m), so the routing Linear's weights mean the same on
    both paths.

    The kernels take three components and at least one direction. A problem in fewer
    dimensions enters with zero trailing components, and K = 0 as one zero direction
    with a zero routing weight. Both add exact zeros to every norm, dot product and
    bias sum, so the arithmetic is the unpadded region's; the padded parts of the
    outputs and gradients are dropped."""
    if not HAS_TRITON:
        raise RuntimeError("fused geo region requires triton")
    if not isinstance(lin, torch.nn.Linear):
        raise ValueError("fused geo region takes the linear routing-bias head, nn.Linear(n_geo, 1)")
    weight = lin.weight
    bias = lin.bias if lin.bias is not None else weight.new_zeros(1)
    lin_bf16 = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    if torch.is_autocast_enabled("cuda") and not lin_bf16:
        raise RuntimeError("fused geo region supports autocast in bfloat16 only")
    k = int(d_hat.shape[2] if n_global is None else n_global)
    if k == 0:
        d_hat = torch.cat([d_hat.new_zeros(*d_hat.shape[:2], 1, d_hat.shape[3]), d_hat], dim=2)
        weight = torch.cat([weight[:, :2], weight.new_zeros(1, 1), weight[:, 2:]], dim=1)
    if r.shape[-1] < 3:
        pad = (0, 3 - r.shape[-1])
        r, n_hat, d_hat, z_pos, m_s = (F.pad(t, pad) for t in (r, n_hat, d_hat, z_pos, m_s))
    bias_out, mix, pooled = _geo_region_fwd(
        logits_pre, r, n_hat, d_hat, z_pos, m_s, weight, bias, float(eps), bool(lin_bf16), max(k, 1)
    )
    if k == 0:
        pooled = torch.cat([pooled[..., :2], pooled[..., 3:]], dim=-1)
    return bias_out, mix, pooled
