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

r"""The tree read with its kernel term fused in (Triton). Between forward and backward
it keeps nothing per pair but the kernel basis and the plan.

The reference read (:mod:`segment_softmax` behind
:class:`~physicsnemo.experimental.models.isla._tree_read._TreeKernelRead`) takes each
block's kernel term :math:`\kappa_{ph} = \sum_f \phi_{pf} W_{hf}` as a ``(P, H)``
tensor: a matmul of the shared basis :math:`\phi` ``(P, F)`` with the block's kernel
weights. Autograd keeps that tensor for backward in every block (2e8 to 7e8 pairs at
DrivAerML scale). Here the read's kernels compute :math:`\kappa` per tile from the
basis and the weights, in forward and again in backward (as flash attention
recomputes its logits), so a block keeps only its per-point output and log-sum-exp.

The by-point backward kernel recomputes each pair's softmax weight and logit gradient
once, and writes them (float32) in the plan's by-source order. The by-source kernel
sums them against the points' queries and output gradients into the keys' and values'
gradients, one chunk of a source's pairs at a time, and a source's chunks are then
summed in order. It uses no atomics, so the backward is deterministic.

The arithmetic is the reference's but for summation order:

- :math:`\kappa` is a sum of exact products in float32, rounded to the basis's dtype
  (bfloat16 under autocast, as the reference matmul's output is);
- its gradient is rounded to that dtype before it reaches the basis and the weights,
  as the reference read's kernel-term gradient is;
- each block's basis gradient is rounded to that dtype, as in the reference matmul's
  backward;
- the weights' gradient is a float32 sum over every pair, rounded once.

The blocks' basis gradients are summed in place in one buffer, in the order and at the
rounding points of autograd's own sum over the basis's uses: each block's contribution
is rounded, added to the running sum, and rounded again. The sum reaches the basis
once, after the last read's backward (:func:`basis_sink`), and no block's gradient is
materialized beside it. A backward pass that will not reach the basis
(``torch.autograd.grad`` of other inputs, ``backward(inputs=...)``) adds nothing to it.

The small products are float32 broadcast sums. Triton turns a sum over the middle
axis, ``(M, K, 1) * (1, K, N)`` summed over ``K``, into a ``tl.dot``. With bfloat16 or
float16 operands its products are exact and it accumulates in float32 on the tensor
cores, as the reference's bfloat16 GEMM does; the basis gradient under autocast takes
that form, the fastest. With float32 operands the dot is TF32, 1e-3 off, so there the
basis gradient is summed head by head instead.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from physicsnemo.experimental.models.isla._triton import HAS_TRITON, tl, triton
from physicsnemo.experimental.models.isla._segment_softmax import PairPlan, _source_items

#: Pairs per tile of a program's loop, warps per program, and pairs per program of
#: the by-source backward. Launch constants tuned on Blackwell-class GPUs: the outputs
#: depend on them at most through summation order. Re-tune them when the kernels or
#: Triton change.
BLOCK_PAIRS = 32
NUM_WARPS = 1
SOURCE_CHUNK = 512

#: Registers per thread of the forward (under a narrow and under a float32 basis) and of
#: the by-point and by-source backward kernels (under a narrow basis; a float32 one runs
#: them uncapped). Launch constants tuned on Blackwell-class GPUs: these one-warp kernels
#: are latency-bound, and a cap lets more programs reside per multiprocessor. The outputs
#: do not depend on them. Re-tune them when the kernels or Triton change. CUDA only
#: (ROCm's Triton has no maxnreg).
FORWARD_MAXNREG = 96
FORWARD_MAXNREG_FLOAT32 = 128
POINT_MAXNREG = 128
SOURCE_MAXNREG = 96


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def supported(basis: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether the fused read applies: CUDA tensors, Triton importable, no float64 (the
    reference read runs float64 eagerly), and power-of-two widths."""
    h, kc, c, f = q.shape[1], q.shape[2], v.shape[2], basis.shape[1]
    return (
        HAS_TRITON
        and all(t.is_cuda for t in (basis, q, k, v))
        and torch.float64 not in (basis.dtype, q.dtype, k.dtype, v.dtype)
        and all(_pow2(n) for n in (h, kc, c, f, h * kc, h * c))
    )


if HAS_TRITON:

    @triton.jit
    def _weights(w_ptr, H: tl.constexpr, F: tl.constexpr):
        """The kernel weights ``(H, F)`` in float32."""
        return tl.load(w_ptr + tl.arange(0, H)[:, None] * F + tl.arange(0, F)[None, :]).to(tl.float32)

    @triton.jit
    def _weights_padded(w_ptr, H: tl.constexpr, F: tl.constexpr):
        """The kernel weights in their own dtype as a ``(2H, F)`` matrix, row ``2h`` the
        head ``h`` and every odd row zero (``tl.dot`` takes no dimension below 16)."""
        rr = tl.arange(0, 2 * H)
        ff = tl.arange(0, F)
        return tl.load(w_ptr + (rr // 2)[:, None] * F + ff[None, :], mask=(rr % 2 == 0)[:, None], other=0.0)

    @triton.jit
    def _kernel_term(
        basis_ptr, p, msk, w, wp, H: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr, NARROW: tl.constexpr
    ):
        """The pairs' basis rows ``(BLOCK, F)`` (in the basis's dtype when NARROW, else
        float32) and kernel terms ``(BLOCK, H)``, rounded to the basis's dtype and held
        in float32."""
        ff = tl.arange(0, F)
        bas = tl.load(basis_ptr + p[:, None] * F + ff[None, :], mask=msk[:, None], other=0.0)
        if NARROW:
            ker2 = tl.dot(bas, tl.trans(wp))  # (BLOCK, 2H), the heads at the even columns
            ker, _ = tl.split(tl.reshape(ker2, [BLOCK, H, 2]))
        else:
            bas = bas.to(tl.float32)
            ker = tl.sum(bas[:, None, :] * w[None, :, :], axis=2)
        return bas, ker.to(basis_ptr.dtype.element_ty).to(tl.float32)

    @triton.jit
    def _fkr_fwd_kernel(
        off_ptr, bj_ptr, basis_ptr, w_ptr, logm_ptr, q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
        scale, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr,
        NARROW: tl.constexpr,
    ):  # fmt: skip
        """One point: online softmax over its pairs; out[s] (H, C), lse[s] (H)."""
        s = tl.program_id(0).to(tl.int64)
        ### int64 pair offsets whatever the plan's dtype: p * F passes 2**31 at
        ### P > 2**27 pairs (F = 16), and PairPlan.build gives int32 offsets for int32 indices.
        start = tl.load(off_ptr + s).to(tl.int64)
        end = tl.load(off_ptr + s + 1).to(tl.int64)
        hh = tl.arange(0, H)
        w = _weights(w_ptr, H, F)
        wp = _weights_padded(w_ptr, H, F)
        q = tl.reshape(tl.load(q_ptr + s * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        m_run = tl.full([H], float("-inf"), tl.float32)
        den = tl.zeros([H], tl.float32)
        acc = tl.zeros([H, C], tl.float32)
        for p0 in range(start, end, BLOCK):
            p = p0 + tl.arange(0, BLOCK)
            msk = p < end
            bj = tl.load(bj_ptr + p, mask=msk, other=0).to(tl.int64)
            _, ker = _kernel_term(basis_ptr, p, msk, w, wp, H, F, BLOCK, NARROW)
            k = tl.reshape(
                tl.load(
                    k_ptr + bj[:, None] * (H * KC) + tl.arange(0, H * KC)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, KC],
            )
            score = tl.sum(k * q[None, :, :], axis=2)  # (BLOCK, H)
            logm = tl.load(logm_ptr + bj, mask=msk, other=0.0).to(tl.float32)
            z = ker + score * scale + logm[:, None]
            z = tl.where(msk[:, None], z, float("-inf"))
            m_new = tl.maximum(m_run, tl.max(z, axis=0))
            alpha = tl.exp(m_run - m_new)
            pe = tl.exp(z - m_new[None, :])
            den = den * alpha + tl.sum(pe, axis=0)
            v = tl.reshape(
                tl.load(
                    v_ptr + bj[:, None] * (H * C) + tl.arange(0, H * C)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, C],
            )
            acc = acc * alpha[:, None] + tl.sum(pe[:, :, None] * v, axis=0)
            m_run = m_new
        safe = tl.where(den > 0, den, 1.0)
        out = acc / safe[:, None]
        tl.store(out_ptr + s * (H * C) + tl.arange(0, H * C), tl.reshape(out, [H * C]))
        tl.store(lse_ptr + s * H + hh, m_run + tl.log(safe))

    @triton.jit
    def _fkr_bwd_point_kernel(
        off_ptr, bj_ptr, basis_ptr, w_ptr, logm_ptr, q_ptr, k_ptr, v_ptr, lse_ptr, dsum_ptr, dout_ptr,
        inv_ptr, dq_ptr, dbasis_ptr, dw_ptr, prdz_ptr,
        scale, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr,
        BASIS_GRAD: tl.constexpr, NARROW: tl.constexpr, FIRST: tl.constexpr,
    ):  # fmt: skip
        """One point: dq[s]; its part of the weights' gradient, dw[s] (2H, F) float32
        with the heads at the even rows; and each pair's softmax weights and logit
        gradients (H each, float32) in the pair's row of prdz, in the plan's by-source
        order (inv: each pair's position there). With BASIS_GRAD, the basis gradient of
        its pairs through this block's weights is added to dbasis (with FIRST, written:
        the sum's first term). NARROW: the basis is bfloat16 or float16."""
        s = tl.program_id(0).to(tl.int64)
        start = tl.load(off_ptr + s).to(tl.int64)  # int64 pair offsets, as in the forward
        end = tl.load(off_ptr + s + 1).to(tl.int64)
        hh = tl.arange(0, H)
        ff = tl.arange(0, F)
        rr = tl.arange(0, 2 * H)
        w = _weights(w_ptr, H, F)
        wp = _weights_padded(w_ptr, H, F)
        q = tl.reshape(tl.load(q_ptr + s * (H * KC) + tl.arange(0, H * KC)).to(tl.float32), [H, KC])
        lse = tl.load(lse_ptr + s * H + hh)
        dsum = tl.load(dsum_ptr + s * H + hh)
        dout = tl.reshape(tl.load(dout_ptr + s * (H * C) + tl.arange(0, H * C)).to(tl.float32), [H, C])
        dq = tl.zeros([H, KC], tl.float32)
        dw = tl.zeros([H, F], tl.float32)
        dw2 = tl.zeros([2 * H, F], tl.float32)
        for p0 in range(start, end, BLOCK):
            p = p0 + tl.arange(0, BLOCK)
            msk = p < end
            bj = tl.load(bj_ptr + p, mask=msk, other=0).to(tl.int64)
            bas, ker = _kernel_term(basis_ptr, p, msk, w, wp, H, F, BLOCK, NARROW)
            k = tl.reshape(
                tl.load(
                    k_ptr + bj[:, None] * (H * KC) + tl.arange(0, H * KC)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, KC],
            )
            score = tl.sum(k * q[None, :, :], axis=2)
            logm = tl.load(logm_ptr + bj, mask=msk, other=0.0).to(tl.float32)
            z = ker + score * scale + logm[:, None]
            pr = tl.exp(z - lse[None, :])
            pr = tl.where(msk[:, None], pr, 0.0)  # (BLOCK, H)
            v = tl.reshape(
                tl.load(
                    v_ptr + bj[:, None] * (H * C) + tl.arange(0, H * C)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, C],
            )
            dp = tl.sum(v * dout[None, :, :], axis=2)  # (BLOCK, H)
            dz = pr * (dp - dsum[None, :])  # (BLOCK, H)
            dq += tl.sum(dz[:, :, None] * k, axis=0) * scale
            row = prdz_ptr + tl.load(inv_ptr + p, mask=msk, other=0).to(tl.int64)[:, None] * (2 * H)
            tl.store(row + hh[None, :], pr, mask=msk[:, None])
            tl.store(row + H + hh[None, :], dz, mask=msk[:, None])
            ### The kernel-term gradient in the basis's dtype; NARROW: as a (BLOCK, 2H)
            ### matrix with the heads at the even columns, for tl.dot.
            dker_lp = dz.to(basis_ptr.dtype.element_ty)
            if NARROW:
                dker2 = tl.reshape(tl.join(dker_lp, tl.zeros_like(dker_lp)), [BLOCK, 2 * H])
            else:
                dker = dker_lp.to(tl.float32)
            if BASIS_GRAD:
                dt = dbasis_ptr.dtype.element_ty
                if NARROW:
                    dbas = tl.dot(dker2, wp)  # (BLOCK, F)
                else:
                    dbas = tl.zeros([BLOCK, F], tl.float32)
                    for h in tl.static_range(H):
                        col = tl.sum(tl.where(hh[None, :] == h, dker, 0.0), axis=1)
                        dbas += col[:, None] * tl.sum(tl.where(hh[:, None] == h, w, 0.0), axis=0)[None, :]
                dbas = dbas.to(dt).to(tl.float32)
                rows = dbasis_ptr + p[:, None] * F + ff[None, :]
                if FIRST:
                    tl.store(rows, dbas.to(dt), mask=msk[:, None])
                else:
                    run = tl.load(rows, mask=msk[:, None], other=0.0).to(tl.float32)
                    tl.store(rows, (run + dbas).to(dt), mask=msk[:, None])
            if NARROW:
                dw2 += tl.dot(tl.trans(dker2), bas)
            else:
                dw += tl.sum(dker[:, :, None] * bas[:, None, :], axis=0)
        tl.store(dq_ptr + s * (H * KC) + tl.arange(0, H * KC), tl.reshape(dq, [H * KC]))
        if NARROW:
            tl.store(dw_ptr + s * (2 * H * F) + rr[:, None] * F + ff[None, :], dw2)
        else:
            tl.store(dw_ptr + s * (2 * H * F) + (2 * hh)[:, None] * F + ff[None, :], dw)

    @triton.jit
    def _fkr_bwd_source_kernel(
        boff_ptr, item_src_ptr, item_chunk_ptr, qib_ptr, prdz_ptr, q_ptr, dout_ptr, part_ptr, lpart_ptr,
        scale, CHUNK, H: tl.constexpr, C: tl.constexpr, KC: tl.constexpr, BLOCK: tl.constexpr,
        LOGM_GRAD: tl.constexpr,
    ):  # fmt: skip
        """One chunk of source j's pairs, in the plan's by-source order: its part of dk[j]
        and dv[j], in its row of part, from the pairs' softmax weights and logit gradients
        (prdz, from the point kernel). With LOGM_GRAD, also its part of dlogm[j], the
        pairs' logit gradients summed over the heads, in its entry of lpart."""
        item = tl.program_id(0).to(tl.int64)
        j = tl.load(item_src_ptr + item).to(tl.int64)
        chunk = tl.load(item_chunk_ptr + item).to(tl.int64)
        start = tl.load(boff_ptr + j) + chunk * CHUNK
        end = tl.minimum(tl.load(boff_ptr + j + 1), start + CHUNK)
        hh = tl.arange(0, H)
        dk = tl.zeros([H, KC], tl.float32)
        dv = tl.zeros([H, C], tl.float32)
        dl = tl.zeros([BLOCK], tl.float32)
        for i0 in range(start, end, BLOCK):
            i = i0 + tl.arange(0, BLOCK)
            msk = i < end
            ### Streamed once: evicted first, so the gathered point rows stay in cache.
            qi = tl.load(qib_ptr + i, mask=msk, other=0, eviction_policy="evict_first").to(tl.int64)
            prdz = prdz_ptr + i[:, None] * (2 * H) + hh[None, :]
            pr = tl.load(prdz, mask=msk[:, None], other=0.0, eviction_policy="evict_first")
            dz = tl.load(prdz + H, mask=msk[:, None], other=0.0, eviction_policy="evict_first")
            q = tl.reshape(
                tl.load(
                    q_ptr + qi[:, None] * (H * KC) + tl.arange(0, H * KC)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, KC],
            )
            dout = tl.reshape(
                tl.load(
                    dout_ptr + qi[:, None] * (H * C) + tl.arange(0, H * C)[None, :],
                    mask=msk[:, None], other=0.0,
                ).to(tl.float32),
                [BLOCK, H, C],
            )
            dv += tl.sum(pr[:, :, None] * dout, axis=0)
            dk += tl.sum(dz[:, :, None] * q, axis=0) * scale
            if LOGM_GRAD:
                dl += tl.sum(dz, axis=1)
        row = part_ptr + item * (H * (KC + C))
        tl.store(row + tl.arange(0, H * KC), tl.reshape(dk, [H * KC]))
        tl.store(row + H * KC + tl.arange(0, H * C), tl.reshape(dv, [H * C]))
        if LOGM_GRAD:
            tl.store(lpart_ptr + item, tl.sum(dl, axis=0))

    @triton.jit
    def _fkr_source_sum_kernel(
        item_off_ptr, part_ptr, lpart_ptr, dk_ptr, dv_ptr, dlogm_ptr, H: tl.constexpr, C: tl.constexpr,
        KC: tl.constexpr, LOGM_GRAD: tl.constexpr,
    ):  # fmt: skip
        """One source: dk[j] and dv[j] (and with LOGM_GRAD dlogm[j]), the sum of its
        items' parts in item order."""
        j = tl.program_id(0).to(tl.int64)
        a = tl.load(item_off_ptr + j).to(tl.int64)
        b = tl.load(item_off_ptr + j + 1).to(tl.int64)
        dk = tl.zeros([H * KC], tl.float32)
        dv = tl.zeros([H * C], tl.float32)
        dl = tl.zeros([1], tl.float32)
        for t in range(a, b):
            row = part_ptr + t * (H * (KC + C))
            dk += tl.load(row + tl.arange(0, H * KC))
            dv += tl.load(row + H * KC + tl.arange(0, H * C))
            if LOGM_GRAD:
                dl += tl.load(lpart_ptr + t + tl.arange(0, 1))
        tl.store(dk_ptr + j * (H * KC) + tl.arange(0, H * KC), dk)
        tl.store(dv_ptr + j * (H * C) + tl.arange(0, H * C), dv)
        if LOGM_GRAD:
            tl.store(dlogm_ptr + j + tl.arange(0, 1), dl)


if HAS_TRITON:

    @triton.jit
    def _nm_leaf_kernel(
        x_ptr, om_ptr, cells_ptr, leaf_ptr, rs_ptr, rc_ptr, sums_ptr, n_leaves, D,
        BLOCK_D: tl.constexpr, NB: tl.constexpr,
    ):  # fmt: skip
        """NB leaves: each the sum of its cells' rows times their measure, in position order."""
        ll = tl.program_id(0).to(tl.int64) * NB + tl.arange(0, NB)
        lm = ll < n_leaves
        node = tl.load(leaf_ptr + ll, mask=lm, other=0).to(tl.int64)
        st = tl.load(rs_ptr + node, mask=lm, other=0).to(tl.int64)
        cnt = tl.load(rc_ptr + node, mask=lm, other=0).to(tl.int64)
        dd = tl.arange(0, BLOCK_D)
        dm = dd < D
        acc = tl.zeros([NB, BLOCK_D], tl.float32)
        for t in range(0, tl.max(cnt, axis=0)):
            tm = lm & (t < cnt)
            cell = tl.load(cells_ptr + st + t, mask=tm, other=0).to(tl.int64)
            xv = tl.load(x_ptr + cell[:, None] * D + dd[None, :], mask=tm[:, None] & dm[None, :], other=0.0)
            om = tl.load(om_ptr + cell, mask=tm, other=0.0).to(tl.float32)
            acc = tl.where(tm[:, None], acc + xv.to(tl.float32) * om[:, None], acc)
        tl.store(sums_ptr + node[:, None] * D + dd[None, :], acc, mask=lm[:, None] & dm[None, :])

    @triton.jit
    def _nm_up_kernel(sums_ptr, nodes_ptr, left_ptr, right_ptr, n, D, BLOCK_D: tl.constexpr, NB: tl.constexpr):
        """One depth's internal nodes: each the sum of its two children."""
        ii = tl.program_id(0).to(tl.int64) * NB + tl.arange(0, NB)
        m = ii < n
        node = tl.load(nodes_ptr + ii, mask=m, other=0).to(tl.int64)
        lc = tl.load(left_ptr + ii, mask=m, other=0).to(tl.int64)
        rc = tl.load(right_ptr + ii, mask=m, other=0).to(tl.int64)
        dd = tl.arange(0, BLOCK_D)
        mm = m[:, None] & (dd < D)[None, :]
        a = tl.load(sums_ptr + lc[:, None] * D + dd[None, :], mask=mm, other=0.0)
        b = tl.load(sums_ptr + rc[:, None] * D + dd[None, :], mask=mm, other=0.0)
        tl.store(sums_ptr + node[:, None] * D + dd[None, :], a + b, mask=mm)

    @triton.jit
    def _nm_down_kernel(g_ptr, nodes_ptr, left_ptr, right_ptr, n, D, BLOCK_D: tl.constexpr, NB: tl.constexpr):
        """One depth's internal nodes: each adds its gradient to its two children's."""
        ii = tl.program_id(0).to(tl.int64) * NB + tl.arange(0, NB)
        m = ii < n
        node = tl.load(nodes_ptr + ii, mask=m, other=0).to(tl.int64)
        lc = tl.load(left_ptr + ii, mask=m, other=0).to(tl.int64)
        rc = tl.load(right_ptr + ii, mask=m, other=0).to(tl.int64)
        dd = tl.arange(0, BLOCK_D)
        mm = m[:, None] & (dd < D)[None, :]
        g = tl.load(g_ptr + node[:, None] * D + dd[None, :], mask=mm, other=0.0)
        gl = tl.load(g_ptr + lc[:, None] * D + dd[None, :], mask=mm, other=0.0)
        gr = tl.load(g_ptr + rc[:, None] * D + dd[None, :], mask=mm, other=0.0)
        tl.store(g_ptr + lc[:, None] * D + dd[None, :], gl + g, mask=mm)
        tl.store(g_ptr + rc[:, None] * D + dd[None, :], gr + g, mask=mm)

    @triton.jit
    def _nm_cells_kernel(
        g_ptr, om_ptr, cells_ptr, pos_leaf_ptr, leaf_ptr, gx_ptr, n, D, BLOCK_D: tl.constexpr, NB: tl.constexpr,
    ):  # fmt: skip
        """NB positions: each cell's gradient, its leaf's times its measure."""
        pos = tl.program_id(0).to(tl.int64) * NB + tl.arange(0, NB)
        m = pos < n
        cell = tl.load(cells_ptr + pos, mask=m, other=0).to(tl.int64)
        leaf = tl.load(pos_leaf_ptr + pos, mask=m, other=0).to(tl.int64)
        node = tl.load(leaf_ptr + leaf, mask=m, other=0).to(tl.int64)
        om = tl.load(om_ptr + cell, mask=m, other=0.0).to(tl.float32)
        dd = tl.arange(0, BLOCK_D)
        mm = m[:, None] & (dd < D)[None, :]
        g = tl.load(g_ptr + node[:, None] * D + dd[None, :], mask=mm, other=0.0)
        tl.store(gx_ptr + cell[:, None] * D + dd[None, :], (g * om[:, None]).to(gx_ptr.dtype.element_ty), mask=mm)


#: Nodes (or cells) per program of the node-means kernels (a launch constant).
NODE_BLOCK = 8


class _NodeMeans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forest, omega):
        n, d = x.shape
        x, omega = x.contiguous(), omega.contiguous()
        bd = triton.next_power_of_2(d)
        sums = torch.empty(forest.n_nodes, d, device=x.device, dtype=torch.float32)
        n_leaves = int(forest.leaf_ids.shape[0])
        if n_leaves:
            _nm_leaf_kernel[(triton.cdiv(n_leaves, NODE_BLOCK),)](
                x, omega, forest.cells, forest.leaf_ids, forest.range_start, forest.range_count, sums, n_leaves, d,
                BLOCK_D=bd, NB=NODE_BLOCK, enable_fp_fusion=False,
            )  # fmt: skip
        for nodes, left, right in forest.levels:
            _nm_up_kernel[(triton.cdiv(int(nodes.shape[0]), NODE_BLOCK),)](
                sums, nodes, left, right, int(nodes.shape[0]), d, BLOCK_D=bd, NB=NODE_BLOCK
            )
        measure = forest.measure.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
        ctx.save_for_backward(omega, measure)
        ctx.forest, ctx.x_dtype, ctx.n = forest, x.dtype, n
        return (sums / measure[:, None]).to(x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        omega, measure = ctx.saved_tensors
        forest = ctx.forest
        d = g.shape[1]
        bd = triton.next_power_of_2(d)
        gs = (g.to(torch.float32) / measure[:, None]).contiguous()
        for nodes, left, right in reversed(forest.levels):
            _nm_down_kernel[(triton.cdiv(int(nodes.shape[0]), NODE_BLOCK),)](
                gs, nodes, left, right, int(nodes.shape[0]), d, BLOCK_D=bd, NB=NODE_BLOCK
            )
        gx = torch.empty(ctx.n, d, device=g.device, dtype=ctx.x_dtype)
        if ctx.n:
            _nm_cells_kernel[(triton.cdiv(ctx.n, NODE_BLOCK),)](
                gs, omega, forest.cells, forest.position_leaf, forest.leaf_ids, gx, ctx.n, d, BLOCK_D=bd, NB=NODE_BLOCK
            )
        return gx, None, None


def node_means_supported(x: torch.Tensor) -> bool:
    """The fused node means apply: a CUDA tensor, Triton importable, not float64."""
    return HAS_TRITON and x.is_cuda and x.dtype != torch.float64


def fused_node_means(x: torch.Tensor, forest, omega: torch.Tensor) -> torch.Tensor:
    """:func:`~physicsnemo.experimental.models.isla._tree_read.node_means` of ``x``
    ``(N, D)`` in a few kernels. ``forest`` is the case's
    :class:`~physicsnemo.experimental.models.isla._tree_read.BoundaryForest` and
    ``omega`` ``(N,)`` the cells' measure fractions; requires
    :func:`node_means_supported`.

    The forward sums each leaf's cells in position order, then makes one launch per
    depth, every internal node the float32 sum of its two children. The backward adds
    each node's gradient to its children's, depth by depth from the roots. The float32
    operations are the reference's tensor ops but for the order of each leaf's sum (an
    atomic ``index_add`` there). So the backward is bitwise the reference's, and the
    forward is bitwise wherever no leaf holds three or more cells."""
    return _NodeMeans.apply(x, forest, omega)


class _BasisGrad:
    """The running sum of the blocks' basis gradients of one forward pass."""

    acc: torch.Tensor | None = None


class _BasisSink(torch.autograd.Function):
    """The basis's only consumer in the autograd graph: every fused read takes its empty
    token, so its backward runs after every read's and passes the summed gradient to the
    basis."""

    @staticmethod
    def forward(ctx, basis, grad: _BasisGrad):
        ctx.grad = grad
        return basis.new_empty(0)

    @staticmethod
    @once_differentiable
    def backward(ctx, _):
        acc, ctx.grad.acc = ctx.grad.acc, None
        return acc, None


def basis_sink(basis: torch.Tensor) -> tuple[torch.Tensor, _BasisGrad] | None:
    """The token and running sum through which every fused read of one forward pass
    passes its gradient to ``basis``, or ``None`` when ``basis`` needs no gradient.
    Give it to each :func:`fused_kernel_read` of that basis."""
    if not (torch.is_grad_enabled() and basis.requires_grad):
        return None
    grad = _BasisGrad()
    return _BasisSink.apply(basis, grad), grad


def _basis_grad_target(ctx, basis: torch.Tensor, zeroed: bool = True) -> torch.Tensor | None:
    """The running sum that a read's backward adds its basis gradient to. ``None`` when
    this backward pass will not run the basis's sink (a pass of other inputs only): the
    read's part would otherwise stay in the sum and be added to a later pass's.
    ``ctx.grad`` and ``ctx.sink_node`` are the read's sink, set in its forward. A new
    sum is zeros, or uninitialized with ``zeroed=False`` for a first read that writes it."""
    grad = ctx.grad
    if grad is None or not torch._C._will_engine_execute_node(ctx.sink_node):
        return None
    if grad.acc is None:
        grad.acc = torch.zeros_like(basis) if zeroed else torch.empty_like(basis)
    return grad.acc


def _narrow(basis: torch.Tensor) -> bool:
    return basis.dtype in (torch.bfloat16, torch.float16)


def _plan_source_items(plan: PairPlan):
    """The by-source work items of ``plan`` (source and chunk index), each source's
    first item, and each pair's position in the by-source order. Built once (counting
    the items costs a host sync) and kept on the plan for every block's backward."""
    items = getattr(plan, "_fused_source_items", None)
    if items is None:
        src, chunk_idx = _source_items(plan, SOURCE_CHUNK)
        item_off = torch.searchsorted(src, torch.arange(plan.n_b + 1, device=src.device, dtype=src.dtype))
        n = plan.perm.shape[0]
        inv = plan.perm.new_empty(n, dtype=torch.int32 if n < 2**31 else torch.int64)
        inv[plan.perm.long()] = torch.arange(n, device=inv.device, dtype=inv.dtype)
        items = src, chunk_idx, item_off, inv
        plan._fused_source_items = items
    return items


class _FusedKernelRead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, basis, token, grad: _BasisGrad | None, w, logm, q, k, v, plan: PairPlan, scale: float, out_dtype
    ):
        n_q, h, kc = q.shape
        c, f = v.shape[2], basis.shape[1]
        basis, w, logm, q, k, v = (t.contiguous() for t in (basis, w, logm, q, k, v))
        out = torch.empty(n_q, h, c, device=q.device, dtype=torch.float32)
        lse = torch.empty(n_q, h, device=q.device, dtype=torch.float32)
        if n_q:
            _fkr_fwd_kernel[(n_q,)](
                plan.q_offsets, plan.bj, basis, w, logm, q, k, v, out, lse, scale,
                H=h, C=c, KC=kc, F=f, BLOCK=BLOCK_PAIRS, NARROW=_narrow(basis), num_warps=NUM_WARPS,
                **(
                    {"maxnreg": FORWARD_MAXNREG if _narrow(basis) else FORWARD_MAXNREG_FLOAT32}
                    if torch.version.hip is None
                    else {}
                ),
            )  # fmt: skip
        ctx.save_for_backward(basis, w, logm, q, k, v, out, lse)
        ctx.plan, ctx.scale, ctx.grad = plan, scale, grad
        ctx.sink_node = None if token is None else token.grad_fn
        return out.to(out_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        basis, w, logm, q, k, v, out, lse = ctx.saved_tensors
        plan, scale = ctx.plan, ctx.scale
        n_q, h, kc = q.shape
        c, f = v.shape[2], basis.shape[1]
        ### dout arrives in the output's dtype (bfloat16 under autocast) and is read so.
        dout = dout.contiguous()
        dsum = (dout.float() * out).sum(-1)  # (n_q, H)
        dq = torch.empty(n_q, h, kc, device=q.device, dtype=torch.float32)
        ### The first read of a pass writes the sum (every pair's row is one point's); the others add to it.
        first = ctx.grad is not None and ctx.grad.acc is None
        acc = _basis_grad_target(ctx, basis, zeroed=False)
        dbasis = acc if acc is not None else basis
        dw_parts = torch.empty(n_q, 2 * h, f, device=q.device, dtype=torch.float32)
        dk = torch.empty(plan.n_b, h, kc, device=q.device, dtype=torch.float32)
        dv = torch.empty(plan.n_b, h, c, device=q.device, dtype=torch.float32)
        src, chunk_idx, item_off, inv = _plan_source_items(plan)
        prdz = torch.empty(inv.shape[0], 2 * h, device=q.device, dtype=torch.float32)
        ### d logit[p, h] / d logm[bj[p]] = 1: each source's log-measure gradient is the
        ### logit gradients of its pairs summed over the heads, by the source kernels.
        logm_grad = ctx.needs_input_grad[4]
        dlogm = torch.zeros(plan.n_b, device=q.device, dtype=torch.float32) if logm_grad else None
        narrow = _narrow(basis)
        cap = narrow and torch.version.hip is None
        if n_q:
            _fkr_bwd_point_kernel[(n_q,)](
                plan.q_offsets, plan.bj, basis, w, logm, q, k, v, lse, dsum, dout, inv, dq, dbasis, dw_parts, prdz,
                scale, H=h, C=c, KC=kc, F=f, BLOCK=BLOCK_PAIRS, BASIS_GRAD=acc is not None, FIRST=first,
                NARROW=narrow, num_warps=NUM_WARPS, **({"maxnreg": POINT_MAXNREG} if cap else {}),
            )  # fmt: skip
        if plan.n_b:
            ### Each source's dk and dv: every item's part, then their sum in item order.
            parts = torch.empty(src.shape[0], h * (kc + c), device=q.device, dtype=torch.float32)
            lparts = torch.empty(src.shape[0], device=q.device, dtype=torch.float32) if logm_grad else parts
            if src.shape[0]:
                _fkr_bwd_source_kernel[(src.shape[0],)](
                    plan.b_offsets, src, chunk_idx, plan.qi_by_b, prdz, q, dout, parts, lparts, scale, SOURCE_CHUNK,
                    H=h, C=c, KC=kc, BLOCK=BLOCK_PAIRS, LOGM_GRAD=logm_grad, num_warps=NUM_WARPS,
                    **({"maxnreg": SOURCE_MAXNREG} if cap else {}),
                )  # fmt: skip
            del prdz
            _fkr_source_sum_kernel[(plan.n_b,)](
                item_off, parts, lparts, dk, dv, dlogm if logm_grad else dk, H=h, C=c, KC=kc, LOGM_GRAD=logm_grad,
                num_warps=1,
            )  # fmt: skip
        dw = dw_parts[:, 0::2].sum(0).to(w.dtype)
        dtoken = None if acc is None else basis.new_empty(0)
        dlogm = None if dlogm is None else dlogm.to(logm.dtype)
        return None, dtoken, None, dw, dlogm, dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def fused_kernel_read(
    basis: torch.Tensor,
    w: torch.Tensor,
    logm: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: PairPlan,
    scale: float,
    sink: tuple[torch.Tensor, _BasisGrad] | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``segment_softmax_read(basis @ w.T, logm[plan.bj], q, k, v, plan, scale)`` with the
    kernel term computed inside the read. ``basis`` ``(P, F)`` is the pairs' kernel basis,
    ``w`` ``(H, F)`` the block's kernel weights in the basis's dtype, and ``logm``
    ``(n_b,)`` the sources' log measures (per source, not per pair); the rest are as in
    :func:`~physicsnemo.experimental.models.isla._segment_softmax.segment_softmax_read`.
    Requires :func:`supported`.

    Differentiable in ``basis``, ``w``, ``logm``, ``q``, ``k`` and ``v``. ``logm``'s
    gradient, each source's pairs' logit gradients summed over the heads, is computed
    only when it is asked for. Reads of one basis share its :func:`basis_sink`
    (``sink``); without one, the read's basis gradient is its own. The output is rounded
    to ``out_dtype``. Under autocast that is the dtype the next matmul rounds it to
    anyway, so its gradient arrives in that dtype, as the reference's (cast back to
    float32) does."""
    if w.dtype != basis.dtype:
        raise TypeError(f"the kernel weights ({w.dtype}) must be in the basis's dtype ({basis.dtype})")
    if sink is None:
        sink = basis_sink(basis)
    token, grad = sink if sink is not None else (None, None)
    return _FusedKernelRead.apply(basis.detach(), token, grad, w, logm, q, k, v, plan, float(scale), out_dtype)
