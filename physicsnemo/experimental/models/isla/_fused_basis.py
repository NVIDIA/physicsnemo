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

r"""The kernel basis with its pair invariants fused in (Triton). Between forward and
backward it keeps nothing per pair but the basis that the fused read consumes.

The reference (:func:`~physicsnemo.experimental.models.isla._tree_read._chunked_basis_from_plan`)
builds each chunk's pair invariants (:func:`~physicsnemo.experimental.models.isla._tree_read.tree_pair_invariants`)
and runs the basis MLP (:class:`~physicsnemo.experimental.models.isla._tree_read._KernelBasis`)
on them under a checkpoint: a dozen tensor ops per chunk and per layer, each a pass
over every pair. Here one kernel computes, per tile of pairs, the invariants (from the
plan's indices and the points' and sources' geometry), the three layers, and the
basis. The backward kernel rebuilds them per tile and sums the MLP's weight and bias
gradients.

The arithmetic is the reference's but for summation order:

- The invariants are float32, with IEEE division and square root and libdevice's
  logarithm, and each sum of two or three products in the order of PyTorch's
  reductions.
- Each linear layer is a dot product of operands in the MLP's dtype, accumulated in
  float32, plus its bias, rounded to that dtype. Under autocast the invariants, the
  weights and biases, and every layer's output are rounded to bfloat16 (or float16),
  as autocast's GEMMs do. In float32 the dots are IEEE (``input_precision="ieee"``,
  no TF32).
- Under autocast, GELU (the erf form) is PyTorch's own. Its 16-bit input indexes a
  table of PyTorch's float32 GELU and of its gradient factor at every 16-bit value
  (:func:`_gelu_tables`). That is bitwise what PyTorch's bfloat16 and float16 kernels
  compute: they run the float32 kernel on the upcast input and round. (Triton's
  libdevice ``erf`` differs from CUDA's ``erff`` in the last bit at a few hundred
  inputs.) The backward computes the same values instead, with CUDA 13's ``erff``
  ported to Triton (:func:`_erf`), wherever a check of every 16-bit value finds them
  bitwise the tables' (:func:`_computed_gelu_is_pytorchs`).
- In float32, GELU is computed from the erf form, to float32 roundoff of PyTorch's.
- The backward rounds every layer's input gradient to the MLP's dtype, as the
  reference GEMMs' outputs are. Each chunk's weight and bias gradients are float32
  sums over its pairs, rounded once to that dtype. Autograd sums the chunks' through
  one cast of the weights per forward pass (:func:`mlp_weights`), as it sums the
  reference's through autocast's cached cast.
"""

from __future__ import annotations

import logging

import torch
from torch.autograd.function import once_differentiable

from physicsnemo.experimental.models.isla._triton import HAS_TRITON, libdevice, tl, triton

logger = logging.getLogger(__name__)

#: Pairs per tile and warps per program of the forward and backward basis kernels, and
#: the backward's programs per multiprocessor (each sums the weight gradients of its
#: tiles). Launch constants tuned on Blackwell-class GPUs; re-tune them when the kernels
#: or Triton change. The outputs depend on the tile sizes at most through summation
#: order. The backward's program count follows the GPU's multiprocessor count, so its
#: float32 weight-gradient sums group by GPU model: deterministic on one model, not
#: bitwise across models. 12 programs per multiprocessor is a multiple of the 2 or 3
#: that one holds at once (by the kernel's registers), so no last wave runs part-empty.
BASIS_BLOCK = 128
BASIS_WARPS = 8
#: Warps per program of the forward kernel in float32 (its dots run on the CUDA cores),
#: a launch constant like the above.
BASIS_WARPS_FLOAT32 = 4
BACKWARD_BLOCK = 32
BACKWARD_WARPS = 4
BACKWARD_PROGRAMS_PER_SM = 12


def _pow2_at_least_16(n: int) -> bool:
    return n >= 16 and n & (n - 1) == 0


def unsupported_reason(basis: torch.nn.Module, geometry: dict) -> str | None:
    """Why the fused basis does not apply to ``basis`` (a ``_KernelBasis``) on one case's
    ``geometry`` (the keyword arguments of ``tree_pair_invariants``), or ``None`` when
    it does. It needs Triton importable, CUDA float32 geometry and parameters, two or
    three dimensions, power-of-two hidden and basis widths of at least 16, and either no
    autocast or a bfloat16 or float16 autocast with its weight cache on (the reference
    then casts the weights once per forward pass)."""
    if not HAS_TRITON:
        return "Triton with libdevice is not importable"
    lin = (basis.net[0], basis.net[2], basis.net[4])
    params = [t for m in lin for t in (m.weight, m.bias) if t is not None]
    geo = [geometry[k] for k in ("query_r", "query_n", "query_wall", "src_r", "src_n", "g_unit")]
    if geometry["src_vecs"] is not None:
        geo.append(geometry["src_vecs"])
    if not all(t.is_cuda and t.dtype == torch.float32 for t in geo + params):
        return "the geometry or the basis MLP's parameters are not CUDA float32"
    if torch.is_autocast_enabled("cuda") and not (
        torch.get_autocast_dtype("cuda") in (torch.bfloat16, torch.float16) and torch.is_autocast_cache_enabled()
    ):
        return "autocast is not bfloat16 or float16 with its weight cache on"
    if geometry["query_r"].shape[1] not in (2, 3):
        return "the geometry is not two- or three-dimensional"
    if not (_pow2_at_least_16(lin[0].out_features) and _pow2_at_least_16(lin[2].out_features)):
        return "the basis MLP's hidden or basis width is not a power of two of at least 16"
    return None


def mlp_weights(basis: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    """The basis MLP's weights and biases ``(W1, b1, W2, b2, W3, b3)`` as its layers
    read them. Under autocast on their device, each is cast to the autocast dtype once
    per forward pass, as autocast caches one cast of each for every chunk; otherwise they
    are the parameters themselves. A layer without a bias (the last one of
    ``_KernelBasis``) gets exact zeros."""
    params = [
        t if t is not None else m.weight.new_zeros(m.out_features)
        for m in (basis.net[0], basis.net[2], basis.net[4])
        for t in (m.weight, m.bias)
    ]
    dev = params[0].device.type
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        params = [t.to(dt) for t in params]
    return tuple(params)


if HAS_TRITON:

    @triton.jit
    def _sum3(a, b, c, D: tl.constexpr):
        """A sum over a last axis of D = 2 or 3 rounded terms in the order of PyTorch's
        CUDA reduction (two threads: the first takes terms 0 and 2)."""
        if D == 3:
            return (a + c) + b
        else:
            return a + b

    @triton.jit
    def _mm3(ax, ay, az, bx, by, bz, D: tl.constexpr, K: tl.constexpr):
        """A row of a float32 ``(P, D) @ (D, K)`` matmul with D = 2 or 3, in cuBLAS's
        order: for K = 1 the first and last terms fused, then the middle one added; else
        a chain of fused multiply-adds from the first term."""
        if K == 1:
            if D == 3:
                return tl.fma(az, bz, ax * bx) + ay * by
            else:
                return ax * bx + ay * by
        acc = tl.fma(ay, by, ax * bx)
        if D == 3:
            acc = tl.fma(az, bz, acc)
        return acc

    @triton.jit
    def _load3(ptr, row, stride, msk, D: tl.constexpr):
        """The D components of rows ``row`` of a float32 ``(.., D)`` table (the third
        zero when D = 2)."""
        x = tl.load(ptr + row * stride, mask=msk, other=0.0)
        y = tl.load(ptr + row * stride + 1, mask=msk, other=0.0)
        z = tl.zeros_like(x)
        if D == 3:
            z = tl.load(ptr + row * stride + 2, mask=msk, other=0.0)
        return x, y, z

    @triton.jit
    def _invariants(
        qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """The pairs' invariants ``(BLOCK, GP)`` float32, ``tree_pair_invariants``'s
        columns then zeros (and zero rows where ``msk`` is off)."""
        qx, qy, qz = _load3(qr_ptr, qi, D, msk, D)
        sx, sy, sz = _load3(sr_ptr, si, D, msk, D)
        rx, ry, rz = qx - sx, qy - sy, qz - sz
        dist = tl.sqrt_rn(_sum3(rx * rx, ry * ry, rz * rz, D))
        dist_g = tl.maximum(dist, eps)
        hx, hy, hz = tl.div_rn(rx, dist_g), tl.div_rn(ry, dist_g), tl.div_rn(rz, dist_g)
        ax, ay, az = _load3(qn_ptr, qi, D, msk, D)
        jx, jy, jz = _load3(sn_ptr, si, D, msk, D)
        log_r = libdevice.log(dist_g)
        log_w = libdevice.log(tl.maximum(tl.load(qw_ptr + qi, mask=msk, other=1.0), eps))
        cols = tl.arange(0, GP)[None, :]
        inv = tl.zeros([BLOCK, GP], tl.float32)
        inv = tl.where(cols == 0, dist[:, None], inv)
        inv = tl.where(cols == 1, log_r[:, None], inv)
        inv = tl.where(cols == 2, (log_r - log_w)[:, None], inv)
        inv = tl.where(cols == 3, _sum3(hx * jx, hy * jy, hz * jz, D)[:, None], inv)
        inv = tl.where(cols == 4, _sum3(hx * ax, hy * ay, hz * az, D)[:, None], inv)
        inv = tl.where(cols == 5, _sum3(ax * jx, ay * jy, az * jz, D)[:, None], inv)
        for k in tl.static_range(K):
            gx = tl.load(g_ptr + k * D)
            gy = tl.load(g_ptr + k * D + 1)
            gz = 0.0
            if D == 3:
                gz = tl.load(g_ptr + k * D + 2)
            inv = tl.where(cols == 6 + k, _mm3(hx, hy, hz, gx, gy, gz, D, K)[:, None], inv)
            inv = tl.where(cols == 6 + K + k, _mm3(jx, jy, jz, gx, gy, gz, D, K)[:, None], inv)
        for u in tl.static_range(PB):
            ux, uy, uz = _load3(sv_ptr + u * D, si, PB * D, msk, D)
            inv = tl.where(cols == 6 + 2 * K + u, _sum3(hx * ux, hy * uy, hz * uz, D)[:, None], inv)
            inv = tl.where(cols == 6 + 2 * K + PB + u, _sum3(ax * ux, ay * uy, az * uz, D)[:, None], inv)
        return tl.where(msk[:, None], inv, 0.0)

    @triton.jit
    def _dot(a, b, acc, NARROW: tl.constexpr):
        """``acc + a @ b`` in float32: exact products of narrow operands on the tensor
        cores, IEEE float32 products otherwise."""
        if NARROW:
            return tl.dot(a, b, acc)
        else:
            return tl.dot(a, b, acc, input_precision="ieee")

    @triton.jit
    def _linear(x, w_ptr, b_ptr, N_IN: tl.constexpr, IN: tl.constexpr, OUT: tl.constexpr, NARROW: tl.constexpr):
        """``x @ W.T + b`` in float32 for ``x`` ``(BLOCK, IN)`` and ``W`` ``(OUT, N_IN)``
        (``N_IN <= IN``, the rest zero), both in the MLP's dtype."""
        ii = tl.arange(0, IN)
        oo = tl.arange(0, OUT)
        wt = tl.load(w_ptr + oo[None, :] * N_IN + ii[:, None], mask=(ii < N_IN)[:, None], other=0.0)
        acc = _dot(x, wt, tl.zeros([x.shape[0], OUT], tl.float32), NARROW)
        return acc + tl.load(b_ptr + oo).to(tl.float32)[None, :]

    @triton.jit
    def _bits(x):
        """The bit patterns of 16-bit ``x`` as int32 indices 0..65535."""
        return x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF

    @triton.jit
    def _gelu(x, ftab_ptr, NARROW: tl.constexpr):
        """GELU (erf form) of ``x`` in the MLP's dtype: PyTorch's, from its table, for a
        16-bit ``x``; else computed in float32."""
        if NARROW:
            return tl.load(ftab_ptr + _bits(x))
        else:
            return x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067690849304))

    @triton.jit
    def _erf(a):
        """``erff`` of float32 ``a`` as CUDA 13's libdevice (``__nv_erff``) computes it, op
        for op, as PyTorch's CUDA 13 kernels do: Triton's own libdevice is older, with
        other coefficients for ``|a| >= 1.00296``."""
        t = tl.abs(a)
        big = t >= 1.002959966659546
        s = tl.where(big, t, a * a)
        r = tl.fma(
            tl.where(big, 0.00011219871521461755, 8.483494457323104e-05),
            s,
            tl.where(big, -0.0013275252422317863, -0.0008213091641664505),
        )
        r = tl.fma(r, s, tl.where(big, 0.00839653518050909, 0.005213488824665546))
        r = tl.fma(r, s, tl.where(big, -0.04024658352136612, -0.026868773624300957))
        r = tl.fma(r, s, tl.where(big, 0.15950430929660797, 0.11284004896879196))
        r = tl.fma(r, s, tl.where(big, 0.9129176735877991, -0.37612664699554443))
        r = tl.fma(r, s, tl.where(big, 0.6290600299835205, 0.12837915122509003))
        u = tl.where(big, -s, a)
        r = tl.fma(r, u, u)
        e = (1.0 - tl.exp2(r)).to(tl.int32, bitcast=True) & 0x7FFFFFFF
        e = (e | (a.to(tl.int32, bitcast=True) & -2147483648)).to(tl.float32, bitcast=True)
        return tl.where(big, e, r)

    @triton.jit
    def _gelu_and_grad(x, fgtab_ptr, NARROW: tl.constexpr, COMPUTED: tl.constexpr):
        """GELU of ``x`` (in the MLP's dtype) and its gradient factor, both float32. For
        a 16-bit ``x`` they are PyTorch's: computed with :func:`_erf` where ``COMPUTED``
        (:func:`_computed_gelu_is_pytorchs`), else read from one 64-bit table entry (the
        GELU in the low half). For a float32 ``x``, computed in float32 as in
        :func:`_gelu`."""
        if NARROW and not COMPUTED:
            v = tl.load(fgtab_ptr + _bits(x))
            a = (v & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
            g = (v >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        else:
            xf = x.to(tl.float32)
            if NARROW:
                e = _erf(xf * 0.7071067690849304)
            else:
                e = libdevice.erf(xf * 0.7071067690849304)
            a = xf * 0.5 * (1.0 + e)
            pdf = libdevice.exp(-0.5 * xf * xf) * 0.3989422917366028
            g = 0.5 * (1.0 + e) + xf * pdf
        return a, g

    @triton.jit
    def _hidden(inv, w1_ptr, b1_ptr, w2_ptr, b2_ptr, ftab_ptr, G: tl.constexpr, GP: tl.constexpr, HID: tl.constexpr,
                NARROW: tl.constexpr):  # fmt: skip
        """The MLP's input and hidden activations in its dtype: the rounded invariants,
        the first layer's output and GELU, the second's and GELU."""
        dt = w1_ptr.dtype.element_ty
        x0 = inv.to(dt)
        h1 = _linear(x0, w1_ptr, b1_ptr, G, GP, HID, NARROW).to(dt)
        a1 = _gelu(h1, ftab_ptr, NARROW).to(dt)
        h2 = _linear(a1, w2_ptr, b2_ptr, HID, HID, HID, NARROW).to(dt)
        a2 = _gelu(h2, ftab_ptr, NARROW).to(dt)
        return x0, h1, a1, h2, a2

    @triton.jit
    def _pair_rows(qi_ptr, si_ptr, tile, n, q_off, s_off, BLOCK: tl.constexpr):
        """The pair rows of tile ``tile`` (int64), their mask, and the case's point and
        source of each."""
        p = tile * BLOCK + tl.arange(0, BLOCK)
        msk = p < n
        qi = tl.where(msk, tl.load(qi_ptr + p, mask=msk, other=0).to(tl.int64) - q_off, 0)
        si = tl.where(msk, tl.load(si_ptr + p, mask=msk, other=0).to(tl.int64) - s_off, 0)
        return p, msk, qi, si

    @triton.jit
    def _inv_kernel(
        qi_ptr, si_ptr, n, q_off, s_off, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, out_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        """One tile's invariants ``(BLOCK, G)`` float32 (:func:`pair_invariants`)."""
        p, msk, qi, si = _pair_rows(qi_ptr, si_ptr, tl.program_id(0).to(tl.int64), n, q_off, s_off, BLOCK)
        inv = _invariants(qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps, D, K, PB, GP, BLOCK)
        cc = tl.arange(0, GP)
        tl.store(out_ptr + p[:, None] * G + cc[None, :], inv, mask=msk[:, None] & (cc < G)[None, :])

    @triton.jit
    def _basis_fwd_kernel(
        qi_ptr, si_ptr, n, q_off, s_off, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, ftab_ptr, out_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
        HID: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr, NARROW: tl.constexpr,
    ):  # fmt: skip
        """One tile of pairs: invariants, MLP, basis ``(BLOCK, F)`` in the MLP's dtype."""
        p, msk, qi, si = _pair_rows(qi_ptr, si_ptr, tl.program_id(0).to(tl.int64), n, q_off, s_off, BLOCK)
        inv = _invariants(qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps, D, K, PB, GP, BLOCK)
        _, _, _, _, a2 = _hidden(inv, w1_ptr, b1_ptr, w2_ptr, b2_ptr, ftab_ptr, G, GP, HID, NARROW)
        out = _linear(a2, w3_ptr, b3_ptr, HID, HID, F, NARROW).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + p[:, None] * F + tl.arange(0, F)[None, :], out, mask=msk[:, None])

    @triton.jit
    def _basis_bwd_kernel(
        qi_ptr, si_ptr, n, n_tiles, q_off, s_off, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr,
        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, fgtab_ptr, dout_ptr, part_ptr, eps,
        D: tl.constexpr, K: tl.constexpr, PB: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
        HID: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr, NARROW: tl.constexpr, COMPUTED: tl.constexpr,
    ):  # fmt: skip
        """Tiles ``pid, pid + programs, ...``: the MLP rebuilt, its backward from the
        basis gradient, and the program's float32 sums of the weight and bias gradients,
        written to ``part_ptr[pid]`` (W1 ``(HID, GP)``, b1, W2, b2, W3 ``(F, HID)``, b3). The
        weights are loaded once per program; each bias gradient is summed per row of the
        tile over the program's tiles, then over the rows once."""
        pid = tl.program_id(0).to(tl.int64)
        dt = w1_ptr.dtype.element_ty
        hh = tl.arange(0, HID)
        ff = tl.arange(0, F)
        ii = tl.arange(0, GP)
        dw1 = tl.zeros([HID, GP], tl.float32)
        dw2 = tl.zeros([HID, HID], tl.float32)
        dw3 = tl.zeros([F, HID], tl.float32)
        db1 = tl.zeros([BLOCK, HID], tl.float32)
        db2 = tl.zeros([BLOCK, HID], tl.float32)
        db3 = tl.zeros([BLOCK, F], tl.float32)
        w1t = tl.load(w1_ptr + hh[None, :] * G + ii[:, None], mask=(ii < G)[:, None], other=0.0)
        w2t = tl.load(w2_ptr + hh[None, :] * HID + hh[:, None])
        w2 = tl.load(w2_ptr + hh[:, None] * HID + hh[None, :])
        w3 = tl.load(w3_ptr + ff[:, None] * HID + hh[None, :])
        b1 = tl.load(b1_ptr + hh).to(tl.float32)
        b2 = tl.load(b2_ptr + hh).to(tl.float32)
        for tile in range(pid, n_tiles, tl.num_programs(0)):
            p, msk, qi, si = _pair_rows(qi_ptr, si_ptr, tile, n, q_off, s_off, BLOCK)
            inv = _invariants(
                qi, si, msk, qr_ptr, qn_ptr, qw_ptr, sr_ptr, sn_ptr, sv_ptr, g_ptr, eps, D, K, PB, GP, BLOCK
            )
            x0 = inv.to(dt)
            h1 = (_dot(x0, w1t, tl.zeros([BLOCK, HID], tl.float32), NARROW) + b1[None, :]).to(dt)
            a1, g1 = _gelu_and_grad(h1, fgtab_ptr, NARROW, COMPUTED)
            a1 = a1.to(dt)
            h2 = (_dot(a1, w2t, tl.zeros([BLOCK, HID], tl.float32), NARROW) + b2[None, :]).to(dt)
            a2, g2 = _gelu_and_grad(h2, fgtab_ptr, NARROW, COMPUTED)
            a2 = a2.to(dt)
            dy = tl.load(dout_ptr + p[:, None] * F + ff[None, :], mask=msk[:, None], other=0.0)
            dw3 = _dot(tl.trans(dy), a2, dw3, NARROW)
            db3 += dy.to(tl.float32)
            da2 = _dot(dy, w3, tl.zeros([BLOCK, HID], tl.float32), NARROW).to(dt)
            dh2 = (da2.to(tl.float32) * g2).to(dt)
            dw2 = _dot(tl.trans(dh2), a1, dw2, NARROW)
            db2 += dh2.to(tl.float32)
            da1 = _dot(dh2, w2, tl.zeros([BLOCK, HID], tl.float32), NARROW).to(dt)
            dh1 = (da1.to(tl.float32) * g1).to(dt)
            dw1 = _dot(tl.trans(dh1), x0, dw1, NARROW)
            db1 += dh1.to(tl.float32)
        part = part_ptr + pid * (HID * GP + 2 * HID + HID * HID + F * HID + F)
        tl.store(part + hh[:, None] * GP + ii[None, :], dw1)
        part += HID * GP
        tl.store(part + hh, tl.sum(db1, axis=0))
        part += HID
        tl.store(part + hh[:, None] * HID + hh[None, :], dw2)
        part += HID * HID
        tl.store(part + hh, tl.sum(db2, axis=0))
        part += HID
        tl.store(part + ff[:, None] * HID + hh[None, :], dw3)
        part += F * HID
        tl.store(part + ff, tl.sum(db3, axis=0))

    @triton.jit
    def _gelu_check_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        """The computed GELU of 16-bit ``x`` rounded to its dtype, and its factor, as
        float32 pairs (:func:`_computed_gelu_is_pytorchs`)."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        x = tl.load(x_ptr + i, mask=m, other=0.0)
        a, g = _gelu_and_grad(x, x_ptr, True, True)
        tl.store(out_ptr + 2 * i, a.to(x.dtype).to(tl.float32), mask=m)
        tl.store(out_ptr + 2 * i + 1, g, mask=m)


_SM_COUNT: dict = {}
_GELU_TABLES: dict = {}
_GELU_COMPUTED: dict = {}


def _gelu_tables(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch's GELU at every value of the 16-bit ``dtype``, indexed by bit pattern.
    The forward's table holds ``gelu(x)`` rounded to ``dtype``. The backward's holds that
    value as float32 beside the float32 factor ``gelu_backward(1, x)``, in one 64-bit
    entry. Both come from PyTorch's float32 kernels on the upcast values, built once per
    device and dtype."""
    key = (device, dtype)
    if key not in _GELU_TABLES:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            x = torch.arange(1 << 16, dtype=torch.int32, device=device).to(torch.int16).view(dtype).float()
            f = torch.nn.functional.gelu(x).to(dtype)
            g = torch.ops.aten.gelu_backward(torch.ones_like(x), x)
        _GELU_TABLES[key] = (f, torch.stack([f.float(), g], -1).view(torch.int64).reshape(-1))
    return _GELU_TABLES[key]


def _computed_gelu_is_pytorchs(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the backward's computed GELU and its factor (:func:`_erf`) equal the
    backward table's bitwise at every value of the 16-bit ``dtype`` (NaN where it holds
    NaN), as they do where PyTorch's CUDA kernels use CUDA 13's ``erff``. If so, the
    backward computes them; else it reads the table and logs that once.
    Checked once per device and dtype."""
    key = (device, dtype)
    if key not in _GELU_COMPUTED:
        _, fg = _gelu_tables(device, dtype)
        x = torch.arange(1 << 16, dtype=torch.int32, device=device).to(torch.int16).view(dtype)
        out = torch.empty(1 << 16, 2, device=device, dtype=torch.float32)
        _gelu_check_kernel[(triton.cdiv(1 << 16, 1024),)](x, out, 1 << 16, BLOCK=1024, enable_fp_fusion=False)
        ref = fg.view(torch.float32).view(-1, 2)
        same = ((out.view(torch.int32) == ref.view(torch.int32)) | (out.isnan() & ref.isnan())).all(1)
        _GELU_COMPUTED[key] = bool(same.all())
        if not _GELU_COMPUTED[key]:
            logger.info(
                "ISLA fused basis: the backward reads GELU from tables (computed, it is not PyTorch's at %d %s values)",
                int((~same).sum()),
                dtype,
            )
    return _GELU_COMPUTED[key]


def _sm_count(device: torch.device) -> int:
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _SM_COUNT:
        _SM_COUNT[key] = torch.cuda.get_device_properties(key).multi_processor_count
    return _SM_COUNT[key]


def _meta(qr, gu, pb, w1, w3) -> dict:
    hid, g = w1.shape
    return dict(
        D=int(qr.shape[1]), K=int(gu.shape[0]), PB=int(pb), G=int(g), GP=max(16, triton.next_power_of_2(int(g))),
        HID=int(hid), F=int(w3.shape[0]),
    )  # fmt: skip


def _geometry_args(geometry: dict) -> tuple[tuple[torch.Tensor, ...], int]:
    """The case's geometry tensors in the kernels' order, contiguous and detached, and the
    number of rank-1 inputs per source (``src_vecs`` a placeholder when there are none).
    Detached because the rank-1 inputs come from learned maps: an autograd edge into
    their graph would run (and free) it in every backward through the basis, such as
    each chunk's in :meth:`ISLA.query_chunks`."""
    sv = geometry["src_vecs"]
    pb = 0 if sv is None else int(sv.shape[1])
    keys = ("query_r", "query_n", "query_wall", "src_r", "src_n")
    geo = tuple(geometry[k].detach().contiguous() for k in keys)
    return geo + ((sv if pb else geometry["src_r"]).detach().contiguous(), geometry["g_unit"].detach().contiguous()), pb


class _FusedBasis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qi, si, q_off, s_off, eps, pb, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3):
        meta = _meta(qr, gu, pb, w1, w3)
        if meta["G"] != 6 + 2 * meta["K"] + 2 * pb:
            raise ValueError(f"the basis MLP takes {meta['G']} invariants; the geometry gives {6 + 2 * meta['K'] + 2 * pb}")
        meta["NARROW"] = w1.dtype in (torch.bfloat16, torch.float16)
        ftab, fgtab = _gelu_tables(qi.device, w1.dtype) if meta["NARROW"] else (b1, b1)
        n = int(qi.shape[0])
        out = torch.empty(n, meta["F"], device=qi.device, dtype=w1.dtype)
        if n:
            _basis_fwd_kernel[(triton.cdiv(n, BASIS_BLOCK),)](
                qi, si, n, q_off, s_off, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3, ftab, out, eps,
                BLOCK=BASIS_BLOCK, num_warps=BASIS_WARPS if meta["NARROW"] else BASIS_WARPS_FLOAT32,
                enable_fp_fusion=False, **meta,
            )  # fmt: skip
        ctx.save_for_backward(qi, si, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3)
        ctx.args = (q_off, s_off, eps, meta, fgtab)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        qi, si, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3 = ctx.saved_tensors
        q_off, s_off, eps, meta, fgtab = ctx.args
        hid, g, gp, f = meta["HID"], meta["G"], meta["GP"], meta["F"]
        sizes = (hid * gp, hid, hid * hid, hid, f * hid, f)
        n = int(qi.shape[0])
        n_tiles = triton.cdiv(n, BACKWARD_BLOCK)
        progs = max(1, min(n_tiles, BACKWARD_PROGRAMS_PER_SM * _sm_count(qi.device)))
        parts = torch.zeros(progs, sum(sizes), device=qi.device, dtype=torch.float32)
        if n:
            computed = meta["NARROW"] and _computed_gelu_is_pytorchs(qi.device, w1.dtype)
            _basis_bwd_kernel[(progs,)](
                qi, si, n, n_tiles, q_off, s_off, qr, qn, qw, sr, sn, sv, gu, w1, b1, w2, b2, w3, b3, fgtab,
                dout.contiguous(), parts, eps, BLOCK=BACKWARD_BLOCK, num_warps=BACKWARD_WARPS, enable_fp_fusion=False,
                COMPUTED=computed, **meta,
            )  # fmt: skip
        dw1, db1, dw2, db2, dw3, db3 = parts.sum(0).split(sizes)
        ### Each a float32 sum over the chunk's pairs, rounded once, as the reference GEMMs' are.
        grads = (dw1.view(hid, gp)[:, :g], db1, dw2.view(hid, hid), db2, dw3.view(f, hid), db3)
        return (None,) * 13 + tuple(t.to(w1.dtype) for t in grads)


def fused_kernel_basis(
    weights: tuple[torch.Tensor, ...], qi: torch.Tensor, si: torch.Tensor, q_off: int, s_off: int, geometry: dict
) -> torch.Tensor:
    """``_KernelBasis`` of ``tree_pair_invariants(qi - q_off, si - s_off, **geometry)``,
    ``(P, F)`` in the MLP's dtype, with the invariants and the MLP in one kernel.
    ``weights`` come from :func:`mlp_weights`; ``qi`` and ``si`` are the pairs' points and
    sources in the plan's numbering (int32 or int64); ``q_off`` and ``s_off`` are the
    case's first point and source. Differentiable in ``weights`` (the geometry gets no
    gradient: the tree read takes the reference basis when it requires one); requires
    :func:`unsupported_reason` to return ``None``."""
    geo, pb = _geometry_args(geometry)
    return _FusedBasis.apply(qi, si, int(q_off), int(s_off), float(geometry["eps"]), pb, *geo, *weights)


def pair_invariants(qi: torch.Tensor, si: torch.Tensor, q_off: int, s_off: int, geometry: dict) -> torch.Tensor:
    """The fused kernels' invariants ``(P, G)`` float32 of the same pairs, for tests."""
    geo, pb = _geometry_args(geometry)
    k = int(geometry["g_unit"].shape[0])
    g = 6 + 2 * k + 2 * pb
    n = int(qi.shape[0])
    out = torch.empty(n, g, device=qi.device, dtype=torch.float32)
    if n:
        _inv_kernel[(triton.cdiv(n, BASIS_BLOCK),)](
            qi, si, n, int(q_off), int(s_off), *geo, out, float(geometry["eps"]), D=int(geo[0].shape[1]), K=k, PB=pb,
            G=g, GP=max(16, triton.next_power_of_2(g)), BLOCK=BASIS_BLOCK, num_warps=BASIS_WARPS,
            enable_fp_fusion=False,
        )  # fmt: skip
    return out
