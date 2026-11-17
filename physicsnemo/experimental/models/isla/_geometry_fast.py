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

r"""Triton kernels of the fast boundary-geometry search (``kernel="fast"`` of
:func:`~physicsnemo.experimental.models.isla._geometry.exact_boundary_geometry`).

The mesh-only steps before the search compute integers from comparisons of the
vertex coordinates: the representative of each welded vertex, and each cell's
vertices in the lexicographic order of their positions with the parity of that
permutation. The kernels here compute the same integers as the reference's torch
code in one pass each. The torch code launches tens of kernels (the canonical order)
or a slow scan (the weld's running maximum).

Two kernels (:func:`closest_weights`) compute the closest points' weights on
triangles: Ericson's, the edge points recomputed on their edges, the thin-cell guard,
the accurate point and the switch to it on the guarded pairs, and the zero-area
cells' segments. The first runs over the pairs and lists the guarded ones; the second
runs over that list alone. The torch code launches about two hundred small kernels
for the same work. The float64 arithmetic is the torch code's operation by operation,
in the order torch's CUDA kernels evaluate the sums, norms and cross products, without
contraction, so the weights are the reference's bitwise.

Off CUDA or without Triton, each function falls back to the reference's torch code,
so its result never depends on the path taken.
"""

from __future__ import annotations

import torch

from physicsnemo.experimental.models.isla._triton import HAS_TRITON, tl, triton

_SCAN_BLOCK = 1024
_CELL_BLOCK = 256
_ACCURATE_BLOCK = 64


def _usable(t: torch.Tensor) -> bool:
    return HAS_TRITON and t.is_cuda


if HAS_TRITON:

    @triton.jit
    def _max_combine(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def _block_last_start_kernel(new, out, n, BLOCK: tl.constexpr):
        """The last position ``i`` with ``new[i]`` in each block (0 if none)."""
        b = tl.program_id(0)
        i = b * BLOCK + tl.arange(0, BLOCK)
        flag = tl.load(new + i, mask=i < n, other=0)
        tl.store(out + b, tl.max(tl.where(flag != 0, i, 0).to(tl.int64), 0))

    @triton.jit
    def _run_start_kernel(new, before, out, n, BLOCK: tl.constexpr):
        """The running maximum of ``where(new, i, 0)``: the block's own scan, against the
        greatest start of the blocks before it."""
        b = tl.program_id(0)
        i = b * BLOCK + tl.arange(0, BLOCK)
        flag = tl.load(new + i, mask=i < n, other=0)
        scan = tl.associative_scan(tl.where(flag != 0, i, 0).to(tl.int64), 0, _max_combine)
        tl.store(out + i, tl.maximum(scan, tl.load(before + b)), mask=i < n)

    @triton.jit
    def _canonical_kernel(points, cells, vert_out, parity_out, n, K: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        """Each cell's vertices sorted stably by their coordinates, the last first, by
        adjacent swaps of strictly smaller keys (NaN last), as
        :func:`~physicsnemo.experimental.models.isla._geometry._canonical_cells`; and the
        parity of the permutation."""
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = c < n
        v0 = tl.load(cells + K * c, mask=m, other=0)
        v1 = tl.load(cells + K * c + 1, mask=m, other=0)
        s0 = tl.zeros([BLOCK], dtype=tl.int32)
        s1 = s0 + 1
        if K == 3:
            v2 = tl.load(cells + K * c + 2, mask=m, other=0)
            s2 = s0 + 2
        for jj in tl.static_range(D):
            j = D - 1 - jj
            k0 = tl.load(points + D * v0 + j, mask=m, other=0.0).to(tl.float64)
            k1 = tl.load(points + D * v1 + j, mask=m, other=0.0).to(tl.float64)
            ### the pairs (0, 1), then for three vertices (1, 2) and (0, 1) again
            swap = (k1 < k0) | ((k0 != k0) & (k1 == k1))
            v0, v1 = tl.where(swap, v1, v0), tl.where(swap, v0, v1)
            s0, s1 = tl.where(swap, s1, s0), tl.where(swap, s0, s1)
            k0, k1 = tl.where(swap, k1, k0), tl.where(swap, k0, k1)
            if K == 3:
                k2 = tl.load(points + D * v2 + j, mask=m, other=0.0).to(tl.float64)
                swap = (k2 < k1) | ((k1 != k1) & (k2 == k2))
                v1, v2 = tl.where(swap, v2, v1), tl.where(swap, v1, v2)
                s1, s2 = tl.where(swap, s2, s1), tl.where(swap, s1, s2)
                k1, k2 = tl.where(swap, k2, k1), tl.where(swap, k1, k2)
                swap = (k1 < k0) | ((k0 != k0) & (k1 == k1))
                v0, v1 = tl.where(swap, v1, v0), tl.where(swap, v0, v1)
                s0, s1 = tl.where(swap, s1, s0), tl.where(swap, s0, s1)
        inv = (s0 > s1).to(tl.int64)
        tl.store(vert_out + K * c, v0, mask=m)
        tl.store(vert_out + K * c + 1, v1, mask=m)
        if K == 3:
            inv += (s0 > s2).to(tl.int64) + (s1 > s2).to(tl.int64)
            tl.store(vert_out + K * c + 2, v2, mask=m)
        tl.store(parity_out + c, 1 - 2 * (inv % 2), mask=m)
    ### The float64 arithmetic of the closest weights, each operation as torch's CUDA
    ### kernels evaluate it, launched without contraction (enable_fp_fusion=False). A
    ### sum over the last axis of three is ((x0 + x2) + x1) + 0, one over the vertex
    ### axis ((x0 + x1) + x2) + 0 (identity 0: never -0), a norm the root of ((x0^2 +
    ### x2^2) + x1^2), and a cross product's component an fma. 3-vectors are tuples.
    _TINY = tl.constexpr(torch.finfo(torch.float64).tiny)
    _U64 = tl.constexpr(2.0**-53)
    _INF = tl.constexpr(float("inf"))

    @triton.jit
    def _neg(x):
        """``-x`` as torch's, sign flipped (Triton's ``-x`` is ``0 - x``: +0 at +0)."""
        return x * -1.0

    @triton.jit
    def _sub(x, y):
        return x[0] - y[0], x[1] - y[1], x[2] - y[2]

    @triton.jit
    def _dot(x, y):
        return ((x[0] * y[0] + x[2] * y[2]) + x[1] * y[1]) + 0.0

    @triton.jit
    def _norm(x):
        return tl.sqrt((x[0] * x[0] + x[2] * x[2]) + x[1] * x[1])

    @triton.jit
    def _cross(x, y):
        """``torch.linalg.cross(x, y)``."""
        return (
            tl.fma(x[1], y[2], _neg(x[2] * y[1])),
            tl.fma(x[2], y[0], _neg(x[0] * y[2])),
            tl.fma(x[0], y[1], _neg(x[1] * y[0])),
        )

    @triton.jit
    def _amax(x, y):
        """``torch.amax`` of two values (NaN propagates)."""
        return tl.where((x != x) | (x > y), x, y)

    @triton.jit
    def _select(c, x, y):
        return tl.where(c, x[0], y[0]), tl.where(c, x[1], y[1]), tl.where(c, x[2], y[2])

    @triton.jit
    def _distance(q, w, s):
        """``(q - (w[..., None] * s).sum(1)).norm(dim=-1)``."""
        p0 = ((w[0] * s[0][0] + w[1] * s[1][0]) + w[2] * s[2][0]) + 0.0
        p1 = ((w[0] * s[0][1] + w[1] * s[1][1]) + w[2] * s[2][1]) + 0.0
        p2 = ((w[0] * s[0][2] + w[1] * s[1][2]) + w[2] * s[2][2]) + 0.0
        return _norm((q[0] - p0, q[1] - p1, q[2] - p2))

    @triton.jit
    def _segment_t(q, a, b):
        """:func:`~physicsnemo.experimental.models.isla._geometry._segment_t` (its
        clamps return NaN as it is)."""
        ab = _sub(b, a)
        len2 = _dot(ab, ab)
        t = _dot(_sub(q, a), ab) / tl.where(len2 != len2, len2, tl.maximum(len2, _TINY))
        t = tl.where(len2 > 0, t, 0.0)
        return tl.where(t != t, t, tl.minimum(tl.maximum(t, 0.0), 1.0))

    @triton.jit
    def _segment_distance2(q, a, b, t):
        """:func:`~physicsnemo.experimental.models.isla._geometry._segment_distance2`,
        ``t`` its :func:`_segment_t`."""
        ab = _sub(b, a)
        r = ((q[0] - a[0]) - t * ab[0], (q[1] - a[1]) - t * ab[1], (q[2] - a[2]) - t * ab[2])
        return _dot(r, r)

    @triton.jit
    def _clamp_min(x, low):
        return tl.where(x != x, x, tl.maximum(x, low))

    @triton.jit
    def _ericson(q, s):
        """Ericson's weights, as :func:`~physicsnemo.nn.functional.geometry.sdf._closest_point_weights`."""
        ab, ac = _sub(s[1], s[0]), _sub(s[2], s[0])
        qa, qb, qc = _sub(q, s[0]), _sub(q, s[1]), _sub(q, s[2])
        d1, d2, d3, d4, d5, d6 = _dot(ab, qa), _dot(ac, qa), _dot(ab, qb), _dot(ac, qb), _dot(ab, qc), _dot(ac, qc)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        denom = _clamp_min(va + vb + vc, _TINY)
        t_bc = (d4 - d3) / _clamp_min((d4 - d3) + (d5 - d6), _TINY)
        t_ac = d2 / _clamp_min(d2 - d6, _TINY)
        t_ab = d1 / _clamp_min(d1 - d3, _TINY)
        zero = tl.zeros_like(d1)
        w = (va / denom, vb / denom, vc / denom)
        w = _select((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), (zero, 1.0 - t_bc, t_bc), w)
        w = _select((vb <= 0) & (d2 >= 0) & (d6 <= 0), (1.0 - t_ac, zero, t_ac), w)
        w = _select((d6 >= 0) & (d5 <= d6), (zero, zero, zero + 1.0), w)
        w = _select((vc <= 0) & (d1 >= 0) & (d3 <= 0), (1.0 - t_ab, t_ab, zero), w)
        w = _select((d3 >= 0) & (d4 <= d3), (zero, zero + 1.0, zero), w)
        return _select((d1 <= 0) & (d2 <= 0), (zero + 1.0, zero, zero), w)

    @triton.jit
    def _nearer(d, w, d_best, w_best):
        """The running ``argmin`` of the distances: the first minimum, NaN first."""
        take = (d < d_best) | ((d != d) & (d_best == d_best))
        return tl.where(take, d, d_best), _select(take, w, w_best)

    @triton.jit
    def _two_sum(a, b):
        x = a + b
        bv = x - a
        return x, (a - (x - bv)) + (b - bv)

    @triton.jit
    def _two_product(a, b):
        x = a * b
        ca, cb = 134217729.0 * a, 134217729.0 * b
        ah, bh = ca - (ca - a), cb - (cb - b)
        al, bl = a - ah, b - bh
        return x, al * bl - (((x - ah * bh) - al * bh) - ah * bl)

    @triton.jit
    def _exact_cross(v0, v1, v2, i: tl.constexpr, j: tl.constexpr):
        """The component ``(i, j)`` of ``(v1 - v0) x (v2 - v0)``: the expansion of
        :func:`~physicsnemo.experimental.models.isla._geometry._cross_expansion`,
        summed smallest term first."""
        terms = ()
        for k in tl.static_range(3):
            x, y = (v0, v1, v2)[k], (v1, v2, v0)[k]
            p, pe = _two_product(x[i], y[j])
            r, re = _two_product(x[j], y[i])
            terms = terms + (p, pe, _neg(r), _neg(re))
        expansion = (terms[0],)
        for k in tl.static_range(1, 12):
            t = terms[k]
            grown = ()
            for e in tl.static_range(k):
                t, h = _two_sum(t, expansion[e])
                grown = grown + (h,)
            expansion = grown + (t,)
        total = expansion[0]
        for k in tl.static_range(1, 12):
            total = total + expansion[k]
        return total

    @triton.jit
    def _load_rows(q_ptr, s_ptr, r, m):
        """The queries ``(x, y, z)`` and the triangles ``((x, y, z),) * 3`` of rows ``r``."""
        q = (
            tl.load(q_ptr + 3 * r, mask=m, other=0.0),
            tl.load(q_ptr + 3 * r + 1, mask=m, other=0.0),
            tl.load(q_ptr + 3 * r + 2, mask=m, other=0.0),
        )
        s = ()
        for k in tl.static_range(3):
            v = s_ptr + 9 * r + 3 * k
            x = tl.load(v, mask=m, other=0.0), tl.load(v + 1, mask=m, other=0.0), tl.load(v + 2, mask=m, other=0.0)
            s = s + (x,)
        return q, s

    @triton.jit
    def _unit(q, s):
        return (_norm(q) + _amax(_amax(_norm(s[0]), _norm(s[1])), _norm(s[2]))) * _U64

    @triton.jit
    def _closest_weights_kernel(
        q_ptr, s_ptr, flat_ptr, out, guarded, count, n, HAS_FLAT: tl.constexpr, KAPPA: tl.constexpr,
        ERRBOUND: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """The closest weights of each row, as
        :func:`~physicsnemo.experimental.models.isla._geometry._closest_weights` in three
        dimensions (``KAPPA`` is ``_ERICSON_K u64``), except Ericson's on the guarded
        rows. It appends those to ``guarded`` (``count`` of them) for
        :func:`_accurate_kernel`. A block with no zero-area row skips the segments."""
        r = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        m = r < n
        q, s = _load_rows(q_ptr, s_ptr, r, m)
        zero = tl.zeros([BLOCK], dtype=tl.float64)
        ### Ericson's weights, an edge point recomputed on its edge (ends after the zero)
        w = _ericson(q, s)
        edge = ((w[0] > 0).to(tl.int32) + (w[1] > 0).to(tl.int32) + (w[2] > 0).to(tl.int32)) == 2
        gone = tl.where(w[0] == 0, 0, tl.where(w[1] == 0, 1, tl.where(w[2] == 0, 2, 0)))
        t = _segment_t(q, _select(gone == 0, s[1], s[0]), _select(gone == 2, s[1], s[2]))
        on_edge = (
            tl.where(gone == 0, zero, 1 - t),
            tl.where(gone == 0, 1 - t, tl.where(gone == 1, zero, t)),
            tl.where(gone == 2, zero, t),
        )
        w = _select(edge, on_edge, w)
        ### _thin_bounds' kappa and longest edge
        e1, e2, e3 = _sub(s[1], s[0]), _sub(s[2], s[0]), _sub(s[2], s[1])
        cross2 = tl.zeros([BLOCK], dtype=tl.float64)
        bound2 = tl.zeros([BLOCK], dtype=tl.float64)
        for k in tl.static_range(3):
            left, right = e1[(k + 1) % 3] * e2[(k + 2) % 3], e1[(k + 2) % 3] * e2[(k + 1) % 3]
            x = left - right
            cross2 = cross2 + x * x
            x = ERRBOUND * (tl.abs(left) + tl.abs(right))
            bound2 = bound2 + x * x
        longest2 = _amax(_amax(_dot(e1, e1), _dot(e2, e2)), _dot(e3, e3))
        longest = tl.sqrt(longest2)
        rho_low = (tl.sqrt(cross2) - tl.sqrt(bound2)) / longest2
        kappa = tl.where(rho_low > 0, (1.0 / (rho_low * rho_low * longest)) * KAPPA, _INF)
        ### the guard: _ericson_error not below one unit, except on a zero-area row
        dist = _distance(q, w, s)
        y = (_sub(q, s[0]), _sub(q, s[1]), _sub(q, s[2]))
        l2 = _amax(_amax(_dot(y[0], y[0]), _dot(y[1], y[1])), _dot(y[2], y[2]))
        delta = kappa * l2
        far = tl.minimum(delta, delta * delta / (2 * (dist - delta)), propagate_nan=tl.PropagateNan.ALL)
        eps = tl.where(dist > delta, far, delta)
        eps = tl.where(delta <= longest, eps, _INF)
        guard = ~(eps < _unit(q, s)) & m
        if HAS_FLAT:
            flat = tl.load(flat_ptr + r, mask=m, other=0) != 0
            guard = guard & ~flat
        g = guard.to(tl.int32)
        start = tl.atomic_add(count, tl.sum(g, 0))
        tl.store(guarded + start + tl.cumsum(g, 0) - 1, r, mask=guard)
        if HAS_FLAT:
            if tl.max(flat.to(tl.int32), 0) > 0:
                ### a zero-area cell: the nearer of its edges (0, 1) and (1, 2)
                t1, t2 = _segment_t(q, s[0], s[1]), _segment_t(q, s[1], s[2])
                second = _segment_distance2(q, s[1], s[2], t2) < _segment_distance2(q, s[0], s[1], t1)
                w = _select(flat, _select(second, (zero, 1 - t2, t2), (1 - t1, t1, zero)), w)
        for k in tl.static_range(3):
            tl.store(out + 3 * r + k, w[k], mask=m)

    @triton.jit
    def _accurate_kernel(q_ptr, s_ptr, out, guarded, count, SWITCH: tl.constexpr, BLOCK: tl.constexpr):
        """The guarded rows' weights (``guarded``, ``count`` of them; Ericson's in
        ``out``) switched to the accurate point's where their distances differ by more
        than ``SWITCH`` units, as
        :func:`~physicsnemo.experimental.models.isla._geometry._accurate_weights` and
        :func:`~physicsnemo.experimental.models.isla._geometry._closest_weights`."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = tl.load(count)
        if tl.program_id(0) * BLOCK < total:
            m = i < total
            r = tl.load(guarded + i, mask=m, other=0).to(tl.int64)
            q, s = _load_rows(q_ptr, s_ptr, r, m)
            w = (
                tl.load(out + 3 * r, mask=m, other=0.0),
                tl.load(out + 3 * r + 1, mask=m, other=0.0),
                tl.load(out + 3 * r + 2, mask=m, other=0.0),
            )
            zero = tl.zeros([BLOCK], dtype=tl.float64)
            ### the edge points on (0, 1), (0, 2), (1, 2) ...
            t = _segment_t(q, s[0], s[1])
            acc = (1 - t, t, zero)
            d_acc = _distance(q, acc, s)
            t = _segment_t(q, s[0], s[2])
            d_acc, acc = _nearer(_distance(q, (1 - t, zero, t), s), (1 - t, zero, t), d_acc, acc)
            t = _segment_t(q, s[1], s[2])
            d_acc, acc = _nearer(_distance(q, (zero, 1 - t, t), s), (zero, 1 - t, t), d_acc, acc)
            ### ... and the face's foot in the frame of its longest edge AB (the first)
            e1, e2, e3 = _sub(s[1], s[0]), _sub(s[2], s[0]), _sub(s[2], s[1])
            length2 = (_dot(e1, e1), _dot(e2, e2), _dot(e3, e3))
            take = (length2[1] > length2[0]) | ((length2[1] != length2[1]) & (length2[0] == length2[0]))
            best = tl.where(take, 1, 0)
            most = tl.where(take, length2[1], length2[0])
            best = tl.where((length2[2] > most) | ((length2[2] != length2[2]) & (most == most)), 2, best)
            a = _select(best == 2, s[1], s[0])
            b = _select(best == 0, s[1], s[2])
            c = _select(best == 0, s[2], _select(best == 1, s[1], s[0]))
            cross = (_exact_cross(a, b, c, 1, 2), _exact_cross(a, b, c, 2, 0), _exact_cross(a, b, c, 0, 1))
            size = _norm(cross)
            nrm = (cross[0] / size, cross[1] / size, cross[2] / size)
            ab = _sub(b, a)
            lab = _norm(ab)
            e = (ab[0] / lab, ab[1] / lab, ab[2] / lab)
            y = _sub(q, a)
            lam_c = _dot(y, _cross(nrm, e)) / (size / lab)
            lam_b = (_dot(y, e) - lam_c * _dot(_sub(c, a), e)) / lab
            lam_a = 1 - lam_b - lam_c
            face = (
                tl.where(best == 2, lam_c, lam_a),
                tl.where(best == 0, lam_b, tl.where(best == 1, lam_c, lam_a)),
                tl.where(best == 0, lam_c, lam_b),
            )
            on_face = (lam_a >= 0) & (lam_b >= 0) & (lam_c >= 0)
            d_acc, acc = _nearer(tl.where(on_face, tl.abs(_dot(y, nrm)), _INF), face, d_acc, acc)
            ### the switch
            switch = m & (tl.abs(d_acc - _distance(q, w, s)) > SWITCH * _unit(q, s))
            for k in tl.static_range(3):
                tl.store(out + 3 * r + k, acc[k], mask=switch)


def run_start(new: torch.Tensor) -> torch.Tensor:
    """``torch.cummax(torch.where(new, arange, 0), 0).values`` for a boolean ``(N,)``:
    for each position, the last position at or before it where ``new`` holds."""
    n = new.shape[0]
    if not _usable(new) or n == 0:
        return torch.cummax(torch.where(new, torch.arange(n, device=new.device), 0), 0).values
    flags = new.to(torch.int8).contiguous()
    blocks = triton.cdiv(n, _SCAN_BLOCK)
    last = torch.empty(blocks, dtype=torch.int64, device=new.device)
    _block_last_start_kernel[(blocks,)](flags, last, n, BLOCK=_SCAN_BLOCK)
    before = torch.cat([last.new_zeros(1), torch.cummax(last, 0).values[:-1]])
    out = torch.empty(n, dtype=torch.int64, device=new.device)
    _run_start_kernel[(blocks,)](flags, before, out, n, BLOCK=_SCAN_BLOCK)
    return out


def canonical_cells(points: torch.Tensor, cells: torch.Tensor, reference):
    """``reference(points, cells)`` (the reference's
    :func:`~physicsnemo.experimental.models.isla._geometry._canonical_cells`) by one
    Triton kernel on CUDA, for segments and triangles in two and three dimensions."""
    k, d = cells.shape[1], points.shape[1]
    if not _usable(points) or k not in (2, 3) or d not in (2, 3) or cells.shape[0] == 0:
        return reference(points, cells)
    n = cells.shape[0]
    vert = torch.empty_like(cells)
    parity = torch.empty(n, dtype=torch.int64, device=cells.device)
    _canonical_kernel[(triton.cdiv(n, _CELL_BLOCK),)](
        points.contiguous(), cells.contiguous(), vert, parity, n, K=k, D=d, BLOCK=_CELL_BLOCK
    )
    return vert, parity


def closest_weights(q, simplex, flat, reference, ericson_k, switch_units, orient_errbound):
    """``reference(q, simplex, flat)`` (the reference's
    :func:`~physicsnemo.experimental.models.isla._geometry._closest_weights` of
    triangles: Ericson's weights, the edge points recomputed, the thin-cell guard and
    the accurate point of each guarded pair, the zero-area cells' segments) by two
    Triton kernels on CUDA, the second on the guarded pairs alone. ``ericson_k``,
    ``switch_units`` and ``orient_errbound`` are the reference's constants
    ``_ERICSON_K``, ``_SWITCH_UNITS`` and ``_ORIENT_ERRBOUND``. The reference itself
    runs where an input is not contiguous, since its sums over a strided axis may take
    another order."""
    n = q.shape[0]
    contiguous = q.is_contiguous() and simplex.is_contiguous() and (flat is None or flat.is_contiguous())
    if not _usable(q) or n == 0 or not contiguous:
        return reference(q, simplex, flat)
    out = torch.empty_like(q)
    guarded = torch.empty(n, dtype=torch.int32, device=q.device)
    count = torch.zeros(1, dtype=torch.int32, device=q.device)
    _closest_weights_kernel[(triton.cdiv(n, _CELL_BLOCK),)](
        q, simplex, q if flat is None else flat, out, guarded, count, n, HAS_FLAT=flat is not None,
        KAPPA=ericson_k * 2.0**-53, ERRBOUND=orient_errbound, BLOCK=_CELL_BLOCK, enable_fp_fusion=False,
    )
    ### One program per block of rows that may be guarded; those past the count return.
    _accurate_kernel[(triton.cdiv(n, _ACCURATE_BLOCK),)](
        q, simplex, out, guarded, count, SWITCH=switch_units, BLOCK=_ACCURATE_BLOCK, num_warps=2, enable_fp_fusion=False
    )
    return out
