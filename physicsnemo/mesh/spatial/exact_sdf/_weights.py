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

"""The closest point of a segment or a triangle to a query, in float64.

Ericson's weights with the edge points recomputed on the edge alone, the
certified guard on thin triangles and the accurate point that replaces Ericson's
where the guard fires, with the rounding analyses behind each bound.
"""

import torch
import warp as wp

from physicsnemo.mesh.spatial.exact_sdf._predicates import (
    _ORIENT_ERRBOUND,
    _cross_expansion,
)
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights


def _segment_distance2(q, a, b):
    """Squared distance from points ``q`` to segments ``[a, b]`` (broadcast)."""
    r = q - a - _segment_t(q, a, b)[..., None] * (b - a)
    return (r * r).sum(-1)


def _segment_t(q, a, b):
    """The parameter ``t`` in [0, 1] of the point ``a + t (b - a)`` of each
    segment closest to ``q`` (broadcast); exactly 0 or 1 in a vertex's region."""
    ab = b - a
    len2 = (ab * ab).sum(-1)
    t = ((q - a) * ab).sum(-1) / len2.clamp_min(torch.finfo(len2.dtype).tiny)
    return torch.where(len2 > 0, t, 0).clamp(0, 1)


def _closest_weights(q, simplex, flat=None):
    """The barycentric weights ``(P, d)`` of the point of each simplex closest to
    its query (Ericson 2005, Sec. 5.1.5), exact zeros outside the face region; call
    in float64 (the face-region weights are ratios of Gram determinants that cancel
    on thin cells). An edge-region point of a triangle is recomputed on the edge
    alone, so every triangle sharing the edge gives it bitwise. A zero-area triangle
    (``flat`` ``(P,)``, :func:`_zero_area`; canonical vertices) is the union of its
    two edges between consecutive vertices, since on a line the lexicographic order
    is the order along it: its closest point is the nearer of theirs (the first at
    equal distances), computed as a triangle sharing that edge computes it.

    On a thin triangle Ericson's face-region weights move the point along the face
    by up to :func:`_ericson_error`. A pair where that bound reaches one unit of the
    arithmetic, ``u64 (|x| + R_k)`` (``R_k`` the cell's greatest vertex norm), is
    guarded: it keeps Ericson's weights only where its distance is within one unit
    (``_SWITCH_UNITS``) of the accurate point's (:func:`_accurate_weights`), within
    87 units of the exact one (``_FOOT_UNITS``), so that Ericson's is within the tie
    bound's arithmetic share, 91 units (``_BAND_UNITS``); otherwise it takes the
    accurate point. At every pair below the guard, Ericson's weights stand,
    bitwise. Both branches are certified: the guard by ``K`` (:data:`_ERICSON_K`),
    the switch by a band of at least ``_SWITCH_UNITS + _FOOT_UNITS``, each a worst
    case. ``unit`` is measured in the frame of ``q`` and ``simplex``."""
    if q.shape[-1] == 2:
        t = _segment_t(q, simplex[:, 0], simplex[:, 1])
        return torch.stack([1 - t, t], -1)
    w = _closest_point_weights(q, simplex)
    edge = (w > 0).sum(-1) == 2
    ends = torch.tensor([[1, 2], [0, 2], [0, 1]], device=q.device)[
        (w[edge] == 0).to(torch.int8).argmax(-1)
    ]
    s = simplex[edge]
    a, b = (
        s.gather(1, ends[:, :1, None].expand(-1, 1, 3))[:, 0],
        s.gather(1, ends[:, 1:, None].expand(-1, 1, 3))[:, 0],
    )
    t = _segment_t(q[edge], a, b)
    w[edge] = torch.zeros_like(s[..., 0]).scatter(1, ends, torch.stack([1 - t, t], -1))
    _, kappa, longest = _thin_bounds(simplex)
    dist = (q - (w[..., None] * simplex).sum(1)).norm(dim=-1)
    unit = 2.0**-53 * (q.norm(dim=-1) + simplex.norm(dim=-1).amax(-1))
    l2 = (q[:, None] - simplex).square().sum(-1).amax(-1)
    guard = ~(_ericson_error(dist, kappa * l2, longest) < unit)
    if flat is not None:
        guard &= ~flat
    rows = guard.nonzero()[:, 0]
    if rows.numel():
        acc, d_acc = _accurate_weights(q[rows], simplex[rows])
        switch = (d_acc - dist[rows]).abs() > _SWITCH_UNITS * unit[rows]
        w[rows] = torch.where(switch[:, None], acc, w[rows])
    if flat is not None:
        v0, v1, v2 = simplex.unbind(1)
        t1, t2 = _segment_t(q, v0, v1), _segment_t(q, v1, v2)
        zero = torch.zeros_like(t1)
        second = _segment_distance2(q, v1, v2) < _segment_distance2(q, v0, v1)
        seg = torch.where(
            second[:, None],
            torch.stack([zero, 1 - t2, t2], -1),
            torch.stack([1 - t1, t1, zero], -1),
        )
        w = torch.where(flat[:, None], seg, w)
    return w


#: Ericson's float64 closest point on a triangle of nonzero area is misplaced by at
#: most ``Delta = K u l^2 / (rho^2 L)`` (``u = 2^-53``, ``L`` the longest edge, ``rho =
#: 2 A / L^2``, ``l`` the greatest distance from the query to a vertex) wherever
#: ``Delta <= L``. Rounding analysis of :func:`_triangle_distance2` and
#: :func:`_closest_point_weights` as coded, each operation ``(1 + delta)``, ``|delta|
#: <= u``. A term rounded ``n`` times carries a factor ``1 + theta_n``, ``|theta_n| <=
#: n u / (1 - n u)`` exactly (Higham 2002, Lemma 3.1), so each count below is a worst
#: case, with no dropped term:
#:
#: 1. ``d1 .. d6``: an edge vector ``s`` (``ab``, ``ac``, ``|s| <= L``) dotted with a
#:    query offset ``t`` (``ap``, ``bp``, ``cp``, ``|t| <= l``): two differences, a
#:    product and two sums, at most five roundings per term, so ``|d~ - d| <= 5 u |s|
#:    |t| <= 5 u L l``.
#: 2. ``va = d3 d6 - d5 d4`` (``vb``, ``vc`` alike): each of the four factors' error
#:    times its partner (``|d| <= L l``), ``20 u L^2 l^2``, the two products'
#:    roundings, ``2 u L^2 l^2``, and the difference's, ``u |va|``: ``|Dva| = |va~ -
#:    va| <= 22 u L^2 l^2 + u |va|``.
#: 3. In the face region the exact ``va, vb, vc >= 0`` sum to ``D = (2 A)^2 = rho^2
#:    L^4`` and the closest point is ``P = a + (vb ab + vc ac) / D``. The computed one
#:    before the last roundings, ``P' = a + (vb~ ab + vc~ ac) / denom~``, satisfies
#:    exactly ``P' - P = -[(P - a) Dva + (P - b) Dvb + (P - c) Dvc + (P - a) r] /
#:    denom~`` (``r`` the rounding of the two sums of ``denom``, ``|r| <= 2 u D``): it
#:    moves along the face. With ``|P - v| <= L``, ``|P' - P| <= (66 u L^3 l^2 + 5 u L
#:    D) / denom~``, and ``denom~ >= (1 - theta) D``, ``theta = 66 u l^2 / (rho^2 L^2)
#:    + 3 u``.
#: 4. The rest moves the distance by at most ``4 u L + 5.5 u l``: the weights'
#:    divisions ``u L``, ``ap`` ``u l``, ``ab`` and ``ac`` ``u L``, their products with
#:    the weights ``u L``, the two differences ``u (l + L) + u l``, the squared norm
#:    and its root ``2.5 u l``. With step 3's ``5 u L``: ``9 u L + 5.5 u l <= 23.5 u
#:    l``, as ``L <= 2 l``.
#: 5. ``u l <= 1.5 u l^2 / (rho^2 L)``, as ``rho^2 <= 3 / 4`` and ``L <= 2 l``; so
#:    ``|d~ - d| <= (66 + 1.5 * 23.5) / (1 - theta) u l^2 / (rho^2 L) = 101.25 / (1 -
#:    theta) u l^2 / (rho^2 L)``. Where ``Delta <= L``, ``theta <= 66 / K + 3 u``, and
#:    ``K = 101.25 / (1 - 66 / K)`` gives ``K = 167.25`` (the ``5 u L D`` of step 3 is
#:    ``3 u L D`` on the face, where ``sum |v_i| = D``; harmlessly over-counted). The
#:    one denominator that
#:    cancels, ``denom~``, is kept as ``(1 - theta) D``, so its amplification is in
#:    the bound; the products of two error terms (``(5 u L l)^2`` against ``20 u L^2
#:    l^2``) add at most ``3e-15`` relative: 168 is a worst case.
#: 6. A region decided wrongly by rounding: an exact face point taken on the edge
#:    ``ab`` has ``vc <= |Dvc|``, so it lies within ``(vc / D) L <= 22 u l^2 / (rho^2
#:    L) + u L`` of that edge; an exact edge point taken on the face is beyond the
#:    plane's foot, which step 3 bounds; vertex and edge regions confused with each
#:    other cost ``O(u l)`` (``sqrt(75) u l`` at most); each within ``Delta``.
#:
#: The misplacement (steps 3 and 6) is along the face, so for a face-region point
#: ``d(P')^2 = d^2 + |P' - P|^2`` and ``d(P') - d <= min(Delta, Delta^2 / (2 (d(P') -
#: Delta)))`` (:func:`_ericson_error`): that form bounds the in-plane part only. Step
#: 4's roundings are linear in ``u l`` and do not shrink with the height; they are the
#: arithmetic share of the tie bound (``_BAND_UNITS``), as are the weights' sum and the
#: point's assembly from the vertices. :func:`_ericson_error` reads the computed
#: distance, up to ``a`` units above ``d(P')`` (9.5 on the face, 17.5 on an edge), so
#: its value below one unit certifies the in-plane error to ``(1 + sqrt(1 + 8 a)) /
#: 2`` units, at most 4.9 (6.4 on an edge), which the share counts. ``rho`` is
#: replaced by its certified lower bound (:func:`_thin_bounds`), which only raises
#: ``Delta``; the relative rounding of ``kappa``, ``Delta`` and ``eps`` themselves (a
#: few ``u``) is within the margin of ``K`` over 167.25.
_ERICSON_K = 168.0


#: The bound on the accurate point's distance error (:func:`_accurate_weights`), in
#: units of ``u64 (|x| + R_k)``. Rounding analysis as coded (``y = x - A``, ``l`` the
#: greatest distance from ``x`` to a vertex, ``|y| <= l``, ``L = |B - A|`` the longest
#: edge, ``F`` the exact foot):
#:
#: 1. The cross product, rounded once from its exact expansion (the partial sums of
#:    its smaller terms lie below an ulp of the largest): ``u`` per component. Its
#:    norm (three squares, two sums, a root) ``2.5 u`` more, the division ``u``:
#:    ``|n~ - n| <= 5.5 u``.
#: 2. ``B - A`` ``u`` per component, its norm ``3.5 u`` relative, the division ``u``:
#:    ``|e~ - e| <= 5.5 u``; ``m~ = n~ x e~``, ``11 u`` from its factors and ``4 u``
#:    of its own: ``|m~ - m| <= 15 u``.
#: 3. ``y`` ``u l``, each dot product ``3 u l``: ``s1`` and ``h`` err by ``9.5 u l``,
#:    ``s2`` by ``19 u l``.
#: 4. ``hc`` (``8 u`` relative) and the division: the point's offset along ``m``,
#:    ``lambda_C hc``, errs by ``19 u l + 9 lambda_C u L``. ``tc`` errs by ``9.5 u L``;
#:    the offset along ``e``, ``lambda_B |B - A| + lambda_C tc``, by ``9.5 u l + (10.5
#:    lambda_C + 5.5 lambda_B) u L``: ``tc``'s error and the product's rounding times
#:    ``lambda_C``, and the difference, the division and ``|B - A|``'s rounding, which
#:    scale with ``lambda_B |B - A| = s1 - lambda_C tc``; the error of ``lambda_C``
#:    along ``e`` cancels in ``lambda_B``. On an admitted face ``lambda_B + lambda_C <=
#:    1 + u``, so ``A + lambda_B (B - A) + lambda_C (C - A)`` lies in the plane within
#:    ``delta = 28.5 u l + 19.5 u L`` of ``F``; ``lambda_A >= 0`` is decided from
#:    rounded weights, so the point may lie ``u L`` beyond ``BC``; and the sum of the
#:    weights (``2 u``) and the assembly from the vertices (``3 u``) move it by ``5 u
#:    R_k`` more.
#: 5. An admitted face: ``|h|`` errs by ``9.5 u l``, and a foot admitted just outside
#:    the face lies within ``delta + u L`` of it: ``38 u l + 20.5 u L``, 79 units. A
#:    face whose exact foot ``F`` lies inside but which rounding rejects (some weight
#:    below 0): the exact distance is ``|h|``, the computed one the least edge
#:    candidate, at most ``sqrt(h^2 + s^2)`` plus an edge point's error (``14.5 u l + 3
#:    u R_k``, ``_BAND_UNITS``), ``s <= delta + u L`` the distance from ``F`` to the
#:    face's boundary: ``43 u l + 20.5 u L + 3 u R_k``. In this branch the weights are
#:    not bounded, but their error across ``AC`` is damped by ``hc / |AC|`` and across
#:    ``BC`` it stays within the same linear bound (``s <= hc``). An edge candidate
#:    alone errs as an edge point.
#: 6. With ``l <= |x| + R_k`` and ``L <= 2 R_k`` (both in the frame the arithmetic
#:    runs in): at most ``87 u64 (|x| + R_k)``, the rejected face's.
#:
#: Each count is a worst case, as ``_ERICSON_K``'s: on an admitted face no term is
#: divided by a quantity that rounding can make small (``lambda_C`` multiplies only
#: ``tc``'s error), so the products of two error terms add at most ``1.1e-14``
#: relative. The largest error observed against exact rational arithmetic is 2.03
#: units (600,000 pairs of slivers, caps and needles).
_FOOT_UNITS = 87


#: The switch threshold of a guarded pair (:func:`_closest_weights`), in units of
#: ``u64 (|x| + R_k)``: the pair keeps Ericson's point where its distance is within
#: this of the accurate point's, so within ``_SWITCH_UNITS + _FOOT_UNITS`` of the
#: exact one. Any threshold is certified by a band that covers both branches
#: (``_BAND_UNITS``); a larger one keeps Ericson's point on more pairs and widens the
#: band.
_SWITCH_UNITS = 1


#: The tie bound's arithmetic share, ``C u64 (|x'| + R'_k)``, by dimension
#: (:func:`_closest_candidates_impl`; primes the boundary's centred frame, in which
#: the arithmetic runs): a worst case of the float64 pair distance ``|x - sum_i w_i
#: v_i|``, in units of ``u64 (|x'| + R'_k)``, over every path the weights of
#: :func:`_closest_weights` take (each count as ``_FOOT_UNITS``'):
#:
#: - Ericson's point on a pair below the guard: its in-plane misplacement (at most
#:   4.9 units, ``_ERICSON_K``), the weights' divisions and their sum's rounding (3),
#:   the assembly from the vertices (3), the difference and the norm (3.5), and
#:   Ericson's linear roundings: at most 18.4 units.
#: - An edge point (:func:`_segment_t`): ``|t~ - t| |s| <= 11 u l`` along the edge
#:   (``s = b - a``, ``t |s| <= l``), the assembly ``3 u R_k``, the difference and the
#:   norm ``3.5 u l``: ``14.5 u l + 3 u R_k``, 17.5 units, plus Ericson's region test
#:   on the guard (6.4): 24. In two dimensions, with two-term dot products, ``12 u l +
#:   3 u R_k``: 15 units, the only path there.
#: - A vertex: the difference and the norm and the guard, 10 units.
#: - A zero-area triangle: an edge point, and the choice between its two edges
#:   (``_segment_distance2``, ``16.5 u l`` each): 51 units.
#: - A guarded pair switched to the accurate point, its distance recomputed from the
#:   weights: ``32 u l + 19.5 u L + 5 u R_k`` on an admitted face (76 units), and as
#:   ``_FOOT_UNITS`` after a rejected face (87).
#: - A guarded pair that keeps Ericson's point: ``_SWITCH_UNITS + _FOOT_UNITS``, 88.
#:
#: The kept guarded pair's is the greatest, so the band covers the switch by
#: construction (``fl(|d_acc - d~|) <= 1`` unit implies the exact difference is within
#: ``1 + 2 u`` units). The centring adds the rounding of ``x - c`` and ``v - c``, one
#: unit (none for float32 inputs whose exponents lie within 29 bits of ``c``'s), and
#: the second-order terms and the band's own float64 evaluation two more (rounding is
#: monotone, so ``fl(D - E) <= fl(D + E)`` needs no margin): 91 in three dimensions.
#: In two, an edge point's 15 units, the centring's 1 and one more: 17. Torch evaluates
#: each operation by itself (no fused multiply-add, which would only remove roundings in
#: the bounds; :func:`_two_product` and :func:`_two_sum` need it absent). Float64 inputs
#: are covered where every product stays normal, about ``|v| < 1e76`` with nonzero
#: differences above ``1e-76``.
_BAND_UNITS = {2: 17, 3: _SWITCH_UNITS + _FOOT_UNITS + 3}


def _thin_bounds(simplex: torch.Tensor):
    """Per triangle ``abc`` of ``simplex`` ``(F, 3, 3)`` (float64): ``delta``, a bound
    on the angle between its computed float64 normal and the exact one, ``|E| /
    |X|`` with ``X`` the cross product ``(b - a) x (c - a)`` and ``E`` its components'
    a-priori error bounds, Shewchuk's ``_ORIENT_ERRBOUND (|l| + |r|)`` (as
    :func:`_zero_area`'s filter); ``kappa = K u64 / (rho_low^2 L)`` (``_ERICSON_K``),
    with ``rho_low = (|X| - |E|) / L^2`` a certified lower bound of ``2 A / L^2``,
    ``inf`` where it is not positive; and ``L``, the longest edge."""
    a, b, c = simplex.unbind(1)
    e1, e2, e3 = b - a, c - a, c - b
    cross2, bound2 = torch.zeros_like(a[:, 0]), torch.zeros_like(a[:, 0])
    for i, j in ((1, 2), (2, 0), (0, 1)):
        left, right = e1[:, i] * e2[:, j], e1[:, j] * e2[:, i]
        cross2 = cross2 + (left - right).square()
        bound2 = bound2 + (_ORIENT_ERRBOUND * (left.abs() + right.abs())).square()
    cross, bound = cross2.sqrt(), bound2.sqrt()
    longest2 = torch.stack([(e * e).sum(-1) for e in (e1, e2, e3)]).amax(0)
    longest = longest2.sqrt()
    rho_low = (cross - bound) / longest2
    kappa = torch.where(
        rho_low > 0, _ERICSON_K * 2.0**-53 / (rho_low * rho_low * longest), torch.inf
    )
    return bound / cross, kappa, longest


def _ericson_error(dist, delta, longest):
    """The bound on the error of Ericson's distance ``dist`` to a cell whose point
    is misplaced by at most ``delta`` (``kappa l^2``, :func:`_thin_bounds`) along the
    face: ``min(delta, delta^2 / (2 (dist - delta)))``, ``delta`` where ``dist <=
    delta``, and ``inf`` where ``delta`` exceeds the cell's longest edge (the bound's
    analysis then does not hold), as :func:`_ericson_error_wp`."""
    eps = torch.where(
        dist > delta, torch.minimum(delta, delta * delta / (2 * (dist - delta))), delta
    )
    return torch.where(delta <= longest, eps, torch.inf)


@wp.func
def _ericson_error_wp(
    dist: wp.float64, delta: wp.float64, longest: wp.float64
) -> wp.float64:
    """:func:`_ericson_error` in Warp."""
    if not (delta <= longest):
        return wp.float64(wp.inf)
    if dist <= delta:
        return delta
    return wp.min(delta, delta * delta / (wp.float64(2.0) * (dist - delta)))


def _accurate_weights(q: torch.Tensor, simplex: torch.Tensor):
    """The barycentric weights ``(P, 3)`` of the point of each triangle ``simplex``
    ``(P, 3, 3)`` (canonical, nonzero area) closest to its query ``q`` ``(P, 3)``,
    float64, accurate whatever its shape, and that point's distance ``(P,)``: the
    nearest (the first at equal distances) of the three edge points, computed as
    :func:`_closest_weights` computes an edge point (on the canonical pairs (0, 1),
    (0, 2), (1, 2), so bitwise every neighbour's), and of the face foot where it lies
    on the face, at distance ``|h|``. The foot is found in the frame of the longest
    edge ``AB`` (C the third vertex): ``n = X / |X|``, ``X`` the cross product ``(B -
    A) x (C - A)`` rounded once from its exact value (:func:`_cross_expansion`,
    summed smallest term first), ``e = (B - A) / |B - A|``, ``m = n x e`` (towards
    ``C``), ``s1 = (x - A) . e``, ``s2 = (x - A) . m``, ``h = (x - A) . n``; with ``hc =
    |X| / |B - A|`` and ``tc = (C - A) . e``, ``lambda_C = s2 / hc``, ``lambda_B = (s1 -
    lambda_C tc) / |B - A|``, ``lambda_A = 1 - lambda_B - lambda_C``, the foot lying on
    the face when all three are ``>= 0``. Along ``e`` the error of ``lambda_C``
    cancels in ``lambda_B``'s, so the point is accurate to rounding of the
    coordinates whatever ``rho``."""
    pairs = ((0, 1), (0, 2), (1, 2))
    weights, dists = [], []
    for i, j in pairs:
        t = _segment_t(q, simplex[:, i], simplex[:, j])
        w = torch.zeros_like(q)
        w[:, i], w[:, j] = 1 - t, t
        weights.append(w)
        dists.append((q - (w[..., None] * simplex).sum(1)).norm(dim=-1))
    length2 = torch.stack(
        [((simplex[:, j] - simplex[:, i]) ** 2).sum(-1) for i, j in pairs], 1
    )
    slots = torch.tensor([[0, 1, 2], [0, 2, 1], [1, 2, 0]], device=q.device)[
        length2.argmax(1)
    ]
    a, b, c = (
        simplex.gather(1, slots[:, k, None, None].expand(-1, 1, 3))[:, 0]
        for k in range(3)
    )
    cross = []
    for i, j in ((1, 2), (2, 0), (0, 1)):
        expansion = _cross_expansion(a, b, c, i, j)
        total = expansion[0]
        for term in expansion[1:]:
            total = total + term
        cross.append(total)
    cross = torch.stack(cross, -1)
    size = cross.norm(dim=-1)
    n = cross / size[:, None]
    ab = b - a
    lab = ab.norm(dim=-1)
    e = ab / lab[:, None]
    m = torch.linalg.cross(n, e, dim=-1)
    y = q - a
    s1, s2, h = (y * e).sum(-1), (y * m).sum(-1), (y * n).sum(-1)
    lam_c = s2 / (size / lab)
    lam_b = (s1 - lam_c * ((c - a) * e).sum(-1)) / lab
    lam = torch.stack([1 - lam_b - lam_c, lam_b, lam_c], -1)
    weights.append(torch.zeros_like(q).scatter(1, slots, lam))
    dists.append(torch.where((lam >= 0).all(-1), h.abs(), torch.inf))
    dists = torch.stack(dists, 1)
    best = dists.argmin(1)
    return torch.stack(weights, 1).gather(1, best[:, None, None].expand(-1, 1, 3))[
        :, 0
    ], dists.gather(1, best[:, None])[:, 0]
