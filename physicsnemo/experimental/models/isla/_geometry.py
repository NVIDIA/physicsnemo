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

"""ISLA's geometric statistics of the boundary.

:func:`rms_pairwise_distance` is the model's one length unit, a quadrature over
the boundary tokens; :func:`exact_boundary_geometry` gives the prediction
points' built-in signed distance and direction to the boundary *surface*, from
the exact closest point on its cells. Both are built from differences of
positions only, so they refer to no frame origin."""

import itertools
import math
from typing import Any

import torch
import warp as wp
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.experimental.models.isla import _geometry_fast as geometry_fast
from physicsnemo.mesh.geometry import compute_cell_normals
from physicsnemo.mesh.spatial import canonical_frame, frame_coordinates
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes
from physicsnemo.mesh.spatial.cluster_tree import _WARP_MAX_ELEMENTS as WARP_MAX_ELEMENTS
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights

### The (query, cell) pairs per chunk of the two-dimensional search, and the (point,
### subset) pairs per chunk of the least-norm solve: a bound on their blocks. Both
### run in a custom op, so their loops over chunks do not specialize a dynamic query
### count under torch.compile.
_SEARCH_PAIRS = 2**23

#: The softening of the logarithmic length channels, in the frame's length unit
#: (:func:`soft_log_length` gives its reason).
LENGTH_SOFTENING = 1e-6

#: The far-field criterion of the winding number that signs the distance
#: (:func:`_packed_winding_sum`; Barill et al. 2018's ``beta``): a BVH node's
#: expansion stands in for its cells when the query lies farther than this many node
#: radii from its centre. A performance setting, like the tree read's
#: ``TREE_READ_THETA``: the error falls to zero as it grows. At 2 its error against
#: the exact sum (float64, 3,000 queries each) is at most 3.1e-2 (median 1e-3) on an
#: icosphere of 5,120 triangles, 1.9e-2 on a torus, and 1.2e-2 on a cube with a face
#: removed. Off a closed or double-sided surface the exact ``w`` is an integer, half
#: away from the threshold, so the sign is the surface's. The sign can depend on the
#: tree only where the exact ``|w - sigma / 2|`` is below that error (behind an
#: isolated sheet, across a hole).
WINDING_NUMBER_BETA = 2.0


def soft_log_length(s: torch.Tensor) -> torch.Tensor:
    r"""The logarithmic channel of a signed length ``s`` in the frame's length unit:
    :math:`\operatorname{asinh}(s / \varepsilon)`.

    For :math:`\lvert s \rvert \gg \varepsilon` it is
    :math:`\operatorname{sign}(s) \log(2 \lvert s \rvert / \varepsilon)` up to
    :math:`O((\varepsilon / s)^2)`, so it resolves the distances to a wall over many
    decades, as a logarithm does. Near zero it is linear, :math:`s / \varepsilon`:
    odd, smooth and continuous through :math:`s = 0`, with slope at most
    :math:`1 / \varepsilon`. Lupton, Gunn and Szalay (1999, "asinh magnitudes")
    soften a logarithm the same way at the noise level of what it measures. A signed
    logarithm :math:`\operatorname{sign}(s) \log(\lvert s \rvert + \epsilon)` instead
    jumps by :math:`2 \lvert \log \epsilon \rvert` at :math:`s = 0`. A point on a
    wall, whose distance is zero or a rounding error of either sign, then gets a code
    that rounding or a rigid motion can change by tens.

    The softening :math:`\varepsilon = 10^{-6}` is a numerical resolution, not a
    model parameter. It is in the frame's length unit, so the channel is invariant to
    a uniform scaling of the geometry. It is about 17 times float32's unit roundoff
    :math:`2^{-24}`, the resolution of a distance relative to the frame's length
    wherever the geometry sits in its domain. The channel is linear for
    :math:`\lvert s \rvert` below :math:`\varepsilon`. For the whole boundary of an
    external flow the length unit is the domain's size, so that range can reach into
    the boundary layer.
    """
    return torch.asinh(s / LENGTH_SOFTENING)


def rms_pairwise_distance(
    points: torch.Tensor,
    measure_weights: torch.Tensor | None,
    second_moments: torch.Tensor | None = None,
) -> torch.Tensor:
    """Measure-weighted RMS pairwise distance of a sampled boundary, shape (B, 1, 1).

    ``L^2 = sum_ij w~_i w~_j |x_i - x_j|^2`` with ``w~ = w / sum(w)`` (uniform
    weights when ``measure_weights`` is ``None``). It is evaluated in O(N) through the
    variance identity ``L^2 = 2 sum_i w~_i (|x_i - c|^2 + J_i)``, with ``c`` the
    weighted mean and ``J_i`` each point's own second moment about itself, per unit
    measure (``second_moments`` ``(B, N)``, :func:`cell_second_moments`; ``None`` for
    points). With the cells' centroids, measures and second moments it is the exact
    RMS distance between two points of the piecewise-flat surface, the same for every
    subdivision of its cells.

    As a normalized two-point statistic, it converges under refinement of any
    boundary whose normalized measure converges (fractal boundaries of divergent
    total measure included). It is invariant to a uniform rescale of the weights and
    covariant under a geometric rescale. The mean ``c`` is an intermediate of this
    scalar only; no centre is exposed. Its two floors, on the total weight and on the
    result, are the dtype's smallest normal number. They guard only a boundary of no
    measure, and bind at no length scale the dtype represents.
    """
    b, n = points.shape[0], points.shape[1]
    if measure_weights is None:
        w = points.new_ones(b, n, 1)
    else:
        w = measure_weights.reshape(b, n, 1).to(points.dtype)
    tiny = torch.finfo(points.dtype).tiny
    w_n = w / w.sum(dim=1, keepdim=True).clamp_min(tiny)
    c = (w_n * points).sum(dim=1, keepdim=True)
    sq = (points - c).square().sum(-1, keepdim=True)
    if second_moments is not None:
        sq = sq + second_moments.reshape(b, n, 1).to(points.dtype)
    var = (w_n * sq).sum(dim=1, keepdim=True)
    return (2.0 * var).sqrt().clamp_min(tiny)


def cell_second_moments(mesh) -> torch.Tensor:
    r"""Each simplex cell's second moment about its centroid per unit measure,
    :math:`\frac{1}{|K|}\int_K |x - g|^2 = \frac{1}{(n+1)(n+2)} \sum_k |v_k - g|^2` for an
    :math:`n`-simplex with vertices :math:`v_k` and centroid :math:`g` (a segment's
    :math:`\ell^2 / 12`), shape ``(n_cells,)``: the parallel-axis term that makes
    :func:`rms_pairwise_distance` exact for the piecewise-flat surface."""
    verts = mesh.points[mesh.cells]  # (C, n + 1, d)
    k = verts.shape[1]
    g = verts.mean(1, keepdim=True)
    return (verts - g).square().sum((1, 2)) / (k * (k + 1))


def _weld(
    points: Float[torch.Tensor, "vertices dim"],
    run_start=None,
) -> Int[torch.Tensor, " vertices"]:
    """For every vertex, the index of one representative of the vertices at
    bitwise the same position (a stable lexicographic sort; static shapes), so
    that a surface split into pieces, or stored as a triangle soup, has the
    connectivity of the surface it describes."""
    n = points.shape[0]
    order = torch.arange(n, device=points.device)
    for k in reversed(range(points.shape[1])):
        order = order[torch.argsort(points[order, k], stable=True)]
    s = points[order]
    new = torch.cat([s.new_ones(1, dtype=torch.bool), (s[1:] != s[:-1]).any(-1)])
    if run_start is None:
        first = torch.cummax(
            torch.where(new, torch.arange(n, device=points.device), 0), 0
        ).values
    else:
        first = run_start(new)
    return torch.zeros_like(order).index_copy(0, order, order[first])


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


def _canonical_cells(points, cells):
    """Each cell's vertices in the lexicographic order of their positions ``(F,
    d)``, and the parity of that permutation ``(F,)`` (+1 or -1). Both are functions
    of the positions alone, the same for every numbering of the vertices and every
    rotation of a cell's vertex list, so the closest points computed on them are too.
    A winding flip changes only the parity, which carries the orientation."""
    k = cells.shape[1]
    vert, slot = list(cells.unbind(1)), [torch.full_like(cells[:, 0], i) for i in range(k)]
    ### A stable sort per coordinate, as argsort's (NaN last), by adjacent swaps of
    ### strictly smaller keys: elementwise, where a sort along rows of length k is not.
    for j in reversed(range(points.shape[1])):
        key = [points[v, j] for v in vert]
        for a in (a for p in range(k - 1) for a in range(k - 1 - p)):
            x, y = key[a], key[a + 1]
            swap = (y < x) | (x.isnan() & ~y.isnan())
            for col in (vert, slot, key):
                col[a], col[a + 1] = torch.where(swap, col[a + 1], col[a]), torch.where(swap, col[a], col[a + 1])
    inversions = sum((slot[a] > slot[b]).long() for a in range(k) for b in range(a + 1, k))
    return torch.stack(vert, 1), 1 - 2 * (inversions % 2)


def _closest_weights(q, simplex, flat=None, fast=False):
    """The barycentric weights ``(P, d)`` of the point of each simplex closest to
    its query (Ericson 2005, Sec. 5.1.5), with exact zeros outside the face region.
    Call it in float64: the face-region weights are ratios of Gram determinants that
    cancel on thin cells. An edge-region point of a triangle is recomputed on the
    edge alone, so every triangle sharing the edge gives it bitwise. A zero-area
    triangle (``flat`` ``(P,)``, :func:`_zero_area`; canonical vertices) is the union
    of its two edges between consecutive vertices, since on a line the lexicographic
    order is the order along it. Its closest point is the nearer of theirs (the first
    at equal distances), computed as a triangle sharing that edge computes it.

    On a thin triangle Ericson's face-region weights move the point along the face
    by up to :func:`_ericson_error`. A pair where that bound reaches one unit of the
    arithmetic, ``u64 (|x| + R_k)`` (``R_k`` the cell's greatest vertex norm), is
    guarded. It keeps Ericson's weights only where its distance is within one unit
    (``_SWITCH_UNITS``) of the accurate point's (:func:`_accurate_weights`), which is
    within 87 units of the exact one (``_FOOT_UNITS``). So Ericson's is within the
    tie bound's arithmetic share, 91 units (``_BAND_UNITS``). Otherwise it takes the
    accurate point. Every pair below the guard keeps Ericson's weights, bitwise. Both
    branches are certified, each by a worst case: the guard by ``K``
    (:data:`_ERICSON_K`), the switch by a band of at least ``_SWITCH_UNITS +
    _FOOT_UNITS``. ``unit`` is measured in the frame of ``q`` and ``simplex``. ``fast`` computes
    the weights of triangles by two Triton kernels on CUDA, bitwise these
    (:func:`~physicsnemo.experimental.models.isla._geometry_fast.closest_weights`)."""
    if q.shape[-1] == 2:
        t = _segment_t(q, simplex[:, 0], simplex[:, 1])
        return torch.stack([1 - t, t], -1)
    if fast:
        constants = (_ERICSON_K, _SWITCH_UNITS, _ORIENT_ERRBOUND)
        return geometry_fast.closest_weights(q, simplex, flat, _closest_weights, *constants)
    w = _closest_point_weights(q, simplex)
    edge = (w > 0).sum(-1) == 2
    ends = torch.tensor([[1, 2], [0, 2], [0, 1]], device=q.device)[(w[edge] == 0).to(torch.int8).argmax(-1)]
    s = simplex[edge]
    a, b = s.gather(1, ends[:, :1, None].expand(-1, 1, 3))[:, 0], s.gather(1, ends[:, 1:, None].expand(-1, 1, 3))[:, 0]
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
        seg = torch.where(second[:, None], torch.stack([zero, 1 - t2, t2], -1), torch.stack([1 - t1, t1, zero], -1))
        w = torch.where(flat[:, None], seg, w)
    return w


#: Shewchuk's (1997) bound on the rounding of a two-dimensional orientation
#: determinant evaluated in float64 from rounded coordinate differences, ``(3 + 16
#: e) e`` with ``e = 2^-53`` times the sum of its two products' magnitudes.
_ORIENT_ERRBOUND = (3 + 16 * 2.0**-53) * 2.0**-53


def _two_sum(a, b):
    """``a + b`` exactly, as ``x + y`` (Knuth)."""
    x = a + b
    bv = x - a
    return x, (a - (x - bv)) + (b - bv)


def _two_product(a, b):
    """``a b`` exactly, as ``x + y`` (Dekker 1971, by splitting; no fused
    multiply-add)."""
    x = a * b
    ca, cb = 134217729.0 * a, 134217729.0 * b
    ah, bh = ca - (ca - a), cb - (cb - b)
    al, bl = a - ah, b - bh
    return x, al * bl - (((x - ah * bh) - al * bh) - ah * bl)


def _zero_area(simplex: torch.Tensor, cells: torch.Tensor):
    """Whether each cell ``simplex`` ``(F, d, d)`` (float64, the coordinates as the
    search reads them) has exactly zero measure ``(F,)``, and whether any has. A
    segment has zero length when its two vertices coincide (welded, one index). A
    triangle has zero area when its cross product is exactly zero (two or three
    vertices coincide, or three distinct ones are collinear). Each component of the
    cross product is first evaluated in float64 with Shewchuk's (1997) error bound,
    which proves it nonzero for every cell of nonzero area but the very thinnest.
    Those left are evaluated exactly: ``a x b + b x c + c x a`` as six exact products
    (:func:`_two_product`) summed as a nonoverlapping expansion (Shewchuk's
    Grow-Expansion), zero exactly when its every term is. The test is exact unless an
    intermediate overflows or underflows: always for float32 coordinates (in float64
    their products lie on a grid of ``2^-298``), and for float64 ones when every
    nonzero magnitude lies in ``[2^-485, 2^510]``. Torch evaluates each operation by
    itself, without a fused multiply-add, and the caller runs in a custom op, which
    torch.compile does not trace."""
    if simplex.shape[-1] == 2:
        flat = cells[:, 0] == cells[:, 1]
        return flat, bool(flat.any())
    a, b, c = simplex.unbind(1)
    e1, e2 = b - a, c - a
    axes = ((1, 2), (2, 0), (0, 1))
    certain = torch.zeros_like(a[:, 0], dtype=torch.bool)
    for i, j in axes:
        left, right = e1[:, i] * e2[:, j], e1[:, j] * e2[:, i]
        certain |= (left - right).abs() > _ORIENT_ERRBOUND * (left.abs() + right.abs())
    rows = (~certain).nonzero()[:, 0]
    flat = torch.zeros_like(certain)
    if rows.numel() == 0:
        return flat, False
    v = simplex[rows].unbind(1)
    zero = torch.ones_like(rows, dtype=torch.bool)
    for i, j in axes:
        zero &= (torch.stack(_cross_expansion(*v, i, j)) == 0).all(0)
    flat[rows] = zero
    return flat, bool(zero.any())


def _cross_expansion(v0, v1, v2, i, j):
    """The component ``(i, j)`` of ``(v1 - v0) x (v2 - v0)`` of each row, exactly, as
    ``v0 x v1 + v1 x v2 + v2 x v0``: six exact products (:func:`_two_product`) summed
    as a nonoverlapping expansion (Shewchuk's Grow-Expansion), a list of terms of
    increasing magnitude (zeros anywhere)."""
    terms = []
    for x, y in ((v0, v1), (v1, v2), (v2, v0)):
        terms += [*_two_product(x[:, i], y[:, j]), *(-t for t in _two_product(x[:, j], y[:, i]))]
    expansion = terms[:1]
    for t in terms[1:]:
        grown = []
        for e in expansion:
            t, h = _two_sum(t, e)
            grown.append(h)
        expansion = [*grown, t]
    return expansion


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
#:    one denominator that cancels, ``denom~``, is kept as ``(1 - theta) D``, so its
#:    amplification is in the bound; the products of two error terms (``(5 u L l)^2`` against ``20 u L^2
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
    kappa = torch.where(rho_low > 0, _ERICSON_K * 2.0**-53 / (rho_low * rho_low * longest), torch.inf)
    return bound / cross, kappa, longest


def _ericson_error(dist, delta, longest):
    """The bound on the error of Ericson's distance ``dist`` to a cell whose point
    is misplaced by at most ``delta`` (``kappa l^2``, :func:`_thin_bounds`) along the
    face: ``min(delta, delta^2 / (2 (dist - delta)))``, ``delta`` where ``dist <=
    delta``, and ``inf`` where ``delta`` exceeds the cell's longest edge (the bound's
    analysis then does not hold), as :func:`_ericson_error_wp`."""
    eps = torch.where(dist > delta, torch.minimum(delta, delta * delta / (2 * (dist - delta))), delta)
    return torch.where(delta <= longest, eps, torch.inf)


@wp.func
def _ericson_error_wp(dist: wp.float64, delta: wp.float64, longest: wp.float64) -> wp.float64:
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
    length2 = torch.stack([((simplex[:, j] - simplex[:, i]) ** 2).sum(-1) for i, j in pairs], 1)
    slots = torch.tensor([[0, 1, 2], [0, 2, 1], [1, 2, 0]], device=q.device)[length2.argmax(1)]
    a, b, c = (simplex.gather(1, slots[:, k, None, None].expand(-1, 1, 3))[:, 0] for k in range(3))
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
    return torch.stack(weights, 1).gather(1, best[:, None, None].expand(-1, 1, 3))[:, 0], dists.gather(1, best[:, None])[:, 0]


def _covered(cells: torch.Tensor, flat: torch.Tensor, n_vertices: int) -> torch.Tensor:
    """Whether each zero-area cell (``flat``, :func:`_zero_area`; canonical, welded
    ``cells``) lies in the cells of nonzero measure, ``(F,)``: a point when it is one
    of their vertices; in three dimensions a segment when its edges between
    consecutive distinct vertices are edges of theirs, or its edge from first to last
    vertex is (on a line the canonical order is the order along it, so that edge
    holds the middle vertex)."""
    regular = cells[~flat]
    vertex = torch.zeros(n_vertices, dtype=torch.bool, device=cells.device)
    vertex[regular.reshape(-1)] = True
    if cells.shape[1] == 2:
        return flat & vertex[cells[:, 0]]
    keys = torch.unique(
        torch.minimum(regular[:, [0, 0, 1]], regular[:, [1, 2, 2]]) * n_vertices
        + torch.maximum(regular[:, [0, 0, 1]], regular[:, [1, 2, 2]])
    )

    def edge(a, b):
        k = torch.minimum(a, b) * n_vertices + torch.maximum(a, b)
        return keys[torch.searchsorted(keys, k).clamp_max(max(keys.shape[0] - 1, 0))] == k if keys.numel() else torch.zeros_like(flat)

    v0, v1, v2 = cells.unbind(1)
    segment = ((v0 == v1) | edge(v0, v1)) & ((v1 == v2) | edge(v1, v2)) | edge(v0, v2)
    return flat & torch.where(v0 == v2, vertex[v0], segment)


def _hull_candidates(cc: torch.Tensor, subsets: list[torch.Tensor]):
    """The candidates of :func:`_min_norm_points` for the rows ``cc`` ``(n, K, d)``
    on the subsets (one ``(S, m)`` tensor of indices per size ``m``): each subset's
    least-norm affine point with non-negative weights ``(n, S, d)`` (infinite where
    there is none), its norm where it meets the optimality condition to rounding,
    and its duality gap where it separates the origin (each ``(n, S)``, infinite
    elsewhere)."""
    d = cc.shape[-1]
    xs = []
    for sub in subsets:
        p = cc[:, sub]  # (n, S, m, d)
        if sub.shape[1] == 2:
            xs.append(0.5 * (p[:, :, 0] + p[:, :, 1]))
            continue
        ### (elementwise products, not matmuls: the matrices are 3 x 3)
        gram = (p[..., :, None, :] * p[..., None, :, :]).sum(-1)
        lam, info = torch.linalg.solve_ex(gram, p.new_ones(p.shape[:-1]))
        ok = (info == 0) & torch.isfinite(lam).all(-1) & (lam >= 0).all(-1)
        x = ((lam / lam.sum(-1, keepdim=True))[..., None] * p).sum(-2)
        xs.append(torch.where(ok[..., None], x, torch.inf))
    x = torch.cat(xs, 1)
    ### Rounding of a d-term dot product of unit vectors, on each side.
    slack = 2 * d * torch.finfo(cc.dtype).eps
    low, sq = (cc[:, None] * x[:, :, None]).sum(-1).amin(-1), (x * x).sum(-1)
    norm = torch.where(low >= sq - slack, x.norm(dim=-1), torch.inf)
    gap = torch.where((low > slack) & torch.isfinite(sq), sq - low, torch.inf)
    return x, norm, gap


def _min_norm_points(c: torch.Tensor) -> torch.Tensor:
    """For each row of ``c`` ``(n, K, d)`` (``K`` distinct unit vectors), the point
    of their convex hull closest to the origin ``(n, d)``. That point lies in the
    relative interior of a face spanned by at most ``d`` of them (Caratheodory), where
    it is the face's least-norm affine point with non-negative weights, and every
    such point lies in the hull. So it is the least-norm one over those faces (a
    pair's midpoint, since the vectors have equal norms; the first in the enumeration
    at equal norms) that meets the optimality condition ``c_k . x >= |x|^2``. Only
    the hull's least-norm point meets it; other faces' points can match its norm to
    rounding where vectors nearly coincide. When none meets it to rounding, the
    origin is outside the hull exactly when a candidate separates it (``c_k . x > 0``
    for every ``k``; the least-norm point does). The result is then the candidate
    with the least duality gap ``g = |x|^2 - min_k c_k . x`` (``|x - x*| <= sqrt(2
    g)``), and otherwise 0. ``c`` comes in a canonical order, so the result is a
    function of the set. The work is chunked over the rows. Where one row's subsets
    and directions overflow a chunk (a vertex shared by hundreds of cells), it runs
    one row at a time over chunks of its subsets, with the same result. The work
    grows as ``K**(d + 1)``: a query tied with hundreds of cells (on the axis of a
    faceted tube, at a sphere's centre) takes seconds. An active-set solver (Wolfe's)
    over every tied row at once, certified as here, would remove that term; it is a
    follow-up."""
    n, k, d = c.shape
    if k == 1:
        return c[:, 0]
    sizes, budget = range(2, min(k, d) + 1), _SEARCH_PAIRS // 8
    total = sum(math.comb(k, m) for m in sizes)
    if total * k <= budget:
        subsets = [torch.tensor(list(itertools.combinations(range(k), m)), device=c.device) for m in sizes]
        chunk = max(1, budget // total)
        out = []
        for s in range(0, n, chunk):
            x, norm, gap = _hull_candidates(c[s : s + chunk], subsets)
            certified = torch.isfinite(norm).any(1)
            pick = torch.where(certified, norm.argmin(1), gap.argmin(1))
            best = x.gather(1, pick[:, None, None].expand(-1, 1, d))[:, 0]
            out.append(torch.where((certified | torch.isfinite(gap).any(1))[:, None], best, 0.0))
        return torch.cat(out)
    ### The subsets are generated in the same order, a chunk at a time, and the first
    ### least norm and the first least gap kept (a later chunk only when smaller).
    out = []
    for cc in c.split(1):
        least = [c.new_full((1,), torch.inf), c.new_full((1,), torch.inf)]
        best = [c.new_zeros(1, d), c.new_zeros(1, d)]
        for m in sizes:
            subsets = itertools.combinations(range(k), m)
            while sub := list(itertools.islice(subsets, max(1, budget // k))):
                x, *scores = _hull_candidates(cc, [torch.tensor(sub, device=c.device)])
                for j, score in enumerate(scores):
                    i = score.argmin(1, keepdim=True)
                    value = score.gather(1, i)[:, 0]
                    smaller = value < least[j]
                    least[j] = torch.where(smaller, value, least[j])
                    point = x.gather(1, i[..., None].expand(-1, 1, d))[:, 0]
                    best[j] = torch.where(smaller[:, None], point, best[j])
        out.append(
            torch.where(
                torch.isfinite(least[0])[:, None],
                best[0],
                torch.where(torch.isfinite(least[1])[:, None], best[1], 0.0),
            )
        )
    return torch.cat(out)


def _grouped_min_norm(group: torch.Tensor, c: torch.Tensor, n_groups: int) -> torch.Tensor:
    """The least-norm point of the hull of each group's distinct directions
    (:func:`_min_norm_points`), ``(n_groups, d)``, zero for an empty group; the rows
    ``c`` ``(P, d)`` belong to the groups ``group`` ``(P,)``. The directions are
    de-duplicated and sorted per group, so each result is a function of its set."""
    out = c.new_zeros(n_groups, c.shape[1])
    if group.numel() == 0:
        return out
    rows = torch.unique(torch.cat([group[:, None].to(c.dtype), c], 1), dim=0)
    group, c = rows[:, 0].long(), rows[:, 1:]
    ids, counts = torch.unique_consecutive(group, return_counts=True)
    start = torch.cumsum(counts, 0) - counts
    for k in torch.unique(counts).tolist():
        here = (counts == k).nonzero()[:, 0]
        out[ids[here]] = _min_norm_points(c[start[here, None] + torch.arange(k, device=c.device)])
    return out


@wp.func
def _triangle_distance2(p: Any, a: Any, b: Any, c: Any):
    """The squared distance from ``p`` to the triangle ``abc`` (Ericson 2005, Sec.
    5.1.5), in the precision of the vectors (float64 or float32); NaN on a degenerate
    triangle, which the caller keeps."""
    zero = type(p[0])(0.0)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= zero and d2 <= zero:
        return wp.dot(ap, ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= zero and d4 <= d3:
        return wp.dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= zero and d1 >= zero and d3 <= zero:
        r = ap - ab * (d1 / (d1 - d3))
        return wp.dot(r, r)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= zero and d5 <= d6:
        return wp.dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= zero and d2 >= zero and d6 <= zero:
        r = ap - ac * (d2 / (d2 - d6))
        return wp.dot(r, r)
    va = d3 * d6 - d5 * d4
    if va <= zero and d4 - d3 >= zero and d5 - d6 >= zero:
        r = bp - (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        return wp.dot(r, r)
    denom = va + vb + vc
    r = ap - ab * (vb / denom) - ac * (vc / denom)
    return wp.dot(r, r)


@wp.func
def _beyond(x: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, thin: wp.vec2d, r: wp.float64, r2: wp.float64) -> bool:
    """Whether the triangle ``abc`` lies provably farther than ``r`` (``r2`` its
    square) from ``x``: Ericson's float64 distance less the bound on its error
    (:func:`_ericson_error_wp`, ``thin`` the cell's ``kappa`` and longest edge,
    :func:`_thin_bounds`) exceeds it; never on a degenerate triangle (NaN)."""
    d2 = _triangle_distance2(x, a, b, c)
    far = d2 > r2
    if far:
        l2 = wp.max(wp.max(wp.length_sq(x - a), wp.length_sq(x - b)), wp.length_sq(x - c))
        dt = wp.sqrt(d2)
        far = dt - _ericson_error_wp(dt, thin[0] * l2, thin[1]) > r
    return far


@wp.struct
class _NodeChildren:
    """What the seed and ball traversals read of an inner node, in one record: its two
    children's boxes, slabs (:func:`_node_slabs`) and references (a node, or ``-1 -
    cell`` at a leaf)."""

    lo_l: wp.vec3f
    hi_l: wp.vec3f
    lo_r: wp.vec3f
    hi_r: wp.vec3f
    slab_l: wp.vec3f
    slab_r: wp.vec3f
    slab_lo_l: wp.float32
    slab_hi_l: wp.float32
    slab_lo_r: wp.float32
    slab_hi_r: wp.float32
    ref_l: wp.int32
    ref_r: wp.int32


@wp.kernel
def _ball_candidates_kernel(
    node_lo: wp.array(dtype=wp.vec3f),
    node_hi: wp.array(dtype=wp.vec3f),
    root: wp.int32,
    children: wp.array(dtype=_NodeChildren),
    corners: wp.array(dtype=wp.vec3d),
    thin: wp.array(dtype=wp.vec2d),
    planes: wp.array(dtype=wp.vec4f),
    query: wp.array(dtype=wp.vec3d),
    query32: wp.array(dtype=wp.vec3f),
    radius: wp.array(dtype=wp.float64),
    radius32: wp.array(dtype=wp.float32),
    reach32: wp.array(dtype=wp.float32),
    rows: wp.array(dtype=wp.int64),
    cap: wp.int64,
    count: wp.array(dtype=wp.int64),
    cells_out: wp.array(dtype=wp.int64),
    overflow_query: wp.array(dtype=wp.int64),
    overflow_cell: wp.array(dtype=wp.int64),
    overflow_count: wp.array(dtype=wp.int64),
):
    """Every cell within ``radius`` of the query ``rows[tid]``. A depth-first
    traversal of the BVH from ``root``, left child first, reads one record per inner
    node (``children``): both children are tested where their parent is read, and the
    right one waits on a stack. It prunes the nodes whose boxes lie farther than
    ``radius32`` (float32 boxes rounded outward, a radius padded over float32
    rounding) and the inner nodes whose slabs lie farther than ``reach32``. At a leaf,
    a float32 lower bound on the cell's distance first rejects the far cells: the
    distance to the cell's plane and beyond its farthest edge, from the unit normals
    in ``planes`` (the plane's, then each edge's outward one in the plane, with their
    offsets), against ``reach32``. Then it prunes the cells farther than ``radius``
    (float64, :func:`_beyond`; ``corners`` holds each cell's three vertices, ``thin``
    its ``kappa`` and longest edge). It writes the first ``cap`` cells from ``tid *
    cap`` on, and the others as (query, cell) pairs to the overflow list while it has
    room (its counter, shared by every thread, counts them all). ``count`` gets the
    query's total."""
    tid = wp.tid()
    q = rows[tid]
    x = query[q]
    x32 = query32[q]
    r2 = radius[q] * radius[q]
    r2_32 = radius32[q] * radius32[q]
    reach2 = reach32[q] * reach32[q]
    zero = wp.float32(0.0)
    k = wp.int64(0)
    stack = _nearest_stack()
    top = int(0)
    ref = root
    go = _box_gap2(node_lo[0], node_hi[0], x32) <= r2_32
    while go or top > 0:
        if not go:
            top -= 1
            ref = stack[top]
        go = False
        if ref < wp.int32(0):
            c = wp.int64(-1) - wp.int64(ref)
            j = c * wp.int64(4)
            h = planes[j]
            s = h[0] * x32[0] + h[1] * x32[1] + h[2] * x32[2] - h[3]
            o = zero
            for e in range(1, 4):
                h = planes[j + wp.int64(e)]
                o = wp.max(o, h[0] * x32[0] + h[1] * x32[1] + h[2] * x32[2] - h[3])
            ### (not ``<=``: a NaN, from a degenerate cell, keeps it)
            i = c * wp.int64(3)
            if not (s * s + o * o > reach2) and not _beyond(
                x, corners[i], corners[i + wp.int64(1)], corners[i + wp.int64(2)], thin[c], radius[q], r2
            ):
                if k < cap:
                    cells_out[wp.int64(tid) * cap + k] = c
                else:
                    slot = wp.atomic_add(overflow_count, 0, wp.int64(1))
                    if slot < overflow_query.shape[0]:
                        overflow_query[slot] = q
                        overflow_cell[slot] = c
                k += wp.int64(1)
        else:
            ch = children[ref]
            ### (a slab's distance is never NaN: a NaN direction or offset counts as 0)
            hit_l = _box_gap2(ch.lo_l, ch.hi_l, x32) <= r2_32
            if hit_l and ch.ref_l >= wp.int32(0):
                hit_l = _slab_gap2(ch.slab_l, ch.slab_lo_l, ch.slab_hi_l, x32) <= reach2
            hit_r = _box_gap2(ch.lo_r, ch.hi_r, x32) <= r2_32
            if hit_r and ch.ref_r >= wp.int32(0):
                hit_r = _slab_gap2(ch.slab_r, ch.slab_lo_r, ch.slab_hi_r, x32) <= reach2
            if hit_l:
                if hit_r:
                    stack[top] = ch.ref_r
                    top += 1
                ref = ch.ref_l
                go = True
            elif hit_r:
                ref = ch.ref_r
                go = True
    count[tid] = k


#: Candidate slots per query of :func:`_ball_candidates`, before its overflow list.
#: Its second pass, which runs only when that list is full, takes about a whole
#: traversal's time however few queries it holds.
#: A performance setting: the outputs do not depend on it; re-tune it when the search
#: changes.
_BALL_SLOTS = 64


@wp.kernel
def _node_boxes_kernel(
    inner: wp.array(dtype=wp.int64),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    lo: wp.array(dtype=wp.vec3d),
    hi: wp.array(dtype=wp.vec3d),
):
    """The box of each node ``inner[tid]`` from its two children's."""
    i = inner[wp.tid()]
    lo[i] = wp.min(lo[left[i]], lo[right[i]])
    hi[i] = wp.max(hi[left[i]], hi[right[i]])


def _bottom_up(kernel, levels, arrays, ref: torch.Tensor):
    """``kernel`` over the inner nodes of each level of a BVH, deepest first, one
    launch per level (``arrays`` follow the node ids)."""
    device, stream = FunctionSpec.warp_launch_context(ref)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        for inner in reversed(levels):
            wp.launch(kernel, dim=inner.shape[0], inputs=[wp.from_torch(inner, dtype=wp.int64), *arrays], device=device, stream=stream)


def _morton_order(simplex: torch.Tensor) -> torch.Tensor:
    """The cells ``simplex`` ``(F, 3, 3)`` in the Morton order of their centroids in the
    boundary's canonical frame (:func:`~physicsnemo.mesh.spatial._frame.canonical_frame`),
    stable at equal codes."""
    f = simplex.shape[0]
    e = simplex[:, 1:] - simplex[:, :1]
    frame = canonical_frame(simplex.mean(1), torch.linalg.cross(e[:, 0], e[:, 1], dim=-1).norm(dim=-1))
    ### Column products, not a GEMM (see frame_coordinates).
    return _compute_morton_codes(frame_coordinates(simplex.reshape(-1, 3), frame).reshape(f, 3, 3).mean(1)).argsort(stable=True)


def _tree_boxes(simplex, left, right, levels, leaf_cell):
    """Each node's box ``(N, 3)`` twice (least and greatest corner): its cell's at a
    leaf, its children's above (one launch per level), in float32 rounded outward."""
    leaf = leaf_cell >= 0
    lo, hi = simplex.new_full((left.shape[0], 3), torch.inf), simplex.new_full((left.shape[0], 3), -torch.inf)
    lo[leaf], hi[leaf] = simplex[leaf_cell[leaf]].amin(1), simplex[leaf_cell[leaf]].amax(1)
    children = [wp.from_torch(c.contiguous(), dtype=wp.int64) for c in (left, right)]
    _bottom_up(_node_boxes_kernel, levels, [*children, wp.from_torch(lo, dtype=wp.vec3d), wp.from_torch(hi, dtype=wp.vec3d)], simplex)
    lo32, hi32 = lo.float(), hi.float()
    lo32 = torch.where(lo32.double() > lo, lo32.nextafter(lo32.new_tensor(-torch.inf)), lo32)
    hi32 = torch.where(hi32.double() < hi, hi32.nextafter(hi32.new_tensor(torch.inf)), hi32)
    return lo32, hi32


@wp.kernel
def _preorder_kernel(
    f: wp.int64,
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    escape: wp.array(dtype=wp.int64),
    leaf_cell: wp.array(dtype=wp.int64),
    depth: wp.array(dtype=wp.int32),
):
    """The midpoint-split tree over ``f`` sorted cells, its nodes in depth-first
    preorder: the thread of sorted position ``i`` descends from the root and writes
    every node whose range starts at ``i`` (a node at ``p`` over ``[s, s + n)`` has its
    left child at ``p + 1`` over ``[s, s + n // 2)``, its right child at ``p + 2 (n //
    2)``, and the node after its subtree at ``p + 2 n - 1``); leaf ``i`` holds the
    cell of sorted position ``i``."""
    i = wp.int64(wp.tid())
    total = wp.int64(2) * f - wp.int64(1)
    p = wp.int64(0)
    s = wp.int64(0)
    n = f
    d = wp.int32(0)
    while n > wp.int64(1):
        h = n // wp.int64(2)
        if i == s:
            left[p] = p + wp.int64(1)
            right[p] = p + wp.int64(2) * h
            e = p + wp.int64(2) * n - wp.int64(1)
            if e >= total:
                e = wp.int64(-1)
            escape[p] = e
            leaf_cell[p] = wp.int64(-1)
            depth[p] = d
        if i < s + h:
            p = p + wp.int64(1)
            n = h
        else:
            p = p + wp.int64(2) * h
            s = s + h
            n = n - h
        d += wp.int32(1)
    left[p] = wp.int64(-1)
    right[p] = wp.int64(-1)
    e = p + wp.int64(1)
    if e >= total:
        e = wp.int64(-1)
    escape[p] = e
    leaf_cell[p] = i
    depth[p] = d


def _parents(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Each node's parent ``(N,)``, -1 at the root, from the nodes' children ``left``
    and ``right`` (-1 at a leaf)."""
    node = torch.arange(left.shape[0], device=left.device)
    ### (a leaf's -1 lands in a spare first slot)
    parent = torch.full((left.shape[0] + 1,), -1, dtype=torch.int64, device=left.device)
    parent[left + 1] = node
    parent[right + 1] = node
    return parent[1:]


def _inner_per_level(f: int) -> list[int]:
    """The inner nodes at each depth, root first, of the midpoint-split tree over
    ``f`` items with one per leaf (``build_lbvh_topology``'s host recurrence)."""
    sizes, counts = {f: 1}, []
    while any(s > 1 for s in sizes):
        counts.append(sum(c for s, c in sizes.items() if s > 1))
        nxt = {}
        for s, c in sizes.items():
            if s > 1:
                for t in (s // 2, s - s // 2):
                    nxt[t] = nxt.get(t, 0) + c
        sizes = nxt
    return counts


def _preorder_cell_tree(simplex: torch.Tensor):
    """A BVH over the cells ``simplex`` ``(F, 3, 3)``, one cell per leaf: the
    midpoint-split LBVH of :class:`physicsnemo.mesh.spatial.BVH`, built by one kernel,
    its nodes in depth-first preorder, so a node's left child is the next node. It
    holds what a stackless traversal needs: each node's two children; its escape
    pointer (the node after its subtree, -1 at the end); the inner nodes of each
    level, root first; each node's cell (-1 inside); and each node's box ``(N, 3)``
    twice (least and greatest corner), in float32 rounded outward for the traversals'
    pruning. The cells are put in Morton order in the boundary's canonical frame
    (:func:`~physicsnemo.mesh.spatial._frame.canonical_frame`), so the tree, and the
    far field of :func:`_packed_winding_sum` on it, is the same in every pose of the
    case. The boxes are in the queries' frame. Leaf ``i`` holds row ``i`` of
    ``simplex[order]``, returned with ``order`` ``(F,)``."""
    f, dev = simplex.shape[0], simplex.device
    order = _morton_order(simplex)
    n = 2 * f - 1
    left, right, escape, leaf_cell = (torch.empty(n, dtype=torch.int64, device=dev) for _ in range(4))
    depth = torch.empty(n, dtype=torch.int32, device=dev)
    device, stream = FunctionSpec.warp_launch_context(simplex)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _preorder_kernel,
            dim=f,
            inputs=[f, *(wp.from_torch(t, dtype=wp.int64) for t in (left, right, escape, leaf_cell)), wp.from_torch(depth, dtype=wp.int32)],
            device=device,
            stream=stream,
        )
    counts = _inner_per_level(f)
    inner = torch.argsort(torch.where(left >= 0, depth, torch.iinfo(torch.int32).max), stable=True)[: sum(counts)]
    levels = list(inner.split(counts))
    s_sorted = simplex[order]
    return (left, right, escape, levels, leaf_cell, *_tree_boxes(s_sorted, left, right, levels, leaf_cell)), order, s_sorted


def _ball_candidates(query: torch.Tensor, radius: torch.Tensor, simplex: torch.Tensor, tree, children, reach, flat=None):
    """The (query, cell) pairs ``(P,)``, ``(P,)`` of every cell of ``simplex`` ``(F,
    3, 3)`` within ``radius`` ``(Q,)`` of its query ``(Q, 3)``, all float64. A Warp
    traversal of the BVH ``tree`` over the cells (:func:`_preorder_cell_tree`, its
    nodes' ``children`` from :func:`_child_records`) prunes with its boxes and the
    queries in float32 (rounded outward and padded), and takes the distances to the
    cells in float64.

    An inner node is also skipped where its slab (:func:`_node_slabs`) lies farther
    than the padded radius of the cells' float32 lower bound. With the slab's
    direction ``n`` as stored (``|n| <= 1 + 2^-23``) and offsets ``lo <= n . y <= hi``
    over the node's cells (float64, rounded outward), a point ``y`` of a cell is at
    least ``(n . x - hi) / |n|`` from ``x``. The float32 product, the direction's
    length and the float64 offsets' rounding stay far inside that pad (``32 u32`` of
    the magnitudes and ``1e-5`` of the radius), so every cell within ``radius`` is
    still found. A cell the search keeps only for lack of a certificate (a zero-area
    or thin one) can go with a node it lies beyond. It is then farther than
    ``radius``, and no output of the caller depends on such a pair, wherever the
    pairs' distances are numbers (the magnitudes ``_BAND_UNITS`` covers): it is not
    the least, it fails the tie test, and a pair that ties on a vertex or an edge of
    its cell would put the cell within ``radius``.

    One pass fills ``_BALL_SLOTS`` slots per query and an overflow list of ``Q``
    pairs. When that list is full, the queries that overflow their slots are searched
    again with as many slots as they need, so no candidate is dropped. The pairs come
    in no particular order (the caller sorts them). The queries are traversed in
    Morton order, so neighbouring threads visit neighbouring nodes. ``reach`` is the
    greatest norm of a vertex.

    The zero-area cells ``flat`` ``(F,)`` (:func:`_zero_area`) have no plane, and the
    float64 distance is not defined on them: they are kept wherever their boxes and
    the slabs above them are reached. Nor are the planes used for a cell whose float64
    normal is not certified to its float32 rounding (``delta > u32``,
    :func:`_thin_bounds`): its float64 distance rejects it only beyond the bound on
    its error (:func:`_beyond`)."""
    dev = query.device
    lo, hi = tree[5:]
    ### Each cell's unit normal and its edges' outward unit normals in its plane
    ### (outward by the vertex order alone, the edges taken cyclically), with their
    ### offsets: a point's distance to the cell is at least the root sum of squares
    ### of its height over the plane and of its largest edge offset.
    a, b, c = simplex.unbind(1)
    n = torch.nn.functional.normalize(torch.linalg.cross(b - a, c - a, dim=-1), dim=-1, eps=0.0)
    rows_ = [torch.cat([n, (n * a).sum(-1, keepdim=True)], -1)]
    for p0, p1 in ((a, b), (b, c), (c, a)):
        m = torch.nn.functional.normalize(torch.linalg.cross(p1 - p0, n, dim=-1), dim=-1, eps=0.0)
        rows_.append(torch.cat([m, (m * p0).sum(-1, keepdim=True)], -1))
    planes = torch.stack([r.float() for r in rows_], 1).reshape(-1, 4)
    del rows_, n
    delta, kappa, longest = _thin_bounds(simplex)
    ### (NaN fails every rejection test of the kernel)
    unsure = delta > torch.finfo(torch.float32).eps / 2
    corners = simplex
    if flat is not None:
        unsure = unsure | flat
        corners = torch.where(flat[:, None, None], torch.nan, simplex)
    planes = torch.where(unsure.repeat_interleave(4)[:, None], torch.nan, planes)
    q32 = query.float()
    ### (the query's own rounding, then a relative pad far above float32's on the
    ### box test's few operations)
    r32 = ((radius + (query - q32.double()).norm(dim=-1)) * (1 + 1e-5)).float()
    r32 = r32.nextafter(r32.new_tensor(torch.inf))
    ### (the lower bound's float32 rounding: the products and offsets, against the
    ### coordinates' magnitude)
    reach32 = ((r32.double() + 16 * torch.finfo(torch.float32).eps * (query.norm(dim=-1) + reach)) * (1 + 1e-5)).float()
    device, stream = FunctionSpec.warp_launch_context(query)
    fixed = [
        wp.from_torch(lo.contiguous(), dtype=wp.vec3f),
        wp.from_torch(hi.contiguous(), dtype=wp.vec3f),
        ### (one cell: the root is its leaf)
        0 if simplex.shape[0] > 1 else -1,
        children[0],
        wp.from_torch(corners.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
        wp.from_torch(torch.stack([kappa, longest], -1).contiguous(), dtype=wp.vec2d),
        wp.from_torch(planes.contiguous(), dtype=wp.vec4f),
        wp.from_torch(query.contiguous(), dtype=wp.vec3d),
        wp.from_torch(q32.contiguous(), dtype=wp.vec3f),
        wp.from_torch(radius.contiguous(), dtype=wp.float64),
        wp.from_torch(r32.contiguous(), dtype=wp.float32),
        wp.from_torch(reach32.contiguous(), dtype=wp.float32),
    ]
    ### One overflow list for every launch, its counter atomic (int64: an int32 one
    ### would wrap past 2**31 candidates to a negative slot).
    overflow_query, overflow_cell = (torch.empty(query.shape[0], dtype=torch.int64, device=dev) for _ in range(2))
    overflow_count = torch.zeros(1, dtype=torch.int64, device=dev)
    overflow = [wp.from_torch(t, dtype=wp.int64) for t in (overflow_query, overflow_cell, overflow_count)]

    def search(rows, cap):
        count = torch.zeros(rows.shape[0], dtype=torch.int64, device=dev)
        out = torch.empty(rows.shape[0] * cap, dtype=torch.int64, device=dev)
        ### Blocks of rows whose slots Warp takes as one array (under 2**31 entries:
        ### 33,554,431 rows of 64); one block below that.
        block = (WARP_MAX_ELEMENTS - 1) // cap
        with FunctionSpec.warp_stream_scope(stream):
            wp.init()
            for s in range(0, rows.shape[0], block):
                e = min(s + block, rows.shape[0])
                wp.launch(
                    _ball_candidates_kernel,
                    dim=e - s,
                    inputs=[*fixed, wp.from_torch(rows[s:e], dtype=wp.int64), cap,
                            wp.from_torch(count[s:e], dtype=wp.int64),
                            wp.from_torch(out[s * cap : e * cap], dtype=wp.int64), *overflow],
                    device=device,
                    stream=stream,
                )
        return count, out.reshape(-1, cap)

    rows = _compute_morton_codes(query).argsort()
    count, out = search(rows, _BALL_SLOTS)
    n_overflow = int(overflow_count[0])
    if n_overflow <= overflow_query.shape[0]:
        keep = torch.arange(_BALL_SLOTS, device=dev) < count[:, None]
        li, pk = rows[:, None].expand_as(out)[keep], out[keep]
        return torch.cat([li, overflow_query[:n_overflow]]), torch.cat([pk, overflow_cell[:n_overflow]])
    over = count > _BALL_SLOTS
    keep = (torch.arange(_BALL_SLOTS, device=dev) < count[:, None]) & ~over[:, None]
    li, pk = rows[:, None].expand_as(out)[keep], out[keep]
    ### (the list is full, so some query overflowed its slots)
    rows = rows[over]
    count, out = search(rows, int(count.max()))
    keep = torch.arange(out.shape[1], device=dev) < count[:, None]
    return torch.cat([li, rows[:, None].expand_as(out)[keep]]), torch.cat([pk, out[keep]])


#: Entries of the traversal stacks of :func:`_nearest_cells` and
#: :func:`_ball_candidates`; a midpoint-split tree of
#: ``F`` cells is ``ceil(log2 F)`` deep and needs one more. The kernels take int32 node
#: ids, so ``2 F - 1 < 2^31`` and the tree is at most 30 deep.
_NEAREST_STACK = 64
_nearest_stack = wp.types.vector(length=_NEAREST_STACK, dtype=wp.int32)
_nearest_gaps = wp.types.vector(length=_NEAREST_STACK, dtype=wp.float32)

#: The first run of :func:`_nearest_cell_kernel` skips a node whose box's squared
#: distance times this (1.05 squared) reaches the nearest cell's so far. A performance
#: setting: the second run is exact whatever its value. Inside a finely meshed far
#: field most cells are nearly as far as the nearest one, so a single exact run opens
#: nearly every box that holds the query before it meets a near cell, and a kernel
#: takes as long as its slowest thread. Re-tune it when the search changes.
_NEAREST_FIRST_PASS = 1.05**2

#: :func:`_node_slabs` costs one more pass over the cells, in which every leaf walks to
#: the root, and saves the seed search and the ball search a bounded time per query. So
#: the slabs are built up to this many cells per query; above it both searches skip
#: nodes by their boxes alone (:func:`_no_slabs`). A performance setting: the outputs
#: are the same with the slabs as without; re-tune it when the searches change.
_SLAB_CELLS_PER_QUERY = 32


@wp.func
def _vec3f(v: wp.vec3d) -> wp.vec3f:
    return wp.vec3f(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def _box_gap2(lo: wp.vec3f, hi: wp.vec3f, x: wp.vec3f) -> wp.float32:
    """The squared distance from ``x`` to the box ``[lo, hi]``."""
    zero = wp.float32(0.0)
    g0 = wp.max(wp.max(lo[0] - x[0], x[0] - hi[0]), zero)
    g1 = wp.max(wp.max(lo[1] - x[1], x[1] - hi[1]), zero)
    g2 = wp.max(wp.max(lo[2] - x[2], x[2] - hi[2]), zero)
    return g0 * g0 + g1 * g1 + g2 * g2


@wp.func
def _slab_gap2(n: wp.vec3f, lo: wp.float32, hi: wp.float32, x: wp.vec3f) -> wp.float32:
    """The squared distance from ``x`` to the slab ``lo <= n . y <= hi``."""
    s = wp.dot(n, x)
    g = wp.max(wp.max(lo - s, s - hi), wp.float32(0.0))
    return g * g


@wp.kernel
def _node_sum_kernel(
    inner: wp.array(dtype=wp.int64),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    value: wp.array(dtype=wp.vec3d),
):
    """The sum of each node ``inner[tid]``'s two children's vectors."""
    i = inner[wp.tid()]
    value[i] = value[left[i]] + value[right[i]]


@wp.kernel
def _node_slab_kernel(
    leaf_cell: wp.array(dtype=wp.int64),
    parent: wp.array(dtype=wp.int64),
    corners: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    lo: wp.array(dtype=wp.float64),
    hi: wp.array(dtype=wp.float64),
):
    """The extent of each node's cells along the node's direction. The thread of each
    leaf walks up from it; a minimum and a maximum are exact, so the result does not
    depend on the threads' order."""
    node = wp.tid()
    c = leaf_cell[node]
    if c < wp.int64(0):
        return
    i = c * wp.int64(3)
    a = corners[i]
    b = corners[i + wp.int64(1)]
    c2 = corners[i + wp.int64(2)]
    p = wp.int64(node)
    while p >= wp.int64(0):
        n = direction[p]
        sa = wp.dot(n, a)
        sb = wp.dot(n, b)
        sc = wp.dot(n, c2)
        wp.atomic_min(lo, p, wp.min(wp.min(sa, sb), sc))
        wp.atomic_max(hi, p, wp.max(wp.max(sa, sb), sc))
        p = parent[p]


def _node_slabs(simplex: torch.Tensor, parity: torch.Tensor, tree):
    """Per node of ``tree`` (:func:`_preorder_cell_tree`, its leaves holding the rows of
    ``simplex`` ``(F, 3, 3)`` with orientations ``parity`` ``(F,)``), a slab that holds
    the node's cells: a direction ``(N, 3)`` and the least and the greatest offset of
    their vertices along it ``(N,)`` twice, in float32. The direction is the mean
    normal of the node's cells (their summed area vectors, normalized; zero where they
    cancel). A point's distance to the node's cells is at least its distance to the
    slab. Where the cells form a patch of a smooth surface the slab is as thin as the
    patch is curved, while the patch's box, unless it lies along the axes, is as thick
    as the patch is wide. :func:`_nearest_cells` uses the slab to skip nodes without a
    pad. The offsets' rounding (outward; the query's own product is not padded) can
    then cost it only a cell nearer by float32's roundoff of the coordinates, the
    accuracy of its float32 distances themselves. :func:`_ball_candidates`, which must
    not miss a cell, tests the slab against a padded radius."""
    left, right, _, levels, leaf_cell, _, _ = tree
    n, leaf = left.shape[0], leaf_cell >= 0
    cell = leaf_cell[leaf]
    e = simplex[cell, 1:] - simplex[cell, :1]
    total = simplex.new_zeros(n, 3)
    total[leaf] = parity[cell, None] * torch.linalg.cross(e[:, 0], e[:, 1], dim=-1)
    children = [wp.from_torch(c.contiguous(), dtype=wp.int64) for c in (left, right)]
    _bottom_up(_node_sum_kernel, levels, [*children, wp.from_torch(total, dtype=wp.vec3d)], simplex)
    direction = torch.nn.functional.normalize(total, dim=-1).float()
    parent = _parents(left, right)
    lo, hi = simplex.new_full((n,), torch.inf), simplex.new_full((n,), -torch.inf)
    device, stream = FunctionSpec.warp_launch_context(simplex)
    with FunctionSpec.warp_stream_scope(stream):
        wp.launch(
            _node_slab_kernel,
            dim=n,
            inputs=[
                wp.from_torch(leaf_cell.contiguous(), dtype=wp.int64),
                wp.from_torch(parent, dtype=wp.int64),
                wp.from_torch(simplex.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
                wp.from_torch(direction.double(), dtype=wp.vec3d),
                wp.from_torch(lo, dtype=wp.float64),
                wp.from_torch(hi, dtype=wp.float64),
            ],
            device=device,
            stream=stream,
        )
    lo32, hi32 = lo.float(), hi.float()
    lo32 = torch.where(lo32.double() > lo, lo32.nextafter(lo32.new_tensor(-torch.inf)), lo32)
    hi32 = torch.where(hi32.double() < hi, hi32.nextafter(hi32.new_tensor(torch.inf)), hi32)
    return direction.contiguous(), lo32, hi32


def _no_slabs(tree, like: torch.Tensor):
    """Slabs that hold every point (a zero direction between infinite offsets), in
    :func:`_node_slabs`' layout: with them :func:`_nearest_cells` and
    :func:`_ball_candidates` skip nodes by their boxes alone."""
    n = tree[0].shape[0]
    hi = torch.full((n,), torch.inf, dtype=torch.float32, device=like.device)
    return torch.zeros(n, 3, dtype=torch.float32, device=like.device), -hi, hi


@wp.kernel
def _pack_children_kernel(
    node_lo: wp.array(dtype=wp.vec3f),
    node_hi: wp.array(dtype=wp.vec3f),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    leaf_cell: wp.array(dtype=wp.int64),
    slab_n: wp.array(dtype=wp.vec3f),
    slab_lo: wp.array(dtype=wp.float32),
    slab_hi: wp.array(dtype=wp.float32),
    children: wp.array(dtype=_NodeChildren),
):
    """One :class:`_NodeChildren` record per internal node ``i``: its two children's
    boxes and slabs, and each child as a node id or, for a leaf, ``-1 - cell``."""
    i = wp.tid()
    a = left[i]
    if a < wp.int64(0):
        return
    b = right[i]
    ch = _NodeChildren()
    ch.lo_l = node_lo[a]
    ch.hi_l = node_hi[a]
    ch.lo_r = node_lo[b]
    ch.hi_r = node_hi[b]
    ch.slab_l = slab_n[a]
    ch.slab_r = slab_n[b]
    ch.slab_lo_l = slab_lo[a]
    ch.slab_hi_l = slab_hi[a]
    ch.slab_lo_r = slab_lo[b]
    ch.slab_hi_r = slab_hi[b]
    ch.ref_l = wp.int32(a)
    if leaf_cell[a] >= wp.int64(0):
        ch.ref_l = wp.int32(-1) - wp.int32(leaf_cell[a])
    ch.ref_r = wp.int32(b)
    if leaf_cell[b] >= wp.int64(0):
        ch.ref_r = wp.int32(-1) - wp.int32(leaf_cell[b])
    children[i] = ch


def _child_records(tree, slabs):
    """One :class:`_NodeChildren` per node of ``tree`` (:func:`_preorder_cell_tree`;
    unset at the leaves) from the nodes' boxes and ``slabs`` (:func:`_node_slabs`), as
    the Warp array the seed and ball traversals read, with the buffer that holds it. A
    traversal takes as long as its slowest query's chain of loads. With both children
    in their parent's record, that is one load per opened node, not one per node
    tested (the ball search) or two per opened node (the seed) from the nodes' own
    arrays."""
    left, right, _, _, leaf_cell, lo, hi = tree
    device, stream = FunctionSpec.warp_launch_context(left)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        buffer = torch.empty(left.shape[0] * wp.types.type_size_in_bytes(_NodeChildren), dtype=torch.uint8, device=left.device)
        children = wp.array(ptr=buffer.data_ptr(), dtype=_NodeChildren, shape=(left.shape[0],), device=device)
        wp.launch(
            _pack_children_kernel,
            dim=left.shape[0],
            inputs=[
                wp.from_torch(lo.contiguous(), dtype=wp.vec3f),
                wp.from_torch(hi.contiguous(), dtype=wp.vec3f),
                *(wp.from_torch(t.contiguous(), dtype=wp.int64) for t in (left, right, leaf_cell)),
                wp.from_torch(slabs[0], dtype=wp.vec3f),
                wp.from_torch(slabs[1], dtype=wp.float32),
                wp.from_torch(slabs[2], dtype=wp.float32),
                children,
            ],
            device=device,
            stream=stream,
        )
    return children, buffer


@wp.kernel
def _nearest_cell_kernel(
    node_lo: wp.array(dtype=wp.vec3f),
    node_hi: wp.array(dtype=wp.vec3f),
    root: wp.int32,
    children: wp.array(dtype=_NodeChildren),
    corners: wp.array(dtype=wp.vec3f),
    query32: wp.array(dtype=wp.vec3f),
    rows: wp.array(dtype=wp.int64),
    nearest: wp.array(dtype=wp.int64),
):
    """A cell nearest to the query ``rows[tid]`` in float32. A depth-first traversal
    of the BVH from ``root`` reads one record per inner node (``children``), nearer
    child first. It skips the nodes no nearer than the nearest cell so far, each
    stacked with its distance: the larger of its box's and its slab's
    (:func:`_node_slabs`). It runs twice. The first run also skips the nodes within
    the factor ``_NEAREST_FIRST_PASS`` of the nearest so far, so it ends early at a
    cell within that factor of the nearest; the second, exact, starts from that
    cell's distance. A NaN distance
    (a degenerate cell) is never the nearest; with no other cell, ``nearest`` keeps
    its value."""
    q = rows[wp.tid()]
    x = query32[q]
    best = wp.float32(wp.inf)
    found = wp.int64(-1)
    nodes = _nearest_stack()
    gaps = _nearest_gaps()
    run = int(0)
    while run < 2:
        scale = wp.float32(1.0)
        if run == 0:
            scale = wp.float32(_NEAREST_FIRST_PASS)
        nodes[0] = root
        gaps[0] = _box_gap2(node_lo[0], node_hi[0], x)
        top = int(1)
        while top > 0:
            top -= 1
            if gaps[top] * scale < best:
                ref = nodes[top]
                if ref < wp.int32(0):
                    c = wp.int64(-1) - wp.int64(ref)
                    i = c * wp.int64(3)
                    d2 = _triangle_distance2(x, corners[i], corners[i + wp.int64(1)], corners[i + wp.int64(2)])
                    if d2 < best:
                        best = d2
                        found = c
                else:
                    ch = children[ref]
                    near = ch.ref_l
                    far = ch.ref_r
                    ### (neither distance is ever NaN: Warp's ``max`` skips a NaN, so a NaN
                    ### bound counts as 0 and its node is opened, with the slabs as without)
                    g_near = wp.max(_box_gap2(ch.lo_l, ch.hi_l, x), _slab_gap2(ch.slab_l, ch.slab_lo_l, ch.slab_hi_l, x))
                    g_far = wp.max(_box_gap2(ch.lo_r, ch.hi_r, x), _slab_gap2(ch.slab_r, ch.slab_lo_r, ch.slab_hi_r, x))
                    ### (at equal distances, as inside both boxes, the nearer centre first)
                    swap = g_far < g_near
                    if g_far == g_near:
                        e_near = wp.float32(0.5) * (ch.lo_l + ch.hi_l) - x
                        e_far = wp.float32(0.5) * (ch.lo_r + ch.hi_r) - x
                        swap = wp.dot(e_far, e_far) < wp.dot(e_near, e_near)
                    if swap:
                        near = ch.ref_r
                        far = ch.ref_l
                        g = g_near
                        g_near = g_far
                        g_far = g
                    if g_far * scale < best:
                        nodes[top] = far
                        gaps[top] = g_far
                        top += 1
                    if g_near * scale < best:
                        nodes[top] = near
                        gaps[top] = g_near
                        top += 1
        run += 1
    if found >= wp.int64(0):
        nearest[q] = found


def _nearest_cells(query: torch.Tensor, simplex: torch.Tensor, tree, children) -> torch.Tensor:
    """For each query ``(Q, 3)``, the index of a cell of ``simplex`` ``(F, 3, 3)``
    nearest to it in float32 ``(Q,)``, up to the rounding of the slab test, by a Warp
    traversal of the BVH ``tree`` over the cells (:func:`_preorder_cell_tree`) on its
    nodes' ``children`` (:func:`_child_records`); which of the cells at equal distances
    is unspecified, and it is cell 0 when every distance is NaN. Its exact distance
    bounds the closest one, so any cell would do for the search radius; a near one
    keeps the radius, and so the candidate pairs, few. The queries are traversed in
    Morton order."""
    lo, hi = tree[5:]
    nearest = torch.zeros(query.shape[0], dtype=torch.int64, device=query.device)
    device, stream = FunctionSpec.warp_launch_context(query)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _nearest_cell_kernel,
            dim=query.shape[0],
            inputs=[
                wp.from_torch(lo.contiguous(), dtype=wp.vec3f),
                wp.from_torch(hi.contiguous(), dtype=wp.vec3f),
                ### (one cell: the root is its leaf)
                0 if simplex.shape[0] > 1 else -1,
                children[0],
                wp.from_torch(simplex.reshape(-1, 3).float().contiguous(), dtype=wp.vec3f),
                wp.from_torch(query.float().contiguous(), dtype=wp.vec3f),
                wp.from_torch(_compute_morton_codes(query).argsort(), dtype=wp.int64),
                wp.from_torch(nearest, dtype=wp.int64),
            ],
            device=device,
            stream=stream,
        )
    return nearest


def _segment_candidates(query: torch.Tensor, points: torch.Tensor, cells: torch.Tensor, margin: torch.Tensor):
    """The (query, segment) pairs whose distance, computed in the queries'
    precision, is within ``margin`` ``(Q,)`` of the query's least: a brute force
    over the segments, chunked over the queries."""
    a, b = points[cells[:, 0]][None], points[cells[:, 1]][None]
    chunk = max(1, _SEARCH_PAIRS // cells.shape[0])
    ### (seeded empty, so that zero queries give zero pairs)
    li = [torch.empty(0, dtype=torch.long, device=query.device)]
    pk = [torch.empty(0, dtype=torch.long, device=query.device)]
    for s in range(0, query.shape[0], chunk):
        d2 = _segment_distance2(query[s : s + chunk, None], a, b)
        reach = (d2.amin(-1, keepdim=True).sqrt() + margin[s : s + chunk, None]).square()
        i, k = (d2 <= reach).nonzero(as_tuple=True)
        li.append(s + i)
        pk.append(k)
    return torch.cat(li), torch.cat(pk)


@wp.func
def _solid_angle(x: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    """The signed solid angle of the triangle ``abc`` seen from ``x`` (Van Oosterom
    and Strackee 1983), positive when ``x`` lies on the side opposite to ``(b - a) x
    (c - a)``."""
    p = a - x
    q = b - x
    r = c - x
    lp = wp.length(p)
    lq = wp.length(q)
    lr = wp.length(r)
    den = lp * lq * lr + wp.dot(p, q) * lr + wp.dot(p, r) * lq + wp.dot(q, r) * lp
    return wp.float64(2.0) * wp.atan2(wp.dot(p, wp.cross(q, r)), den)


@wp.kernel
def _node_moments_kernel(
    inner: wp.array(dtype=wp.int64),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    weight: wp.array(dtype=wp.float64),
    centre: wp.array(dtype=wp.vec3d),
    dipole: wp.array(dtype=wp.vec3d),
    spread: wp.array(dtype=wp.mat33d),
    radius: wp.array(dtype=wp.float64),
):
    """The far-field moments of each node ``inner[tid]`` from its two children's
    (:func:`_packed_winding_sum`)."""
    i = inner[wp.tid()]
    a = left[i]
    b = right[i]
    s = weight[a] + weight[b]
    p = wp.float64(0.5) * (centre[a] + centre[b])
    if s > wp.float64(0.0):
        p = (weight[a] * centre[a] + weight[b] * centre[b]) / s
    weight[i] = s
    centre[i] = p
    dipole[i] = dipole[a] + dipole[b]
    ea = centre[a] - p
    eb = centre[b] - p
    spread[i] = spread[a] + wp.outer(dipole[a], ea) + spread[b] + wp.outer(dipole[b], eb)
    radius[i] = wp.max(wp.length(ea) + radius[a], wp.length(eb) + radius[b])


def _winding_moments(simplex: torch.Tensor, parity: torch.Tensor, area: torch.Tensor, tree):
    """The far-field data of :func:`_packed_winding_sum` per node of ``tree``, as
    the Warp arrays its traversal reads: the centre, the squared far-field radius,
    the dipole and its spread, then the cells' corners and parities."""
    left, right, escape, levels, leaf_cell, _, _ = tree
    n, leaf = left.shape[0], leaf_cell >= 0
    cell = leaf_cell[leaf]
    centroid = simplex.mean(1)
    weight, centre, dipole = simplex.new_zeros(n), simplex.new_zeros(n, 3), simplex.new_zeros(n, 3)
    spread, radius = simplex.new_zeros(n, 3, 3), simplex.new_zeros(n)
    weight[leaf], centre[leaf], dipole[leaf] = area[cell].norm(dim=-1), centroid[cell], area[cell]
    ### (one vertex at a time: the same norms and maximum, without an (F, 3, 3)
    ### temporary)
    radius[leaf] = torch.stack([(simplex[cell, k] - centroid[cell]).norm(dim=-1) for k in range(simplex.shape[1])], -1).amax(-1)
    _bottom_up(
        _node_moments_kernel,
        levels,
        [
            *(wp.from_torch(c.contiguous(), dtype=wp.int64) for c in (left, right)),
            wp.from_torch(weight, dtype=wp.float64),
            wp.from_torch(centre, dtype=wp.vec3d),
            wp.from_torch(dipole, dtype=wp.vec3d),
            wp.from_torch(spread, dtype=wp.mat33d),
            wp.from_torch(radius, dtype=wp.float64),
        ],
        simplex,
    )
    ### (each float64 moment freed as it is rounded)
    del weight, centroid
    far2 = (WINDING_NUMBER_BETA * radius).square().float().contiguous()
    del radius
    dipole = dipole.float().contiguous()
    spread = spread.float().contiguous()
    return [
        wp.from_torch(centre.contiguous(), dtype=wp.vec3d),
        wp.from_torch(far2, dtype=wp.float32),
        wp.from_torch(dipole, dtype=wp.vec3f),
        wp.from_torch(spread, dtype=wp.mat33f),
        wp.from_torch(simplex.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
        wp.from_torch(parity.contiguous(), dtype=wp.float64),
    ]


@wp.struct
class _WindingNode:
    """What :func:`_packed_winding_kernel` reads of a node, in one record: the dipole and
    its spread, the squared far-field radius, the node after its subtree, its cell (-1
    inside) and the centre (last: with the float64 centre first, :func:`_pack_winding_kernel`
    runs several times slower)."""

    dipole: wp.vec3f
    spread: wp.mat33f
    far2: wp.float32
    escape: wp.int32
    cell: wp.int32
    centre: wp.vec3d


@wp.kernel
def _pack_winding_kernel(
    centre: wp.array(dtype=wp.vec3d),
    far2: wp.array(dtype=wp.float32),
    dipole: wp.array(dtype=wp.vec3f),
    spread: wp.array(dtype=wp.mat33f),
    escape: wp.array(dtype=wp.int64),
    leaf_cell: wp.array(dtype=wp.int64),
    nodes: wp.array(dtype=_WindingNode),
):
    i = wp.tid()
    node = _WindingNode()
    node.centre = centre[i]
    node.far2 = far2[i]
    node.dipole = dipole[i]
    node.spread = spread[i]
    node.escape = wp.int32(escape[i])
    node.cell = wp.int32(leaf_cell[i])
    nodes[i] = node


@wp.kernel
def _packed_winding_kernel(
    nodes: wp.array(dtype=_WindingNode),
    corners: wp.array(dtype=wp.vec3d),
    parity: wp.array(dtype=wp.float64),
    query: wp.array(dtype=wp.vec3d),
    rows: wp.array(dtype=wp.int64),
    out: wp.array(dtype=wp.float64),
):
    """The sum of the signed solid angles of the cells seen from each query: a
    stackless depth-first traversal of the BVH (:func:`_preorder_cell_tree`, an inner
    node's left child is the next node; its nodes packed) that takes a node's expansion
    (in float32, from the float64 offset of its centre) where the query is farther than
    ``far2`` (squared) from it, and the exact solid angle (float64, the canonical
    vertices ``corners`` times the cell's ``parity``) at the near leaves."""
    q = rows[wp.tid()]
    x = query[q]
    total = wp.float64(0.0)
    node = wp.int32(0)
    while node >= 0:
        nd = nodes[node]
        r = _vec3f(nd.centre - x)
        r2 = wp.dot(r, r)
        if r2 > nd.far2:
            inv2 = wp.float32(1.0) / r2
            s = nd.spread
            near = wp.dot(nd.dipole, r) + s[0, 0] + s[1, 1] + s[2, 2]
            total += wp.float64((near - wp.float32(3.0) * wp.dot(r, wp.mul(s, r)) * inv2) * inv2 * wp.sqrt(inv2))
            node = nd.escape
        else:
            c = nd.cell
            if c >= wp.int32(0):
                i = wp.int64(c) * wp.int64(3)
                total += parity[c] * _solid_angle(x, corners[i], corners[i + wp.int64(1)], corners[i + wp.int64(2)])
                node = nd.escape
            else:
                node = node + wp.int32(1)
    out[q] = total


def _packed_winding_sum(query: torch.Tensor, simplex: torch.Tensor, parity: torch.Tensor, area: torch.Tensor, tree):
    """The generalized winding number times ``4 pi`` of the oriented triangles
    ``simplex`` ``(F, 3, 3)`` (canonical, orientation ``parity`` ``(F,)``, area vectors
    ``area`` ``(F, 3)``) at each query ``(Q, 3)``, all float64. It follows Barill,
    Dickson, Schmidt, Levin and Jacobson (2018) on the BVH ``tree`` from
    :func:`_preorder_cell_tree`, whose leaves hold the rows of ``simplex``. Every node
    carries the area-weighted centre ``p`` of its cells, a radius about it, their
    dipole moment ``m = sum a_f n_f`` and its spread ``N = sum a_f n_f (c_f - p)^T``
    (one bottom-up pass over the levels). A node farther than ``WINDING_NUMBER_BETA``
    radii takes the expansion of the solid angle about ``p`` to first order, ``m . r /
    |r|^3 + tr(N) / |r|^3 - 3 r . N r / |r|^5`` with ``r = p - x``; the near leaves
    take the exact solid angle. The moments come first, then one record per node, then
    one traversal with the queries in Morton order. The lengths are first divided by
    the power of two nearest above the cells' extent. That is exact, so the sum is the
    same at every power-of-two scale, and the float32 moments (the spread grows as a
    length cubed, its product with ``r`` as the fifth power) neither overflow nor
    underflow at any scale."""
    unit = torch.ldexp(torch.ones_like(simplex[0, 0, 0]), torch.frexp(simplex.abs().amax()).exponent)
    query, simplex, area = query / unit, simplex / unit, area / unit.square()
    left, _, escape, _, leaf_cell, _, _ = tree
    centre, far2, dipole, spread, corners, par = _winding_moments(simplex, parity, area, tree)
    out = torch.empty(query.shape[0], dtype=torch.float64, device=query.device)
    device, stream = FunctionSpec.warp_launch_context(query)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        size = wp.types.type_size_in_bytes(_WindingNode)
        buffer = torch.empty(left.shape[0] * size, dtype=torch.uint8, device=query.device)
        nodes = wp.array(ptr=buffer.data_ptr(), dtype=_WindingNode, shape=(left.shape[0],), device=device)
        wp.launch(
            _pack_winding_kernel,
            dim=left.shape[0],
            inputs=[centre, far2, dipole, spread, wp.from_torch(escape, dtype=wp.int64), wp.from_torch(leaf_cell, dtype=wp.int64), nodes],
            device=device,
            stream=stream,
        )
        wp.launch(
            _packed_winding_kernel,
            dim=query.shape[0],
            inputs=[
                nodes,
                corners,
                par,
                wp.from_torch(query.contiguous(), dtype=wp.vec3d),
                wp.from_torch(_compute_morton_codes(query).argsort(), dtype=wp.int64),
                wp.from_torch(out, dtype=wp.float64),
            ],
            device=device,
            stream=stream,
        )
    return out


@wp.kernel
def _segment_winding_kernel(
    ends: wp.array(dtype=wp.vec2d),
    parity: wp.array(dtype=wp.float64),
    query: wp.array(dtype=wp.vec2d),
    out: wp.array(dtype=wp.float64),
):
    """The sum of the signed angles the segments (``ends`` two by two, times their
    ``parity``) subtend at each query, in the cells' order."""
    tid = wp.tid()
    x = query[tid]
    total = wp.float64(0.0)
    for c in range(parity.shape[0]):
        p = ends[2 * c] - x
        q = ends[2 * c + 1] - x
        total -= parity[c] * wp.atan2(p[0] * q[1] - p[1] * q[0], p[0] * q[0] + p[1] * q[1])
    out[tid] = total


def _segment_winding_numbers(query: torch.Tensor, simplex: torch.Tensor, parity: torch.Tensor):
    """The winding number of the oriented segments ``simplex`` ``(F, 2, 2)``
    (canonical, orientation ``parity`` ``(F,)``) at each query ``(Q, 2)``, float64:
    the sum of the signed angles they subtend, over ``2 pi``, positive for a query on
    the side opposite to the normals; a brute force over the segments, one Warp
    thread per query."""
    out = torch.empty(query.shape[0], dtype=torch.float64, device=query.device)
    device, stream = FunctionSpec.warp_launch_context(query)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _segment_winding_kernel,
            dim=query.shape[0],
            inputs=[
                wp.from_torch(simplex.reshape(-1, 2).contiguous(), dtype=wp.vec2d),
                wp.from_torch(parity.contiguous(), dtype=wp.float64),
                wp.from_torch(query.contiguous(), dtype=wp.vec2d),
                wp.from_torch(out, dtype=wp.float64),
            ],
            device=device,
            stream=stream,
        )
    return out / (2 * torch.pi)


def _winding_sign(sigma: torch.Tensor, w: torch.Tensor):
    """The sign of the distance at each query, float64: ``sign(sigma / 2 - w)``.
    ``w`` ``(Q,)`` is the generalized winding number of the oriented cells at the
    queries (Jacobson, Kavan and Sorkine-Hornung 2013; :func:`_packed_winding_sum` in
    three dimensions, :func:`_segment_winding_numbers` in two). ``sigma`` is the sign
    of the volume they enclose (:func:`_oriented_areas`), ``V = sum_f a_f n_f . (c_f -
    g) / d`` about their area-weighted centroid ``g`` (+1 at ``V = 0`` and at ``w =
    sigma / 2``). For a closed surface ``w`` is ``(sigma - 1) / 2`` on the side the
    normals point to and ``(sigma + 1) / 2`` on the other (0 and 1 for outward
    normals, -1 and 0 for normals into the region they bound). So ``sigma / 2 - w`` is
    +1/2 or -1/2, positive on the side the normals point to. Coincident cells of
    opposite orientation (a zero-thickness fold) cancel in ``w``, so both of its sides
    take the sign of the region around it."""
    return torch.where(0.5 * sigma - w >= 0, 1.0, -1.0).to(w)


def _oriented_areas(simplex: torch.Tensor, parity: torch.Tensor):
    """The oriented area vectors ``a_f n_f`` of the cells ``(F, d)`` and the sign
    ``sigma`` of the volume they enclose (:func:`_winding_sign`)."""
    e = simplex[:, 1:] - simplex[:, :1]
    if simplex.shape[-1] == 2:
        area = torch.stack([-e[:, 0, 1], e[:, 0, 0]], -1)
    else:
        area = 0.5 * torch.linalg.cross(e[:, 0], e[:, 1], dim=-1)
    area = parity[:, None] * area
    centroid, measure = simplex.mean(1), area.norm(dim=-1)
    g = (measure[:, None] * centroid).sum(0) / measure.sum().clamp_min(torch.finfo(measure.dtype).tiny)
    return area, torch.where((area * (centroid - g)).sum() >= 0, 1.0, -1.0)


def _centroid(simplex: torch.Tensor, flat=None) -> torch.Tensor:
    """The measure-weighted centroid ``(d,)`` of the cells ``simplex`` ``(F, d, d)``
    (float64; the zero-area ones ``flat`` weigh 0), a function of the set of cells,
    bitwise, whatever their order: each cell's measure and measure-weighted midpoint
    in float64, summed exactly in int64 fixed point (each column scaled so that ``F``
    times its greatest magnitude stays below ``2^62``, then truncated). The origin
    when every measure is 0."""
    if simplex.shape[-1] == 3:
        m = torch.linalg.cross(simplex[:, 1] - simplex[:, 0], simplex[:, 2] - simplex[:, 0], dim=-1).norm(dim=-1)
    else:
        m = (simplex[:, 1] - simplex[:, 0]).norm(dim=-1)
    if flat is not None:
        m = torch.where(flat, 0.0, m)
    vals = torch.cat([m[:, None], m[:, None] * simplex.mean(1)], 1)
    scale = vals.abs().amax(0) * vals.shape[0]
    e = torch.frexp(scale.clamp_min(torch.finfo(torch.float64).tiny)).exponent
    tot = torch.ldexp(torch.ldexp(vals, 62 - e).trunc().long().sum(0).double(), e - 62)
    if not bool(tot[0] > 0):
        return simplex.new_zeros(simplex.shape[-1])
    return tot[1:] / tot[0]


def _certified_tied(tq: torch.Tensor, c: torch.Tensor, tau: torch.Tensor, nq: int) -> torch.Tensor:
    """Per query ``(nq,)``, whether its tie rows (``tq`` ``(P,)``, sorted) hold two
    directions ``c`` ``(P, d)`` farther apart than the sum of their bounds ``tau``
    ``(P,)`` (with zero bounds, two that differ). Chunked over the queries."""
    tied = torch.zeros(nq, dtype=torch.bool, device=c.device)
    if tq.numel() == 0:
        return tied
    ids, counts = torch.unique_consecutive(tq, return_counts=True)
    start = torch.cumsum(counts, 0) - counts
    for k in torch.unique(counts).tolist():
        if k < 2:
            continue
        here = (counts == k).nonzero()[:, 0]
        for s in range(0, here.shape[0], 65536):
            h = here[s : s + 65536]
            rows = start[h, None] + torch.arange(k, device=c.device)
            cc, tt = c[rows], tau[rows]
            dist = (cc[:, :, None] - cc[:, None, :]).norm(dim=-1)
            tied[ids[h]] = (dist > tt[:, :, None] + tt[:, None, :]).flatten(1).any(1)
    return tied


def _closest_pair(li, pk, dist, err, points, cells, parity, nq):
    """Per query ``(nq,)``: the pair of its closest point (an index into the pairs
    ``li``, ``pk`` with distances ``dist`` and error bounds ``err``), the least
    distance, and the least upper bound ``dist + err``. Cells at exactly the least
    distance (cells sharing the closest point compute it bitwise) are ordered by their
    positions, the lexicographic order of their canonical vertices as given
    (``points[cells]``) and then their ``parity``, not by their indices, so the
    closest cell, its bound and its weights do not depend on the cells' order."""
    inf = dist.new_full((nq,), torch.inf)
    d_min = inf.scatter_reduce(0, li, dist, "amin")
    upper = inf.scatter_reduce(0, li, dist + err, "amin")
    pair = torch.arange(li.shape[0], device=li.device)
    first = torch.full((nq,), li.shape[0], dtype=torch.long, device=li.device).scatter_reduce(
        0, li, torch.where(dist == d_min[li], pair, li.shape[0]), "amin"
    )
    ### The tie-break of the docstring, one key column at a time, each keeping the rows
    ### at the query's least value;
    ### identical keys are the same cell, so the least pair among them is the cell.
    least = dist == d_min[li]
    key = torch.cat([points[cells[pk]].flatten(1), parity[pk, None].to(points.dtype)], 1)
    for k in key.unbind(1):
        k_min = k.new_full((nq,), torch.inf).scatter_reduce(0, li, torch.where(least, k, torch.inf), "amin")
        least = least & (k == k_min[li])
    pick = torch.full_like(first, li.shape[0]).scatter_reduce(0, li, torch.where(least, pair, li.shape[0]), "amin")
    first = torch.where(pick < li.shape[0], pick, first)
    return first, d_min, upper


def _feature_reached(li, pk, w, cells, flat, n_vertices):
    """Per pair, whether every cell sharing the feature of its closest point (a vertex
    or an edge by the weights ``w``; a cell's interior is reached) is among the
    query's pairs ``li``, ``pk``: a vertex or an edge point is a local minimum of the
    distance on the surface only where every cell sharing that feature reaches the
    same point. The zero-area cells ``flat`` (or ``None``) count their distinct
    vertices and the edges between consecutive distinct ones once. Also each pair's
    count of nonzero weights."""
    nonzero = w > 0
    n_nonzero = nonzero.sum(-1)
    corners = cells[pk]
    key = torch.where(n_nonzero == 1, corners.gather(1, w.argmax(-1, keepdim=True))[:, 0], -1)
    if flat is not None:
        ### A zero-area cell's features are its distinct vertices and the edges between
        ### consecutive distinct ones, each counted once (n_vertices: none).
        consecutive = cells[:, 1:] != cells[:, :-1]
        vertices = torch.where(flat[:, None] & ~torch.cat([torch.ones_like(consecutive[:, :1]), consecutive], 1), n_vertices, cells)
        count = torch.bincount(vertices.reshape(-1), minlength=n_vertices + 1)[key.clamp_min(0)]
    else:
        count = torch.bincount(cells.reshape(-1), minlength=n_vertices)[key.clamp_min(0)]
    if cells.shape[1] == 3:
        a, b = cells[:, [0, 0, 1]], cells[:, [1, 2, 2]]
        if flat is not None:
            edges = torch.stack([consecutive[:, 0], torch.zeros_like(flat), consecutive[:, 1]], 1)
            a, b = (torch.where(flat[:, None] & ~edges, n_vertices, x) for x in (a, b))
        edge_keys, edge_valence = torch.unique(
            n_vertices + torch.minimum(a, b) * n_vertices + torch.maximum(a, b), return_counts=True
        )
        ends = corners.masked_fill(~nonzero, n_vertices)
        ends = ends.sort(-1).values  # the edge's two vertices, ascending
        edge = n_vertices + ends[:, 0] * n_vertices + ends[:, 1]
        key = torch.where(n_nonzero == 2, edge, key)
        count = torch.where(
            n_nonzero == 2,
            edge_valence[torch.searchsorted(edge_keys, edge).clamp_max(edge_keys.shape[0] - 1)],
            count,
        )
    reached = torch.ones_like(nonzero[:, 0])
    on_feature = key >= 0
    group = torch.stack([li[on_feature], key[on_feature]], -1)
    _, inverse, counts = torch.unique(group, dim=0, return_inverse=True, return_counts=True)
    reached[on_feature] = counts[inverse] == count[on_feature]
    return reached, n_nonzero


def _tie_uncertainty(st, q64, tq, eta, e_t, d_t, wall, on_cell):
    """The certified uncertainty ``tau`` of each tie row's direction (cells ``st``,
    queries ``q64[tq]``, storage roundings ``eta``, bands ``e_t``, distances ``d_t``):
    ``sqrt(2) s`` (2 where ``s >= 1``), ``s`` a bound on the sine of the angle between
    the computed direction and the exact one of the inputs as given; 0 on the surface
    (``wall``), where normals that differ tie. ``on_cell`` rows take the cell's normal
    with Ericson's misplacement, the others the direction from the point."""
    ### Off the surface, a tie is two certified-distinct directions, |c_i - c_j| >
    ### tau_i + tau_j; tau = sqrt(2) s is the chord of a sine error s < 1. A cell's
    ### normal: its vertices each within eta = u |v| of the exact ones (their
    ### storage rounding), so |dX| <= 4 eta L + 4 eta^2 against the certified
    ### |X| - |E| (Shewchuk), with the float64 normal's own bound (delta) and its
    ### normalization. A direction from a point: the point within the band E_k (and
    ### Ericson's misplacement on a face), over D_k.
    u64 = torch.finfo(torch.float64).eps / 2
    if st.shape[-1] == 3:
        delta_t, kappa_t, longest_t = _thin_bounds(st)
        size = torch.linalg.cross(st[:, 1] - st[:, 0], st[:, 2] - st[:, 0], dim=-1).norm(dim=-1)
        s_n = (4 * eta * longest_t + 4 * eta * eta) / (size * (1 - delta_t)) + delta_t + 4 * u64
        mis = kappa_t * (q64[tq][:, None] - st).square().sum(-1).amax(-1)
    else:
        length = (st[:, 1] - st[:, 0]).norm(dim=-1)
        s_n = 2 * eta / (length - 2 * eta) + 4 * u64
        mis = torch.zeros_like(length)
    s_n = torch.where(s_n >= 0, s_n, torch.inf)
    s_p = torch.where(d_t > 0, (e_t + 4 * u64 * d_t) / d_t, torch.inf)
    s_c = s_n + torch.where(d_t > 0, (mis + e_t) / d_t, torch.inf)
    s = torch.where(wall, s_n, torch.where(on_cell, s_c, s_p))
    ### On the surface the bound is 0 (normals that differ tie). The least-norm point of
    ### their hull lies within tau_k + max_j tau_j of each member k, the group's
    ### certified uncertainty, so it needs no certificate. It is also the certified
    ### case's own value, so u does not depend on the queries' precision.
    return torch.where(wall, 0.0, torch.where(s < 1, 2.0**0.5 * s, 2.0))


def _closest_candidates_impl(query_points, points, cells, parity, face_normals, fast: bool):
    """The closest-feature search (no gradient); see :func:`exact_boundary_geometry`.
    For every query it gives the cell of its closest point ``(Q,)``; whether several
    features tie for it ``(Q,)``; at a tie, the least-norm point of the hull of their
    directions ``(Q, d)``; the sign of the distance from the winding number of the
    cells (:func:`_winding_sign`) ``(Q,)``; whether the query is on the surface
    (below) ``(Q,)``; and the barycentric weights of its closest point on its cell
    (:func:`_closest_weights`, float64) ``(Q, d)``. ``cells`` come in canonical order
    with their orientations ``parity`` (:func:`_canonical_cells`), the normals in
    float64.

    The search runs in the boundary's centred frame: every float64 coordinate less
    ``c``, the measure-weighted centroid of its distinct cells (:func:`_centroid`),
    rounded to the queries' precision (so float32 inputs centre exactly). Its
    arithmetic then scales with the boundary's size, not with its distance from the
    origin. The exact predicates (:func:`_zero_area`) and the normals stay on the
    coordinates as given.

    Every candidate distance is computed in float64 on the canonical cell. The error
    bound of the distance to cell ``k`` is ``E_k = u (|x| + R_k) + C u_64 (|x'| +
    R'_k)``, with ``R_k = max_{v in k} |v|`` and primes in the centred frame. Its first
    term is the rounding of the coordinates as they were given, ``u`` the unit
    roundoff of the queries' precision. So it is measured in the frame they were
    given in: a case posed far from the origin is rounded more coarsely, and a term
    measured about ``c`` would not bound it. Its second term is float64's arithmetic
    in the frame it runs in, ``C`` a worst case (``_BAND_UNITS``: 91 in three
    dimensions, 17 in two). It is 1.7e-7 of the bound for float32 queries, and
    matters only for float64 ones.

    Cell ``k`` ties when it may be the closest one, ``D_k - E_k <= min_j (D_j +
    E_j)``, and its closest point is a local minimum of the distance on the surface.
    On a vertex or an edge, that needs every cell sharing that feature to reach the
    same point; otherwise a neighbouring cell holds a closer point of the same patch,
    as beside the foot of the perpendicular on a cell next to the vertex. On the
    surface the second condition is dropped, since every cell within the bound holds
    the query. The query is on the surface when its closest cell's distance interval
    reaches 0, ``D_min - E_min <= 0``: its coordinates cannot tell it from a point of
    the surface. Cells at exactly the least distance are ordered by their positions,
    not by their indices (:func:`_closest_pair`).

    A tie's directions are the unit vectors from the surface to the query. On the
    surface they are the unit normals of the tied cells that have one; with none, the
    query keeps its closest cell's normal. On the surface the query is tied when two
    of these normals differ, certified distinct or not. Their least-norm point lies in
    their hull, so within ``tau_k + max_j tau_j`` of each member ``k`` (the group's
    certified uncertainty) where none is certified distinct. It is also a function of
    the set, where one of the normals would depend on the cells' order and the
    certificate on the queries' precision. Off the surface the query is tied when two
    of its directions are certified distinct: farther apart than the sum of their
    bounds ``tau`` (:func:`_tie_uncertainty`). The least-norm point is then taken over
    all of the tie's directions. On the surface it is odd in them (half its value at
    the normals less its value at their negations), so a winding flip negates it
    bitwise.

    The exact search runs on candidate pairs that contain every cell with ``D_k -
    max_j E_j`` within that bound. In three dimensions a Warp traversal of a BVH over
    the cells finds a cell nearest in float32 (:func:`_nearest_cells`). Its exact
    distance plus its error bound bounds the closest distance, and a second traversal
    of the same BVH returns the cells within that radius. In two dimensions a brute
    force in the queries' precision, on the centred coordinates, keeps the segments
    within its own rounding and the error bounds of the least. That rounding is ``40
    u (|x'| + max |v'|)`` on each side, with room over a first-order count of 33.5
    units: the brute-force distance 14, the centred copies' rounding 1, the
    comparison's square and root 2.5, and the arithmetic share of the least's bound
    16. The BVH is built in depth-first preorder with its cells in leaf order
    (:func:`_preorder_cell_tree`). In three dimensions the winding number takes the
    same BVH, on packed node records (:func:`_packed_winding_sum`), unless cells
    repeat.

    ``fast`` computes the closest weights (:func:`_closest_weights`: Ericson's, the
    thin-cell guard and the accurate point) by two Triton kernels on CUDA, bitwise the
    torch code's."""
    nq, dim = query_points.shape
    dev = query_points.device
    q64, simplex = query_points.double(), points.double()[cells]
    ### A zero-area cell is the segment or the point its vertices span. Where that set
    ### lies in the other cells' it adds nothing to the surface and is left out, so the
    ### outputs are those without it.
    flat, any_flat = _zero_area(simplex, cells)
    kept = None
    if any_flat:
        covered = _covered(cells, flat, points.shape[0])
        if bool(covered.any()):
            kept = (~covered).nonzero()[:, 0]
            cells, parity, face_normals, simplex, flat = (x[kept] for x in (cells, parity, face_normals, simplex, flat))
            any_flat = bool(flat.any())
    face_n = face_normals.double()
    ### The winding number is the surface's, not the cell list's: identical oriented
    ### cells count once. The cells go in the sorted order of their keys, so a surface
    ### stored with repeated cells has the same BVH, and the same far field, as the
    ### surface stored once: the stable lexicographic order of (cells, parity), as in
    ### _weld, by two stable sorts of keys that pack adjacent columns.
    low = cells[:, -1] * 2 + (parity > 0)
    high = cells[:, 0] * points.shape[0] + cells[:, 1] if dim == 3 else cells[:, 0]
    order = torch.argsort(low, stable=True)
    order = order[torch.argsort(high[order], stable=True)]
    low, high = low[order], high[order]
    new = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    del low, high
    rep = order[torch.cat([new.new_ones(1), new])]
    del new
    ### The band: the storage term on the magnitudes as given, the arithmetic's on
    ### the centred ones. The float32 copies the BVH and the 2-D brute force read are
    ### the centred coordinates cast once.
    u_in, u64 = torch.finfo(query_points.dtype).eps / 2, torch.finfo(torch.float64).eps / 2
    given_reach, given_q = simplex.norm(dim=-1).amax(-1), q64.norm(dim=-1)
    origin = _centroid(simplex[rep], flat[rep] if any_flat else None).to(query_points.dtype).double()
    q64, simplex = q64 - origin, simplex - origin
    query_c, points_c = q64.to(query_points.dtype), (points.double() - origin).to(points.dtype)
    reach, q_norm = simplex.norm(dim=-1).amax(-1), q64.norm(dim=-1)
    share = _BAND_UNITS[dim] * u64

    def band(qi, ki):
        return u_in * (given_q[qi] + given_reach[ki]) + share * (q_norm[qi] + reach[ki])

    err_max = u_in * (given_q + given_reach.max()) + share * (q_norm + reach.max())
    s_rep, p_rep = simplex[rep], parity[rep].to(q64)
    ### (no normal, no area, no solid angle: its parity 0 in the winding number)
    if any_flat:
        p_rep = torch.where(flat[rep], 0.0, p_rep)
    area, sigma = _oriented_areas(s_rep, p_rep)
    if dim == 3:
        ### Without repeated cells the two trees are one: the ball search and the
        ### winding number traverse the same tree.
        once = rep.shape[0] == cells.shape[0]
        s_ord = s_rep if once else simplex[order]
        f_ord = flat[order] if any_flat else None
        tree, leaf_order, s_leaf = _preorder_cell_tree(s_ord)
        if once:
            ### (the leaves hold the cells; the winding number reads them there)
            s_ord = s_rep = None
        if s_leaf.shape[0] <= _SLAB_CELLS_PER_QUERY * nq:
            slabs = _node_slabs(s_leaf, parity[order][leaf_order].to(q64), tree)
        else:
            slabs = _no_slabs(tree, s_leaf)
        children = _child_records(tree, slabs)
        seed = order[leaf_order[_nearest_cells(query_c, s_leaf, tree, children)]]
        w0 = _closest_weights(q64, simplex[seed], flat[seed] if any_flat else None, fast)
        bound = (q64 - (w0[..., None] * simplex[seed]).sum(1)).norm(dim=-1) + band(torch.arange(nq, device=dev), seed)
        ### (padded by twice the bound, the kernel's float64 distances against these,
        ### and by the rounding of its box test)
        radius = (bound + 3 * err_max) * (1 + 1e-12)
        li, pk = _ball_candidates(
            q64, radius, s_leaf, tree, children, reach=reach.max(), flat=None if f_ord is None else f_ord[leaf_order]
        )
        del children
        pk = order[leaf_order[pk]]
        ### (the winding number reads the tree's links and leaves, not its boxes)
        tree = (*tree[:5], None, None)
        if not once:
            tree, leaf_order, s_leaf = _preorder_cell_tree(s_rep)
        w_sum = _packed_winding_sum(q64, s_leaf, p_rep[leaf_order], area[leaf_order], tree)
    else:
        rounding = 40 * torch.finfo(query_points.dtype).eps / 2 * (q_norm + reach.max())
        li, pk = _segment_candidates(query_c, points_c, cells, (2 * (rounding + err_max)).to(query_points.dtype))
    ### In the order of (query, cell).
    f = cells.shape[0]
    key = (li * f + pk).sort().values
    li, pk = key // f, key % f
    tri = simplex[pk]
    w = _closest_weights(q64[li], tri, flat[pk] if any_flat else None, fast)
    diff = q64[li] - (w[..., None] * tri).sum(1)
    dist = diff.norm(dim=-1)
    err = band(li, pk)
    first, d_min, upper = _closest_pair(li, pk, dist, err, points, cells, parity, nq)
    idx, on_wall = pk[first], d_min - err[first] <= 0
    reached, n_nonzero = _feature_reached(li, pk, w, cells, flat if any_flat else None, points.shape[0])
    ### On the surface the directions are the cells' normals, renormalized in float64
    ### (the least-norm step needs unit vectors); a degenerate cell has none.
    wall_n = face_n / face_n.norm(dim=-1, keepdim=True)
    has_n = torch.isfinite(wall_n).all(-1)
    if any_flat:
        ### (a zero-area cell's computed normal can be a unit vector of rounding)
        has_n = has_n & ~flat
    tie = (dist - err <= upper[li]) & torch.where(on_wall[li], has_n[pk], reached)
    tq = li[tie]
    fn, on_cell = face_n[pk[tie]], n_nonzero[tie] == dim
    norm = fn.norm(dim=-1, keepdim=True)
    fn = torch.where(norm > 0, fn / norm.clamp_min(torch.finfo(norm.dtype).tiny), fn)
    dt = diff[tie]
    plane = (dt * fn).sum(-1)
    tc = torch.where(
        on_cell[:, None],
        torch.where((plane >= 0)[:, None], fn, -fn),
        dt / dist[tie].clamp_min(torch.finfo(dist.dtype).tiny)[:, None],
    )
    wall = on_wall[tq]
    tc = torch.where(wall[:, None], wall_n[pk[tie]], tc)
    pt = pk[tie]
    tau = _tie_uncertainty(simplex[pt], q64, tq, u_in * given_reach[pt], err[tie], dist[tie], wall, on_cell)
    tied = _certified_tied(tq, tc, tau, nq)
    if any_flat:
        ### On a zero-area closest cell the direction is the tie's over the cells
        ### that hold the query and have a normal: theirs, or 0 with none (the
        ### least-norm element of the distance's gradient on a segment or a point).
        tied = tied | (on_wall & flat[idx])
    sel = tied[tq]
    g, c, wall = tq[sel], tc[sel], wall[sel]
    m = _grouped_min_norm(torch.cat([g, g[wall] + nq]), torch.cat([c, -c[wall]]), 2 * nq)
    tie_u = torch.where(on_wall[:, None], 0.5 * (m[:nq] - m[nq:]), m[:nq])
    winding = w_sum / (4 * torch.pi) if dim == 3 else _segment_winding_numbers(q64, s_rep, p_rep)
    sign = _winding_sign(sigma, winding)
    if kept is not None:
        idx = kept[idx]
    return idx, tied, tie_u.to(query_points.dtype), sign.to(query_points.dtype), on_wall, w[first]


_SearchOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@torch.library.custom_op("physicsnemo::isla_closest_candidates", mutates_args=())
def _closest_candidates(
    query_points: torch.Tensor, points: torch.Tensor, cells: torch.Tensor, parity: torch.Tensor, face_normals: torch.Tensor
) -> _SearchOutputs:
    """The closest-feature search, :func:`_closest_candidates_impl`: the reference,
    which defines it."""
    return _closest_candidates_impl(query_points, points, cells, parity, face_normals, fast=False)


@torch.library.custom_op("physicsnemo::isla_closest_candidates_fast", mutates_args=())
def _closest_candidates_fast(
    query_points: torch.Tensor, points: torch.Tensor, cells: torch.Tensor, parity: torch.Tensor, face_normals: torch.Tensor
) -> _SearchOutputs:
    """The closest-feature search, :func:`_closest_candidates_impl` with ``fast``: the
    reference's outputs bitwise, from Triton kernels on CUDA."""
    return _closest_candidates_impl(query_points, points, cells, parity, face_normals, fast=True)


@_closest_candidates.register_fake
@_closest_candidates_fast.register_fake
def _(query_points, points, cells, parity, face_normals):
    nq = query_points.shape[0]
    return (
        query_points.new_empty(nq, dtype=torch.long),
        query_points.new_empty(nq, dtype=torch.bool),
        query_points.new_empty(query_points.shape),
        query_points.new_empty(nq),
        query_points.new_empty(nq, dtype=torch.bool),
        query_points.new_empty(query_points.shape, dtype=torch.float64),
    )


class _StraightThrough(torch.autograd.Function):
    """``value`` in the forward pass, bitwise; the backward pass sends the whole
    gradient to ``surrogate``, and none to ``value``."""

    @staticmethod
    def forward(ctx, value, surrogate):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return None, grad


def _closest_features(
    query_points: Float[torch.Tensor, "queries dim"],
    points: Float[torch.Tensor, "vertices dim"],
    cells: Int[torch.Tensor, "cells dim"],
    kernel: str = "reference",
):
    """The closest surface point of each query and its feature; see
    :func:`exact_boundary_geometry`. Returns, in order: the offset ``x - p`` from the
    point ``(Q, d)``, evaluated in float64 as ``(x - v_0) - sum_{i > 0} w_i (v_i -
    v_0)`` from the coordinates as given and rounded once to the queries' precision;
    whether the point is interior to its cell ``(Q,)``; the cell's unit normal ``(Q,
    d)``; whether there is a closest-feature tie (:func:`_closest_candidates`)
    ``(Q,)`` and its direction ``(Q, d)``; the sign of the distance ``(Q,)``; the
    distance at a tie ``(Q,)``; whether the query is on the surface ``(Q,)``; whether
    the point lies inside an edge of a triangle (off the surface, untied) ``(Q,)``;
    and there the direction from the edge's closest point ``(Q, d)``, in float64 from
    the query and the edge's vertices.

    The last two serve the gradient. The offset is rebuilt from frozen weights, which
    hold the point fixed, while on an edge the point slides with the query and the
    vertices, so the direction takes its gradient from the sliding one
    (:class:`_StraightThrough`). ``kernel`` names the search's implementation."""
    search = {"reference": _closest_candidates, "fast": _closest_candidates_fast}[kernel]
    points = points.to(query_points.dtype)
    if kernel == "fast":
        cells, parity = geometry_fast.canonical_cells(points, _weld(points, geometry_fast.run_start)[cells], _canonical_cells)
    else:
        cells, parity = _canonical_cells(points, _weld(points)[cells])
    cell_points = points[cells]
    all_n = parity[:, None].to(points.dtype) * compute_cell_normals(cell_points[:, 1:] - cell_points[:, :1])
    ### The normals the on-wall rule and the tie directions read, in float64 from the
    ### coordinates as given: the least-norm point of near-duplicate normals amplifies
    ### their rounding, so they must not depend on the queries' precision.
    cell64 = cell_points.double()
    n64 = parity[:, None].double() * compute_cell_normals(cell64[:, 1:] - cell64[:, :1])
    del cell_points, cell64
    with torch.no_grad():
        q, y = query_points, points
        if q.dtype in (torch.float16, torch.bfloat16):
            q, y = q.float(), y.float()
        idx, tied, tie_u, sign, on_wall, weights = search(q, y, cells, parity, n64)
        corners = cells[idx]  # (Q, d)
    simplex = points[corners]  # (Q, d, d)
    s64 = simplex.double()
    diff = ((query_points.double() - s64[:, 0]) - (weights[:, 1:, None] * (s64[:, 1:] - s64[:, :1])).sum(1)).to(query_points.dtype)
    on_cell = (weights > 0).all(-1)
    dtype = query_points.dtype
    ### (in float64: off the surface a lower precision can round it to 0, losing the sign)
    tie_dist = (query_points.double() - (weights[..., None] * simplex.double()).sum(1)).norm(dim=-1).to(dtype)
    ### (in half precision a small cell's cross product underflows to 0, so off the
    ### surface too the normal comes from float64)
    off_n = n64 if dtype in (torch.float16, torch.bfloat16) else all_n
    face_n = torch.where(on_wall[:, None], n64[idx].to(dtype), off_n[idx].to(dtype))
    slide = (weights > 0).sum(-1) == 2
    slide = slide & ~on_wall & ~tied & (query_points.shape[-1] == 3)
    ### (on every row, finite with a finite gradient: the rows torch.where leaves out
    ### still receive a zero gradient)
    k = (~(weights > 0)).to(torch.int8).argmax(-1)  # the vertex off the edge
    a, b = (
        s64.gather(
            1, ((k + i) % s64.shape[1])[:, None, None].expand(-1, 1, s64.shape[2])
        )[:, 0]
        for i in (1, 2)
    )
    x, ab = query_points.double(), b - a
    ab2 = (ab * ab).sum(-1)
    t = ((x - a) * ab).sum(-1) / torch.where(ab2 > 0, ab2, 1.0)
    off = x - (a + t[:, None] * ab)
    r = off.norm(dim=-1, keepdim=True)
    slide_u = (off / torch.where(r > 0, r, 1.0)).to(dtype)
    return diff, on_cell, face_n, tied, tie_u.to(dtype), sign.to(dtype), tie_dist, on_wall, slide, slide_u


def exact_boundary_geometry(
    query_points: Float[torch.Tensor, "queries dim"],
    points: Float[torch.Tensor, "vertices dim"],
    cells: Int[torch.Tensor, "cells dim"],
    kernel: str = "reference",
) -> tuple[Float[torch.Tensor, " queries"], Float[torch.Tensor, "queries dim"]]:
    r"""Signed distance and direction from a boundary surface to each query.

    The surface is the union of the cells ``cells`` (``(F, dim)`` vertex
    indices into ``points``, at least one; none raises a ``ValueError``):
    segments in two dimensions, triangles in three.
    For every query :math:`x` the closest point :math:`p` on the surface is
    found exactly (a Warp BVH in three dimensions, a brute force over the
    segments in two), and

    .. math::

        d = \operatorname{sign}\bigl(\tfrac{\sigma}{2} - w(x)\bigr)\,\lvert x - p \rvert,
        \qquad
        u = \begin{cases}
            \operatorname{sign}\bigl((x - p) \cdot \hat n_f\bigr)\, \hat n_f & p \text{ interior to the cell } f, \\
            (x - p) / \lvert x - p \rvert & \text{otherwise,}
        \end{cases}

    the distance to the surface signed by the generalized winding number
    :math:`w` of the oriented cells (Jacobson, Kavan and Sorkine-Hornung, 2013;
    evaluated as Barill et al., 2018, :func:`_winding_sign`) against half the sign
    :math:`\sigma` of the volume they enclose. For a closed surface
    :math:`\sigma / 2 - w` is :math:`+1/2` on the side the normals point to and
    :math:`-1/2` on the other, whichever way the normals point, so :math:`d` is
    **positive on the side the normals point to**. The two sheets of a
    zero-thickness fold (coincident cells of opposite orientation) cancel in
    :math:`w`, so both of its sides take the sign of the region around it. An open
    surface is signed by the standard threshold :math:`w = 1/2` (on the side of
    :math:`\sigma`). Its :math:`\sigma` depends on its shape, so close the boundary
    (give its far field) for a reliable sign. :math:`u` is the direction from the
    surface to the query (``sign(0) = +1``).

    A query is on the surface when its float64 distance is within the error bound of
    that distance (:func:`_closest_candidates`): its coordinates cannot tell it from
    a point of the surface. It takes :math:`d = 0`, and :math:`u` follows the same
    rule with the unit normals of the cells that hold it in place of the directions
    (below): the cell's normal inside a cell, and the least-norm point of the hull of
    the normals of the cells sharing an edge or a vertex. Its :math:`d` has the
    gradient of the distance along :math:`u`: :math:`u` in the query, as beside the
    surface inside a cell.

    Coincident vertices are welded first, so a surface stored as pieces or as a
    triangle soup has the connectivity of the surface it describes. A cell of zero
    area (two or three coincident vertices, or three collinear ones, by an exact
    test, :func:`_zero_area`) is the segment or the point its vertices span. If it
    lies along the edges of other cells or at one of their vertices, it adds nothing
    to the surface and is left out. Otherwise its closest point lies on that set; it
    has no normal and no solid angle, and a query on it takes the normals of the
    cells there that have one, or :math:`u = 0` with none.

    Where several features tie for the closest point within the error bound of the
    distance (:func:`_closest_candidates`), as on the medial surface (a plate's
    mid-plane, a corner's bisector), :math:`u` is the point of the convex hull of
    their directions :math:`u_k` closest to the origin. So
    :math:`\operatorname{sign}(d)\, u` is the least-norm element of the Clarke
    generalized gradient of :math:`d`: the midpoint of two directions, and zero when
    they cancel (no preferred direction). It is a function of the tie set, not of
    the order of the vertices, the cells or the search. Where cells lie at exactly
    the least distance and their directions do not tie, the closest cell is chosen by
    their positions, not their indices. A winding flip negates :math:`d` and
    :math:`\operatorname{sign}(d)\, u` bitwise, since every closest point is computed
    on its cell's vertices in the lexicographic order of their positions.

    :math:`d` and :math:`u` are functions of the surface alone, not of how a sample
    of it was drawn. They are invariant to splitting a cell and converge as
    :math:`O(h^2)` under refinement of a smooth surface. Under every rigid motion and
    reflection (with the cells re-oriented so their normals follow the map),
    :math:`d` is invariant and :math:`u` covariant; :math:`d` scales with the
    geometry.

    Every decision (a tie off the surface, the query on the surface, and the cells
    that hold it) is certified against the inputs' own rounding. The error bound is
    the rounding of the coordinates as given, in the queries' precision, plus a worst
    case of the float64 arithmetic. That arithmetic runs in the boundary's centred
    frame (its measure-weighted centroid as the origin), so a boundary far from the
    origin costs it nothing. Off the surface, directions tie only where they are
    certified distinct. On the surface the cells that hold the query are certified,
    and :math:`u` is the least-norm point of their normals wherever they differ.

    The search runs without gradient. The offset :math:`x - p` is then rebuilt in
    float64 from the query and its cell's vertices, with the barycentric weights of
    its Voronoi region (computed in float64; Ericson 2005, Sec. 5.1.5), and rounded
    once to the queries' precision. So :math:`d` and :math:`u` are differentiable in
    the query and the vertex positions. :math:`d` has the gradient of the envelope
    theorem (the closest point held fixed). :math:`u` has the derivative of the
    direction to the closest point as it moves: fixed at a vertex, sliding along a
    triangle's edge, the normal inside a cell. At a tie :math:`u` carries no
    gradient.

    ``kernel`` names the implementation of the search, not what it computes.
    ``"reference"`` defines it, and ``"fast"`` gives the same outputs bitwise from
    Triton kernels on CUDA for the weld, the canonical cells and the closest points'
    weights (``_geometry_fast``).
    """
    if query_points.shape[-1] not in (2, 3):
        raise NotImplementedError(
            f"the exact boundary geometry is implemented in two and three dimensions; got {query_points.shape[-1]}"
        )
    if cells.shape[0] == 0:
        raise ValueError("the boundary has no cells; there is no surface to measure distance to")
    diff, on_cell, face_n, tied, tie_u, sign, tie_dist, on_wall, slide, slide_u = _closest_features(
        query_points, points, cells, kernel
    )
    dist = diff.norm(dim=-1)
    plane = (diff * face_n).sum(-1)  # the distance on a cell's interior, signed by its normal
    u = torch.where(
        on_cell[:, None],
        torch.where((plane >= 0)[:, None], face_n, -face_n),
        diff / dist.clamp_min(torch.finfo(dist.dtype).tiny)[:, None],
    )
    ### (inside an edge the point slides: u keeps its value, with the sliding
    ### point's gradient)
    u = torch.where(slide[:, None], _StraightThrough.apply(u, slide_u), u)
    u = torch.where(on_wall[:, None], face_n, u)
    u = torch.where(tied[:, None], tie_u, u)
    ### (on the wall the distance along u, exactly zero, keeps its gradient: u)
    along = (diff * u).sum(-1)
    d = torch.where(
        on_wall,
        along - along.detach(),
        sign * torch.where(tied, tie_dist, torch.where(on_cell, plane.abs(), dist)),
    )
    return d, u

