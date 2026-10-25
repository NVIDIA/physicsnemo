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

"""The closest-feature search and :func:`exact_signed_distance`.

The search finds, for every query, the cell of its closest point on the surface,
whether several features tie for it (certified against the inputs' rounding), at a
tie the least-norm point of their directions, the sign from the winding number,
and whether the query is on the surface. It runs without gradient, in one custom
op; the offset from the closest point is then rebuilt differentiably.
"""

import itertools

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.spatial.exact_sdf._cell_tree import (
    _SEARCH_PAIRS,
    _ball_candidates,
    _nearest_cells,
    _preorder_cell_tree,
    _segment_candidates,
)
from physicsnemo.mesh.spatial.exact_sdf._predicates import (
    _canonical_cells,
    _centroid,
    _covered,
    _oriented_areas,
    _weld,
    _zero_area,
)
from physicsnemo.mesh.spatial.exact_sdf._weights import (
    _BAND_UNITS,
    _closest_weights,
    _thin_bounds,
)
from physicsnemo.mesh.spatial.exact_sdf._winding import (
    _packed_winding_sum,
    _winding_sign,
)


def _min_norm_points(c: torch.Tensor) -> torch.Tensor:
    """For each row of ``c`` ``(n, K, d)`` (``K`` distinct unit vectors), the point
    of their convex hull closest to the origin ``(n, d)``. It lies in the relative
    interior of a face spanned by at most ``d`` of them (Caratheodory), where it is
    that face's least-norm affine point with non-negative weights, and every such
    point lies in the hull: so it is the least-norm one over those faces (a pair's
    midpoint, since the vectors have equal norms; the first in the enumeration at
    equal norms) that meets the optimality condition ``c_k . x >= |x|^2``, which
    only the hull's least-norm point does (other faces' points can match its norm
    to rounding where vectors nearly coincide). When none meets it to rounding, the
    origin is outside the hull exactly when a candidate separates it (``c_k . x >
    0`` for every ``k``; the least-norm point does): then the candidate with the
    least duality gap ``g = |x|^2 - min_k c_k . x`` (``|x - x*| <= sqrt(2 g)``),
    otherwise 0. ``c`` in a canonical order, so the result is a function of the
    set. Chunked over the rows."""
    n, k, d = c.shape
    if k == 1:
        return c[:, 0]
    subsets = [
        torch.tensor(list(itertools.combinations(range(k), m)), device=c.device)
        for m in range(2, min(k, d) + 1)
    ]
    chunk = max(1, _SEARCH_PAIRS // 8 // sum(s.shape[0] for s in subsets))
    out = []
    for s in range(0, n, chunk):
        cc = c[s : s + chunk]
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
        slack = 2 * d * torch.finfo(c.dtype).eps
        low, sq = (cc[:, None] * x[:, :, None]).sum(-1).amin(-1), (x * x).sum(-1)
        norm = torch.where(low >= sq - slack, x.norm(dim=-1), torch.inf)
        gap = torch.where((low > slack) & torch.isfinite(sq), sq - low, torch.inf)
        certified = torch.isfinite(norm).any(1)
        pick = torch.where(certified, norm.argmin(1), gap.argmin(1))
        best = x.gather(1, pick[:, None, None].expand(-1, 1, d))[:, 0]
        out.append(
            torch.where((certified | torch.isfinite(gap).any(1))[:, None], best, 0.0)
        )
    return torch.cat(out)


def _grouped_min_norm(
    group: torch.Tensor, c: torch.Tensor, n_groups: int
) -> torch.Tensor:
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
        out[ids[here]] = _min_norm_points(
            c[start[here, None] + torch.arange(k, device=c.device)]
        )
    return out


def _certified_tied(
    tq: torch.Tensor, c: torch.Tensor, tau: torch.Tensor, nq: int
) -> torch.Tensor:
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


def _closest_candidates_impl(query_points, points, cells, parity, face_normals):
    """The closest-feature search (no gradient): for every query the cell of its
    closest point ``(Q,)``, whether several features tie for it ``(Q,)``, at a tie
    the least-norm point of the hull of their directions ``(Q, d)``, the sign of the
    distance from the winding number of the cells (:func:`_winding_sign`) ``(Q,)``,
    whether the query is on the surface (below) ``(Q,)``, and the barycentric
    weights of its closest point on its cell (:func:`_closest_weights`, float64)
    ``(Q, d)``; see
    :func:`exact_signed_distance`. ``cells`` in canonical order with their
    orientations ``parity`` (:func:`_canonical_cells`), the normals in float64.

    The search runs in the boundary's centred frame: every float64 coordinate less
    ``c``, the measure-weighted centroid of its distinct cells (:func:`_centroid`),
    rounded to the queries' precision (so float32 inputs centre exactly), so its
    arithmetic scales with the boundary's size, not with its distance from the
    origin. The exact predicates (:func:`_zero_area`) and the normals stay on the
    coordinates as given. Every candidate distance is computed in float64 on the
    canonical cell; the features tie when their distances are equal within the
    error bound ``E_k = u (|x| + R_k) + C u_64 (|x'| + R'_k)`` of the distance to
    cell ``k`` (``R_k = max_{v in k} |v|``; primes in the centred frame). Its first
    term is the rounding of the coordinates as they were given, ``u`` the unit
    roundoff of the queries' precision, so it is measured in the frame they were
    given in: a case posed far from the origin is rounded more coarsely, and a term
    measured about ``c`` would no longer bound it. Its second is float64's
    arithmetic in the frame it runs in, ``C`` a worst case (``_BAND_UNITS``: 91 in
    three dimensions, 17 in two; for float32 queries 1.7e-7 of the bound, for
    float64 ones it decides). Cell ``k`` ties when
    ``D_k - E_k <= min_j (D_j + E_j)``, i.e. when it may be the closest one, and
    its closest point is a local minimum of the distance on the surface: on a
    vertex or an edge, every cell sharing that feature reaches the same point
    (otherwise a neighbouring cell holds a closer point of the same patch, as
    beside the foot of the perpendicular on a cell next to the vertex), except on
    the surface, where every cell within the bound holds the query. The query is
    on the surface when its closest cell's distance interval reaches 0, ``D_min -
    E_min <= 0``: its coordinates cannot tell it from a point of the surface. Cells
    at exactly the least distance (cells sharing the closest point compute it
    bitwise) are ordered by their positions, the lexicographic order of their
    canonical vertices as given and then their parity, not by their indices, so the
    closest cell, its bound and its weights do not depend on the cells' order. A tie's
    directions are the unit vectors from the surface to the query, and on the
    surface the unit normals of the tied cells that have one (with none, the query
    keeps its closest cell's normal). On the surface the query is tied when two of
    these normals differ, certified distinct or not: their least-norm point lies in
    their hull, so within ``tau_k + max_j tau_j`` of each member ``k`` (the group's
    certified uncertainty) where none is certified distinct, and it is a function of
    the set, where one of them would depend on the cells' order and the certificate
    on the queries' precision. Off the surface it is tied when two of its directions
    are certified distinct, farther apart than the sum of their bounds
    ``tau = sqrt(2) s`` (2 where ``s >= 1``), ``s`` a bound on the sine of the angle
    between a computed direction and the exact one of the inputs as given (below).
    The least-norm point is then taken over all of the tie's directions, odd in them
    on the surface (half its value at the normals less its value at their negations),
    so a winding flip negates it bitwise.

    The exact search runs on candidate pairs that contain every cell with
    ``D_k - max_j E_j`` within that bound. In three dimensions a Warp traversal of a
    BVH over the cells (:func:`_preorder_cell_tree`) finds a cell nearest in float32
    (:func:`_nearest_cells`), whose exact distance plus its error bound bounds the
    closest distance, and a second traversal of the same BVH returns the cells within
    that radius (:func:`_ball_candidates`); in two
    a brute force in the queries' precision, on the centred coordinates, keeps the
    segments within its own rounding (``40 u (|x'| + max |v'|)`` on each side, with
    room over a first-order count of 33.5 units: the brute-force distance 14, the
    centred copies' rounding 1, the comparison's square and root 2.5, and the
    arithmetic share of the least's bound 16) and the error bounds of the least. The winding
    number takes the same BVH in three dimensions, unless cells repeat
    (:func:`_packed_winding_sum`)."""
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
            cells, parity, face_normals, simplex, flat = (
                x[kept] for x in (cells, parity, face_normals, simplex, flat)
            )
            any_flat = bool(flat.any())
    face_n = face_normals.double()
    ### The winding number is the surface's, not the cell list's: identical oriented
    ### cells count once. The cells go in the sorted order of their keys, so a surface
    ### stored with repeated cells has the same BVH, and the same far field, as the
    ### surface stored once (a stable lexicographic sort, as in _weld).
    ### (by two stable sorts of keys that pack adjacent columns)
    low = cells[:, -1] * 2 + (parity > 0)
    high = cells[:, 0] * points.shape[0] + cells[:, 1] if dim == 3 else cells[:, 0]
    order = torch.argsort(low, stable=True)
    order = order[torch.argsort(high[order], stable=True)]
    low, high = low[order], high[order]
    new = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    rep = order[torch.cat([new.new_ones(1), new])]
    del new, low, high
    ### The band: the storage term on the magnitudes as given, the arithmetic's on
    ### the centred ones. The float32 copies the BVH and the 2-D brute force read are
    ### the centred coordinates cast once.
    u_in, u64 = (
        torch.finfo(query_points.dtype).eps / 2,
        torch.finfo(torch.float64).eps / 2,
    )
    given_reach, given_q = simplex.norm(dim=-1).amax(-1), q64.norm(dim=-1)
    origin = (
        _centroid(simplex[rep], flat[rep] if any_flat else None)
        .to(query_points.dtype)
        .double()
    )
    q64, simplex = q64 - origin, simplex - origin
    query_c, points_c = (
        q64.to(query_points.dtype),
        (points.double() - origin).to(points.dtype),
    )
    reach, q_norm = simplex.norm(dim=-1).amax(-1), q64.norm(dim=-1)
    share = _BAND_UNITS[dim] * u64

    def band(qi, ki):
        return u_in * (given_q[qi] + given_reach[ki]) + share * (q_norm[qi] + reach[ki])

    err_max = u_in * (given_q + given_reach.max()) + share * (q_norm + reach.max())
    s_rep, p_rep = simplex[rep], parity[rep].to(q64)
    ### (no normal, no area, no solid angle: its parity 0 in the winding number)
    if any_flat:
        p_rep = torch.where(flat[rep], 0.0, p_rep)
    oriented = _oriented_areas(s_rep, p_rep)
    if dim == 3:
        ### Without repeated cells the two trees are one, and one traversal serves the
        ### ball search and the winding number.
        once = rep.shape[0] == cells.shape[0]
        s_ord = s_rep if once else simplex[order]
        f_ord = flat[order] if any_flat else None
        tree, leaf_order, s_leaf = _preorder_cell_tree(s_ord)
        if once:
            ### (the leaves hold the cells; the winding number reads them there)
            s_ord = s_rep = None
        seed = order[leaf_order[_nearest_cells(query_c, s_leaf, tree)]]
        w0 = _closest_weights(q64, simplex[seed], flat[seed] if any_flat else None)
        bound = (q64 - (w0[..., None] * simplex[seed]).sum(1)).norm(dim=-1) + band(
            torch.arange(nq, device=dev), seed
        )
        ### (padded by twice the bound, the kernel's float64 distances against these,
        ### and by the rounding of its box test)
        radius = (bound + 3 * err_max) * (1 + 1e-12)
        li, pk = _ball_candidates(
            q64,
            radius,
            s_leaf,
            tree,
            reach=reach.max(),
            flat=None if f_ord is None else f_ord[leaf_order],
        )
        pk = order[leaf_order[pk]]
        ### (the winding number reads the tree's links and leaves, not its boxes)
        tree = (*tree[:5], None, None)
        if not once:
            tree, leaf_order, s_leaf = _preorder_cell_tree(s_rep)
        w_sum = _packed_winding_sum(
            q64, s_leaf, p_rep[leaf_order], oriented[0][leaf_order], tree
        )
    else:
        w_sum = None
        rounding = 40 * torch.finfo(query_points.dtype).eps / 2 * (q_norm + reach.max())
        li, pk = _segment_candidates(
            query_c, points_c, cells, (2 * (rounding + err_max)).to(query_points.dtype)
        )
    ### In the order of (query, cell), whatever order the search found them in.
    f = cells.shape[0]
    key = (li * f + pk).sort().values
    li, pk = key // f, key % f
    tri = simplex[pk]
    w = _closest_weights(q64[li], tri, flat[pk] if any_flat else None)
    diff = q64[li] - (w[..., None] * tri).sum(1)
    dist = diff.norm(dim=-1)
    err = band(li, pk)
    inf = dist.new_full((nq,), torch.inf)
    d_min = inf.scatter_reduce(0, li, dist, "amin")
    upper = inf.scatter_reduce(0, li, dist + err, "amin")
    pair = torch.arange(li.shape[0], device=dev)
    first = torch.full((nq,), li.shape[0], dtype=torch.long, device=dev).scatter_reduce(
        0, li, torch.where(dist == d_min[li], pair, li.shape[0]), "amin"
    )
    ### Equal least distances (cells sharing the closest point compute it bitwise, so
    ### they tie exactly) are broken by the cells' positions, not by their indices: the
    ### lexicographic order of their canonical vertices as given, then their parity.
    ### So the closest cell, its band and its weights are functions of the positions.
    ### One key column at a time, each keeping the rows at the query's least value;
    ### identical keys are the same cell, so the least pair among them is the cell.
    least = dist == d_min[li]
    key = torch.cat(
        [points[cells[pk]].flatten(1), parity[pk, None].to(points.dtype)], 1
    )
    for k in key.unbind(1):
        k_min = k.new_full((nq,), torch.inf).scatter_reduce(
            0, li, torch.where(least, k, torch.inf), "amin"
        )
        least = least & (k == k_min[li])
    pick = torch.full_like(first, li.shape[0]).scatter_reduce(
        0, li, torch.where(least, pair, li.shape[0]), "amin"
    )
    first = torch.where(pick < li.shape[0], pick, first)
    idx, on_wall = pk[first], d_min - err[first] <= 0
    ### The feature of each closest point (a vertex, an edge, or -1 for a cell's
    ### interior), and whether all of its cells reach it.
    n_vertices = points.shape[0]
    nonzero = w > 0
    n_nonzero = nonzero.sum(-1)
    corners = cells[pk]
    key = torch.where(
        n_nonzero == 1, corners.gather(1, w.argmax(-1, keepdim=True))[:, 0], -1
    )
    if any_flat:
        ### A zero-area cell's features are its distinct vertices and the edges between
        ### consecutive distinct ones, each counted once (n_vertices: none).
        consecutive = cells[:, 1:] != cells[:, :-1]
        vertices = torch.where(
            flat[:, None]
            & ~torch.cat([torch.ones_like(consecutive[:, :1]), consecutive], 1),
            n_vertices,
            cells,
        )
        count = torch.bincount(vertices.reshape(-1), minlength=n_vertices + 1)[
            key.clamp_min(0)
        ]
    else:
        count = torch.bincount(cells.reshape(-1), minlength=n_vertices)[
            key.clamp_min(0)
        ]
    if dim == 3:
        a, b = cells[:, [0, 0, 1]], cells[:, [1, 2, 2]]
        if any_flat:
            edges = torch.stack(
                [consecutive[:, 0], torch.zeros_like(flat), consecutive[:, 1]], 1
            )
            a, b = (torch.where(flat[:, None] & ~edges, n_vertices, x) for x in (a, b))
        edge_keys, edge_valence = torch.unique(
            n_vertices + torch.minimum(a, b) * n_vertices + torch.maximum(a, b),
            return_counts=True,
        )
        ends = corners.masked_fill(~nonzero, n_vertices)
        ends = ends.sort(-1).values  # the edge's two vertices, ascending
        edge = n_vertices + ends[:, 0] * n_vertices + ends[:, 1]
        key = torch.where(n_nonzero == 2, edge, key)
        count = torch.where(
            n_nonzero == 2,
            edge_valence[
                torch.searchsorted(edge_keys, edge).clamp_max(edge_keys.shape[0] - 1)
            ],
            count,
        )
    reached = torch.ones_like(nonzero[:, 0])
    on_feature = key >= 0
    group = torch.stack([li[on_feature], key[on_feature]], -1)
    _, inverse, counts = torch.unique(
        group, dim=0, return_inverse=True, return_counts=True
    )
    reached[on_feature] = counts[inverse] == count[on_feature]
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
    ### Off the surface, a tie is two certified-distinct directions, |c_i - c_j| >
    ### tau_i + tau_j, tau = sqrt(2) s (the chord of a sine error s < 1; else 2).
    ### A cell's normal: its vertices each within eta = u |v| of the exact ones (their
    ### storage rounding), so |dX| <= 4 eta L + 4 eta^2 against the certified
    ### |X| - |E| (Shewchuk), with the float64 normal's own bound (delta) and its
    ### normalization. A direction from a point: the point within the band E_k (and
    ### Ericson's misplacement on a face), over D_k.
    pt = pk[tie]
    st = simplex[pt]
    eta = u_in * given_reach[pt]
    if dim == 3:
        delta_t, kappa_t, longest_t = _thin_bounds(st)
        size = torch.linalg.cross(
            st[:, 1] - st[:, 0], st[:, 2] - st[:, 0], dim=-1
        ).norm(dim=-1)
        s_n = (
            (4 * eta * longest_t + 4 * eta * eta) / (size * (1 - delta_t))
            + delta_t
            + 4 * u64
        )
        mis = kappa_t * (q64[tq][:, None] - st).square().sum(-1).amax(-1)
    else:
        length = (st[:, 1] - st[:, 0]).norm(dim=-1)
        s_n = 2 * eta / (length - 2 * eta) + 4 * u64
        mis = torch.zeros_like(length)
    s_n = torch.where(s_n >= 0, s_n, torch.inf)
    e_t, d_t = err[tie], dist[tie]
    s_p = torch.where(d_t > 0, (e_t + 4 * u64 * d_t) / d_t, torch.inf)
    s_c = s_n + torch.where(d_t > 0, (mis + e_t) / d_t, torch.inf)
    s = torch.where(wall, s_n, torch.where(on_cell, s_c, s_p))
    ### On the surface the bound is 0 (normals that differ tie): the least-norm point of
    ### their hull lies within tau_k + max_j tau_j of each member k, the group's
    ### certified uncertainty, so it needs no certificate, and it is the certified case's own value, so u does not depend on
    ### the queries' precision.
    tied = _certified_tied(
        tq, tc, torch.where(wall, 0.0, torch.where(s < 1, 2.0**0.5 * s, 2.0)), nq
    )
    if any_flat:
        ### On a zero-area closest cell the direction is the tie's over the cells
        ### that hold the query and have a normal: theirs, or 0 with none (the
        ### least-norm element of the distance's gradient on a segment or a point).
        tied = tied | (on_wall & flat[idx])
    sel = tied[tq]
    g, c, wall = tq[sel], tc[sel], wall[sel]
    m = _grouped_min_norm(
        torch.cat([g, g[wall] + nq]), torch.cat([c, -c[wall]]), 2 * nq
    )
    tie_u = torch.where(on_wall[:, None], 0.5 * (m[:nq] - m[nq:]), m[:nq])
    sign = _winding_sign(q64, s_rep, p_rep, oriented[1], w_sum)
    if kept is not None:
        idx = kept[idx]
    return (
        idx,
        tied,
        tie_u.to(query_points.dtype),
        sign.to(query_points.dtype),
        on_wall,
        w[first],
    )


_SearchOutputs = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


@torch.library.custom_op("physicsnemo::mesh_closest_features", mutates_args=())
def _closest_features_op(
    query_points: torch.Tensor,
    points: torch.Tensor,
    cells: torch.Tensor,
    parity: torch.Tensor,
    face_normals: torch.Tensor,
) -> _SearchOutputs:
    """The closest-feature search, :func:`_closest_candidates_impl`."""
    return _closest_candidates_impl(query_points, points, cells, parity, face_normals)


@_closest_features_op.register_fake
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


def _closest_features(
    query_points: Float[torch.Tensor, "queries dim"],
    points: Float[torch.Tensor, "vertices dim"],
    cells: Int[torch.Tensor, "cells dim"],
):
    """The closest surface point of each query and its feature: the offset ``x -
    p`` from the point ``(Q, d)``, evaluated in float64 as ``(x - v_0) - sum_{i > 0}
    w_i (v_i - v_0)`` from the coordinates as given and rounded once to the queries'
    precision, whether the point is interior to its cell ``(Q,)``, the cell's unit
    normal ``(Q, d)``, at a closest-feature tie (:func:`_closest_features_op`)
    whether there is one ``(Q,)`` and the direction ``(Q, d)``, the sign of the
    distance ``(Q,)``, at a tie the distance ``(Q,)``, and whether the query is on the
    surface ``(Q,)``; see
    :func:`exact_signed_distance`."""
    points = points.to(query_points.dtype)
    cells, parity = _canonical_cells(points, _weld(points)[cells])
    cell_points = points[cells]
    all_n = parity[:, None].to(points.dtype) * compute_cell_normals(
        cell_points[:, 1:] - cell_points[:, :1]
    )
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
        idx, tied, tie_u, sign, on_wall, weights = _closest_features_op(
            q, y, cells, parity, n64
        )
        corners = cells[idx]  # (Q, d)
    simplex = points[corners]  # (Q, d, d)
    s64 = simplex.double()
    diff = (
        (query_points.double() - s64[:, 0])
        - (weights[:, 1:, None] * (s64[:, 1:] - s64[:, :1])).sum(1)
    ).to(query_points.dtype)
    on_cell = (weights > 0).all(-1)
    dtype = query_points.dtype
    ### (in float64: off the surface a lower precision can round it to 0, losing the sign)
    tie_dist = (
        (query_points.double() - (weights[..., None] * simplex.double()).sum(1))
        .norm(dim=-1)
        .to(dtype)
    )
    face_n = torch.where(on_wall[:, None], n64[idx].to(dtype), all_n[idx])
    return (
        diff,
        on_cell,
        face_n,
        tied,
        tie_u.to(dtype),
        sign.to(dtype),
        tie_dist,
        on_wall,
    )


def exact_signed_distance(
    mesh: Mesh,
    query_points: Float[torch.Tensor, "... n_spatial_dims"],
) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "... n_spatial_dims"]]:
    r"""Exact signed distance and direction from a boundary mesh to each query.

    The surface is the union of the cells of ``mesh``: segments in two
    dimensions, triangles in three. For every query :math:`x` the closest point
    :math:`p` on the surface is found exactly (a Warp BVH in three dimensions, a
    brute force over the segments in two), in float64, and

    .. math::

        d = \operatorname{sign}\bigl(\tfrac{\sigma}{2} - w(x)\bigr)\,\lvert x - p \rvert,
        \qquad
        u = \begin{cases}
            \operatorname{sign}\bigl((x - p) \cdot \hat n_f\bigr)\, \hat n_f & p \text{ interior to the cell } f, \\
            (x - p) / \lvert x - p \rvert & \text{otherwise,}
        \end{cases}

    the distance to the surface signed by the generalized winding number
    :math:`w` of the oriented cells (Jacobson, Kavan and Sorkine-Hornung, 2013;
    evaluated as Barill et al., 2018) against half the sign :math:`\sigma` of the
    volume they enclose. For a closed surface :math:`\sigma / 2 - w` is
    :math:`+1/2` on the side the normals point to and :math:`-1/2` on the other,
    whichever way the normals point, so :math:`d` is **positive on the side the
    normals point to**. The two sheets of a zero-thickness fold (coincident cells
    of opposite orientation) cancel in :math:`w`, so both of its sides take the
    sign of the region around it, and an open surface is signed by the standard
    threshold :math:`w = 1/2` (on the side of :math:`\sigma`). :math:`u` is the
    direction from the surface to the query (``sign(0) = +1``).

    A query on the surface (its float64 distance within the error bound of that
    distance: its coordinates cannot tell it from a point of the surface) takes
    :math:`d = 0` and the same rule with the directions replaced by the unit
    normals of the cells that hold it: the cell's normal inside a cell, the
    least-norm point of the hull of the normals of the cells sharing an edge or a
    vertex. Coincident vertices are welded first, so a surface stored as pieces
    or as a triangle soup has the connectivity of the surface it describes. A
    cell of zero area (two or three coincident vertices, or three collinear ones,
    by an exact test) is the segment or the point its vertices span. Lying along
    the edges of other cells or at one of their vertices, it adds nothing to the
    surface and is left out; otherwise its closest point lies on that set, it has
    no normal and no solid angle, and a query on it takes the normals of the cells
    there that have one, or :math:`u = 0` with none.

    Where several features tie for the closest point (the medial surface: a
    plate's mid-plane, a corner's bisector), within the error bound of the
    distance computation, :math:`u` is the point of the convex hull of their
    directions :math:`u_k` closest to the origin, so
    :math:`\operatorname{sign}(d)\, u` is the least-norm element of the Clarke
    generalized gradient of :math:`d`: the midpoint of two directions, zero when
    they cancel. It is a function of the tie set, not of the order of the
    vertices, the cells or the search (where cells lie at exactly the least
    distance and their directions do not tie, the closest cell is chosen by their
    positions, not their indices), and a winding flip negates :math:`d` and
    :math:`\operatorname{sign}(d)\, u` bitwise.

    :math:`d` and :math:`u` are functions of the surface alone: invariant to
    splitting a cell, converging as :math:`O(h^2)` under refinement of a smooth
    surface, :math:`d` invariant and :math:`u` covariant under every rigid motion
    and reflection (with the cells re-oriented so their normals follow the map),
    :math:`d` scaling with the geometry.

    Every decision (a tie off the surface, the query on the surface and the
    cells that hold it) is certified against the inputs' own rounding: the error
    bound is the rounding of the coordinates as given in the queries' precision
    plus a worst case of the float64 arithmetic, which runs in the boundary's
    centred frame (its measure-weighted centroid as the origin), so a boundary
    far from the origin costs the arithmetic nothing. Off the surface,
    directions tie only where they are certified distinct; on it, the cells that
    hold the query are certified, and :math:`u` is the least-norm point of their
    normals wherever they differ.

    The search runs without gradient; the offset :math:`x - p` is then rebuilt
    from the query and its cell's vertices with the barycentric weights of its
    Voronoi region (computed in float64, Ericson 2005, Sec. 5.1.5), in float64,
    and rounded once to the queries' precision, so :math:`d` and :math:`u` are
    differentiable in ``query_points`` and ``mesh.points``, with the gradient of
    the envelope theorem (at a tie, :math:`u` carries none).

    Parameters
    ----------
    mesh : Mesh
        A boundary mesh of codimension one: segments in 2D
        (``n_manifold_dims == 1``) or triangles in 3D (``n_manifold_dims == 2``).
        It may be open, folded, self-intersecting, a triangle soup, or hold
        cells of zero area.
    query_points : torch.Tensor
        Query points, shape ``(..., n_spatial_dims)``, on the device of
        ``mesh``. The mesh's points are cast to their dtype. ``float16`` and
        ``bfloat16`` queries are searched in float32.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(distance, direction)``: the signed distance, shape
        ``query_points.shape[:-1]``, and the unit direction from the surface to
        the query (the least-norm point of the tied directions at a tie, which
        may be shorter than one), shape ``query_points.shape``, both in the dtype
        of ``query_points``.

    Raises
    ------
    ValueError
        If ``mesh`` is not of codimension one (``n_manifold_dims ==
        n_spatial_dims - 1``), if the last dimension of ``query_points`` is not
        ``mesh.n_spatial_dims``, if ``mesh`` has no cells, or if ``mesh`` and
        ``query_points`` are on different devices.
    NotImplementedError
        If ``mesh.n_spatial_dims`` is not 2 or 3.

    See Also
    --------
    signed_distance_field : Warp's float32 mesh query on a triangle mesh in 3D.

    Notes
    -----
    :func:`signed_distance_field` is faster on ordinary closed triangle meshes
    (several times, measured on surfaces of 10^4 to 10^7 triangles), works in
    float32 and makes no host synchronizations. Prefer this function where the
    distance must be exact in float64 and its decisions certified, in two
    dimensions, on open, degenerate or folded surfaces, or on boundaries of large
    extent (such as a vehicle inside its whole flow domain). It synchronizes with
    the host several times per call.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> from physicsnemo.mesh.spatial import exact_signed_distance
    >>> sphere = sphere_icosahedral.load(subdivisions=2)
    >>> query = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    >>> distance, direction = exact_signed_distance(sphere, query)
    >>> distance.shape, direction.shape
    (torch.Size([2]), torch.Size([2, 3]))
    >>> bool(distance[0] < 0 < distance[1])
    True
    """
    if mesh.n_manifold_dims != mesh.n_spatial_dims - 1:
        raise ValueError(
            "exact_signed_distance requires a mesh of codimension one "
            f"(n_manifold_dims == n_spatial_dims - 1), but got {mesh.n_manifold_dims=} "
            f"and {mesh.n_spatial_dims=}."
        )
    if query_points.shape[-1] != mesh.n_spatial_dims:
        raise ValueError(
            f"query_points must have last dimension {mesh.n_spatial_dims=}, "
            f"but got {query_points.shape=}."
        )
    if mesh.n_spatial_dims not in (2, 3):
        raise NotImplementedError(
            "exact_signed_distance is implemented in two and three dimensions, "
            f"but got {mesh.n_spatial_dims=}."
        )
    if mesh.n_cells == 0:
        raise ValueError(
            "mesh has no cells; there is no surface to measure distance to"
        )
    if mesh.points.device != query_points.device:
        raise ValueError(
            "mesh and query_points must be on the same device, but got "
            f"{mesh.points.device=} and {query_points.device=}."
        )
    diff, on_cell, face_n, tied, tie_u, sign, tie_dist, on_wall = _closest_features(
        query_points.reshape(-1, mesh.n_spatial_dims), mesh.points, mesh.cells
    )
    dist = diff.norm(dim=-1)
    plane = (diff * face_n).sum(
        -1
    )  # the distance on a cell's interior, signed by its normal
    u = torch.where(
        on_cell[:, None],
        torch.where((plane >= 0)[:, None], face_n, -face_n),
        diff / dist.clamp_min(torch.finfo(dist.dtype).tiny)[:, None],
    )
    u = torch.where(on_wall[:, None], face_n, u)
    u = torch.where(tied[:, None], tie_u, u)
    d = torch.where(
        on_wall,
        torch.zeros_like(dist),
        sign * torch.where(tied, tie_dist, torch.where(on_cell, plane.abs(), dist)),
    )
    return d.reshape(query_points.shape[:-1]), u.reshape(query_points.shape)
