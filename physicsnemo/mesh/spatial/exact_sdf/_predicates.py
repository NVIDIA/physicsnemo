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

"""Exact predicates on the cells of a boundary mesh.

The bitwise weld of coincident vertices, each cell's vertices in the order of
their positions, the exact zero-area test (Shewchuk's filtered expansions), the
cells that add nothing to the surface, the exactly summed centroid and the
oriented areas.
"""

import torch
from jaxtyping import Float, Int


def _weld(
    points: Float[torch.Tensor, "vertices dim"],
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
    first = torch.cummax(
        torch.where(new, torch.arange(n, device=points.device), 0), 0
    ).values
    return torch.zeros_like(order).index_copy(0, order, order[first])


def _canonical_cells(points, cells):
    """Each cell's vertices in the lexicographic order of their positions, ``(F,
    d)``, and the parity of that permutation ``(F,)`` (+1 or -1): a function of the
    positions alone, the same for every numbering of the vertices and every
    rotation of a cell's vertex list, so the closest points computed on it are too;
    a winding flip changes only the parity, which carries the orientation."""
    k = cells.shape[1]
    vert, slot = (
        list(cells.unbind(1)),
        [torch.full_like(cells[:, 0], i) for i in range(k)],
    )
    ### A stable sort per coordinate, as argsort's (NaN last), by adjacent swaps of
    ### strictly smaller keys: elementwise, where a sort along rows of length k is not.
    for j in reversed(range(points.shape[1])):
        key = [points[v, j] for v in vert]
        for a in (a for p in range(k - 1) for a in range(k - 1 - p)):
            x, y = key[a], key[a + 1]
            swap = (y < x) | (x.isnan() & ~y.isnan())
            for col in (vert, slot, key):
                col[a], col[a + 1] = (
                    torch.where(swap, col[a + 1], col[a]),
                    torch.where(swap, col[a], col[a + 1]),
                )
    inversions = sum(
        (slot[a] > slot[b]).long() for a in range(k) for b in range(a + 1, k)
    )
    return torch.stack(vert, 1), 1 - 2 * (inversions % 2)


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
    search reads them) has zero measure, exactly, ``(F,)``, and whether any has. A
    segment has zero length when its two vertices coincide (welded, one index); a
    triangle has zero area when its cross product is exactly zero (two or three
    vertices coincide, or three distinct ones are collinear). Each component of the
    cross product is first evaluated in float64 with Shewchuk's (1997) error bound,
    which proves it nonzero for every cell of nonzero area but the very thinnest;
    those left are evaluated exactly, ``a x b + b x c + c x a`` as six exact products
    (:func:`_two_product`) summed as a nonoverlapping expansion (Shewchuk's
    Grow-Expansion), zero exactly when its every term is. Exact unless an
    intermediate overflows or underflows: never for float32 coordinates (in float64
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
        terms += [
            *_two_product(x[:, i], y[:, j]),
            *(-t for t in _two_product(x[:, j], y[:, i])),
        ]
    expansion = terms[:1]
    for t in terms[1:]:
        grown = []
        for e in expansion:
            t, h = _two_sum(t, e)
            grown.append(h)
        expansion = [*grown, t]
    return expansion


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
        return (
            keys[torch.searchsorted(keys, k).clamp_max(max(keys.shape[0] - 1, 0))] == k
            if keys.numel()
            else torch.zeros_like(flat)
        )

    v0, v1, v2 = cells.unbind(1)
    segment = ((v0 == v1) | edge(v0, v1)) & ((v1 == v2) | edge(v1, v2)) | edge(v0, v2)
    return flat & torch.where(v0 == v2, vertex[v0], segment)


def _centroid(simplex: torch.Tensor, flat=None) -> torch.Tensor:
    """The measure-weighted centroid ``(d,)`` of the cells ``simplex`` ``(F, d, d)``
    (float64; the zero-area ones ``flat`` weigh 0), a function of the set of cells,
    bitwise, whatever their order: each cell's measure and measure-weighted midpoint
    in float64, summed exactly in int64 fixed point (each column scaled so that ``F``
    times its greatest magnitude stays below ``2^62``, then truncated). The origin
    when every measure is 0."""
    if simplex.shape[-1] == 3:
        m = torch.linalg.cross(
            simplex[:, 1] - simplex[:, 0], simplex[:, 2] - simplex[:, 0], dim=-1
        ).norm(dim=-1)
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
    g = (measure[:, None] * centroid).sum(0) / measure.sum().clamp_min(
        torch.finfo(measure.dtype).tiny
    )
    return area, torch.where((area * (centroid - g)).sum() >= 0, 1.0, -1.0)
