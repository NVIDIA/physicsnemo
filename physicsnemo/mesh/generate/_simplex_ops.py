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

"""Batched simplex primitives shared by the implicit-domain mesher.

Everything here is dimension-generic: cells are ``(n_cells, d+1)`` integer
tensors over ``(n_points, d)`` coordinates, for any spatial dimension ``d``.
"""

import math
from itertools import combinations

import torch

__all__ = [
    "boundary_vertex_mask",
    "boundary_is_closed_manifold",
    "compact_mesh",
    "facet_census",
    "orient_positive",
    "signed_volumes",
    "triangle_angles",
    "volume_length_quality",
]


def signed_volumes(points, cells):
    """Signed simplex volumes: ``det(edge matrix) / d!``, shape ``(n_cells,)``."""
    d = points.shape[1]
    p0 = points[cells[:, 0]]
    rel = points[cells[:, 1:]] - p0[:, None, :]
    return torch.linalg.det(rel) / math.factorial(d)


def volume_length_quality(points, cells):
    """Volume-length quality ``q in (0, 1]``; the regular simplex scores 1.

    ``q = c_d * V / l_rms^d`` with the constant chosen so every degeneracy
    mode (sliver, needle, cap) drives ``q -> 0``. Signed: inverted simplices
    score negative. This is the d-dimensional volume-length measure.
    """
    d = points.shape[1]
    vol = signed_volumes(points, cells)
    verts = points[cells]
    ii, jj = torch.triu_indices(d + 1, d + 1, offset=1)
    e = verts[:, ii, :] - verts[:, jj, :]
    l_rms = (e * e).sum(-1).mean(dim=1).sqrt()
    c_d = math.factorial(d) * math.sqrt(2.0**d) / math.sqrt(d + 1.0)
    return c_d * vol / (l_rms**d).clamp_min(1e-300)


def triangle_angles(points, cells):
    """Interior angles in degrees, shape ``(n_cells, 3)``. 2D reporting metric."""
    verts = points[cells]
    angles = []
    for i in range(3):
        u = verts[:, (i + 1) % 3] - verts[:, i]
        v = verts[:, (i + 2) % 3] - verts[:, i]
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-300)
        angles.append(torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0))))
    return torch.stack(angles, dim=1)


def _sub_simplices(cells, sub_size):
    """All ``sub_size``-vertex sub-simplices, sorted per row.

    Returns ``(subs (n_cells * C, sub_size), owner (n_cells * C,))`` where C
    is ``comb(d+1, sub_size)`` and ``owner`` maps each row to its cell.
    """
    m, dp1 = cells.shape
    combos = torch.tensor(
        list(combinations(range(dp1), sub_size)),
        dtype=torch.int64,
        device=cells.device,
    )
    subs, _ = torch.sort(cells[:, combos], dim=-1)
    owner = torch.arange(m, device=cells.device).repeat_interleave(combos.shape[0])
    return subs.reshape(-1, sub_size), owner


# Vertex-id packing bound: rows pack into one int64 when every id fits in
# floor(63 / row_width) bits, which makes `unique` a 1D sort instead of a
# lexicographic row sort (the mesher's hottest operation).
def _unique_rows(rows):
    """`torch.unique(dim=0)` with a packed-int64 fast path for small ids."""
    n, w = rows.shape
    bits = 63 // w
    if n > 0 and rows.max() < (1 << bits):
        packed = rows[:, 0].clone()
        for i in range(1, w):
            packed = (packed << bits) | rows[:, i]
        uniq_p, inverse, counts = torch.unique(
            packed, return_inverse=True, return_counts=True
        )
        # Recover representative rows for the unique keys.
        first = torch.zeros_like(uniq_p, dtype=torch.int64).scatter_reduce_(
            0,
            inverse,
            torch.arange(n, device=rows.device),
            reduce="amin",
            include_self=False,
        )
        return rows[first], inverse, counts
    return torch.unique(rows, dim=0, return_inverse=True, return_counts=True)


def facet_census(cells):
    """Unique ``(d-1)``-facets with counts and owner recovery.

    Returns ``(facets (F, d), counts (F,), owners (n_cells*(d+1),),
    inverse (n_cells*(d+1),))``: ``inverse[i]`` is the unique-facet index of
    the ``i``-th (cell, facet-slot) incidence and ``owners[i]`` its cell.
    """
    d = cells.shape[1] - 1
    subs, owner = _sub_simplices(cells, d)
    uniq, inverse, counts = _unique_rows(subs)
    return uniq, counts, owner, inverse


def boundary_vertex_mask(points, cells):
    """Boolean mask of vertices lying on any boundary (count-1) facet."""
    uniq, counts, _, _ = facet_census(cells)
    mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    mask[uniq[counts == 1].reshape(-1)] = True
    return mask


def boundary_is_closed_manifold(cells) -> bool:
    """True iff every boundary ridge lies in exactly two boundary facets."""
    uniq, counts, _, _ = facet_census(cells)
    bnd = uniq[counts == 1]
    if bnd.shape[0] == 0:
        return False
    dd = bnd.shape[1]
    if dd == 1:
        return True
    ridges = []
    for drop in range(dd):
        keep = [i for i in range(dd) if i != drop]
        ridges.append(bnd[:, keep])
    ridges, _ = torch.sort(torch.cat(ridges, dim=0), dim=1)
    _, _, rcounts = _unique_rows(ridges)
    return bool((rcounts == 2).all())


def orient_positive(points, cells):
    """Swap two vertices of negatively-oriented cells; returns new cells."""
    vol = signed_volumes(points, cells)
    neg = vol < 0
    if neg.any():
        cells = cells.clone()
        cells[neg] = cells[neg][:, [1, 0] + list(range(2, cells.shape[1]))]
    return cells


def compact_mesh(points, cells):
    """Drop unreferenced points and reindex cells."""
    used, inverse = torch.unique(cells.reshape(-1), return_inverse=True)
    return points[used], inverse.reshape(cells.shape)
