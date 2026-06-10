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

"""Vectorized ear clipping for the rare non-convex polygons.

A vertex-0 fan is only valid for convex polygons; for a non-convex polygon its
fan triangles overlap and exit the polygon, so their unsigned areas no longer
sum to the polygon's. Ear clipping produces a true triangulation (`k - 2`
triangles, no added points) instead.

Ear clipping is sequential per polygon, but is vectorized across polygons by
grouping them by valence ``k`` (the ``torch.unique`` group-by-count pattern used
in :mod:`physicsnemo.mesh.calculus._neighborhoods`): every polygon in a group is
clipped in lockstep over a static ``k - 2`` rounds. This is the rare fallback,
reached only when :func:`physicsnemo.mesh.tessellation.triangulate` finds a
non-convex cell, so it is allowed the ``torch.unique`` topology graph break.
"""

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.mesh.neighbors._adjacency import Adjacency
from physicsnemo.mesh.utilities._tolerances import safe_eps

#: Containment tolerance for the ear test, relative to each candidate ear's own
#: area: an active vertex whose signed area against an ear edge is within this
#: fraction of the ear's ``|2 x area|`` is treated as lying *on* that edge and
#: blocks the clip. Without it, a vertex sitting exactly on an ear edge slips
#: through a strict interior test, letting the ear be clipped anyway and emit
#: overlapping triangles that over-count the polygon's unsigned area. The
#: magnitude matches ``triangulate``'s reflex tolerance and stays comfortably
#: above float32 round-off for well-conditioned (origin-centered) polygons.
_IN_TRIANGLE_REL_TOL: float = 1e-6


def reclip_nonconvex(
    points: Float[torch.Tensor, "n_points 3"],
    polygons: Adjacency,
    normals: Float[torch.Tensor, "n_polygons 3"],
    cells: Int[torch.Tensor, "n_triangles 3"],
    nonconvex: Bool[torch.Tensor, " n_polygons"],
) -> Int[torch.Tensor, "n_triangles 3"]:
    """Overwrite each non-convex polygon's fan triangles with an ear-clip.

    Polygons are grouped by valence; each group is a dense ``(m, k)`` batch
    ear-clipped together. The output triangle block of each non-convex polygon
    (contiguous and `k - 2` long, matching the fan layout) is replaced in place,
    so ``parent_index`` and triangle ordering are unchanged.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates embedded in 3D, shape ``(n_points, 3)``.
    polygons : Adjacency
        Cell-to-vertex incidence (CSR).
    normals : torch.Tensor
        Per-polygon Newell normals, shape ``(n_polygons, 3)``.
    cells : torch.Tensor
        Fan triangulation to patch, shape ``(n_triangles, 3)``.
    nonconvex : torch.Tensor
        Boolean mask of polygons to re-triangulate, shape ``(n_polygons,)``.

    Returns
    -------
    torch.Tensor
        ``cells`` with the non-convex blocks replaced (mutated in place).
    """
    starts = polygons.offsets[:-1]
    n_tris_per_poly = polygons.counts - 2
    tri_starts = (
        torch.cumsum(n_tris_per_poly, 0) - n_tris_per_poly
    )  # first out-tri per polygon

    nonconvex_polys = torch.nonzero(nonconvex, as_tuple=False).flatten()
    nonconvex_counts = polygons.counts[nonconvex_polys]
    device = points.device

    # Bounded loop over distinct valences; each iteration is fully vectorized.
    for valence in torch.unique(nonconvex_counts).tolist():
        group = nonconvex_polys[nonconvex_counts == valence]
        cols = torch.arange(valence, device=device)
        vertex_ids = polygons.indices[
            starts[group].unsqueeze(1) + cols
        ]  # (m, k) global
        local_tris = _parallel_ear_clip(
            points[vertex_ids], normals[group]
        )  # (m, k-2, 3)

        n_per = valence - 2
        global_tris = torch.gather(
            vertex_ids, 1, local_tris.reshape(group.shape[0], -1)
        ).reshape(-1, 3)
        dst = (
            tri_starts[group].unsqueeze(1) + torch.arange(n_per, device=device)
        ).reshape(-1)
        cells.index_copy_(0, dst, global_tris)

    return cells


def _parallel_ear_clip(
    coords: Float[torch.Tensor, "m k 3"], normals: Float[torch.Tensor, "m 3"]
) -> Int[torch.Tensor, "m k_minus_2 3"]:
    """Ear-clip a batch of equal-valence polygons into local-index triangles.

    Each polygon is projected to a 2D frame in which it is counter-clockwise,
    then clipped over ``k - 2`` static rounds: every round finds one ear per
    polygon (vectorized), emits it, and compacts that vertex out of the ring.
    Output triangles inherit the counter-clockwise winding, so their normals
    agree with the polygon's Newell normal (no separate winding fix needed).
    """
    m, k, _ = coords.shape
    device = coords.device
    coords_2d = _project_to_ccw_plane(coords, normals)  # (m, k, 2)

    active = torch.arange(k, device=device).expand(m, k).contiguous()  # ring order
    triangles = torch.empty((m, k - 2, 3), dtype=torch.long, device=device)
    rows = torch.arange(m, device=device)

    for r in range(k - 3):
        prev, nxt = active.roll(1, dims=1), active.roll(-1, dims=1)
        # First ear per polygon (argmax picks the first True; valid simple
        # polygons always have an ear, so the all-False fallback to 0 is inert).
        pick = _ear_mask(coords_2d, prev, active, nxt).to(coords_2d.dtype).argmax(dim=1)
        triangles[:, r, 0] = prev[rows, pick]
        triangles[:, r, 1] = active[rows, pick]
        triangles[:, r, 2] = nxt[rows, pick]
        # Compact the clipped vertex out of every polygon's ring.
        keep = torch.arange(active.shape[1], device=device) != pick.unsqueeze(1)
        active = active[keep].reshape(m, active.shape[1] - 1)

    triangles[:, k - 3] = active[:, :3]  # the final three survivors, in ring order
    return triangles


def _ear_mask(
    coords_2d: Float[torch.Tensor, "m k 2"],
    prev: Int[torch.Tensor, "m length"],
    active: Int[torch.Tensor, "m length"],
    nxt: Int[torch.Tensor, "m length"],
) -> Bool[torch.Tensor, "m length"]:
    """Whether each active vertex is an ear tip (convex and empty)."""
    a, tip, c = (_gather_rows(coords_2d, idx) for idx in (prev, active, nxt))
    convex_tip = _signed_area2(a, tip, c) > 0  # CCW polygon -> ear tip is a left turn

    # No other active vertex may lie inside the candidate ear (tip, cand) grid.
    inside = _point_in_triangle(
        tip[:, None, :, :], a[:, :, None, :], tip[:, :, None, :], c[:, :, None, :]
    )
    is_corner = (
        (active[:, None, :] == prev[:, :, None])
        | (active[:, None, :] == active[:, :, None])
        | (active[:, None, :] == nxt[:, :, None])
    )
    return convex_tip & ~(inside & ~is_corner).any(dim=2)


# ---------------------------------------------------------------------------
# 2D geometry primitives (broadcast over arbitrary leading dimensions)
# ---------------------------------------------------------------------------


def _project_to_ccw_plane(
    coords: Float[torch.Tensor, "m k 3"], normals: Float[torch.Tensor, "m 3"]
) -> Float[torch.Tensor, "m k 2"]:
    """Project each polygon onto an in-plane (u, w) frame with ``u x w = n_hat``.

    Because the frame is right-handed about the Newell normal, the polygon's
    original vertex order is counter-clockwise in these 2D coordinates.
    """
    eps = safe_eps(coords.dtype)
    normal_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(eps)
    u = _arbitrary_perp(normal_hat)
    w = torch.linalg.cross(normal_hat, u)
    return torch.stack(
        [(coords * u[:, None, :]).sum(-1), (coords * w[:, None, :]).sum(-1)], dim=-1
    )


def _arbitrary_perp(
    normal_hat: Float[torch.Tensor, "m 3"],
) -> Float[torch.Tensor, "m 3"]:
    """A unit vector orthogonal to each normal (numerically well-conditioned)."""
    eps = safe_eps(normal_hat.dtype)
    world_axis = torch.eye(3, device=normal_hat.device, dtype=normal_hat.dtype)[
        normal_hat.abs().argmin(dim=-1)  # axis least aligned with the normal
    ]
    perp = world_axis - (world_axis * normal_hat).sum(-1, keepdim=True) * normal_hat
    return perp / perp.norm(dim=-1, keepdim=True).clamp_min(eps)


def _signed_area2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Twice the signed area of 2D triangle ``(a, b, c)`` (positive = CCW)."""
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (c[..., 0] - a[..., 0])


def _point_in_triangle(
    p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    r"""Whether ``p`` lies in the *closed* CCW triangle ``(a, b, c)``.

    The test is boundary-inclusive: a point lying on an edge -- within
    :data:`_IN_TRIANGLE_REL_TOL` of it, relative to the triangle's own area --
    counts as contained. Ear clipping needs this. A vertex sitting exactly on a
    candidate ear's edge must block the clip; a strict interior test (``> 0``)
    classifies that vertex as outside, lets the ear be clipped, and produces
    overlapping triangles whose unsigned areas over-count the polygon (see
    :data:`_IN_TRIANGLE_REL_TOL`).

    Parameters
    ----------
    p : torch.Tensor
        Query point(s), shape :math:`(\dots, 2)`.
    a, b, c : torch.Tensor
        Triangle vertices in counter-clockwise order, each shape
        :math:`(\dots, 2)`.

    Returns
    -------
    torch.Tensor
        Boolean tensor, broadcast over the leading dimensions of the inputs,
        ``True`` where ``p`` is inside or on the boundary of ``(a, b, c)``.
    """
    # Tolerance relative to the triangle's own size (its ``|2 x area|``) keeps
    # the boundary test scale-free; a point is "on" an edge when its signed area
    # against that edge is within round-off of zero.
    tol = _IN_TRIANGLE_REL_TOL * _signed_area2(a, b, c).abs()
    return (
        (_signed_area2(a, b, p) >= -tol)
        & (_signed_area2(b, c, p) >= -tol)
        & (_signed_area2(c, a, p) >= -tol)
    )


def _gather_rows(
    values: Float[torch.Tensor, "m k d"], idx: Int[torch.Tensor, "m length"]
) -> Float[torch.Tensor, "m length d"]:
    """Gather ``values[i, idx[i, j]]`` -> ``(m, length, d)``."""
    rows = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
    return values[rows, idx]
