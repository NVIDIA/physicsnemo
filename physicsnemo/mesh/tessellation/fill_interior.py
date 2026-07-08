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

"""Mesh-native interior filling: boundary ``Mesh`` in, volume ``Mesh`` out.

``fill_interior`` takes a closed codimension-one boundary mesh (edge loops
in 2D; a watertight surface in 3D, planned) and produces a quality simplex
mesh of the enclosed interior, preserving the boundary exactly: every input
vertex appears bit-identically in the output, and boundary facets are never
moved off the input geometry (they may be subdivided during refinement).
This is the exact-boundary counterpart to
``physicsnemo.mesh.generate.mesh_implicit_domain``, which meshes
implicit domains approximately but works in any dimension.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

__all__ = ["fill_interior"]


def _extract_loops(points_2d, edges):
    """Ordered vertex-index loops from a closed 1-manifold edge mesh.

    Requires every referenced vertex to have degree exactly 2. Returns a
    list of 1D int64 index tensors (one per loop, arbitrary orientation).
    """
    n_points = points_2d.shape[0]
    degree = torch.zeros(n_points, dtype=torch.int64)
    degree.index_add_(
        0, edges.reshape(-1), torch.ones(edges.numel(), dtype=torch.int64)
    )
    used = degree > 0
    if not bool((degree[used] == 2).all()):
        bad = torch.nonzero(used & (degree != 2)).reshape(-1)[:5].tolist()
        raise ValueError(
            f"boundary must be a closed 1-manifold (every vertex on exactly "
            f"2 edges); vertices {bad} have degree != 2. Open curves, "
            f"T-junctions, and duplicated edges are not fillable."
        )
    # Two neighbor slots per vertex.
    nbr = torch.full((n_points, 2), -1, dtype=torch.int64)
    slot = torch.zeros(n_points, dtype=torch.int64)
    for a, b in edges.tolist():
        nbr[a, slot[a]] = b
        slot[a] += 1
        nbr[b, slot[b]] = a
        slot[b] += 1
    visited = torch.zeros(n_points, dtype=torch.bool)
    visited[~used] = True
    loops = []
    for start in range(n_points):
        if visited[start]:
            continue
        loop = [start]
        visited[start] = True
        prev, cur = start, int(nbr[start, 0])
        while cur != start:
            loop.append(cur)
            visited[cur] = True
            a, b = int(nbr[cur, 0]), int(nbr[cur, 1])
            prev, cur = cur, (b if a == prev else a)
        if len(loop) < 3:
            raise ValueError(
                f"boundary contains a degenerate loop of {len(loop)} "
                f"vertices (needs >= 3)"
            )
        loops.append(torch.tensor(loop, dtype=torch.int64))
    return loops


def _point_in_polygon(point, poly):
    """Even-odd crossing test (poly: (N, 2) float64, point: (2,))."""
    a = poly
    b = torch.roll(poly, -1, dims=0)
    straddle = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    t = (point[1] - a[:, 1]) / (b[:, 1] - a[:, 1] + 1e-300)
    x_cross = a[:, 0] + t * (b[:, 0] - a[:, 0])
    return bool((straddle & (x_cross > point[0])).sum() % 2 == 1)


def _group_components(loop_polys):
    """Group loops into (outer, [holes]) components by containment depth.

    Even nesting depth = a component's outer boundary; its holes are the
    loops one level deeper that it directly contains. Islands inside holes
    start new components (any nesting depth is supported).
    """
    n = len(loop_polys)
    contains = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                contains[i][j] = _point_in_polygon(loop_polys[j][0], loop_polys[i])
    depth = [sum(contains[i][j] for i in range(n)) for j in range(n)]
    components = []
    for j in range(n):
        if depth[j] % 2 == 0:
            holes = [k for k in range(n) if depth[k] == depth[j] + 1 and contains[j][k]]
            components.append((j, holes))
    return components


def fill_interior(
    boundary: "Mesh",
    *,
    max_cell_size: float | None = None,
    min_angle_degrees: float = 30.0,
    smooth_iterations: int = 0,
) -> "Mesh":
    r"""Fill the interior of a closed boundary mesh with quality simplices.

    Dimension-generic contract: given a closed codimension-one boundary
    ``Mesh[n-1, n]``, produce a volume ``Mesh[n, n]`` of the enclosed
    interior such that

    - every input vertex appears **bit-identically** in the output (the
      leading rows of ``points``, in input order);
    - boundary facets are never moved off the input geometry — refinement
      may *subdivide* them, but the union of output boundary facets equals
      the input boundary exactly;
    - interior (Steiner) vertices are inserted to meet the quality bounds.

    Currently implemented for ``n = 2`` (edge loops -> triangles), where
    every output triangle is **guaranteed** a minimum angle of
    ``min_angle_degrees`` (Ruppert refinement; deterministic, bitwise
    reproducible). ``n = 3`` (watertight surface -> tetrahedra) raises
    :class:`NotImplementedError`; for *approximate* interior meshing of a
    surface today, build an SDF from it
    (``physicsnemo.mesh.spatial.signed_distance_field_mesh``) and use
    ``physicsnemo.mesh.generate.mesh_implicit_domain``, which trades
    the exact-boundary contract for dimension generality.

    Parameters
    ----------
    boundary : Mesh
        Closed codimension-one boundary: ``n_manifold_dims ==
        n_spatial_dims - 1``. In 2D, an edge mesh forming one or more
        disjoint simple loops (any orientation, any order); nesting is
        resolved automatically — loops at even containment depth bound
        components, loops one level deeper bound holes, islands inside
        holes are supported. Vertices not referenced by any edge are
        ignored. Loops of distinct components/holes must be disjoint, and
        segments of distinct loops should meet at angles of at least ~60°
        for the refinement termination guarantee.
    max_cell_size : float, optional
        Maximum cell measure (area in 2D). ``None`` disables the size
        bound. For a target edge length :math:`h`, pass the equilateral
        measure :math:`\sqrt{3}/4\,h^2`.
    min_angle_degrees : float, default 30.0
        Guaranteed minimum triangle angle, in :math:`[0, 33]` (2D).
    smooth_iterations : int, default 0
        Quality-gated ODT smoothing passes after refinement; boundary
        vertices never move, and the quality bounds are preserved.

    Returns
    -------
    Mesh
        Volume mesh on the input's device and dtype, positively oriented,
        with provenance in ``point_data``:

        - ``"boundary_marker"`` (int64): 1 for vertices on the input
          boundary (input vertices and refinement midpoints inserted on
          it), 0 for interior Steiner vertices.
        - ``"source_point"`` (int64): for vertices inherited from the
          input, the index of the originating input vertex; -1 for
          generated vertices. Use it to propagate input ``point_data``
          onto the output.

    Raises
    ------
    ValueError
        If the boundary is not a closed 1-manifold, loops are degenerate
        or crossing, or quality parameters are out of range.
    NotImplementedError
        For ``n_spatial_dims != 2`` (3D tetrahedralization is planned;
        the contract above is dimension-generic by design).

    Examples
    --------
    Fill an annulus given as one edge mesh containing both circles:

    >>> import math, torch
    >>> from physicsnemo.mesh import Mesh
    >>> from physicsnemo.mesh.tessellation import fill_interior
    >>> def circle(r, n, start):
    ...     t = torch.arange(n, dtype=torch.float64) / n * 2 * math.pi
    ...     pts = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1)
    ...     e = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    ...     return pts, e + start
    >>> p1, e1 = circle(1.0, 32, 0)
    >>> p2, e2 = circle(0.4, 16, 32)
    >>> ring = Mesh(points=torch.cat([p1, p2]), cells=torch.cat([e1, e2]))
    >>> filled = fill_interior(ring, max_cell_size=0.02)
    >>> (filled.n_manifold_dims, filled.n_spatial_dims)
    (2, 2)
    >>> bool(torch.equal(filled.points[:48], ring.points))  # exact boundary
    True
    """
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.tessellation.delaunay import delaunay_mesh_2d

    n = boundary.n_spatial_dims
    if boundary.n_manifold_dims != n - 1:
        raise ValueError(
            f"boundary must be codimension-one (n_manifold_dims == "
            f"n_spatial_dims - 1); got Mesh[{boundary.n_manifold_dims}, {n}]"
        )
    if n == 3:
        raise NotImplementedError(
            "fill_interior for n=3 (watertight surface -> tetrahedra) is "
            "not implemented yet. For approximate interior meshing today, "
            "build an SDF from the surface via "
            "physicsnemo.mesh.spatial.signed_distance_field_mesh and mesh "
            "it with physicsnemo.mesh.generate.mesh_implicit_domain (the "
            "boundary becomes O(h^2)-approximate rather than exact)."
        )
    if n != 2:
        raise NotImplementedError(
            f"fill_interior supports n_spatial_dims == 2 (n == 3 planned); got n == {n}"
        )

    device = boundary.points.device
    dtype = boundary.points.dtype
    pts64 = boundary.points.detach().to(device="cpu", dtype=torch.float64)
    edges = boundary.cells.detach().cpu()

    loops_idx = _extract_loops(pts64, edges)
    loop_polys = [pts64[idx] for idx in loops_idx]
    components = _group_components(loop_polys)

    all_points, all_cells = [], []
    all_marker, all_source = [], []
    offset = 0
    for outer, holes in components:
        comp_loops = [outer, *holes]
        engine_loops = [loop_polys[k] for k in comp_loops]
        source_ids = torch.cat([loops_idx[k] for k in comp_loops])
        points, triangles, markers, _segments = delaunay_mesh_2d(
            engine_loops,
            max_area=max_cell_size,
            min_angle_degrees=min_angle_degrees,
            smooth_iterations=smooth_iterations,
        )
        source = torch.full((points.shape[0],), -1, dtype=torch.int64)
        source[: source_ids.shape[0]] = source_ids
        all_points.append(points)
        all_cells.append(triangles + offset)
        all_marker.append(markers)
        all_source.append(source)
        offset += points.shape[0]

    points = torch.cat(all_points)
    cells = torch.cat(all_cells)
    marker = torch.cat(all_marker)
    source = torch.cat(all_source)

    # Reorder so ALL inherited input vertices lead, in input order (the
    # documented contract) — per-component concatenation interleaves each
    # component's Steiner vertices otherwise.
    inherited = torch.nonzero(source >= 0).reshape(-1)
    inherited = inherited[torch.argsort(source[inherited])]
    generated = torch.nonzero(source < 0).reshape(-1)
    order = torch.cat([inherited, generated])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.shape[0])

    return Mesh(
        points=points[order].to(device=device, dtype=dtype),
        cells=inverse[cells].to(device),
        point_data={
            "boundary_marker": marker[order].to(device),
            "source_point": source[order].to(device),
        },
    )
