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
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

__all__ = ["fill_interior"]


def _extract_loops(
    edges: Int[torch.Tensor, "n_edges 2"], n_points: int
) -> list[Int[torch.Tensor, " n_i"]]:
    """Ordered vertex-index loops from a closed 1-manifold edge mesh.

    Requires every referenced vertex to have degree exactly 2. Returns a
    list of 1D int64 index tensors (one per loop, arbitrary orientation).
    Pure-Python adjacency (built once from a single ``tolist``) rather than
    per-edge tensor indexing: ~1 us/edge instead of ~33 us/edge.
    """
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
    nbr: list[list[int]] = [[] for _ in range(n_points)]
    for a, b in edges.tolist():
        nbr[a].append(b)
        nbr[b].append(a)
    visited = [not bool(u) for u in used.tolist()]
    loops = []
    for start in range(n_points):
        if visited[start]:
            continue
        loop = [start]
        visited[start] = True
        prev, cur = start, nbr[start][0]
        while cur != start:
            loop.append(cur)
            visited[cur] = True
            a, b = nbr[cur]
            prev, cur = cur, (b if a == prev else a)
        if len(loop) < 3:
            raise ValueError(
                f"boundary contains a degenerate loop of {len(loop)} "
                f"vertices (needs >= 3)"
            )
        loops.append(torch.tensor(loop, dtype=torch.int64))
    return loops


def _group_components(
    loop_polys: list[Float[torch.Tensor, "n_i 2"]],
) -> list[tuple[int, list[int]]]:
    """Group loops into (outer, [holes]) components by containment depth.

    Even nesting depth = a component's outer boundary; its holes are the
    loops one level deeper that it directly contains. Islands inside holes
    start new components (any nesting depth is supported). Containment
    between disjoint curves is probed with the loops' own vertices -- ALL
    of them, via one vectorized ``_points_in_polygon`` call per loop --
    so touching or crossing loops surface as a mixed-membership error
    rather than an arbitrary classification.

    Crossing loops that end up in *different* components (the engine's own
    crossing check only sees one component at a time) are detected by
    pairwise segment intersection, bounding-box prefiltered.
    """
    from physicsnemo.mesh.tessellation.delaunay import _points_in_polygon

    n = len(loop_polys)
    polys_np = [poly.numpy() for poly in loop_polys]
    # Containment between DISJOINT curves is decided by any single boundary
    # point, so probe with the loops' own vertices -- testing all of them
    # both makes the answer robust and detects touching/crossing loops
    # (mixed membership) instead of classifying them arbitrarily. (An
    # interior point of the polygon is the WRONG probe here: an outer
    # square's interior point can fall inside its own hole, misclassifying
    # the outer loop as nested -- found in review by melo-gonzo.)
    contains = [[False] * n for _ in range(n)]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if not others:
            break
        stacked = __import__("numpy").concatenate([polys_np[j] for j in others])
        inside = _points_in_polygon(stacked, polys_np[i])
        off = 0
        for j in others:
            m = polys_np[j].shape[0]
            block = inside[off : off + m]
            off += m
            if block.all():
                contains[i][j] = True
            elif block.any():
                raise ValueError(
                    f"boundary loops {i} and {j} touch or cross (some of "
                    f"loop {j}'s vertices are inside loop {i} and some are "
                    f"outside); loops must be disjoint simple polylines"
                )
    depth = [sum(contains[i][j] for i in range(n)) for j in range(n)]
    components = []
    for j in range(n):
        if depth[j] % 2 == 0:
            holes = [k for k in range(n) if depth[k] == depth[j] + 1 and contains[j][k]]
            components.append((j, holes))

    comp_of = {}
    for ci, (outer, holes) in enumerate(components):
        comp_of[outer] = ci
        for k in holes:
            comp_of[k] = ci
    _check_no_cross_component_crossings(loop_polys, comp_of)
    return components


def _segments_intersect(
    a: Float[torch.Tensor, "n_a 2 2"], b: Float[torch.Tensor, "n_b 2 2"]
) -> bool:
    """True if any segment of ``a`` properly intersects any segment of ``b``."""

    def orient(p, q, r):  # sign of the (q-p) x (r-p) cross product
        return torch.sign(
            (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
            - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
        )

    p1, p2 = a[:, None, 0], a[:, None, 1]
    q1, q2 = b[None, :, 0], b[None, :, 1]
    hit = (orient(p1, p2, q1) * orient(p1, p2, q2) < 0) & (
        orient(q1, q2, p1) * orient(q1, q2, p2) < 0
    )
    return bool(hit.any())


def _check_no_cross_component_crossings(
    loop_polys: list[Float[torch.Tensor, "n_i 2"]],
    comp_of: dict[int, int],
) -> None:
    n = len(loop_polys)
    boxes = [(poly.min(dim=0).values, poly.max(dim=0).values) for poly in loop_polys]
    segs = [
        torch.stack([poly, torch.roll(poly, -1, dims=0)], dim=1) for poly in loop_polys
    ]
    for i in range(n):
        for j in range(i + 1, n):
            if comp_of.get(i) == comp_of.get(j):
                continue  # same component: the engine validates these
            if bool((boxes[i][1] < boxes[j][0]).any()) or bool(
                (boxes[j][1] < boxes[i][0]).any()
            ):
                continue  # disjoint bounding boxes cannot cross
            if _segments_intersect(segs[i], segs[j]):
                raise ValueError(
                    f"boundary loops {i} and {j} cross each other; loops "
                    f"must be disjoint simple polylines"
                )


def fill_interior(
    boundary: "Mesh",
    *,
    max_cell_size: float | None = None,
    min_angle_degrees: float = 30.0,
    smooth_iterations: int = 0,
    provenance: bool = False,
) -> "Mesh":
    r"""Fill the interior of a closed boundary mesh with quality simplices.

    Dimension-generic contract: given a closed codimension-one boundary
    ``Mesh[n-1, n]``, produce a volume ``Mesh[n, n]`` of the enclosed
    interior such that

    - every input vertex *referenced by an edge* appears **bit-identically**
      in the output (the leading rows of ``points``, in input order);
    - boundary facets are never moved off the input geometry — refinement
      may *subdivide* them, but the union of output boundary facets equals
      the input boundary exactly;
    - interior (Steiner) vertices are inserted to meet the quality bounds.

    Currently implemented for ``n = 2`` (edge loops -> triangles), where
    every output triangle is **guaranteed** a minimum angle of
    ``min_angle_degrees`` (Ruppert refinement; deterministic, bitwise
    reproducible). ``n = 3`` (watertight surface -> tetrahedra) raises
    :class:`NotImplementedError` pending exact 3D boundary recovery.

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
        Quality-gated optimal-Delaunay-triangulation (ODT) smoothing
        passes after refinement (each interior vertex moves to the
        area-weighted average of its incident triangles' circumcenters,
        Chen & Xu 2004); boundary
        vertices never move, and the quality bounds are preserved.
    provenance : bool, default False
        When ``True``, attach provenance fields to the output's
        ``point_data`` (opt-in, since these claim keys in a user-owned
        namespace):

        - ``"boundary_marker"`` (int64): 1 for vertices on the input
          boundary (input vertices and refinement midpoints inserted on
          it), 0 for interior Steiner vertices.
        - ``"source_point"`` (int64): for vertices inherited from the
          input, the index of the originating input vertex; -1 for
          generated vertices. Use it to propagate input ``point_data``
          onto the output.

    Returns
    -------
    Mesh
        Volume mesh on the input's device and dtype, positively oriented,
        with an empty ``point_data`` unless ``provenance=True``.

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
            "not implemented yet; exact 3D boundary recovery is planned."
        )
    if n != 2:
        raise NotImplementedError(
            f"fill_interior supports n_spatial_dims == 2 (n == 3 planned); got n == {n}"
        )

    device = boundary.points.device
    dtype = boundary.points.dtype
    pts64 = boundary.points.detach().to(device="cpu", dtype=torch.float64)
    edges = boundary.cells.detach().cpu()

    if edges.shape[0] == 0:
        raise ValueError(
            "boundary contains no edges; nothing to fill. Pass a Mesh "
            "whose cells are the closed boundary loops."
        )
    loops_idx = _extract_loops(edges, pts64.shape[0])
    loop_polys = [pts64[idx] for idx in loops_idx]
    components = _group_components(loop_polys)
    if not components:
        raise ValueError(
            "no boundary loop sits at even containment depth, so there is "
            "no domain to fill; this typically means coincident duplicate "
            "loops or otherwise invalid nesting"
        )

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

    point_data = None
    if provenance:
        point_data = {
            "boundary_marker": marker[order].to(device),
            "source_point": source[order].to(device),
        }
    return Mesh(
        points=points[order].to(device=device, dtype=dtype),
        cells=inverse[cells].to(device),
        point_data=point_data,
    )
