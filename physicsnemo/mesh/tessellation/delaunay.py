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

r"""Planar quality mesh generation: constrained Delaunay + Ruppert refinement.

The public entry point :func:`delaunay_mesh_2d` meshes a polygonal domain given as
closed loops (one outer boundary plus optional hole loops) into a quality
triangle mesh: a conforming constrained Delaunay triangulation refined until no
triangle is smaller than a minimum-angle bound or larger than a maximum-area
bound, and optionally smoothed. :func:`polygon_interior_point` is a companion
utility that returns a point strictly inside a simple polygon.

This is a from-scratch implementation of the published algorithms:

- Incremental Delaunay insertion with a bounding super-triangle follows
  A. Bowyer, "Computing Dirichlet tessellations", *The Computer Journal* 24(2),
  1981, and D. F. Watson, "Computing the n-dimensional Delaunay tessellation
  with application to Voronoi polytopes", *The Computer Journal* 24(2), 1981
  (the "Bowyer-Watson" cavity algorithm).
- Constrained-edge recovery flips the edges crossing each input segment until
  the segment appears, per S. W. Sloan, "A fast algorithm for generating
  constrained Delaunay triangulations", *Computers & Structures* 47(3), 1993;
  local Delaunayhood is then restored by Lawson's flip criterion
  (C. L. Lawson, "Software for C1 surface interpolation", in *Mathematical
  Software III*, 1977). Constrained edges are never flipped afterwards.
- Quality refinement is J. Ruppert's Delaunay-refinement algorithm, "A Delaunay
  refinement algorithm for quality 2-dimensional mesh generation", *Journal of
  Algorithms* 18(3), 1995: encroached boundary subsegments are bisected at
  their midpoints, and skinny or oversized triangles are fixed by inserting
  their circumcenters -- deferring to subsegment splits whenever a circumcenter
  would encroach upon one.
- Exterior and hole removal is the even-odd parity flood fill over the dual
  graph used by CGAL's ``mark_domain_in_triangulation``: the recovered
  constrained segments tile the closed input loops exactly, so the parity of
  the number of constrained edges crossed on any path from the unbounded
  exterior classifies every triangle, with no interior seed points and no
  geometric predicates.
- Optional smoothing is the optimal-Delaunay-triangulation vertex update of
  L. Chen and J.-c. Xu, "Optimal Delaunay triangulations", *Journal of
  Computational Mathematics* 22(2), 2004 (popularized as "ODT smoothing" by
  L. Chen, "Mesh smoothing schemes based on optimal Delaunay triangulations",
  *13th International Meshing Roundtable*, 2004): each interior vertex moves
  to the area-weighted average of its incident triangles' circumcenters,
  gated so the local minimum angle never decreases, followed by a Lawson
  re-legalization pass -- so the refinement quality bounds survive smoothing.

Robustness model (documented rather than hidden): all geometry is computed in
float64 on coordinates normalized once into the unit box, so the ``orient2d``
and ``incircle`` determinants run at a uniform, well-understood precision.
Sign tests are strict (ties count as "not inside" / "not crossing"), and cavity
retriangulation carries an explicit star-shapedness repair, so near-degenerate
(collinear / cocircular) inputs yield a valid -- if not bitwise-unique --
triangulation instead of a corrupted one. Everything is deterministic: queues
are FIFO, iteration orders are index orders, and there is no randomness, so
identical inputs produce bitwise-identical outputs across runs.

Ruppert's termination guarantee holds for minimum-angle bounds up to
:math:`\arcsin(1/2\sqrt{2}) \approx 20.7^\circ` in theory and to roughly
:math:`33^\circ` in practice (Ruppert 1995, section 5), assuming input segments
meet at angles of at least :math:`60^\circ` -- true of the polygonalized smooth
boundaries this mesher targets, whose adjacent segments turn by only a few
degrees. ``min_angle_degrees`` is therefore capped at 33.

The implementation is generation-time CPU code: internals are pure Python +
NumPy over flat arrays, with torch tensors only at the API boundary (matching
the conventions of :func:`physicsnemo.mesh.tessellation.triangulate`).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Float, Int

#: Half-extent of the bounding super-triangle in normalized (unit-box)
#: coordinates. Large enough that the super-vertices essentially never fall
#: inside the circumcircle of a well-shaped interior triangle (so the interior
#: is genuinely Delaunay), small enough that determinant entries stay ~1e6 and
#: float64 keeps ~10 significant digits in every predicate involving them; a
#: final Lawson pass cleans up any hull-adjacent tie the compromise loses.
_SUPER_HALF_EXTENT: float = 1024.0

#: Iteration caps that convert would-be infinite loops (possible only under
#: broken invariants or inputs violating the documented assumptions) into
#: informative errors. Walk/pipe caps scale with mesh size at the call sites.
_MAX_CAVITY_REPAIR_ROUNDS: int = 64
_MAX_LEGALIZE_PASSES: int = 64

#: Relative threshold on the incircle determinant for Lawson legalization
#: flips: an edge is flipped only when the violation exceeds this fraction of
#: the determinant's own term magnitudes. Exactly-cocircular quads (ubiquitous
#: in symmetric inputs) evaluate to pure roundoff (~1e-16 relative) with a
#: sign that changes when the flip changes the operand order, so a strict
#: zero-threshold flip criterion cycles forever; this margin (~1000x roundoff,
#: yet far below any genuine violation) makes ties stable no-ops.
_INCIRCLE_REL_EPS: float = 1e-13


def delaunay_mesh_2d(
    loops: Sequence[Float[torch.Tensor, "n_i 2"] | np.ndarray],
    *,
    max_area: float | None = None,
    min_angle_degrees: float = 30.0,
    smooth_iterations: int = 0,
) -> tuple[
    Float[torch.Tensor, "n_points 2"],
    Int[torch.Tensor, "n_triangles 3"],
    Int[torch.Tensor, " n_points"],
    Int[torch.Tensor, " n_segments"],
]:
    r"""Quality-mesh a polygonal domain (outer loop plus optional holes).

    Builds the constrained Delaunay triangulation of the input loops
    (Bowyer-Watson insertion + Sloan segment recovery), removes the exterior
    and the holes by even-odd parity flood fill across the recovered
    constrained segments, then applies Ruppert's Delaunay refinement until
    every triangle has minimum angle at least ``min_angle_degrees`` and (if
    given) area at most ``max_area``, optionally followed by
    ``smooth_iterations`` passes of quality-gated ODT smoothing.

    Parameters
    ----------
    loops : sequence of torch.Tensor or numpy.ndarray
        Closed polylines of shape :math:`(N_i, 2)`, :math:`N_i \geq 3`. The
        first loop is the outer boundary; every further loop bounds a hole.
        Loops are closed implicitly (do not repeat the first vertex), must be
        simple, mutually disjoint, and free of duplicate vertices; either
        winding is accepted. Segments of distinct loops must meet at angles of
        at least ~60 degrees for the refinement termination guarantee (any
        polygonalization of smooth curves qualifies).
    max_area : float, optional
        Maximum triangle area. ``None`` (default) disables the area bound.
        For a target interior edge length :math:`h`, pass the equilateral
        area :math:`\sqrt{3}/4 \, h^2`.
    min_angle_degrees : float, default 30.0
        Minimum-angle quality bound in degrees, in :math:`[0, 33]`. ``0``
        disables the angle criterion. Values above 33 are rejected because
        Ruppert refinement is no longer guaranteed to terminate there.
    smooth_iterations : int, default 0
        Number of ODT smoothing passes applied after refinement (``0``
        disables smoothing). Each pass moves every interior (Steiner) vertex
        to the area-weighted average of its incident triangles' circumcenters
        -- the optimal-Delaunay-triangulation update of Chen and Xu -- accepts
        the move only if the smallest angle among those triangles does not
        decrease and no triangle grows beyond ``max_area``, and then restores
        Delaunayhood by Lawson legalization. Boundary vertices never move, so
        the first return value's leading rows stay bit-identical to the
        input. The quality bounds above are preserved exactly: a final
        refinement pass re-splits the rare over-bound triangle a legalization
        flip can produce, so smoothing may add a few Steiner vertices. The
        *typical* angle improves markedly (interiors approach the hexagonal
        ideal); 2 to 5 passes capture most of the benefit.

    Returns
    -------
    points : torch.Tensor
        Vertex coordinates, shape :math:`(N_\text{points}, 2)`, float64. The
        first ``sum(N_i)`` rows are the input loop vertices, bit-identical and
        in input order; refinement (Steiner) vertices follow.
    triangles : torch.Tensor
        Triangle connectivity, shape :math:`(N_\text{triangles}, 3)`, int64,
        counterclockwise (positive signed area).
    vertex_markers : torch.Tensor
        Shape :math:`(N_\text{points},)`, int64: ``1`` for vertices on the
        input boundary polyline (original loop vertices and midpoints inserted
        on segments), ``0`` for interior Steiner vertices.
    boundary_segments : torch.Tensor
        The final constrained subsegments as vertex-index pairs, shape
        :math:`(N_\text{segments}, 2)`, int64. Their union is exactly the
        input polyline: every input segment appears as a chain of these
        subsegments, and every row lies on some input segment. Each row is an
        edge of ``triangles``, directed with the domain interior on its left.

    Raises
    ------
    ValueError
        If ``loops`` is empty, a loop has fewer than 3 vertices or a
        non-``(N, 2)`` shape, coordinates are non-finite, vertices are
        duplicated, all points are coincident, a hole loop is not strictly
        inside the outer boundary loop, a hole loop lies inside another hole,
        ``max_area`` is non-positive, ``min_angle_degrees`` is outside
        :math:`[0, 33]`, ``smooth_iterations`` is negative, or input segments
        cross each other.
    RuntimeError
        If refinement exceeds its insertion budget or an internal geometric
        invariant fails -- both indicate inputs outside the documented
        assumptions (e.g. nearly-touching loops).

    Notes
    -----
    Deterministic: identical inputs give bitwise-identical outputs across
    runs. There is no randomness; insertion and refinement queues are FIFO
    and iterate in index order.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.tessellation.delaunay import delaunay_mesh_2d
    >>> square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> points, triangles, markers, segments = delaunay_mesh_2d(
    ...     [square], max_area=0.1, min_angle_degrees=30.0
    ... )
    >>> bool((markers[:4] == 1).all())  # input vertices are boundary-marked
    True
    >>> from math import degrees, acos
    >>> p = points[triangles]  # all angles >= 30 degrees
    >>> e = [p[:, (i + 1) % 3] - p[:, i] for i in range(3)]
    >>> min_cos = max(
    ...     float(
    ...         (
    ...             (e[i] * -e[(i + 2) % 3]).sum(-1)
    ...             / (e[i].norm(dim=-1) * e[(i + 2) % 3].norm(dim=-1))
    ...         ).max()
    ...     )
    ...     for i in range(3)
    ... )
    >>> degrees(acos(min_cos)) >= 30.0 - 1e-9
    True
    """
    loop_arrays = _validate_loops(loops)
    if max_area is not None and not (math.isfinite(max_area) and max_area > 0.0):
        raise ValueError(f"max_area must be finite and positive, got {max_area}")
    if not (math.isfinite(min_angle_degrees) and 0.0 <= min_angle_degrees <= 33.0):
        raise ValueError(
            f"min_angle_degrees must lie in [0, 33], got {min_angle_degrees}. "
            f"Ruppert refinement is not guaranteed to terminate above 33 degrees."
        )
    if smooth_iterations < 0:
        raise ValueError(f"smooth_iterations must be >= 0, got {smooth_iterations}")

    all_points = np.concatenate(loop_arrays, axis=0)
    seen: dict[tuple[float, float], int] = {}
    for index, (x, y) in enumerate(all_points.tolist()):
        if (x, y) in seen:
            raise ValueError(
                f"duplicate vertex at ({x}, {y}): loops must not repeat points "
                f"(vertex {index} coincides with vertex {seen[(x, y)]})"
            )
        seen[(x, y)] = index

    # Containment validation (after the duplicate check, so exactly-shared
    # vertices get the clearer error): loops must not cross -- recovery
    # checks that exactly -- so one misplaced vertex condemns its whole loop.
    for index, hole in enumerate(loop_arrays[1:], start=1):
        if not _points_in_polygon(hole, loop_arrays[0]).all():
            raise ValueError(
                f"loop {index} is not inside the outer boundary loop (loop 0); "
                f"every loop after the first must bound a hole in the domain"
            )
        for other_index, other in enumerate(loop_arrays[1:], start=1):
            if other_index != index and _points_in_polygon(hole[:1], other).any():
                raise ValueError(
                    f"hole loop {index} lies inside hole loop {other_index}; "
                    f"hole loops must be mutually disjoint"
                )

    if min_angle_degrees > 0.0:
        # Ruppert's termination guarantee assumes adjacent input segments
        # meet at >= ~60 degrees; sharper corners make refinement chase
        # itself into the corner, emitting thousands of sub-float32-area
        # triangles near the apex with no error (found in review by
        # melo-gonzo). Validate rather than assume.
        for li, arr in enumerate(loop_arrays):
            prev = np.roll(arr, 1, axis=0) - arr
            nxt = np.roll(arr, -1, axis=0) - arr
            cosang = (prev * nxt).sum(axis=1) / (
                np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1)
            )
            ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            k = int(ang.argmin())
            if ang[k] < 60.0 - 1e-9:
                raise ValueError(
                    f"loop {li} has adjacent segments meeting at "
                    f"{ang[k]:.1f} degrees (vertex {k} at "
                    f"{arr[k].tolist()}); the min-angle refinement "
                    f"guarantee requires input corners of at least ~60 "
                    f"degrees. Pass min_angle_degrees=0.0 for an "
                    f"unrefined constrained triangulation, or blunt the "
                    f"corner geometrically."
                )

    # Normalize into the unit box: every predicate then runs on ~unit-scale
    # float64 operands regardless of the physical coordinate range.
    lower = all_points.min(axis=0)
    scale = float((all_points.max(axis=0) - lower).max())
    if scale <= 0.0:
        raise ValueError("all loop vertices are collinear along an axis or coincident")
    normalized = (all_points - lower) / scale

    tri = _Triangulation()
    tri.insert_points(normalized)

    # Constrained-edge recovery: consecutive loop vertices, loop by loop.
    segments: list[tuple[int, int]] = []
    offset = 0
    for loop in loop_arrays:
        n = loop.shape[0]
        segments.extend((offset + i, offset + (i + 1) % n) for i in range(n))
        offset += n
    for a, b in segments:
        tri.recover_segment(3 + a, 3 + b)  # +3 skips the super-vertices
    tri.legalize_all()

    tri.remove_exterior_and_holes()

    max_area_normalized = None if max_area is None else max_area / (scale * scale)
    sin_min = math.sin(math.radians(min_angle_degrees))
    b_squared = None if min_angle_degrees <= 0.0 else 1.0 / (4.0 * sin_min * sin_min)
    budget = all_points.shape[0] + 100_000
    if max_area_normalized is not None:
        # The unit box bounds the domain area; x16 covers boundary grading.
        budget += int(16.0 / max_area_normalized)
    refine_requested = max_area is not None or min_angle_degrees > 0.0

    if refine_requested:
        tri.refine(max_area_normalized, b_squared, budget)

    if smooth_iterations > 0:
        tri.smooth(smooth_iterations, max_area_normalized)
        if refine_requested:
            # Legalization flips during smoothing can, rarely, merge two
            # near-bound triangles into one over-bound triangle; a final
            # refinement pass restores both bounds exactly (usually a no-op).
            tri.refine(max_area_normalized, b_squared, budget)

    points, triangles, markers, boundary_segments = tri.extract(
        n_input=all_points.shape[0], original=all_points, lower=lower, scale=scale
    )
    return (
        torch.from_numpy(points),
        torch.from_numpy(triangles),
        torch.from_numpy(markers),
        torch.from_numpy(boundary_segments),
    )


def polygon_interior_point(
    loop: Float[torch.Tensor, "n 2"] | np.ndarray,
) -> Float[torch.Tensor, " 2"]:
    """Return a point strictly inside a simple closed polygon.

    Returns the centroid of the polygon's largest *ear*: a convex vertex
    whose neighbor-to-neighbor triangle strictly contains no other polygon
    vertex, which the two-ears theorem guarantees to exist for every simple
    polygon. Ear triangles lie inside the polygon, so their centroids do too,
    and taking the largest keeps the point robustly away from the boundary
    even for polygons with near-degenerate (collinear) vertices. As a final
    guard against exactly-degenerate inputs, the centroid is verified by an
    even-odd ray-crossing test before being returned, falling through to the
    next-largest ear on failure.

    The scan is vectorized NumPy: O(n) memory and O(n^2) worst-case time,
    near-linear in practice because candidate ears are tested largest-first.
    Deterministic: identical inputs return bitwise-identical outputs.

    Parameters
    ----------
    loop : torch.Tensor or numpy.ndarray
        Polygon vertices, shape :math:`(N, 2)` with :math:`N \\geq 3`, closed
        implicitly (do not repeat the first vertex). Must be simple; either
        winding is accepted.

    Returns
    -------
    torch.Tensor
        A point strictly inside the polygon, shape :math:`(2,)`, float64.

    Raises
    ------
    ValueError
        If the loop has fewer than 3 vertices, a non-``(N, 2)`` shape,
        non-finite coordinates, duplicate consecutive vertices, zero area
        (fully degenerate), or no verifiable ear (the boundary
        self-intersects).
    """
    (loop_array,) = _validate_loops([loop])
    ring = loop_array
    doubled_area = float(
        np.sum(
            ring[:, 0] * (np.roll(ring[:, 1], -1) - np.roll(ring[:, 1], 1)),
        )
    )
    if doubled_area < 0.0:
        ring = ring[::-1]  # counterclockwise from here on
    n = ring.shape[0]

    # Classify on unit-box-normalized coordinates so the strict sign tests
    # below run at uniform float64 precision at any physical scale. Distinct
    # vertices (validated above) guarantee a positive extent.
    lower = ring.min(axis=0)
    normalized = (ring - lower) / float((ring.max(axis=0) - lower).max())
    previous = np.roll(normalized, 1, axis=0)
    following = np.roll(normalized, -1, axis=0)
    # Doubled signed ear area at each vertex; > 0 marks a convex corner.
    ear_area = (normalized[:, 0] - previous[:, 0]) * (
        following[:, 1] - normalized[:, 1]
    ) - (normalized[:, 1] - previous[:, 1]) * (following[:, 0] - normalized[:, 0])
    candidates = np.nonzero(ear_area > 0.0)[0]
    if candidates.size == 0:
        raise ValueError(
            "polygon is degenerate (zero area); cannot find an interior point"
        )
    order = candidates[np.argsort(-ear_area[candidates], kind="stable")]

    # A candidate is an ear iff no other vertex lies strictly inside its
    # triangle (vertices exactly on the triangle boundary do not block; the
    # even-odd verification below covers those exactly-degenerate touches).
    # Chunked so the (candidates x vertices) sign matrices stay ~memory-flat.
    chunk_size = max(1, 2_000_000 // n)
    for start in range(0, order.size, chunk_size):
        chunk = order[start : start + chunk_size]
        a = previous[chunk][:, None]
        b = normalized[chunk][:, None]
        c = following[chunk][:, None]
        p = normalized[None]
        s_ab = (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (
            b[..., 1] - a[..., 1]
        ) * (p[..., 0] - a[..., 0])
        s_bc = (c[..., 0] - b[..., 0]) * (p[..., 1] - b[..., 1]) - (
            c[..., 1] - b[..., 1]
        ) * (p[..., 0] - b[..., 0])
        s_ca = (a[..., 0] - c[..., 0]) * (p[..., 1] - c[..., 1]) - (
            a[..., 1] - c[..., 1]
        ) * (p[..., 0] - c[..., 0])
        blocked = ((s_ab > 0.0) & (s_bc > 0.0) & (s_ca > 0.0)).any(axis=1)
        for index in chunk[~blocked]:
            i = int(index)
            centroid = (ring[i - 1] + ring[i] + ring[(i + 1) % n]) / 3.0
            if _points_in_polygon(centroid[None], ring)[0]:
                return torch.from_numpy(centroid)
    raise ValueError("no verifiable ear found; the polygon boundary self-intersects")


def _validate_loops(
    loops: Sequence[Float[torch.Tensor, "n_i 2"] | np.ndarray],
) -> list[np.ndarray]:
    """Convert loops to float64 numpy arrays and check basic well-formedness."""
    if len(loops) == 0:
        raise ValueError("loops must contain at least the outer boundary loop")
    arrays = []
    for index, loop in enumerate(loops):
        if torch.is_tensor(loop):
            array = loop.detach().cpu().to(torch.float64).numpy()
        else:
            array = np.asarray(loop, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 3:
            raise ValueError(
                f"loop {index} must have shape (n >= 3, 2), got {tuple(array.shape)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"loop {index} contains non-finite coordinates")
        if (
            np.any(np.linalg.norm(np.diff(array, axis=0), axis=1) == 0.0)
            or np.linalg.norm(array[0] - array[-1]) == 0.0
        ):
            raise ValueError(f"loop {index} contains duplicate consecutive points")
        arrays.append(np.ascontiguousarray(array))
    return arrays


def _points_in_polygon(
    points: Float[np.ndarray, "n_query 2"],
    loop: Float[np.ndarray, "n_loop 2"],
) -> np.ndarray:
    """Vectorized even-odd (ray-crossing) point-in-polygon test.

    Casts a ray toward +x from each query point and counts crossings of the
    implicitly-closed polygon's edges, with the half-open vertical rule (an
    edge counts only when exactly one endpoint's y is <= the query's) so a
    ray through a vertex is never double-counted. Points exactly on the
    boundary classify arbitrarily, which both callers tolerate: containment
    validation treats a boundary-touching loop as misconfigured either way,
    and ear verification queries strictly-interior centroids.
    """
    x = points[:, 0:1]
    y = points[:, 1:2]
    ax = loop[None, :, 0]
    ay = loop[None, :, 1]
    bx = np.roll(loop[:, 0], -1)[None]
    by = np.roll(loop[:, 1], -1)[None]
    straddles = (ay <= y) != (by <= y)
    crosses = x < ax + (y - ay) * (bx - ax) / (by - ay + (ay == by))
    return (straddles & crosses).sum(axis=1) % 2 == 1


# ---------------------------------------------------------------------------
# Geometric predicates (float64 determinants on normalized coordinates)
# ---------------------------------------------------------------------------


def _orient2d(ax: float, ay: float, bx: float, by: float, cx: float, cy: float):
    """Twice the signed area of triangle (a, b, c); > 0 iff counterclockwise.

    Plain float64 evaluation. On unit-box coordinates its roundoff is a few
    ulps of the operand products (~1e-16), which the strict sign conventions
    of the callers absorb: ties and sub-roundoff values classify as "straight"
    / "not crossing", and cavity construction repairs any resulting
    non-star-shapedness explicitly instead of trusting the sign.
    """
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _incircle_with_magnitude(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """Incircle determinant plus the magnitude scale of its three terms.

    The magnitude (sum of the terms' absolute values) is the natural yardstick
    for a *relative* epsilon: float64 roundoff on the determinant is a few
    ulps of it regardless of how small the triangle is, so callers can
    distinguish genuine incircle violations from cocircular ties at any scale
    (see ``_INCIRCLE_REL_EPS``).
    """
    adx = ax - dx
    ady = ay - dy
    bdx = bx - dx
    bdy = by - dy
    cdx = cx - dx
    cdy = cy - dy
    term_a = (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
    term_b = (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
    term_c = (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    return (
        term_a + term_b + term_c,
        abs(term_a) + abs(term_b) + abs(term_c),
    )


# ---------------------------------------------------------------------------
# Core triangulation structure
# ---------------------------------------------------------------------------


class _Triangulation:
    """Mutable constrained-Delaunay triangulation over normalized points.

    Flat-list storage tuned for pure-Python speed (this is the hot path of
    generation-time meshing; numpy scalar indexing would be slower):

    - ``px`` / ``py``: vertex coordinates. Vertices 0-2 are the bounding
      super-triangle; real vertices follow in insertion order.
    - ``marker``: per-vertex boundary marker (1 = on the input polyline).
    - ``tv``: triangle vertices, 3 slots per triangle, counterclockwise.
      ``tv[3 t] == -2`` marks a dead (recycled) triangle.
    - ``tn``: neighbor triangle across the edge *opposite* each local vertex
      (edge opposite local ``i`` joins locals ``i+1`` and ``i+2``, directed
      counterclockwise); ``-1`` = no neighbor.
    - ``tc``: per-edge constrained flags, parallel to ``tn``. Constrained
      edges are input (sub)segments: never flipped, never crossed by cavities.
    - ``vt``: for each vertex, some living incident triangle (kept fresh by
      triangle creation; used to seed rotations and edge lookups).
    """

    def __init__(self) -> None:
        k = _SUPER_HALF_EXTENT
        self.px: list[float] = [0.5 - k, 0.5 + k, 0.5]
        self.py: list[float] = [0.5 - k, 0.5 - k, 0.5 + k]
        self.marker: list[int] = [0, 0, 0]
        self.tv: list[int] = [0, 1, 2]
        self.tn: list[int] = [-1, -1, -1]
        self.tc: list[bool] = [False, False, False]
        self.vt: list[int] = [0, 0, 0]
        self.free: list[int] = []

    # -- vertices and triangles ------------------------------------------------

    def _add_vertex(self, x: float, y: float, marker: int) -> int:
        self.px.append(x)
        self.py.append(y)
        self.marker.append(marker)
        self.vt.append(-1)
        return len(self.px) - 1

    def _new_triangle(self, a: int, b: int, c: int) -> int:
        """Allocate a live counterclockwise triangle (a, b, c)."""
        tv, tn, tc, vt = self.tv, self.tn, self.tc, self.vt
        if self.free:
            t = self.free.pop()
            base = 3 * t
            tv[base] = a
            tv[base + 1] = b
            tv[base + 2] = c
            tn[base] = tn[base + 1] = tn[base + 2] = -1
            tc[base] = tc[base + 1] = tc[base + 2] = False
        else:
            t = len(tv) // 3
            tv.extend((a, b, c))
            tn.extend((-1, -1, -1))
            tc.extend((False, False, False))
        vt[a] = t
        vt[b] = t
        vt[c] = t
        return t

    def _kill(self, t: int) -> None:
        self.tv[3 * t] = -2
        self.free.append(t)

    def _alive(self, t: int) -> bool:
        return self.tv[3 * t] >= 0

    def _living(self) -> list[int]:
        tv = self.tv
        return [t for t in range(len(tv) // 3) if tv[3 * t] >= 0]

    # -- point location ---------------------------------------------------------

    def _locate(self, x: float, y: float, start: int) -> int:
        """Orientation walk to a triangle containing (x, y).

        Valid only while the super-triangle still bounds everything (before
        exterior removal), so the walk can never fall off the mesh. Ties
        (point exactly on an edge) count as inside, so the walk stops at a
        triangle whose closure contains the point.
        """
        tv, tn, px, py = self.tv, self.tn, self.px, self.py
        t = start
        for _ in range(4 * (len(tv) // 3) + 16):
            base = 3 * t
            a = tv[base]
            b = tv[base + 1]
            c = tv[base + 2]
            axv = px[a]
            ayv = py[a]
            bxv = px[b]
            byv = py[b]
            cxv = px[c]
            cyv = py[c]
            if (bxv - axv) * (y - ayv) - (byv - ayv) * (x - axv) < 0.0:
                t = tn[base + 2]  # exit across edge (a, b)
            elif (cxv - bxv) * (y - byv) - (cyv - byv) * (x - bxv) < 0.0:
                t = tn[base]  # exit across edge (b, c)
            elif (axv - cxv) * (y - cyv) - (ayv - cyv) * (x - cxv) < 0.0:
                t = tn[base + 1]  # exit across edge (c, a)
            else:
                return t
            if t < 0:
                raise RuntimeError(
                    "point location walked off the mesh; the query point lies "
                    "outside the bounding super-triangle"
                )
        raise RuntimeError("point location did not terminate")

    # -- Bowyer-Watson insertion -------------------------------------------------

    def _grow_cavity(
        self, x: float, y: float, seed: int
    ) -> tuple[list[int], dict[int, bool]]:
        """Depth-first collection of triangles whose circumdisk contains (x, y).

        The Bowyer-Watson cavity: grown from the containing triangle across
        non-constrained edges only, with a strict incircle test (ties stay
        out, keeping the cavity conservative under roundoff).
        """
        tv, tn, tc, px, py = self.tv, self.tn, self.tc, self.px, self.py
        cavity = [seed]
        in_cavity = {seed: True}
        stack = [seed]
        while stack:
            t = stack.pop()
            base = 3 * t
            for j in (0, 1, 2):
                n = tn[base + j]
                if n < 0 or n in in_cavity or tc[base + j]:
                    continue
                nb = 3 * n
                a = tv[nb]
                b = tv[nb + 1]
                c = tv[nb + 2]
                adx = px[a] - x
                ady = py[a] - y
                bdx = px[b] - x
                bdy = py[b] - y
                cdx = px[c] - x
                cdy = py[c] - y
                if (
                    (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
                    + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
                    + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
                ) > 0.0:
                    in_cavity[n] = True
                    cavity.append(n)
                    stack.append(n)
        return cavity, in_cavity

    def _cavity_boundary(
        self, cavity: list[int], in_cavity: dict[int, bool]
    ) -> list[tuple[int, int, int, int, bool]]:
        """Directed boundary edges of a cavity.

        Each entry is ``(u, v, n, k, constrained)``: the edge runs u -> v with
        the cavity on its left, ``n`` is the surviving triangle across it
        (``-1`` if none) with back-reference slot ``k``.
        """
        tv, tn, tc = self.tv, self.tn, self.tc
        boundary = []
        for t in cavity:
            base = 3 * t
            for j in (0, 1, 2):
                n = tn[base + j]
                if n >= 0 and n in in_cavity:
                    continue
                u = tv[base + (1, 2, 0)[j]]
                v = tv[base + (2, 0, 1)[j]]
                if n >= 0:
                    nb = 3 * n
                    k = 0 if tn[nb] == t else (1 if tn[nb + 1] == t else 2)
                else:
                    k = -1
                boundary.append((u, v, n, k, tc[base + j]))
        return boundary

    def _cavity_for_point(
        self,
        x: float,
        y: float,
        seed: int,
        split_edge: tuple[int, int] | None = None,
    ):
        """Cavity + boundary for inserting (x, y), repaired to be star-shaped.

        Floating-point incircle tests can leave the cavity non-star-shaped
        (a boundary edge not strictly visible from the new point), which would
        produce inverted fan triangles; such edges are repaired by absorbing
        the triangle across them (Bowyer-Watson with explicit cavity repair).
        Repair may not cross constrained edges: hitting one means the point
        essentially lies on a segment, which the caller must resolve.

        Returns ``("ok", cavity, in_cavity, boundary)`` on success or
        ``("constrained", (u, v))`` if repair was blocked by constrained /
        hull edge ``(u, v)``. ``split_edge`` (for midpoint insertion of a
        constrained subsegment) names the directed boundary edge the point
        lies on, which is exempt from the visibility requirement.
        """
        px, py = self.px, self.py
        cavity, in_cavity = self._grow_cavity(x, y, seed)
        for _ in range(_MAX_CAVITY_REPAIR_ROUNDS):
            boundary = self._cavity_boundary(cavity, in_cavity)
            expand = []
            for u, v, n, _k, constrained in boundary:
                if split_edge is not None and u == split_edge[0] and v == split_edge[1]:
                    continue
                ux = px[u]
                uy = py[u]
                if (px[v] - ux) * (y - uy) - (py[v] - uy) * (x - ux) <= 0.0:
                    if n < 0 or constrained:
                        return ("constrained", (u, v))
                    if n not in in_cavity:
                        expand.append(n)
            if not expand:
                # A cavity that is a triangulated polygon with no interior
                # vertices satisfies len(cavity) == len(boundary) - 2 (Euler);
                # violating it would silently orphan a vertex.
                if len(cavity) != len(boundary) - 2:
                    raise RuntimeError(
                        "insertion cavity is not a simple polygon; input "
                        "geometry is too degenerate for float64 predicates"
                    )
                return ("ok", cavity, in_cavity, boundary)
            for n in expand:
                if n not in in_cavity:
                    in_cavity[n] = True
                    cavity.append(n)
        raise RuntimeError("cavity star-shapedness repair did not converge")

    def _build_fan(
        self,
        p: int,
        cavity: list[int],
        boundary: list[tuple[int, int, int, int, bool]],
        split_edge: tuple[int, int] | None = None,
    ) -> list[int]:
        """Retriangulate a cavity as the fan of its boundary edges around p.

        For a midpoint insertion (``split_edge`` given) the edge the point
        lies on emits no (degenerate) triangle; the two dangling fan edges
        (u, p) and (p, v) become the constrained child subsegments instead.
        """
        tn, tc = self.tn, self.tc
        open_edges: dict[int, tuple[int, int]] = {}
        new_triangles = []
        for u, v, n, k, constrained in boundary:
            if split_edge is not None and u == split_edge[0] and v == split_edge[1]:
                continue
            t = self._new_triangle(p, u, v)
            base = 3 * t
            tn[base] = n
            tc[base] = constrained
            if n >= 0:
                tn[3 * n + k] = t
            for x, slot in ((u, 2), (v, 1)):  # edges (p, u) and (v, p)
                partner = open_edges.pop(x, None)
                if partner is None:
                    open_edges[x] = (t, slot)
                else:
                    t2, s2 = partner
                    tn[base + slot] = t2
                    tn[3 * t2 + s2] = t
            new_triangles.append(t)
        if split_edge is None:
            if open_edges:
                raise RuntimeError("insertion cavity boundary was not a closed loop")
        else:
            for t2, s2 in open_edges.values():
                tc[3 * t2 + s2] = True  # child subsegments; tn stays -1 (hull)
        for t in cavity:
            self._kill(t)
        return new_triangles

    def insert_points(self, points: np.ndarray) -> None:
        """Bulk-insert points (rows of an ``(n, 2)`` array) in input order.

        Each point is located by an orientation walk starting from the last
        insertion's fan, which is nearly O(1) for the spatially coherent
        orders this module feeds it (loop order, refinement order).
        """
        start = 0
        for x, y in points.tolist():
            seed = self._locate(x, y, start)
            status = self._cavity_for_point(x, y, seed)
            if status[0] != "ok":
                raise RuntimeError("unconstrained insertion cannot be blocked")
            _, cavity, _in_cavity, boundary = status
            p = self._add_vertex(x, y, marker=1)
            new_triangles = self._build_fan(p, cavity, boundary)
            start = new_triangles[-1]

    # -- edge lookup and flips ---------------------------------------------------

    def _triangles_around(self, v: int) -> list[tuple[int, int]]:
        """All living triangles incident to v as ``(t, local_index_of_v)``.

        Rotates counterclockwise from ``vt[v]``; if the star is cut by a hull
        (post-removal boundary), finishes the remainder clockwise.
        """
        tv, tn = self.tv, self.tn
        t0 = self.vt[v]
        out = []
        t = t0
        while True:
            base = 3 * t
            i = 0 if tv[base] == v else (1 if tv[base + 1] == v else 2)
            out.append((t, i))
            t = tn[base + (2, 0, 1)[i]]  # cross edge (v, v_next): rotate CCW
            if t == t0:
                return out
            if t < 0:
                break
        t = t0
        while True:
            base = 3 * t
            i = 0 if tv[base] == v else (1 if tv[base + 1] == v else 2)
            t = tn[base + (1, 2, 0)[i]]  # cross edge (v_prev, v): rotate CW
            if t < 0 or t == t0:
                return out
            base = 3 * t
            i = 0 if tv[base] == v else (1 if tv[base + 1] == v else 2)
            out.append((t, i))

    def _edge_slot(self, u: int, v: int) -> tuple[int, int] | None:
        """Find edge (u, v) as ``(triangle, slot_opposite_the_edge)``.

        Searches the star of u; returns ``None`` if u and v are not currently
        adjacent (e.g. the edge has been split away). Direction-insensitive.
        """
        tv = self.tv
        for t, i in self._triangles_around(u):
            base = 3 * t
            if tv[base + (1, 2, 0)[i]] == v:
                return (t, (2, 0, 1)[i])
            if tv[base + (2, 0, 1)[i]] == v:
                return (t, (1, 2, 0)[i])
        return None

    def _flip(self, t: int, j: int) -> int:
        """Flip the edge opposite local vertex j of triangle t.

        Replaces triangles (A, B, C) and (D, C, B) sharing edge (B, C) with
        (A, B, D) and (A, D, C) sharing the other diagonal (A, D). The caller
        guarantees the quad is strictly convex and the edge unconstrained.
        Returns the neighbor triangle id (which now holds (A, D, C)).
        """
        tv, tn, tc, vt = self.tv, self.tn, self.tc, self.vt
        bt = 3 * t
        n = tn[bt + j]
        j1 = (j + 1) % 3
        j2 = (j + 2) % 3
        a = tv[bt + j]
        b = tv[bt + j1]
        c = tv[bt + j2]
        bn = 3 * n
        k = 0 if tn[bn] == t else (1 if tn[bn + 1] == t else 2)
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        d = tv[bn + k]
        nb_j1 = tn[bt + j1]
        c_j1 = tc[bt + j1]
        nb_j2 = tn[bt + j2]
        c_j2 = tc[bt + j2]
        nb_k1 = tn[bn + k1]
        c_k1 = tc[bn + k1]
        nb_k2 = tn[bn + k2]
        c_k2 = tc[bn + k2]
        tv[bt] = a
        tv[bt + 1] = b
        tv[bt + 2] = d
        tn[bt] = nb_k1
        tc[bt] = c_k1
        tn[bt + 1] = n
        tc[bt + 1] = False
        tn[bt + 2] = nb_j2
        tc[bt + 2] = c_j2
        tv[bn] = a
        tv[bn + 1] = d
        tv[bn + 2] = c
        tn[bn] = nb_k2
        tc[bn] = c_k2
        tn[bn + 1] = nb_j1
        tc[bn + 1] = c_j1
        tn[bn + 2] = t
        tc[bn + 2] = False
        if nb_k1 >= 0:
            nbase = 3 * nb_k1
            s = 0 if tn[nbase] == n else (1 if tn[nbase + 1] == n else 2)
            tn[nbase + s] = t
        if nb_j1 >= 0:
            nbase = 3 * nb_j1
            s = 0 if tn[nbase] == t else (1 if tn[nbase + 1] == t else 2)
            tn[nbase + s] = n
        vt[a] = t
        vt[b] = t
        vt[d] = t
        vt[c] = n
        return n

    # -- constrained segment recovery ---------------------------------------------

    def _mark_if_edge(self, a: int, b: int) -> bool:
        """If edge (a, b) exists, flag it constrained on both sides."""
        found = self._edge_slot(a, b)
        if found is None:
            return False
        t, j = found
        self.tc[3 * t + j] = True
        n = self.tn[3 * t + j]
        if n >= 0:
            nb = 3 * n
            k = 0 if self.tn[nb] == t else (1 if self.tn[nb + 1] == t else 2)
            self.tc[nb + k] = True
        return True

    def _first_crossing(self, a: int, b: int):
        """First obstruction on the way from vertex a toward vertex b.

        Scans the star of a for the wedge containing the ray a -> b. Returns
        ``("edge", t, j)`` where the edge opposite ``j`` in ``t`` is the first
        edge crossed by segment (a, b), or ``("vertex", w)`` if an existing
        vertex lies exactly on the segment (recovery then splits at w).
        """
        tv, px, py = self.tv, self.px, self.py
        ax = px[a]
        ay = py[a]
        bx = px[b]
        by = py[b]
        best: tuple[float, int, int] | None = None
        for t, i in self._triangles_around(a):
            base = 3 * t
            u = tv[base + (1, 2, 0)[i]]
            v = tv[base + (2, 0, 1)[i]]
            ou = (bx - ax) * (py[u] - ay) - (by - ay) * (px[u] - ax)
            ov = (bx - ax) * (py[v] - ay) - (by - ay) * (px[v] - ax)
            if ou == 0.0 and (px[u] - ax) * (bx - ax) + (py[u] - ay) * (by - ay) > 0.0:
                return ("vertex", u)
            if ov == 0.0 and (px[v] - ax) * (bx - ax) + (py[v] - ay) * (by - ay) > 0.0:
                return ("vertex", v)
            straddles = (ou >= 0.0 >= ov or ou <= 0.0 <= ov) and (
                ou != 0.0 or ov != 0.0
            )
            if straddles:
                # The sign pattern of (ou, ov) alone cannot distinguish the
                # wedge containing the forward ray from the one containing
                # the backward ray (found in review by melo-gonzo: the
                # original strict test selected backward wedges, and the
                # subsequent pipe walk exited the hull, silently dropping
                # coverage). Require a PROPER segment-segment intersection
                # of (a, b) with the opposite edge (u, v), and take the
                # crossing closest to a: the segment may properly cross a
                # non-convex link polygon several times, and the walk must
                # start at the first crossing.
                oa = (px[v] - px[u]) * (ay - py[u]) - (py[v] - py[u]) * (ax - px[u])
                ob = (px[v] - px[u]) * (by - py[u]) - (py[v] - py[u]) * (bx - px[u])
                if oa * ob < 0.0:
                    t_param = oa / (oa - ob)
                    if best is None or t_param < best[0]:
                        best = (t_param, t, i)
        if best is not None:
            return ("edge", best[1], best[2])
        raise ValueError(
            "segment recovery found no crossing toward the segment "
            "endpoint; input segments most likely cross each other (loops "
            "must be disjoint simple polylines), or the geometry is "
            "degenerate beyond float64 predicates"
        )

    def _collect_pipe(self, a: int, b: int):
        """Edges crossed by segment (a, b), walked in order from a to b.

        Returns ``("pipe", deque_of_vertex_pairs)``, or ``("vertex", w)`` when
        a vertex sits exactly on the segment (the caller splits recovery at
        it). Crossing a constrained edge means two input segments intersect,
        which is invalid input.
        """
        tv, tn, tc, px, py = self.tv, self.tn, self.tc, self.px, self.py
        first = self._first_crossing(a, b)
        if first[0] == "vertex":
            return first
        _, t, i = first
        ax = px[a]
        ay = py[a]
        bx = px[b]
        by = py[b]
        pipe: deque[tuple[int, int]] = deque()
        for _ in range(4 * (len(tv) // 3) + 16):
            base = 3 * t
            if tc[base + i]:
                raise ValueError(
                    "input segments cross each other; loops must be disjoint "
                    "simple polygons"
                )
            u = tv[base + (1, 2, 0)[i]]  # left of (a, b) by construction
            v = tv[base + (2, 0, 1)[i]]  # right of (a, b)
            pipe.append((u, v))
            n = tn[base + i]
            if n < 0:
                # Python's negative indexing would otherwise read garbage
                # adjacency off the list tails and corrupt the walk.
                raise RuntimeError(
                    "segment pipe walk exited the triangulation; input "
                    "geometry is too degenerate for float64 predicates"
                )
            nb = 3 * n
            k = 0 if tn[nb] == t else (1 if tn[nb + 1] == t else 2)
            w = tv[nb + k]
            if w == b:
                return ("pipe", pipe)
            ow = (bx - ax) * (py[w] - ay) - (by - ay) * (px[w] - ax)
            if ow == 0.0:
                return ("vertex", w)
            # Continue through the far triangle: its slots are (k: w, k+1: v,
            # k+2: u); the next crossed edge keeps w on the side it fell on.
            t = n
            i = (k + 2) % 3 if ow > 0.0 else (k + 1) % 3
        raise RuntimeError("segment pipe walk did not terminate")

    def recover_segment(self, a: int, b: int) -> None:
        """Force edge (a, b) into the triangulation and mark it constrained.

        Sloan's algorithm: repeatedly flip the edges crossing the segment,
        deferring (re-queueing) edges whose surrounding quad is not yet
        strictly convex, until the segment appears as an edge. Flipped-in
        diagonals that still cross the segment rejoin the queue. Local
        Delaunayhood of the neighborhood is restored afterwards by the global
        :meth:`legalize_all` pass that follows recovery.
        """
        tv, px, py = self.tv, self.px, self.py
        work = [(a, b)]
        while work:
            a, b = work.pop()
            if self._mark_if_edge(a, b):
                continue
            hit = self._collect_pipe(a, b)
            if hit[0] == "vertex":
                w = hit[1]
                work.append((w, b))
                work.append((a, w))
                continue
            pipe = hit[1]
            ax = px[a]
            ay = py[a]
            bx = px[b]
            by = py[b]
            for _ in range(16 * len(pipe) * (len(pipe) + 8)):
                if not pipe:
                    break
                u, v = pipe.popleft()
                found = self._edge_slot(u, v)
                if found is None:
                    continue
                t, j = found
                bt = 3 * t
                n = self.tn[bt + j]
                aa = tv[bt + j]
                nb = 3 * n
                k = 0 if self.tn[nb] == t else (1 if self.tn[nb + 1] == t else 2)
                dd = tv[nb + k]
                # Flip only strictly convex quads (u, v strictly on opposite
                # sides of the candidate diagonal (aa, dd)).
                o1 = (px[dd] - px[aa]) * (py[u] - py[aa]) - (py[dd] - py[aa]) * (
                    px[u] - px[aa]
                )
                o2 = (px[dd] - px[aa]) * (py[v] - py[aa]) - (py[dd] - py[aa]) * (
                    px[v] - px[aa]
                )
                if (o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0):
                    self._flip(t, j)
                    if aa != a and aa != b and dd != a and dd != b:
                        oa = (bx - ax) * (py[aa] - ay) - (by - ay) * (px[aa] - ax)
                        od = (bx - ax) * (py[dd] - ay) - (by - ay) * (px[dd] - ax)
                        if (oa > 0.0 > od) or (od > 0.0 > oa):
                            pipe.append((aa, dd))
                else:
                    pipe.append((u, v))  # not flippable yet; retry later
            else:
                raise RuntimeError("segment recovery flipping did not terminate")
            if not self._mark_if_edge(a, b):
                raise RuntimeError(
                    "segment recovery finished flipping but the segment is "
                    "still absent; input geometry is too degenerate"
                )

    def legalize_all(self) -> None:
        """Lawson flip passes until every unconstrained edge is locally Delaunay.

        Runs after segment recovery (whose flips are guided by crossings, not
        by the Delaunay criterion) and doubles as a safety net for hull-
        adjacent ties of the finite super-triangle. Each flip is gated on an
        incircle violation exceeding the relative roundoff margin
        ``_INCIRCLE_REL_EPS`` (exactly-cocircular quads are ties whose
        determinant sign is pure noise and would otherwise flip forever) *and*
        on strict convexity of the quad; every accepted flip then genuinely
        improves the triangulation, and the process terminates by Lawson's
        classical argument.
        """
        tv, tn, tc, px, py = self.tv, self.tn, self.tc, self.px, self.py
        for _ in range(_MAX_LEGALIZE_PASSES):
            flips = 0
            for t in range(len(tv) // 3):
                bt = 3 * t
                if tv[bt] < 0:
                    continue
                for j in (0, 1, 2):
                    n = tn[bt + j]
                    if n < t or tc[bt + j]:  # each pair once; skips hull (-1)
                        continue
                    a = tv[bt]
                    b = tv[bt + 1]
                    c = tv[bt + 2]
                    nb = 3 * n
                    k = 0 if tn[nb] == t else (1 if tn[nb + 1] == t else 2)
                    d = tv[nb + k]
                    det, magnitude = _incircle_with_magnitude(
                        px[a], py[a], px[b], py[b], px[c], py[c], px[d], py[d]
                    )
                    if det > _INCIRCLE_REL_EPS * magnitude:
                        aa = tv[bt + j]
                        u = tv[bt + (1, 2, 0)[j]]
                        v = tv[bt + (2, 0, 1)[j]]
                        if (
                            _orient2d(px[aa], py[aa], px[u], py[u], px[d], py[d]) > 0.0
                            and _orient2d(px[aa], py[aa], px[d], py[d], px[v], py[v])
                            > 0.0
                        ):
                            self._flip(t, j)
                            flips += 1
            if flips == 0:
                return
        raise RuntimeError("Delaunay legalization did not converge")

    # -- exterior and hole removal ---------------------------------------------

    def remove_exterior_and_holes(self) -> None:
        """Delete triangles outside the outer loop and inside the holes.

        Even-odd parity flood fill over the dual graph (the classification
        CGAL ships as ``mark_domain_in_triangulation``): triangles touching a
        super-vertex are the unbounded exterior at parity 0, and crossing a
        constrained edge flips parity. The recovered constrained segments
        tile the closed input loops exactly, so the parity of a triangle is
        path-independent, and -- with hole containment already validated --
        odd parity is precisely the domain interior. Purely topological: no
        seed points and no geometric predicates, so nearly-touching loops
        cannot misclassify.
        """
        tv, tn, tc = self.tv, self.tn, self.tc
        living = self._living()
        parity: dict[int, int] = {}
        queue: deque[int] = deque()
        for t in living:
            base = 3 * t
            if tv[base] < 3 or tv[base + 1] < 3 or tv[base + 2] < 3:
                parity[t] = 0
                queue.append(t)
        while queue:
            t = queue.popleft()
            p = parity[t]
            base = 3 * t
            for j in (0, 1, 2):
                n = tn[base + j]
                if n >= 0 and n not in parity:
                    parity[n] = p ^ (1 if tc[base + j] else 0)
                    queue.append(n)
        survivors = []
        for t in living:
            if parity.get(t, 0) & 1:
                survivors.append(t)
            else:
                self._kill(t)
        if not survivors:
            raise ValueError(
                "no triangles remain after exterior/hole removal; the outer "
                "loop is degenerate"
            )
        vt = self.vt
        for t in survivors:
            base = 3 * t
            for j in (0, 1, 2):
                v = tv[base + j]
                if v < 3:
                    raise ValueError(
                        "domain interior leaked to the bounding super-triangle;"
                        " the outer loop is not a closed simple polygon"
                    )
                vt[v] = t
                n = tn[base + j]
                if n >= 0 and not parity.get(n, 0) & 1:
                    tn[base + j] = -1

    # -- Ruppert refinement -------------------------------------------------------

    def _is_bad(self, t: int, max_area: float | None, b_squared: float | None) -> bool:
        """Quality test: area above ``max_area`` or minimum angle below bound.

        The angle test uses the circumradius-to-shortest-edge form of the
        bound (Ruppert 1995): with :math:`B = 1 / (2 \\sin \\theta_{\\min})`,
        a triangle has an angle below :math:`\\theta_{\\min}` iff
        :math:`R / \\ell_{\\min} > B`, evaluated here multiplicatively as
        :math:`\\ell_1^2 \\ell_2^2 \\ell_3^2 > (2 A)^2 B^2 \\ell_{\\min}^2`
        (from :math:`R = \\ell_1 \\ell_2 \\ell_3 / 4A`), which needs no
        square roots or divisions.
        """
        tv, px, py = self.tv, self.px, self.py
        base = 3 * t
        a = tv[base]
        b = tv[base + 1]
        c = tv[base + 2]
        ax = px[a]
        ay = py[a]
        bx = px[b]
        by = py[b]
        cx = px[c]
        cy = py[c]
        doubled_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if max_area is not None and doubled_area > 2.0 * max_area:
            return True
        if b_squared is None:
            return False
        lab = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
        lbc = (cx - bx) * (cx - bx) + (cy - by) * (cy - by)
        lca = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy)
        lmin = lab if lab < lbc else lbc
        if lca < lmin:
            lmin = lca
        return lab * lbc * lca > 4.0 * doubled_area * doubled_area * b_squared * lmin

    def _circumcenter(self, t: int) -> tuple[float, float]:
        tv, px, py = self.tv, self.px, self.py
        base = 3 * t
        a = tv[base]
        b = tv[base + 1]
        c = tv[base + 2]
        ax = px[a]
        ay = py[a]
        dx = px[b] - ax
        dy = py[b] - ay
        ex = px[c] - ax
        ey = py[c] - ay
        d2 = dx * dx + dy * dy
        e2 = ex * ex + ey * ey
        denominator = 2.0 * (dx * ey - dy * ex)
        if denominator <= 0.0:
            raise RuntimeError("degenerate triangle has no circumcenter")
        return (
            ax + (ey * d2 - dy * e2) / denominator,
            ay + (dx * e2 - ex * d2) / denominator,
        )

    def _walk_toward(self, t: int, qx: float, qy: float):
        """Straight-line walk from triangle t's centroid toward (qx, qy).

        Crosses exactly the edges the segment crosses, so a constrained edge
        on the way means (qx, qy) is genuinely not visible from the triangle
        -- the signal Ruppert refinement needs. Returns ``("found", triangle)``
        or ``("blocked", (triangle, slot))`` at the offending edge.
        """
        tv, tn, tc, px, py = self.tv, self.tn, self.tc, self.px, self.py
        base = 3 * t
        a = tv[base]
        b = tv[base + 1]
        c = tv[base + 2]
        sx = (px[a] + px[b] + px[c]) / 3.0
        sy = (py[a] + py[b] + py[c]) / 3.0
        current = t
        for _ in range(4 * (len(tv) // 3) + 16):
            base = 3 * current
            exit_slot = -1
            for j in (0, 1, 2):
                u = tv[base + (1, 2, 0)[j]]
                v = tv[base + (2, 0, 1)[j]]
                ux = px[u]
                uy = py[u]
                if (px[v] - ux) * (qy - uy) - (py[v] - uy) * (qx - ux) < 0.0:
                    ou = (qx - sx) * (uy - sy) - (qy - sy) * (ux - sx)
                    ov = (qx - sx) * (py[v] - sy) - (qy - sy) * (px[v] - sx)
                    if ou >= 0.0 >= ov:
                        exit_slot = j
                        break
            if exit_slot < 0:
                return ("found", current)
            if tc[base + exit_slot] or tn[base + exit_slot] < 0:
                return ("blocked", (current, exit_slot))
            current = tn[base + exit_slot]
        raise RuntimeError("circumcenter walk did not terminate")

    def _split_segment(self, t: int, j: int, seg_queue, tri_queue, budget: int):
        """Bisect the constrained subsegment opposite slot j of triangle t.

        The midpoint is a boundary vertex (marker 1); the two child
        subsegments inherit the constraint and are re-queued for encroachment
        checks, as are any other constrained edges whose apex changed.
        """
        tv = self.tv
        base = 3 * t
        u = tv[base + (1, 2, 0)[j]]
        v = tv[base + (2, 0, 1)[j]]
        mx = 0.5 * (self.px[u] + self.px[v])
        my = 0.5 * (self.py[u] + self.py[v])
        status = self._cavity_for_point(mx, my, t, split_edge=(u, v))
        if status[0] != "ok":
            raise RuntimeError(
                "subsegment midpoint insertion was blocked by another "
                "constrained edge; boundary loops nearly touch"
            )
        _, cavity, _in_cavity, boundary = status
        p = self._add_vertex(mx, my, marker=1)
        if len(self.px) > budget:
            raise RuntimeError(
                "Delaunay refinement exceeded its vertex budget; the input "
                "violates the >= 60 degree segment-angle assumption"
            )
        new_triangles = self._build_fan(p, cavity, boundary, split_edge=(u, v))
        seg_queue.append((u, p, False))
        seg_queue.append((p, v, False))
        for bu, bv, _n, _k, constrained in boundary:
            if constrained and not (bu == u and bv == v):
                seg_queue.append((bu, bv, False))
        for tn_ in new_triangles:
            nb = 3 * tn_
            tri_queue.append((tn_, tv[nb], tv[nb + 1], tv[nb + 2]))

    def refine(
        self, max_area: float | None, b_squared: float | None, budget: int
    ) -> None:
        """Ruppert Delaunay refinement (Ruppert 1995, section 3).

        Two FIFO queues, processed deterministically with segments taking
        priority over triangles:

        1. A constrained subsegment is *encroached* when a vertex lies
           strictly inside its diametral circle; encroached subsegments are
           split at their midpoints. Testing only the apexes of the adjacent
           triangles suffices in a Delaunay triangulation.
        2. A bad triangle (angle/area bound violated) asks for its
           circumcenter. If the straight path to the circumcenter is blocked
           by a subsegment, or the circumcenter would encroach upon
           subsegments on its cavity boundary, those subsegments are split
           instead and the triangle is re-queued; otherwise the circumcenter
           is inserted (marker 0).

        Entries are validated lazily at pop time (the mesh may have changed
        since they were queued), so stale entries are simply skipped.
        """
        tv, tc, px, py = self.tv, self.tc, self.px, self.py
        seg_queue: deque[tuple[int, int, bool]] = deque()
        tri_queue: deque[tuple[int, int, int, int]] = deque()
        for t in self._living():
            base = 3 * t
            for j in (0, 1, 2):
                if tc[base + j]:
                    seg_queue.append(
                        (tv[base + (1, 2, 0)[j]], tv[base + (2, 0, 1)[j]], False)
                    )
            tri_queue.append((t, tv[base], tv[base + 1], tv[base + 2]))

        while seg_queue or tri_queue:
            if seg_queue:
                u, v, forced = seg_queue.popleft()
                found = self._edge_slot(u, v)
                if found is None:
                    continue  # already split away
                t, j = found
                base = 3 * t
                if not tc[base + j]:
                    continue
                if not forced:
                    w = tv[base + j]  # apex of the only living side
                    if (px[u] - px[w]) * (px[v] - px[w]) + (py[u] - py[w]) * (
                        py[v] - py[w]
                    ) >= 0.0:
                        continue  # apex on/outside the diametral circle
                self._split_segment(t, j, seg_queue, tri_queue, budget)
                continue

            entry = tri_queue.popleft()
            t, a, b, c = entry
            base = 3 * t
            if tv[base] != a or tv[base + 1] != b or tv[base + 2] != c:
                continue  # stale: triangle was retriangulated
            if not self._is_bad(t, max_area, b_squared):
                continue
            ccx, ccy = self._circumcenter(t)
            walked = self._walk_toward(t, ccx, ccy)
            if walked[0] == "blocked":
                tb, jb = walked[1]
                nb = 3 * tb
                seg_queue.append((tv[nb + (1, 2, 0)[jb]], tv[nb + (2, 0, 1)[jb]], True))
                tri_queue.append(entry)
                continue
            container = walked[1]
            cb = 3 * container
            if any(px[tv[cb + j]] == ccx and py[tv[cb + j]] == ccy for j in (0, 1, 2)):
                continue  # circumcenter coincides with an existing vertex
            status = self._cavity_for_point(ccx, ccy, container)
            if status[0] == "constrained":
                su, sv = status[1]
                seg_queue.append((su, sv, True))
                tri_queue.append(entry)
                continue
            _, cavity, _in_cavity, boundary = status
            encroached = [
                (bu, bv)
                for bu, bv, _n, _k, constrained in boundary
                if constrained
                and (px[bu] - ccx) * (px[bv] - ccx) + (py[bu] - ccy) * (py[bv] - ccy)
                < 0.0
            ]
            if encroached:
                for bu, bv in encroached:
                    seg_queue.append((bu, bv, True))
                tri_queue.append(entry)
                continue
            p = self._add_vertex(ccx, ccy, marker=0)
            if len(px) > budget:
                raise RuntimeError(
                    "Delaunay refinement exceeded its vertex budget; the "
                    "input violates the documented angle assumptions"
                )
            new_triangles = self._build_fan(p, cavity, boundary)
            for bu, bv, _n, _k, constrained in boundary:
                if constrained:
                    seg_queue.append((bu, bv, False))
            for tn_ in new_triangles:
                nb = 3 * tn_
                tri_queue.append((tn_, tv[nb], tv[nb + 1], tv[nb + 2]))

    # -- ODT smoothing --------------------------------------------------------------

    def _star_min_quality(self, star: list[tuple[int, int]], x: float, y: float):
        """Minimum squared-sine over all angles of v's star with v at (x, y).

        For a counterclockwise triangle with doubled area :math:`2A` and
        squared edge lengths :math:`l_1^2, l_2^2, l_3^2`, the squared sine of
        the angle between edges 1 and 2 is :math:`(2A)^2 / (l_1^2 l_2^2)`, so
        the smallest angle's squared sine is :math:`(2A)^2` over the largest
        pairwise product. A non-positive orientation scores ``-inf``.
        ``sin^2`` cannot tell an angle from its supplement, but a near-180
        corner forces a near-0 corner in the same triangle, so the minimum
        still detects every degeneracy. Angles at all three corners count --
        the star's outer-ring angles at fixed vertices matter just as much
        as the angles at v.
        """
        tv, px, py = self.tv, self.px, self.py
        worst = math.inf
        for t, i in star:
            base = 3 * t
            u = tv[base + (1, 2, 0)[i]]
            w = tv[base + (2, 0, 1)[i]]
            ux = px[u]
            uy = py[u]
            wx = px[w]
            wy = py[w]
            doubled = (ux - x) * (wy - y) - (uy - y) * (wx - x)
            if doubled <= 0.0:
                return -math.inf
            l_vu = (ux - x) * (ux - x) + (uy - y) * (uy - y)
            l_uw = (wx - ux) * (wx - ux) + (wy - uy) * (wy - uy)
            l_wv = (x - wx) * (x - wx) + (y - wy) * (y - wy)
            d = l_vu * l_uw
            if l_vu * l_wv > d:
                d = l_vu * l_wv
            if l_uw * l_wv > d:
                d = l_uw * l_wv
            quality = doubled * doubled / d
            if quality < worst:
                worst = quality
        return worst

    def _star_max_doubled_area(self, star: list[tuple[int, int]], x: float, y: float):
        """Largest doubled triangle area over v's star with v at (x, y)."""
        tv, px, py = self.tv, self.px, self.py
        largest = 0.0
        for t, i in star:
            base = 3 * t
            u = tv[base + (1, 2, 0)[i]]
            w = tv[base + (2, 0, 1)[i]]
            doubled = (px[u] - x) * (py[w] - y) - (py[u] - y) * (px[w] - x)
            if doubled > largest:
                largest = doubled
        return largest

    def smooth(self, iterations: int, max_area: float | None) -> None:
        """Quality-gated ODT smoothing (Chen and Xu 2004; module docstring).

        Each pass sweeps the interior (marker-0) vertices in index order,
        Gauss-Seidel style: the proposed position is the area-weighted
        average of the incident triangles' circumcenters -- the optimal-
        Delaunay-triangulation update -- and the move is accepted only when
        the smallest angle among those triangles does not decrease and (when
        ``max_area`` is given) no incident triangle grows beyond it, so
        triangles never invert and the refinement's bounds survive. A Lawson
        legalization pass then restores Delaunayhood (Delaunay flips never
        reduce a quad's smallest angle either) before the next sweep. Stops
        early once a sweep accepts no move. Deterministic: index-order
        sweeps, no randomness.
        """
        px, py, marker = self.px, self.py, self.marker
        tv = self.tv
        for _ in range(iterations):
            moved = 0
            for v in range(3, len(px)):
                if marker[v] != 0:
                    continue
                star = self._triangles_around(v)
                weight_sum = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                for t, _i in star:
                    base = 3 * t
                    a = tv[base]
                    b = tv[base + 1]
                    c = tv[base + 2]
                    doubled = (px[b] - px[a]) * (py[c] - py[a]) - (py[b] - py[a]) * (
                        px[c] - px[a]
                    )
                    center_x, center_y = self._circumcenter(t)
                    weight_sum += doubled
                    weighted_x += doubled * center_x
                    weighted_y += doubled * center_y
                if weight_sum <= 0.0:
                    continue
                new_x = weighted_x / weight_sum
                new_y = weighted_y / weight_sum
                if new_x == px[v] and new_y == py[v]:
                    continue
                if self._star_min_quality(star, new_x, new_y) < (
                    self._star_min_quality(star, px[v], py[v])
                ):
                    continue
                if max_area is not None:
                    # Keep every incident triangle within the area bound; a
                    # star already over the bound (possible transiently after
                    # a legalization flip) may still shrink toward it.
                    new_max = self._star_max_doubled_area(star, new_x, new_y)
                    if new_max > 2.0 * max_area and new_max > (
                        self._star_max_doubled_area(star, px[v], py[v])
                    ):
                        continue
                px[v] = new_x
                py[v] = new_y
                moved += 1
            if moved == 0:
                return
            self.legalize_all()

    # -- output -------------------------------------------------------------------

    def extract(
        self, n_input: int, original: np.ndarray, lower: np.ndarray, scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the final arrays in un-normalized coordinates.

        Input vertices are returned bit-identically from ``original``; Steiner
        vertices are mapped back through the affine normalization. Triangles
        and constrained subsegments are emitted in triangle-id order.
        """
        tv, tc = self.tv, self.tc
        n_vertices = len(self.px) - 3
        points = np.empty((n_vertices, 2), dtype=np.float64)
        points[:n_input] = original
        added = np.stack(
            (
                np.asarray(self.px[3 + n_input :], dtype=np.float64),
                np.asarray(self.py[3 + n_input :], dtype=np.float64),
            ),
            axis=1,
        )
        points[n_input:] = added * scale + lower
        markers = np.asarray(self.marker[3:], dtype=np.int64)
        triangles = []
        boundary_segments = []
        for t in self._living():
            base = 3 * t
            triangles.append((tv[base] - 3, tv[base + 1] - 3, tv[base + 2] - 3))
            boundary_segments.extend(
                (tv[base + (1, 2, 0)[j]] - 3, tv[base + (2, 0, 1)[j]] - 3)
                for j in (0, 1, 2)
                if tc[base + j]
            )
        return (
            points,
            np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
            markers,
            np.asarray(boundary_segments, dtype=np.int64).reshape(-1, 2),
        )


def _delaunay_triangulation(points: np.ndarray) -> np.ndarray:
    """Unconstrained Delaunay triangulation of a point set (testing hook).

    Normalizes, runs Bowyer-Watson insertion plus a final legalization pass,
    strips the super-triangle, and returns the ``(n_triangles, 3)`` int64
    connectivity in input-point indices. Exposed for the unit tests that
    verify the empty-circumcircle property directly.
    """
    points = np.asarray(points, dtype=np.float64)
    lower = points.min(axis=0)
    scale = float((points.max(axis=0) - lower).max())
    if scale <= 0.0:
        raise ValueError("points are coincident or collinear along an axis")
    tri = _Triangulation()
    tri.insert_points((points - lower) / scale)
    tri.legalize_all()
    tv = tri.tv
    triangles = []
    for t in tri._living():
        base = 3 * t
        a, b, c = tv[base], tv[base + 1], tv[base + 2]
        if a >= 3 and b >= 3 and c >= 3:
            triangles.append((a - 3, b - 3, c - 3))
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
