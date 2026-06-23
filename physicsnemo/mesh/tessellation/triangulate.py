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

"""Decompose non-simplicial cells into a simplex connectivity.

The public entry point :func:`triangulate` branches on the manifold dimension,
mirroring how the rest of the mesh package handles dimension-generic ops (e.g.
``from_pyvista`` branches on ``manifold_dim``, ``compute_cell_areas`` on
``n_manifold_dims``). Only ``manifold_dim == 2`` (polygon ring -> triangles) is
implemented; higher dimensions raise ``NotImplementedError`` because a
polyhedron -> tetrahedron decomposition needs an explicit face hierarchy and a
different non-convex fallback (a non-convex polyhedron may require Steiner
points; cf. Schoenhardt's polyhedron).

The polygon path is pure PyTorch and vectorized:

- Convex polygons (the overwhelming majority of CFD surface cells) are
  fan-triangulated from vertex 0 in a single ``_ragged_arange`` pass, which is
  ``torch.compile``-traceable with no graph break.
- Non-convex polygons (rare) are ear-clipped so the unsigned per-triangle areas
  sum to the true polygon area -- a bare fan would emit overlapping triangles
  and over-count viscous / scalar-area integrals. Ear clipping is vectorized by
  grouping polygons of equal valence and clipping the whole group in lockstep.

Both paths emit exactly ``k - 2`` triangles per ``k``-gon, so per-polygon data
is broadcast to the output identically via the returned ``parent_index``
(``cell_data[parent_index]``).
"""

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.mesh.neighbors._adjacency import Adjacency
from physicsnemo.mesh.spatial._ragged import _ragged_arange
from physicsnemo.mesh.utilities._tolerances import safe_eps

#: Absolute tolerance on the (dimensionless) sine of a vertex turn below which
#: the turn is treated as straight rather than reflex, so near-collinear
#: vertices stay on the cheap convex fan path.
_REFLEX_SIN_TOL: float = 1e-6

#: ``(poly_id, prev_pos, next_pos)`` from :func:`_ring_neighbors`: for each flat
#: connectivity slot, its owning polygon and the slots of its cyclic neighbors.
_RingNeighbors = tuple[
    Int[torch.Tensor, " n_ring_positions"],
    Int[torch.Tensor, " n_ring_positions"],
    Int[torch.Tensor, " n_ring_positions"],
]


def triangulate(
    points: Float[torch.Tensor, "n_points n_spatial"],
    polygons: Adjacency,
    *,
    manifold_dim: int = 2,
    assume_convex: bool = False,
) -> tuple[
    Int[torch.Tensor, "n_simplices d_plus_one"], Int[torch.Tensor, " n_simplices"]
]:
    r"""Decompose cells into simplices, branching on manifold dimension.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates, shape :math:`(N_\text{points}, D)` with
        :math:`D \in \{2, 3\}`.
    polygons : Adjacency
        Cell-to-vertex incidence in CSR form: cell ``c`` is the vertex ring
        ``polygons.indices[polygons.offsets[c] : polygons.offsets[c + 1]]``.
        Build one from a flat VTK-style soup with
        ``Adjacency(offsets=..., indices=connectivity)``.
    manifold_dim : int, default 2
        Dimension of the cells to decompose. Only ``2`` (polygon -> triangle)
        is implemented.
    assume_convex : bool, default False
        If ``True``, skip the convexity test and ear-clip fallback and
        fan-triangulate every cell. Correct only when all cells are convex;
        this is the fully ``torch.compile``-traceable fast path.

    Returns
    -------
    cells : torch.Tensor
        Simplex connectivity, shape
        :math:`(N_\text{simplices}, \text{manifold\_dim} + 1)`, dtype int64.
    parent_index : torch.Tensor
        Source cell of each simplex, shape :math:`(N_\text{simplices},)`.
        Broadcast per-cell data to the simplices with ``data[parent_index]``.

    Raises
    ------
    NotImplementedError
        If ``manifold_dim != 2``.
    ValueError
        If any polygon has fewer than three vertices, or if ``polygons.indices``
        contains a negative index or one ``>= n_points`` (i.e. outside the valid
        :math:`[0, N_\text{points})` range; both checked off the
        ``torch.compile`` path).

    Notes
    -----
    Each polygon ring must be a *simple* polygon (no self-intersections) with no
    repeated consecutive vertices or zero-length edges. Degenerate or
    self-intersecting rings are not detected and produce undefined results. The
    non-convex (ear-clip) path additionally assumes each ring is approximately
    planar: a badly non-planar non-convex ring can project to a
    self-intersecting 2-D polygon and triangulate incorrectly (convex rings are
    unaffected by planarity).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.neighbors import Adjacency
    >>> from physicsnemo.mesh.tessellation import triangulate
    >>> points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
    ...                        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    >>> polygons = Adjacency(offsets=torch.tensor([0, 4]),  # one quad
    ...                      indices=torch.tensor([0, 1, 2, 3]))
    >>> cells, parent_index = triangulate(points, polygons)
    >>> cells.tolist()
    [[0, 1, 2], [0, 2, 3]]
    >>> parent_index.tolist()
    [0, 0]
    """
    if manifold_dim != 2:
        raise NotImplementedError(
            f"triangulate supports manifold_dim=2 (polygon -> triangle) only; got "
            f"{manifold_dim=}. Higher-dimensional decomposition (e.g. polyhedron -> "
            f"tetrahedron) needs an explicit face hierarchy and is not yet implemented."
        )
    return _triangulate_polygons(points, polygons, assume_convex=assume_convex)


# ---------------------------------------------------------------------------
# Polygon triangulation (manifold_dim == 2)
# ---------------------------------------------------------------------------


def _triangulate_polygons(
    points: Float[torch.Tensor, "n_points n_spatial"],
    polygons: Adjacency,
    *,
    assume_convex: bool,
) -> tuple[Int[torch.Tensor, "n_triangles 3"], Int[torch.Tensor, " n_triangles"]]:
    """Triangulate a 2D polygon soup: fan the convex cells, ear-clip the rest.

    Runs in two passes so the common all-convex case stays on a single
    vectorized, ``torch.compile``-traceable kernel:

    1. Fan every polygon from its first vertex (:func:`_fan`). This is correct
       for convex cells and emits ``k - 2`` contiguous triangles per ``k``-gon.
    2. Unless ``assume_convex``, classify cells from their Newell normal and
       per-vertex turn signs (:func:`_polygon_normals`, :func:`_convex_mask`)
       and overwrite only the non-convex triangle blocks with a true ear-clip
       triangulation (``reclip_nonconvex``). A bare fan over a non-convex cell
       emits overlapping triangles whose unsigned areas over-count the polygon,
       which would corrupt any area-weighted integral.

    Both passes emit the same ``k - 2`` triangles per polygon in the same order,
    so ``parent_index`` is path-independent and per-polygon data broadcasts to
    the triangles identically via ``data[parent_index]``.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates, shape ``(n_points, n_spatial)`` with ``n_spatial``
        in ``{2, 3}``. Read only on the ear-clip path; the fan is purely
        combinatorial.
    polygons : Adjacency
        Cell-to-vertex rings in CSR form.
    assume_convex : bool
        If ``True``, return the bare fan with no convexity test or ear-clip
        fallback. Correct only when every cell is convex, and the only fully
        ``torch.compile``-traceable path.

    Returns
    -------
    cells : torch.Tensor
        Triangle connectivity, shape ``(n_triangles, 3)``.
    parent_index : torch.Tensor
        Source polygon of each triangle, shape ``(n_triangles,)``.

    Raises
    ------
    ValueError
        If any polygon has fewer than three vertices, or references a vertex
        index outside ``[0, n_points)`` (negative or too large; both checked
        only off the ``torch.compile`` path, where a host sync would force a
        graph break).
    """
    counts = polygons.counts
    # Cheap structural validation, off the torch.compile path (each check is a
    # host sync that would otherwise force a graph break).
    if not torch.compiler.is_compiling() and counts.numel() > 0:
        if bool((counts < 3).any()):
            raise ValueError(
                f"Every polygon needs >= 3 vertices to triangulate; got a "
                f"polygon with {int(counts.min())} vertices."
            )
        if bool((polygons.indices < 0).any()):
            raise ValueError(
                f"polygons.indices must be non-negative, but the minimum is "
                f"{int(polygons.indices.min())}. A negative index would silently "
                f"wrap around the points array under PyTorch gather semantics and "
                f"triangulate the wrong vertex rather than raising."
            )
        if bool((polygons.indices >= points.shape[0]).any()):
            raise ValueError(
                f"polygons.indices reference vertex "
                f"{int(polygons.indices.max())}, but points has only "
                f"{points.shape[0]} vertices."
            )

    # Fan every polygon (correct for convex cells; non-convex blocks are
    # overwritten below). This is the only path under ``assume_convex``.
    cells, parent_index = _fan(polygons)
    if assume_convex:
        return cells.long(), parent_index

    points = _to_3d(points)  # normals / projection need a 3D embedding
    ring = _ring_neighbors(polygons)  # shared by normals + convexity, computed once
    normals = _polygon_normals(points, polygons, ring)
    nonconvex = ~_convex_mask(points, polygons, normals, ring)
    if bool(nonconvex.any()):  # the only host sync on the all-convex common path
        from physicsnemo.mesh.tessellation._ear_clipping import reclip_nonconvex

        cells = reclip_nonconvex(points, polygons, normals, cells, nonconvex)
    return cells.long(), parent_index


def _fan(
    polygons: Adjacency,
) -> tuple[Int[torch.Tensor, "n_triangles 3"], Int[torch.Tensor, " n_triangles"]]:
    """Fan-triangulate every polygon from its first vertex, fully vectorized.

    A ``k``-gon with vertices ``(v_0, ..., v_{k-1})`` becomes the ``k - 2``
    triangles ``(v_0, v_{j+1}, v_{j+2})`` for ``j = 0 .. k - 3``. The whole soup
    is expanded with one :func:`_ragged_arange` (no Python loop, no host sync),
    so the fan is fully ``torch.compile``-traceable.

    A polygon's triangles are emitted contiguously and in order, so
    ``parent_index`` is simply each polygon id repeated ``k - 2`` times. That
    contiguous, ordered layout is the contract ``reclip_nonconvex`` relies on to
    patch non-convex cells in place. The fan is geometrically valid only for
    convex polygons; for a non-convex cell its triangles overlap and spill
    outside the ring.

    Parameters
    ----------
    polygons : Adjacency
        Cell-to-vertex rings in CSR form.

    Returns
    -------
    cells : torch.Tensor
        Triangle connectivity, shape ``(n_triangles, 3)``.
    parent_index : torch.Tensor
        Source polygon of each triangle, shape ``(n_triangles,)``.
    """
    conn = polygons.indices
    poly_starts = polygons.offsets[:-1]

    # One entry per output triangle: ``parent_index`` is its polygon and
    # ``positions`` walks the polygon's connectivity (poly_start + fan index j).
    positions, parent_index = _ragged_arange(poly_starts, polygons.counts - 2)
    cells = torch.stack(
        [conn[poly_starts[parent_index]], conn[positions + 1], conn[positions + 2]],
        dim=-1,
    )
    return cells, parent_index


# ---------------------------------------------------------------------------
# Convexity (Newell normal + per-vertex turn sign)
# ---------------------------------------------------------------------------


def _polygon_normals(
    points: Float[torch.Tensor, "n_points 3"],
    polygons: Adjacency,
    ring: _RingNeighbors | None = None,
) -> Float[torch.Tensor, "n_polygons 3"]:
    """Per-polygon (unnormalized) Newell normal ``sum_i v_i x v_{i+1}``.

    The Newell formula sums the cross products of consecutive edges around each
    ring, so it stays robust for slightly non-planar or non-convex polygons
    rather than trusting a single corner. The result points along the vertex
    winding by the right-hand rule, and its length is twice the polygon's area:
    callers reuse it both to orient the projection plane (:func:`_convex_mask`,
    ear clipping) and to detect degenerate (near-zero-area) cells.

    Each ring is centered on its own first vertex before the cross-sum. This is
    translation-invariant yet keeps the summands small, avoiding catastrophic
    cancellation for meshes far from the origin in float32.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates embedded in 3D, shape ``(n_points, 3)``.
    polygons : Adjacency
        Cell-to-vertex rings in CSR form.
    ring : tuple of torch.Tensor, optional
        Precomputed ``(poly_id, prev_pos, next_pos)`` from
        :func:`_ring_neighbors`. Both this function and :func:`_convex_mask`
        need the same ring decomposition, so :func:`_triangulate_polygons`
        computes it once and passes it to both, avoiding a redundant pass on the
        common convex path. Defaults to ``None``, in which case it is computed
        internally so the function stays correct when called on its own.

    Returns
    -------
    torch.Tensor
        Unnormalized per-polygon normals, shape ``(n_polygons, 3)``.
    """
    if ring is None:
        ring = _ring_neighbors(polygons)
    poly_id, _, next_pos = ring
    conn = polygons.indices
    ref = points[conn[polygons.offsets[:-1]]][poly_id]  # this position's polygon v0
    edge_cross = torch.linalg.cross(points[conn] - ref, points[conn[next_pos]] - ref)

    normals = points.new_zeros((polygons.n_sources, 3))
    normals.index_add_(0, poly_id, edge_cross)
    return normals


def _convex_mask(
    points: Float[torch.Tensor, "n_points 3"],
    polygons: Adjacency,
    normals: Float[torch.Tensor, "n_polygons 3"],
    ring: _RingNeighbors | None = None,
) -> Bool[torch.Tensor, " n_polygons"]:
    """Flag polygons that are convex, and therefore safe to fan-triangulate.

    A simple polygon is convex iff it has no reflex vertex. At each vertex the
    turn from the incoming to the outgoing edge is measured as the signed sine
    ``(edge_in x edge_out) . n_hat / (|edge_in| |edge_out|)``, which is
    scale-free and lies in ``[-1, 1]``. Taken against the polygon's own normal a
    convex (left) turn is positive and a reflex (right) turn negative, so a
    vertex counts as reflex when its signed sine falls below ``-_REFLEX_SIN_TOL``;
    that small tolerance keeps near-collinear vertices on the cheap fan path. A
    polygon is convex when its reflex count is zero.

    Degenerate (near-zero-area) polygons have an ill-defined normal and are
    reported convex so they, too, stay on the fan path instead of entering ear
    clipping.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates embedded in 3D, shape ``(n_points, 3)``.
    polygons : Adjacency
        Cell-to-vertex rings in CSR form.
    normals : torch.Tensor
        Per-polygon Newell normals from :func:`_polygon_normals`, shape
        ``(n_polygons, 3)``.
    ring : tuple of torch.Tensor, optional
        Precomputed ``(poly_id, prev_pos, next_pos)`` from
        :func:`_ring_neighbors`, shared with :func:`_polygon_normals` to avoid
        recomputing it on the common convex path (see that function). Defaults
        to ``None``, in which case it is computed internally.

    Returns
    -------
    torch.Tensor
        Boolean mask, ``True`` where a polygon is convex (or degenerate), shape
        ``(n_polygons,)``.
    """
    if ring is None:
        ring = _ring_neighbors(polygons)
    poly_id, prev_pos, next_pos = ring
    conn = polygons.indices
    eps = safe_eps(points.dtype)

    v_cur = points[conn]
    edge_in = v_cur - points[conn[prev_pos]]
    edge_out = points[conn[next_pos]] - v_cur
    normal_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(eps)
    sin_turn = (torch.linalg.cross(edge_in, edge_out) * normal_hat[poly_id]).sum(-1) / (
        edge_in.norm(dim=-1) * edge_out.norm(dim=-1)
    ).clamp_min(eps)

    reflex_count = torch.zeros(
        polygons.n_sources, dtype=torch.int64, device=points.device
    )
    reflex_count.index_add_(0, poly_id, (sin_turn < -_REFLEX_SIN_TOL).to(torch.int64))
    degenerate = normals.norm(dim=-1) < eps
    return (reflex_count == 0) | degenerate


def _ring_neighbors(polygons: Adjacency) -> _RingNeighbors:
    """Cyclic previous/next neighbors of every vertex in the flattened rings.

    Walks the flat connectivity (one slot per polygon-vertex incidence) and, for
    each slot, returns the owning polygon together with the connectivity indices
    of the cyclically previous and next vertices of the *same* polygon. The
    successor of a ring's last vertex wraps to its first, via modular arithmetic
    confined to that polygon's ``[start, start + valence)`` block.

    Both :func:`_polygon_normals` and :func:`_convex_mask` use these to gather
    each vertex's two incident edges (``v - v_prev`` and ``v_next - v``) in one
    vectorized pass over the whole soup.

    Parameters
    ----------
    polygons : Adjacency
        Cell-to-vertex rings in CSR form.

    Returns
    -------
    poly_id : torch.Tensor
        Owning polygon of each connectivity slot, shape ``(n_ring_positions,)``.
    prev_pos : torch.Tensor
        Connectivity index of the cyclically previous vertex, same shape.
    next_pos : torch.Tensor
        Connectivity index of the cyclically next vertex, same shape.
    """
    poly_id, _ = polygons.expand_to_pairs()  # owning polygon of each ring position
    starts = polygons.offsets[:-1][poly_id]
    valence = polygons.counts[poly_id]
    local = torch.arange(polygons.indices.shape[0], device=poly_id.device) - starts
    prev_pos = starts + (local - 1 + valence) % valence
    next_pos = starts + (local + 1) % valence
    return poly_id, prev_pos, next_pos


def _to_3d(
    points: Float[torch.Tensor, "n_points n_spatial"],
) -> Float[torch.Tensor, "n_points 3"]:
    """Lift points onto the ``z = 0`` plane so 3D cross products are defined.

    The normal and convexity machinery is written with
    :func:`torch.linalg.cross`, which is defined only for 3-vectors, so planar
    (``n_spatial == 2``) inputs are padded with a zero z-column. Already-3D
    inputs pass through untouched. Any other dimensionality is rejected, making
    the ``n_spatial in {2, 3}`` contract of :func:`triangulate` explicit instead
    of letting, say, a 1D point cloud silently produce meaningless normals.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates, shape ``(n_points, n_spatial)``.

    Returns
    -------
    torch.Tensor
        Coordinates embedded in 3D, shape ``(n_points, 3)``.

    Raises
    ------
    ValueError
        If ``n_spatial`` is neither 2 nor 3.
    """
    if points.shape[-1] == 3:
        return points
    if points.shape[-1] != 2:
        raise ValueError(
            f"triangulate supports 2-D or 3-D point coordinates; got "
            f"{points.shape[-1]}-D points."
        )
    pad = points.new_zeros((points.shape[0], 1))
    return torch.cat([points, pad], dim=-1)
