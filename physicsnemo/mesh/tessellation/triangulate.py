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
        If any polygon has fewer than three vertices.

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
    points: torch.Tensor, polygons: Adjacency, *, assume_convex: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangulate a polygon soup: fan the convex cells, ear-clip the rest."""
    counts = polygons.counts
    if (
        not torch.compiler.is_compiling()
        and counts.numel() > 0
        and bool((counts < 3).any())
    ):
        raise ValueError(
            f"Every polygon needs >= 3 vertices to triangulate; got a polygon with "
            f"{int(counts.min())} vertices."
        )

    # Fan every polygon (correct for convex cells; non-convex blocks are
    # overwritten below). This is the only path under ``assume_convex``.
    cells, parent_index = _fan(polygons)
    if assume_convex:
        return cells, parent_index

    points = _to_3d(points)  # normals / projection need a 3D embedding
    normals = _polygon_normals(points, polygons)
    nonconvex = ~_convex_mask(points, polygons, normals)
    if bool(nonconvex.any()):  # the only host sync on the all-convex common path
        from physicsnemo.mesh.tessellation._ear_clipping import reclip_nonconvex

        cells = reclip_nonconvex(points, polygons, normals, cells, nonconvex)
    return cells, parent_index


def _fan(polygons: Adjacency) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized fan-from-vertex-0 triangulation of every polygon.

    Polygon ``p`` emits triangles ``(v0, v_{j+1}, v_{j+2})`` for
    ``j = 0 .. k - 3``. Built on :func:`_ragged_arange` so it is fully
    ``torch.compile``-traceable.
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
    points: Float[torch.Tensor, "n_points 3"], polygons: Adjacency
) -> Float[torch.Tensor, "n_polygons 3"]:
    """Per-polygon (unnormalized) Newell normal: ``sum_i v_i x v_{i+1}``.

    Each polygon is centered at its first vertex before the cross-sum, which is
    translation-invariant but keeps the summands small (avoids catastrophic
    cancellation for meshes far from the origin in float32).
    """
    poly_id, _, next_pos = _ring_neighbors(polygons)
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
) -> Bool[torch.Tensor, " n_polygons"]:
    """Boolean mask, ``True`` where a polygon is convex (or degenerate).

    A polygon is convex iff no vertex is reflex. The signed sine of
    each vertex turn is measured relative to the polygon normal (scale-free, in
    ``[-1, 1]``) and reflex vertices are counted per polygon. Degenerate
    (zero-area) polygons are reported convex so they stay on the cheap fan path.
    """
    poly_id, prev_pos, next_pos = _ring_neighbors(polygons)
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


def _ring_neighbors(
    polygons: Adjacency,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each connectivity position, its polygon id and cyclic prev/next positions."""
    poly_id, _ = polygons.expand_to_pairs()  # owning polygon of each ring position
    starts = polygons.offsets[:-1][poly_id]
    valence = polygons.counts[poly_id]
    local = torch.arange(polygons.indices.shape[0], device=poly_id.device) - starts
    prev_pos = starts + (local - 1 + valence) % valence
    next_pos = starts + (local + 1) % valence
    return poly_id, prev_pos, next_pos


def _to_3d(points: torch.Tensor) -> torch.Tensor:
    """Embed 2D points in 3D (z = 0) so cross products are well-defined."""
    if points.shape[-1] == 3:
        return points
    if points.shape[-1] != 2:
        raise ValueError(
            f"triangulate supports 2-D or 3-D point coordinates; got "
            f"{points.shape[-1]}-D points."
        )
    pad = points.new_zeros((points.shape[0], 1))
    return torch.cat([points, pad], dim=-1)
