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

"""Stitch T-junctions in triangle meshes.

A T-junction is a vertex that lies on an open edge of a neighboring triangle
without being one of its vertices, as where two patches with different
refinement meet. Splitting the triangle through the vertex makes the seam
conforming, without moving any point.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Int

from physicsnemo.mesh.utilities._tolerances import default_length_tolerance
from physicsnemo.utils._index_tuple_ops import unique_index_tuples

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _open_half_edges(
    cells: Int[torch.Tensor, "n_cells 3"],
    n_points: int,
) -> tuple[
    Int[torch.Tensor, "n_open 2"],
    Int[torch.Tensor, " n_open"],
    Int[torch.Tensor, " n_open"],
]:
    """Directed half-edges of the edges used by only one triangle.

    Returns the half-edges ``(cells[c, k], cells[c, (k + 1) % 3])`` as their
    owner cell ``c`` traverses them, with ``c`` and the local edge index ``k``.
    """
    half_edges = torch.stack([cells, cells.roll(-1, dims=1)], dim=-1).reshape(-1, 2)
    _, inverse, counts = unique_index_tuples(
        half_edges.sort(dim=1).values,
        index_bound=n_points,
        return_inverse=True,
        return_counts=True,
    )
    flat_idx = torch.nonzero(counts[inverse] == 1).squeeze(1)
    return half_edges[flat_idx], flat_idx // 3, flat_idx % 3


def stitch_t_junctions(
    mesh: "Mesh",
    tolerance: float | None = None,
) -> tuple["Mesh", dict[str, int]]:
    """Close T-junction seams by splitting triangles at the vertices on their open edges.

    For each open edge (an edge used by one triangle), finds the vertices of
    other open edges that lie on it: within ``tolerance`` of the segment and
    more than ``tolerance`` from both of its ends. The owning triangle
    ``(a, b, o)`` is split into the fan ``(a, m1, o), (m1, m2, o), ...,
    (mk, b, o)`` through those vertices, sorted along the edge. No point moves,
    the winding of every child follows its parent, and children copy their
    parent's ``cell_data``. A triangle is split at one edge per round, and
    rounds repeat until no open edge has a vertex on it.

    Parameters
    ----------
    mesh : Mesh
        Input triangle mesh (``n_manifold_dims == 2``), in 2D or 3D space.
    tolerance : float, optional
        Distance below which a vertex lies on an edge. Defaults to
        ``max(1e-6, 100 * eps)`` times the bounding-box diagonal, where ``eps``
        is the machine epsilon of the point dtype.

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (stitched_mesh, stats_dict) where stats_dict contains:

        - ``"n_cells_split"``: Number of triangle splits (a triangle split at
          two of its edges, in two rounds, counts twice).
        - ``"n_vertices_inserted"``: Number of vertices inserted into edges.
        - ``"n_rounds"``: Number of rounds that split at least one triangle.

    Raises
    ------
    ValueError
        If mesh is not a triangle mesh.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # A triangle whose long edge (0, 1) meets the two short edges of a pair
    >>> # of triangles through vertex 3, which sits on it
    >>> mesh = Mesh(
    ...     points=torch.tensor(
    ...         [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, -1.0]]
    ...     ),
    ...     cells=torch.tensor([[0, 1, 2], [0, 4, 3], [3, 4, 1]]),
    ... )
    >>> stitched, stats = stitch_t_junctions(mesh)
    >>> stats
    {'n_cells_split': 1, 'n_vertices_inserted': 1, 'n_rounds': 1}
    >>> stitched.cells.tolist()
    [[0, 4, 3], [3, 4, 1], [0, 3, 2], [3, 1, 2]]
    """
    if mesh.n_manifold_dims != 2:
        raise ValueError(
            f"T-junction stitching is only implemented for triangle meshes. "
            f"Got {mesh.n_manifold_dims=}."
        )
    if tolerance is None:
        tolerance = default_length_tolerance(mesh.points)

    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.spatial import BVH

    points = mesh.points
    cells = mesh.cells
    device = cells.device
    parent = torch.arange(mesh.n_cells, device=device)  # original cell of each cell
    n_cells_split = n_vertices_inserted = n_rounds = 0

    while True:
        ### Find vertices of open edges that lie strictly inside other open edges
        open_edges, owner, local = _open_half_edges(cells, len(points))
        if len(open_edges) == 0:
            break
        candidates = torch.unique(open_edges)
        bvh = BVH.from_mesh(Mesh(points=points, cells=open_edges))
        query, edge = bvh.find_candidate_cells(
            points[candidates], max_candidates_per_point=None, aabb_tolerance=tolerance
        ).expand_to_pairs()
        vertex = candidates[query]
        a = points[open_edges[edge, 0]]
        ab = points[open_edges[edge, 1]] - a
        length_sq = (ab * ab).sum(dim=-1)
        t = ((points[vertex] - a) * ab).sum(dim=-1) / length_sq
        distance = (points[vertex] - a - t[:, None] * ab).norm(dim=-1)
        margin = tolerance / length_sq.sqrt()
        on_edge = (
            (distance < tolerance)
            & (t > margin)
            & (t < 1 - margin)
            & (cells[owner[edge]] != vertex[:, None]).all(dim=-1)
        )
        if not on_edge.any():
            break
        edge, vertex, t = edge[on_edge], vertex[on_edge], t[on_edge]

        ### Split each owner cell at only its lowest-numbered edge with hits this round
        first_edge = torch.full((len(cells),), len(open_edges), device=device)
        first_edge.scatter_reduce_(0, owner[edge], edge, reduce="amin")
        keep = edge == first_edge[owner[edge]]
        edge, vertex, t = edge[keep], vertex[keep], t[keep]
        split_edges = torch.unique(edge)

        ### Order each split edge's vertices along it: a, m1, ..., mk, b
        seq_edge = torch.cat([split_edges, edge, split_edges])
        seq_vertex = torch.cat(
            [open_edges[split_edges, 0], vertex, open_edges[split_edges, 1]]
        )
        seq_t = torch.cat([torch.zeros_like(t[: len(split_edges)]), t])
        seq_t = torch.cat([seq_t, torch.ones_like(seq_t[: len(split_edges)])])
        order = torch.argsort(seq_t, stable=True)
        order = order[torch.argsort(seq_edge[order], stable=True)]
        seq_edge, seq_vertex = seq_edge[order], seq_vertex[order]

        ### Build the fan (m_i, m_{i+1}, o) of each split edge
        same = seq_edge[:-1] == seq_edge[1:]
        child_edge = seq_edge[:-1][same]
        opposite = cells[owner[child_edge], (local[child_edge] + 2) % 3]
        children = torch.stack(
            [seq_vertex[:-1][same], seq_vertex[1:][same], opposite], dim=1
        )
        is_split = torch.zeros(len(cells), dtype=torch.bool, device=device)
        is_split[owner[split_edges]] = True
        cells = torch.cat([cells[~is_split], children])
        parent = torch.cat([parent[~is_split], parent[owner[child_edge]]])

        n_cells_split += len(split_edges)
        n_vertices_inserted += len(edge)
        n_rounds += 1

    stats = {
        "n_cells_split": n_cells_split,
        "n_vertices_inserted": n_vertices_inserted,
        "n_rounds": n_rounds,
    }
    if n_rounds == 0:
        return mesh, stats

    stitched = Mesh(
        points=points,
        cells=cells,
        point_data=mesh.point_data.clone(),
        cell_data=mesh.cell_data[parent],
        global_data=mesh.global_data.clone(),
    )
    return stitched, stats
