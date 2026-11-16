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

"""Self-intersection detection for triangle surface meshes."""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool, Float, Int

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

# Query triangles per broad-phase slice, and candidate pairs per narrow-phase
# slice: they bound the memory of the candidate pairs and of the narrow phase.
_QUERY_CHUNK_SIZE = 16384
_PAIR_CHUNK_SIZE = 131072


def _edges(
    simplex: Float[torch.Tensor, "n k 3"],
) -> Float[torch.Tensor, "n n_edges 3"]:
    """Edge vectors of segments (``k == 2``) or triangles (``k == 3``)."""
    if simplex.shape[1] == 2:
        return simplex[:, 1:] - simplex[:, :1]
    return simplex.roll(-1, dims=1) - simplex


def _hulls_intersect(
    p: Float[torch.Tensor, "n k 3"],
    q: Float[torch.Tensor, "n 3 3"],
) -> Bool[torch.Tensor, " n"]:
    """Whether segment or triangle ``p`` and triangle ``q`` meet (closed sets).

    Separating-axis test: two convex sets are disjoint if and only if their
    projections onto some axis are disjoint. For a segment or triangle against
    a triangle, the axes that can separate are the triangle normals, the cross
    products of one edge of each, and, for coplanar inputs, each normal crossed
    with each edge (Eberly, "Intersection of Convex Objects: The Method of
    Separating Axes"). Axes that vanish (parallel edges) never separate. Uses
    only products and comparisons, so touching sets are found exactly when
    the products are.
    """
    ### Coordinates relative to a vertex of q, to keep products small
    p, q = p - q[:, :1], q - q[:, :1]
    ep, eq = _edges(p), _edges(q)
    normals = [torch.linalg.cross(eq[:, 0], eq[:, 1])[:, None]]
    if p.shape[1] == 3:
        normals.append(torch.linalg.cross(ep[:, 0], ep[:, 1])[:, None])
    normals = torch.cat(normals, dim=1)  # (n, n_normals, 3)
    edges = torch.cat([ep, eq], dim=1)  # (n, n_edges, 3)
    axes = torch.cat(
        [
            normals,
            torch.linalg.cross(ep[:, :, None], eq[:, None]).flatten(1, 2),
            torch.linalg.cross(normals[:, :, None], edges[:, None]).flatten(1, 2),
        ],
        dim=1,
    )  # (n, n_axes, 3)

    ### Disjoint if the projection intervals are disjoint along any axis
    proj_p = axes @ p.transpose(1, 2)  # (n, n_axes, k)
    proj_q = axes @ q.transpose(1, 2)  # (n, n_axes, 3)
    separated = (proj_p.amax(-1) < proj_q.amin(-1)) | (
        proj_q.amax(-1) < proj_p.amin(-1)
    )
    return ~separated.any(dim=-1)


def _pairs_intersect(
    tri_a: Float[torch.Tensor, "n 3 3"],
    tri_b: Float[torch.Tensor, "n 3 3"],
    cells_a: Int[torch.Tensor, "n 3"],
    cells_b: Int[torch.Tensor, "n 3"],
) -> Bool[torch.Tensor, " n"]:
    """Whether each triangle pair meets anywhere outside its shared vertices."""
    shared = cells_a[:, :, None] == cells_b[:, None, :]  # (n, 3, 3)
    in_b = shared.any(dim=2)  # (n, 3): vertex k of a is a vertex of b
    in_a = shared.any(dim=1)  # (n, 3): vertex k of b is a vertex of a
    n_shared = in_b.sum(dim=1)
    hit = torch.zeros_like(n_shared, dtype=torch.bool)

    ### No shared vertex: the closed triangles meet
    m = n_shared == 0
    hit[m] = _hulls_intersect(tri_a[m], tri_b[m])

    ### One shared vertex: the edge opposite it in one triangle meets the other.
    # Such a hit is not the shared vertex, and conversely the far end of any
    # longer intersection lies on one of these edges.
    m = n_shared == 1
    a, b = tri_a[m], tri_b[m]
    rows = torch.arange(len(a), device=a.device)[:, None]
    after = torch.tensor([1, 2], device=a.device)
    edge_a = a[rows, (in_b[m].int().argmax(dim=1, keepdim=True) + after) % 3]
    edge_b = b[rows, (in_a[m].int().argmax(dim=1, keepdim=True) + after) % 3]
    hit[m] = _hulls_intersect(edge_a, b) | _hulls_intersect(edge_b, a)

    ### One shared edge (u, v): the triangles overlap beyond it only when they
    # are coplanar and their apexes lie on the same side of it (a fold)
    m = n_shared == 2
    a, b = tri_a[m], tri_b[m]
    rows = torch.arange(len(a), device=a.device)
    apex_a = a[rows, (~in_b[m]).int().argmax(dim=1)]
    apex_b = b[rows, (~in_a[m]).int().argmax(dim=1)]
    uv = a[in_b[m]].view(-1, 2, 3)
    edge = uv[:, 1] - uv[:, 0]
    side_a = torch.linalg.cross(edge, apex_a - uv[:, 0])
    side_b = torch.linalg.cross(edge, apex_b - uv[:, 0])
    coplanar = (side_a * (apex_b - uv[:, 0])).sum(dim=1) == 0
    hit[m] = coplanar & ((side_a * side_b).sum(dim=1) > 0)

    return hit


def find_self_intersections(mesh: "Mesh") -> Int[torch.Tensor, "n_pairs 2"]:
    r"""Find the pairs of triangles of a surface mesh that intersect.

    A pair of triangles is reported when the two closed triangles meet anywhere
    other than in the vertices they share (by index):

    - triangles that share no vertex are reported when they touch or cross;
    - triangles that share one vertex are reported when they also meet
      somewhere else (a crossing, or an edge of one touching the other);
    - triangles that share an edge are reported when they overlap beyond it,
      which happens only when they are coplanar and fold onto the same side
      of the edge (a fold that is only nearly coplanar meets the other
      triangle on the edge alone, so it is not reported);
    - triangles with all three vertices in common (duplicate cells) are not
      reported; :func:`~physicsnemo.mesh.repair.remove_duplicate_cells` finds
      those.

    Vertices that coincide in space but have different indices are not
    shared, so the triangles around an unmerged seam are reported as
    touching. Zero-area triangles give no meaningful result; remove them
    first.

    The test is computed in float64 whatever the dtype of ``mesh.points``,
    with each pair's coordinates taken relative to one of its vertices. It
    decides exactly up to float64 rounding: a pair within rounding error of
    touching (or of being coplanar) may go either way. Candidate pairs come
    from a :class:`~physicsnemo.mesh.spatial.BVH` over the triangles'
    bounding boxes, queried in slices so memory stays bounded on large
    meshes.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh in 3D (``n_manifold_dims == 2``,
        ``n_spatial_dims == 3``).

    Returns
    -------
    torch.Tensor
        Intersecting pairs of cell indices ``(i, j)`` with ``i < j``, sorted
        lexicographically, shape ``(n_pairs, 2)``, dtype int64, on the mesh's
        device.

    Raises
    ------
    NotImplementedError
        If ``mesh`` is not a triangle mesh in 3D.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Two triangles that cross, plus one far away
    >>> points = torch.tensor(
    ...     [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
    ...      [0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [1.5, 1.5, 1.0],
    ...      [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]]
    ... )
    >>> cells = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> find_self_intersections(Mesh(points=points, cells=cells)).tolist()
    [[0, 1]]
    """
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise NotImplementedError(
            "Self-intersection detection supports only triangle meshes in 3D "
            f"(n_manifold_dims=2, n_spatial_dims=3); got {mesh.n_manifold_dims=}, "
            f"{mesh.n_spatial_dims=}."
        )
    from physicsnemo.mesh.spatial.bvh import BVH

    cells = mesh.cells
    points = mesh.points.to(torch.float64)
    triangles = points[cells]  # (n_cells, 3, 3)
    box_min = triangles.amin(dim=1)
    box_max = triangles.amax(dim=1)
    bvh = BVH.from_mesh(mesh)

    pairs: list[torch.Tensor] = [
        torch.empty((0, 2), dtype=torch.long, device=cells.device)
    ]
    for start in range(0, mesh.n_cells, _QUERY_CHUNK_SIZE):
        ### Broad phase: triangles whose bounding boxes overlap, each pair once
        stop = start + _QUERY_CHUNK_SIZE
        candidates = bvh.find_overlapping_cells(
            box_min[start:stop], box_max[start:stop]
        )
        i, j = candidates.expand_to_pairs()
        i = i + start
        keep = i < j
        candidate_pairs = torch.stack([i[keep], j[keep]], dim=1)

        ### Narrow phase: exact-geometry test of each candidate pair
        for chunk in candidate_pairs.split(_PAIR_CHUNK_SIZE):
            i, j = chunk.unbind(dim=1)
            hit = _pairs_intersect(triangles[i], triangles[j], cells[i], cells[j])
            pairs.append(chunk[hit])
    return torch.cat(pairs)
