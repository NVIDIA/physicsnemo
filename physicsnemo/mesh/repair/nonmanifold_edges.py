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

"""Split non-manifold edges of triangle meshes into manifold wedges.

An edge used by three or more triangles is split by pairing its triangles
into manifold wedges, one copy of the edge per wedge, and optionally by
dropping the fins and pillows left open along the split edges.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Int

from physicsnemo.mesh.utilities._duplicate_detection import (
    vectorized_connected_components,
)
from physicsnemo.utils._index_tuple_ops import unique_index_tuples

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _corner_links(
    cells: Int[torch.Tensor, "n_cells 3"],
    first: Int[torch.Tensor, " n_pairs"],
    second: Int[torch.Tensor, " n_pairs"],
) -> Int[torch.Tensor, "n_links 2"]:
    """Link the corners at either end of two half-edges of one edge.

    Half-edge ``3 * c + j`` runs from corner ``j`` of cell ``c`` to corner
    ``(j + 1) % 3``; corner ``j`` of cell ``c`` has index ``3 * c + j``.
    Corners are linked where they hold the same vertex.
    """
    flat = cells.reshape(-1)
    first_head = first - first % 3 + (first + 1) % 3
    second_head = second - second % 3 + (second + 1) % 3
    same = flat[first] == flat[second]
    return torch.stack(
        [
            torch.cat([first, first_head]),
            torch.cat(
                [
                    torch.where(same, second, second_head),
                    torch.where(same, second_head, second),
                ]
            ),
        ],
        dim=1,
    )


def split_nonmanifold_edges(
    mesh: "Mesh",
    drop_fins: bool = True,
) -> tuple["Mesh", dict[str, int]]:
    """Split each edge used by three or more triangles into manifold wedges.

    The triangles around such an edge are sorted by angle around it, and each
    is paired with its neighbor in that order that traverses the edge the
    other way, where the two enclose the inside of an outward-oriented
    surface. Each pair, and each triangle left unpaired, gets its own copy of
    the edge: an end vertex is duplicated for every part of its star that the
    split disconnects, as :func:`~physicsnemo.mesh.generate._repair.split_pinched_vertices`
    does for pinched vertices. Two triangles at equal angles never pair with
    each other, so a zero-volume pillow opens rather than closes.

    The input should be consistently oriented: two triangles that wind the
    same way around an edge never pair.

    Parameters
    ----------
    mesh : Mesh
        Input triangle mesh in 3D.
    drop_fins : bool
        If True, also remove every facet-connected part whose triangles at
        split edges are all left unpaired: fins of any number of triangles,
        and pillows, which the split leaves open along the edge copies. Their
        points stay, as with
        :func:`~physicsnemo.mesh.repair.remove_degenerate_cells`.

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (split_mesh, stats_dict) where stats_dict contains:

        - ``"n_edges_split"``: Number of edges used by three or more triangles
          whose kept triangles now use two or more copies of the edge.
        - ``"n_vertices_duplicated"``: Number of points added as copies.
        - ``"n_cells_dropped"``: Number of cells removed as fins.

    Raises
    ------
    ValueError
        If mesh is not a triangle mesh in 3D.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> from physicsnemo.mesh.primitives.surfaces import cube_surface
    >>> cube = cube_surface.load()
    >>> # A fin on the diagonal (1, 7) of the top face
    >>> points = torch.cat([cube.points, torch.tensor([[0.0, 0.0, 1.0]])])
    >>> cells = torch.cat([cube.cells, torch.tensor([[1, 7, 8]])])
    >>> fixed, stats = split_nonmanifold_edges(Mesh(points=points, cells=cells))
    >>> stats
    {'n_edges_split': 0, 'n_vertices_duplicated': 0, 'n_cells_dropped': 1}
    >>> torch.equal(fixed.cells, cube.cells)
    True
    """
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise ValueError(
            f"Non-manifold edge splitting needs a triangle mesh in 3D. "
            f"Got {mesh.n_manifold_dims=} and {mesh.n_spatial_dims=}."
        )
    n_points, n_cells = mesh.n_points, mesh.n_cells
    cells, points = mesh.cells, mesh.points
    device = cells.device
    no_change = {"n_edges_split": 0, "n_vertices_duplicated": 0, "n_cells_dropped": 0}
    if n_cells == 0:
        return mesh, no_change

    ### Half-edges and the edges they traverse
    tail = cells.reshape(-1)
    head = cells.roll(-1, dims=1).reshape(-1)
    apex = cells.roll(-2, dims=1).reshape(-1)
    lo, hi = torch.minimum(tail, head), torch.maximum(tail, head)
    _, edge_ids, counts = unique_index_tuples(
        torch.stack([lo, hi], dim=1),
        index_bound=n_points,
        return_inverse=True,
        return_counts=True,
    )
    if not bool((counts > 2).any()):
        return mesh, no_change

    ### Angle of each half-edge's triangle around its edge, from the edge's first
    axis = points[hi] - points[lo]
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(axis.dtype).tiny
    )
    w = points[apex] - points[lo]
    w = w - (w * axis).sum(dim=-1, keepdim=True) * axis
    half_edges = torch.arange(3 * n_cells, device=device)
    first = torch.full_like(counts, 3 * n_cells).scatter_reduce_(
        0, edge_ids, half_edges, reduce="amin"
    )
    u = w[first[edge_ids]]
    angle = torch.atan2(
        (torch.linalg.cross(u, w) * axis).sum(dim=-1), (u * w).sum(dim=-1)
    )

    ### Sort by edge, then angle, then forward (lo -> hi) half-edges first
    forward = tail == lo
    order = torch.argsort((~forward).to(torch.int8), stable=True)
    order = order[torch.argsort(angle[order], stable=True)]
    order = order[torch.argsort(edge_ids[order], stable=True)]

    ### Pair each half-edge with the next one around its edge
    edge = edge_ids[order]
    start = (torch.cumsum(counts, dim=0) - counts)[edge]
    nxt = order[start + (half_edges - start + 1) % counts[edge]]
    is_nonmanifold = counts[edge] > 2
    is_manifold = (counts[edge] == 2) & (half_edges == start)
    # Backward then forward, counterclockwise: the inside of an outward surface
    is_wedge = is_nonmanifold & ~forward[order] & forward[nxt]
    joined = is_manifold | is_wedge

    ### Drop the facet-connected parts left unpaired at every split edge they use
    drop = torch.zeros(n_cells, dtype=torch.bool, device=device)
    if drop_fins:
        paired = torch.zeros(3 * n_cells, dtype=torch.bool, device=device)
        paired[order[is_wedge]] = True
        paired[nxt[is_wedge]] = True
        unpaired = (counts[edge_ids] > 2) & ~paired
        part = vectorized_connected_components(
            torch.stack([order[joined] // 3, nxt[joined] // 3], dim=1), n_cells
        )
        is_fin = torch.zeros(n_cells, dtype=torch.bool, device=device)
        is_fin[part[unpaired.nonzero().squeeze(1) // 3]] = True
        is_fin[part[order[is_wedge] // 3]] = False
        drop = is_fin[part]

    ### Star components of the corners, after (split) and before (whole) the split
    split_links = _corner_links(cells, order[joined], nxt[joined])
    whole_links = torch.cat(
        [split_links, _corner_links(cells, order[is_nonmanifold], nxt[is_nonmanifold])]
    )
    kept = ~drop.repeat_interleave(3)
    split_links = split_links[kept[split_links].all(dim=1)]
    whole_links = whole_links[kept[whole_links].all(dim=1)]
    split_labels = vectorized_connected_components(split_links, 3 * n_cells)
    whole_labels = vectorized_connected_components(whole_links, 3 * n_cells)

    ### One part of each star disconnected by the split keeps the vertex; the rest copy it
    keeper = torch.full_like(whole_labels, 3 * n_cells).scatter_reduce_(
        0, whole_labels[kept], split_labels[kept], reduce="amin"
    )
    copies = kept & (split_labels != keeper[whole_labels])
    copied, copy_ids = torch.unique(split_labels[copies], return_inverse=True)
    new_cells = tail.clone()
    new_cells[copies] = n_points + copy_ids
    sources = torch.cat([torch.arange(n_points, device=device), tail[copied]])
    kept_cells = (~drop).nonzero().squeeze(1)

    ### Count the non-manifold edges whose kept triangles use several copies
    at_split = kept & (counts[edge_ids] > 2)
    new_tail = new_cells[at_split]
    new_head = new_cells.reshape(n_cells, 3).roll(-1, dims=1).reshape(-1)[at_split]
    edge_copies = unique_index_tuples(
        torch.stack(
            [
                edge_ids[at_split],
                torch.minimum(new_tail, new_head),
                torch.maximum(new_tail, new_head),
            ],
            dim=1,
        ),
        index_bound=max(counts.shape[0], sources.shape[0]),
    )
    n_edges_split = int((torch.bincount(edge_copies[:, 0]) > 1).sum())

    from physicsnemo.mesh.mesh import Mesh

    split_mesh = Mesh(
        points=points[sources],
        cells=new_cells.reshape(n_cells, 3)[kept_cells],
        point_data=mesh.point_data[sources],
        cell_data=mesh.cell_data[kept_cells],
        global_data=mesh.global_data.clone(),
    )
    n_cells_dropped = n_cells - kept_cells.shape[0]
    return split_mesh, {
        "n_edges_split": n_edges_split,
        "n_vertices_duplicated": copied.shape[0],
        "n_cells_dropped": n_cells_dropped,
    }
