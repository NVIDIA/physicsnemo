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

"""Compute point-based adjacency relationships in simplicial meshes.

This module provides functions to compute:
- Point-to-cells adjacency (star of each vertex)
- Point-to-points adjacency (graph edges)
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.neighbors._adjacency import Adjacency

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def get_point_to_cells_adjacency(mesh: "Mesh") -> Adjacency:
    """Compute the star of each vertex (all cells containing each point).

    For each point in the mesh, finds all cells that contain that point. This
    is the graph-theoretic "star" operation on vertices.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh.

    Returns
    -------
    Adjacency
        Adjacency where ``adjacency.to_list()[i]`` contains all cell indices that
        contain point ``i``. Isolated points (not in any cells) have empty lists.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Triangle mesh with 4 points, 2 triangles
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> adj = get_point_to_cells_adjacency(mesh)
    >>> adj.to_list()
    [[0], [0, 1], [0, 1], [1]]
    """
    ### Handle empty mesh
    if mesh.n_cells == 0 or mesh.n_points == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_points + 1, dtype=torch.int64, device=mesh.points.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.points.device),
            device=mesh.points.device,
        )

    ### Sort the flattened cells by point id
    # The flattened cells list each cell's points in cell order, so a stable
    # sort by point id alone keeps each point's cells ascending, and the cell of
    # each sorted entry is its flat position // n_vertices_per_cell. No
    # composite (point, cell) keys and no cell-id array are needed.
    n_vertices_per_cell = mesh.cells.shape[1]
    sorted_point_ids, order = torch.sort(mesh.cells.reshape(-1), stable=True)

    ### offsets[i] is the number of entries whose point is below i
    offsets = torch.searchsorted(
        sorted_point_ids,
        torch.arange(
            mesh.n_points + 1, dtype=sorted_point_ids.dtype, device=mesh.points.device
        ),
    )
    del sorted_point_ids
    indices = order // n_vertices_per_cell
    return Adjacency(offsets=offsets, indices=indices, device=mesh.points.device)


def get_point_to_points_adjacency(mesh: "Mesh") -> Adjacency:
    """Compute point-to-point adjacency (graph edges of the mesh).

    For each point, finds all other points that share a cell with it. In simplicial
    meshes, this is equivalent to finding all points connected by an edge, since
    all vertices in a simplex are pairwise connected.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh.

    Returns
    -------
    Adjacency
        Adjacency where ``adjacency.to_list()[i]`` contains all point indices that
        share a cell (edge) with point ``i``. Isolated points have empty lists.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Three points forming a single triangle
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]])
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> adj = get_point_to_points_adjacency(mesh)
    >>> adj.to_list()
    [[1, 2], [0, 2], [0, 1]]
    """
    from physicsnemo.mesh.boundaries import extract_candidate_facets
    from physicsnemo.mesh.utilities._topology import extract_unique_edges
    from physicsnemo.utils._index_tuple_ops import sort_index_tuples

    ### Handle empty mesh, and vertex cells, which join no two points
    if mesh.n_cells == 0 or mesh.n_points == 0 or mesh.n_manifold_dims == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_points + 1, dtype=torch.int64, device=mesh.points.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.points.device),
            device=mesh.points.device,
        )

    from physicsnemo.mesh.neighbors._adjacency import (
        _PACKED_PAIR_KEY_LIMIT,
        adjacency_from_sorted_keys,
        build_adjacency_from_pairs,
    )

    n_points = mesh.n_points
    if n_points * n_points >= _PACKED_PAIR_KEY_LIMIT:
        ### The packed edge keys below could overflow: build from edge rows
        unique_edges, _ = extract_unique_edges(mesh)
        # For each edge [a, b], create both [a, b] and [b, a]
        bidirectional_edges = torch.cat([unique_edges, unique_edges.flip(dims=[1])])
        return build_adjacency_from_pairs(
            source_indices=bidirectional_edges[:, 0],
            target_indices=bidirectional_edges[:, 1],
            n_sources=n_points,
            n_targets=n_points,
        )

    ### Pack each canonical candidate edge (a <= b) into one key, a * n + b
    if mesh.n_manifold_dims == 1:
        candidate_edges = sort_index_tuples(mesh.cells)
    else:
        candidate_edges, parent_cell_indices = extract_candidate_facets(
            mesh.cells, manifold_codimension=mesh.n_manifold_dims - 1
        )
        del parent_cell_indices
    edge_keys = candidate_edges[:, 0].long() * n_points + candidate_edges[:, 1]
    del candidate_edges

    ### Deduplicate the edges, then add the reverse direction (b, a) of each
    # A degenerate self-edge (a, a) is listed twice, as with the edge rows.
    forward_keys = torch.unique(edge_keys)
    del edge_keys
    reverse_keys = (forward_keys % n_points) * n_points + forward_keys // n_points
    sorted_keys = torch.sort(torch.cat([forward_keys, reverse_keys])).values
    del forward_keys, reverse_keys
    return adjacency_from_sorted_keys(
        sorted_keys, n_points, n_points, dtype=mesh.cells.dtype
    )
