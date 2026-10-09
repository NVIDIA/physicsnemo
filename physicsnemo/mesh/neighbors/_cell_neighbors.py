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

"""Compute cell-based adjacency relationships in simplicial meshes.

This module provides functions to compute:
- Cell-to-cells adjacency based on shared facets
- Cell-to-points adjacency (vertices of each cell)
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.neighbors._adjacency import Adjacency
from physicsnemo.utils._index_tuple_ops import unique_index_tuples

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def get_cell_to_cells_adjacency(
    mesh: "Mesh",
    adjacency_codimension: int = 1,
) -> Adjacency:
    """Compute cell-to-cells adjacency based on shared facets.

    Two cells are considered adjacent if they share a k-codimension facet.
    For example:

    - codimension=1: Share an (n-1)-facet (e.g., triangles sharing an edge in 2D,
      tetrahedra sharing a triangular face in 3D)
    - codimension=2: Share an (n-2)-facet (e.g., tetrahedra sharing an edge in 3D)
    - codimension=k: Share any (n-k)-facet

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh.
    adjacency_codimension : int, optional
        Codimension of shared facets defining adjacency.

        - 1 (default): Cells must share a codimension-1 facet (most restrictive)
        - 2: Cells must share a codimension-2 facet (more permissive)
        - k: Cells must share a codimension-k facet

    Returns
    -------
    Adjacency
        Adjacency where ``adjacency.to_list()[i]`` contains all cell indices that
        share a k-codimension facet with cell ``i``. Each neighbor appears exactly
        once per source cell.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Two triangles sharing an edge
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> adj = get_cell_to_cells_adjacency(mesh, adjacency_codimension=1)
    >>> adj.to_list()
    [[1], [0]]
    """
    from physicsnemo.mesh.boundaries import extract_candidate_facets
    from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs

    device = mesh.cells.device

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return Adjacency(
            offsets=torch.zeros(1, dtype=torch.int64, device=device),
            indices=torch.zeros(0, dtype=torch.int64, device=device),
            device=device,
        )

    ### Extract all candidate facets from cells
    # candidate_facets: (n_cells * n_facets_per_cell, n_vertices_per_facet)
    # parent_cell_indices: (n_cells * n_facets_per_cell,)
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=adjacency_codimension,
    )

    ### Label each candidate with its unique facet and that facet's cell count
    _, facet_ids, facet_counts = unique_index_tuples(
        candidate_facets,
        index_bound=mesh.n_points,
        return_inverse=True,
        return_counts=True,
    )

    ### Group the candidates of each facet into one contiguous run
    order = torch.argsort(facet_ids)
    sorted_cells = parent_cell_indices[order]
    sorted_facet_ids = facet_ids[order]
    facet_starts = torch.cumsum(facet_counts, dim=0) - facet_counts
    group_starts = facet_starts[sorted_facet_ids]  # run start of each candidate
    positions = torch.arange(len(sorted_cells), dtype=torch.int64, device=device)
    local_indices = positions - group_starts  # rank of each candidate in its run

    ### Pair every candidate with each other candidate of its run
    # A facet shared by k cells yields k * (k - 1) directed pairs; a boundary
    # facet (k = 1) yields none. Fully vectorized, with one device sync for the
    # data-dependent number of pairs.
    n_pairs_per_candidate = facet_counts[sorted_facet_ids] - 1
    n_pairs = int(n_pairs_per_candidate.sum())
    source_positions = torch.repeat_interleave(
        positions, n_pairs_per_candidate, output_size=n_pairs
    )

    # For a source at local index i of a run of k, the targets are the local
    # indices 0, ..., i - 1, i + 1, ..., k - 1: count 0..k-2, skipping i.
    pair_starts = torch.cumsum(n_pairs_per_candidate, dim=0) - n_pairs_per_candidate
    counter = (
        torch.arange(n_pairs, dtype=torch.int64, device=device)
        - pair_starts[source_positions]
    )
    target_local_indices = counter + (counter >= local_indices[source_positions])
    target_positions = group_starts[source_positions] + target_local_indices

    # Stack into pairs (source, target)
    # Shape: (n_pairs, 2)
    cell_pairs_tensor = torch.stack(
        [sorted_cells[source_positions], sorted_cells[target_positions]], dim=1
    )

    ### Remove duplicate pairs (can happen if cells share multiple facets)
    # This ensures each neighbor appears exactly once per source
    unique_pairs = unique_index_tuples(cell_pairs_tensor, index_bound=mesh.n_cells)

    ### Build adjacency using shared utility
    return build_adjacency_from_pairs(
        source_indices=unique_pairs[:, 0],
        target_indices=unique_pairs[:, 1],
        n_sources=mesh.n_cells,
        n_targets=mesh.n_cells,
    )


def get_cell_to_points_adjacency(mesh: "Mesh") -> Adjacency:
    """Get the vertices (points) that comprise each cell.

    This is a simple wrapper around the cells array that returns it in the
    standard Adjacency format for consistency with other neighbor queries.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh.

    Returns
    -------
    Adjacency
        Adjacency where ``adjacency.to_list()[i]`` contains all point indices that
        are vertices of cell ``i``. For simplicial meshes, all cells have the same
        number of vertices (``n_manifold_dims + 1``).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Triangle mesh with 2 cells
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> adj = get_cell_to_points_adjacency(mesh)
    >>> adj.to_list()
    [[0, 1, 2], [1, 3, 2]]
    """
    ### Handle empty mesh
    if mesh.n_cells == 0:
        return Adjacency(
            offsets=torch.zeros(1, dtype=torch.int64, device=mesh.cells.device),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
            device=mesh.cells.device,
        )

    n_cells, n_vertices_per_cell = mesh.cells.shape

    ### Create uniform offsets (each cell has exactly n_vertices_per_cell vertices)
    # offsets[i] = i * n_vertices_per_cell
    offsets = (
        torch.arange(
            n_cells + 1,
            dtype=torch.int64,
            device=mesh.cells.device,
        )
        * n_vertices_per_cell
    )

    ### Flatten cells array to get all point indices
    indices = mesh.cells.reshape(-1)

    return Adjacency(offsets=offsets, indices=indices, device=mesh.cells.device)
