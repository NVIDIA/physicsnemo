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
        Adjacency where adjacency.to_list()[i] contains all cell indices that
        share a k-codimension facet with cell i. Each neighbor appears exactly
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
    from physicsnemo.mesh.boundaries import (
        categorize_facets_by_count,
        extract_candidate_facets,
    )

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return Adjacency(
            offsets=torch.zeros(1, dtype=torch.int64, device=mesh.cells.device),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
        )

    ### Extract all candidate facets from cells
    # candidate_facets: (n_cells * n_facets_per_cell, n_vertices_per_facet)
    # parent_cell_indices: (n_cells * n_facets_per_cell,)
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=adjacency_codimension,
    )

    ### Find shared facets (those appearing in 2+ cells)
    _, inverse_indices, _ = categorize_facets_by_count(
        candidate_facets, target_counts="shared"
    )

    ### Filter to only keep candidate facets that are shared
    # inverse_indices maps candidates to unique shared facets (or -1 if not shared)
    candidate_is_shared = inverse_indices >= 0

    # Extract only the parent cells and inverse indices for shared facets
    shared_parent_cells = parent_cell_indices[candidate_is_shared]
    shared_inverse = inverse_indices[candidate_is_shared]

    ### Handle case where no cells share facets
    if len(shared_parent_cells) == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_cells + 1, dtype=torch.int64, device=mesh.cells.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
        )

    ### Build cell-to-cell pairs using vectorized operations
    # Sort by unique facet index to group cells sharing the same facet
    sort_by_facet = torch.argsort(shared_inverse)
    sorted_cells = shared_parent_cells[sort_by_facet]
    sorted_facet_ids = shared_inverse[sort_by_facet]

    # Find boundaries of each unique shared facet
    # diff != 0 marks transitions between different facets
    facet_changes = torch.cat(
        [
            sorted_facet_ids.new_zeros(1),
            torch.where(sorted_facet_ids[1:] != sorted_facet_ids[:-1])[0] + 1,
            sorted_facet_ids.new_tensor([len(sorted_facet_ids)]),
        ]
    )

    # Generate all pairs for cells sharing each facet
    # Fully vectorized implementation - no Python loops over facets

    ### Compute the size (number of cells) for each unique shared facet
    # Shape: (n_unique_shared_facets,)
    facet_sizes = facet_changes[1:] - facet_changes[:-1]

    ### Filter to only facets shared by 2+ cells (can form pairs)
    # Single-cell facets cannot form pairs
    multi_cell_facet_mask = facet_sizes > 1

    if not multi_cell_facet_mask.any():
        # No facets shared by multiple cells
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_cells + 1, dtype=torch.int64, device=mesh.cells.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.cells.device),
        )

    ### Build arrays for vectorized pair generation
    # For each facet, we'll generate all directed pairs (i, j) where i != j
    # Fully vectorized - no Python loops whatsoever

    # Get sizes only for facets with multiple cells
    valid_facet_sizes = facet_sizes[multi_cell_facet_mask]
    n_valid_facets = len(valid_facet_sizes)

    # Filter facet_changes to only include valid facets
    valid_facet_starts = facet_changes[:-1][multi_cell_facet_mask]

    ### Extract all cells belonging to valid facets (those with 2+ cells)
    # Fully vectorized - no Python loops or .tolist() calls
    total_cells_in_valid_facets = valid_facet_sizes.sum()

    # Generate indices into sorted_cells for all cells in valid facets
    # For each facet: [start, start+1, ..., end-1]
    # Vectorized: repeat each start by facet_size, then add [0,1,2,...,size-1]

    # Generate local indices [0, 1, 2, ..., size-1] for each facet
    # For facet_sizes [2, 3, 2], we want [0, 1, 0, 1, 2, 0, 1]
    # Fully vectorized approach: use cumulative indexing with group offsets

    # Create cumulative index for all positions
    cumulative_idx = torch.arange(
        total_cells_in_valid_facets, dtype=torch.int64, device=mesh.cells.device
    )

    # For each position, compute the start index of its facet group
    # First, compute cumulative starts: [0, size[0], size[0]+size[1], ...]
    facet_cumulative_starts = torch.cat(
        [
            torch.tensor([0], dtype=torch.int64, device=mesh.cells.device),
            torch.cumsum(valid_facet_sizes[:-1], dim=0),
        ]
    )

    # Expand starts to match each cell position
    start_indices_per_cell = torch.repeat_interleave(
        facet_cumulative_starts, valid_facet_sizes
    )

    # Local index = cumulative_idx - start_of_its_group
    local_indices = cumulative_idx - start_indices_per_cell

    # Generate indices into sorted_cells
    # Start indices in sorted_cells repeated by facet size + local offset
    valid_facet_starts_expanded = torch.repeat_interleave(
        valid_facet_starts, valid_facet_sizes
    )
    cell_indices_into_sorted = valid_facet_starts_expanded + local_indices

    # Extract cell IDs
    cells_in_valid_facets = sorted_cells[cell_indices_into_sorted]

    # Assign facet ID to each cell
    # Shape: (total_cells_in_valid_facets,)
    facet_ids_per_cell = torch.repeat_interleave(
        torch.arange(n_valid_facets, dtype=torch.int64, device=mesh.cells.device),
        valid_facet_sizes,
    )

    ### Generate all directed pairs (i, j) where i != j
    # Each cell needs (facet_size - 1) pairs
    facet_sizes_per_cell = valid_facet_sizes[facet_ids_per_cell]
    n_pairs_per_cell = facet_sizes_per_cell - 1

    # Repeat source cells by (facet_size - 1)
    source_cells = torch.repeat_interleave(cells_in_valid_facets, n_pairs_per_cell)
    source_facet_ids = torch.repeat_interleave(facet_ids_per_cell, n_pairs_per_cell)
    source_local_indices = torch.repeat_interleave(local_indices, n_pairs_per_cell)

    # Generate target local indices: for each source at local_idx i in facet of size n,
    # generate [0, 1, ..., i-1, i+1, ..., n-1] (all indices except i)
    # Fully vectorized approach using boundary-based cumulative counter

    # For each source cell, generate a counter: 0, 1, 2, ..., (facet_size-2)
    # For n_pairs_per_cell [1, 2, 1], we want [0, 0, 1, 0]
    # Same vectorization approach as local_indices

    # Create cumulative index for all pair positions
    # Total pairs = length of the repeated source_cells tensor
    total_pairs = len(source_cells)
    pair_cumulative_idx = torch.arange(
        total_pairs, dtype=torch.int64, device=mesh.cells.device
    )

    # Compute cumulative starts for each cell's target block
    pair_cumulative_starts = torch.cat(
        [
            torch.tensor([0], dtype=torch.int64, device=mesh.cells.device),
            torch.cumsum(n_pairs_per_cell[:-1], dim=0),
        ]
    )

    # Expand starts to match each pair position
    pair_start_indices = torch.repeat_interleave(
        pair_cumulative_starts, n_pairs_per_cell
    )

    # Counter = cumulative_idx - start_of_its_block
    within_facet_counter = pair_cumulative_idx - pair_start_indices

    # Adjust counters to skip the source cell's local index
    target_local_indices = (
        within_facet_counter + (within_facet_counter >= source_local_indices).long()
    )

    # Convert target local indices to global cell IDs
    # For each target, we need: cells_in_valid_facets[facet_start + local_idx]
    facet_cumsum = torch.cat(
        [
            torch.tensor([0], dtype=torch.int64, device=mesh.cells.device),
            torch.cumsum(valid_facet_sizes, dim=0)[:-1],
        ]
    )
    target_global_positions = facet_cumsum[source_facet_ids] + target_local_indices
    target_cells = cells_in_valid_facets[target_global_positions]

    # Stack into pairs (source, target)
    # Shape: (total_pairs, 2)
    cell_pairs_tensor = torch.stack([source_cells, target_cells], dim=1)

    ### Remove duplicate pairs (can happen if cells share multiple facets)
    # This ensures each neighbor appears exactly once per source
    from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs

    unique_pairs = torch.unique(cell_pairs_tensor, dim=0)

    ### Build adjacency using shared utility
    return build_adjacency_from_pairs(
        source_indices=unique_pairs[:, 0],
        target_indices=unique_pairs[:, 1],
        n_sources=mesh.n_cells,
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
        Adjacency where adjacency.to_list()[i] contains all point indices that
        are vertices of cell i. For simplicial meshes, all cells have the same
        number of vertices (n_manifold_dims + 1).

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

    return Adjacency(offsets=offsets, indices=indices)
