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

"""Support volume computation for Discrete Exterior Calculus.

Support volumes are geometric regions associated with primal simplices, formed by
the convex hull of the simplex and its circumcentric dual cell. These are fundamental
to DEC formulas for sharp and flat operators.

Key concept (Hirani Def. 2.4.9, line 2034):
    V_σᵏ = convex hull(σᵏ, ⋆σᵏ)

The support volumes perfectly tile the mesh: their union is |K| and intersections
have measure zero.

For implementing sharp/flat operators, we need the intersection of support volumes
with n-simplices (cells). Hirani Prop. 5.5.1 (lines 2345-2390) proves that these
can be computed efficiently using pyramid volumes.

References:
    Hirani (2003) Section 2.4, Proposition 5.5.1, Figure 5.4
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_edge_support_volume_cell_fractions(
    mesh: "Mesh",
    edges: torch.Tensor,
) -> torch.Tensor:
    """Compute |⋆edge ∩ cell| / |⋆edge| for all edge-cell pairs.

    For each edge and each cell containing it, computes the fraction of the edge's
    dual 1-cell (and support volume) that lies within that cell.

    This is needed for the DPP-flat operator (Hirani Eq. 5.5.3, line 2398):
        ⟨X♭, edge⟩ = Σ_{cells ⊃ edge} (|⋆edge ∩ cell|/|⋆edge|) × X(cell) · edge⃗

    From Hirani Prop. 5.5.1 (line 2348), this equals:
        |⋆edge ∩ cell| / |⋆edge| = |V_edge ∩ cell| / |V_edge|

    And from the pyramid volume analysis (lines 2361-2388), for dimension n:
        |V_edge ∩ cell| = 2 × (1/(n+1)) × |edge|/2 × |⋆edge ∩ cell|
        |V_edge| = Σ_{cells ⊃ edge} |V_edge ∩ cell|

    So: fraction = |⋆edge ∩ cell| / |⋆edge| = |⋆edge ∩ cell| / Σ|⋆edge ∩ cell|

    For 2D triangles, |⋆edge ∩ triangle| is the length of the dual edge segment
    from edge midpoint to triangle circumcenter.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh (must be 2D for now)
    edges : torch.Tensor
        Edge connectivity, shape (n_edges, 2)

    Returns
    -------
    torch.Tensor
        Sparse representation of fractions, shape (n_edges, max_cells_per_edge)
        where max_cells_per_edge = 2 for manifold meshes without boundary.

        For boundary edges (only 1 adjacent cell), the fraction is 1.0.
        For interior edges (2 adjacent cells), fractions sum to 1.0.

    Algorithm (2D specific):
        For each edge:
        1. Find all triangles containing it (typically 1 or 2)
        2. Compute circumcenter of each triangle
        3. Dual edge length in triangle = distance from edge midpoint to circumcenter
        4. Total dual edge length = sum over all triangles
        5. Fraction = (dual length in triangle) / (total dual length)

    Examples
    --------
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> edges = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 3], [2, 3]])
        >>> fractions = compute_edge_support_volume_cell_fractions(mesh, edges)
        >>> # fractions[i, j] = fraction of edge i's support volume in its j-th cell
    """
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"Support volume fractions only implemented for 2D manifolds. "
            f"Got {mesh.n_manifold_dims=}"
        )

    from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

    n_edges = len(edges)
    device = mesh.points.device
    dtype = mesh.points.dtype

    ### Find which cells contain each edge
    # Use facet extraction to map edges → parent cells
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    candidate_edges, parent_cells = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,  # Extract 1-simplices (edges) from 2-simplices (triangles)
    )

    ### Build mapping from edges to their parent cells
    # Each edge maps to a list of cell indices
    # Most edges have 1 (boundary) or 2 (interior) adjacent cells
    # Store as (n_edges, 2) with -1 for missing second cell
    from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

    edge_indices, matches = find_edges_in_reference(edges, candidate_edges)
    edge_to_cells = torch.full(
        (n_edges, 2), -1, dtype=torch.long, device=device
    )  # (n_edges, 2)

    ### Vectorized fill of edge_to_cells matrix
    # Filter to only matched candidates
    matched_edge_indices = edge_indices[matches]
    matched_cell_indices = parent_cells[matches]

    if len(matched_edge_indices) > 0:
        ### Sort by edge index to group edges together
        sort_order = torch.argsort(matched_edge_indices, stable=True)
        sorted_edges_idx = matched_edge_indices[sort_order]
        sorted_cells_idx = matched_cell_indices[sort_order]

        ### Compute within-group position (0, 1, 2, ...) for each entry
        # Find group boundaries where edge index changes
        group_starts = torch.cat(
            [
                sorted_edges_idx.new_zeros(1),
                torch.where(sorted_edges_idx[1:] != sorted_edges_idx[:-1])[0] + 1,
            ]
        )

        # Compute cumulative position within each group
        # positions[i] = i - group_start for entry i
        positions = torch.arange(len(sorted_edges_idx), device=device)
        group_ids = torch.searchsorted(group_starts, positions, right=True) - 1
        within_group_positions = positions - group_starts[group_ids]

        ### Keep only first 2 entries per edge (slot 0 and slot 1)
        valid_mask = within_group_positions < 2
        final_edge_indices = sorted_edges_idx[valid_mask]
        final_cell_indices = sorted_cells_idx[valid_mask]
        final_slots = within_group_positions[valid_mask]

        ### Fill matrix using advanced indexing
        edge_to_cells[final_edge_indices, final_slots] = final_cell_indices

    ### Compute circumcenters of all cells
    cell_vertices = mesh.points[mesh.cells]  # (n_cells, 3, n_spatial_dims)
    circumcenters = compute_circumcenters(cell_vertices)  # (n_cells, n_spatial_dims)

    ### For each edge, compute dual edge length segments
    # Dual edge goes from edge midpoint to circumcenters of adjacent cells
    edge_midpoints = (
        mesh.points[edges[:, 0]] + mesh.points[edges[:, 1]]
    ) / 2  # (n_edges, n_spatial_dims)

    ### Compute |⋆edge ∩ cell| for each edge-cell pair
    dual_edge_segments = torch.zeros(
        (n_edges, 2), dtype=dtype, device=device
    )  # (n_edges, 2)

    for slot in range(2):
        valid_mask = edge_to_cells[:, slot] >= 0
        # Clamp indices to 0 for invalid slots (distances will be zeroed by mask)
        safe_cell_indices = edge_to_cells[:, slot].clamp(min=0)

        # Distance from edge midpoint to circumcenter
        distances = torch.norm(
            circumcenters[safe_cell_indices] - edge_midpoints,
            dim=-1,
        )  # (n_edges,)

        # Zero out distances for invalid slots (no adjacent cell)
        dual_edge_segments[:, slot] = torch.where(
            valid_mask, distances, distances.new_zeros(())
        )

    ### Compute total dual edge length for each edge
    total_dual_lengths = dual_edge_segments.sum(dim=1)  # (n_edges,)

    ### Compute fractions: |⋆edge ∩ cell| / |⋆edge|
    fractions = dual_edge_segments / total_dual_lengths.unsqueeze(-1).clamp(
        min=safe_eps(total_dual_lengths.dtype)
    )

    return fractions  # (n_edges, 2) - fractions for up to 2 adjacent cells


def compute_vertex_support_volume_cell_fractions(
    mesh: "Mesh",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute |⋆vertex ∩ cell| / |⋆vertex| for all vertex-cell pairs.

    For each vertex v and each cell containing it, computes the fraction of v's
    total dual 0-cell volume (Voronoi region) that lies within that cell. These
    fractions sum to 1.0 for each vertex by construction, since
    :math:`|⋆v| = \sum_{\text{cells } c \ni v} |⋆v \cap c|`.

    This normalization is required by the PP-sharp operator so that it exactly
    reproduces constant gradients:

    .. math::

        \alpha^\sharp(v) = \sum_{\text{edges } [v,\sigma^0]}
            \langle \alpha, [v,\sigma^0] \rangle
            \sum_{\text{cells } \sigma^n \supset \text{edge}}
            \frac{|⋆v \cap \sigma^n|}{|⋆v|}
            \nabla\varphi_{\sigma^0, \sigma^n}

    For 2D triangles, :math:`|⋆v \cap \text{cell}|` is the area of the Voronoi
    region within the triangle, computed using the Meyer mixed area formula
    (Eq. 7 for acute triangles, Fig. 4 for obtuse).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of ``(fractions, cell_vertex_pairs)``:

        - ``fractions``: shape ``(n_pairs,)`` - the weight
          :math:`|⋆v \cap \text{cell}| / |⋆v|`
        - ``cell_vertex_pairs``: shape ``(n_pairs, 2)`` -
          ``[cell_idx, local_vertex_idx]``

        Fractions are guaranteed to sum to 1.0 for each vertex.

    Notes
    -----
    For non-2D manifolds, uses the barycentric approximation where each
    vertex's Voronoi region in a cell is ``|cell| / n_vertices_per_cell``.

    Returns a flat array of all (cell, vertex) pairs to avoid a dense tensor.
    """
    device = mesh.points.device
    dtype = mesh.points.dtype
    n_cells = mesh.n_cells
    n_vertices_per_cell = mesh.n_manifold_dims + 1

    ### Initialize storage for raw Voronoi areas |⋆v ∩ cell|
    n_pairs = n_cells * n_vertices_per_cell
    voronoi_areas = torch.zeros(n_pairs, dtype=dtype, device=device)
    cell_indices_out = torch.arange(n_cells, device=device).repeat_interleave(
        n_vertices_per_cell
    )
    local_vertex_indices = torch.arange(n_vertices_per_cell, device=device).repeat(
        n_cells
    )

    if mesh.n_manifold_dims != 2:
        ### Non-2D: barycentric approximation |⋆v ∩ cell| ≈ |cell| / n_verts
        cell_areas = mesh.cell_areas  # (n_cells,)
        approx_voronoi = cell_areas / n_vertices_per_cell  # (n_cells,)
        for local_v_idx in range(n_vertices_per_cell):
            pair_indices = (
                torch.arange(n_cells, device=device) * n_vertices_per_cell + local_v_idx
            )
            voronoi_areas[pair_indices] = approx_voronoi
    else:
        ### 2D manifolds: Meyer mixed area computation for |⋆v ∩ cell|
        from physicsnemo.mesh.geometry.dual_meshes import (
            _compute_meyer_mixed_voronoi_areas,
        )

        cell_vertices = mesh.points[mesh.cells]  # (n_cells, 3, n_spatial_dims)
        cell_areas = mesh.cell_areas  # (n_cells,)

        voronoi_areas[:] = _compute_meyer_mixed_voronoi_areas(
            cell_vertices, cell_areas
        )  # (n_cells * 3,)

    ### Normalize per vertex: fraction = |⋆v ∩ cell| / |⋆v|
    # Map each (cell, local_vertex) pair to its global vertex index
    global_vertex_indices = mesh.cells[cell_indices_out, local_vertex_indices]

    # Sum Voronoi areas per vertex to get total dual volume |⋆v|
    dual_volumes = torch.zeros(mesh.n_points, dtype=dtype, device=device)
    dual_volumes.scatter_add_(0, global_vertex_indices, voronoi_areas)

    # Divide each per-cell area by the vertex total (guaranteed to sum to 1.0)
    fractions = voronoi_areas / dual_volumes[global_vertex_indices].clamp(
        min=safe_eps(dtype)
    )

    cell_vertex_pairs = torch.stack([cell_indices_out, local_vertex_indices], dim=1)
    return fractions, cell_vertex_pairs


def compute_dual_edge_volumes_in_cells(
    mesh: "Mesh",
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute |⋆edge ∩ cell| for all edge-cell adjacencies.

    Returns the actual volume (not fraction) of dual 1-cell within each cell.
    This is the |⋆edge ∩ cell| term from Hirani Eq. 5.5.3.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh (2D for now)
    edges : torch.Tensor
        Edge connectivity, shape (n_edges, 2)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (dual_volumes_in_cells, edge_cell_mapping):
        - dual_volumes_in_cells: shape (n_edge_cell_pairs,)
        - edge_cell_mapping: shape (n_edge_cell_pairs, 2) - [edge_idx, cell_idx]

    Algorithm (2D):
        For each edge-cell pair:
        |⋆edge ∩ cell| = distance from edge midpoint to cell circumcenter
    """
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"Dual edge volumes only implemented for 2D. Got {mesh.n_manifold_dims=}"
        )

    from physicsnemo.mesh.boundaries import extract_candidate_facets
    from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters

    ### Extract all edges with their parent cells
    candidate_edges, parent_cells = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Match candidates to input edges
    from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

    edge_indices_for_candidates, matches = find_edges_in_reference(
        edges, candidate_edges
    )

    ### Filter to only matched pairs
    edge_indices = edge_indices_for_candidates[matches]
    cell_indices = parent_cells[matches]

    ### Compute circumcenters
    cell_vertices = mesh.points[mesh.cells]
    circumcenters = compute_circumcenters(cell_vertices)

    ### Compute edge midpoints
    edge_midpoints = (mesh.points[edges[:, 0]] + mesh.points[edges[:, 1]]) / 2

    ### For each matched pair, compute dual edge segment length
    # |⋆edge ∩ cell| = ||midpoint - circumcenter||
    dual_volumes = torch.norm(
        circumcenters[cell_indices] - edge_midpoints[edge_indices],
        dim=-1,
    )  # (n_matched,)

    ### Package output
    edge_cell_mapping = torch.stack([edge_indices, cell_indices], dim=1)

    return dual_volumes, edge_cell_mapping
