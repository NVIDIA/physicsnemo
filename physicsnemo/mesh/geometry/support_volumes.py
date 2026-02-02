# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

    Args:
        mesh: Simplicial mesh (must be 2D for now)
        edges: Edge connectivity, shape (n_edges, 2)

    Returns:
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

    Example:
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

    from physicsnemo.mesh.calculus._circumcentric_dual import compute_circumcenters

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

    ### Sort edges canonically for matching
    sorted_candidate_edges, _ = torch.sort(candidate_edges, dim=-1)
    sorted_edges, _ = torch.sort(edges, dim=-1)

    ### Build mapping from edges to their parent cells
    # Each edge maps to a list of cell indices
    # Use hash for efficient lookup
    max_vertex = max(edges.max(), candidate_edges.max()) + 1
    edge_hash = sorted_edges[:, 0] * max_vertex + sorted_edges[:, 1]
    candidate_hash = (
        sorted_candidate_edges[:, 0] * max_vertex + sorted_candidate_edges[:, 1]
    )

    ### For each edge, find all cells containing it
    # Most edges have 1 (boundary) or 2 (interior) adjacent cells
    # Store as (n_edges, 2) with -1 for missing second cell
    edge_to_cells = torch.full(
        (n_edges, 2), -1, dtype=torch.long, device=device
    )  # (n_edges, 2)

    ### Build reverse mapping: for each candidate edge, which slot in edges array?
    edge_hash_sorted, sort_idx = torch.sort(edge_hash)
    positions = torch.searchsorted(edge_hash_sorted, candidate_hash)
    positions = positions.clamp(max=len(edge_hash_sorted) - 1)

    matches = edge_hash_sorted[positions] == candidate_hash
    edge_indices = sort_idx[positions]  # Map candidate → edge index

    ### Count how many cells we've seen for each edge
    edge_cell_counts = torch.zeros(n_edges, dtype=torch.long, device=device)

    ### Fill in edge_to_cells matrix
    for i in range(len(candidate_edges)):
        if matches[i]:
            edge_idx = edge_indices[i]
            cell_idx = parent_cells[i]
            slot = edge_cell_counts[edge_idx]
            if slot < 2:
                edge_to_cells[edge_idx, slot] = cell_idx
                edge_cell_counts[edge_idx] += 1

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
        if not valid_mask.any():
            continue

        valid_edges = torch.where(valid_mask)[0]
        cell_indices = edge_to_cells[valid_edges, slot]

        # Distance from edge midpoint to circumcenter
        distances = torch.norm(
            circumcenters[cell_indices] - edge_midpoints[valid_edges],
            dim=-1,
        )  # (n_valid,)

        dual_edge_segments[valid_edges, slot] = distances

    ### Compute total dual edge length for each edge
    total_dual_lengths = dual_edge_segments.sum(dim=1)  # (n_edges,)

    ### Compute fractions: |⋆edge ∩ cell| / |⋆edge|
    fractions = dual_edge_segments / total_dual_lengths.unsqueeze(-1).clamp(min=1e-10)

    return fractions  # (n_edges, 2) - fractions for up to 2 adjacent cells


def compute_vertex_support_volume_cell_fractions(
    mesh: "Mesh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute |⋆vertex ∩ cell| / |cell| for all vertex-cell pairs.

    For each vertex and each cell containing it, computes the fraction of the vertex's
    dual 0-cell volume (Voronoi region) that lies within that cell, divided by the cell volume.

    This is needed for the PP-sharp operator (Hirani Eq. 5.8.1, line 2596):
        α♯(v) = Σ_{edges from v} ⟨α,edge⟩ × Σ_{cells ⊃ edge} (|⋆v ∩ cell|/|cell|) × ∇φ

    For 2D triangles, |⋆v ∩ triangle| is the area of the Voronoi region within
    the triangle. This was already computed as part of `compute_dual_volumes_0()`.

    Args:
        mesh: Simplicial mesh (must be 2D for now)

    Returns:
        Tuple of (fractions, cell_vertex_pairs):
        - fractions: shape (n_pairs,) - the weight |⋆v ∩ cell| / |cell|
        - cell_vertex_pairs: shape (n_pairs, 2) - [cell_idx, local_vertex_idx]

        For each pair (cell_i, vertex_j in cell_i), gives the geometric weight.

    Algorithm (2D):
        Uses the same Meyer mixed area computation as in compute_dual_volumes_0():
        - Acute triangles: Use Eq. 7 cotangent formula
        - Obtuse triangles: Use Fig. 4 mixed area subdivision

        The per-cell contribution is already the |⋆v ∩ cell| value.
        Divide by cell area to get the required fraction.

    Note:
        This returns a flat array of all (cell, vertex) pairs to avoid dense tensor.
        The sparse representation is more memory-efficient.
    """
    device = mesh.points.device
    dtype = mesh.points.dtype
    n_cells = mesh.n_cells
    n_vertices_per_cell = mesh.n_manifold_dims + 1

    if mesh.n_manifold_dims != 2:
        ### For non-2D: use uniform weighting (barycentric approximation)
        # Each vertex gets equal fraction in each incident cell
        uniform_fraction = 1.0 / n_vertices_per_cell

        n_pairs = n_cells * n_vertices_per_cell
        fractions = torch.full((n_pairs,), uniform_fraction, dtype=dtype, device=device)

        cell_indices = torch.arange(n_cells, device=device).repeat_interleave(
            n_vertices_per_cell
        )
        local_vertex_indices = torch.arange(n_vertices_per_cell, device=device).repeat(
            n_cells
        )
        cell_vertex_pairs = torch.stack([cell_indices, local_vertex_indices], dim=1)

        return fractions, cell_vertex_pairs

    ### 2D manifolds: Use rigorous Meyer mixed area computation
    ### We need to recompute the per-cell Voronoi contributions
    # (These are the |⋆v ∩ cell| values before summing over all incident cells)

    cell_vertices = mesh.points[mesh.cells]  # (n_cells, 3, n_spatial_dims)
    cell_areas = mesh.cell_areas  # (n_cells,)

    from physicsnemo.mesh.curvature._utils import compute_triangle_angles

    ### Compute angles
    angles_0 = compute_triangle_angles(
        cell_vertices[:, 0, :],
        cell_vertices[:, 1, :],
        cell_vertices[:, 2, :],
    )
    angles_1 = compute_triangle_angles(
        cell_vertices[:, 1, :],
        cell_vertices[:, 2, :],
        cell_vertices[:, 0, :],
    )
    angles_2 = compute_triangle_angles(
        cell_vertices[:, 2, :],
        cell_vertices[:, 0, :],
        cell_vertices[:, 1, :],
    )
    all_angles = torch.stack([angles_0, angles_1, angles_2], dim=1)  # (n_cells, 3)

    is_obtuse = torch.any(all_angles > torch.pi / 2, dim=1)  # (n_cells,)

    ### Initialize storage for (cell_idx, local_vertex_idx, fraction) tuples
    # We'll have n_cells × 3 pairs
    n_pairs = n_cells * 3
    fractions = torch.zeros(n_pairs, dtype=dtype, device=device)
    cell_indices_out = torch.arange(n_cells, device=device).repeat_interleave(3)
    local_vertex_indices = torch.tensor([0, 1, 2], device=device).repeat(n_cells)

    ### Compute fractions for acute triangles
    non_obtuse_mask = ~is_obtuse

    if non_obtuse_mask.any():
        non_obtuse_indices = torch.where(non_obtuse_mask)[0]
        non_obtuse_vertices = cell_vertices[non_obtuse_mask]
        non_obtuse_angles = all_angles[non_obtuse_mask]
        non_obtuse_areas = cell_areas[non_obtuse_mask]

        for local_v_idx in range(3):
            next_idx = (local_v_idx + 1) % 3
            prev_idx = (local_v_idx + 2) % 3

            edge_to_next = (
                non_obtuse_vertices[:, next_idx, :]
                - non_obtuse_vertices[:, local_v_idx, :]
            )
            edge_to_prev = (
                non_obtuse_vertices[:, prev_idx, :]
                - non_obtuse_vertices[:, local_v_idx, :]
            )

            edge_to_next_sq = (edge_to_next**2).sum(dim=-1)
            edge_to_prev_sq = (edge_to_prev**2).sum(dim=-1)

            cot_prev = torch.cos(non_obtuse_angles[:, prev_idx]) / torch.sin(
                non_obtuse_angles[:, prev_idx]
            ).clamp(min=1e-10)
            cot_next = torch.cos(non_obtuse_angles[:, next_idx]) / torch.sin(
                non_obtuse_angles[:, next_idx]
            ).clamp(min=1e-10)

            ### Voronoi contribution (Eq. 7)
            voronoi_in_cell = (
                edge_to_next_sq * cot_prev + edge_to_prev_sq * cot_next
            ) / 8.0

            ### Fraction = |⋆v ∩ cell| / |cell|
            fraction = voronoi_in_cell / non_obtuse_areas

            ### Store in output array
            pair_indices = non_obtuse_indices * 3 + local_v_idx
            fractions[pair_indices] = fraction

    ### Compute fractions for obtuse triangles
    if is_obtuse.any():
        obtuse_indices = torch.where(is_obtuse)[0]
        obtuse_areas = cell_areas[is_obtuse]
        obtuse_angles = all_angles[is_obtuse]

        for local_v_idx in range(3):
            is_obtuse_at_vertex = obtuse_angles[:, local_v_idx] > torch.pi / 2

            ### Mixed area contribution (Fig. 4)
            voronoi_in_cell = torch.where(
                is_obtuse_at_vertex,
                obtuse_areas / 2.0,
                obtuse_areas / 4.0,
            )

            ### Fraction = |⋆v ∩ cell| / |cell|
            fraction = voronoi_in_cell / obtuse_areas

            ### Store in output array
            pair_indices = obtuse_indices * 3 + local_v_idx
            fractions[pair_indices] = fraction

    ### Package output
    cell_vertex_pairs = torch.stack([cell_indices_out, local_vertex_indices], dim=1)

    return fractions, cell_vertex_pairs


def compute_dual_edge_volumes_in_cells(
    mesh: "Mesh",
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute |⋆edge ∩ cell| for all edge-cell adjacencies.

    Returns the actual volume (not fraction) of dual 1-cell within each cell.
    This is the |⋆edge ∩ cell| term from Hirani Eq. 5.5.3.

    Args:
        mesh: Simplicial mesh (2D for now)
        edges: Edge connectivity, shape (n_edges, 2)

    Returns:
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
    from physicsnemo.mesh.calculus._circumcentric_dual import compute_circumcenters

    ### Extract all edges with their parent cells
    candidate_edges, parent_cells = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Match candidates to sorted edges
    sorted_candidates, _ = torch.sort(candidate_edges, dim=-1)
    sorted_edges_input, _ = torch.sort(edges, dim=-1)

    max_vertex = max(edges.max(), candidate_edges.max()) + 1
    candidate_hash = sorted_candidates[:, 0] * max_vertex + sorted_candidates[:, 1]
    edge_hash = sorted_edges_input[:, 0] * max_vertex + sorted_edges_input[:, 1]

    edge_hash_sorted, sort_idx = torch.sort(edge_hash)
    positions = torch.searchsorted(edge_hash_sorted, candidate_hash)
    positions = positions.clamp(max=len(edge_hash_sorted) - 1)

    matches = edge_hash_sorted[positions] == candidate_hash
    edge_indices_for_candidates = sort_idx[positions]

    ### Filter to only matched pairs
    matched_mask = matches
    edge_indices = edge_indices_for_candidates[matched_mask]
    cell_indices = parent_cells[matched_mask]

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
