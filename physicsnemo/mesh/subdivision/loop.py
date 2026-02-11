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

"""Loop subdivision for simplicial meshes.

Loop subdivision is an approximating scheme where both old and new vertices
are repositioned. It produces smooth limit surfaces for triangular meshes.

Original vertices are moved using valence-based weights, and new edge midpoints
use weighted averages. This provides C² continuity for regular vertices.

Reference: Charles Loop, "Smooth Subdivision Surfaces Based on Triangles" (1987)
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs
from physicsnemo.mesh.subdivision._data import propagate_cell_data_to_children
from physicsnemo.mesh.subdivision._topology import (
    extract_unique_edges,
    generate_child_cells,
    get_subdivision_pattern,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def reposition_original_vertices_2d(
    mesh: "Mesh",
    unique_edges: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reposition original vertices using Loop's valence-based formula.

    For each vertex, compute new position as:
        new_pos = (1 - n*beta) * old_pos + beta * sum(neighbor_positions)

    where n is the vertex valence and beta depends on n.

    This implementation is fully vectorized using the Adjacency structure directly,
    avoiding any Python loops over mesh elements.

    Parameters
    ----------
    mesh : Mesh
        Input 2D manifold mesh
    unique_edges : torch.Tensor | None, optional
        Pre-computed unique edges (optional). If provided, uses these
        instead of recomputing them, which saves significant time.

    Returns
    -------
    torch.Tensor
        Repositioned vertex positions, shape (n_points, n_spatial_dims)
    """
    device = mesh.points.device
    n_points = mesh.n_points

    ### Get point-to-point adjacency (vertex neighbors)
    if unique_edges is not None:
        # Build adjacency directly from pre-computed edges (avoids re-extracting
        # edges from cells, which is the expensive part of get_point_to_points_adjacency)
        sources = torch.cat([unique_edges[:, 0], unique_edges[:, 1]])
        targets = torch.cat([unique_edges[:, 1], unique_edges[:, 0]])
        adjacency = build_adjacency_from_pairs(sources, targets, n_sources=n_points)
    else:
        from physicsnemo.mesh.neighbors import get_point_to_points_adjacency

        adjacency = get_point_to_points_adjacency(mesh)

    ### Compute valences for all points at once
    # valences[i] = offsets[i+1] - offsets[i]
    # Shape: (n_points,)
    valences = adjacency.offsets[1:] - adjacency.offsets[:-1]

    ### Compute beta weights for all valences at once
    # Vectorize the beta formula
    # If valence == 3: beta = 3/16
    # Else: beta = (1/n) * (5/8 - (3/8 + 1/4 * cos(2π/n))²)
    # Shape: (n_points,)

    cos_term = 3.0 / 8.0 + 0.25 * torch.cos(2.0 * torch.pi / valences.float())
    beta_else = (1.0 / valences.float()) * (5.0 / 8.0 - cos_term * cos_term)
    beta = torch.where(valences == 3, 3.0 / 16.0, beta_else)
    # Handle isolated vertices (valence=0) - beta should be 0 to keep original position
    beta = torch.where(valences > 0, beta, 0.0)

    ### Compute neighbor position sums for all points using scatter_add
    # For each neighbor relationship, add neighbor's position to source point's sum
    # Shape: (n_points, n_spatial_dims)
    neighbor_sums = torch.zeros_like(mesh.points)

    # Get source point indices by expanding offsets
    # For adjacency.indices[i], the source point is the one whose offset range contains i
    # We can use searchsorted or create source indices directly
    source_point_indices = torch.repeat_interleave(
        torch.arange(n_points, dtype=torch.int64, device=device),
        valences,
    )

    # Get neighbor positions and scatter-add to source points
    # adjacency.indices contains the neighbor point indices
    neighbor_positions = mesh.points[
        adjacency.indices
    ]  # (total_neighbors, n_spatial_dims)

    # Expand source_point_indices for scatter_add
    source_point_indices_expanded = source_point_indices.unsqueeze(-1).expand(
        -1, mesh.n_spatial_dims
    )

    neighbor_sums.scatter_add_(
        dim=0,
        index=source_point_indices_expanded,
        src=neighbor_positions,
    )

    ### Apply Loop formula for all points at once
    # new_pos = (1 - n*beta) * old_pos + beta * sum(neighbors)
    # Shape: (n_points, n_spatial_dims)
    valences_expanded = valences.unsqueeze(-1).float()  # (n_points, 1)
    beta_expanded = beta.unsqueeze(-1)  # (n_points, 1)

    new_positions = (
        1 - valences_expanded * beta_expanded
    ) * mesh.points + beta_expanded * neighbor_sums

    return new_positions


def compute_loop_edge_positions_2d(
    mesh: "Mesh",
    unique_edges: torch.Tensor,
) -> torch.Tensor:
    """Compute new edge vertex positions using Loop's edge rule.

    For an interior edge with endpoints v0, v1 and opposite vertices opp0, opp1:
        new_pos = 3/8 * (v0 + v1) + 1/8 * (opp0 + opp1)

    For boundary edges, use simple average: (v0 + v1) / 2

    Parameters
    ----------
    mesh : Mesh
        Input 2D manifold mesh
    unique_edges : torch.Tensor
        Edge connectivity, shape (n_edges, 2)

    Returns
    -------
    torch.Tensor
        Edge vertex positions, shape (n_edges, n_spatial_dims)
    """
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    n_edges = len(unique_edges)
    device = mesh.points.device

    ### Build edge-to-cells mapping
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )

    _, inverse_indices = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )

    ### Count adjacent cells for each edge
    # Shape: (n_edges,)
    adjacent_counts = torch.bincount(inverse_indices, minlength=n_edges)

    ### Identify boundary vs interior edges
    is_interior = adjacent_counts == 2
    is_boundary = ~is_interior

    ### Initialize edge positions
    edge_positions = torch.zeros(
        (n_edges, mesh.n_spatial_dims),
        dtype=mesh.points.dtype,
        device=device,
    )

    ### Compute boundary edge positions (simple average)
    # Shape: (n_boundary_edges, n_spatial_dims)
    boundary_edges = unique_edges[is_boundary]
    if len(boundary_edges) > 0:
        v0_pos = mesh.points[boundary_edges[:, 0]]
        v1_pos = mesh.points[boundary_edges[:, 1]]
        edge_positions[is_boundary] = (v0_pos + v1_pos) / 2

    ### Compute interior edge positions (Loop's formula)
    interior_edge_indices = torch.where(is_interior)[0]
    n_interior = len(interior_edge_indices)

    if n_interior > 0:
        ### For each interior edge, find its two adjacent cells (vectorized)
        # Filter candidate edges to only those belonging to interior edges
        is_interior_candidate = is_interior[inverse_indices]
        interior_inverse = inverse_indices[is_interior_candidate]
        interior_parents = parent_cell_indices[is_interior_candidate]

        # Sort by edge index to group candidates belonging to same edge
        sort_indices = torch.argsort(interior_inverse)
        sorted_parents = interior_parents[sort_indices]

        # Reshape to (n_interior, 2) - each interior edge has exactly 2 adjacent cells
        # Shape: (n_interior, 2)
        adjacent_cells = sorted_parents.reshape(n_interior, 2)

        ### Get the triangles
        # Shape: (n_interior, 2, 3)
        triangles = mesh.cells[adjacent_cells]

        ### Get edge vertices
        # Shape: (n_interior, 2)
        interior_edges = unique_edges[interior_edge_indices]

        ### Find opposite vertices for each triangle
        # For each triangle, find the vertex that's not in the edge
        # Shape: (n_interior, 2, 3) - broadcast comparison
        # Create masks for which vertices are in the edge
        edge_v0 = interior_edges[:, 0].unsqueeze(1).unsqueeze(2)  # (n_interior, 1, 1)
        edge_v1 = interior_edges[:, 1].unsqueeze(1).unsqueeze(2)  # (n_interior, 1, 1)

        # Check if each triangle vertex matches edge vertices
        # Shape: (n_interior, 2, 3)
        is_edge_vertex = (triangles == edge_v0) | (triangles == edge_v1)

        # The opposite vertex is where is_edge_vertex is False
        # Shape: (n_interior, 2, 3)
        opposite_mask = ~is_edge_vertex

        # Extract opposite vertices using argmax (finds first True in mask)
        # Shape: (n_interior, 2)
        # torch.argmax on the opposite_mask gives us the index of the opposite vertex
        opposite_vertex_indices = torch.argmax(
            opposite_mask.int(), dim=2
        )  # (n_interior, 2)

        # Gather the actual vertex IDs
        # Shape: (n_interior, 2)
        opposite_vertices = torch.gather(
            triangles,  # (n_interior, 2, 3)
            dim=2,
            index=opposite_vertex_indices.unsqueeze(2),  # (n_interior, 2, 1)
        ).squeeze(2)  # (n_interior, 2)

        ### Compute Loop edge rule: 3/8 * (v0 + v1) + 1/8 * (opp0 + opp1)
        v0_pos = mesh.points[interior_edges[:, 0]]  # (n_interior, n_spatial_dims)
        v1_pos = mesh.points[interior_edges[:, 1]]  # (n_interior, n_spatial_dims)
        opp0_pos = mesh.points[opposite_vertices[:, 0]]  # (n_interior, n_spatial_dims)
        opp1_pos = mesh.points[opposite_vertices[:, 1]]  # (n_interior, n_spatial_dims)

        edge_positions[interior_edge_indices] = (3.0 / 8.0) * (v0_pos + v1_pos) + (
            1.0 / 8.0
        ) * (opp0_pos + opp1_pos)

    return edge_positions


def subdivide_loop(mesh: "Mesh") -> "Mesh":
    """Perform one level of Loop subdivision on the mesh.

    Loop subdivision is an approximating scheme that:
    1. Repositions original vertices using valence-weighted averaging
    2. Creates new edge vertices using weighted stencils
    3. Connects vertices to form 4 triangles per original triangle

    Properties:
    - Approximating: original vertices move to new positions
    - Produces C² smooth limit surfaces for regular meshes
    - Designed for 2D manifolds (triangular meshes)
    - For non-2D manifolds: raises NotImplementedError

    The result is a smoother mesh that approximates (rather than interpolates)
    the original surface.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to subdivide (must be 2D manifold)

    Returns
    -------
    Mesh
        Subdivided mesh with Loop-repositioned vertices

    Raises
    ------
    NotImplementedError
        If n_manifold_dims is not 2

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Smooth a rough triangulated surface
        >>> mesh = sphere_icosahedral.load(subdivisions=2)
        >>> smooth = subdivide_loop(mesh)
        >>> # Original vertices have moved; result is smoother
    """
    from physicsnemo.mesh.mesh import Mesh

    ### Check manifold dimension
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"Loop subdivision currently only supports 2D manifolds (triangular meshes). "
            f"Got {mesh.n_manifold_dims=}. "
            f"For other dimensions, use linear subdivision instead."
        )

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return mesh

    ### Extract unique edges
    unique_edges, edge_inverse = extract_unique_edges(mesh)
    n_original_points = mesh.n_points

    ### Reposition original vertices (pass unique_edges to avoid recomputation)
    repositioned_vertices = reposition_original_vertices_2d(
        mesh, unique_edges=unique_edges
    )

    ### Compute new edge vertex positions
    edge_vertices = compute_loop_edge_positions_2d(mesh, unique_edges)

    ### Combine repositioned original vertices and new edge vertices
    new_points = torch.cat([repositioned_vertices, edge_vertices], dim=0)

    ### Interpolate point_data
    # For Loop subdivision, data should ideally be repositioned like geometry,
    # but for simplicity, use linear interpolation for edge data
    from physicsnemo.mesh.subdivision._data import interpolate_point_data_to_edges

    new_point_data = interpolate_point_data_to_edges(
        point_data=mesh.point_data,
        edges=unique_edges,
        n_original_points=n_original_points,
    )

    ### Get subdivision pattern
    subdivision_pattern = get_subdivision_pattern(mesh.n_manifold_dims)
    subdivision_pattern = subdivision_pattern.to(mesh.cells.device)

    ### Generate child cells
    child_cells, parent_indices = generate_child_cells(
        parent_cells=mesh.cells,
        edge_inverse=edge_inverse,
        n_original_points=n_original_points,
        subdivision_pattern=subdivision_pattern,
    )

    ### Propagate cell_data
    new_cell_data = propagate_cell_data_to_children(
        cell_data=mesh.cell_data,
        parent_indices=parent_indices,
        n_total_children=len(child_cells),
    )

    ### Create and return subdivided mesh
    return Mesh(
        points=new_points,
        cells=child_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,
    )
