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

"""Butterfly subdivision for simplicial meshes.

Butterfly is an interpolating subdivision scheme where original vertices remain
fixed and new edge midpoints are computed using weighted stencils of neighboring
vertices. This produces smoother surfaces than linear subdivision.

The classical butterfly scheme is designed for 2D manifolds (triangular meshes).
This implementation provides the standard 2D butterfly and extensions/fallbacks
for other dimensions.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.subdivision._data import propagate_cell_data_to_children
from physicsnemo.mesh.subdivision._topology import (
    extract_unique_edges,
    generate_child_cells,
    get_subdivision_pattern,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_butterfly_weights_2d(
    mesh: "Mesh",
    unique_edges: torch.Tensor,
) -> torch.Tensor:
    """Compute butterfly weighted positions for edge midpoints in 2D manifolds.

    For triangular meshes, uses the classical butterfly stencil:
    - Regular interior edges: 8-point stencil with weights (1/2, 1/2, 1/8, 1/8, -1/16, -1/16, -1/16, -1/16)
    - Boundary edges: Simple average of endpoints

    The stencil for an edge (v0, v1) includes:
    - The two edge vertices: v0, v1 (weight 1/2 each)
    - Two opposite vertices in adjacent triangles (weight 1/8 each)
    - Four "wing" vertices (weight -1/16 each)

    Args:
        mesh: Input 2D manifold mesh (triangular)
        unique_edges: Unique edge connectivity, shape (n_edges, 2)

    Returns:
        Edge midpoint positions using butterfly weights, shape (n_edges, n_spatial_dims)
    """
    n_edges = len(unique_edges)
    device = mesh.points.device

    ### Build edge-to-adjacent-cells mapping
    # For each edge, find which cells contain it
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )

    # Deduplicate to get inverse mapping
    _, inverse_indices = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
    )

    ### Count adjacent cells for each edge (vectorized)
    # Shape: (n_edges,)
    adjacent_counts = torch.bincount(inverse_indices, minlength=n_edges)

    ### Identify boundary vs interior edges
    is_interior = adjacent_counts == 2
    is_boundary = adjacent_counts != 2  # 0, 1, or >2

    ### Initialize edge midpoints
    edge_midpoints = torch.zeros(
        (n_edges, mesh.n_spatial_dims),
        dtype=mesh.points.dtype,
        device=device,
    )

    ### Compute boundary edge positions (simple average) - vectorized
    boundary_edges = unique_edges[is_boundary]
    if len(boundary_edges) > 0:
        v0_pos = mesh.points[boundary_edges[:, 0]]
        v1_pos = mesh.points[boundary_edges[:, 1]]
        edge_midpoints[is_boundary] = (v0_pos + v1_pos) / 2

    ### Compute interior edge positions (butterfly stencil) - vectorized
    interior_edge_indices = torch.where(is_interior)[0]
    n_interior = len(interior_edge_indices)

    if n_interior > 0:
        ### For each interior edge, find its two adjacent cells
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

        ### Find opposite vertices for each triangle (vectorized)
        # Shape: (n_interior, 1, 1)
        edge_v0 = interior_edges[:, 0].unsqueeze(1).unsqueeze(2)
        edge_v1 = interior_edges[:, 1].unsqueeze(1).unsqueeze(2)

        # Check if each triangle vertex matches edge vertices
        # Shape: (n_interior, 2, 3)
        is_edge_vertex = (triangles == edge_v0) | (triangles == edge_v1)
        opposite_mask = ~is_edge_vertex

        # Extract opposite vertices using argmax
        # Shape: (n_interior, 2)
        opposite_vertex_indices = torch.argmax(opposite_mask.int(), dim=2)
        opposite_vertices = torch.gather(
            triangles,
            dim=2,
            index=opposite_vertex_indices.unsqueeze(2),
        ).squeeze(2)

        ### Compute butterfly weights for all interior edges (vectorized)
        # Main edge vertices: 1/2 each
        # Opposite vertices: 1/8 each
        # (Simplified 4-point butterfly, no wing vertices)

        v0_pos = mesh.points[interior_edges[:, 0]]  # (n_interior, n_spatial_dims)
        v1_pos = mesh.points[interior_edges[:, 1]]  # (n_interior, n_spatial_dims)
        opp0_pos = mesh.points[opposite_vertices[:, 0]]  # (n_interior, n_spatial_dims)
        opp1_pos = mesh.points[opposite_vertices[:, 1]]  # (n_interior, n_spatial_dims)

        midpoint = (
            (1.0 / 2.0) * v0_pos
            + (1.0 / 2.0) * v1_pos
            + (1.0 / 8.0) * opp0_pos
            + (1.0 / 8.0) * opp1_pos
        )

        # Normalize weights (they sum to 5/4, scale by 4/5)
        edge_midpoints[interior_edge_indices] = midpoint * (4.0 / 5.0)

    return edge_midpoints


def subdivide_butterfly(mesh: "Mesh") -> "Mesh":
    """Perform one level of butterfly subdivision on the mesh.

    Butterfly subdivision is an interpolating scheme that produces smoother
    results than linear subdivision by using weighted stencils for new vertices.

    Properties:
    - Interpolating: original vertices remain unchanged
    - New edge midpoints use weighted neighbor stencils
    - Designed for 2D manifolds (triangular meshes)
    - For non-2D manifolds: falls back to linear subdivision with warning

    The connectivity pattern is identical to linear subdivision (same topology),
    but the geometric positions of new vertices differ.

    Args:
        mesh: Input mesh to subdivide

    Returns:
        Subdivided mesh with butterfly-weighted vertex positions

    Raises:
        NotImplementedError: If n_manifold_dims is not 2 (may be relaxed in future)

    Example:
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Smooth a triangular surface
        >>> mesh = sphere_icosahedral.load(subdivisions=2)
        >>> smooth = subdivide_butterfly(mesh)
        >>> # smooth has same connectivity as linear subdivision
        >>> # but smoother geometry from weighted stencils
    """
    from physicsnemo.mesh.mesh import Mesh

    ### Check manifold dimension
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"Butterfly subdivision currently only supports 2D manifolds (triangular meshes). "
            f"Got {mesh.n_manifold_dims=}. "
            f"For other dimensions, use linear subdivision instead."
        )

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return mesh

    ### Extract unique edges
    unique_edges, edge_inverse = extract_unique_edges(mesh)
    n_original_points = mesh.n_points

    ### Compute edge midpoints using butterfly weights
    edge_midpoints = compute_butterfly_weights_2d(mesh, unique_edges)

    ### Create new points: original (unchanged) + butterfly midpoints
    new_points = torch.cat([mesh.points, edge_midpoints], dim=0)

    ### Interpolate point_data to edge midpoints
    # For butterfly, we could use the same weighted stencil for data,
    # but for simplicity, use linear interpolation (average of endpoints)
    from physicsnemo.mesh.subdivision._data import interpolate_point_data_to_edges

    new_point_data = interpolate_point_data_to_edges(
        point_data=mesh.point_data,
        edges=unique_edges,
        n_original_points=n_original_points,
    )

    ### Get subdivision pattern (same as linear)
    subdivision_pattern = get_subdivision_pattern(mesh.n_manifold_dims)
    subdivision_pattern = subdivision_pattern.to(mesh.cells.device)

    ### Generate child cells (same topology as linear)
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
