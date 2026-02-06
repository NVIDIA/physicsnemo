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

"""Angle computation for curvature calculations.

Computes angles and solid angles at vertices in n-dimensional simplicial meshes.
Uses dimension-agnostic formulas based on Gram determinants and stable atan2.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.curvature._utils import (
    compute_triangle_angles,
    stable_angle_between_vectors,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_solid_angle_at_tet_vertex(
    vertex_pos: torch.Tensor,
    opposite_vertices: torch.Tensor,
) -> torch.Tensor:
    """Compute solid angle at apex of tetrahedron using van Oosterom-Strackee formula.

    For a tetrahedron with apex at vertex_pos and opposite triangular face
    defined by opposite_vertices, computes the solid angle subtended.

    Uses the stable atan2-based formula:
        Ω = 2 * atan2(|det(a, b, c)|, denominator)
    where:
        a, b, c are vectors from vertex to the three opposite vertices
        denominator = ||a|| ||b|| ||c|| + (a·b)||c|| + (b·c)||a|| + (c·a)||b||

    Parameters
    ----------
    vertex_pos : torch.Tensor
        Position of apex vertex, shape (..., n_spatial_dims)
    opposite_vertices : torch.Tensor
        Positions of three opposite vertices,
        shape (..., 3, n_spatial_dims)

    Returns
    -------
    torch.Tensor
        Solid angle in steradians, shape (...)
        Range: [0, 2π) for valid tetrahedra

    Reference:
        van Oosterom & Strackee (1983), "The Solid Angle of a Plane Triangle"
        IEEE Trans. Biomed. Eng. BME-30(2):125-126
    """
    ### Compute edge vectors from vertex to opposite face vertices
    # Shape: (..., 3, n_spatial_dims)
    a = opposite_vertices[..., 0, :] - vertex_pos
    b = opposite_vertices[..., 1, :] - vertex_pos
    c = opposite_vertices[..., 2, :] - vertex_pos

    ### Compute norms
    norm_a = torch.norm(a, dim=-1)  # (...)
    norm_b = torch.norm(b, dim=-1)
    norm_c = torch.norm(c, dim=-1)

    ### Compute dot products
    ab = (a * b).sum(dim=-1)
    bc = (b * c).sum(dim=-1)
    ca = (c * a).sum(dim=-1)

    ### Compute determinant |det([a, b, c])|
    # For 3D: det = a · (b × c)
    # General: Use torch.det on stacked matrix
    # Stack as matrix: (..., 3, n_spatial_dims) where rows are a, b, c

    if a.shape[-1] == 3:
        # 3D case: use cross product (faster)
        cross_bc = torch.cross(b, c, dim=-1)
        det = (a * cross_bc).sum(dim=-1)
    else:
        # Higher dimensional case: use determinant
        # Need square matrix, so take first 3 spatial dimensions
        # This is an approximation for n_spatial_dims > 3
        matrix = torch.stack([a[..., :3], b[..., :3], c[..., :3]], dim=-2)
        det = torch.det(matrix)

    numerator = torch.abs(det)

    ### Compute denominator
    denominator = norm_a * norm_b * norm_c + ab * norm_c + bc * norm_a + ca * norm_b

    ### Compute solid angle using atan2 (stable)
    solid_angle = 2 * torch.atan2(numerator, denominator)

    return solid_angle


def compute_angles_at_vertices(mesh: "Mesh") -> torch.Tensor:
    """Compute sum of angles at each vertex over all incident cells.

    Uses dimension-specific formulas:
    - 1D manifolds (edges): Angle between incident edges
    - 2D manifolds (triangles): Sum of corner angles in incident triangles
    - 3D manifolds (tets): Sum of solid angles at vertex in incident tets

    All formulas use numerically stable atan2-based computation.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_points,) containing sum of angles at each vertex.
        For isolated vertices, angle is 0.

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> triangle_mesh = two_triangles_2d.load()
        >>> angles = compute_angles_at_vertices(triangle_mesh)
        >>> # Angles are computed at each vertex
    """
    device = mesh.points.device
    n_points = mesh.n_points
    n_manifold_dims = mesh.n_manifold_dims

    ### Initialize angle sums
    angle_sums = torch.zeros(n_points, dtype=mesh.points.dtype, device=device)

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return angle_sums

    ### Get point-to-cells adjacency
    from physicsnemo.mesh.neighbors import get_point_to_cells_adjacency

    adjacency = get_point_to_cells_adjacency(mesh)

    ### Compute angles based on manifold dimension
    if n_manifold_dims == 1:
        ### 1D manifolds (edges): Interior angle at each vertex in polygon
        # For closed polygons, must handle reflex angles (> π) correctly
        # Use signed angle based on cross product (2D) or ordering

        ### Group points by number of incident edges
        neighbor_counts = adjacency.offsets[1:] - adjacency.offsets[:-1]  # (n_points,)

        ### Handle most common case: exactly 2 incident edges (vectorized)
        two_edge_mask = neighbor_counts == 2
        two_edge_indices = torch.where(two_edge_mask)[0]  # (n_two_edge,)

        if len(two_edge_indices) > 0:
            # Extract the two incident edges for each vertex
            offsets_two_edge = adjacency.offsets[two_edge_indices]  # (n_two_edge,)
            edge0_cells = adjacency.indices[offsets_two_edge]  # (n_two_edge,)
            edge1_cells = adjacency.indices[offsets_two_edge + 1]  # (n_two_edge,)

            # Get edge vertices: (n_two_edge, 2)
            edge0_verts = mesh.cells[edge0_cells]
            edge1_verts = mesh.cells[edge1_cells]

            # Determine incoming/outgoing edges
            # Incoming: point_idx is at position 1 (edge = [prev, point_idx])
            # Outgoing: point_idx is at position 0 (edge = [point_idx, next])

            # Check if point is at position 1 of edge0
            edge0_is_incoming = edge0_verts[:, 1] == two_edge_indices  # (n_two_edge,)

            # Select prev/next vertices based on edge configuration
            # If edge0 is incoming: prev=edge0[0], next=edge1[1]
            # If edge1 is incoming: prev=edge1[0], next=edge0[1]
            prev_vertex = torch.where(
                edge0_is_incoming,
                edge0_verts[:, 0],
                edge1_verts[:, 0],
            )  # (n_two_edge,)
            next_vertex = torch.where(
                edge0_is_incoming,
                edge1_verts[:, 1],
                edge0_verts[:, 1],
            )  # (n_two_edge,)

            # Compute vectors
            v_from_prev = (
                mesh.points[two_edge_indices] - mesh.points[prev_vertex]
            )  # (n_two_edge, n_spatial_dims)
            v_to_next = (
                mesh.points[next_vertex] - mesh.points[two_edge_indices]
            )  # (n_two_edge, n_spatial_dims)

            # Compute interior angles
            if mesh.n_spatial_dims == 2:
                # 2D: Use signed angle with cross product
                cross_z = (
                    v_from_prev[:, 0] * v_to_next[:, 1]
                    - v_from_prev[:, 1] * v_to_next[:, 0]
                )  # (n_two_edge,)
                dot = (v_from_prev * v_to_next).sum(dim=-1)  # (n_two_edge,)

                # Signed angle in range [-π, π]
                signed_angle = torch.atan2(cross_z, dot)

                # Interior angle: π - signed_angle
                interior_angles = torch.pi - signed_angle
            else:
                # Higher dimensions: Use unsigned angle
                interior_angles = stable_angle_between_vectors(v_from_prev, v_to_next)

            # Assign angles to vertices
            angle_sums[two_edge_indices] = interior_angles

        ### Handle vertices with >2 edges (junctions) - rare, so small loop acceptable
        # Note: This case is uncommon (junction points in 1D meshes)
        # Full vectorization is complex due to variable edge counts
        multi_edge_mask = neighbor_counts > 2
        multi_edge_indices = torch.where(multi_edge_mask)[0]

        for point_idx_tensor in multi_edge_indices:
            point_idx = int(point_idx_tensor)
            offset_start = int(adjacency.offsets[point_idx])
            offset_end = int(adjacency.offsets[point_idx + 1])
            incident_cells = adjacency.indices[offset_start:offset_end]
            n_incident = len(incident_cells)

            # Get all incident edge vertices
            edge_verts = mesh.cells[incident_cells]  # (n_incident, 2)

            # Find the "other" vertex in each edge (not point_idx)
            # Create mask for vertices that equal point_idx
            is_point = edge_verts == point_idx
            other_indices = torch.where(
                ~is_point, edge_verts, edge_verts.new_full(edge_verts.shape, -1)
            )
            other_vertices = other_indices.max(dim=1).values  # (n_incident,)

            # Compute vectors from point to all neighbors
            vectors = (
                mesh.points[other_vertices] - mesh.points[point_idx]
            )  # (n_incident, n_spatial_dims)

            # Compute all pairwise angles using broadcasting
            # Expand vectors for pairwise computation
            v_i = vectors.unsqueeze(1)  # (n_incident, 1, n_spatial_dims)
            v_j = vectors.unsqueeze(0)  # (1, n_incident, n_spatial_dims)

            # Compute pairwise angles for all combinations
            # We only need upper triangle (i < j)
            pairwise_angles = stable_angle_between_vectors(
                v_i.expand(-1, n_incident, -1).reshape(-1, mesh.n_spatial_dims),
                v_j.expand(n_incident, -1, -1).reshape(-1, mesh.n_spatial_dims),
            ).reshape(n_incident, n_incident)

            # Sum only upper triangle (i < j) to avoid double-counting
            triu_indices = torch.triu_indices(
                n_incident, n_incident, offset=1, device=device
            )
            angle_sum = pairwise_angles[triu_indices[0], triu_indices[1]].sum()

            angle_sums[point_idx] = angle_sum

    elif n_manifold_dims == 2:
        ### 2D manifolds (triangles): Sum of corner angles
        # For each triangle and each vertex, compute the corner angle

        # Vectorized: For all cells, compute all three corner angles
        # Shape: (n_cells, 3, n_spatial_dims)
        cell_vertices = mesh.points[mesh.cells]

        # Compute angle at each corner
        # Corner 0: angle at vertex 0, between edges to vertices 1 and 2
        angles_corner0 = compute_triangle_angles(
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
        )  # (n_cells,)

        # Corner 1: angle at vertex 1
        angles_corner1 = compute_triangle_angles(
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
        )

        # Corner 2: angle at vertex 2
        angles_corner2 = compute_triangle_angles(
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
        )

        ### Scatter angles to corresponding vertices
        # Each cell contributes one angle to each of its three vertices
        angle_sums.scatter_add_(0, mesh.cells[:, 0], angles_corner0)
        angle_sums.scatter_add_(0, mesh.cells[:, 1], angles_corner1)
        angle_sums.scatter_add_(0, mesh.cells[:, 2], angles_corner2)

    elif n_manifold_dims == 3:
        ### 3D manifolds (tetrahedra): Sum of solid angles
        # For each tet and each vertex, compute solid angle at that vertex

        # Vectorized computation for all tets
        # Shape: (n_cells, 4, n_spatial_dims)
        cell_vertices = mesh.points[mesh.cells]
        n_cells = mesh.n_cells

        # Compute all 4 solid angles per tet in parallel
        # For each local vertex position, get opposite triangle vertices
        # Vertex 0: opposite vertices are [1, 2, 3]
        # Vertex 1: opposite vertices are [0, 2, 3]
        # Vertex 2: opposite vertices are [0, 1, 3]
        # Vertex 3: opposite vertices are [0, 1, 2]

        # Stack all apex vertices: (n_cells, 4, n_spatial_dims)
        all_apexes = cell_vertices  # (n_cells, 4, n_spatial_dims)

        # Stack all opposite triangles: (n_cells, 4, 3, n_spatial_dims)
        # For each of 4 vertices, select the 3 opposite vertices
        # Opposite vertices of vertex i are all vertices except i
        opposite_vertex_indices = torch.tensor(
            [[j for j in range(4) if j != i] for i in range(4)],
            device=mesh.cells.device,
            dtype=torch.long,
        )  # (4, 3)

        # Gather opposite vertices: (n_cells, 4, 3, n_spatial_dims)
        all_opposites = torch.gather(
            cell_vertices.unsqueeze(1).expand(
                -1, 4, -1, -1
            ),  # (n_cells, 4, 4, n_spatial_dims)
            dim=2,
            index=opposite_vertex_indices.unsqueeze(0)
            .unsqueeze(-1)
            .expand(n_cells, -1, -1, mesh.n_spatial_dims),
        )  # (n_cells, 4, 3, n_spatial_dims)

        # Reshape for batch computation
        apexes_flat = all_apexes.reshape(n_cells * 4, mesh.n_spatial_dims)
        opposites_flat = all_opposites.reshape(n_cells * 4, 3, mesh.n_spatial_dims)

        # Compute all solid angles at once
        solid_angles_flat = compute_solid_angle_at_tet_vertex(
            apexes_flat, opposites_flat
        )

        # Scatter all angles to vertices in one operation
        # Flatten vertex indices and solid angles together
        vertex_indices_flat = mesh.cells.reshape(-1)  # (n_cells * 4,)
        angle_sums.scatter_add_(0, vertex_indices_flat, solid_angles_flat)

    else:
        raise NotImplementedError(
            f"Angle computation not implemented for {n_manifold_dims=}. "
            f"Currently supported: 1D (edges), 2D (triangles), 3D (tetrahedra)."
        )

    return angle_sums
