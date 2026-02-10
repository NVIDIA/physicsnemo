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

"""Angle computation for curvature calculations.

Computes angles and solid angles at vertices in n-dimensional simplicial meshes.
For manifold dimension >= 2, delegates to the unified formula in
:mod:`physicsnemo.mesh.geometry._angles`. The 1D case (edge meshes) requires
special handling because the relevant quantity is the *inter-cell* turning
angle between adjacent edges, not an intra-cell interior angle.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.curvature._utils import stable_angle_between_vectors
from physicsnemo.mesh.geometry._angles import compute_vertex_angle_sums

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_angles_at_vertices(mesh: "Mesh") -> torch.Tensor:
    """Compute sum of angles at each vertex over all incident cells.

    For manifold dimension >= 2, uses the unified correlation-matrix formula
    from :func:`~physicsnemo.mesh.geometry._angles.compute_vertex_angle_sums`.
    For 1D manifolds (edge meshes), computes the inter-cell turning angle
    between adjacent edges at each vertex.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_points,)`` containing sum of angles at each vertex.
        For isolated vertices, angle is 0.

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> triangle_mesh = two_triangles_2d.load()
        >>> angles = compute_angles_at_vertices(triangle_mesh)
        >>> # Angles are computed at each vertex
    """
    ### For 2D+ manifolds, delegate to the unified geometry primitive
    if mesh.n_manifold_dims >= 2:
        return compute_vertex_angle_sums(mesh)

    ### 1D manifolds (edges): inter-cell turning angle at each vertex
    # This is conceptually different from intra-cell angles: we need the
    # angle between adjacent edges sharing a vertex, which requires
    # examining the cell connectivity.

    device = mesh.points.device
    n_points = mesh.n_points

    angle_sums = torch.zeros(n_points, dtype=mesh.points.dtype, device=device)

    if mesh.n_cells == 0:
        return angle_sums

    from physicsnemo.mesh.neighbors import get_point_to_cells_adjacency

    adjacency = get_point_to_cells_adjacency(mesh)

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
        edge0_is_incoming = edge0_verts[:, 1] == two_edge_indices  # (n_two_edge,)

        # Select prev/next vertices based on edge configuration
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

            # Signed angle in range [-pi, pi]
            signed_angle = torch.atan2(cross_z, dot)

            # Interior angle: pi - signed_angle
            interior_angles = torch.pi - signed_angle
        else:
            # Higher dimensions: Use unsigned angle
            interior_angles = stable_angle_between_vectors(v_from_prev, v_to_next)

        # Assign angles to vertices
        angle_sums[two_edge_indices] = interior_angles

    ### Handle vertices with >2 edges (junctions) - rare, so small loop is acceptable
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
        v_i = vectors.unsqueeze(1)  # (n_incident, 1, n_spatial_dims)
        v_j = vectors.unsqueeze(0)  # (1, n_incident, n_spatial_dims)

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

    return angle_sums
