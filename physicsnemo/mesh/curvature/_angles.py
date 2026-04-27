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

For manifold dimension >= 2, delegates to the unified intra-cell formula in
:mod:`physicsnemo.mesh.geometry._angles`. The 1D case (edge meshes) requires
special handling because the relevant quantity is the *inter-cell* turning
angle between adjacent edges, not an intra-cell interior angle.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from physicsnemo.mesh.geometry._angles import (
    compute_vertex_angle_sums,
    stable_angle_between_vectors,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_angles_at_vertices(mesh: "Mesh") -> Float[torch.Tensor, " n_points"]:
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
    if mesh.n_manifold_dims >= 2:
        return compute_vertex_angle_sums(mesh)

    return _compute_1d_turning_angle_sums(mesh)


def _compute_1d_turning_angle_sums(mesh: "Mesh") -> torch.Tensor:
    """Compute inter-edge turning angles at vertices of a 1D edge mesh.

    For 1D manifolds, the relevant curvature quantity is the turning angle
    between adjacent edges sharing a vertex, not an intra-cell interior angle.
    For 2D ambient space, signed angles (via atan2) capture orientation;
    for higher ambient dimensions, unsigned angles are used.

    Parameters
    ----------
    mesh : Mesh
        1D edge mesh (``n_manifold_dims == 1``).

    Returns
    -------
    torch.Tensor
        Turning angle sum at each vertex, shape ``(n_points,)``.
    """
    device = mesh.points.device
    n_points = mesh.n_points

    angle_sums = torch.zeros(n_points, dtype=mesh.points.dtype, device=device)

    if mesh.n_cells == 0:
        return angle_sums

    adjacency = mesh.get_point_to_cells_adjacency()
    neighbor_counts = adjacency.counts  # (n_points,)

    ### Handle most common case: exactly 2 incident edges (vectorized)
    two_edge_mask = neighbor_counts == 2
    two_edge_indices = torch.where(two_edge_mask)[0]  # (n_two_edge,)

    if len(two_edge_indices) > 0:
        offsets_two_edge = adjacency.offsets[two_edge_indices]  # (n_two_edge,)
        edge0_cells = adjacency.indices[offsets_two_edge]  # (n_two_edge,)
        edge1_cells = adjacency.indices[offsets_two_edge + 1]  # (n_two_edge,)

        edge0_verts = mesh.cells[edge0_cells]  # (n_two_edge, 2)
        edge1_verts = mesh.cells[edge1_cells]  # (n_two_edge, 2)

        # Determine which edge is incoming (point is at position 1)
        edge0_is_incoming = edge0_verts[:, 1] == two_edge_indices

        prev_vertex = torch.where(
            edge0_is_incoming,
            edge0_verts[:, 0],
            edge1_verts[:, 0],
        )
        next_vertex = torch.where(
            edge0_is_incoming,
            edge1_verts[:, 1],
            edge0_verts[:, 1],
        )

        v_from_prev = mesh.points[two_edge_indices] - mesh.points[prev_vertex]
        v_to_next = mesh.points[next_vertex] - mesh.points[two_edge_indices]

        if mesh.n_spatial_dims == 2:
            # 2D: signed angle via cross product -> interior angle = pi - signed
            cross_z = (
                v_from_prev[:, 0] * v_to_next[:, 1]
                - v_from_prev[:, 1] * v_to_next[:, 0]
            )
            dot = (v_from_prev * v_to_next).sum(dim=-1)
            interior_angles = torch.pi - torch.atan2(cross_z, dot)
        else:
            interior_angles = stable_angle_between_vectors(v_from_prev, v_to_next)

        angle_sums[two_edge_indices] = interior_angles

    ### Handle vertices with >2 edges (junctions) - rare, Python loop acceptable
    multi_edge_indices = torch.where(neighbor_counts > 2)[0]

    for point_idx_tensor in multi_edge_indices:
        point_idx = int(point_idx_tensor)
        offset_start = int(adjacency.offsets[point_idx])
        offset_end = int(adjacency.offsets[point_idx + 1])
        incident_cells = adjacency.indices[offset_start:offset_end]
        n_incident = len(incident_cells)

        edge_verts = mesh.cells[incident_cells]  # (n_incident, 2)

        # Find the "other" vertex in each edge (not point_idx)
        is_point = edge_verts == point_idx
        other_indices = torch.where(
            ~is_point, edge_verts, edge_verts.new_full(edge_verts.shape, -1)
        )
        other_vertices = other_indices.max(dim=1).values  # (n_incident,)

        vectors = mesh.points[other_vertices] - mesh.points[point_idx]

        # Pairwise angles between all neighbor directions
        v_i = (
            vectors.unsqueeze(1)
            .expand(-1, n_incident, -1)
            .reshape(-1, mesh.n_spatial_dims)
        )
        v_j = (
            vectors.unsqueeze(0)
            .expand(n_incident, -1, -1)
            .reshape(-1, mesh.n_spatial_dims)
        )
        pairwise_angles = stable_angle_between_vectors(v_i, v_j).reshape(
            n_incident, n_incident
        )

        # Sum upper triangle only (avoid double-counting)
        triu_idx = torch.triu_indices(n_incident, n_incident, offset=1, device=device)
        angle_sums[point_idx] = pairwise_angles[triu_idx[0], triu_idx[1]].sum()

    return angle_sums
