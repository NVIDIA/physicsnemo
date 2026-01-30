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

"""Divergence operator for vector fields.

Implements divergence using both DEC and LSQ methods.

DEC formula (from paper lines 1610-1654):
    div(X)(v₀) = (1/|⋆v₀|) Σ_{edges from v₀} |⋆edge∩cell| × (X·edge_unit)

Physical interpretation: Net flux through dual cell boundary per unit volume.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_divergence_points_dec(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute divergence at vertices using DEC: div = -δ♭.

    Uses the explicit formula from DEC paper for divergence of a dual vector field.

    Args:
        mesh: Simplicial mesh
        vector_field: Vectors at vertices, shape (n_points, n_spatial_dims)

    Returns:
        Divergence at vertices, shape (n_points,)
    """
    from physicsnemo.mesh.calculus._circumcentric_dual import (
        get_or_compute_dual_volumes_0,
    )

    n_points = mesh.n_points

    ### Get dual volumes
    dual_volumes = get_or_compute_dual_volumes_0(mesh)  # |⋆v₀|

    ### Extract edges
    # Use facet extraction to get all edges
    codim_to_edges = mesh.n_manifold_dims - 1
    edge_mesh = mesh.get_facet_mesh(manifold_codimension=codim_to_edges)
    edges = edge_mesh.cells  # (n_edges, 2)

    # Sort edges for canonical ordering
    sorted_edges, _ = torch.sort(edges, dim=-1)

    ### Get edge vectors
    edge_vectors = mesh.points[sorted_edges[:, 1]] - mesh.points[sorted_edges[:, 0]]
    edge_lengths = torch.norm(edge_vectors, dim=-1)
    edge_unit = edge_vectors / edge_lengths.unsqueeze(-1).clamp(min=1e-10)

    ### Compute divergence at each vertex
    # Simplified implementation: for each vertex, sum flux through edges
    divergence = torch.zeros(
        n_points, dtype=vector_field.dtype, device=mesh.points.device
    )

    ### Vectorized edge contributions
    v0_indices = sorted_edges[:, 0]  # (n_edges,)
    v1_indices = sorted_edges[:, 1]  # (n_edges,)

    # Vector field at edges (average of endpoints): (n_edges, n_spatial_dims)
    v_edge = (vector_field[v0_indices] + vector_field[v1_indices]) / 2

    # Flux through all edges: v·edge_direction (n_edges,)
    flux = (v_edge * edge_unit).sum(dim=-1)

    # Scatter-add contributions with appropriate signs
    # v0: positive flux (outward)
    # v1: negative flux (inward)
    divergence.scatter_add_(0, v0_indices, flux)
    divergence.scatter_add_(0, v1_indices, -flux)

    ### Normalize by dual volumes
    divergence = divergence / dual_volumes.clamp(min=1e-10)

    return divergence


def compute_divergence_points_lsq(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute divergence at vertices using LSQ gradient of each component.

    For vector field v = [vₓ, vᵧ, vᵧ]:
        div(v) = ∂vₓ/∂x + ∂vᵧ/∂y + ∂vᵧ/∂z

    Computes gradient of each component, then takes trace.

    Args:
        mesh: Simplicial mesh
        vector_field: Vectors at vertices, shape (n_points, n_spatial_dims)

    Returns:
        Divergence at vertices, shape (n_points,)
    """
    from physicsnemo.mesh.calculus._lsq_reconstruction import compute_point_gradient_lsq

    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims

    ### Compute gradient of each component
    # For 3D: ∇vₓ, ∇vᵧ, ∇vᵧ
    # Each is (n_points, n_spatial_dims)

    divergence = torch.zeros(
        n_points, dtype=vector_field.dtype, device=mesh.points.device
    )

    for dim in range(n_spatial_dims):
        component = vector_field[:, dim]  # (n_points,)
        grad_component = compute_point_gradient_lsq(
            mesh, component
        )  # (n_points, n_spatial_dims)

        # Take diagonal: ∂v_dim/∂dim
        divergence += grad_component[:, dim]

    return divergence
