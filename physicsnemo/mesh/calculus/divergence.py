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

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_divergence_points_dec(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute divergence at vertices using DEC: div = -δ♭.

    Uses the explicit formula from DEC paper for divergence of a dual vector field:

        div(X)(v₀) = (1/|⋆v₀|) Σ_{edges from v₀} |⋆e| × (X·edge_unit)

    where:
        - |⋆v₀| is the dual 0-cell volume (Voronoi area at vertex v₀)
        - |⋆e| is the dual 1-cell volume (dual edge length)
        - X·edge_unit is the flux component along the edge

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    vector_field : torch.Tensor
        Vectors at vertices, shape (n_points, n_spatial_dims)

    Returns
    -------
    torch.Tensor
        Divergence at vertices, shape (n_points,)
    """
    from physicsnemo.mesh.calculus._circumcentric_dual import (
        compute_dual_volumes_1,
        get_or_compute_dual_volumes_0,
    )

    n_points = mesh.n_points

    ### Get dual volumes
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)  # |⋆v₀| at vertices
    dual_volumes_1 = compute_dual_volumes_1(mesh)  # |⋆e| at edges

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
    edge_unit = edge_vectors / edge_lengths.unsqueeze(-1).clamp(min=safe_eps(edge_lengths.dtype))

    ### Compute divergence at each vertex
    divergence = torch.zeros(
        n_points, dtype=vector_field.dtype, device=mesh.points.device
    )

    ### Vectorized edge contributions
    v0_indices = sorted_edges[:, 0]  # (n_edges,)
    v1_indices = sorted_edges[:, 1]  # (n_edges,)

    # Vector field at edges (average of endpoints): (n_edges, n_spatial_dims)
    v_edge = (vector_field[v0_indices] + vector_field[v1_indices]) / 2

    # Flux through all edges: v·edge_direction (n_edges,)
    # This is the component of velocity along the edge direction
    flux_component = (v_edge * edge_unit).sum(dim=-1)

    # Weight by dual 1-cell volumes |⋆e| to get the actual flux through dual edge
    # Physically: flux = velocity_component × dual_edge_length
    weighted_flux = flux_component * dual_volumes_1

    # Scatter-add contributions with appropriate signs
    # v0: positive flux (outward from v0's dual cell)
    # v1: negative flux (inward to v1's dual cell)
    divergence.scatter_add_(0, v0_indices, weighted_flux)
    divergence.scatter_add_(0, v1_indices, -weighted_flux)

    ### Normalize by dual 0-cell volumes to get divergence per unit area
    divergence = divergence / dual_volumes_0.clamp(min=safe_eps(dual_volumes_0.dtype))

    return divergence


def compute_divergence_points_lsq(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute divergence at vertices using LSQ gradient of each component.

    For vector field v = [vₓ, vᵧ, vᵧ]:
        div(v) = ∂vₓ/∂x + ∂vᵧ/∂y + ∂vᵧ/∂z

    Computes gradient of each component, then takes trace.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    vector_field : torch.Tensor
        Vectors at vertices, shape (n_points, n_spatial_dims)

    Returns
    -------
    torch.Tensor
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
