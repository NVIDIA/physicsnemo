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

"""Laplace-Beltrami operator for scalar fields.

The Laplace-Beltrami operator is the generalization of the Laplacian to
curved manifolds. In DEC: Δ = δd = -⋆d⋆d

For functions (0-forms), this gives the discrete Laplace-Beltrami operator
which reduces to the standard Laplacian on flat manifolds.

DEC formula (from Desbrun et al. lines 1689-1705):
    Δf(v₀) = -(1/|⋆v₀|) Σ_{edges from v₀} (|⋆e|/|e|)(f(v) - f(v₀))

This is the cotangent Laplacian, intrinsic to the manifold.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _apply_cotan_laplacian_operator(
    n_vertices: int,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    data: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply cotangent Laplacian operator to data via scatter-add.

    Computes: (L @ data)[i] = Σ_{j adjacent to i} w_ij * (data[j] - data[i])

    This is the core scatter-add pattern shared by all cotangent Laplacian computations.
    Used by both compute_laplacian_points_dec() for scalar fields and
    compute_laplacian_at_points() in curvature module for point coordinates.

    Args:
        n_vertices: Number of vertices
        edges: Edge connectivity, shape (n_edges, 2)
        cotan_weights: Cotangent weights for each edge, shape (n_edges,)
        data: Data at vertices, shape (n_vertices, *data_shape)
        device: Device for computation

    Returns:
        Laplacian applied to data, shape (n_vertices, *data_shape)

    Example:
        >>> # For scalar field
        >>> laplacian = _apply_cotan_laplacian_operator(n_points, edges, weights, scalar_field, device)
        >>> # For vector field (point coordinates)
        >>> laplacian = _apply_cotan_laplacian_operator(n_points, edges, weights, points, device)
    """
    ### Initialize output with same shape as data
    if data.ndim == 1:
        laplacian = torch.zeros(n_vertices, dtype=data.dtype, device=device)
    else:
        laplacian = torch.zeros_like(data)

    ### Extract vertex indices
    v0_indices = edges[:, 0]  # (n_edges,)
    v1_indices = edges[:, 1]  # (n_edges,)

    ### Compute weighted differences
    if data.ndim == 1:
        # Scalar case
        contrib_v0 = cotan_weights * (data[v1_indices] - data[v0_indices])
        contrib_v1 = cotan_weights * (data[v0_indices] - data[v1_indices])
        laplacian.scatter_add_(0, v0_indices, contrib_v0)
        laplacian.scatter_add_(0, v1_indices, contrib_v1)
    else:
        # Multi-dimensional case (vectors, tensors)
        # Broadcast weights to match data dimensions
        weights_expanded = cotan_weights.view(-1, *([1] * (data.ndim - 1)))
        contrib_v0 = weights_expanded * (data[v1_indices] - data[v0_indices])
        contrib_v1 = weights_expanded * (data[v0_indices] - data[v1_indices])

        # Flatten for scatter_add
        laplacian_flat = laplacian.reshape(n_vertices, -1)
        contrib_v0_flat = contrib_v0.reshape(len(edges), -1)
        contrib_v1_flat = contrib_v1.reshape(len(edges), -1)

        v0_expanded = v0_indices.unsqueeze(-1).expand(-1, contrib_v0_flat.shape[1])
        v1_expanded = v1_indices.unsqueeze(-1).expand(-1, contrib_v1_flat.shape[1])

        laplacian_flat.scatter_add_(0, v0_expanded, contrib_v0_flat)
        laplacian_flat.scatter_add_(0, v1_expanded, contrib_v1_flat)

        laplacian = laplacian_flat.reshape(laplacian.shape)

    return laplacian


def compute_laplacian_points_dec(
    mesh: "Mesh",
    point_values: torch.Tensor,
) -> torch.Tensor:
    """Compute Laplace-Beltrami at vertices using DEC cotangent formula.

    This is the INTRINSIC Laplacian - it automatically respects the manifold structure.

    Formula: Δf(v₀) = -(1/|⋆v₀|) Σ_{edges from v₀} (|⋆e|/|e|)(f(v) - f(v₀))

    Where:
    - |⋆v₀| is the dual 0-cell volume (Voronoi cell around vertex)
    - |⋆e| is the dual 1-cell volume (dual to edge)
    - |e| is the edge length
    - The ratio |⋆e|/|e| are the cotangent weights

    Args:
        mesh: Simplicial mesh
        point_values: Values at vertices, shape (n_points,) or (n_points, ...)

    Returns:
        Laplacian at vertices, same shape as input
    """
    from physicsnemo.mesh.calculus._circumcentric_dual import (
        compute_cotan_weights_triangle_mesh,
        get_or_compute_dual_volumes_0,
    )

    n_points = mesh.n_points
    device = mesh.points.device

    ### Validate manifold dimension
    if mesh.n_manifold_dims != 2:
        raise NotImplementedError(
            f"DEC Laplace-Beltrami currently only implemented for triangle meshes (2D manifolds). "
            f"Got {mesh.n_manifold_dims=}. Use LSQ-based Laplacian via div(grad(.)) instead."
        )

    ### Get cotangent weights and edges (uses standard formula with factor of 1/2)
    cotan_weights, sorted_edges = compute_cotan_weights_triangle_mesh(
        mesh, return_edges=True
    )

    ### Apply cotangent Laplacian operator using shared utility
    laplacian = _apply_cotan_laplacian_operator(
        n_vertices=n_points,
        edges=sorted_edges,
        cotan_weights=cotan_weights,
        data=point_values,
        device=device,
    )

    ### Normalize by Voronoi areas
    # Standard cotangent Laplacian: Δf_i = (1/A_voronoi_i) × accumulated_sum
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)

    if point_values.ndim == 1:
        laplacian = laplacian / dual_volumes_0.clamp(min=1e-10)
    else:
        laplacian = laplacian / dual_volumes_0.view(
            -1, *([1] * (point_values.ndim - 1))
        ).clamp(min=1e-10)

    return laplacian


def compute_laplacian_points(
    mesh: "Mesh",
    point_values: torch.Tensor,
) -> torch.Tensor:
    """Compute Laplace-Beltrami at vertices using DEC.

    This is a convenience wrapper for compute_laplacian_points_dec.

    Args:
        mesh: Simplicial mesh
        point_values: Values at vertices

    Returns:
        Laplacian at vertices
    """
    return compute_laplacian_points_dec(mesh, point_values)
