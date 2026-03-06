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

"""Laplace-Beltrami operator for scalar fields.

The Laplace-Beltrami operator is the generalization of the Laplacian to
curved manifolds.

This implementation uses the analyst's sign convention:
    Δf(v₀) = (1/|⋆v₀|) Σ_{edges from v₀} (|⋆e|/|e|)(f(v) - f(v₀))

which is positive for locally convex functions (e.g., Δ(x²) = 2).

For functions (0-forms), this gives the discrete Laplace-Beltrami operator
which reduces to the standard Laplacian on flat manifolds.

This is the cotangent Laplacian, intrinsic to the manifold.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _apply_cotan_laplacian_operator(
    n_vertices: int,
    edges: torch.Tensor,
    cotan_weights: torch.Tensor,
    data: torch.Tensor,
) -> torch.Tensor:
    """Apply cotangent Laplacian operator to data via scatter-add.

    Computes: (L @ data)[i] = Σ_{j adjacent to i} w_ij * (data[j] - data[i])

    This is the core scatter-add pattern shared by all cotangent Laplacian computations.
    Used by both compute_laplacian_points_dec() for scalar fields and
    compute_laplacian_at_points() in curvature module for point coordinates.

    Parameters
    ----------
    n_vertices : int
        Number of vertices
    edges : torch.Tensor
        Edge connectivity, shape (n_edges, 2)
    cotan_weights : torch.Tensor
        Cotangent weights for each edge, shape (n_edges,)
    data : torch.Tensor
        Data at vertices, shape (n_vertices, *data_shape)

    Returns
    -------
    torch.Tensor
        Laplacian applied to data, shape (n_vertices, *data_shape)

    Examples
    --------
    >>> import torch
    >>> # For scalar field
    >>> n_points, edges = 4, torch.tensor([[0, 1], [1, 2], [0, 2]])
    >>> weights = torch.ones(3)
    >>> scalar_field = torch.randn(4)
    >>> laplacian = _apply_cotan_laplacian_operator(n_points, edges, weights, scalar_field)
    """
    ### Initialize output with same shape as data
    device = data.device
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

    Formula: Δf(v₀) = (1/|⋆v₀|) Σ_{edges from v₀} (|⋆e|/|e|)(f(v) - f(v₀))

    Where:
    - |⋆v₀| is the dual 0-cell volume (Voronoi cell around vertex)
    - |⋆e| is the dual 1-cell volume (dual to edge)
    - |e| is the edge length
    - The ratio |⋆e|/|e| are the cotangent weights

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    point_values : torch.Tensor
        Values at vertices, shape (n_points,) or (n_points, ...)

    Returns
    -------
    torch.Tensor
        Laplacian at vertices, same shape as input
    """
    from physicsnemo.mesh.geometry.dual_meshes import (
        compute_cotan_weights_fem,
        get_or_compute_dual_volumes_0,
    )

    ### Get cotangent weights and edges via FEM stiffness matrix (works for any dimension)
    cotan_weights, sorted_edges = compute_cotan_weights_fem(mesh)

    ### Apply cotangent Laplacian operator using shared utility
    laplacian = _apply_cotan_laplacian_operator(
        n_vertices=mesh.n_points,
        edges=sorted_edges,
        cotan_weights=cotan_weights,
        data=point_values,
    )

    ### Normalize by Voronoi areas
    # Standard cotangent Laplacian: Δf_i = (1/A_voronoi_i) × accumulated_sum
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)

    if point_values.ndim == 1:
        laplacian = laplacian / dual_volumes_0.clamp(min=safe_eps(dual_volumes_0.dtype))
    else:
        laplacian = laplacian / dual_volumes_0.view(
            -1, *([1] * (point_values.ndim - 1))
        ).clamp(min=safe_eps(dual_volumes_0.dtype))

    return laplacian
