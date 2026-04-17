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

"""Weighted least-squares gradient reconstruction for unstructured meshes.

This implements the standard CFD approach for computing gradients on irregular
meshes using weighted least-squares fitting.

The method solves for the gradient that best fits the function differences
to neighboring points/cells, weighted by inverse distance.

Reference: Standard in CFD literature (Barth & Jespersen, AIAA 1989)
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _to_mesh_gradient_layout(
    gradients: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Convert functional layout ``(n, dims, ...)`` to mesh layout ``(n, ..., dims)``."""
    if values.ndim == 1:
        return gradients
    perm = [0] + list(range(2, gradients.ndim)) + [1]
    return gradients.permute(*perm)


def compute_point_gradient_lsq(
    mesh: "Mesh",
    point_values: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 0,
) -> torch.Tensor:
    """Compute gradient at vertices using weighted least-squares reconstruction.

    For each vertex, solves:
        min_{∇φ} Σ_neighbors w_i ||∇φ·(x_i - x_0) - (φ_i - φ_0)||²

    Where weights w_i = 1/||x_i - x_0||^α (typically α=2).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    point_values : torch.Tensor
        Values at vertices, shape (n_points,) or (n_points, ...)
    weight_power : float
        Exponent for inverse distance weighting (default: 2.0)
    min_neighbors : int
        Minimum neighbors required for gradient computation. Points with
        fewer neighbors get zero gradients. The default of 0 means all
        points are processed: ``lstsq`` naturally returns the minimum-norm
        solution for under-determined systems (fewer neighbors than spatial
        dims) and zero for isolated points with no neighbors.

    Returns
    -------
    torch.Tensor
        Gradients at vertices, shape (n_points, n_spatial_dims) for scalars,
        or (n_points, n_spatial_dims, ...) for tensor fields

    Notes
    -----
    Algorithm:
        Solve weighted least-squares: (A^T W A) ∇φ = A^T W b
        where:
            A = [x₁-x₀, x₂-x₀, ...]^T  (n_neighbors × n_spatial_dims)
            b = [φ₁-φ₀, φ₂-φ₀, ...]^T  (n_neighbors,)
            W = diag([w₁, w₂, ...])     (n_neighbors × n_neighbors)

    Implementation:
        Fully vectorized using batched operations. Groups points by neighbor count
        and processes each group in parallel to handle ragged neighbor structure.
    """
    ### Get point-to-point adjacency
    adjacency = mesh.get_point_to_points_adjacency()

    ### Delegate LSQ solve to the functional API using the torch backend.
    from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient import (
        mesh_lsq_gradient,
    )

    gradients = mesh_lsq_gradient(
        points=mesh.points,
        values=point_values,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation="torch",
    )
    return _to_mesh_gradient_layout(gradients, point_values)


def compute_cell_gradient_lsq(
    mesh: "Mesh",
    cell_values: torch.Tensor,
    weight_power: float = 2.0,
) -> torch.Tensor:
    """Compute gradient at cells using weighted least-squares reconstruction.

    Uses cell-to-cell adjacency to build LSQ system around each cell centroid.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    cell_values : torch.Tensor
        Values at cells, shape (n_cells,) or (n_cells, ...)
    weight_power : float
        Exponent for inverse distance weighting (default: 2.0)

    Returns
    -------
    torch.Tensor
        Gradients at cells, shape (n_cells, n_spatial_dims) for scalars,
        or (n_cells, n_spatial_dims, ...) for tensor fields

    Notes
    -----
    Implementation:
        Fully vectorized using batched operations. Groups cells by neighbor count
        and processes each group in parallel.
    """
    ### Get cell-to-cell adjacency
    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)

    ### Get cell centroids
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)

    ### Delegate LSQ solve to the functional API using the torch backend.
    from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient import (
        mesh_lsq_gradient,
    )

    gradients = mesh_lsq_gradient(
        points=cell_centroids,
        values=cell_values,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=0,  # Cells may have fewer neighbors than points.
        implementation="torch",
    )
    return _to_mesh_gradient_layout(gradients, cell_values)
