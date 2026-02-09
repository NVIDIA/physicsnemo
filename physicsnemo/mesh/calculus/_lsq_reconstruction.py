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

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _solve_batched_lsq_gradients(
    positions: torch.Tensor,  # shape: (n_entities, n_spatial_dims)
    values: torch.Tensor,  # shape: (n_entities, ...)
    adjacency,  # Adjacency object
    weight_power: float,
    min_neighbors: int = 0,
) -> torch.Tensor:
    """Core batched LSQ gradient solver (shared by point and cell versions).

    For each entity (point or cell), solves a weighted least-squares problem:
        min_{∇φ} Σ_neighbors w_i ||∇φ·(x_i - x_0) - (φ_i - φ_0)||²

    Parameters
    ----------
    positions : torch.Tensor
        Entity positions (points or cell centroids)
    values : torch.Tensor
        Values at entities (scalars or tensor fields)
    adjacency
        Adjacency structure (entity-to-entity neighbors)
    weight_power : float
        Exponent for inverse distance weighting
    min_neighbors : int
        Minimum neighbors required for gradient computation

    Returns
    -------
    torch.Tensor
        Gradients at entities, shape (n_entities, n_spatial_dims) for scalars,
        or (n_entities, n_spatial_dims, ...) for tensor fields.
        Entities with insufficient neighbors have zero gradients.
    """
    n_entities = len(positions)
    n_spatial_dims = positions.shape[1]
    device = positions.device
    dtype = values.dtype

    ### Determine output shape
    is_scalar = values.ndim == 1
    if is_scalar:
        gradient_shape = (n_entities, n_spatial_dims)
    else:
        gradient_shape = (n_entities, n_spatial_dims) + values.shape[1:]

    gradients = torch.zeros(gradient_shape, dtype=dtype, device=device)

    ### Process each neighbor-count group in parallel
    from physicsnemo.mesh.calculus._neighborhoods import iter_neighborhood_batches

    for batch in iter_neighborhood_batches(
        positions, adjacency, min_neighbors=min_neighbors
    ):
        entity_indices = batch.entity_indices
        neighbors_flat = batch.neighbor_indices
        A = batch.relative_positions  # (n_group, n_neighbors, n_spatial_dims)
        n_group = len(entity_indices)
        n_neighbors = batch.n_neighbors

        ### Entities with no neighbors retain their zero-initialized gradient
        if n_neighbors == 0:
            continue

        ### Function differences (b vector)
        b = values[neighbors_flat] - values[entity_indices].unsqueeze(1)

        ### Compute inverse-distance weights
        distances = torch.linalg.vector_norm(A, dim=-1)  # (n_group, n_neighbors)
        weights = 1.0 / distances.pow(weight_power).clamp(min=safe_eps(distances.dtype))

        ### Apply sqrt-weights to system
        sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_group, n_neighbors, 1)
        A_weighted = sqrt_w * A  # (n_group, n_neighbors, n_spatial_dims)

        ### Solve batched least-squares
        if is_scalar:
            b_weighted = sqrt_w.squeeze(-1) * b  # (n_group, n_neighbors)
            solution = torch.linalg.lstsq(
                A_weighted, b_weighted.unsqueeze(-1), rcond=None
            ).solution.squeeze(-1)  # (n_group, n_spatial_dims)

            gradients[entity_indices] = solution
        else:
            # Tensor field: flatten extra dims, solve, reshape back
            b_weighted = sqrt_w * b  # (n_group, n_neighbors, ...)
            orig_shape = b.shape[2:]
            b_flat = b_weighted.reshape(n_group, n_neighbors, -1)

            solution = torch.linalg.lstsq(
                A_weighted, b_flat, rcond=None
            ).solution  # (n_group, n_spatial_dims, n_components)

            solution_reshaped = solution.reshape(n_group, n_spatial_dims, *orig_shape)
            # Permute spatial_dims to second position
            perm = [0] + list(range(2, solution_reshaped.ndim)) + [1]
            gradients[entity_indices] = solution_reshaped.permute(*perm)

    return gradients


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

    ### Use shared batched LSQ solver
    return _solve_batched_lsq_gradients(
        positions=mesh.points,
        values=point_values,
        adjacency=adjacency,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
    )


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

    ### Use shared batched LSQ solver
    return _solve_batched_lsq_gradients(
        positions=cell_centroids,
        values=cell_values,
        adjacency=adjacency,
        weight_power=weight_power,
        min_neighbors=0,  # Cells may have fewer neighbors than points
    )
