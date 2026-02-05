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

    ### Group entities by neighbor count for efficient batched processing
    neighbor_counts = adjacency.offsets[1:] - adjacency.offsets[:-1]  # (n_entities,)
    unique_counts, inverse_indices = torch.unique(neighbor_counts, return_inverse=True)

    ### Process each neighbor-count group in parallel
    for count_idx, n_neighbors in enumerate(unique_counts):
        n_neighbors = int(n_neighbors)

        # Skip if too few neighbors or no neighbors
        if n_neighbors < min_neighbors or n_neighbors == 0:
            continue

        # Find all entities with this neighbor count
        entity_mask = inverse_indices == count_idx
        entity_indices = torch.where(entity_mask)[0]  # (n_group,)
        n_group = len(entity_indices)

        if n_group == 0:
            continue

        ### Extract neighbor indices for this group
        # Shape: (n_group, n_neighbors)
        offsets_group = adjacency.offsets[entity_indices]  # (n_group,)
        neighbor_idx_ranges = offsets_group.unsqueeze(1) + torch.arange(
            n_neighbors, device=device
        ).unsqueeze(0)  # (n_group, n_neighbors)
        neighbors_flat = adjacency.indices[
            neighbor_idx_ranges
        ]  # (n_group, n_neighbors)

        ### Build LSQ matrices for all entities in group
        # Current entity positions: (n_group, n_spatial_dims)
        x0 = positions[entity_indices]  # (n_group, n_spatial_dims)

        # Neighbor positions: (n_group, n_neighbors, n_spatial_dims)
        x_neighbors = positions[neighbors_flat]

        # Relative positions (A matrix): (n_group, n_neighbors, n_spatial_dims)
        A = x_neighbors - x0.unsqueeze(1)

        # Function differences (b vector)
        if is_scalar:
            # (n_group,) and (n_group, n_neighbors)
            b = values[neighbors_flat] - values[entity_indices].unsqueeze(1)
        else:
            # (n_group, extra_dims...) and (n_group, n_neighbors, extra_dims...)
            b = values[neighbors_flat] - values[entity_indices].unsqueeze(1)

        ### Compute weights
        distances = torch.norm(A, dim=-1)  # (n_group, n_neighbors)
        weights = 1.0 / distances.pow(weight_power).clamp(min=safe_eps(distances.dtype))

        ### Apply weights to system
        sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_group, n_neighbors, 1)
        A_weighted = sqrt_w * A  # (n_group, n_neighbors, n_spatial_dims)

        ### Solve batched least-squares
        try:
            if is_scalar:
                # b_weighted: (n_group, n_neighbors)
                b_weighted = sqrt_w.squeeze(-1) * b
                # Solve batched system
                solution = torch.linalg.lstsq(
                    A_weighted,  # (n_group, n_neighbors, n_spatial_dims)
                    b_weighted.unsqueeze(-1),  # (n_group, n_neighbors, 1)
                    rcond=None,
                ).solution.squeeze(-1)  # (n_group, n_spatial_dims)

                gradients[entity_indices] = solution
            else:
                # Tensor field case
                b_weighted = sqrt_w * b  # (n_group, n_neighbors, extra_dims...)
                orig_shape = b.shape[2:]  # Extra dimensions
                b_flat = b_weighted.reshape(
                    n_group, n_neighbors, -1
                )  # (n_group, n_neighbors, n_components)

                solution = torch.linalg.lstsq(
                    A_weighted,  # (n_group, n_neighbors, n_spatial_dims)
                    b_flat,  # (n_group, n_neighbors, n_components)
                    rcond=None,
                ).solution  # (n_group, n_spatial_dims, n_components)

                # Reshape and permute: (n_group, n_spatial_dims, *orig_shape)
                solution_reshaped = solution.reshape(
                    n_group, n_spatial_dims, *orig_shape
                )
                # Move spatial_dims to second position
                perm = [0] + list(range(2, solution_reshaped.ndim)) + [1]
                gradients[entity_indices] = solution_reshaped.permute(*perm)

        except torch.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "Singular linear system in LSQ gradient reconstruction. "
                "Affected vertices (e.g., isolated or collinear neighborhoods) "
                "will have zero gradients.",
                stacklevel=2,
            )

    return gradients


def compute_point_gradient_lsq(
    mesh: "Mesh",
    point_values: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 3,
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
        Minimum neighbors required for reliable gradient

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
