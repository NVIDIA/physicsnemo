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

"""PCA-based tangent space estimation for manifolds.

For higher codimension manifolds (e.g., curves in 3D, surfaces in 4D+), normal
vectors are not uniquely defined. PCA on local neighborhoods provides a robust
method to estimate the tangent space.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def estimate_tangent_space_pca(
    mesh: "Mesh",
    k_neighbors: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate tangent space at each point using PCA on local neighborhoods.

    For each point, gathers k-nearest neighbors and performs PCA on their
    relative positions. The eigenvectors corresponding to the largest eigenvalues
    span the tangent space, while those with smallest eigenvalues span the normal space.

    Parameters
    ----------
    mesh : Mesh
        Input mesh
    k_neighbors : int | None
        Number of neighbors to use for PCA. If None, uses
        min(2 * n_manifold_dims + 1, available_neighbors)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (tangent_basis, normal_basis) where:
        - tangent_basis: (n_points, n_manifold_dims, n_spatial_dims)
            Orthonormal basis vectors spanning tangent space at each point
        - normal_basis: (n_points, codimension, n_spatial_dims)
            Orthonormal basis vectors spanning normal space at each point

    Notes
    -----
    Algorithm:
        1. For each point, gather k nearest neighbors
        2. Center the neighborhood (subtract mean)
        3. Compute covariance matrix C = (1/k) Σ (x_i - mean)(x_i - mean)^T
        4. Eigen-decompose: C = V Λ V^T
        5. Sort eigenvectors by eigenvalue (descending)
        6. First n_manifold_dims eigenvectors span tangent space
        7. Remaining eigenvectors span normal space

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.curves import helix_3d
    >>> curve_mesh = helix_3d.load()
    >>> tangent_basis, normal_basis = estimate_tangent_space_pca(curve_mesh)
    >>> # tangent_basis: (n_points, 1, 3) - tangent direction
    >>> # normal_basis: (n_points, 2, 3) - normal plane basis
    """
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims
    n_manifold_dims = mesh.n_manifold_dims
    codimension = mesh.codimension
    device = mesh.points.device
    dtype = mesh.points.dtype

    ### Determine k_neighbors if not specified
    if k_neighbors is None:
        k_neighbors = min(2 * n_manifold_dims + 1, n_points - 1)

    k_neighbors = max(k_neighbors, n_manifold_dims + 1)  # Need at least n+1 neighbors

    ### Get point-to-point adjacency
    adjacency = mesh.get_point_to_points_adjacency()

    ### Initialize output tensors
    tangent_basis = torch.zeros(
        (n_points, n_manifold_dims, n_spatial_dims),
        dtype=dtype,
        device=device,
    )
    normal_basis = torch.zeros(
        (n_points, codimension, n_spatial_dims),
        dtype=dtype,
        device=device,
    )

    ### Compute neighbor counts per point
    neighbor_counts = adjacency.offsets[1:] - adjacency.offsets[:-1]  # (n_points,)

    ### Clamp to k_neighbors and group by effective neighbor count
    effective_counts = torch.minimum(
        neighbor_counts, torch.tensor(k_neighbors, dtype=torch.int64, device=device)
    )
    unique_counts, inverse_indices = torch.unique(effective_counts, return_inverse=True)

    ### Process each neighbor-count group in vectorized batches
    for count_idx, n_neighbors in enumerate(unique_counts):
        n_neighbors = int(n_neighbors)

        ### Skip if too few neighbors
        if n_neighbors < n_manifold_dims + 1:
            # Identity fallback for insufficient neighbors
            points_mask = inverse_indices == count_idx
            point_indices = torch.where(points_mask)[0]

            # Tangent basis: first n_manifold_dims standard basis vectors
            for i in range(min(n_manifold_dims, n_spatial_dims)):
                tangent_basis[point_indices, i, i] = 1.0

            # Normal basis: remaining standard basis vectors
            for i in range(min(codimension, n_spatial_dims - n_manifold_dims)):
                normal_basis[point_indices, i, n_manifold_dims + i] = 1.0

            continue

        ### Find all points with this neighbor count
        points_mask = inverse_indices == count_idx
        point_indices = torch.where(points_mask)[0]  # (n_group,)
        n_group = len(point_indices)

        if n_group == 0:
            continue

        ### Extract neighbor indices for this group (vectorized)
        # Shape: (n_group, n_neighbors)
        offsets_group = adjacency.offsets[point_indices]  # (n_group,)
        neighbor_idx_ranges = offsets_group.unsqueeze(1) + torch.arange(
            n_neighbors, device=device
        ).unsqueeze(0)  # (n_group, n_neighbors)
        neighbors_flat = adjacency.indices[
            neighbor_idx_ranges
        ]  # (n_group, n_neighbors)

        ### Gather neighborhood positions
        # Shape: (n_group, n_neighbors, n_spatial_dims)
        neighborhood_points = mesh.points[neighbors_flat]

        # Current point positions: (n_group, n_spatial_dims)
        center_points = mesh.points[point_indices]

        ### Center the neighborhoods
        # Shape: (n_group, n_neighbors, n_spatial_dims)
        centered = neighborhood_points - center_points.unsqueeze(1)

        ### Compute covariance matrices for all points in group
        # C = (1/k) X^T X where X is centered data
        # Use batch matrix multiplication: (n_group, n_spatial_dims, n_neighbors) @ (n_group, n_neighbors, n_spatial_dims)
        # Result: (n_group, n_spatial_dims, n_spatial_dims)
        cov_matrices = (
            torch.bmm(
                centered.transpose(1, 2),  # (n_group, n_spatial_dims, n_neighbors)
                centered,  # (n_group, n_neighbors, n_spatial_dims)
            )
            / n_neighbors
        )

        ### Batch eigen-decomposition
        # eigenvalues: (n_group, n_spatial_dims)
        # eigenvectors: (n_group, n_spatial_dims, n_spatial_dims)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices)

        ### Sort eigenvectors by eigenvalue (descending) for each point
        # Get sorting indices: (n_group, n_spatial_dims)
        sorted_indices = torch.argsort(eigenvalues, dim=1, descending=True)

        # Apply sorting to eigenvectors using gather
        # Expand indices for gathering: (n_group, n_spatial_dims, n_spatial_dims)
        sorted_idx_expanded = sorted_indices.unsqueeze(1).expand_as(eigenvectors)
        eigenvectors_sorted = torch.gather(
            eigenvectors, dim=2, index=sorted_idx_expanded
        )

        ### Extract tangent and normal bases
        # First n_manifold_dims eigenvectors span tangent space
        # eigenvectors_sorted: (n_group, n_spatial_dims, n_spatial_dims)
        #   where eigenvectors_sorted[i, :, j] is the j-th eigenvector for point i
        tangent_vecs = eigenvectors_sorted[
            :, :, :n_manifold_dims
        ]  # (n_group, n_spatial_dims, n_manifold_dims)
        tangent_basis[point_indices] = tangent_vecs.transpose(
            1, 2
        )  # (n_group, n_manifold_dims, n_spatial_dims)

        # Remaining eigenvectors span normal space
        normal_vecs = eigenvectors_sorted[
            :, :, n_manifold_dims:
        ]  # (n_group, n_spatial_dims, codimension)
        normal_basis[point_indices] = normal_vecs.transpose(
            1, 2
        )  # (n_group, codimension, n_spatial_dims)

    return tangent_basis, normal_basis


def project_gradient_to_tangent_space_pca(
    mesh: "Mesh",
    gradients: torch.Tensor,
    k_neighbors: int | None = None,
) -> torch.Tensor:
    """Project gradients onto PCA-estimated tangent space.

    For higher codimension manifolds, uses PCA to estimate tangent space
    and projects gradients accordingly.

    Parameters
    ----------
    mesh : Mesh
        Input mesh
    gradients : torch.Tensor
        Extrinsic gradients, shape (n_points, n_spatial_dims) or
        (n_points, n_spatial_dims, ...)
    k_neighbors : int | None
        Number of neighbors for PCA estimation

    Returns
    -------
    torch.Tensor
        Intrinsic gradients projected onto tangent space, same shape as input

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.primitives.curves import helix_3d
    >>> mesh = helix_3d.load()
    >>> gradients = torch.randn(mesh.n_points, mesh.n_spatial_dims)
    >>> grad_intrinsic = project_gradient_to_tangent_space_pca(mesh, gradients)
    """
    ### Estimate tangent space using PCA
    tangent_basis, _ = estimate_tangent_space_pca(mesh, k_neighbors)
    # tangent_basis: (n_points, n_manifold_dims, n_spatial_dims)

    ### Project gradient onto tangent space
    # For each point: grad_intrinsic = Σ_i (grad · t_i) t_i
    # where t_i are the tangent basis vectors

    if gradients.ndim == 2:
        ### Scalar gradient case: (n_points, n_spatial_dims)
        # Compute projection onto each tangent vector
        # grad · t_i for all i: (n_points, n_manifold_dims)
        projections = torch.einsum("ij,ikj->ik", gradients, tangent_basis)

        # Reconstruct in tangent space: Σ_i (grad · t_i) t_i
        grad_intrinsic = torch.einsum("ik,ikj->ij", projections, tangent_basis)

        return grad_intrinsic
    else:
        ### Tensor gradient case: (n_points, n_spatial_dims, ...)
        # More complex - need to handle extra dimensions

        # Compute projections: grad · t_i
        # Shape: (n_points, n_manifold_dims, ...)
        projections = torch.einsum("ij...,ikj->ik...", gradients, tangent_basis)

        # Reconstruct
        grad_intrinsic = torch.einsum("ik...,ikj->ij...", projections, tangent_basis)

        return grad_intrinsic
