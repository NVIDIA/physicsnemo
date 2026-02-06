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

"""Intrinsic LSQ gradient reconstruction on manifolds.

For manifolds embedded in higher dimensions, solves LSQ in the local tangent space
rather than solving in ambient space and projecting. This avoids ill-conditioning.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_point_gradient_lsq_intrinsic(
    mesh: "Mesh",
    point_values: torch.Tensor,
    weight_power: float = 2.0,
) -> torch.Tensor:
    """Compute intrinsic gradient on manifold using tangent-space LSQ.

    For surfaces in 3D, solves LSQ in the local 2D tangent plane at each vertex.
    This avoids the ill-conditioning that occurs when solving in full ambient space.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh (assumed to be a manifold)
    point_values : torch.Tensor
        Values at vertices, shape (n_points,) or (n_points, ...)
    weight_power : float
        Exponent for inverse distance weighting (default: 2.0)

    Returns
    -------
    torch.Tensor
        Intrinsic gradients (living in tangent space, represented in ambient coordinates).
        Shape: (n_points, n_spatial_dims) for scalars, or (n_points, n_spatial_dims, ...) for tensor fields

    Notes
    -----
    Algorithm:
        For each point:
        1. Estimate tangent space using point normals
        2. Project neighbor positions onto tangent space
        3. Solve LSQ in tangent space (reduced dimension)
        4. Express result as vector in ambient space

    Implementation:
        Fully vectorized using batched operations. Groups points by neighbor count
        and processes each group in parallel.
    """
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims
    n_manifold_dims = mesh.n_manifold_dims
    device = mesh.points.device
    dtype = point_values.dtype

    if mesh.codimension == 0:
        # No manifold structure: use standard LSQ
        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_point_gradient_lsq,
        )

        return compute_point_gradient_lsq(mesh, point_values, weight_power)

    ### Get adjacency
    adjacency = mesh.get_point_to_points_adjacency()

    ### Determine output shape
    is_scalar = point_values.ndim == 1
    if is_scalar:
        gradient_shape = (n_points, n_spatial_dims)
    else:
        gradient_shape = (n_points, n_spatial_dims) + point_values.shape[1:]

    gradients = torch.zeros(gradient_shape, dtype=dtype, device=device)

    ### Build tangent space basis for all points (vectorized)
    # For codim-1: use point normals and construct orthogonal basis
    if mesh.codimension == 1:
        # Get point normals (already vectorized and cached)
        point_normals = mesh.point_normals  # (n_points, n_spatial_dims)

        # Build tangent basis for all points at once
        tangent_bases = _build_tangent_bases_vectorized(
            point_normals, n_manifold_dims
        )  # (n_points, n_spatial_dims, n_manifold_dims)

        ### Group points by neighbor count for efficient batched processing
        neighbor_counts = adjacency.offsets[1:] - adjacency.offsets[:-1]  # (n_points,)
        unique_counts, inverse_indices = torch.unique(
            neighbor_counts, return_inverse=True
        )

        ### Process each neighbor-count group in parallel
        for count_idx, n_neighbors in enumerate(unique_counts):
            n_neighbors = int(n_neighbors)

            # Skip if too few neighbors
            if n_neighbors < 2:
                continue

            # Find all points with this neighbor count
            points_mask = inverse_indices == count_idx
            point_indices = torch.where(points_mask)[0]  # (n_group,)
            n_group = len(point_indices)

            if n_group == 0:
                continue

            ### Extract neighbor indices for this group
            # Shape: (n_group, n_neighbors)
            offsets_group = adjacency.offsets[point_indices]  # (n_group,)
            neighbor_idx_ranges = offsets_group.unsqueeze(1) + torch.arange(
                n_neighbors, device=device
            ).unsqueeze(0)  # (n_group, n_neighbors)
            neighbors_flat = adjacency.indices[
                neighbor_idx_ranges
            ]  # (n_group, n_neighbors)

            ### Build LSQ matrices in ambient space
            # Current point positions: (n_group, n_spatial_dims)
            x0 = mesh.points[point_indices]  # (n_group, n_spatial_dims)

            # Neighbor positions: (n_group, n_neighbors, n_spatial_dims)
            x_neighbors = mesh.points[neighbors_flat]

            # Relative positions (A matrix): (n_group, n_neighbors, n_spatial_dims)
            A_ambient = x_neighbors - x0.unsqueeze(1)

            ### Project LSQ system into tangent space
            # Tangent bases for this group: (n_group, n_spatial_dims, n_manifold_dims)
            tangent_basis = tangent_bases[point_indices]

            # Project A into tangent space: A_tangent = A_ambient @ tangent_basis
            # For each group element: A_ambient[i, :, :] @ tangent_basis[i, :, :]
            # (n_group, n_neighbors, n_spatial_dims) @ (n_group, n_spatial_dims, n_manifold_dims)
            # = (n_group, n_neighbors, n_manifold_dims)
            A_tangent = torch.einsum("gns,gsm->gnm", A_ambient, tangent_basis)

            # Function differences
            if is_scalar:
                b = point_values[neighbors_flat] - point_values[
                    point_indices
                ].unsqueeze(1)  # (n_group, n_neighbors)
            else:
                b = point_values[neighbors_flat] - point_values[
                    point_indices
                ].unsqueeze(1)  # (n_group, n_neighbors, ...)

            ### Compute weights (based on ambient distances)
            distances = torch.norm(A_ambient, dim=-1)  # (n_group, n_neighbors)
            weights = 1.0 / distances.pow(weight_power).clamp(
                min=safe_eps(distances.dtype)
            )

            ### Apply weights to tangent-space system
            sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_group, n_neighbors, 1)
            A_tangent_weighted = (
                sqrt_w * A_tangent
            )  # (n_group, n_neighbors, n_manifold_dims)

            ### Solve batched least-squares in tangent space
            try:
                if is_scalar:
                    b_weighted = sqrt_w.squeeze(-1) * b  # (n_group, n_neighbors)
                    # Solve for gradient in tangent coordinates
                    grad_tangent = torch.linalg.lstsq(
                        A_tangent_weighted,  # (n_group, n_neighbors, n_manifold_dims)
                        b_weighted.unsqueeze(-1),  # (n_group, n_neighbors, 1)
                        rcond=None,
                    ).solution.squeeze(-1)  # (n_group, n_manifold_dims)

                    # Convert back to ambient coordinates
                    # grad_ambient = tangent_basis @ grad_tangent
                    # (n_group, n_spatial_dims, n_manifold_dims) @ (n_group, n_manifold_dims)
                    grad_ambient = torch.einsum(
                        "gsm,gm->gs", tangent_basis, grad_tangent
                    )  # (n_group, n_spatial_dims)

                    gradients[point_indices] = grad_ambient
                else:
                    # Tensor field case
                    b_weighted = sqrt_w * b  # (n_group, n_neighbors, ...)
                    orig_shape = b.shape[2:]  # Extra dimensions
                    b_flat = b_weighted.reshape(
                        n_group, n_neighbors, -1
                    )  # (n_group, n_neighbors, n_components)

                    grad_tangent = torch.linalg.lstsq(
                        A_tangent_weighted,  # (n_group, n_neighbors, n_manifold_dims)
                        b_flat,  # (n_group, n_neighbors, n_components)
                        rcond=None,
                    ).solution  # (n_group, n_manifold_dims, n_components)

                    # Convert to ambient: (n_group, n_spatial_dims, n_manifold_dims) @ (n_group, n_manifold_dims, n_components)
                    grad_ambient = torch.bmm(
                        tangent_basis,  # (n_group, n_spatial_dims, n_manifold_dims)
                        grad_tangent,  # (n_group, n_manifold_dims, n_components)
                    )  # (n_group, n_spatial_dims, n_components)

                    # Reshape: (n_group, n_spatial_dims, *orig_shape)
                    grad_ambient_reshaped = grad_ambient.reshape(
                        n_group, n_spatial_dims, *orig_shape
                    )
                    # Move spatial_dims to second position: (n_group, *orig_shape, n_spatial_dims)
                    perm = [0] + list(range(2, grad_ambient_reshaped.ndim)) + [1]
                    gradients[point_indices] = grad_ambient_reshaped.permute(*perm)

            except torch.linalg.LinAlgError:
                import warnings

                warnings.warn(
                    "Singular linear system in intrinsic LSQ gradient reconstruction. "
                    "Affected vertices (e.g., isolated or collinear neighborhoods) "
                    "will have zero gradients.",
                    stacklevel=2,
                )

    return gradients


def _build_tangent_bases_vectorized(
    normals: torch.Tensor,
    n_manifold_dims: int,
) -> torch.Tensor:
    """Build orthonormal tangent space bases from normal vectors (vectorized).

    Parameters
    ----------
    normals : torch.Tensor
        Unit normal vectors, shape (n_points, n_spatial_dims)
    n_manifold_dims : int
        Dimension of the manifold

    Returns
    -------
    torch.Tensor
        Tangent bases, shape (n_points, n_spatial_dims, n_manifold_dims)
        where tangent_bases[i, :, :] contains n_manifold_dims orthonormal tangent vectors
        as columns

    Notes
    -----
    Algorithm:
        Uses Gram-Schmidt to construct orthonormal basis from arbitrary starting vectors.
    """
    n_points, n_spatial_dims = normals.shape
    device = normals.device
    dtype = normals.dtype

    ### Start with arbitrary vectors not parallel to normals
    # Use standard basis vector least aligned with normal
    # For each point, choose e_i where |normal · e_i| is smallest
    standard_basis = torch.eye(
        n_spatial_dims, device=device, dtype=dtype
    )  # (n_spatial_dims, n_spatial_dims)

    # Compute |normal · e_i| for all i: (n_points, n_spatial_dims)
    alignment = torch.abs(normals @ standard_basis)  # (n_points, n_spatial_dims)

    # Choose least-aligned basis vector for each point
    least_aligned_idx = torch.argmin(alignment, dim=-1)  # (n_points,)
    v1 = standard_basis[least_aligned_idx]  # (n_points, n_spatial_dims)

    ### Project v1 onto tangent plane: v1 = v1 - (v1·n)n
    v1_dot_n = (v1 * normals).sum(dim=-1, keepdim=True)  # (n_points, 1)
    v1 = v1 - v1_dot_n * normals  # (n_points, n_spatial_dims)
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True).clamp(min=safe_eps(v1.dtype))

    if n_manifold_dims == 1:
        # 1D manifold (curves): single tangent vector
        return v1.unsqueeze(-1)  # (n_points, n_spatial_dims, 1)

    elif n_manifold_dims == 2:
        # 2D manifold (surfaces): two tangent vectors
        # Second tangent vector: v2 = n × v1
        if n_spatial_dims == 3:
            v2 = torch.linalg.cross(normals, v1)  # (n_points, 3)
            v2 = v2 / torch.norm(v2, dim=-1, keepdim=True).clamp(min=safe_eps(v2.dtype))
            return torch.stack([v1, v2], dim=-1)  # (n_points, 3, 2)
        else:
            raise ValueError(
                f"2D manifolds require 3D ambient space, got {n_spatial_dims=}"
            )

    else:
        raise NotImplementedError(
            f"Tangent basis construction for {n_manifold_dims=} not implemented"
        )
