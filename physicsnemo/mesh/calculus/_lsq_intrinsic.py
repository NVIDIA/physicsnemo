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

        ### Process each neighbor-count group in parallel
        from physicsnemo.mesh.calculus._neighborhoods import iter_neighborhood_batches

        for batch in iter_neighborhood_batches(mesh.points, adjacency, min_neighbors=2):
            point_indices = batch.entity_indices
            neighbors_flat = batch.neighbor_indices
            A_ambient = (
                batch.relative_positions
            )  # (n_group, n_neighbors, n_spatial_dims)
            n_group = len(point_indices)
            n_neighbors = batch.n_neighbors

            ### Project into tangent space
            tangent_basis = tangent_bases[
                point_indices
            ]  # (n_group, n_spatial_dims, n_manifold_dims)
            A_tangent = torch.einsum("gns,gsm->gnm", A_ambient, tangent_basis)

            ### Function differences
            b = point_values[neighbors_flat] - point_values[point_indices].unsqueeze(1)

            ### Compute inverse-distance weights (using ambient distances)
            distances = torch.linalg.vector_norm(A_ambient, dim=-1)
            weights = 1.0 / distances.pow(weight_power).clamp(
                min=safe_eps(distances.dtype)
            )

            ### Apply sqrt-weights to tangent-space system
            sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_group, n_neighbors, 1)
            A_tangent_weighted = sqrt_w * A_tangent

            ### Solve batched least-squares in tangent space
            if is_scalar:
                b_weighted = sqrt_w.squeeze(-1) * b
                grad_tangent = torch.linalg.lstsq(
                    A_tangent_weighted, b_weighted.unsqueeze(-1), rcond=None
                ).solution.squeeze(-1)  # (n_group, n_manifold_dims)

                # Map back to ambient coordinates
                grad_ambient = torch.einsum("gsm,gm->gs", tangent_basis, grad_tangent)
                gradients[point_indices] = grad_ambient
            else:
                # Tensor field: flatten extra dims, solve, map back
                b_weighted = sqrt_w * b
                orig_shape = b.shape[2:]
                b_flat = b_weighted.reshape(n_group, n_neighbors, -1)

                grad_tangent = torch.linalg.lstsq(
                    A_tangent_weighted, b_flat, rcond=None
                ).solution  # (n_group, n_manifold_dims, n_components)

                grad_ambient = torch.bmm(tangent_basis, grad_tangent)
                grad_ambient_reshaped = grad_ambient.reshape(
                    n_group, n_spatial_dims, *orig_shape
                )
                perm = [0] + list(range(2, grad_ambient_reshaped.ndim)) + [1]
                gradients[point_indices] = grad_ambient_reshaped.permute(*perm)

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
    v1 = v1 / torch.linalg.vector_norm(v1, dim=-1, keepdim=True).clamp(
        min=safe_eps(v1.dtype)
    )

    if n_manifold_dims == 1:
        # 1D manifold (curves): single tangent vector
        return v1.unsqueeze(-1)  # (n_points, n_spatial_dims, 1)

    elif n_manifold_dims == 2:
        # 2D manifold (surfaces): two tangent vectors
        # Second tangent vector: v2 = n × v1
        if n_spatial_dims == 3:
            v2 = torch.linalg.cross(normals, v1)  # (n_points, 3)
            v2 = v2 / torch.linalg.vector_norm(v2, dim=-1, keepdim=True).clamp(
                min=safe_eps(v2.dtype)
            )
            return torch.stack([v1, v2], dim=-1)  # (n_points, 3, 2)
        else:
            raise ValueError(
                f"2D manifolds require 3D ambient space, got {n_spatial_dims=}"
            )

    else:
        raise NotImplementedError(
            f"Tangent basis construction for {n_manifold_dims=} not implemented"
        )
