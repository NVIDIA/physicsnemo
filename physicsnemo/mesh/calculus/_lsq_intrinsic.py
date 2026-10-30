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

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

from physicsnemo.mesh.utilities._tolerances import safe_eps
from physicsnemo.utils._small_linalg import (
    _lstsq_solver,
    _segment_lstsq_solver,
    _uses_normal_equations,
    batched_lstsq,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_point_gradient_lsq_intrinsic(
    mesh: "Mesh",
    point_values: Float[torch.Tensor, "n_points ..."],
    weight_power: float = 2.0,
) -> Float[torch.Tensor, "n_points n_spatial_dims ..."]:
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
        Shape: (n_points, n_spatial_dims) for scalars, or (n_points, n_spatial_dims, ...) for tensor fields.
        The derivative-coordinate axis is always axis 1.

    Notes
    -----
    Algorithm:
        For each point:
        1. Estimate tangent space using point normals
        2. Project neighbor positions onto tangent space
        3. Solve LSQ in tangent space (reduced dimension)
        4. Express result as vector in ambient space

    Implementation:
        Fully vectorized using batched operations. Assembles the tangent-space
        systems of all points at once where the normal equations solve them
        (float32 and 16-bit points); otherwise groups points by neighbor count
        and processes each group in parallel.
    """
    return _compute_point_gradients_lsq_intrinsic(mesh, [point_values], weight_power)[0]


def _compute_point_gradients_lsq_intrinsic(
    mesh: "Mesh",
    point_values: Sequence[Float[torch.Tensor, "n_points ..."]],
    weight_power: float = 2.0,
    *,
    grouped: bool = False,
) -> list[Float[torch.Tensor, "n_points n_spatial_dims ..."]]:
    """Intrinsic gradients of several fields, sharing their tangent-space systems.

    Each gradient is that of :func:`compute_point_gradient_lsq_intrinsic`, bit
    for bit. The fields share the tangent bases and the weighted systems and
    their factorization; each field is then solved on its own, so the peak
    memory is that of one field. The systems are assembled and solved at once
    where the normal equations solve them, as in
    :func:`~physicsnemo.nn.functional.derivatives.mesh_lsq_gradient._torch_impl._mesh_lsq_gradients_torch`;
    otherwise, or with ``grouped``, in groups of one neighbor count each.
    """
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims
    n_manifold_dims = mesh.n_manifold_dims
    device = mesh.points.device

    if mesh.codimension == 0:
        # No manifold structure: use standard LSQ
        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            _compute_point_gradients_lsq,
        )

        return _compute_point_gradients_lsq(mesh, point_values, weight_power)

    ### Get adjacency
    adjacency = mesh.get_point_to_points_adjacency()

    ### Determine output shapes
    def zero_gradients(i: int) -> torch.Tensor:
        """Zero gradients of field ``i``."""
        shape = (n_points, n_spatial_dims) + point_values[i].shape[1:]
        return torch.zeros(shape, dtype=point_values[i].dtype, device=device)

    ### Build per-point tangent-space bases, expressed in ambient coordinates with
    # shape (n_points, n_spatial_dims, n_manifold_dims): each column is a tangent vector.
    if mesh.codimension == 1:
        # Codimension-1: derive the tangent plane from the (unique) point normal.
        point_normals = mesh.point_normals  # (n_points, n_spatial_dims)
        tangent_bases = _build_tangent_bases_vectorized(
            point_normals, n_manifold_dims
        )  # (n_points, n_spatial_dims, n_manifold_dims)
    else:
        # Codimension >= 2 (e.g. a curve in 3D): normals are not unique, so estimate
        # the tangent space via local PCA. estimate_tangent_space_pca returns
        # (n_points, n_manifold_dims, n_spatial_dims); transpose to the column
        # convention used by the tangent-space solve below. (Without this branch the
        # zero-initialized `gradients` would be returned unchanged -- a silent
        # all-zero gradient for the default intrinsic path on codim>=2 meshes.)
        from physicsnemo.mesh.calculus._pca_tangent import estimate_tangent_space_pca

        pca_tangent_bases, _ = estimate_tangent_space_pca(mesh)
        tangent_bases = pca_tangent_bases.transpose(-2, -1).contiguous()

    dtype = torch.promote_types(mesh.points.dtype, tangent_bases.dtype)
    if not grouped and _uses_normal_equations(dtype, n_manifold_dims):
        ### Assemble every point's system at once, from one weighted row per edge
        # The rows are those of the groups below, by the same operations.
        n_edges = adjacency.indices.shape[0]
        sources = torch.repeat_interleave(adjacency.counts, output_size=n_edges)
        A_ambient = mesh.points[adjacency.indices]
        A_ambient -= mesh.points[sources]
        A_tangent = (A_ambient.unsqueeze(-1) * tangent_bases[sources]).sum(dim=-2)
        distances = torch.linalg.vector_norm(A_ambient, dim=-1)
        del A_ambient
        weights = 1.0 / distances.pow(weight_power).clamp(min=safe_eps(distances.dtype))
        sqrt_w = weights.sqrt()  # (n_edges,)
        del distances, weights
        solve = _segment_lstsq_solver(
            sqrt_w.unsqueeze(-1) * A_tangent, adjacency.offsets
        )
        del A_tangent
        # Points with fewer than 2 neighbors are not solved: zero gradients
        unsolved = adjacency.counts < 2

        gradients = []
        for field in point_values:
            b = field[adjacency.indices]
            b -= field[sources]
            b_weighted = sqrt_w.view(n_edges, *([1] * (b.ndim - 1))) * b
            del b
            grad_tangent = solve(b_weighted.reshape(n_edges, field.shape[1:].numel()))
            del b_weighted  # grad_tangent: (n_points, n_manifold_dims, n_components)

            ### Map back to ambient coordinates
            grad_ambient = (
                tangent_bases.unsqueeze(-1) * grad_tangent.unsqueeze(1)
            ).sum(dim=-2)
            del grad_tangent
            grad_ambient = grad_ambient.masked_fill(unsolved[:, None, None], 0.0)
            gradients.append(
                grad_ambient.reshape(n_points, n_spatial_dims, *field.shape[1:]).to(
                    field.dtype
                )
            )
            del grad_ambient
        return gradients

    # Several fields: each field's gradients are created before its first
    # solve, as one field at a time would
    several = len(point_values) > 1
    gradients = [None if several else zero_gradients(0) for _ in point_values]

    ### Process each neighbor-count group in parallel
    from physicsnemo.mesh.calculus._neighborhoods import iter_neighborhood_batches

    # Several fields: the group of the largest systems first, so that the peak
    # memory falls where only the gradients of the fields solved so far exist
    for batch in iter_neighborhood_batches(
        mesh.points, adjacency, min_neighbors=2, largest_first=several
    ):
        point_indices = batch.entity_indices
        neighbors_flat = batch.neighbor_indices
        A_ambient = batch.relative_positions  # (n_group, n_neighbors, n_spatial_dims)
        n_group = len(point_indices)
        n_neighbors = batch.n_neighbors

        ### Project into tangent space
        tangent_basis = tangent_bases[
            point_indices
        ]  # (n_group, n_spatial_dims, n_manifold_dims)
        # Broadcast-and-sum contractions: batched matmuls of these tiny matrices
        # (as einsum would dispatch) are much slower on GPUs.
        A_tangent = (A_ambient.unsqueeze(-1) * tangent_basis.unsqueeze(1)).sum(dim=-2)

        ### Compute inverse-distance weights (using ambient distances)
        distances = torch.linalg.vector_norm(A_ambient, dim=-1)
        weights = 1.0 / distances.pow(weight_power).clamp(min=safe_eps(distances.dtype))

        ### Apply sqrt-weights to tangent-space system
        sqrt_w = weights.sqrt().unsqueeze(-1)  # (n_group, n_neighbors, 1)
        A_tangent_weighted = sqrt_w * A_tangent
        # Only the weighted system is needed from here on
        del batch, A_ambient, A_tangent, distances, weights

        ### Factorize the tangent-space system once, for all fields
        if not several:
            solve = functools.partial(batched_lstsq, A_tangent_weighted)
        else:
            solve = _lstsq_solver(A_tangent_weighted)
        del A_tangent_weighted  # kept by the solver if it needs it

        for i, field in enumerate(point_values):
            ### Function differences
            b = field[neighbors_flat] - field[point_indices].unsqueeze(1)
            is_last = i == len(point_values) - 1
            if gradients[i] is None:
                gradients[i] = zero_gradients(i)
            field_gradients = gradients[i]

            ### Solve batched least-squares in tangent space
            if field.ndim == 1:
                b_weighted = sqrt_w.squeeze(-1) * b
                del b
                if is_last:
                    del sqrt_w
                grad_tangent = solve(b_weighted.unsqueeze(-1)).squeeze(
                    -1
                )  # (n_group, n_manifold_dims)
                del b_weighted

                # Map back to ambient coordinates
                grad_ambient = (tangent_basis * grad_tangent.unsqueeze(1)).sum(dim=-1)
                field_gradients[point_indices] = grad_ambient
                del grad_tangent, grad_ambient
            else:
                # Tensor field: flatten extra dims, solve, map back
                value_weights = sqrt_w.view(n_group, n_neighbors, *([1] * (b.ndim - 2)))
                b_weighted = value_weights * b
                orig_shape = b.shape[2:]
                b_flat = b_weighted.reshape(n_group, n_neighbors, -1)
                del b, value_weights, b_weighted
                if is_last:
                    del sqrt_w

                grad_tangent = solve(b_flat)  # (n_group, n_manifold_dims, n_components)
                del b_flat

                grad_ambient = (
                    tangent_basis.unsqueeze(-1) * grad_tangent.unsqueeze(1)
                ).sum(dim=-2)
                grad_ambient_reshaped = grad_ambient.reshape(
                    n_group, n_spatial_dims, *orig_shape
                )
                field_gradients[point_indices] = grad_ambient_reshaped
                del grad_tangent, grad_ambient, grad_ambient_reshaped
        del solve  # before the next group's

    # Zero for fields never solved
    return [
        zero_gradients(i) if gradients[i] is None else gradients[i]
        for i in range(len(point_values))
    ]


def _build_tangent_bases_vectorized(
    normals: Float[torch.Tensor, "n_points n_spatial_dims"],
    n_manifold_dims: int,
) -> Float[torch.Tensor, "n_points n_spatial_dims n_manifold_dims"]:
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
