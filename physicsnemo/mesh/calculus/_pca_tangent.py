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

"""PCA-based tangent space estimation for manifolds.

For higher codimension manifolds (e.g., curves in 3D, surfaces in 4D+), normal
vectors are not uniquely defined. PCA on local neighborhoods provides a robust
method to estimate the tangent space.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def estimate_tangent_space_pca(
    mesh: "Mesh",
    k_neighbors: int | None = None,
) -> tuple[
    Float[torch.Tensor, "n_points n_manifold_dims n_spatial_dims"],
    Float[torch.Tensor, "n_points codimension n_spatial_dims"],
]:
    r"""Estimate tangent space at each point using PCA on local neighborhoods.

    For each point, gathers :math:`k` nearest neighbors and performs PCA on
    their relative positions. The eigenvectors with the largest eigenvalues
    span the tangent space; those with the smallest span the normal space.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    k_neighbors : int or None, optional
        Number of neighbors to use for PCA. If ``None``, uses
        ``min(2 * n_manifold_dims + 1, available_neighbors)``.

    Returns
    -------
    tuple[Float[torch.Tensor, "n_points n_manifold_dims n_spatial_dims"], Float[torch.Tensor, "n_points codimension n_spatial_dims"]]
        Tuple of ``(tangent_basis, normal_basis)``:

        - ``tangent_basis``: orthonormal basis vectors spanning the tangent
          space at each point, shape
          ``(n_points, n_manifold_dims, n_spatial_dims)``.
        - ``normal_basis``: orthonormal basis vectors spanning the normal
          space at each point, shape
          ``(n_points, codimension, n_spatial_dims)``.

    Notes
    -----
    Algorithm:

    1. For each point :math:`x`, gather :math:`k` nearest neighbors.
    2. Take their offsets :math:`x_i - x` from the point.
    3. Compute the second-moment matrix
       :math:`C = \tfrac{1}{k} \sum_i (x_i - x)(x_i - x)^\top`.
    4. Eigen-decompose: :math:`C = V \Lambda V^\top`.
    5. Sort eigenvectors by eigenvalue (descending).
    6. First ``n_manifold_dims`` eigenvectors span the tangent space.
    7. Remaining eigenvectors span the normal space.

    Derivatives of the bases are those of the tangent and normal spaces: each
    basis turns only into the other space, so the derivatives stay finite where
    eigenvalues within one space tie. Only first derivatives are exact.

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

    k_neighbors = max(k_neighbors, n_manifold_dims + 1)  # Use n+1 where available

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

    ### Identity fallback for points with insufficient neighbors
    # The offsets from the point are not centred, so n_manifold_dims neighbors
    # in general position span the tangent space
    min_required = max(n_manifold_dims, 1)
    neighbor_counts = adjacency.counts
    effective_counts = torch.minimum(
        neighbor_counts,
        torch.tensor(k_neighbors, dtype=neighbor_counts.dtype, device=device),
    )
    insufficient_mask = effective_counts < min_required
    if insufficient_mask.any():
        insufficient_indices = torch.where(insufficient_mask)[0]
        for i in range(min(n_manifold_dims, n_spatial_dims)):
            tangent_basis[insufficient_indices, i, i] = 1.0
        for i in range(min(codimension, n_spatial_dims - n_manifold_dims)):
            normal_basis[insufficient_indices, i, n_manifold_dims + i] = 1.0

    ### Process each neighbor-count group via shared iterator
    from physicsnemo.mesh.calculus._neighborhoods import iter_neighborhood_batches

    for batch in iter_neighborhood_batches(
        mesh.points, adjacency, min_neighbors=min_required, max_neighbors=k_neighbors
    ):
        point_indices = batch.entity_indices
        # (n_group, n_neighbors, n_spatial_dims); 16-bit in float32, which eigh needs
        offsets = batch.relative_positions.to(torch.promote_types(dtype, torch.float32))
        n_neighbors = batch.n_neighbors

        ### Covariance matrix: C = (1/k) X^T X
        cov_matrices = (
            torch.bmm(
                offsets.transpose(1, 2),  # (n_group, n_spatial_dims, n_neighbors)
                offsets,  # (n_group, n_neighbors, n_spatial_dims)
            )
            / n_neighbors
        )

        ### Batch eigen-decomposition, without derivatives (see below)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices.detach())

        ### Sort eigenvectors by eigenvalue (descending)
        eigenvalues, sorted_indices = eigenvalues.sort(dim=1, descending=True)
        sorted_idx_expanded = sorted_indices.unsqueeze(1).expand_as(eigenvectors)
        eigenvectors_sorted = torch.gather(
            eigenvectors, dim=2, index=sorted_idx_expanded
        )

        ### Extract tangent and normal bases
        tangent_vecs = eigenvectors_sorted[:, :, :n_manifold_dims]
        normal_vecs = eigenvectors_sorted[:, :, n_manifold_dims:]

        ### Derivatives of the tangent and normal spaces
        # eigh's backward divides by every eigenvalue gap, so it is NaN or wrong
        # where eigenvalues tie, such as the zero normal eigenvalues of a flat
        # neighborhood. The spaces depend only on the gaps between tangent and
        # normal eigenvalues: to first order, a change of the matrix turns each
        # space into the other by this rotation, which is zero in value (so only
        # first derivatives are exact). Where such a gap is zero, the tangent
        # space is undefined and gets no derivative.
        gaps = (
            eigenvalues[:, None, :n_manifold_dims]
            - eigenvalues[:, n_manifold_dims:, None]
        )  # (n_group, codimension, n_manifold_dims)
        d_cov = cov_matrices - cov_matrices.detach()
        rotation = (
            normal_vecs.mT @ d_cov @ tangent_vecs / gaps.where(gaps > 0, torch.inf)
        )
        tangent_vecs, normal_vecs = (
            tangent_vecs + normal_vecs @ rotation,
            normal_vecs - tangent_vecs @ rotation.mT,
        )

        tangent_basis[point_indices] = tangent_vecs.transpose(1, 2).to(dtype)
        normal_basis[point_indices] = normal_vecs.transpose(1, 2).to(dtype)

    return tangent_basis, normal_basis


def project_gradient_to_tangent_space_pca(
    mesh: "Mesh",
    gradients: Float[torch.Tensor, "n_points n_spatial_dims ..."],
    k_neighbors: int | None = None,
) -> Float[torch.Tensor, "n_points n_spatial_dims ..."]:
    """Project gradients onto PCA-estimated tangent space.

    For higher codimension manifolds, uses PCA to estimate tangent space
    and projects gradients accordingly.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.
    gradients : Float[torch.Tensor, "n_points n_spatial_dims ..."]
        Extrinsic gradients, shape ``(n_points, n_spatial_dims, ...)``.
    k_neighbors : int or None, optional
        Number of neighbors for PCA estimation.

    Returns
    -------
    Float[torch.Tensor, "n_points n_spatial_dims ..."]
        Intrinsic gradients projected onto the tangent space, same shape as
        ``gradients``.

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
