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

r"""Gradient operators using both DEC and LSQ methods.

Provides gradient computation via:

1. DEC (Discrete Exterior Calculus):
   :math:`\operatorname{grad}(f) = \sharp(df)`, rigorous differential geometry.
2. LSQ (Least-Squares): weighted reconstruction, the standard CFD approach.

Both methods support intrinsic (tangent space) and extrinsic (ambient space)
gradients for manifolds embedded in higher-dimensional spaces.
"""

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_gradient_points_dec(
    mesh: "Mesh",
    point_values: Float[torch.Tensor, "n_points ..."],
) -> Float[torch.Tensor, "n_points n_spatial_dims ..."]:
    r"""Compute gradient at vertices using DEC: :math:`\operatorname{grad}(f) = \sharp(df)`.

    Steps:

    1. Apply the exterior derivative :math:`d_0` to get a 1-form on edges.
    2. Apply the sharp operator :math:`\sharp` to convert the 1-form to a
       vector field.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    point_values : Float[torch.Tensor, "n_points ..."]
        Values at vertices.

    Returns
    -------
    Float[torch.Tensor, "n_points n_spatial_dims ..."]
        Gradient vectors at vertices, shape
        ``(n_points, n_spatial_dims, ...)``.
    """
    from physicsnemo.mesh.calculus._exterior_derivative import exterior_derivative_0
    from physicsnemo.mesh.calculus._sharp_flat import sharp

    ### Step 1: Compute df (exterior derivative)
    edge_1form, edges = exterior_derivative_0(mesh, point_values)

    ### Step 2: Apply sharp to convert 1-form to vector field
    gradient_vectors = sharp(mesh, edge_1form, edges)

    return gradient_vectors


def compute_gradient_points_lsq(
    mesh: "Mesh",
    point_values: Float[torch.Tensor, "n_points ..."],
    weight_power: float = 2.0,
    intrinsic: bool = False,
) -> Float[torch.Tensor, "n_points n_spatial_dims ..."]:
    r"""Compute gradient at vertices using weighted least-squares.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    point_values : Float[torch.Tensor, "n_points ..."]
        Values at vertices.
    weight_power : float
        Exponent for inverse-distance weighting.
    intrinsic : bool
        If ``True`` and mesh is a manifold, solve LSQ in tangent space.

    Returns
    -------
    Float[torch.Tensor, "n_points n_spatial_dims ..."]
        Gradient vectors at vertices, shape
        ``(n_points, n_spatial_dims, ...)``.
    """
    if intrinsic and mesh.codimension > 0:
        # Use intrinsic LSQ (solves in tangent space)
        from physicsnemo.mesh.calculus._lsq_intrinsic import (
            compute_point_gradient_lsq_intrinsic,
        )

        return compute_point_gradient_lsq_intrinsic(mesh, point_values, weight_power)
    else:
        # Use standard ambient-space LSQ
        from physicsnemo.mesh.calculus._lsq_reconstruction import (
            compute_point_gradient_lsq,
        )

        return compute_point_gradient_lsq(mesh, point_values, weight_power)


def compute_gradient_cells_lsq(
    mesh: "Mesh",
    cell_values: Float[torch.Tensor, "n_cells ..."],
    weight_power: float = 2.0,
) -> Float[torch.Tensor, "n_cells n_spatial_dims ..."]:
    r"""Compute gradient at cells using weighted least-squares.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    cell_values : Float[torch.Tensor, "n_cells ..."]
        Values at cells.
    weight_power : float
        Exponent for inverse-distance weighting.

    Returns
    -------
    Float[torch.Tensor, "n_cells n_spatial_dims ..."]
        Gradient vectors at cells, shape
        ``(n_cells, n_spatial_dims, ...)``.
    """
    from physicsnemo.mesh.calculus._lsq_reconstruction import compute_cell_gradient_lsq

    return compute_cell_gradient_lsq(mesh, cell_values, weight_power)


def project_to_tangent_space(
    mesh: "Mesh",
    gradients: Float[torch.Tensor, "n n_spatial_dims ..."],
    location: Literal["points", "cells"],
) -> Float[torch.Tensor, "n n_spatial_dims ..."]:
    r"""Project gradients onto manifold tangent space for intrinsic derivatives.

    For manifolds where ``n_manifold_dims < n_spatial_dims`` (e.g. surfaces in
    3D), the intrinsic gradient lies in the tangent space of the manifold.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    gradients : Float[torch.Tensor, "n n_spatial_dims ..."]
        Extrinsic gradients.
    location : {"points", "cells"}
        Whether gradients are at points or cells.

    Returns
    -------
    Float[torch.Tensor, "n n_spatial_dims ..."]
        Intrinsic gradients (projected onto tangent space), same shape as input.

    Notes
    -----
    For codimension-1 manifolds:

    1. get the normal vector at each point/cell,
    2. project the gradient,
       :math:`\nabla_\text{int} f = \nabla f - (\nabla f \cdot \hat{n}) \, \hat{n}`.

    For higher codimension, PCA on a local neighborhood is used to estimate
    the tangent space.
    """
    if mesh.codimension == 0:
        # Manifold fills the space: intrinsic = extrinsic
        return gradients

    elif mesh.codimension == 1:
        ### Codimension-1: use normals for projection
        if location == "cells":
            # Use cell normals
            normals = mesh.cell_normals  # (n_cells, n_spatial_dims)
        else:
            # For points, use area-weighted averaged normals from adjacent cells
            # This is already computed and cached by mesh.point_normals
            normals = mesh.point_normals  # (n_points, n_spatial_dims)

        ### Project: grad_intrinsic = grad - (grad·n)n
        # grad·n contracts along the spatial dimension (dim=1 for gradients)
        if gradients.ndim == 2:
            # Scalar gradient: (n, n_spatial_dims)
            grad_dot_n = (gradients * normals).sum(dim=-1, keepdim=True)  # (n, 1)
            grad_intrinsic = gradients - grad_dot_n * normals  # (n, n_spatial_dims)
        else:
            # Tensor gradient: (n, n_spatial_dims, ...)
            # Contract along spatial dimension (dim=1)
            # normals is (n, n_spatial_dims), need to broadcast to match gradient shape

            # Expand normals to (n, n_spatial_dims, 1, 1, ...)
            normals_expanded = normals.view(
                normals.shape[0],  # n
                normals.shape[1],  # n_spatial_dims
                *([1] * (gradients.ndim - 2)),  # broadcast dimensions
            )

            # Dot product: sum over spatial dimension
            grad_dot_n = (gradients * normals_expanded).sum(
                dim=1, keepdim=True
            )  # (n, 1, ...)

            # Project out normal component
            grad_intrinsic = (
                gradients - grad_dot_n * normals_expanded
            )  # (n, n_spatial_dims, ...)

        return grad_intrinsic

    else:
        ### Higher codimension: use PCA to estimate tangent space
        from physicsnemo.mesh.calculus._pca_tangent import (
            project_gradient_to_tangent_space_pca,
        )

        return project_gradient_to_tangent_space_pca(mesh, gradients)
