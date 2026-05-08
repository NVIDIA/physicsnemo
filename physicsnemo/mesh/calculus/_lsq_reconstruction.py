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
from jaxtyping import Float

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
    point_values: Float[torch.Tensor, "n_points ..."],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
) -> Float[torch.Tensor, "n_points n_spatial_dims ..."]:
    r"""Compute gradient at vertices using weighted least-squares reconstruction.

    For each vertex with center :math:`x_0` and value :math:`\varphi_0`, and
    a neighborhood :math:`N = \{x_1, \ldots, x_k\}` with values
    :math:`\varphi_1, \ldots, \varphi_k`, solves

    .. math::

        \min_{\nabla \varphi} \sum_{i = 1}^{k}
            w_i \, \bigl\| \nabla \varphi \cdot (x_i - x_0)
                          - (\varphi_i - \varphi_0) \bigr\|^2,

    with weights :math:`w_i = 1 / \|x_i - x_0\|^{\alpha}` (typically
    :math:`\alpha = 2`).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    point_values : Float[torch.Tensor, "n_points ..."]
        Values at vertices.
    weight_power : float
        Exponent :math:`\alpha` for inverse-distance weighting (default 2.0).
    min_neighbors : int
        Minimum neighbors required for gradient computation. Points with
        fewer neighbors get zero gradients. The default of 0 means all
        points are processed: ``lstsq`` naturally returns the minimum-norm
        solution for under-determined systems (fewer neighbors than spatial
        dims) and zero for isolated points with no neighbors.

    Returns
    -------
    Float[torch.Tensor, "n_points n_spatial_dims ..."]
        Gradients at vertices, shape ``(n_points, n_spatial_dims, ...)``.

    Notes
    -----
    Solves the weighted least-squares system
    :math:`(A^\top W A) \, \nabla \varphi = A^\top W b`, where, with
    :math:`k` neighbors and :math:`d` spatial dimensions,

    .. math::

        A &= [x_1 - x_0, x_2 - x_0, \ldots, x_k - x_0]^\top
            \quad (k \times d), \\
        b &= [\varphi_1 - \varphi_0, \varphi_2 - \varphi_0, \ldots,
              \varphi_k - \varphi_0]^\top
            \quad (k,), \\
        W &= \operatorname{diag}(w_1, w_2, \ldots, w_k)
            \quad (k \times k).

    Here ``k`` is the number of neighbors of the point and ``d`` equals
    ``n_spatial_dims``. Fully vectorized using batched operations. Points are
    grouped by neighbor count and processed in parallel to handle the ragged
    neighbor structure.
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
    cell_values: Float[torch.Tensor, "n_cells ..."],
    weight_power: float = 2.0,
) -> Float[torch.Tensor, "n_cells n_spatial_dims ..."]:
    r"""Compute gradient at cells using weighted least-squares reconstruction.

    Uses cell-to-cell adjacency to build a LSQ system around each cell centroid.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    cell_values : Float[torch.Tensor, "n_cells ..."]
        Values at cells.
    weight_power : float
        Exponent for inverse-distance weighting (default 2.0).

    Returns
    -------
    Float[torch.Tensor, "n_cells n_spatial_dims ..."]
        Gradients at cells, shape ``(n_cells, n_spatial_dims, ...)``.

    Notes
    -----
    Fully vectorized using batched operations. Cells are grouped by neighbor
    count and processed in parallel.
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
