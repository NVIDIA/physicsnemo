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

r"""Curl operator for 2D and 3D vector fields.

Implements curl using weighted least-squares reconstruction.

In 2D, curl maps vectors to the scalar out-of-plane component. In 3D, it
maps vectors to vectors.
"""

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_curl_points_lsq(
    mesh: "Mesh",
    vector_field: Float[torch.Tensor, "n_points n_spatial_dims"],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, " n_points"] | Float[torch.Tensor, "n_points 3"]:
    r"""Compute curl at vertices using LSQ gradient method.

    For a 2D vector field :math:`v = (v_x, v_y)`, the scalar out-of-plane
    curl is

    .. math::

        \operatorname{curl}(v) =
            \frac{\partial v_y}{\partial x}
            - \frac{\partial v_x}{\partial y}.

    For a 3D vector field :math:`v = (v_x, v_y, v_z)`,

    .. math::

        \operatorname{curl}(v) = \begin{pmatrix}
            \partial v_z / \partial y - \partial v_y / \partial z \\
            \partial v_x / \partial z - \partial v_z / \partial x \\
            \partial v_y / \partial x - \partial v_x / \partial y
        \end{pmatrix}.

    Computes the derivative-first Jacobian of the vector field, then takes its
    antisymmetric part. The Jacobian layout is
    ``jacobian[i, k, j] = ∂v_j/∂x_k``.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    vector_field : Float[torch.Tensor, "n_points n_spatial_dims"]
        Vectors at vertices on a 2D or 3D mesh.
    weight_power : float, optional
        Exponent for inverse-distance weighting.
    min_neighbors : int, optional
        Points with fewer than this many neighbors receive zero curl.
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    torch.Tensor
        Scalar curl with shape ``(n_points,)`` in 2D, or vector curl with
        shape ``(n_points, 3)`` in 3D.

    Raises
    ------
    ValueError
        If ``n_spatial_dims`` is not 2 or 3.
    """
    if mesh.n_spatial_dims not in (2, 3):
        raise ValueError(
            "Curl is only defined for 2D or 3D vector fields, "
            f"got {mesh.n_spatial_dims=}"
        )

    from physicsnemo.nn.functional.derivatives.mesh_lsq_curl import mesh_lsq_curl

    adjacency = mesh.get_point_to_points_adjacency()
    return mesh_lsq_curl(
        points=mesh.points,
        vector_field=vector_field,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation=implementation,
    )


def compute_curl_cells_lsq(
    mesh: "Mesh",
    vector_field: Float[torch.Tensor, "n_cells n_spatial_dims"],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, " n_cells"] | Float[torch.Tensor, "n_cells 3"]:
    r"""Compute curl at cell centers using LSQ gradient method.

    Cell-centered analogue of :func:`compute_curl_points_lsq`: computes the
    Jacobian of the vector field via the cell-neighbour LSQ gradient, then
    takes its antisymmetric part.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    vector_field : Float[torch.Tensor, "n_cells n_spatial_dims"]
        Vectors at cell centers on a 2D or 3D mesh.
    weight_power : float, optional
        Exponent for inverse-distance weighting.
    min_neighbors : int, optional
        Cells with fewer than this many neighbors receive zero curl.
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    torch.Tensor
        Scalar curl with shape ``(n_cells,)`` in 2D, or vector curl with
        shape ``(n_cells, 3)`` in 3D.

    Raises
    ------
    ValueError
        If ``n_spatial_dims`` is not 2 or 3.
    """
    if mesh.n_spatial_dims not in (2, 3):
        raise ValueError(
            "Curl is only defined for 2D or 3D vector fields, "
            f"got {mesh.n_spatial_dims=}"
        )

    from physicsnemo.nn.functional.derivatives.mesh_lsq_curl import mesh_lsq_curl

    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
    return mesh_lsq_curl(
        points=mesh.cell_centroids,
        vector_field=vector_field,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation=implementation,
    )
