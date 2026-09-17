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

r"""Curl operator for vector fields (3D only).

Implements curl using both DEC and LSQ methods.

DEC formula: :math:`\operatorname{curl} = \star\, d\, \flat`,

1. apply flat :math:`\flat` to convert vector field to 1-form,
2. apply exterior derivative :math:`d` to get 2-form,
3. apply Hodge star :math:`\star` to get dual 1-form,
4. convert back to vector field.

For 3D: curl maps vectors to vectors.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_curl_points_lsq(
    mesh: "Mesh",
    vector_field: Float[torch.Tensor, "n_points 3"],
) -> Float[torch.Tensor, "n_points 3"]:
    r"""Compute curl at vertices using LSQ gradient method.

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
    vector_field : Float[torch.Tensor, "n_points 3"]
        Vectors at vertices.

    Returns
    -------
    Float[torch.Tensor, "n_points 3"]
        Curl at vertices.

    Raises
    ------
    ValueError
        If ``n_spatial_dims != 3``.
    """
    if mesh.n_spatial_dims != 3:
        raise ValueError(
            f"Curl is only defined for 3D vector fields, got {mesh.n_spatial_dims=}"
        )

    from physicsnemo.mesh.calculus._lsq_reconstruction import compute_point_gradient_lsq

    ### Compute full Jacobian in one batched LSQ solve
    # vector_field: (n_points, 3) -> jacobian: (n_points, 3, 3)
    # jacobian[i, k, j] = ∂v_j/∂x_k (derivative axis first)
    jacobian = compute_point_gradient_lsq(mesh, vector_field)

    return _curl_from_jacobian(jacobian)


def compute_curl_cells_lsq(
    mesh: "Mesh",
    vector_field: Float[torch.Tensor, "n_cells 3"],
) -> Float[torch.Tensor, "n_cells 3"]:
    r"""Compute curl at cell centers using LSQ gradient method.

    Cell-centered analogue of :func:`compute_curl_points_lsq`: computes the
    Jacobian of the vector field via the cell-neighbour LSQ gradient, then
    takes its antisymmetric part.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    vector_field : Float[torch.Tensor, "n_cells 3"]
        Vectors at cell centers.

    Returns
    -------
    Float[torch.Tensor, "n_cells 3"]
        Curl at cell centers.

    Raises
    ------
    ValueError
        If ``n_spatial_dims != 3``.
    """
    if mesh.n_spatial_dims != 3:
        raise ValueError(
            f"Curl is only defined for 3D vector fields, got {mesh.n_spatial_dims=}"
        )

    from physicsnemo.mesh.calculus._lsq_reconstruction import compute_cell_gradient_lsq

    # vector_field: (n_cells, 3) -> jacobian: (n_cells, 3, 3)
    # jacobian[i, k, j] = ∂v_j/∂x_k (derivative axis first)
    jacobian = compute_cell_gradient_lsq(mesh, vector_field)

    return _curl_from_jacobian(jacobian)


def _curl_from_jacobian(
    jacobian: Float[torch.Tensor, "n 3 3"],
) -> Float[torch.Tensor, "n 3"]:
    r"""Extract the curl (antisymmetric part) from a batch of 3D Jacobians.

    With ``jacobian[i, k, j] = ∂v_j/∂x_k``:
    curl = [∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y].
    """
    return torch.stack(
        [
            jacobian[:, 1, 2] - jacobian[:, 2, 1],
            jacobian[:, 2, 0] - jacobian[:, 0, 2],
            jacobian[:, 0, 1] - jacobian[:, 1, 0],
        ],
        dim=-1,
    )
