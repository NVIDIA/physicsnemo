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

"""Curl operator for vector fields (3D only).

Implements curl using both DEC and LSQ methods.

DEC formula: curl = ⋆d♭
    1. Apply flat ♭ to convert vector field to 1-form
    2. Apply exterior derivative d to get 2-form
    3. Apply Hodge star ⋆ to get dual 1-form
    4. Convert back to vector field

For 3D: curl maps vectors to vectors.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_curl_points_lsq(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute curl at vertices using LSQ gradient method.

    For 3D vector field v = [vₓ, vᵧ, v_z]:
        curl(v) = [∂vᵧ/∂y - ∂vᵧ/∂z, ∂vₓ/∂z - ∂vᵧ/∂x, ∂vᵧ/∂x - ∂vₓ/∂y]

    Computes Jacobian of vector field, then takes antisymmetric part.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    vector_field : torch.Tensor
        Vectors at vertices, shape (n_points, 3)

    Returns
    -------
    torch.Tensor
        Curl at vertices, shape (n_points, 3)

    Raises
    ------
    ValueError
        If n_spatial_dims != 3
    """
    if mesh.n_spatial_dims != 3:
        raise ValueError(
            f"Curl is only defined for 3D vector fields, got {mesh.n_spatial_dims=}"
        )

    from physicsnemo.mesh.calculus._lsq_reconstruction import compute_point_gradient_lsq

    n_points = mesh.n_points

    ### Compute full Jacobian in one batched LSQ solve
    # vector_field: (n_points, 3) -> jacobian: (n_points, 3, 3)
    # jacobian[i, j, k] = ∂v_j/∂x_k
    jacobian = compute_point_gradient_lsq(mesh, vector_field)

    ### Compute curl from Jacobian
    # curl = [∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y]
    curl = torch.zeros(
        (n_points, 3), dtype=vector_field.dtype, device=mesh.points.device
    )

    curl[:, 0] = jacobian[:, 2, 1] - jacobian[:, 1, 2]  # ∂vz/∂y - ∂vy/∂z
    curl[:, 1] = jacobian[:, 0, 2] - jacobian[:, 2, 0]  # ∂vx/∂z - ∂vz/∂x
    curl[:, 2] = jacobian[:, 1, 0] - jacobian[:, 0, 1]  # ∂vy/∂x - ∂vx/∂y

    return curl
