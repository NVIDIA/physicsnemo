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

from __future__ import annotations

import math

import torch

from physicsnemo.utils._small_linalg import small_det

from .utils import validate_inputs


def mesh_green_gauss_gradient_torch(
    points: torch.Tensor,
    cells: torch.Tensor,
    neighbors: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute Green-Gauss cell gradients with eager PyTorch tensor ops."""
    ### Validate mesh/value tensors and geometry compatibility.
    validate_inputs(points=points, cells=cells, neighbors=neighbors, values=values)

    n_cells = cells.shape[0]
    dims = points.shape[1]
    n_faces = cells.shape[1]
    value_shape = values.shape[1:]
    values_flat = values.reshape(n_cells, -1)
    n_components = values_flat.shape[1]

    cells_i64 = cells.to(dtype=torch.int64)
    neighbors_i64 = neighbors.to(dtype=torch.int64)

    cell_points = points[cells_i64]
    centroids = cell_points.mean(dim=1)

    # Simplex volume |det(E)| / d!, with E the edge vectors from vertex 0
    edges = cell_points[:, 1:] - cell_points[:, :1]  # (n_cells, dims, dims)
    cell_volume = small_det(edges).abs() / math.factorial(dims)
    cell_volume = torch.clamp(cell_volume, min=1.0e-12)

    grad_flat = torch.zeros(
        (n_cells, dims, n_components),
        device=values.device,
        dtype=values.dtype,
    )

    ### Accumulate Green-Gauss face fluxes into per-cell gradients.
    for face_idx in range(n_faces):
        face_local = [idx for idx in range(n_faces) if idx != face_idx]
        verts = cell_points[:, face_local, :]

        if dims == 2:
            va = verts[:, 0]
            vb = verts[:, 1]
            edge = vb - va
            normal = torch.stack((edge[:, 1], -edge[:, 0]), dim=-1)
            face_center = 0.5 * (va + vb)
        else:
            va = verts[:, 0]
            vb = verts[:, 1]
            vc = verts[:, 2]
            normal = 0.5 * torch.cross(vb - va, vc - va, dim=-1)
            face_center = (va + vb + vc) / 3.0

        to_face = face_center - centroids
        sign = torch.where(
            torch.einsum("bi,bi->b", normal, to_face) >= 0.0,
            1.0,
            -1.0,
        ).unsqueeze(-1)
        coeff = (sign * normal) / cell_volume.unsqueeze(-1)
        coeff = coeff.to(dtype=values.dtype)

        neigh = neighbors_i64[:, face_idx]
        face_values = values_flat
        interior = neigh >= 0
        if torch.any(interior):
            face_values = values_flat.clone()
            face_values[interior] = 0.5 * (
                values_flat[interior] + values_flat[neigh[interior]]
            )

        grad_flat = grad_flat + coeff.unsqueeze(-1) * face_values.unsqueeze(1)

    ### Restore gradient output layout.
    return grad_flat.reshape(n_cells, dims, *value_shape)
