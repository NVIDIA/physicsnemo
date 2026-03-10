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

"""Flat plane in 3D space.

Dimensional: 2D manifold in 3D space (has boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    size: float = 2.0,
    subdivisions: int = 10,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a flat triangulated plane in 3D space.

    Parameters
    ----------
    size : float
        Size of the plane (length of each side).
    subdivisions : int
        Number of subdivisions per edge. Creates (subdivisions+1)^2 vertices
        and 2*subdivisions^2 triangles.
    normal : tuple[float, float, float]
        Normal vector to the plane (will be normalized).
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be at least 1, got {subdivisions=}")

    n = subdivisions + 1

    # Create grid of points in xy-plane
    x = torch.linspace(-size / 2, size / 2, n, device=device)
    y = torch.linspace(-size / 2, size / 2, n, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    points_flat = torch.stack(
        [xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())], dim=1
    )

    # Rotate to align with normal if not (0, 0, 1)
    normal_t = torch.tensor(normal, dtype=torch.float32, device=device)
    normal_t = normal_t / torch.norm(normal_t)

    if not torch.allclose(normal_t, torch.tensor([0.0, 0.0, 1.0], device=device)):
        # Find rotation axis and angle
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        axis = torch.cross(z_axis, normal_t)
        axis_norm = torch.norm(axis)

        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = torch.acos(torch.dot(z_axis, normal_t))

            # Rodrigues' rotation formula
            K = torch.tensor(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ],
                dtype=torch.float32,
                device=device,
            )
            R = (
                torch.eye(3, device=device)
                + torch.sin(angle) * K
                + (1 - torch.cos(angle)) * torch.mm(K, K)
            )
            points = torch.mm(points_flat, R.T)
        else:
            points = points_flat
    else:
        points = points_flat

    # Create triangular cells
    cells = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            idx = i * n + j
            # Two triangles per quad
            cells.append([idx, idx + 1, idx + n])
            cells.append([idx + 1, idx + n + 1, idx + n])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
