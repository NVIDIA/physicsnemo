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

"""Hemisphere surface in 3D space.

Dimensional: 2D manifold in 3D space (has boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    theta_resolution: int = 30,
    phi_resolution: int = 15,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a hemisphere surface in 3D space.

    Parameters
    ----------
    radius : float
        Radius of the hemisphere.
    theta_resolution : int
        Number of points around the equator.
    phi_resolution : int
        Number of points from equator to pole.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if theta_resolution < 3:
        raise ValueError(
            f"theta_resolution must be at least 3, got {theta_resolution=}"
        )
    if phi_resolution < 2:
        raise ValueError(f"phi_resolution must be at least 2, got {phi_resolution=}")

    # Parametric hemisphere (upper half)
    theta = torch.linspace(0, 2 * torch.pi, theta_resolution + 1, device=device)[:-1]
    phi = torch.linspace(0, torch.pi / 2, phi_resolution, device=device)

    points = []
    for phi_val in phi:
        for theta_val in theta:
            x = radius * torch.sin(phi_val) * torch.cos(theta_val)
            y = radius * torch.sin(phi_val) * torch.sin(theta_val)
            z = radius * torch.cos(phi_val)
            points.append([x.item(), y.item(), z.item()])

    points = torch.tensor(points, dtype=torch.float32, device=device)

    # Create cells
    cells = []
    for i in range(phi_resolution - 1):
        for j in range(theta_resolution):
            idx = i * theta_resolution + j
            next_j = (j + 1) % theta_resolution

            # Two triangles per quad
            cells.append([idx, idx + next_j - j, idx + theta_resolution])
            cells.append(
                [
                    idx + next_j - j,
                    idx + theta_resolution + next_j - j,
                    idx + theta_resolution,
                ]
            )

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
