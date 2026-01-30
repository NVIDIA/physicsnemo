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

"""Flat disk in 3D space.

Dimensional: 2D manifold in 3D space (has boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    n_radial: int = 10,
    n_angular: int = 32,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a flat disk in 3D space (lying in xy-plane).

    Parameters
    ----------
    radius : float
        Radius of the disk.
    n_radial : int
        Number of points in radial direction.
    n_angular : int
        Number of points around the circumference.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if n_radial < 1:
        raise ValueError(f"n_radial must be at least 1, got {n_radial=}")
    if n_angular < 3:
        raise ValueError(f"n_angular must be at least 3, got {n_angular=}")

    # Center point
    points = [torch.tensor([0.0, 0.0, 0.0], device=device)]

    # Radial rings
    for i in range(1, n_radial + 1):
        r = radius * i / n_radial
        theta = torch.linspace(0, 2 * torch.pi, n_angular + 1, device=device)[:-1]
        for theta_val in theta:
            x = r * torch.cos(theta_val)
            y = r * torch.sin(theta_val)
            points.append(torch.tensor([x.item(), y.item(), 0.0], device=device))

    points = torch.stack(points, dim=0)

    # Create cells
    cells = []

    # Innermost ring connected to center
    for j in range(n_angular):
        next_j = (j + 1) % n_angular
        cells.append([0, 1 + next_j, 1 + j])

    # Outer rings
    for i in range(n_radial - 1):
        for j in range(n_angular):
            idx = 1 + i * n_angular + j
            next_j = 1 + i * n_angular + (j + 1) % n_angular
            idx_outer = 1 + (i + 1) * n_angular + j
            next_j_outer = 1 + (i + 1) * n_angular + (j + 1) % n_angular

            # Two triangles per quad
            cells.append([idx, next_j, idx_outer])
            cells.append([next_j, next_j_outer, idx_outer])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
