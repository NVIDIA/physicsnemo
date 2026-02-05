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

"""Cylinder surface with caps in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    height: float = 2.0,
    n_circ: int = 32,
    n_height: int = 10,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a cylinder surface (with caps) in 3D space.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.
    height : float
        Height of the cylinder.
    n_circ : int
        Number of points around the circumference.
    n_height : int
        Number of points along the height.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if n_circ < 3:
        raise ValueError(f"n_circ must be at least 3, got {n_circ=}")
    if n_height < 2:
        raise ValueError(f"n_height must be at least 2, got {n_height=}")

    ### Create cylindrical side
    theta = torch.linspace(0, 2 * torch.pi, n_circ + 1, device=device)[:-1]
    z = torch.linspace(-height / 2, height / 2, n_height, device=device)

    # Side points
    points_side = []
    for z_val in z:
        for theta_val in theta:
            x = radius * torch.cos(theta_val)
            y = radius * torch.sin(theta_val)
            points_side.append([x.item(), y.item(), z_val.item()])

    # Create cells for side
    cells_side = []
    for i in range(n_height - 1):
        for j in range(n_circ):
            idx = i * n_circ + j
            next_j = (j + 1) % n_circ

            # Two triangles per quad
            cells_side.append([idx, idx + next_j - j, idx + n_circ])
            cells_side.append(
                [idx + next_j - j, idx + n_circ + next_j - j, idx + n_circ]
            )

    ### Add caps
    # Bottom cap center
    bottom_center_idx = len(points_side)
    points_side.append([0.0, 0.0, -height / 2])

    # Top cap center
    top_center_idx = len(points_side)
    points_side.append([0.0, 0.0, height / 2])

    # Bottom cap triangles
    for j in range(n_circ):
        next_j = (j + 1) % n_circ
        cells_side.append([bottom_center_idx, next_j, j])

    # Top cap triangles
    top_ring_offset = (n_height - 1) * n_circ
    for j in range(n_circ):
        next_j = (j + 1) % n_circ
        cells_side.append(
            [top_center_idx, top_ring_offset + j, top_ring_offset + next_j]
        )

    points = torch.tensor(points_side, dtype=torch.float32, device=device)
    cells = torch.tensor(cells_side, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
