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

"""Regular polygon triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    n_sides: int = 6, radius: float = 1.0, device: torch.device | str = "cpu"
) -> Mesh:
    """Create a regular polygon triangulated in 2D space.

    The polygon is triangulated by connecting all vertices to the center point.

    Parameters
    ----------
    n_sides : int
        Number of sides (must be >= 3).
    radius : float
        Distance from center to vertices.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if n_sides < 3:
        raise ValueError(f"n_sides must be at least 3, got {n_sides=}")

    # Create vertices around the circle
    theta = torch.linspace(0, 2 * torch.pi, n_sides + 1, device=device)[:-1]
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    # Add center point
    points = torch.cat(
        [
            torch.zeros((1, 2), dtype=torch.float32, device=device),
            torch.stack([x, y], dim=1),
        ],
        dim=0,
    )

    # Create triangular cells from center to each edge
    cells = []
    for i in range(n_sides):
        next_i = (i + 1) % n_sides
        cells.append([0, i + 1, next_i + 1])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
