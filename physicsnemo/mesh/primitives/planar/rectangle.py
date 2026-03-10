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

"""Rectangle triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    width: float = 2.0,
    height: float = 1.0,
    n_x: int = 10,
    n_y: int = 5,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a triangulated rectangle in 2D space.

    Parameters
    ----------
    width : float
        Width of the rectangle.
    height : float
        Height of the rectangle.
    n_x : int
        Number of points in x-direction.
    n_y : int
        Number of points in y-direction.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if n_x < 2:
        raise ValueError(f"n_x must be at least 2, got {n_x=}")
    if n_y < 2:
        raise ValueError(f"n_y must be at least 2, got {n_y=}")

    # Create grid of points
    x = torch.linspace(0.0, width, n_x, device=device)
    y = torch.linspace(0.0, height, n_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Create triangular cells
    cells = []
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            idx = i * n_y + j
            # Two triangles per quad
            cells.append([idx, idx + 1, idx + n_y])
            cells.append([idx + 1, idx + n_y + 1, idx + n_y])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
