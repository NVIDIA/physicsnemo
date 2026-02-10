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

"""Unit square triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(subdivisions: int = 1, device: torch.device | str = "cpu") -> Mesh:
    """Create a triangulated unit square in 2D space.

    Parameters
    ----------
    subdivisions : int
        Number of subdivision levels (0 = 2 triangles). Each level quadruples
        the number of triangles: 0 → 2, 1 → 8, 2 → 32, etc.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions=}")

    n = 2**subdivisions + 1

    # Create grid of points
    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Create triangular cells
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            # Two triangles per quad
            cells.append([idx, idx + 1, idx + n])
            cells.append([idx + 1, idx + n + 1, idx + n])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
