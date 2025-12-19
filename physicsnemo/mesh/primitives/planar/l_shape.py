# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""L-shaped domain triangulated in 2D space.

Dimensional: 2D manifold in 2D space (non-convex).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    size: float = 1.0, n_subdivisions: int = 5, device: torch.device | str = "cpu"
) -> Mesh:
    """Create an L-shaped non-convex domain in 2D space.

    Parameters
    ----------
    size : float
        Size of the L-shape.
    n_subdivisions : int
        Number of subdivisions per edge.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if n_subdivisions < 1:
        raise ValueError(f"n_subdivisions must be at least 1, got {n_subdivisions=}")

    # Create L-shape vertices
    # The L-shape is made of two rectangles
    points = []
    cells = []

    n = n_subdivisions + 1

    # Bottom horizontal part
    for i in range(n):
        for j in range(n):
            x = i * size / n_subdivisions
            y = j * size / (2 * n_subdivisions)
            points.append([x, y])

    # Top vertical part
    for i in range(n):
        for j in range(1, n):  # Skip j=0 to avoid overlap
            x = i * size / (2 * n_subdivisions)
            y = size / 2 + j * size / (2 * n_subdivisions)
            points.append([x, y])

    points = torch.tensor(points, dtype=torch.float32, device=device)

    # Create triangular cells for bottom part
    for i in range(n_subdivisions):
        for j in range(n_subdivisions):
            idx = i * n + j
            cells.append([idx, idx + 1, idx + n])
            cells.append([idx + 1, idx + n + 1, idx + n])

    # Offset for top part
    offset = n * n

    # Create triangular cells for top part
    for i in range(n_subdivisions):
        for j in range(n_subdivisions - 1):
            idx = offset + i * (n - 1) + j
            cells.append([idx, idx + 1, idx + (n - 1)])
            cells.append([idx + 1, idx + (n - 1) + 1, idx + (n - 1)])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
