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

"""Structured triangular grid in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    n_x: int = 11,
    n_y: int = 11,
    device: torch.device | str = "cpu",
) -> Mesh[2, 2]:
    """Create a structured triangular grid in 2D space.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate.
    x_max : float
        Maximum x coordinate.
    y_min : float
        Minimum y coordinate.
    y_max : float
        Maximum y coordinate.
    n_x : int
        Number of points in x-direction.
    n_y : int
        Number of points in y-direction.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh[2, 2]
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if n_x < 2:
        raise ValueError(f"n_x must be at least 2, got {n_x=}")
    if n_y < 2:
        raise ValueError(f"n_y must be at least 2, got {n_y=}")

    # Create grid of points
    x = torch.linspace(x_min, x_max, n_x, device=device)
    y = torch.linspace(y_min, y_max, n_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    ### Triangulate: two triangles per grid quad, fully vectorized
    ii, jj = torch.meshgrid(
        torch.arange(n_x - 1, device=device),
        torch.arange(n_y - 1, device=device),
        indexing="ij",
    )
    ii, jj = ii.reshape(-1), jj.reshape(-1)

    v00 = ii * n_y + jj
    v01 = v00 + 1
    v10 = v00 + n_y
    v11 = v10 + 1

    cells = torch.stack(
        [
            torch.stack([v00, v01, v10], dim=-1),
            torch.stack([v01, v11, v10], dim=-1),
        ],
        dim=1,
    ).reshape(-1, 3)

    return Mesh(points=points, cells=cells)
