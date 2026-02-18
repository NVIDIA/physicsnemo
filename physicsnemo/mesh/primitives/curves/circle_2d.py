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

"""Closed circle curve in 2D space.

Dimensional: 1D manifold in 2D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0, n_points: int = 32, device: torch.device | str = "cpu"
) -> Mesh:
    """Create a closed circle curve in 2D space.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    n_points : int
        Number of points around the circle.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=2, n_cells=n_points.
    """
    if n_points < 3:
        raise ValueError(f"n_points must be at least 3, got {n_points=}")

    theta = torch.linspace(0, 2 * torch.pi, n_points + 1, device=device)[:-1]

    points = torch.stack(
        [radius * torch.cos(theta), radius * torch.sin(theta)],
        dim=1,
    )

    # Create edge cells, including wrap-around edge
    cells = torch.stack(
        [
            torch.arange(n_points, device=device),
            torch.cat(
                [
                    torch.arange(1, n_points, device=device),
                    torch.tensor([0], device=device),
                ]
            ),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
