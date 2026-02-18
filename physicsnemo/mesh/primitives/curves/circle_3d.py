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

"""Closed circle curve in 3D space.

Dimensional: 1D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    n_points: int = 32,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a closed circle curve in 3D space.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    center : tuple[float, float, float]
        Center point (x, y, z).
    normal : tuple[float, float, float]
        Normal vector to the circle plane (will be normalized).
    n_points : int
        Number of points around the circle.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=3, n_cells=n_points.
    """
    if n_points < 3:
        raise ValueError(f"n_points must be at least 3, got {n_points=}")

    # Normalize the normal vector
    normal_t = torch.tensor(normal, dtype=torch.float32, device=device)
    normal_t = normal_t / torch.norm(normal_t)

    # Find two orthogonal vectors in the plane
    if abs(normal_t[0].item()) < 0.9:
        u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    else:
        u = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    u = u - torch.dot(u, normal_t) * normal_t
    u = u / torch.norm(u)
    v = torch.linalg.cross(normal_t, u)

    theta = torch.linspace(0, 2 * torch.pi, n_points + 1, device=device)[:-1]
    center_t = torch.tensor(center, dtype=torch.float32, device=device)

    # Parametric circle: center + radius * (cos(theta) * u + sin(theta) * v)
    points = (
        center_t.unsqueeze(0)
        + radius * torch.cos(theta).unsqueeze(1) * u.unsqueeze(0)
        + radius * torch.sin(theta).unsqueeze(1) * v.unsqueeze(0)
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
