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

"""Straight line segment in 3D space.

Dimensional: 1D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    end: tuple[float, float, float] = (1.0, 1.0, 1.0),
    n_points: int = 10,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a straight line segment in 3D space.

    Parameters
    ----------
    start : tuple[float, float, float]
        Starting point (x, y, z).
    end : tuple[float, float, float]
        Ending point (x, y, z).
    n_points : int
        Number of points along the line.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=3, n_cells=n_points-1.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points=}")

    # Interpolate between start and end
    t = torch.linspace(0.0, 1.0, n_points, device=device).unsqueeze(1)
    start_t = torch.tensor(start, dtype=torch.float32, device=device)
    end_t = torch.tensor(end, dtype=torch.float32, device=device)
    points = start_t * (1 - t) + end_t * t

    # Create edge cells
    cells = torch.stack(
        [
            torch.arange(n_points - 1, device=device),
            torch.arange(1, n_points, device=device),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
