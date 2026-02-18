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

"""Archimedean spiral in 2D space.

Dimensional: 1D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    n_turns: float = 3.0,
    spacing: float = 0.5,
    n_points: int = 100,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create an Archimedean spiral in 2D space.

    The spiral follows r = spacing * theta.

    Parameters
    ----------
    n_turns : float
        Number of complete turns.
    spacing : float
        Radial spacing between turns.
    n_points : int
        Number of points along the spiral.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=2, n_cells=n_points-1.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points=}")

    theta = torch.linspace(0, 2 * torch.pi * n_turns, n_points, device=device)
    r = spacing * theta

    points = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)

    # Create edge cells
    cells = torch.stack(
        [
            torch.arange(n_points - 1, device=device),
            torch.arange(1, n_points, device=device),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
