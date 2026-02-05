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

"""Möbius strip surface in 3D space.

Dimensional: 2D manifold in 3D space (non-orientable, has boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    width: float = 0.3,
    n_circ: int = 48,
    n_width: int = 5,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a Möbius strip surface in 3D space.

    Parameters
    ----------
    radius : float
        Radius of the center circle.
    width : float
        Width of the strip.
    n_circ : int
        Number of points around the circle.
    n_width : int
        Number of points across the width.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3 (non-orientable).
    """
    if n_circ < 3:
        raise ValueError(f"n_circ must be at least 3, got {n_circ=}")
    if n_width < 2:
        raise ValueError(f"n_width must be at least 2, got {n_width=}")

    # Parametric Möbius strip
    u = torch.linspace(0, 2 * torch.pi, n_circ + 1, device=device)[:-1]
    v = torch.linspace(-width / 2, width / 2, n_width, device=device)

    points = []
    for u_val in u:
        for v_val in v:
            # Möbius strip parameterization
            x = (radius + v_val * torch.cos(u_val / 2)) * torch.cos(u_val)
            y = (radius + v_val * torch.cos(u_val / 2)) * torch.sin(u_val)
            z = v_val * torch.sin(u_val / 2)
            points.append([x.item(), y.item(), z.item()])

    points = torch.tensor(points, dtype=torch.float32, device=device)

    # Create cells
    cells = []
    for i in range(n_circ):
        for j in range(n_width - 1):
            idx = i * n_width + j
            next_j = i * n_width + j + 1

            # Handle Möbius twist at wrap-around
            if i == n_circ - 1:  # Last slice connecting back to first
                # Flip width index for the half-twist: j -> (n_width - 1 - j)
                next_i = n_width - 1 - j
                next_both = n_width - 2 - j
            else:
                next_i = (i + 1) * n_width + j
                next_both = (i + 1) * n_width + j + 1

            # Two triangles per quad
            cells.append([idx, next_j, next_i])
            cells.append([next_j, next_both, next_i])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
