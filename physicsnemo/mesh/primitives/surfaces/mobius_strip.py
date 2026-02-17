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

    # Vectorized point generation
    U, V = torch.meshgrid(u, v, indexing="ij")
    x = (radius + V * torch.cos(U / 2)) * torch.cos(U)
    y = (radius + V * torch.cos(U / 2)) * torch.sin(U)
    z = V * torch.sin(U / 2)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(dtype=torch.float32)

    # Vectorized cell generation
    # Regular cells (i < n_circ - 1): open in v, no twist
    i_reg = torch.arange(n_circ - 1, device=device)
    j_reg = torch.arange(n_width - 1, device=device)
    ii, jj = torch.meshgrid(i_reg, j_reg, indexing="ij")
    ii_flat = ii.reshape(-1)
    jj_flat = jj.reshape(-1)

    idx_r = ii_flat * n_width + jj_flat
    next_j_r = ii_flat * n_width + jj_flat + 1
    next_i_r = (ii_flat + 1) * n_width + jj_flat
    next_both_r = (ii_flat + 1) * n_width + jj_flat + 1

    tri1_reg = torch.stack([idx_r, next_j_r, next_i_r], dim=1)
    tri2_reg = torch.stack([next_j_r, next_both_r, next_i_r], dim=1)

    # Twist cells (i = n_circ - 1): connects back to first slice with flipped v
    j_tw = torch.arange(n_width - 1, device=device)
    last_i = n_circ - 1

    idx_t = last_i * n_width + j_tw
    next_j_t = last_i * n_width + j_tw + 1
    next_i_t = n_width - 1 - j_tw
    next_both_t = n_width - 2 - j_tw

    tri1_twist = torch.stack([idx_t, next_j_t, next_i_t], dim=1)
    tri2_twist = torch.stack([next_j_t, next_both_t, next_i_t], dim=1)

    cells = torch.cat([tri1_reg, tri2_reg, tri1_twist, tri2_twist], dim=0)

    return Mesh(points=points, cells=cells)
