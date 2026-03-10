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

"""Cone surface in 3D space.

Dimensional: 2D manifold in 3D space (has boundary at base).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    height: float = 2.0,
    n_circ: int = 32,
    n_height: int = 10,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a cone surface in 3D space.

    Parameters
    ----------
    radius : float
        Radius of the base.
    height : float
        Height of the cone.
    n_circ : int
        Number of points around the base circumference.
    n_height : int
        Number of points along the height.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if n_circ < 3:
        raise ValueError(f"n_circ must be at least 3, got {n_circ=}")
    if n_height < 2:
        raise ValueError(f"n_height must be at least 2, got {n_height=}")

    ### Create conical side
    theta = torch.linspace(0, 2 * torch.pi, n_circ + 1, device=device)[:-1]
    z_levels = torch.linspace(0, height, n_height, device=device)

    # Vectorized side point generation (radius decreases with height)
    i_vals = torch.arange(n_height, device=device, dtype=torch.float32)
    r_vals = radius * (1 - i_vals / (n_height - 1))
    R, THETA = torch.meshgrid(r_vals, theta, indexing="ij")
    Z, _ = torch.meshgrid(z_levels, theta, indexing="ij")
    x = R * torch.cos(THETA)
    y = R * torch.sin(THETA)
    side_points = torch.stack([x, y, Z], dim=-1).reshape(-1, 3).to(dtype=torch.float32)

    # Vectorized side cell generation (periodic in theta, open in height)
    i_idx = torch.arange(n_height - 1, device=device)
    j_idx = torch.arange(n_circ, device=device)
    ii, jj = torch.meshgrid(i_idx, j_idx, indexing="ij")
    ii_flat = ii.reshape(-1)
    jj_flat = jj.reshape(-1)

    p00 = ii_flat * n_circ + jj_flat
    p01 = ii_flat * n_circ + (jj_flat + 1) % n_circ
    p10 = (ii_flat + 1) * n_circ + jj_flat
    p11 = (ii_flat + 1) * n_circ + (jj_flat + 1) % n_circ

    tri1 = torch.stack([p00, p01, p10], dim=1)
    tri2 = torch.stack([p01, p11, p10], dim=1)

    ### Add base cap
    base_center_idx = n_height * n_circ
    base_center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    # Vectorized base cap triangles
    j_cap = torch.arange(n_circ, device=device)
    next_j_cap = (j_cap + 1) % n_circ
    base_cells = torch.stack(
        [
            torch.full((n_circ,), base_center_idx, dtype=torch.int64, device=device),
            next_j_cap,
            j_cap,
        ],
        dim=1,
    )

    points = torch.cat([side_points, base_center], dim=0)
    cells = torch.cat([tri1, tri2, base_cells], dim=0)

    return Mesh(points=points, cells=cells)
