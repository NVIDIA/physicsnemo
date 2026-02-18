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

"""Flat disk in 3D space.

Dimensional: 2D manifold in 3D space (has boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    n_radial: int = 10,
    n_angular: int = 32,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a flat disk in 3D space (lying in xy-plane).

    Parameters
    ----------
    radius : float
        Radius of the disk.
    n_radial : int
        Number of points in radial direction.
    n_angular : int
        Number of points around the circumference.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if n_radial < 1:
        raise ValueError(f"n_radial must be at least 1, got {n_radial=}")
    if n_angular < 3:
        raise ValueError(f"n_angular must be at least 3, got {n_angular=}")

    # Center point
    center = torch.zeros(1, 3, dtype=torch.float32, device=device)

    # Vectorized radial ring point generation
    r_vals = (
        radius
        * torch.arange(1, n_radial + 1, device=device, dtype=torch.float32)
        / n_radial
    )
    theta = torch.linspace(0, 2 * torch.pi, n_angular + 1, device=device)[:-1]
    R, THETA = torch.meshgrid(r_vals, theta, indexing="ij")
    x = R * torch.cos(THETA)
    y = R * torch.sin(THETA)
    z = torch.zeros_like(x)
    ring_points = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(dtype=torch.float32)

    points = torch.cat([center, ring_points], dim=0)

    # Vectorized cell generation
    # Innermost ring connected to center
    j_idx = torch.arange(n_angular, device=device)
    next_j = (j_idx + 1) % n_angular
    inner_cells = torch.stack(
        [
            torch.zeros(n_angular, dtype=torch.int64, device=device),
            1 + next_j,
            1 + j_idx,
        ],
        dim=1,
    )

    cells_parts = [inner_cells]

    # Outer rings
    if n_radial > 1:
        i_idx = torch.arange(n_radial - 1, device=device)
        j_idx_outer = torch.arange(n_angular, device=device)
        ii, jj = torch.meshgrid(i_idx, j_idx_outer, indexing="ij")
        ii_flat = ii.reshape(-1)
        jj_flat = jj.reshape(-1)

        idx = 1 + ii_flat * n_angular + jj_flat
        next_j_o = 1 + ii_flat * n_angular + (jj_flat + 1) % n_angular
        idx_outer = 1 + (ii_flat + 1) * n_angular + jj_flat
        next_j_outer = 1 + (ii_flat + 1) * n_angular + (jj_flat + 1) % n_angular

        tri1 = torch.stack([idx, next_j_o, idx_outer], dim=1)
        tri2 = torch.stack([next_j_o, next_j_outer, idx_outer], dim=1)
        cells_parts.extend([tri1, tri2])

    cells = torch.cat(cells_parts, dim=0)

    return Mesh(points=points, cells=cells)
