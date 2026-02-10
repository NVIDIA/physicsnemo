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

"""Hemisphere surface in 3D space.

Dimensional: 2D manifold in 3D space (has boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    theta_resolution: int = 30,
    phi_resolution: int = 15,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a hemisphere surface in 3D space.

    Parameters
    ----------
    radius : float
        Radius of the hemisphere.
    theta_resolution : int
        Number of points around the equator.
    phi_resolution : int
        Number of points from equator to pole.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if theta_resolution < 3:
        raise ValueError(
            f"theta_resolution must be at least 3, got {theta_resolution=}"
        )
    if phi_resolution < 2:
        raise ValueError(f"phi_resolution must be at least 2, got {phi_resolution=}")

    # Parametric hemisphere (upper half)
    theta = torch.linspace(0, 2 * torch.pi, theta_resolution + 1, device=device)[:-1]
    phi = torch.linspace(0, torch.pi / 2, phi_resolution, device=device)

    # Vectorized point generation
    PHI, THETA = torch.meshgrid(phi, theta, indexing="ij")
    x = radius * torch.sin(PHI) * torch.cos(THETA)
    y = radius * torch.sin(PHI) * torch.sin(THETA)
    z = radius * torch.cos(PHI)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(dtype=torch.float32)

    # Vectorized cell generation (periodic in theta, open in phi)
    i_idx = torch.arange(phi_resolution - 1, device=device)
    j_idx = torch.arange(theta_resolution, device=device)
    ii, jj = torch.meshgrid(i_idx, j_idx, indexing="ij")
    ii_flat = ii.reshape(-1)
    jj_flat = jj.reshape(-1)

    p00 = ii_flat * theta_resolution + jj_flat
    p01 = ii_flat * theta_resolution + (jj_flat + 1) % theta_resolution
    p10 = (ii_flat + 1) * theta_resolution + jj_flat
    p11 = (ii_flat + 1) * theta_resolution + (jj_flat + 1) % theta_resolution

    tri1 = torch.stack([p00, p01, p10], dim=1)
    tri2 = torch.stack([p01, p11, p10], dim=1)
    cells = torch.cat([tri1, tri2], dim=0)

    return Mesh(points=points, cells=cells)
