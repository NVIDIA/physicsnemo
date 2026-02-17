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

"""Torus surface in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    n_major: int = 48,
    n_minor: int = 24,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a torus surface in 3D space.

    Parameters
    ----------
    major_radius : float
        Distance from center to tube center.
    minor_radius : float
        Radius of the tube.
    n_major : int
        Number of points around the major circle.
    n_minor : int
        Number of points around the minor circle.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    if n_major < 3:
        raise ValueError(f"n_major must be at least 3, got {n_major=}")
    if n_minor < 3:
        raise ValueError(f"n_minor must be at least 3, got {n_minor=}")
    if minor_radius >= major_radius:
        raise ValueError(
            f"minor_radius must be < major_radius, got {minor_radius=}, {major_radius=}"
        )

    # Parametric torus
    u = torch.linspace(0, 2 * torch.pi, n_major + 1, device=device)[:-1]
    v = torch.linspace(0, 2 * torch.pi, n_minor + 1, device=device)[:-1]

    # Vectorized point generation
    U, V = torch.meshgrid(u, v, indexing="ij")
    x = (major_radius + minor_radius * torch.cos(V)) * torch.cos(U)
    y = (major_radius + minor_radius * torch.cos(V)) * torch.sin(U)
    z = minor_radius * torch.sin(V)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(dtype=torch.float32)

    # Vectorized cell generation (closed in both u and v)
    i_idx = torch.arange(n_major, device=device)
    j_idx = torch.arange(n_minor, device=device)
    ii, jj = torch.meshgrid(i_idx, j_idx, indexing="ij")
    ii_flat = ii.reshape(-1)
    jj_flat = jj.reshape(-1)

    idx = ii_flat * n_minor + jj_flat
    next_i = ((ii_flat + 1) % n_major) * n_minor + jj_flat
    next_j = ii_flat * n_minor + (jj_flat + 1) % n_minor
    next_both = ((ii_flat + 1) % n_major) * n_minor + (jj_flat + 1) % n_minor

    tri1 = torch.stack([idx, next_j, next_i], dim=1)
    tri2 = torch.stack([next_j, next_both, next_i], dim=1)
    cells = torch.cat([tri1, tri2], dim=0)

    return Mesh(points=points, cells=cells)
