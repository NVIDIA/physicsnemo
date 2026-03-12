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

"""Cylinder surface with caps in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
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
    """Create a cylinder surface (with caps) in 3D space.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.
    height : float
        Height of the cylinder.
    n_circ : int
        Number of points around the circumference.
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

    ### Create cylindrical side
    theta = torch.linspace(0, 2 * torch.pi, n_circ + 1, device=device)[:-1]
    z_vals = torch.linspace(-height / 2, height / 2, n_height, device=device)

    # Vectorized side point generation
    Z, THETA = torch.meshgrid(z_vals, theta, indexing="ij")
    x = radius * torch.cos(THETA)
    y = radius * torch.sin(THETA)
    side_points = torch.stack([x, y, Z], dim=-1).reshape(-1, 3).to(dtype=torch.float32)

    # Vectorized side cell generation (periodic in theta, open in z)
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

    ### Add caps
    bottom_center_idx = n_height * n_circ
    top_center_idx = n_height * n_circ + 1
    cap_points = torch.tensor(
        [[0.0, 0.0, -height / 2], [0.0, 0.0, height / 2]],
        dtype=torch.float32,
        device=device,
    )

    # Vectorized bottom cap triangles
    j_cap = torch.arange(n_circ, device=device)
    next_j_cap = (j_cap + 1) % n_circ
    bottom_cells = torch.stack(
        [
            torch.full((n_circ,), bottom_center_idx, dtype=torch.int64, device=device),
            next_j_cap,
            j_cap,
        ],
        dim=1,
    )

    # Vectorized top cap triangles
    top_ring_offset = (n_height - 1) * n_circ
    top_cells = torch.stack(
        [
            torch.full((n_circ,), top_center_idx, dtype=torch.int64, device=device),
            top_ring_offset + j_cap,
            top_ring_offset + next_j_cap,
        ],
        dim=1,
    )

    points = torch.cat([side_points, cap_points], dim=0)
    cells = torch.cat([tri1, tri2, bottom_cells, top_cells], dim=0)

    return Mesh(points=points, cells=cells)
