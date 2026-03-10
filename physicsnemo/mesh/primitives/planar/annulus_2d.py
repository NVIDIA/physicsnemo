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

"""Annulus (ring) triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    n_radial: int = 5,
    n_angular: int = 32,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create an annulus (ring) triangulated in 2D space.

    Parameters
    ----------
    inner_radius : float
        Inner radius of the annulus.
    outer_radius : float
        Outer radius of the annulus.
    n_radial : int
        Number of points in radial direction.
    n_angular : int
        Number of points around the circumference.
    device : torch.device or str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.planar import annulus_2d
    >>> mesh = annulus_2d.load()
    >>> mesh.n_manifold_dims, mesh.n_spatial_dims
    (2, 2)
    """
    if inner_radius >= outer_radius:
        raise ValueError(
            f"inner_radius must be < outer_radius, got {inner_radius=}, {outer_radius=}"
        )
    if n_radial < 2:
        raise ValueError(f"n_radial must be at least 2, got {n_radial=}")
    if n_angular < 3:
        raise ValueError(f"n_angular must be at least 3, got {n_angular=}")

    ### Generate points via polar coordinate grid
    r = torch.linspace(inner_radius, outer_radius, n_radial, device=device)
    theta = torch.linspace(0, 2 * torch.pi, n_angular + 1, device=device)[:-1]
    r_grid, theta_grid = torch.meshgrid(r, theta, indexing="ij")

    x = r_grid * torch.cos(theta_grid)
    y = r_grid * torch.sin(theta_grid)
    points = torch.stack([x.flatten(), y.flatten()], dim=1)

    ### Generate triangle connectivity (vectorized)
    cells = _triangulate_ring_quads(
        n_radial - 1, n_angular, ring_offset=0, device=device
    )

    return Mesh(points=points, cells=cells)


def _triangulate_ring_quads(
    n_rings: int, n_angular: int, ring_offset: int, device: torch.device | str
) -> torch.Tensor:
    """Triangulate quads between concentric rings (vectorized).

    Each ring has n_angular points. Quads connect ring i to ring i+1.

    Parameters
    ----------
    n_rings : int
        Number of ring pairs to triangulate (quads between ring i and ring i+1).
    n_angular : int
        Number of points per ring.
    ring_offset : int
        Starting point index of the first ring.
    device : torch.device or str
        Compute device.

    Returns
    -------
    torch.Tensor
        Triangle connectivity with shape (n_rings * n_angular * 2, 3).
    """
    if n_rings == 0:
        return torch.empty(0, 3, dtype=torch.int64, device=device)

    # Grid indices for (ring, sector)
    i = torch.arange(n_rings, device=device)
    j = torch.arange(n_angular, device=device)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")

    # Four corners of each quad
    p00 = ring_offset + i_grid * n_angular + j_grid  # inner ring, current angle
    p10 = ring_offset + (i_grid + 1) * n_angular + j_grid  # outer ring, current angle
    # Roll along angular dimension for "next angle" neighbors (wraps around)
    p01 = torch.roll(p00, shifts=-1, dims=1)  # inner ring, next angle
    p11 = torch.roll(p10, shifts=-1, dims=1)  # outer ring, next angle

    # Two triangles per quad, stacked and interleaved
    tri1 = torch.stack([p00, p10, p11], dim=-1)  # shape: (n_rings, n_angular, 3)
    tri2 = torch.stack([p00, p11, p01], dim=-1)

    return torch.stack([tri1, tri2], dim=2).reshape(-1, 3)
