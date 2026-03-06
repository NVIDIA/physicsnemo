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

"""Filled disk (circle) triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    n_radial: int = 10,
    n_angular: int = 32,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a filled disk (circle) triangulated in 2D space.

    The disk is meshed with a center point connected by a triangle fan
    to the innermost ring, and concentric rings connected by quads
    (split into triangles).

    Parameters
    ----------
    radius : float
        Radius of the disk.
    n_radial : int
        Number of rings (including center point as ring 0).
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
    >>> from physicsnemo.mesh.primitives.planar import circle_2d
    >>> mesh = circle_2d.load()
    >>> mesh.n_manifold_dims, mesh.n_spatial_dims
    (2, 2)
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius=}")
    if n_radial < 2:
        raise ValueError(f"n_radial must be at least 2, got {n_radial=}")
    if n_angular < 3:
        raise ValueError(f"n_angular must be at least 3, got {n_angular=}")

    ### Generate points
    n_rings = n_radial - 1  # rings excluding center

    # Center point at origin
    center = torch.zeros(1, 2, device=device)

    # Concentric rings
    r = torch.linspace(radius / n_rings, radius, n_rings, device=device)
    theta = torch.linspace(0, 2 * torch.pi, n_angular + 1, device=device)[:-1]
    r_grid, theta_grid = torch.meshgrid(r, theta, indexing="ij")

    x = r_grid * torch.cos(theta_grid)
    y = r_grid * torch.sin(theta_grid)
    ring_points = torch.stack([x.flatten(), y.flatten()], dim=1)

    points = torch.cat([center, ring_points], dim=0)

    ### Generate triangle connectivity (vectorized)
    # Center fan: triangles from center (0) to first ring (1 to n_angular)
    center_fan = _triangulate_pole_fan(
        pole_idx=0,
        ring_start=1,
        n_angular=n_angular,
        pole_is_north=True,
        device=device,
    )

    # Ring quads: between rings 1..n_rings
    ring_quads = _triangulate_ring_quads(
        n_rings=n_rings - 1,
        n_angular=n_angular,
        ring_offset=1,
        device=device,
    )

    cells = torch.cat([center_fan, ring_quads], dim=0)

    return Mesh(points=points, cells=cells)


def _triangulate_pole_fan(
    pole_idx: int,
    ring_start: int,
    n_angular: int,
    pole_is_north: bool,
    device: torch.device | str,
) -> torch.Tensor:
    """Triangulate a fan from a pole to an adjacent ring (vectorized).

    Parameters
    ----------
    pole_idx : int
        Index of the pole vertex.
    ring_start : int
        Starting index of the ring vertices.
    n_angular : int
        Number of vertices in the ring.
    pole_is_north : bool
        If True, winding order for outward normal pointing "up" (away from ring).
        If False, winding order for outward normal pointing "down".
    device : torch.device or str
        Compute device.

    Returns
    -------
    torch.Tensor
        Triangle connectivity with shape (n_angular, 3).
    """
    j = torch.arange(n_angular, device=device)

    pole = torch.full((n_angular,), pole_idx, dtype=torch.int64, device=device)
    p1 = ring_start + j
    p2 = torch.roll(p1, shifts=-1)  # next angle (wraps around)

    if pole_is_north:
        return torch.stack([pole, p1, p2], dim=1)
    else:
        return torch.stack([p1, pole, p2], dim=1)


def _triangulate_ring_quads(
    n_rings: int, n_angular: int, ring_offset: int, device: torch.device | str
) -> torch.Tensor:
    """Triangulate quads between concentric rings (vectorized).

    Parameters
    ----------
    n_rings : int
        Number of ring pairs to triangulate.
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

    i = torch.arange(n_rings, device=device)
    j = torch.arange(n_angular, device=device)
    i_grid, j_grid = torch.meshgrid(i, j, indexing="ij")

    p00 = ring_offset + i_grid * n_angular + j_grid
    p10 = ring_offset + (i_grid + 1) * n_angular + j_grid
    # Roll along angular dimension for "next angle" neighbors (wraps around)
    p01 = torch.roll(p00, shifts=-1, dims=1)
    p11 = torch.roll(p10, shifts=-1, dims=1)

    tri1 = torch.stack([p00, p10, p11], dim=-1)
    tri2 = torch.stack([p00, p11, p01], dim=-1)

    return torch.stack([tri1, tri2], dim=2).reshape(-1, 3)
