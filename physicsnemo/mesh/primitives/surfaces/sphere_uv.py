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

"""UV sphere using latitude/longitude parameterization in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    theta_resolution: int = 30,
    phi_resolution: int = 30,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a UV sphere using latitude/longitude parameterization.

    The sphere is generated using spherical coordinates:
    - phi (latitude): 0 at north pole, π at south pole
    - theta (longitude): 0 to 2π around the equator

    Poles are handled with triangle fans; the body uses quad strips.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    theta_resolution : int
        Number of points around the equator (longitude divisions).
    phi_resolution : int
        Number of latitude rings from pole to pole (including poles).
    device : torch.device or str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_uv
    >>> mesh = sphere_uv.load()
    >>> mesh.n_manifold_dims, mesh.n_spatial_dims
    (2, 3)
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius=}")
    if theta_resolution < 3:
        raise ValueError(
            f"theta_resolution must be at least 3, got {theta_resolution=}"
        )
    if phi_resolution < 3:
        raise ValueError(f"phi_resolution must be at least 3, got {phi_resolution=}")

    n_theta = theta_resolution
    n_phi = phi_resolution
    n_rings = n_phi - 2  # interior rings (excluding poles)

    ### Generate points
    north_pole = torch.tensor([[0.0, 0.0, radius]], device=device)
    south_pole = torch.tensor([[0.0, 0.0, -radius]], device=device)

    # Interior rings via spherical coordinates
    phi = torch.linspace(
        torch.pi / (n_phi - 1),
        torch.pi * (n_phi - 2) / (n_phi - 1),
        n_rings,
        device=device,
    )
    theta = torch.linspace(0, 2 * torch.pi, n_theta + 1, device=device)[:-1]
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")

    x = radius * torch.sin(phi_grid) * torch.cos(theta_grid)
    y = radius * torch.sin(phi_grid) * torch.sin(theta_grid)
    z = radius * torch.cos(phi_grid)

    ring_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    points = torch.cat([north_pole, ring_points, south_pole], dim=0)

    ### Generate triangle connectivity (vectorized)
    north_idx = 0
    south_idx = len(points) - 1

    # North pole fan
    north_fan = _triangulate_pole_fan(
        pole_idx=north_idx,
        ring_start=1,
        n_angular=n_theta,
        pole_is_north=True,
        device=device,
    )

    # Ring quads
    ring_quads = _triangulate_ring_quads(
        n_rings=n_rings - 1,
        n_angular=n_theta,
        ring_offset=1,
        device=device,
    )

    # South pole fan
    last_ring_start = 1 + (n_rings - 1) * n_theta
    south_fan = _triangulate_pole_fan(
        pole_idx=south_idx,
        ring_start=last_ring_start,
        n_angular=n_theta,
        pole_is_north=False,
        device=device,
    )

    cells = torch.cat([north_fan, ring_quads, south_fan], dim=0)

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
        If True, winding order for outward normal pointing away from ring.
        If False, winding order for normal pointing toward ring.
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
