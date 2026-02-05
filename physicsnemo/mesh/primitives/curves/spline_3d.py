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

"""3D spiral curve (equivalent to PyVista's load_spline example).

Dimensional: 1D manifold in 3D space.

This curve is a parametric spiral defined by:
    theta = linspace(-4*pi, 4*pi, n)
    z = linspace(-2, 2, n)
    r = z^2 + 1
    x = r * sin(theta)
    y = r * cos(theta)

This produces a helix-like curve where the radius grows quadratically
as z departs from zero, creating a "spool" or "hourglass" shape.

Note
----
Identical to PyVista's ``pv.examples.load_spline()`` but implemented
without PyVista dependency.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(n_points: int = 1000, device: torch.device | str = "cpu") -> Mesh:
    """Create a 3D spiral curve.

    The curve is a parametric spiral where the radius varies with z-coordinate,
    creating a helix that widens at the ends. This is the same curve as
    PyVista's ``pv.examples.load_spline()``.

    Parameters
    ----------
    n_points : int
        Number of points along the curve (default: 1000, matching PyVista).
    device : torch.device or str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=3, n_cells=n_points-1.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.curves import spline_3d
    >>> mesh = spline_3d.load()
    >>> mesh.n_points, mesh.n_cells
    (1000, 999)
    >>> mesh.n_spatial_dims, mesh.n_manifold_dims
    (3, 1)
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points=}")

    ### Parametric curve sampling
    # Parameter t ∈ [0, 1] maps to both theta and z
    theta = torch.linspace(-4 * torch.pi, 4 * torch.pi, n_points, device=device)
    z = torch.linspace(-2.0, 2.0, n_points, device=device)

    # Radius grows quadratically with distance from z=0
    r = z**2 + 1  # shape: (n_points,)

    # Convert cylindrical (r, theta, z) to Cartesian (x, y, z)
    x = r * torch.sin(theta)
    y = r * torch.cos(theta)

    points = torch.stack([x, y, z], dim=1)  # shape: (n_points, 3)

    ### Edge connectivity for polyline
    cells = torch.stack(
        [
            torch.arange(n_points - 1, device=device),
            torch.arange(1, n_points, device=device),
        ],
        dim=1,
    )  # shape: (n_points - 1, 2)

    return Mesh(points=points, cells=cells)
