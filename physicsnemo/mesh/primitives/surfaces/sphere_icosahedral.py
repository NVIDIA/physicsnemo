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

"""Icosahedral sphere surface in 3D space.

A sphere created by subdividing an icosahedron and projecting vertices
onto the sphere surface. This produces a more uniform triangulation than
UV-parameterized spheres.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import icosahedron_surface


def load(
    radius: float = 1.0,
    subdivisions: int = 2,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a sphere by subdividing an icosahedron and projecting to sphere.

    This method produces a more uniform triangulation than UV-parameterized
    spheres (no pole singularities). Each subdivision level quadruples the
    number of triangles.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    subdivisions : int
        Number of subdivision levels to apply. Each level quadruples the
        triangle count:
        - 0: 20 triangles (base icosahedron)
        - 1: 80 triangles
        - 2: 320 triangles
        - 3: 1280 triangles
        - 4: 5120 triangles
    device : torch.device or str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(radius=1.0, subdivisions=2)
    >>> mesh.n_manifold_dims, mesh.n_spatial_dims
    (2, 3)
    >>> mesh.n_cells  # 20 * 4^2 = 320 triangles
    320
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius=}")
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions=}")

    ### Start with base icosahedron
    mesh = icosahedron_surface.load(radius=1.0, device=device)

    ### Apply subdivision levels
    if subdivisions > 0:
        mesh = mesh.subdivide(levels=subdivisions, filter="linear")

        ### Project all points back onto the sphere surface
        # After linear subdivision, new vertices are at edge midpoints which
        # lie inside the sphere. Project them back to the sphere surface.
        norms = torch.norm(mesh.points, dim=-1, keepdim=True)
        mesh = Mesh(
            points=mesh.points / norms * radius,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )

    return mesh
