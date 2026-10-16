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

"""Tetrahedral cylinder volume mesh in 3D space.

Dimensional: 3D manifold in 3D space.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.core.version_check import OptionalImport, require_version_spec
from physicsnemo.mesh.mesh import Mesh

### Optional dependency. ``pv`` is a lazy proxy: construction does not import
### pyvista; the friendly ``ImportError`` (with the ``[mesh-extras]`` install
### hint) fires only on first attribute access. The
### ``@require_version_spec("pyvista")`` decorator on ``load`` raises that
### same error proactively before any function-body work happens.
if TYPE_CHECKING:
    import pyvista as pv
else:
    pv = OptionalImport("pyvista")


@require_version_spec("pyvista")
def load(
    radius: float = 1.0,
    height: float = 2.0,
    resolution: int = 20,
    device: torch.device | str = "cpu",
) -> Mesh[3, 3]:
    """Create a tetrahedral volume mesh of a cylinder.

    The interior is filled with a fan of tetrahedra from the center to each
    triangle of a PyVista cylinder surface. The cylinder is convex, so the
    tetrahedra fill it without overlap.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.
    height : float
        Height of the cylinder (centered at origin).
    resolution : int
        Controls the density of points (higher = more tetrahedra).
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh[3, 3]
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    from physicsnemo.mesh.generate._simplex_ops import orient_positive
    from physicsnemo.mesh.io.io_pyvista import from_pyvista

    ### Create a cylinder surface
    # PyVista's Cylinder is centered at origin by default
    cylinder = pv.Cylinder(
        radius=radius,
        height=height,
        resolution=resolution,
        capping=True,
    )
    # Weld coincident points, with a tolerance relative to the size
    surface = from_pyvista(cylinder).clean(tolerance=1e-6 * min(radius, height))

    ### Fill the convex surface with a fan of tetrahedra from its center
    points = torch.cat([surface.points, surface.points.new_zeros(1, 3)], dim=0)
    center = torch.full((surface.n_cells, 1), surface.n_points, dtype=torch.int64)
    cells = torch.cat([center, surface.cells], dim=1)
    mesh = Mesh(points=points, cells=orient_positive(points, cells))
    return mesh.to(device=device)
