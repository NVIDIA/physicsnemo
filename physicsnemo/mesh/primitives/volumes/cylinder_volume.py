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

import torch

from physicsnemo.core.version_check import require_version_spec
from physicsnemo.mesh.mesh import Mesh


@require_version_spec("pyvista")
def load(
    radius: float = 1.0,
    height: float = 2.0,
    resolution: int = 20,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a tetrahedral volume mesh of a cylinder.

    The cylinder is filled with tetrahedra using PyVista's delaunay_3d filter.

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
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    import importlib

    pv = importlib.import_module("pyvista")

    from physicsnemo.mesh.io.io_pyvista import from_pyvista

    ### Create a cylinder surface and fill it with tetrahedra
    # PyVista's Cylinder is centered at origin by default
    cylinder = pv.Cylinder(
        radius=radius,
        height=height,
        resolution=resolution,
        capping=True,
    )

    ### Use delaunay_3d to fill the interior with tetrahedra
    volume = cylinder.delaunay_3d()

    ### Convert to physicsnemo Mesh and move to device
    return from_pyvista(volume).to(device=device)
