# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tetrahedral sphere volume mesh in 3D space.

Dimensional: 3D manifold in 3D space.
"""

import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0, resolution: int = 20, device: torch.device | str = "cpu"
) -> Mesh:
    """Create a tetrahedral volume mesh of a sphere.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    resolution : int
        Resolution of the initial surface mesh.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    # Create surface sphere
    surface = pv.Sphere(
        radius=radius, theta_resolution=resolution, phi_resolution=resolution
    )

    # Fill with tetrahedra using Delaunay 3D
    volume = surface.delaunay_3d()

    mesh = from_pyvista(volume, manifold_dim=3)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
