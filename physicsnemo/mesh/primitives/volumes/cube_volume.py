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

"""Tetrahedral cube volume mesh in 3D space.

Dimensional: 3D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    size: float = 1.0, n_subdivisions: int = 5, device: torch.device | str = "cpu"
) -> Mesh:
    """Create a tetrahedral volume mesh of a cube.

    The cube is divided into a regular grid of smaller cubes, and each small
    cube is split into 5 tetrahedra using a consistent diagonal scheme.

    Parameters
    ----------
    size : float
        Side length of the cube.
    n_subdivisions : int
        Number of subdivisions per edge.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    if n_subdivisions < 1:
        raise ValueError(f"n_subdivisions must be at least 1, got {n_subdivisions=}")

    n = n_subdivisions + 1  # Number of points per edge

    ### Generate grid points
    coords_1d = torch.linspace(-size / 2, size / 2, n, device=device)
    x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    ### Generate tetrahedra by splitting each cube into 5 tetrahedra
    # For each cube cell, we split it into 5 tetrahedra using the
    # "5-tetrahedra" decomposition with consistent diagonal orientation.
    cells_list = []

    for i in range(n_subdivisions):
        for j in range(n_subdivisions):
            for k in range(n_subdivisions):
                # 8 vertices of the cube cell (indexed in the flattened grid)
                # Vertex ordering: v0=(i,j,k), v1=(i+1,j,k), v2=(i,j+1,k), etc.
                v0 = i * n * n + j * n + k
                v1 = (i + 1) * n * n + j * n + k
                v2 = i * n * n + (j + 1) * n + k
                v3 = (i + 1) * n * n + (j + 1) * n + k
                v4 = i * n * n + j * n + (k + 1)
                v5 = (i + 1) * n * n + j * n + (k + 1)
                v6 = i * n * n + (j + 1) * n + (k + 1)
                v7 = (i + 1) * n * n + (j + 1) * n + (k + 1)

                # Split cube into 5 tetrahedra using consistent diagonal scheme
                # This decomposition uses the body diagonal from v0 to v7
                cells_list.extend(
                    [
                        [v0, v1, v3, v7],  # tet 1
                        [v0, v3, v2, v7],  # tet 2
                        [v0, v2, v6, v7],  # tet 3
                        [v0, v6, v4, v7],  # tet 4
                        [v0, v4, v5, v7],  # tet 5 (closes with v1)
                    ]
                )

    cells = torch.tensor(cells_list, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
