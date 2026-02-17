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
    size: float = 1.0, subdivisions: int = 5, device: torch.device | str = "cpu"
) -> Mesh:
    """Create a tetrahedral volume mesh of a cube.

    The cube is divided into a regular grid of smaller cubes, and each small
    cube is split into 6 tetrahedra using the Kuhn triangulation. This
    triangulation naturally produces matching faces between adjacent cubes,
    ensuring a valid watertight volume mesh.

    Parameters
    ----------
    size : float
        Side length of the cube.
    subdivisions : int
        Number of subdivisions per edge.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be at least 1, got {subdivisions=}")

    n = subdivisions + 1  # Number of points per edge

    ### Generate grid points
    coords_1d = torch.linspace(-size / 2, size / 2, n, device=device)
    x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    ### Generate tetrahedra using Kuhn triangulation (6 tets per cube)
    # The Kuhn triangulation splits each cube into 6 tetrahedra based on the
    # 6 permutations of (x, y, z). Each tet connects v0 to v7 through a path
    # that increments one coordinate at a time. This decomposition naturally
    # produces matching triangular faces between adjacent cubes.

    # Create all (i, j, k) cell indices via meshgrid
    cell_idx = torch.arange(subdivisions, device=device)
    ii, jj, kk = torch.meshgrid(cell_idx, cell_idx, cell_idx, indexing="ij")
    ii, jj, kk = ii.flatten(), jj.flatten(), kk.flatten()  # Each: (num_cubes,)

    # Compute all 8 vertex indices for all cubes at once
    # Vertex ordering: v0=(i,j,k), v1=(i+1,j,k), v2=(i,j+1,k), etc.
    v0 = ii * n * n + jj * n + kk
    v1 = (ii + 1) * n * n + jj * n + kk
    v2 = ii * n * n + (jj + 1) * n + kk
    v3 = (ii + 1) * n * n + (jj + 1) * n + kk
    v4 = ii * n * n + jj * n + (kk + 1)
    v5 = (ii + 1) * n * n + jj * n + (kk + 1)
    v6 = ii * n * n + (jj + 1) * n + (kk + 1)
    v7 = (ii + 1) * n * n + (jj + 1) * n + (kk + 1)

    cube_verts = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1)  # (num_cubes, 8)

    ### Kuhn triangulation: 6 tetrahedra per cube
    # Each tet corresponds to one of the 6 permutations of incrementing (x,y,z).
    # All tets share the body diagonal v0-v7, ensuring consistent face diagonals.
    tet_pattern = torch.tensor(
        [
            [0, 1, 3, 7],  # perm (x, y, z): v0 -> v1 -> v3 -> v7
            [0, 1, 5, 7],  # perm (x, z, y): v0 -> v1 -> v5 -> v7
            [0, 2, 3, 7],  # perm (y, x, z): v0 -> v2 -> v3 -> v7
            [0, 2, 6, 7],  # perm (y, z, x): v0 -> v2 -> v6 -> v7
            [0, 4, 5, 7],  # perm (z, x, y): v0 -> v4 -> v5 -> v7
            [0, 4, 6, 7],  # perm (z, y, x): v0 -> v4 -> v6 -> v7
        ],
        dtype=torch.int64,
        device=device,
    )

    # Advanced indexing: (num_cubes, 8)[:, (6, 4)] -> (num_cubes, 6, 4)
    cells = cube_verts[:, tet_pattern].reshape(-1, 4)

    return Mesh(points=points, cells=cells)
