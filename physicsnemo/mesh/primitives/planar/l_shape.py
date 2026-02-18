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

"""L-shaped domain triangulated in 2D space.

Dimensional: 2D manifold in 2D space (non-convex).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    size: float = 1.0, subdivisions: int = 5, device: torch.device | str = "cpu"
) -> Mesh:
    """Create an L-shaped non-convex domain in 2D space.

    The L-shape consists of:
    - Bottom rectangle: [0, size] x [0, size/2]
    - Top rectangle: [0, size/2] x [size/2, size]

    Both parts use uniform grid spacing of size/(2*subdivisions), and the
    vertices at y=size/2 for x in [0, size/2] are shared between the parts.

    Parameters
    ----------
    size : float
        Size of the L-shape (both overall width and height).
    subdivisions : int
        Number of subdivisions per half-edge (so the full width has
        2*subdivisions cells).
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be at least 1, got {subdivisions=}")

    ### Grid parameters
    step = size / (2 * subdivisions)
    n_cols_bottom = 2 * subdivisions + 1  # x points spanning [0, size]
    n_cols_top = subdivisions + 1  # x points spanning [0, size/2]
    n_rows = subdivisions + 1  # y points per rectangle half

    points = []
    cells = []

    ### Bottom rectangle vertices: x in [0, size], y in [0, size/2]
    for i in range(n_cols_bottom):
        for j in range(n_rows):
            x = i * step
            y = j * step
            points.append([x, y])

    ### Top rectangle vertices: x in [0, size/2], y in (size/2, size]
    # Skip y=size/2 (j=0) since those vertices are shared with the bottom part
    for i in range(n_cols_top):
        for j in range(1, n_rows):
            x = i * step
            y = size / 2 + j * step
            points.append([x, y])

    points = torch.tensor(points, dtype=torch.float32, device=device)

    ### Triangulate bottom rectangle
    for i in range(n_cols_bottom - 1):
        for j in range(n_rows - 1):
            idx = i * n_rows + j
            cells.append([idx, idx + 1, idx + n_rows])
            cells.append([idx + 1, idx + n_rows + 1, idx + n_rows])

    ### Triangulate top rectangle
    # The bottom row of cells connects to shared vertices from the bottom part
    offset = n_cols_bottom * n_rows  # Start index of top-only vertices
    n_top_rows = n_rows - 1  # Rows per column in top-only vertex storage

    for i in range(n_cols_top - 1):
        for j in range(subdivisions):
            if j == 0:
                # Bottom row: reference shared vertices from bottom part
                # Shared vertices are at y=size/2 (j=subdivisions in bottom grid)
                bl = i * n_rows + subdivisions
                br = (i + 1) * n_rows + subdivisions
                # Top vertices are first row of top-only part
                tl = offset + i * n_top_rows
                tr = offset + (i + 1) * n_top_rows
            else:
                # Interior rows: all vertices in top-only part
                bl = offset + i * n_top_rows + (j - 1)
                br = offset + (i + 1) * n_top_rows + (j - 1)
                tl = offset + i * n_top_rows + j
                tr = offset + (i + 1) * n_top_rows + j

            cells.append([bl, tl, br])
            cells.append([tl, tr, br])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
