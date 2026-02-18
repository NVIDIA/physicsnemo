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

"""Cube surface triangulated in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import itertools

import torch

from physicsnemo.mesh.mesh import Mesh


def load(size: float = 1.0, device: torch.device | str = "cpu") -> Mesh:
    """Create a cube surface triangulated in 3D space.

    The cube is centered at the origin with vertices at (±size/2, ±size/2, ±size/2).
    Each face is split into 2 triangles with consistent outward-facing normals
    (counter-clockwise winding when viewed from outside).

    Parameters
    ----------
    size : float
        Side length of the cube.
    device : torch.device or str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3, 8 vertices, 12 triangles.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import cube_surface
    >>> mesh = cube_surface.load()
    >>> mesh.n_points, mesh.n_cells
    (8, 12)
    >>> mesh.n_manifold_dims, mesh.n_spatial_dims
    (2, 3)
    """
    s = size / 2

    # 8 vertices via Cartesian product
    points = torch.tensor(
        list(itertools.product([-s, +s], repeat=3)),
        dtype=torch.float32,
        device=device,
    )

    # 6 face quads with CCW winding for outward normals.
    # Vertex indices from itertools.product([-s, +s], repeat=3):
    #   0=(-,-,-), 1=(-,-,+), 2=(-,+,-), 3=(-,+,+)
    #   4=(+,-,-), 5=(+,-,+), 6=(+,+,-), 7=(+,+,+)
    _FACE_QUADS = (
        (0, 1, 3, 2),  # -X face
        (4, 6, 7, 5),  # +X face
        (0, 4, 5, 1),  # -Y face
        (2, 3, 7, 6),  # +Y face
        (0, 2, 6, 4),  # -Z face
        (1, 5, 7, 3),  # +Z face
    )

    # Triangulate each quad: (v0, v1, v2, v3) → (v0, v1, v2) and (v0, v2, v3)
    cells_list = []
    for q in _FACE_QUADS:
        cells_list.extend(
            [
                [q[0], q[1], q[2]],
                [q[0], q[2], q[3]],
            ]
        )
    cells = torch.tensor(cells_list, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
