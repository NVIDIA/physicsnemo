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

"""Regular tetrahedron surface in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(side_length: float = 1.0, device: torch.device | str = "cpu") -> Mesh:
    """Create a regular tetrahedron surface in 3D space.

    Parameters
    ----------
    side_length : float
        Length of each edge.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    # Regular tetrahedron vertices
    # Place center at origin
    a = side_length / (2**0.5)

    vertices = [
        [a, 0, -a / (2**0.5)],
        [-a, 0, -a / (2**0.5)],
        [0, a, a / (2**0.5)],
        [0, -a, a / (2**0.5)],
    ]

    # 4 triangular faces
    faces = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ]

    points = torch.tensor(vertices, dtype=torch.float32, device=device)
    cells = torch.tensor(faces, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
