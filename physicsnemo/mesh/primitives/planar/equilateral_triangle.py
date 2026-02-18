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

"""Equilateral triangle triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    side_length: float = 1.0,
    subdivisions: int = 0,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create an equilateral triangle in 2D space.

    Parameters
    ----------
    side_length : float
        Length of each side.
    subdivisions : int
        Number of subdivision levels. Each level quadruples the number of
        triangles: 0 → 1, 1 → 4, 2 → 16, etc.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions=}")

    # Create vertices of equilateral triangle
    height = side_length * (3**0.5) / 2
    points = torch.tensor(
        [[0.0, 0.0], [side_length, 0.0], [side_length / 2, height]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)

    mesh = Mesh(points=points, cells=cells)

    # Apply subdivisions if requested
    if subdivisions > 0:
        mesh = mesh.subdivide(levels=subdivisions, filter="linear")

    return mesh
