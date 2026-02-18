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

"""Zigzag polyline in 2D space.

Dimensional: 1D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    n_segments: int = 10,
    amplitude: float = 0.5,
    wavelength: float = 1.0,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a zigzag polyline in 2D space.

    Parameters
    ----------
    n_segments : int
        Number of segments in the polyline.
    amplitude : float
        Amplitude of the zigzag.
    wavelength : float
        Wavelength of the zigzag pattern.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=2, n_cells=n_segments.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be at least 1, got {n_segments=}")

    n_points = n_segments + 1
    x = torch.linspace(0, n_segments * wavelength, n_points, device=device)
    y = amplitude * (2 * (torch.arange(n_points, device=device) % 2) - 1).float()

    points = torch.stack([x, y], dim=1)

    # Create edge cells
    cells = torch.stack(
        [
            torch.arange(n_segments, device=device),
            torch.arange(1, n_segments + 1, device=device),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
