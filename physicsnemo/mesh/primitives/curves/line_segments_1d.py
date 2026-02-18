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

"""Multiple disconnected line segments in 1D space.

Dimensional: 1D manifold in 1D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    n_segments: int = 3, gap: float = 0.2, device: torch.device | str = "cpu"
) -> Mesh:
    """Create multiple disconnected line segments in 1D space.

    Parameters
    ----------
    n_segments : int
        Number of disconnected segments.
    gap : float
        Gap between segments.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=1, n_cells=n_segments.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be at least 1, got {n_segments=}")

    points = []
    cells = []

    for i in range(n_segments):
        start = i * (1.0 + gap)
        end = start + 1.0
        points.extend([[start], [end]])
        cells.append([2 * i, 2 * i + 1])

    points = torch.tensor(points, dtype=torch.float32, device=device)
    cells = torch.tensor(cells, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
