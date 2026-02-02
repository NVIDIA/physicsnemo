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

"""Remove isolated vertices from meshes.

Removes vertices that are not referenced by any cell.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def remove_isolated_vertices(
    mesh: "Mesh",
) -> tuple["Mesh", dict[str, int]]:
    """Remove vertices not appearing in any cell.

    Identifies vertices not referenced by any cell and removes them,
    updating cell indices accordingly.

    Args:
        mesh: Input mesh

    Returns:
        Tuple of (cleaned_mesh, stats_dict) where stats_dict contains:
        - "n_isolated_removed": Number of isolated vertices removed
        - "n_points_original": Original number of points
        - "n_points_final": Final number of points

    Example:
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> mesh_clean, stats = remove_isolated_vertices(mesh)
        >>> assert stats["n_isolated_removed"] == 0  # no isolated in clean mesh
    """
    n_original = mesh.n_points
    device = mesh.points.device

    if n_original == 0 or mesh.n_cells == 0:
        return mesh, {
            "n_isolated_removed": 0,
            "n_points_original": n_original,
            "n_points_final": n_original,
        }

    ### Find vertices that appear in at least one cell
    used_vertices = torch.unique(mesh.cells.flatten())
    n_used = len(used_vertices)
    n_isolated = n_original - n_used

    if n_isolated == 0:
        # No isolated vertices
        return mesh, {
            "n_isolated_removed": 0,
            "n_points_original": n_original,
            "n_points_final": n_original,
        }

    ### Create mapping from old to new indices
    old_to_new = torch.full((n_original,), -1, device=device, dtype=torch.long)
    old_to_new[used_vertices] = torch.arange(n_used, device=device, dtype=torch.long)

    ### Build new mesh
    new_points = mesh.points[used_vertices]
    new_cells = old_to_new[mesh.cells]

    ### Transfer data (excluding cache)
    new_point_data = mesh.point_data.exclude("_cache")[used_vertices]
    new_cell_data = mesh.cell_data.exclude("_cache").clone()
    new_global_data = mesh.global_data.clone()

    from physicsnemo.mesh.mesh import Mesh

    cleaned_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )

    stats = {
        "n_isolated_removed": n_isolated,
        "n_points_original": n_original,
        "n_points_final": n_used,
    }

    return cleaned_mesh, stats
