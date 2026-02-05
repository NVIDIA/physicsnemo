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

"""Remove duplicate vertices from meshes.

Merges vertices that are coincident within a tolerance and updates cell
connectivity accordingly.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._cache import CACHE_KEY
from physicsnemo.mesh.utilities._duplicate_detection import compute_canonical_indices

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def remove_duplicate_vertices(
    mesh: "Mesh",
    tolerance: float = 1e-6,
) -> tuple["Mesh", dict[str, int]]:
    """Merge coincident vertices and update cell connectivity.

    Identifies pairs of vertices closer than tolerance and merges them,
    updating all cell references to use the merged vertex indices.

    Parameters
    ----------
    mesh : Mesh
        Input mesh
    tolerance : float
        Distance threshold for considering vertices duplicates

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (cleaned_mesh, stats_dict) where stats_dict contains:
        - "n_duplicates_merged": Number of duplicate vertices merged
        - "n_points_original": Original number of points
        - "n_points_final": Final number of points

    Notes
    -----
    Uses BVH spatial data structure (via ``compute_canonical_indices``)
    for O(n log n) complexity.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh_clean, stats = remove_duplicate_vertices(mesh, tolerance=1e-6)
    >>> assert mesh_clean.validate()["valid"]
    """
    n_original = mesh.n_points
    device = mesh.points.device

    no_change_stats = {
        "n_duplicates_merged": 0,
        "n_points_original": n_original,
        "n_points_final": n_original,
    }

    if n_original <= 1:
        return mesh, no_change_stats

    ### Compute canonical representative for each point
    canonical = compute_canonical_indices(mesh.points, tolerance)

    ### Determine unique representatives and build compact remapping
    unique_canonical = torch.unique(canonical)
    n_unique = len(unique_canonical)
    n_merged = n_original - n_unique

    if n_merged == 0:
        return mesh, no_change_stats

    old_to_new = torch.empty(n_original, device=device, dtype=torch.long)
    old_to_new[unique_canonical] = torch.arange(
        n_unique, device=device, dtype=torch.long
    )
    old_to_new = old_to_new[canonical]

    ### Build cleaned mesh
    from tensordict import TensorDict

    from physicsnemo.mesh.mesh import Mesh

    new_points = mesh.points[unique_canonical]
    new_cells = old_to_new[mesh.cells]

    point_data_filtered = mesh.point_data.exclude(CACHE_KEY)
    new_point_data = TensorDict(
        point_data_filtered[unique_canonical], batch_size=[n_unique]
    )
    new_cell_data = TensorDict(
        mesh.cell_data.exclude(CACHE_KEY), batch_size=mesh.cell_data.batch_size
    )
    new_global_data = TensorDict(
        mesh.global_data, batch_size=mesh.global_data.batch_size
    )

    cleaned_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )

    stats = {
        "n_duplicates_merged": n_merged,
        "n_points_original": n_original,
        "n_points_final": n_unique,
    }

    return cleaned_mesh, stats
