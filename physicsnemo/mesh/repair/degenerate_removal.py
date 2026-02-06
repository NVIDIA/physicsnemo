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

"""Remove degenerate cells from meshes.

Removes cells with zero or near-zero area/volume, and cells with duplicate vertices.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def remove_degenerate_cells(
    mesh: "Mesh",
    area_tolerance: float = 1e-10,
) -> tuple["Mesh", dict[str, int]]:
    """Remove cells with area < tolerance or duplicate vertices.

    Identifies and removes degenerate cells that have:
    1. Area/volume below tolerance (nearly zero or negative)
    2. Duplicate vertex indices (invalid simplices)

    Parameters
    ----------
    mesh : Mesh
        Input mesh
    area_tolerance : float
        Minimum acceptable cell area

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (cleaned_mesh, stats_dict) where stats_dict contains:
        - "n_zero_area_cells": Number of cells removed for zero area
        - "n_duplicate_vertex_cells": Number of cells with duplicate vertices
        - "n_cells_original": Original number of cells
        - "n_cells_final": Final number of cells

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh_clean, stats = remove_degenerate_cells(mesh)
    >>> assert stats["n_cells_final"] == mesh.n_cells  # no degenerates in clean mesh
    """
    n_original = mesh.n_cells

    if n_original == 0:
        return mesh, {
            "n_zero_area_cells": 0,
            "n_duplicate_vertex_cells": 0,
            "n_cells_original": 0,
            "n_cells_final": 0,
        }

    ### Check 1: Zero area cells
    cell_areas = mesh.cell_areas
    non_degenerate_by_area = cell_areas >= area_tolerance
    n_zero_area = (~non_degenerate_by_area).sum().item()

    ### Check 2: Cells with duplicate vertices (vectorized)
    # For each cell, check if all vertices are unique
    # Sort vertices in each cell and check for adjacent duplicates
    cells_sorted = torch.sort(mesh.cells, dim=1).values  # (n_cells, n_verts)

    # Check if any adjacent sorted vertices are equal
    has_duplicates = (cells_sorted[:, 1:] == cells_sorted[:, :-1]).any(dim=-1)

    has_unique_vertices = ~has_duplicates

    n_duplicate_vertex = (~has_unique_vertices).sum().item()

    ### Combined mask: keep cells that are good
    keep_mask = non_degenerate_by_area & has_unique_vertices
    n_keep = keep_mask.sum().item()

    if n_keep == n_original:
        # No degenerate cells
        return mesh, {
            "n_zero_area_cells": 0,
            "n_duplicate_vertex_cells": 0,
            "n_cells_original": n_original,
            "n_cells_final": n_original,
        }

    ### Filter cells
    new_cells = mesh.cells[keep_mask]

    ### Transfer data (excluding cache)
    new_cell_data = mesh.cell_data.exclude(CACHE_KEY)[keep_mask]

    ### Keep all points and point data (will be cleaned by remove_isolated_points if needed)
    from physicsnemo.mesh.mesh import Mesh

    cleaned_mesh = Mesh(
        points=mesh.points,
        cells=new_cells,
        point_data=mesh.point_data.exclude(CACHE_KEY).clone(),
        cell_data=new_cell_data,
        global_data=mesh.global_data.clone(),
    )

    stats = {
        "n_zero_area_cells": n_zero_area,
        "n_duplicate_vertex_cells": n_duplicate_vertex,
        "n_cells_original": n_original,
        "n_cells_final": n_keep,
    }

    return cleaned_mesh, stats
