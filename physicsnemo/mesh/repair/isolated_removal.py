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

Thin wrapper around :func:`physicsnemo.mesh.boundaries._cleaning.remove_unused_points`
that accepts and returns :class:`Mesh` objects with statistics.
"""

from typing import TYPE_CHECKING

from physicsnemo.mesh.boundaries._cleaning import remove_unused_points
from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def remove_isolated_vertices(
    mesh: "Mesh",
) -> tuple["Mesh", dict[str, int]]:
    """Remove vertices not appearing in any cell.

    Identifies vertices not referenced by any cell and removes them,
    updating cell indices accordingly. Delegates to
    :func:`~physicsnemo.mesh.boundaries._cleaning.remove_unused_points`
    for the core computation.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (cleaned_mesh, stats_dict) where stats_dict contains:
        - "n_isolated_removed": Number of isolated vertices removed
        - "n_points_original": Original number of points
        - "n_points_final": Final number of points

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh_clean, stats = remove_isolated_vertices(mesh)
    >>> assert stats["n_isolated_removed"] == 0  # no isolated in clean mesh
    """
    n_original = mesh.n_points

    ### Delegate to the tensor-level primitive in _cleaning
    new_points, new_cells, new_point_data, _ = remove_unused_points(
        points=mesh.points,
        cells=mesh.cells,
        point_data=mesh.point_data.exclude(CACHE_KEY),
    )

    n_final = new_points.shape[0]
    n_isolated = n_original - n_final

    ### Short-circuit if nothing changed
    if n_isolated == 0:
        return mesh, {
            "n_isolated_removed": 0,
            "n_points_original": n_original,
            "n_points_final": n_original,
        }

    ### Build cleaned mesh
    from physicsnemo.mesh.mesh import Mesh

    cleaned_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=mesh.cell_data.exclude(CACHE_KEY).clone(),
        global_data=mesh.global_data.clone(),
    )

    return cleaned_mesh, {
        "n_isolated_removed": n_isolated,
        "n_points_original": n_original,
        "n_points_final": n_final,
    }
