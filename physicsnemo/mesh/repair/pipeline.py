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

"""Comprehensive mesh repair pipeline.

Combines multiple repair operations into a single convenient function.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def repair_mesh(
    mesh: "Mesh",
    merge_points: bool = True,
    remove_degenerates: bool = True,
    remove_isolated: bool = True,
    fix_orientation: bool = False,  # Requires 3D, has loops
    fill_holes: bool = False,  # Expensive, opt-in
    tolerance: float = 1e-6,
    area_tolerance: float = 1e-10,
    max_hole_edges: int = 10,
) -> tuple["Mesh", dict[str, dict]]:
    """Apply multiple repair operations in sequence.

    Applies a series of mesh repair operations to clean up common problems.
    Operations are applied in a specific order to maximize effectiveness.

    Order of operations:
    1. Remove degenerate cells (zero area)
    2. Merge duplicate points
    3. Remove isolated points
    4. Fix orientation (if enabled)
    5. Fill holes (if enabled)

    Parameters
    ----------
    mesh : Mesh
        Input mesh to repair.
    merge_points : bool, optional
        Merge coincident points within *tolerance*.
    remove_degenerates : bool, optional
        Remove zero-area cells and cells with duplicate vertices.
    remove_isolated : bool, optional
        Remove points not referenced by any cell.
    fix_orientation : bool, optional
        Ensure consistent face normals (2D in 3D only).
    fill_holes : bool, optional
        Close boundary loops (expensive).
    tolerance : float, optional
        Absolute L2 distance threshold for merging duplicate points.
    area_tolerance : float, optional
        Area threshold for degenerate cell detection. Cells with area
        below this value are considered degenerate and removed. Defaults
        to ``1e-10``, matching :func:`remove_degenerate_cells`'s own default.
    max_hole_edges : int, optional
        Maximum hole size to fill.

    Returns
    -------
    tuple[Mesh, dict[str, dict]]
        Tuple of (repaired_mesh, all_stats) where all_stats is a dict
        mapping operation name to its individual stats dict.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh_clean, stats = repair_mesh(mesh, merge_points=True)
    >>> assert "merge_points" in stats
    """
    current_mesh = mesh
    all_stats = {}

    ### Operation 1: Remove degenerate cells
    if remove_degenerates:
        from physicsnemo.mesh.repair.degenerate_removal import (
            remove_degenerate_cells as remove_deg,
        )

        current_mesh, stats = remove_deg(current_mesh, area_tolerance=area_tolerance)
        all_stats["degenerates"] = stats

    ### Operation 2: Merge duplicate points (via clean_mesh)
    if merge_points:
        from physicsnemo.mesh.repair._cleaning import clean_mesh

        n_before = current_mesh.n_points
        current_mesh, clean_stats = clean_mesh(
            current_mesh,
            tolerance=tolerance,
            merge_points=True,
            deduplicate_cells=False,
            drop_unused_points=False,
        )
        n_after = current_mesh.n_points
        all_stats["merge_points"] = {
            "n_points_original": n_before,
            "n_points_final": n_after,
            "n_duplicates_merged": n_before - n_after,
        }
        all_stats["clean"] = clean_stats

    ### Operation 3: Remove isolated points
    if remove_isolated:
        from physicsnemo.mesh.repair._cleaning import (
            remove_isolated_points as remove_iso,
        )

        current_mesh, stats = remove_iso(current_mesh)
        all_stats["isolated"] = stats

    ### Operation 4: Fix orientation
    if fix_orientation:
        if current_mesh.n_manifold_dims == 2 and current_mesh.n_spatial_dims == 3:
            from physicsnemo.mesh.repair.orientation import (
                fix_orientation as fix_orient,
            )

            current_mesh, stats = fix_orient(current_mesh)
            all_stats["orientation"] = stats
        else:
            all_stats["orientation"] = {"skipped": "Only for 2D manifolds in 3D"}

    ### Operation 5: Fill holes
    if fill_holes:
        if current_mesh.n_manifold_dims == 2:
            from physicsnemo.mesh.repair.hole_filling import fill_holes as fill_h

            current_mesh, stats = fill_h(current_mesh, max_hole_edges=max_hole_edges)
            all_stats["holes"] = stats
        else:
            all_stats["holes"] = {"skipped": "Only for 2D manifolds"}

    return current_mesh, all_stats
