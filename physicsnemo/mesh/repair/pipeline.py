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
    fix_orientation: bool = False,
    fill_holes: bool = False,  # Expensive, opt-in
    tolerance: float | None = None,
    area_tolerance: float | None = None,
    max_hole_edges: int = 10,
    remove_duplicates: bool = True,
    stitch_t_junctions: bool = True,
    split_nonmanifold_edges: bool = False,
) -> tuple["Mesh", dict[str, dict]]:
    """Apply multiple repair operations in sequence.

    Applies a series of mesh repair operations to clean up common problems.
    Operations are applied in a specific order to maximize effectiveness.

    Order of operations:
    1. Merge duplicate points
    2. Remove duplicate cells (triangle meshes): copies with the same vertices
       and the same winding
    3. Remove degenerate cells (zero area), including those the merge collapses
    4. Stitch T-junctions (triangle meshes), which also closes the slits that
       removing a zero-area cell opens
    5. Split non-manifold edges and drop fins, pillows and reversed copies of
       faces (triangle meshes in 3D, if enabled)
    6. Remove isolated points
    7. Fix orientation (if enabled)
    8. Fill holes (if enabled)

    Parameters
    ----------
    mesh : Mesh
        Input mesh to repair.
    merge_points : bool, optional
        Merge coincident points within ``tolerance``.
    remove_degenerates : bool, optional
        Remove zero-area cells and cells with duplicate vertices.
    remove_isolated : bool, optional
        Remove points not referenced by any cell.
    fix_orientation : bool, optional
        Orient all cells consistently (cells with at least two vertices). Each
        connected component follows the winding of its lowest-index cell,
        except that cells filling their space, such as tetrahedra in 3D, get
        positive volumes. See :func:`~physicsnemo.mesh.repair.fix_orientation`.
    fill_holes : bool, optional
        Close boundary loops (expensive).
    tolerance : float, optional
        Absolute L2 distance threshold for merging duplicate points, for
        stitching T-junctions, and for the slit and planarity tests of hole
        filling. Defaults to ``max(1e-6, 100 * eps)`` times the
        bounding-box diagonal, where ``eps`` is the machine epsilon of the
        point dtype, as in :func:`~physicsnemo.mesh.repair.stitch_t_junctions`
        (before, it defaulted to an absolute ``1e-6``).
    area_tolerance : float, optional
        Area threshold for degenerate cell detection. Cells with area (the
        measure of the cell, in its own dimension) below this value are
        considered degenerate and removed. Defaults to ``tolerance`` raised to
        the manifold dimension (before, it defaulted to an absolute ``1e-10``).
    max_hole_edges : int, optional
        Maximum hole size to fill.
    remove_duplicates : bool, optional
        Remove the later copies of triangles with the same vertices and the
        same winding. Reversed copies are left to
        :func:`~physicsnemo.mesh.repair.split_nonmanifold_edges`, which keeps
        the copy wound like the surface; where it does not run (triangle
        meshes in 2D, or ``split_nonmanifold_edges=False``), they survive.
    stitch_t_junctions : bool, optional
        Split triangles at the vertices lying on their open edges. See
        :func:`~physicsnemo.mesh.repair.stitch_t_junctions`.
    split_nonmanifold_edges : bool, optional
        Split edges used by three or more triangles into manifold wedges and
        drop fins. Off by default, since it deletes geometry: thin open sheets
        attached along an edge, such as baffles, zero-thickness plates and
        splitter plates, are dropped as fins too. See
        :func:`~physicsnemo.mesh.repair.split_nonmanifold_edges`.

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

    A cube surface with one face doubled, wound the other way, keeps its own
    face and drops the copy:

    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> from physicsnemo.mesh.primitives.surfaces import cube_surface
    >>> cube = cube_surface.load()
    >>> copy = cube.cells[:2, [0, 2, 1]]
    >>> doubled = Mesh(points=cube.points, cells=torch.cat([copy, cube.cells]))
    >>> repaired, stats = repair_mesh(doubled, split_nonmanifold_edges=True)
    >>> stats["nonmanifold_edges"]["n_cells_dropped"]
    2
    >>> torch.equal(repaired.cells, cube.cells)
    True
    """
    from physicsnemo.mesh.utilities._tolerances import default_length_tolerance

    current_mesh = mesh
    all_stats = {}
    is_triangle_mesh = mesh.n_manifold_dims == 2
    if tolerance is None:
        tolerance = default_length_tolerance(mesh.points)
    if area_tolerance is None:
        area_tolerance = tolerance**mesh.n_manifold_dims

    ### Operation 1: Merge duplicate points (via clean_mesh)
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

    ### Operation 2: Remove duplicate cells with the same winding
    # remove_duplicate_cells keeps the first copy whatever its winding, so the
    # parity of each triangle's vertex order rides along in its indices, as
    # 2 * index + parity. Three such indices pack into one int64 key up to
    # about 1M points; beyond, remove_duplicate_cells takes the slower
    # row-wise unique, with the same result.
    if remove_duplicates:
        if is_triangle_mesh:
            from physicsnemo.mesh.mesh import Mesh
            from physicsnemo.mesh.repair._cleaning import remove_duplicate_cells

            cells = current_mesh.cells
            a, b, c = cells.unbind(dim=1)
            parity = ((a > b).long() + (a > c).long() + (b > c).long()) % 2
            n_points = current_mesh.n_points
            unique_cells, cell_data = remove_duplicate_cells(
                2 * cells + parity[:, None],
                current_mesh.cell_data,
                index_bound=2 * n_points,
            )
            all_stats["duplicates"] = {
                "n_cells_original": current_mesh.n_cells,
                "n_cells_final": unique_cells.shape[0],
            }
            if unique_cells.shape[0] < current_mesh.n_cells:
                current_mesh = Mesh(
                    points=current_mesh.points,
                    cells=unique_cells // 2,
                    point_data=current_mesh.point_data,
                    cell_data=cell_data,
                    global_data=current_mesh.global_data,
                )
        else:
            all_stats["duplicates"] = {"skipped": "Only for triangle meshes"}

    ### Operation 3: Remove degenerate cells
    if remove_degenerates:
        from physicsnemo.mesh.repair.degenerate_removal import (
            remove_degenerate_cells as remove_deg,
        )

        current_mesh, stats = remove_deg(current_mesh, area_tolerance=area_tolerance)
        all_stats["degenerates"] = stats

    ### Operation 4: Stitch T-junctions
    if stitch_t_junctions:
        if is_triangle_mesh:
            from physicsnemo.mesh.repair._stitching import (
                stitch_t_junctions as stitch,
            )

            current_mesh, stats = stitch(current_mesh, tolerance=tolerance)
            all_stats["t_junctions"] = stats
        else:
            all_stats["t_junctions"] = {"skipped": "Only for triangle meshes"}

    ### Operation 5: Split non-manifold edges, dropping fins
    if split_nonmanifold_edges:
        if is_triangle_mesh and current_mesh.n_spatial_dims == 3:
            from physicsnemo.mesh.repair.nonmanifold_edges import (
                split_nonmanifold_edges as split_edges,
            )

            current_mesh, stats = split_edges(current_mesh)
            all_stats["nonmanifold_edges"] = stats
        else:
            all_stats["nonmanifold_edges"] = {
                "skipped": "Only for triangle meshes in 3D"
            }

    ### Operation 6: Remove isolated points
    if remove_isolated:
        from physicsnemo.mesh.repair._cleaning import (
            remove_isolated_points as remove_iso,
        )

        current_mesh, stats = remove_iso(current_mesh)
        all_stats["isolated"] = stats

    ### Operation 7: Fix orientation
    if fix_orientation:
        if current_mesh.n_manifold_dims >= 1:
            from physicsnemo.mesh.repair.orientation import (
                fix_orientation as fix_orient,
            )

            current_mesh, stats = fix_orient(current_mesh)
            all_stats["orientation"] = stats
        else:
            all_stats["orientation"] = {"skipped": "Points have no orientation"}

    ### Operation 8: Fill holes
    if fill_holes:
        if current_mesh.n_manifold_dims == 2:
            from physicsnemo.mesh.repair.hole_filling import fill_holes as fill_h

            current_mesh, stats = fill_h(
                current_mesh, max_hole_edges=max_hole_edges, tolerance=tolerance
            )
            all_stats["holes"] = stats
        else:
            all_stats["holes"] = {"skipped": "Only for 2D manifolds"}

    return current_mesh, all_stats
