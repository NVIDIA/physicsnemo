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

"""Mesh cleaning operations.

This module provides functions to clean and repair meshes:
- Merge duplicate points within tolerance
- Remove duplicate cells
- Remove unused points
- Remove isolated points (mesh-level wrapper)
"""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from physicsnemo.mesh.utilities._cache import CACHE_KEY
from physicsnemo.mesh.utilities._duplicate_detection import compute_canonical_indices
from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def merge_duplicate_points(
    points: torch.Tensor,  # shape: (n_points, n_spatial_dims)
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
    point_data: TensorDict,
    tolerance: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, TensorDict, torch.Tensor]:
    """Merge duplicate points within tolerance.

    Points whose L2 distance is below *tolerance* are merged into a single
    representative, and cell connectivity is updated accordingly.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape (n_points, n_spatial_dims).
    cells : torch.Tensor
        Cell connectivity, shape (n_cells, n_vertices_per_cell).
    point_data : TensorDict
        Point data to merge (values are averaged across merged groups).
    tolerance : float, optional
        Absolute L2 distance threshold for considering two points as
        duplicates.

    Returns
    -------
    merged_points : torch.Tensor
        Deduplicated points, shape (n_unique_points, n_spatial_dims).
    updated_cells : torch.Tensor
        Updated cell connectivity, shape (n_cells, n_vertices_per_cell).
    merged_point_data : TensorDict
        Averaged point data for merged points.
    point_mapping : torch.Tensor
        Mapping from old to new point indices, shape (n_points,).

    Examples
    --------
    >>> import torch
    >>> from tensordict import TensorDict
    >>> # Two points at same location
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 0.]])
    >>> cells = torch.tensor([[0, 1], [1, 2]])
    >>> merged_points, updated_cells, _, mapping = merge_duplicate_points(
    ...     points, cells, TensorDict({}, batch_size=[3])
    ... )
    >>> # Points 0 and 2 are merged
    >>> assert len(merged_points) == 2
    >>> assert torch.equal(mapping, torch.tensor([0, 1, 0]))
    """
    n_points = len(points)
    device = points.device

    if n_points == 0:
        return (
            points,
            cells,
            point_data,
            torch.arange(0, device=device, dtype=torch.int64),
        )

    ### Compute canonical indices via shared BVH-based primitive
    point_mapping = compute_canonical_indices(points, tolerance)

    ### Get unique points and remap connectivity
    unique_indices = torch.unique(point_mapping)
    n_unique = len(unique_indices)

    ### Create reverse mapping from old unique indices to new compact indices
    reverse_mapping = torch.zeros(n_points, device=device, dtype=torch.int64)
    reverse_mapping[unique_indices] = torch.arange(
        n_unique, device=device, dtype=torch.int64
    )

    ### Apply reverse mapping to point_mapping to get final compact indices
    final_point_mapping = reverse_mapping[point_mapping]

    ### Extract merged points
    merged_points = points[unique_indices]

    ### Update cell connectivity
    updated_cells = final_point_mapping[cells]

    ### Merge point data by averaging
    merged_point_data = _merge_point_data(
        point_data=point_data,
        point_mapping=point_mapping,
        unique_indices=unique_indices,
        n_unique=n_unique,
    )

    return merged_points, updated_cells, merged_point_data, final_point_mapping


def _merge_point_data(
    point_data: TensorDict,
    point_mapping: torch.Tensor,
    unique_indices: torch.Tensor,
    n_unique: int,
) -> TensorDict:
    """Merge point data by averaging over merged points.

    Parameters
    ----------
    point_data : TensorDict
        Original point data
    point_mapping : torch.Tensor
        Mapping from original to merged points
    unique_indices : torch.Tensor
        Indices of unique points in original array
    n_unique : int
        Number of unique points

    Returns
    -------
    TensorDict
        Merged point data
    """
    if len(point_data.keys()) == 0:
        return TensorDict(
            {},
            batch_size=torch.Size([n_unique]),
            device=point_data.device,
        )

    ### Create reverse mapping: unique_indices[i] corresponds to output index i
    device = point_mapping.device
    reverse_map = torch.zeros(len(point_mapping), dtype=torch.int64, device=device)
    reverse_map[unique_indices] = torch.arange(
        n_unique, device=device, dtype=torch.int64
    )

    ### Get output indices for all input points
    output_indices = reverse_map[point_mapping]

    ### For each unique point, average the data from all points that map to it
    def _merge_tensor(tensor: torch.Tensor) -> torch.Tensor:
        ### Use scatter aggregation utility
        return scatter_aggregate(
            src_data=tensor,
            src_to_dst_mapping=output_indices,
            n_dst=n_unique,
            weights=None,
            aggregation="mean",
        )

    return point_data.apply(
        _merge_tensor,
        batch_size=torch.Size([n_unique]),
    )


def remove_duplicate_cells(
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
    cell_data: TensorDict,
) -> tuple[torch.Tensor, TensorDict]:
    """Remove duplicate cells from mesh.

    Cells are considered duplicates if they contain the same set of vertex indices
    (regardless of order). When duplicates are found, only the first occurrence is kept.

    Parameters
    ----------
    cells : torch.Tensor
        Cell connectivity, shape (n_cells, n_vertices_per_cell)
    cell_data : TensorDict
        Cell data

    Returns
    -------
    unique_cells : torch.Tensor
        Deduplicated cells, shape (n_unique_cells, n_vertices_per_cell)
    unique_cell_data : TensorDict
        Cell data for unique cells

    Examples
    --------
    >>> import torch
    >>> from tensordict import TensorDict
    >>> # Two cells with same vertices
    >>> cells = torch.tensor([[0, 1, 2], [1, 0, 2], [3, 4, 5]])
    >>> unique_cells, _ = remove_duplicate_cells(
    ...     cells, TensorDict({}, batch_size=[3])
    ... )
    >>> assert len(unique_cells) == 2  # cells 0 and 1 are duplicates
    """
    if len(cells) == 0:
        return cells, cell_data

    ### Sort vertices within each cell to canonical form
    sorted_cells = torch.sort(cells, dim=-1)[0]

    ### Find unique cells using vectorized first-occurrence detection
    n_cells = len(cells)
    device = cells.device

    ### Use torch.unique to identify duplicate groups
    # inverse_indices maps each cell to its unique group
    _, inverse_indices = torch.unique(
        sorted_cells,
        dim=0,
        return_inverse=True,
    )

    ### Vectorized first-occurrence detection
    # Sort cell indices by their inverse_indices to group duplicates together
    # Then mark only the first cell in each group
    sorted_order = torch.argsort(inverse_indices, stable=True)
    sorted_inverse = inverse_indices[sorted_order]

    # Find group boundaries: where the group ID changes
    # First element is always a boundary (first occurrence)
    is_first_in_group = torch.cat(
        [
            torch.tensor([True], device=device),
            sorted_inverse[1:] != sorted_inverse[:-1],
        ]
    )

    # Map back to original indices: first_occurrence_indices are the cells to keep
    first_occurrence_indices = sorted_order[is_first_in_group]

    # Build keep_mask from first occurrences
    keep_mask = torch.zeros(n_cells, dtype=torch.bool, device=device)
    keep_mask[first_occurrence_indices] = True

    ### Filter cells and data
    unique_cells = cells[keep_mask]
    unique_cell_data = (
        cell_data[keep_mask]
        if len(cell_data.keys()) > 0
        else TensorDict(
            {},
            batch_size=torch.Size([keep_mask.sum().item()]),
            device=cell_data.device,
        )
    )

    return unique_cells, unique_cell_data


def remove_unused_points(
    points: torch.Tensor,  # shape: (n_points, n_spatial_dims)
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
    point_data: TensorDict,
) -> tuple[torch.Tensor, torch.Tensor, TensorDict, torch.Tensor]:
    """Remove points that are not referenced by any cell.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape (n_points, n_spatial_dims)
    cells : torch.Tensor
        Cell connectivity, shape (n_cells, n_vertices_per_cell)
    point_data : TensorDict
        Point data

    Returns
    -------
    used_points : torch.Tensor
        Points that are used by cells, shape (n_used_points, n_spatial_dims)
    updated_cells : torch.Tensor
        Updated cell connectivity, shape (n_cells, n_vertices_per_cell)
    used_point_data : TensorDict
        Point data for used points
    point_mapping : torch.Tensor
        Mapping from old to new point indices, shape (n_points,)
        Unused points map to -1

    Examples
    --------
    >>> import torch
    >>> from tensordict import TensorDict
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [2., 2.]])
    >>> cells = torch.tensor([[0, 1, 2]])  # Point 3 is unused
    >>> used_points, updated_cells, _, mapping = remove_unused_points(
    ...     points, cells, TensorDict({}, batch_size=[4])
    ... )
    >>> assert len(used_points) == 3
    >>> assert torch.equal(mapping, torch.tensor([0, 1, 2, -1]))
    """
    n_points = len(points)
    device = points.device

    if len(cells) == 0:
        ### No cells means no points are used
        return (
            torch.empty((0, points.shape[1]), dtype=points.dtype, device=device),
            cells,
            TensorDict({}, batch_size=torch.Size([0]), device=device),
            torch.full((n_points,), -1, dtype=torch.int64, device=device),
        )

    ### Find which points are used by cells
    used_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    used_mask.scatter_(0, cells.flatten(), True)

    ### Get indices of used points
    used_indices = torch.where(used_mask)[0]
    n_used = len(used_indices)

    ### Create mapping from old to new indices
    point_mapping = torch.full((n_points,), -1, dtype=torch.int64, device=device)
    point_mapping[used_indices] = torch.arange(n_used, device=device, dtype=torch.int64)

    ### Extract used points and data
    used_points = points[used_indices]
    used_point_data = (
        point_data[used_indices]
        if len(point_data.keys()) > 0
        else TensorDict(
            {},
            batch_size=torch.Size([n_used]),
            device=device,
        )
    )

    ### Update cell connectivity
    updated_cells = point_mapping[cells]

    return used_points, updated_cells, used_point_data, point_mapping


def clean_mesh(
    mesh: "Mesh",
    tolerance: float = 1e-12,
    merge_points: bool = True,
    deduplicate_cells: bool = True,
    drop_unused_points: bool = True,
) -> tuple["Mesh", dict]:
    """Clean and repair a mesh.

    Performs various cleaning operations to fix common mesh issues:
    1. Merge duplicate points within tolerance
    2. Remove duplicate cells
    3. Remove unused points

    Parameters
    ----------
    mesh : Mesh
        Input mesh to clean.
    tolerance : float, optional
        Absolute L2 distance threshold for merging duplicate points.
    merge_points : bool, optional
        Whether to merge duplicate points.
    deduplicate_cells : bool, optional
        Whether to remove duplicate cells.
    drop_unused_points : bool, optional
        Whether to remove unused points.

    Returns
    -------
    tuple[Mesh, dict]
        Tuple of (cleaned_mesh, stats) where stats tracks what was done:
        - ``"n_points_before_merge"`` / ``"n_points_after_merge"``
        - ``"n_cells_before_dedup"`` / ``"n_cells_after_dedup"``
        - ``"n_points_before_drop"`` / ``"n_points_after_drop"``

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Mesh with duplicate points
    >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [1., 1.]])
    >>> cells = torch.tensor([[0, 1, 3], [2, 1, 3]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> cleaned, stats = clean_mesh(mesh)
    >>> assert cleaned.n_points == 3  # points 0 and 2 merged
    """
    points = mesh.points
    cells = mesh.cells
    point_data = mesh.point_data.exclude(CACHE_KEY)
    cell_data = mesh.cell_data.exclude(CACHE_KEY)
    global_data = mesh.global_data
    stats: dict = {}

    ### Step 1: Merge duplicate points
    if merge_points:
        n_before = points.shape[0]
        points, cells, point_data, _ = merge_duplicate_points(
            points=points,
            cells=cells,
            point_data=point_data,
            tolerance=tolerance,
        )
        stats["n_points_before_merge"] = n_before
        stats["n_points_after_merge"] = points.shape[0]

    ### Step 2: Remove duplicate cells
    if deduplicate_cells:
        n_before = cells.shape[0]
        cells, cell_data = remove_duplicate_cells(
            cells=cells,
            cell_data=cell_data,
        )
        stats["n_cells_before_dedup"] = n_before
        stats["n_cells_after_dedup"] = cells.shape[0]

    ### Step 3: Remove unused points
    if drop_unused_points:
        n_before = points.shape[0]
        points, cells, point_data, _ = remove_unused_points(
            points=points,
            cells=cells,
            point_data=point_data,
        )
        stats["n_points_before_drop"] = n_before
        stats["n_points_after_drop"] = points.shape[0]

    ### Create cleaned mesh
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=global_data,
    ), stats


def remove_isolated_points(
    mesh: "Mesh",
) -> tuple["Mesh", dict[str, int]]:
    """Remove points not appearing in any cell.

    Identifies points not referenced by any cell and removes them,
    updating cell indices accordingly. Delegates to
    :func:`~physicsnemo.mesh.repair._cleaning.remove_unused_points`
    for the core computation.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (cleaned_mesh, stats_dict) where stats_dict contains:
        - "n_isolated_removed": Number of isolated points removed
        - "n_points_original": Original number of points
        - "n_points_final": Final number of points

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh_clean, stats = remove_isolated_points(mesh)
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
