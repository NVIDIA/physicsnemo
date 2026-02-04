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

"""Mesh cleaning operations.

This module provides functions to clean and repair meshes:
- Merge duplicate points within tolerance
- Remove duplicate cells
- Remove unused points
"""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _compute_duplicate_mask(
    points: torch.Tensor,  # shape: (n_points, n_spatial_dims)
    rtol: float,
    atol: float,
) -> torch.Tensor:
    """Compute pairwise duplicate mask based on distance tolerance.

    Two points are considered duplicates if:
        ||p1 - p2|| <= atol + rtol * max(||p1||, ||p2||)

    Args:
        points: Point coordinates
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Boolean mask of shape (n_points, n_points) where True indicates duplicates
    """
    ### Compute pairwise distances: ||pi - pj||
    # Shape: (n_points, n_points)
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # (n_points, n_points, n_dims)
    distances = torch.norm(diff, dim=-1)  # (n_points, n_points)

    ### Compute threshold for each pair: atol + rtol * max(||pi||, ||pj||)
    # Shape: (n_points,)
    point_norms = torch.norm(points, dim=-1)

    ### Threshold matrix: atol + rtol * max(||pi||, ||pj||)
    # Use max to ensure symmetry
    threshold_matrix = atol + rtol * torch.maximum(
        point_norms.unsqueeze(1),
        point_norms.unsqueeze(0),
    )

    ### Find duplicate pairs: distance <= threshold
    return distances <= threshold_matrix


def merge_duplicate_points(
    points: torch.Tensor,  # shape: (n_points, n_spatial_dims)
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
    point_data: TensorDict,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, TensorDict, torch.Tensor]:
    """Merge duplicate points within tolerance.

    Points are considered duplicates if ||p1 - p2|| <= atol + rtol * ||p1||.
    When duplicates are found, they are merged into a single point, and cell
    connectivity is updated accordingly.

    Args:
        points: Point coordinates, shape (n_points, n_spatial_dims)
        cells: Cell connectivity, shape (n_cells, n_vertices_per_cell)
        point_data: Point data to merge
        rtol: Relative tolerance for distance comparison
        atol: Absolute tolerance for distance comparison

    Returns:
        merged_points: Deduplicated points, shape (n_unique_points, n_spatial_dims)
        updated_cells: Updated cell connectivity, shape (n_cells, n_vertices_per_cell)
        merged_point_data: Averaged point data for merged points
        point_mapping: Mapping from old to new point indices, shape (n_points,)

    Example:
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

    ### Use pairwise distance computation for small meshes
    # For large meshes, we should use spatial hashing or KD-tree, but those
    # require additional dependencies. For now, we use a vectorized approach
    # that works well up to ~100k points.

    ### Compute pairwise distances efficiently using broadcasting
    # For very large meshes (>100k points), this may run out of memory
    # In such cases, we process in chunks

    chunk_size = 10000  # Process in chunks to avoid OOM
    point_mapping = torch.arange(n_points, device=device, dtype=torch.int64)

    if n_points <= chunk_size:
        ### Small mesh: compute all pairwise distances at once
        point_mapping = _merge_points_pairwise(points, rtol, atol)
    else:
        ### Large mesh: use spatial hashing for efficiency
        point_mapping = _merge_points_spatial_hash(points, rtol, atol)

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


def _merge_points_pairwise(
    points: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    """Merge points using pairwise distance computation.

    Args:
        points: Point coordinates
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        point_mapping: Mapping from each point to its representative
    """
    n_points = len(points)
    device = points.device

    ### Compute duplicate mask using shared tolerance computation
    is_duplicate = _compute_duplicate_mask(points, rtol, atol)

    ### Build connected components using union-find
    # Start with each point mapping to itself
    point_mapping = torch.arange(n_points, device=device, dtype=torch.int64)

    ### Process each point and merge with lower-indexed duplicates only
    for i in range(n_points):
        if point_mapping[i] != i:
            # Already merged to a lower index
            continue

        # Find all points that should merge with i
        # Only consider j < i to avoid double-processing
        for j in range(i):
            if point_mapping[j] != j:
                # j already merged elsewhere
                continue
            if is_duplicate[i, j]:
                # Merge i into j (j has lower index)
                point_mapping[i] = j
                break

    ### Apply transitive closure via path compression
    # This ensures that if A~B and B~C, then C is also mapped to A's representative.
    # Each iteration halves the tree depth, so convergence is O(log n) iterations.
    max_iterations = 100
    for iteration in range(max_iterations):
        old_mapping = point_mapping.clone()
        point_mapping = point_mapping[point_mapping]
        if torch.equal(point_mapping, old_mapping):
            break
    else:
        import warnings

        warnings.warn(
            f"Transitive closure in pairwise merge did not converge in {max_iterations} "
            "iterations. This should never happen for valid meshes (expected O(log n) iterations)."
        )

    return point_mapping


def _merge_points_spatial_hash(
    points: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    """Merge points using spatial hashing for large meshes.

    This is more memory-efficient than pairwise distances but requires
    more complex implementation.

    Args:
        points: Point coordinates
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        point_mapping: Mapping from each point to its representative
    """
    n_points = len(points)
    device = points.device

    ### Compute a conservative grid size based on tolerance
    # We want cells large enough that duplicates are in same or adjacent cells
    # Use the larger of atol and rtol * typical_scale
    typical_scale = torch.norm(points.max(dim=0)[0] - points.min(dim=0)[0])
    cell_size = atol + rtol * typical_scale

    ### Ensure cell size is positive
    cell_size = max(cell_size, 1e-20)

    ### Map points to grid cells
    # Shape: (n_points, n_dims)
    grid_coords = (points / cell_size).floor().long()

    ### Create a hash for each grid cell
    # Use a simple hash function that maps n-dimensional grid coords to 1D
    # Hash = x + y * prime1 + z * prime2 + ...
    primes = torch.tensor([1, 999983, 999979, 999961, 999959], device=device)[
        : points.shape[1]
    ]
    grid_hashes = (grid_coords * primes).sum(dim=-1)

    ### Sort points by grid hash for efficient processing
    sorted_indices = torch.argsort(grid_hashes)
    sorted_points = points[sorted_indices]
    sorted_hashes = grid_hashes[sorted_indices]

    ### Find groups of points in the same grid cell
    # Points with same hash are potentially close
    unique_hashes, hash_inverse = torch.unique(sorted_hashes, return_inverse=True)

    ### Initialize mapping
    point_mapping = torch.arange(n_points, device=device, dtype=torch.int64)

    ### For each unique hash, check points within that cell and adjacent cells
    for hash_idx in range(len(unique_hashes)):
        ### Get points in this hash bucket
        mask = hash_inverse == hash_idx
        indices_in_bucket = torch.where(mask)[0]

        if len(indices_in_bucket) <= 1:
            continue

        ### Extract points in this bucket
        bucket_points = sorted_points[indices_in_bucket]
        bucket_original_indices = sorted_indices[indices_in_bucket]

        ### Find duplicates within bucket using shared tolerance computation
        is_duplicate = _compute_duplicate_mask(bucket_points, rtol, atol)

        ### Update mapping for duplicates
        for i in range(len(indices_in_bucket)):
            duplicates_local = torch.where(is_duplicate[i])[0]
            if len(duplicates_local) > 0:
                duplicates_global = bucket_original_indices[duplicates_local]
                min_idx = torch.min(duplicates_global)
                point_mapping[duplicates_global] = min_idx

    ### Apply transitive closure via path compression
    # Each iteration halves the tree depth, so convergence is O(log n) iterations.
    # For n points: 1K->10, 1M->20, 1B->30 iterations. Limit of 100 is very safe.
    max_iterations = 100
    for iteration in range(max_iterations):
        old_mapping = point_mapping.clone()
        point_mapping = point_mapping[point_mapping]
        if torch.equal(point_mapping, old_mapping):
            break
    else:
        import warnings

        warnings.warn(
            f"Transitive closure in spatial hash merge did not converge in {max_iterations} "
            "iterations. This should never happen for valid meshes (expected O(log n) iterations)."
        )

    return point_mapping


def _merge_point_data(
    point_data: TensorDict,
    point_mapping: torch.Tensor,
    unique_indices: torch.Tensor,
    n_unique: int,
) -> TensorDict:
    """Merge point data by averaging over merged points.

    Args:
        point_data: Original point data
        point_mapping: Mapping from original to merged points
        unique_indices: Indices of unique points in original array
        n_unique: Number of unique points

    Returns:
        Merged point data
    """
    from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate

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

    Args:
        cells: Cell connectivity, shape (n_cells, n_vertices_per_cell)
        cell_data: Cell data

    Returns:
        unique_cells: Deduplicated cells, shape (n_unique_cells, n_vertices_per_cell)
        unique_cell_data: Cell data for unique cells

    Example:
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

    ### Use a different strategy: mark duplicates and filter
    n_cells = len(cells)
    keep_mask = torch.ones(n_cells, dtype=torch.bool, device=cells.device)

    ### For each pair of cells, check if they're duplicates
    # This is O(n^2) but correct. For large meshes, we'd want a hash-based approach.

    if n_cells < 10000:
        ### Small mesh: pairwise comparison
        for i in range(n_cells):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n_cells):
                if not keep_mask[j]:
                    continue
                if torch.all(sorted_cells[i] == sorted_cells[j]):
                    keep_mask[j] = False
    else:
        ### Large mesh: use torch.unique properly
        # torch.unique returns unique rows, but we need indices
        # Use return_inverse to track which cells are duplicates
        _, inverse_indices = torch.unique(
            sorted_cells,
            dim=0,
            return_inverse=True,
        )

        ### Keep only first occurrence of each unique cell
        # For each unique cell, find its first occurrence
        unique_cell_ids = torch.unique(inverse_indices)
        for cell_id in unique_cell_ids:
            occurrences = torch.where(inverse_indices == cell_id)[0]
            if len(occurrences) > 1:
                keep_mask[occurrences[1:]] = False

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

    Args:
        points: Point coordinates, shape (n_points, n_spatial_dims)
        cells: Cell connectivity, shape (n_cells, n_vertices_per_cell)
        point_data: Point data

    Returns:
        used_points: Points that are used by cells, shape (n_used_points, n_spatial_dims)
        updated_cells: Updated cell connectivity, shape (n_cells, n_vertices_per_cell)
        used_point_data: Point data for used points
        point_mapping: Mapping from old to new point indices, shape (n_points,)
            Unused points map to -1

    Example:
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
    rtol: float = 1e-12,
    atol: float = 1e-12,
    merge_points: bool = True,
    remove_duplicate_cells_flag: bool = True,
    remove_unused_points_flag: bool = True,
) -> "Mesh":
    """Clean and repair a mesh.

    Performs various cleaning operations to fix common mesh issues:
    1. Merge duplicate points within tolerance
    2. Remove duplicate cells
    3. Remove unused points

    Args:
        mesh: Input mesh to clean
        rtol: Relative tolerance for merging points (default 1e-12)
        atol: Absolute tolerance for merging points (default 1e-12)
        merge_points: Whether to merge duplicate points
        remove_duplicate_cells_flag: Whether to remove duplicate cells
        remove_unused_points_flag: Whether to remove unused points

    Returns:
        Cleaned mesh with same structure but repaired topology

    Example:
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> # Mesh with duplicate points
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 3], [2, 1, 3]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> cleaned = clean_mesh(mesh)
        >>> assert cleaned.n_points == 3  # points 0 and 2 merged
    """
    points = mesh.points
    cells = mesh.cells
    point_data = mesh.point_data.exclude(CACHE_KEY)
    cell_data = mesh.cell_data.exclude(CACHE_KEY)
    global_data = mesh.global_data

    ### Step 1: Merge duplicate points
    if merge_points:
        points, cells, point_data, _ = merge_duplicate_points(
            points=points,
            cells=cells,
            point_data=point_data,
            rtol=rtol,
            atol=atol,
        )

    ### Step 2: Remove duplicate cells
    if remove_duplicate_cells_flag:
        cells, cell_data = remove_duplicate_cells(
            cells=cells,
            cell_data=cell_data,
        )

    ### Step 3: Remove unused points
    if remove_unused_points_flag:
        points, cells, point_data, _ = remove_unused_points(
            points=points,
            cells=cells,
            point_data=point_data,
        )

    ### Create cleaned mesh
    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=global_data,
    )
