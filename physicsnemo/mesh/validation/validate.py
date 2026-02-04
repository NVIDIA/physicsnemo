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

"""Mesh validation to detect common errors and degenerate cases.

Provides comprehensive validation of mesh integrity including topology,
geometry, and data consistency checks.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.boundaries import extract_candidate_facets

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _find_duplicate_vertices_spatial_hash(
    points: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Find duplicate vertex pairs using spatial hashing for O(N) average complexity.

    Uses fully vectorized PyTorch operations for GPU compatibility. Each point is
    expanded to its 2^d neighboring cells to correctly handle pairs that span
    cell boundaries.

    Args:
        points: Vertex positions, shape (n_points, n_spatial_dims)
        tolerance: Distance threshold for considering vertices as duplicates

    Returns:
        Tensor of duplicate pairs, shape (n_duplicates, 2), with i < j for each pair
    """
    n_points = points.shape[0]
    n_dims = points.shape[1]
    device = points.device

    if n_points == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    ### Step 1: Assign each point to grid cells
    # Use cell_size = tolerance so that any pair within tolerance spans at most
    # 2 cells in each dimension. By expanding each point to its 2^d neighboring
    # cells, we guarantee that any duplicate pair shares at least one cell.
    cell_size = tolerance if tolerance > 0 else 1.0

    # Shift points to have non-negative coordinates
    min_coords = points.min(dim=0).values
    shifted_points = points - min_coords

    # Compute integer cell indices for each point
    cell_indices = (shifted_points / cell_size).long()  # (n_points, n_dims)

    ### Step 2: Expand each point to 2^d neighboring cells
    # Generate all 2^d offset combinations: {0, 1}^n_dims
    # For 2D: [[0,0], [0,1], [1,0], [1,1]]
    # For 3D: [[0,0,0], [0,0,1], ..., [1,1,1]]
    n_neighbors = 2**n_dims
    neighbor_offsets = torch.zeros((n_neighbors, n_dims), dtype=torch.long, device=device)
    for i in range(n_neighbors):
        for d in range(n_dims):
            neighbor_offsets[i, d] = (i >> d) & 1

    # Expand cell_indices to include all neighboring cells
    # cell_indices: (n_points, n_dims)
    # neighbor_offsets: (2^d, n_dims)
    # Result: expanded_cells of shape (n_points * 2^d, n_dims)
    expanded_cells = cell_indices.unsqueeze(1) + neighbor_offsets.unsqueeze(
        0
    )  # (n_points, 2^d, n_dims)
    expanded_cells = expanded_cells.reshape(-1, n_dims)  # (n_points * 2^d, n_dims)

    # Create corresponding point indices for each expanded cell entry
    # Each point appears 2^d times
    expanded_point_indices = torch.arange(n_points, device=device).repeat_interleave(
        n_neighbors
    )  # (n_points * 2^d,)

    ### Step 3: Group points by cell using torch.unique
    # torch.unique with return_inverse assigns each unique cell a sequential ID
    # and inverse_indices tells us which ID each expanded point belongs to
    _, inverse_indices, counts = torch.unique(
        expanded_cells, dim=0, return_inverse=True, return_counts=True
    )

    # Sort by cell ID to group points in the same cell together
    sorted_order = torch.argsort(inverse_indices)
    sorted_point_indices = expanded_point_indices[sorted_order]

    # Compute bucket boundaries from counts
    # counts[i] = number of points in cell i
    bucket_ends = torch.cumsum(counts, dim=0)
    bucket_starts = torch.cat([torch.tensor([0], device=device), bucket_ends[:-1]])

    ### Step 4: Filter to buckets with 2+ points
    # counts from torch.unique is already the bucket sizes
    bucket_sizes = counts
    multi_point_mask = bucket_sizes >= 2

    if not multi_point_mask.any():
        return torch.empty((0, 2), dtype=torch.long, device=device)

    valid_bucket_sizes = bucket_sizes[multi_point_mask]
    valid_bucket_starts = bucket_starts[multi_point_mask]
    n_valid_buckets = len(valid_bucket_sizes)

    ### Step 5: Generate all pairs within each bucket (vectorized)
    # Following the pattern from _cell_neighbors.py
    # For each bucket, we generate C(n,2) = n*(n-1)/2 pairs

    # Total points across all valid buckets
    total_points_in_valid_buckets = int(valid_bucket_sizes.sum().item())

    # Generate cumulative offsets for indexing into sorted_point_indices
    bucket_cumulative_starts = torch.cat(
        [
            torch.tensor([0], dtype=torch.long, device=device),
            torch.cumsum(valid_bucket_sizes[:-1], dim=0),
        ]
    )

    # Create position index within concatenated valid bucket points
    cumulative_idx = torch.arange(
        total_points_in_valid_buckets, dtype=torch.long, device=device
    )

    # For each position, compute which bucket it belongs to and its local index
    bucket_ids = torch.repeat_interleave(
        torch.arange(n_valid_buckets, dtype=torch.long, device=device),
        valid_bucket_sizes,
    )
    local_indices = cumulative_idx - bucket_cumulative_starts[bucket_ids]

    # Get the actual point indices from sorted_point_indices
    global_positions = valid_bucket_starts[bucket_ids] + local_indices
    point_ids_in_buckets = sorted_point_indices[global_positions]

    ### Step 6: Generate all (i, j) pairs where i < j within each bucket
    # For each point at local index k in a bucket of size n, it pairs with
    # points at local indices k+1, k+2, ..., n-1. That's (n-1-k) pairs.
    # Total pairs per bucket: sum_{k=0}^{n-1} (n-1-k) = n*(n-1)/2

    # Number of pairs each point contributes (points at end of bucket contribute fewer)
    bucket_sizes_per_point = valid_bucket_sizes[bucket_ids]
    pairs_per_point = bucket_sizes_per_point - 1 - local_indices  # (n-1-k) for point at position k

    # Only points that contribute at least 1 pair
    contributing_mask = pairs_per_point > 0
    if not contributing_mask.any():
        return torch.empty((0, 2), dtype=torch.long, device=device)

    contributing_point_ids = point_ids_in_buckets[contributing_mask]
    contributing_bucket_ids = bucket_ids[contributing_mask]
    contributing_local_indices = local_indices[contributing_mask]
    contributing_pairs_count = pairs_per_point[contributing_mask]

    # Repeat each contributing point by its number of pairs
    pair_source_points = torch.repeat_interleave(
        contributing_point_ids, contributing_pairs_count
    )

    # Generate target local indices for each pair
    # For point at local index k, targets are k+1, k+2, ..., n-1
    total_pairs = contributing_pairs_count.sum()
    pair_cumulative_starts = torch.cat(
        [
            torch.tensor([0], dtype=torch.long, device=device),
            torch.cumsum(contributing_pairs_count[:-1], dim=0),
        ]
    )

    pair_idx = torch.arange(total_pairs, dtype=torch.long, device=device)
    pair_source_idx = torch.repeat_interleave(
        torch.arange(len(contributing_pairs_count), dtype=torch.long, device=device),
        contributing_pairs_count,
    )

    # Within-source offset: 0, 1, 2, ... for each source point
    within_source_offset = pair_idx - pair_cumulative_starts[pair_source_idx]

    # Target local index = source_local_index + 1 + offset
    source_local_indices_expanded = contributing_local_indices[pair_source_idx]
    target_local_indices = source_local_indices_expanded + 1 + within_source_offset

    # Convert target local indices back to point IDs
    target_bucket_ids = contributing_bucket_ids[pair_source_idx]
    target_global_positions = (
        valid_bucket_starts[target_bucket_ids]
        + target_local_indices
    )
    pair_target_points = sorted_point_indices[target_global_positions]

    ### Step 7: Compute distances for all candidate pairs (vectorized)
    pair_distances = torch.linalg.vector_norm(
        points[pair_source_points] - points[pair_target_points], dim=1
    )

    # Filter pairs within tolerance
    within_tolerance_mask = pair_distances < tolerance
    if not within_tolerance_mask.any():
        return torch.empty((0, 2), dtype=torch.long, device=device)

    filtered_sources = pair_source_points[within_tolerance_mask]
    filtered_targets = pair_target_points[within_tolerance_mask]

    ### Step 8: Canonicalize pairs (i < j) and deduplicate
    # Due to cell expansion, the same pair may appear in multiple buckets
    pair_min = torch.minimum(filtered_sources, filtered_targets)
    pair_max = torch.maximum(filtered_sources, filtered_targets)

    # Remove self-pairs (can happen if same point appears in multiple expanded cells)
    non_self_mask = pair_min != pair_max
    pair_min = pair_min[non_self_mask]
    pair_max = pair_max[non_self_mask]

    if len(pair_min) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    # Stack and deduplicate
    candidate_pairs = torch.stack([pair_min, pair_max], dim=1)  # (n_candidates, 2)
    unique_pairs = torch.unique(candidate_pairs, dim=0)

    return unique_pairs


def validate_mesh(
    mesh: "Mesh",
    check_degenerate_cells: bool = True,
    check_duplicate_vertices: bool = True,
    check_inverted_cells: bool = False,  # Expensive, opt-in
    check_out_of_bounds: bool = True,
    check_manifoldness: bool = False,  # Only 2D, opt-in
    check_self_intersection: bool = False,  # Very expensive, opt-in
    tolerance: float = 1e-10,
    raise_on_error: bool = False,
) -> Mapping[str, bool | int | torch.Tensor]:
    """Validate mesh integrity and detect common errors.

    Performs a comprehensive set of checks to ensure mesh is well-formed
    and suitable for geometric computations.

    Args:
        mesh: Mesh to validate
        check_degenerate_cells: Check for zero/negative area cells
        check_duplicate_vertices: Check for coincident vertices within tolerance
        check_inverted_cells: Check for cells with negative orientation (expensive)
        check_out_of_bounds: Check that cell indices are valid
        check_manifoldness: Check manifold topology (2D only, expensive)
        check_self_intersection: Check for self-intersecting cells (very expensive)
        tolerance: Tolerance for geometric checks (areas, distances)
        raise_on_error: If True, raise ValueError on first error. If False,
            return dict with all validation results.

    Returns:
        Dictionary with validation results:
            - "valid": bool, True if all enabled checks passed
            - "n_degenerate_cells": int, number of degenerate cells found
            - "degenerate_cell_indices": Tensor of indices (if any found)
            - "n_duplicate_vertices": int, number of duplicate vertex pairs
            - "duplicate_vertex_pairs": Tensor of index pairs (if any found)
            - "n_out_of_bounds_cells": int, cells with invalid indices
            - "out_of_bounds_cell_indices": Tensor of cell indices (if any)
            - "n_inverted_cells": int (if check enabled)
            - "inverted_cell_indices": Tensor (if check enabled and any found)
            - "is_manifold": bool (if check enabled, 2D only)
            - "non_manifold_edges": Tensor of edge indices (if check enabled)

    Raises:
        ValueError: If raise_on_error=True and validation fails

    Example:
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> report = validate_mesh(mesh)
        >>> assert report["valid"] == True
    """
    results = {
        "valid": True,
    }

    ### Check for out-of-bounds indices FIRST (before any geometric computations)
    if check_out_of_bounds:
        if mesh.n_cells > 0:
            min_index = mesh.cells.min()
            max_index = mesh.cells.max()

            out_of_bounds_mask = (mesh.cells < 0) | (mesh.cells >= mesh.n_points)
            out_of_bounds_cells = torch.any(out_of_bounds_mask, dim=1)
            n_out_of_bounds = out_of_bounds_cells.sum().item()

            results["n_out_of_bounds_cells"] = n_out_of_bounds

            if n_out_of_bounds > 0:
                results["valid"] = False
                results["out_of_bounds_cell_indices"] = torch.where(
                    out_of_bounds_cells
                )[0]

                if raise_on_error:
                    raise ValueError(
                        f"Found {n_out_of_bounds} cells with out-of-bounds indices.\n"
                        f"Cell indices must be in range [0, {mesh.n_points}), "
                        f"but got {min_index.item()=} and {max_index.item()=}.\n"
                        f"Problem cells: {results['out_of_bounds_cell_indices'].tolist()[:10]}"
                    )
        else:
            results["n_out_of_bounds_cells"] = 0

    ### Early return if out-of-bounds indices found (can't compute geometry)
    if check_out_of_bounds and results.get("n_out_of_bounds_cells", 0) > 0:
        if raise_on_error:
            # Already raised above
            pass
        else:
            # Skip remaining geometric checks
            return results

    ### Check for duplicate vertices
    if check_duplicate_vertices:
        # Use vectorized spatial hashing for O(N) average complexity
        # Works efficiently for all mesh sizes
        duplicate_pairs = _find_duplicate_vertices_spatial_hash(
            mesh.points, tolerance
        )
        n_duplicates = len(duplicate_pairs)

        results["n_duplicate_vertices"] = n_duplicates

        if n_duplicates > 0:
            results["valid"] = False
            results["duplicate_vertex_pairs"] = duplicate_pairs

            if raise_on_error:
                raise ValueError(
                    f"Found {n_duplicates} pairs of duplicate vertices "
                    f"(within tolerance={tolerance}).\n"
                    f"First few pairs: {duplicate_pairs[:5].tolist()}"
                )

    ### Check for degenerate cells
    if check_degenerate_cells and mesh.n_cells > 0:
        # Compute cell areas
        areas = mesh.cell_areas

        # Scale tolerance for area comparison:
        # - tolerance is in distance units
        # - areas have units of length^n_manifold_dims
        # So use tolerance^n_manifold_dims for a consistent comparison
        area_tolerance = tolerance ** mesh.n_manifold_dims

        # Find cells with area below tolerance
        degenerate_mask = areas < area_tolerance
        n_degenerate = degenerate_mask.sum().item()

        results["n_degenerate_cells"] = n_degenerate

        if n_degenerate > 0:
            results["valid"] = False
            results["degenerate_cell_indices"] = torch.where(degenerate_mask)[0]
            results["degenerate_cell_areas"] = areas[degenerate_mask]

            if raise_on_error:
                raise ValueError(
                    f"Found {n_degenerate} degenerate cells with area < {area_tolerance}.\n"
                    f"Problem cells: {results['degenerate_cell_indices'].tolist()[:10]}\n"
                    f"Areas: {results['degenerate_cell_areas'].tolist()[:10]}"
                )
    elif check_degenerate_cells:
        results["n_degenerate_cells"] = 0

    ### Check for inverted cells (cells with negative orientation)
    if check_inverted_cells and mesh.n_cells > 0:
        # For simplicial meshes, check if determinant is negative
        # This indicates inverted orientation

        if mesh.n_manifold_dims == mesh.n_spatial_dims:
            # Volume mesh: can compute signed volume
            cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_verts, n_dims)

            # Compute signed volume using determinant
            # For n-simplex: V = (1/n!) * det([v1-v0, v2-v0, ..., vn-v0])
            relative_vectors = cell_vertices[:, 1:] - cell_vertices[:, [0]]

            # Compute determinant
            if mesh.n_manifold_dims == 3:
                # 3D case: determinant of 3x3 matrix
                det = torch.det(relative_vectors)  # (n_cells,)

                inverted_mask = det < 0
                n_inverted = inverted_mask.sum().item()

                results["n_inverted_cells"] = n_inverted

                if n_inverted > 0:
                    results["valid"] = False
                    results["inverted_cell_indices"] = torch.where(inverted_mask)[0]

                    if raise_on_error:
                        raise ValueError(
                            f"Found {n_inverted} inverted cells (negative orientation).\n"
                            f"Problem cells: {results['inverted_cell_indices'].tolist()[:10]}"
                        )
            else:
                # For other dimensions, orientation check is more complex
                results["n_inverted_cells"] = -1  # Not implemented
        else:
            # Codimension > 0: orientation not well-defined
            results["n_inverted_cells"] = -1  # Not applicable
    elif check_inverted_cells:
        results["n_inverted_cells"] = 0

    ### Check manifoldness (2D only)
    if check_manifoldness:
        if mesh.n_manifold_dims == 2 and mesh.n_spatial_dims >= 2:
            # Check that each edge is shared by at most 2 triangles
            # Extract all edges (with duplicates)
            edges_with_dupes, parent_cells = extract_candidate_facets(
                mesh.cells, manifold_codimension=1
            )

            # Sort edges to canonical form
            edges_sorted = torch.sort(edges_with_dupes, dim=1).values

            # Find unique edges and their counts
            unique_edges, inverse_indices, counts = torch.unique(
                edges_sorted, dim=0, return_inverse=True, return_counts=True
            )

            # Manifold edges should appear exactly 1 (boundary) or 2 (interior) times
            non_manifold_mask = counts > 2
            n_non_manifold = non_manifold_mask.sum().item()

            results["is_manifold"] = n_non_manifold == 0
            results["n_non_manifold_edges"] = n_non_manifold

            if n_non_manifold > 0:
                results["valid"] = False
                results["non_manifold_edges"] = unique_edges[non_manifold_mask]
                results["non_manifold_edge_counts"] = counts[non_manifold_mask]

                if raise_on_error:
                    raise ValueError(
                        f"Mesh is not manifold: {n_non_manifold} edges shared by >2 faces.\n"
                        f"First few problem edges: {results['non_manifold_edges'][:5].tolist()}"
                    )
        else:
            results["is_manifold"] = None  # Only defined for 2D manifolds
            results["n_non_manifold_edges"] = -1  # Not applicable

    ### Check for self-intersections (very expensive, opt-in only)
    if check_self_intersection:
        # This is very expensive: O(n^2) cell-cell intersection tests
        # For production use, would need BVH acceleration
        results["has_self_intersection"] = None  # Not implemented yet
        results["intersecting_cell_pairs"] = None

        # TODO: Implement BVH-accelerated self-intersection detection
        if raise_on_error:
            raise NotImplementedError(
                "Self-intersection checking not yet implemented.\n"
                "This is a very expensive operation requiring BVH acceleration."
            )

    return results


def check_duplicate_cell_vertices(mesh: "Mesh") -> tuple[int, torch.Tensor]:
    """Check for cells with duplicate vertices (degenerate simplices).

    A valid n-simplex must have n+1 distinct vertices. Cells with duplicate
    vertices are degenerate and should be removed.

    Args:
        mesh: Mesh to check

    Returns:
        Tuple of (n_invalid_cells, invalid_cell_indices)

    Example:
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> n_invalid, indices = check_duplicate_cell_vertices(mesh)
        >>> assert n_invalid == 0  # clean mesh has no duplicate vertices
    """
    if mesh.n_cells == 0:
        return 0, torch.tensor([], dtype=torch.long, device=mesh.cells.device)

    # Vectorized approach: sort vertices within each cell, then check for
    # consecutive duplicates. A cell has duplicates if any adjacent pair
    # in the sorted order is equal.
    sorted_cells = torch.sort(mesh.cells, dim=1).values  # (n_cells, n_verts)

    # Check for consecutive duplicates: sorted_cells[:, i] == sorted_cells[:, i+1]
    has_duplicate = (sorted_cells[:, 1:] == sorted_cells[:, :-1]).any(dim=1)

    # Get indices of cells with duplicates
    invalid_indices = torch.where(has_duplicate)[0]
    n_invalid = len(invalid_indices)

    if n_invalid == 0:
        return 0, torch.tensor([], dtype=torch.long, device=mesh.cells.device)

    return n_invalid, invalid_indices
