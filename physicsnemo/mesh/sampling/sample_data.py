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

"""Spatial sampling of data at query points in a mesh.

Supports both brute-force O(M*N) containment testing and BVH-accelerated
O(M*log(N)) queries. The public API is a single ``sample_data_at_points``
function; pass a ``BVH`` to opt into the accelerated path.
"""

from typing import TYPE_CHECKING, Literal

import torch
from tensordict import TensorDict

from physicsnemo.mesh.neighbors._adjacency import Adjacency, build_adjacency_from_pairs
from physicsnemo.mesh.utilities._cache import CACHE_KEY

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.spatial import BVH


# ---------------------------------------------------------------------------
# Barycentric coordinate solvers
# ---------------------------------------------------------------------------


def _solve_barycentric_system(
    relative_vectors: torch.Tensor,  # shape: (..., n_manifold_dims, n_spatial_dims)
    query_relative: torch.Tensor,  # shape: (..., n_spatial_dims)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core barycentric coordinate solver (shared by both variants).

    Solves the linear system to find barycentric coordinates w_1, ..., w_n such that:
        query_relative = sum(w_i * relative_vectors[i])

    Then computes w_0 = 1 - sum(w_i) and returns all coordinates [w_0, w_1, ..., w_n].

    For codimension != 0 manifolds (n_spatial_dims != n_manifold_dims), this uses
    least squares which projects the query point onto the simplex's affine hull.
    The reconstruction error measures how far the query point is from this projection.

    Parameters
    ----------
    relative_vectors : torch.Tensor
        Edge vectors from first vertex to others,
        shape (..., n_manifold_dims, n_spatial_dims)
    query_relative : torch.Tensor
        Query point relative to first vertex,
        shape (..., n_spatial_dims)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (barycentric_coords, reconstruction_error):
        - barycentric_coords: Barycentric coordinates, shape (..., n_vertices_per_cell)
            where n_vertices_per_cell = n_manifold_dims + 1
        - reconstruction_error: L2 distance from query point to its projection onto
            the simplex's affine hull, shape (...). Zero for codimension-0 manifolds.

    Notes
    -----
    For square systems (n_spatial_dims == n_manifold_dims): use direct solve
    For over/under-determined systems: use least squares
    """
    n_manifold_dims = relative_vectors.shape[-2]
    n_spatial_dims = relative_vectors.shape[-1]

    if n_spatial_dims == n_manifold_dims:
        ### Square system: use torch.linalg.solve
        # Transpose to get (..., n_spatial_dims, n_manifold_dims)
        A = relative_vectors.transpose(-2, -1)
        # query_relative: (..., n_spatial_dims) -> (..., n_spatial_dims, 1)
        b = query_relative.unsqueeze(-1)

        # Solve: A @ x = b
        try:
            weights_1_to_n = torch.linalg.solve(A, b).squeeze(-1)
        except torch.linalg.LinAlgError:
            # Singular matrix - use lstsq as fallback
            weights_1_to_n = torch.linalg.lstsq(A, b).solution.squeeze(-1)

        ### For square systems, reconstruction error is zero (exact solution)
        # Shape: (...) - same batch dimensions as weights_1_to_n but without last dim
        reconstruction_error = torch.zeros(
            weights_1_to_n.shape[:-1],
            dtype=query_relative.dtype,
            device=query_relative.device,
        )

    else:
        ### Over-determined or under-determined system: use least squares
        A = relative_vectors.transpose(-2, -1)
        b = query_relative.unsqueeze(-1)
        weights_1_to_n = torch.linalg.lstsq(A, b).solution.squeeze(-1)

        ### Compute reconstruction error: ||query_relative - reconstructed||
        # reconstructed = sum(w_i * e_i) where e_i = relative_vectors[i]
        # Shape: (..., n_spatial_dims)
        reconstructed = torch.einsum(
            "...m,...ms->...s", weights_1_to_n, relative_vectors
        )
        residual = query_relative - reconstructed  # (..., n_spatial_dims)
        reconstruction_error = torch.linalg.vector_norm(
            residual, dim=-1
        )  # (...) L2 norm

    ### Compute w_0 = 1 - sum(w_i for i=1..n)
    w_0 = 1.0 - weights_1_to_n.sum(dim=-1, keepdim=True)

    ### Concatenate to get all barycentric coordinates
    barycentric_coords = torch.cat([w_0, weights_1_to_n], dim=-1)

    return barycentric_coords, reconstruction_error


def compute_barycentric_coordinates(
    query_points: torch.Tensor,
    cell_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute barycentric coordinates of query points with respect to simplices.

    For each query point and each simplex, computes the barycentric coordinates.
    A point is inside a simplex if all barycentric coordinates are non-negative
    AND the reconstruction error is within tolerance (for codimension != 0 manifolds).

    Parameters
    ----------
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims)
    cell_vertices : torch.Tensor
        Vertices of cells to test, shape (n_cells, n_vertices_per_cell, n_spatial_dims)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (barycentric_coords, reconstruction_error):
        - barycentric_coords: Barycentric coordinates, shape (n_queries, n_cells, n_vertices_per_cell).
            For each query-cell pair, the coordinates sum to 1.
        - reconstruction_error: L2 distance from query point to its projection onto
            the simplex's affine hull, shape (n_queries, n_cells). Zero for codimension-0.

    Notes
    -----
    For a simplex with vertices v0, v1, ..., vn and query point p:
    - Compute relative vectors: e_i = v_i - v_0 for i=1..n
    - Solve: p - v_0 = sum(w_i * e_i) for w_1, ..., w_n
    - Then w_0 = 1 - sum(w_i for i=1..n)
    - Point is inside if all w_i >= 0 (within tolerance)
    """
    ### Compute relative vectors from first vertex to all others
    # Shape: (n_cells, n_vertices_per_cell - 1, n_spatial_dims)
    v0 = cell_vertices[:, 0:1, :]  # (n_cells, 1, n_spatial_dims)
    relative_vectors = (
        cell_vertices[:, 1:, :] - v0
    )  # (n_cells, n_manifold_dims, n_spatial_dims)

    ### Compute query points relative to v0
    # Broadcast query_points and v0 for all combinations
    # Shape: (n_queries, n_cells, n_spatial_dims)
    query_relative = query_points.unsqueeze(1) - v0.squeeze(1).unsqueeze(0)

    ### Solve using shared barycentric solver
    # Expand relative_vectors to broadcast with queries
    # relative_vectors: (n_cells, n_manifold_dims, n_spatial_dims)
    # query_relative: (n_queries, n_cells, n_spatial_dims)
    # Need to expand relative_vectors to (1, n_cells, n_manifold_dims, n_spatial_dims)
    relative_vectors_expanded = relative_vectors.unsqueeze(0)

    # Use shared solver that handles the linear system
    barycentric_coords, reconstruction_error = _solve_barycentric_system(
        relative_vectors_expanded, query_relative
    )

    return barycentric_coords, reconstruction_error


def compute_barycentric_coordinates_pairwise(
    query_points: torch.Tensor,
    cell_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute barycentric coordinates for paired queries and cells.

    Unlike compute_barycentric_coordinates which computes all O(n_queries x n_cells)
    combinations, this computes only n_pairs diagonal elements where each query point
    is paired with exactly one cell. This uses O(n) memory instead of O(n^2).

    This is critical for performance when processing BVH candidate pairs, where we may
    have thousands of pairs but don't need the full cartesian product.

    Parameters
    ----------
    query_points : torch.Tensor
        Query point locations, shape (n_pairs, n_spatial_dims)
    cell_vertices : torch.Tensor
        Vertices of cells, shape (n_pairs, n_vertices_per_cell, n_spatial_dims)
        where cell_vertices[i] is paired with query_points[i]

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (barycentric_coords, reconstruction_error):
        - barycentric_coords: Barycentric coordinates, shape (n_pairs, n_vertices_per_cell).
            For each pair, the coordinates sum to 1.
        - reconstruction_error: L2 distance from query point to its projection onto
            the simplex's affine hull, shape (n_pairs,). Zero for codimension-0.

    Examples
    --------
    >>> import torch
    >>> # For BVH results: each query has specific candidate cells
    >>> n_pairs = 1000
    >>> query_points = torch.randn(n_pairs, 3)
    >>> cell_vertices = torch.randn(n_pairs, 3, 3)  # Triangles in 3D
    >>> bary, recon_err = compute_barycentric_coordinates_pairwise(query_points, cell_vertices)
    >>> assert bary.shape == (1000, 3)  # instead of (1000, 1000, 3) from full version
    >>> assert recon_err.shape == (1000,)
    """

    ### Compute relative vectors from first vertex to all others
    # Shape: (n_pairs, n_manifold_dims, n_spatial_dims)
    v0 = cell_vertices[:, 0, :]  # (n_pairs, n_spatial_dims)
    relative_vectors = cell_vertices[:, 1:, :] - v0.unsqueeze(1)

    ### Compute query points relative to v0
    # Shape: (n_pairs, n_spatial_dims)
    query_relative = query_points - v0

    ### Solve using shared barycentric solver
    # relative_vectors: (n_pairs, n_manifold_dims, n_spatial_dims)
    # query_relative: (n_pairs, n_spatial_dims)
    # Both are already in the right shape for pairwise solving
    barycentric_coords, reconstruction_error = _solve_barycentric_system(
        relative_vectors, query_relative
    )

    return barycentric_coords, reconstruction_error


# ---------------------------------------------------------------------------
# Containment queries
# ---------------------------------------------------------------------------


def find_containing_cells(
    mesh: "Mesh",
    query_points: torch.Tensor,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find which cells contain each query point.

    Parameters
    ----------
    mesh : Mesh
        The mesh to query.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims)
    tolerance : float
        Tolerance for considering a point inside a cell.
        A point is inside if:
        - All barycentric coordinates >= -tolerance, AND
        - Reconstruction error <= tolerance (distance from query point to the
          simplex's affine hull). This ensures points far from codimension != 0
          manifolds (e.g., 2D triangles in 3D space) are not incorrectly reported
          as inside.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (cell_indices, barycentric_coords):
        - cell_indices: Cell index for each query point, shape (n_queries,).
            Value is -1 if no cell contains the point, or the first containing cell index.
        - barycentric_coords: Barycentric coordinates for each query point in its
            containing cell, shape (n_queries, n_vertices_per_cell).
            Values are NaN if no containing cell exists.

    Notes
    -----
    If multiple cells contain a point, only the first is returned.
    Use find_all_containing_cells() to get all containing cells.
    """
    n_queries = query_points.shape[0]
    n_vertices_per_cell = mesh.n_manifold_dims + 1

    ### Get cell vertices: (n_cells, n_vertices_per_cell, n_spatial_dims)
    cell_vertices = mesh.points[mesh.cells]

    ### Compute barycentric coordinates for all query-cell pairs
    # Shape: (n_queries, n_cells, n_vertices_per_cell) and (n_queries, n_cells)
    bary_coords, recon_error = compute_barycentric_coordinates(
        query_points, cell_vertices
    )

    ### Determine which query-cell pairs have the point inside
    # A point is inside if:
    # 1. All barycentric coordinates are >= -tolerance
    # 2. Reconstruction error (distance to affine hull) <= tolerance
    # Shape: (n_queries, n_cells)
    bary_inside = (bary_coords >= -tolerance).all(dim=-1)
    recon_inside = recon_error <= tolerance
    is_inside = bary_inside & recon_inside

    ### For each query, find the first containing cell (vectorized)
    # Shape: (n_queries,)
    cell_indices = torch.full(
        (n_queries,), -1, dtype=torch.long, device=mesh.points.device
    )
    result_bary_coords = torch.full(
        (n_queries, n_vertices_per_cell),
        float("nan"),
        dtype=query_points.dtype,
        device=mesh.points.device,
    )

    ### Vectorized approach: find first True index along each row
    # For each query (row), find the first cell (column) where is_inside is True
    # is_inside shape: (n_queries, n_cells)

    # Get indices of all True values
    query_idx, cell_idx = torch.where(is_inside)

    # For each query, we want the FIRST cell index (smallest cell_idx in original order)
    # Since torch.where returns results in row-major order, we need to find the first
    # occurrence of each query_idx value

    if len(query_idx) > 0:
        # Find where each query_idx changes (marks first occurrence of new query)
        # Prepend True to catch the first element
        is_first_occurrence = torch.cat(
            [
                torch.tensor([True], device=query_idx.device),
                query_idx[1:] != query_idx[:-1],
            ]
        )

        # Get first occurrence indices
        first_occurrence_positions = torch.where(is_first_occurrence)[0]

        # Extract query indices and their corresponding first cells
        queries_with_hits = query_idx[first_occurrence_positions]
        first_cells = cell_idx[first_occurrence_positions]

        # Scatter into result array
        cell_indices[queries_with_hits] = first_cells

        # Get barycentric coords for found cells
        result_bary_coords[queries_with_hits] = bary_coords[
            queries_with_hits,
            first_cells,
        ]

    return cell_indices, result_bary_coords


def find_all_containing_cells(
    mesh: "Mesh",
    query_points: torch.Tensor,
    tolerance: float = 1e-6,
) -> Adjacency:
    """Find all cells that contain each query point.

    Parameters
    ----------
    mesh : Mesh
        The mesh to query.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims)
    tolerance : float
        Tolerance for considering a point inside a cell.
        A point is inside if:
        - All barycentric coordinates >= -tolerance, AND
        - Reconstruction error <= tolerance (distance from query point to the
          simplex's affine hull).

    Returns
    -------
    Adjacency
        Adjacency object where containing cells for query i are at
        ``result.indices[result.offsets[i]:result.offsets[i+1]]``.
        Use ``result.to_list()`` for a list-of-tensors representation.
    """
    ### Get cell vertices: (n_cells, n_vertices_per_cell, n_spatial_dims)
    cell_vertices = mesh.points[mesh.cells]

    ### Compute barycentric coordinates for all query-cell pairs
    bary_coords, recon_error = compute_barycentric_coordinates(
        query_points, cell_vertices
    )

    ### Determine which query-cell pairs have the point inside
    # Check both barycentric bounds and reconstruction error
    bary_inside = (bary_coords >= -tolerance).all(dim=-1)
    recon_inside = recon_error <= tolerance
    is_inside = bary_inside & recon_inside

    ### For each query, collect all containing cells (vectorized)
    # Get all (query_idx, cell_idx) pairs where containment is True
    query_indices, cell_indices = torch.where(is_inside)

    ### Build Adjacency from (query_idx, cell_idx) pairs
    return build_adjacency_from_pairs(
        source_indices=query_indices,
        target_indices=cell_indices,
        n_sources=len(query_points),
    )


def project_point_onto_cell(
    query_point: torch.Tensor,
    cell_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a query point onto a simplex (cell).

    Uses iterative barycentric clipping to find the closest point on the simplex.
    This is more efficient than recursive face enumeration.

    Parameters
    ----------
    query_point : torch.Tensor
        Point to project, shape (n_spatial_dims,)
    cell_vertices : torch.Tensor
        Vertices of the simplex, shape (n_vertices, n_spatial_dims)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (projected_point, squared_distance):
        - projected_point: Closest point on the simplex, shape (n_spatial_dims,)
        - squared_distance: Squared distance from query to projection, scalar tensor
    """
    n_vertices = cell_vertices.shape[0]

    # Handle degenerate cases
    if n_vertices == 1:
        # Single vertex - project to that vertex
        projected = cell_vertices[0]
        dist_sq = ((query_point - projected) ** 2).sum()
        return projected, dist_sq

    # Compute barycentric coordinates on the full simplex
    bary, _ = compute_barycentric_coordinates(
        query_point.unsqueeze(0),
        cell_vertices.unsqueeze(0),
    )
    bary = bary.squeeze(0).squeeze(0)  # (n_vertices,)

    # If all barycentric coords are non-negative, projection is inside the simplex
    if (bary >= 0).all():
        projected = (bary.unsqueeze(-1) * cell_vertices).sum(dim=0)
        dist_sq = ((query_point - projected) ** 2).sum()
        return projected, dist_sq

    # Otherwise, iteratively project onto the active face (vertices with bary > 0)
    # Use clipping algorithm: keep only vertices with positive barycentric coords
    max_iterations = n_vertices  # At most n-1 iterations needed

    for _ in range(max_iterations):
        # Find vertices with positive barycentric coordinates
        active_mask = bary > 0

        # If no positive coords (shouldn't happen), fall back to nearest vertex
        if not active_mask.any():
            dists = ((cell_vertices - query_point.unsqueeze(0)) ** 2).sum(dim=-1)
            nearest_idx = dists.argmin()
            projected = cell_vertices[nearest_idx]
            dist_sq = dists[nearest_idx]
            return projected, dist_sq

        # Keep only active vertices
        active_vertices = cell_vertices[active_mask]

        if active_vertices.shape[0] == 1:
            # Single active vertex
            projected = active_vertices[0]
            dist_sq = ((query_point - projected) ** 2).sum()
            return projected, dist_sq

        # Re-compute barycentric coords on the active face
        bary_active, _ = compute_barycentric_coordinates(
            query_point.unsqueeze(0),
            active_vertices.unsqueeze(0),
        )
        bary_active = bary_active.squeeze(0).squeeze(0)

        # If all non-negative, we found the projection
        if (bary_active >= 0).all():
            projected = (bary_active.unsqueeze(-1) * active_vertices).sum(dim=0)
            dist_sq = ((query_point - projected) ** 2).sum()
            return projected, dist_sq

        # Update for next iteration: map bary_active back to full bary
        bary = torch.zeros_like(bary)
        bary[active_mask] = bary_active

    # Fallback: nearest vertex (shouldn't reach here for valid input)
    dists = ((cell_vertices - query_point.unsqueeze(0)) ** 2).sum(dim=-1)
    nearest_idx = dists.argmin()
    projected = cell_vertices[nearest_idx]
    dist_sq = dists[nearest_idx]
    return projected, dist_sq


def find_nearest_cells(
    mesh: "Mesh",
    query_points: torch.Tensor,
    chunk_size: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest cell for each query point.

    This implementation finds the cell whose centroid is nearest. For large numbers
    of queries or cells, the computation is chunked to avoid memory issues.

    Parameters
    ----------
    mesh : Mesh
        The mesh to query.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims)
    chunk_size : int
        Number of queries to process at once. Larger values use more
        memory but may be faster. Default: 10000

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (cell_indices, projected_points):
        - cell_indices: Nearest cell index for each query point, shape (n_queries,)
        - projected_points: Centroids of nearest cells (approximation of projection),
            shape (n_queries, n_spatial_dims)

    Notes
    -----
    - Uses centroid distances as approximation. Full projection onto simplices
      would require iterative optimization.
    - Complexity is O(n_queries * n_cells). For very large meshes (>100k cells),
      a BVH-based nearest neighbor search could provide O(n_queries * log(n_cells))
      but is not yet implemented.
    """
    n_queries = query_points.shape[0]
    device = mesh.points.device

    ### Compute all cell centroids
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)

    ### For small problems, use fully vectorized approach
    if n_queries * mesh.n_cells <= chunk_size * chunk_size:
        # Compute distances from all queries to all cell centroids
        diffs = query_points.unsqueeze(1) - cell_centroids.unsqueeze(0)
        distances_sq = (diffs**2).sum(dim=-1)  # (n_queries, n_cells)

        # Find nearest cell for each query
        cell_indices = distances_sq.argmin(dim=1)  # (n_queries,)
    else:
        ### For large problems, chunk to avoid memory explosion
        cell_indices = torch.empty(n_queries, dtype=torch.long, device=device)

        for start in range(0, n_queries, chunk_size):
            end = min(start + chunk_size, n_queries)
            query_chunk = query_points[start:end]

            # Compute distances for this chunk
            diffs = query_chunk.unsqueeze(1) - cell_centroids.unsqueeze(0)
            distances_sq = (diffs**2).sum(dim=-1)

            # Find nearest cell for each query in chunk
            cell_indices[start:end] = distances_sq.argmin(dim=1)

    ### Return centroids of nearest cells as approximation of projection
    projected_points = cell_centroids[cell_indices]  # (n_queries, n_spatial_dims)

    return cell_indices, projected_points


# ---------------------------------------------------------------------------
# Containment-pair finders (brute-force vs BVH)
# ---------------------------------------------------------------------------


def _find_containing_pairs_bruteforce(
    mesh: "Mesh",
    query_points: torch.Tensor,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Find (query_idx, cell_idx, bary_coords) via brute-force O(M*N) search.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        (query_indices, cell_indices, bary_coords_for_containing):
        - query_indices: shape (n_containing,)
        - cell_indices: shape (n_containing,)
        - bary_coords_for_containing: shape (n_containing, n_verts) or None if empty
    """
    cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_verts, n_spatial_dims)
    bary_coords_all, recon_error_all = compute_barycentric_coordinates(
        query_points, cell_vertices
    )

    ### Determine containment: barycentric bounds AND reconstruction error
    bary_inside = (bary_coords_all >= -tolerance).all(dim=-1)  # (n_queries, n_cells)
    recon_inside = recon_error_all <= tolerance
    is_inside = bary_inside & recon_inside

    ### Extract flat arrays of containing pairs
    query_indices, cell_indices = torch.where(is_inside)

    if len(query_indices) > 0:
        bary_coords = bary_coords_all[query_indices, cell_indices]
    else:
        bary_coords = None

    return query_indices, cell_indices, bary_coords


def _find_containing_pairs_bvh(
    mesh: "Mesh",
    query_points: torch.Tensor,
    bvh: "BVH",
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Find (query_idx, cell_idx, bary_coords) via BVH-accelerated O(M*log(N)) search.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        Same format as _find_containing_pairs_bruteforce.
    """
    device = mesh.points.device

    ### Get candidate pairs from BVH (AABB overlap test)
    candidate_adjacency = bvh.find_candidate_cells(
        query_points, aabb_tolerance=tolerance
    )

    if candidate_adjacency.n_total_neighbors == 0:
        return (
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
            None,
        )

    query_idx_cand, cell_idx_cand = candidate_adjacency.expand_to_pairs()

    ### Refine candidates with exact barycentric test
    cand_query_pts = query_points[query_idx_cand]
    cand_cell_verts = mesh.points[mesh.cells[cell_idx_cand]]

    bary_coords_cand, recon_error_cand = compute_barycentric_coordinates_pairwise(
        cand_query_pts, cand_cell_verts
    )

    bary_inside = (bary_coords_cand >= -tolerance).all(dim=-1)
    recon_inside = recon_error_cand <= tolerance
    is_inside = bary_inside & recon_inside

    ### Filter to confirmed containments
    query_indices = query_idx_cand[is_inside]
    cell_indices = cell_idx_cand[is_inside]

    if len(query_indices) > 0:
        bary_coords = bary_coords_cand[is_inside]
    else:
        bary_coords = None

    return query_indices, cell_indices, bary_coords


# ---------------------------------------------------------------------------
# Shared accumulation logic
# ---------------------------------------------------------------------------


def _accumulate_sampled_data(
    mesh: "Mesh",
    n_queries: int,
    query_indices: torch.Tensor,
    cell_indices: torch.Tensor,
    bary_coords: torch.Tensor | None,
    data_source: str,
    multiple_cells_strategy: str,
) -> TensorDict:
    """Accumulate sampled data from containing-pair arrays into a TensorDict.

    This is the shared accumulation kernel used by both the brute-force and
    BVH-accelerated paths. It handles scalar/multidimensional data, mean/nan
    strategies, and cell/point data sources.

    Parameters
    ----------
    mesh : Mesh
        Source mesh.
    n_queries : int
        Total number of query points.
    query_indices : torch.Tensor
        Query index for each containing pair, shape (n_containing,).
    cell_indices : torch.Tensor
        Cell index for each containing pair, shape (n_containing,).
    bary_coords : torch.Tensor or None
        Barycentric coordinates for each pair, shape (n_containing, n_verts).
        Required when data_source="points". May be None when no containments exist.
    data_source : str
        "cells" or "points".
    multiple_cells_strategy : str
        "mean" or "nan".

    Returns
    -------
    TensorDict
        Sampled data with shape (n_queries, ...) per field.
    """
    device = mesh.points.device

    ### Count how many cells contain each query point
    query_containment_count = torch.zeros(n_queries, dtype=torch.long, device=device)
    if len(query_indices) > 0:
        query_containment_count.scatter_add_(
            0, query_indices, torch.ones_like(query_indices)
        )

    ### Select data source
    source_data = mesh.cell_data if data_source == "cells" else mesh.point_data

    result = TensorDict(
        {},
        batch_size=torch.Size([n_queries]),
        device=device,
    )

    ### Sample each field
    for key, values in source_data.exclude(CACHE_KEY).items():
        output_shape = (n_queries,) + values.shape[1:]

        # Initialize with NaN
        output = torch.full(
            output_shape, float("nan"), dtype=values.dtype, device=device
        )

        if len(query_indices) == 0:
            result[key] = output
            continue

        ### Get per-pair values
        if data_source == "cells":
            pair_values = values[cell_indices]  # (n_containing, ...)
        else:
            # Interpolate point data using barycentric coordinates
            point_idx = mesh.cells[cell_indices]  # (n_containing, n_verts)
            point_vals = values[point_idx]  # (n_containing, n_verts, ...)

            if values.ndim == 1:
                pair_values = (bary_coords * point_vals).sum(dim=1)
            else:
                bary_expanded = bary_coords.view(
                    bary_coords.shape[0],
                    bary_coords.shape[1],
                    *([1] * (values.ndim - 1)),
                )
                pair_values = (bary_expanded * point_vals).sum(dim=1)

        ### Accumulate into output
        if multiple_cells_strategy == "mean":
            if values.ndim == 1:
                output_sum = torch.zeros(n_queries, dtype=values.dtype, device=device)
                output_sum.scatter_add_(0, query_indices, pair_values)
            else:
                output_sum = torch.zeros(
                    output_shape, dtype=values.dtype, device=device
                )
                idx_expanded = query_indices.view(
                    -1, *([1] * (values.ndim - 1))
                ).expand_as(pair_values)
                output_sum.scatter_add_(0, idx_expanded, pair_values)

            valid = query_containment_count > 0
            if values.ndim == 1:
                output[valid] = (
                    output_sum[valid] / query_containment_count[valid].to(values.dtype)
                )
            else:
                output[valid] = output_sum[valid] / query_containment_count[
                    valid
                ].to(values.dtype).view(-1, *([1] * (values.ndim - 1)))

        else:  # "nan" strategy
            single_cell_mask = query_containment_count == 1
            if single_cell_mask.any():
                has_single = single_cell_mask[query_indices]
                output[query_indices[has_single]] = pair_values[has_single]

        result[key] = output

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_data_at_points(
    mesh: "Mesh",
    query_points: torch.Tensor,
    data_source: Literal["cells", "points"] = "cells",
    multiple_cells_strategy: Literal["mean", "nan"] = "mean",
    project_onto_nearest_cell: bool = False,
    tolerance: float = 1e-6,
    bvh: "BVH | None" = None,
) -> TensorDict:
    """Extract or interpolate mesh data at specified query points.

    This function retrieves mesh data at arbitrary spatial locations. Note that
    "sample" here means "extract/query at specific points" - NOT random sampling.
    For random point sampling, see ``sample_random_points_on_cells``.

    For each query point, the function:
    1. Finds which cell(s) contain the point using barycentric coordinates
    2. Extracts cell data directly (data_source="cells") or interpolates point
       data using barycentric coordinates (data_source="points")

    Two containment-search strategies are available:

    - **Brute-force** (default, ``bvh=None``): Tests all cells for each query.
      Complexity is O(n_queries * n_cells). Supports all features including
      ``project_onto_nearest_cell``.
    - **BVH-accelerated** (``bvh`` provided): Uses a Bounding Volume Hierarchy
      to prune the search space. Complexity is O(n_queries * log(n_cells)).
      For large meshes (>10k cells) this can be dramatically faster. Build
      the BVH once with ``BVH.from_mesh(mesh)`` and reuse it across calls.

    Parameters
    ----------
    mesh : Mesh
        The mesh to extract data from.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims).
    data_source : {"cells", "points"}, optional
        How to retrieve data:
        - "cells": Use cell data directly (no interpolation)
        - "points": Interpolate point data using barycentric coordinates
    multiple_cells_strategy : {"mean", "nan"}, optional
        How to handle query points contained in multiple cells:
        - "mean": Return arithmetic mean of values from all containing cells
        - "nan": Return NaN for ambiguous points
    project_onto_nearest_cell : bool, optional
        If True, projects each query point onto the nearest cell before
        sampling. Useful for codimension != 0 manifolds. Only supported
        with brute-force search (``bvh=None``).
    tolerance : float, optional
        Tolerance for considering a point inside a cell. A point is inside if
        all barycentric coordinates >= -tolerance AND reconstruction error
        <= tolerance.
    bvh : BVH or None, optional
        Pre-built Bounding Volume Hierarchy for accelerated spatial queries.
        If None (default), uses brute-force O(n_queries * n_cells) search.
        If provided, uses O(n_queries * log(n_cells)) BVH search.

    Returns
    -------
    TensorDict
        Sampled data for each query point, with the same keys as
        mesh.cell_data (if data_source="cells") or mesh.point_data
        (if data_source="points"). Values are NaN for query points
        outside the mesh (unless project_onto_nearest_cell=True).

    Raises
    ------
    ValueError
        If data_source or multiple_cells_strategy is invalid.
    NotImplementedError
        If project_onto_nearest_cell=True with BVH acceleration.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh.cell_data["pressure"] = torch.tensor([1.0, 2.0])
    >>> query_pts = torch.tensor([[0.3, 0.3], [0.8, 0.5]])
    >>> sampled = sample_data_at_points(mesh, query_pts, data_source="cells")
    >>> assert "pressure" in sampled.keys()

    BVH-accelerated sampling for large meshes:

    >>> from physicsnemo.mesh.spatial import BVH  # doctest: +SKIP
    >>> bvh = BVH.from_mesh(mesh)  # doctest: +SKIP
    >>> sampled = sample_data_at_points(mesh, query_pts, bvh=bvh)  # doctest: +SKIP
    """
    if data_source not in ("cells", "points"):
        raise ValueError(f"Invalid {data_source=}. Must be 'cells' or 'points'.")

    if multiple_cells_strategy not in ("mean", "nan"):
        raise ValueError(
            f"Invalid {multiple_cells_strategy=}. Must be 'mean' or 'nan'."
        )

    n_queries = query_points.shape[0]

    ### Handle projection onto nearest cell
    if project_onto_nearest_cell:
        if bvh is not None:
            raise NotImplementedError(
                "project_onto_nearest_cell is not yet supported with BVH acceleration. "
                "Pass bvh=None to use brute-force search with projection."
            )
        _, projected_points = find_nearest_cells(mesh, query_points)
        query_points = projected_points

    ### Find containing pairs using the appropriate strategy
    if bvh is not None:
        query_indices, cell_indices, bary_coords = _find_containing_pairs_bvh(
            mesh, query_points, bvh, tolerance
        )
    else:
        query_indices, cell_indices, bary_coords = _find_containing_pairs_bruteforce(
            mesh, query_points, tolerance
        )

    ### Accumulate sampled data (shared logic for both paths)
    return _accumulate_sampled_data(
        mesh=mesh,
        n_queries=n_queries,
        query_indices=query_indices,
        cell_indices=cell_indices,
        bary_coords=bary_coords,
        data_source=data_source,
        multiple_cells_strategy=multiple_cells_strategy,
    )
