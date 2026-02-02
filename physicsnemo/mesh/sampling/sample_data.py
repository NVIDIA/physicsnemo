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

"""Spatial sampling of data at query points in a mesh."""

from typing import TYPE_CHECKING, Literal

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _solve_barycentric_system(
    relative_vectors: torch.Tensor,  # shape: (..., n_manifold_dims, n_spatial_dims)
    query_relative: torch.Tensor,  # shape: (..., n_spatial_dims)
) -> torch.Tensor:
    """Core barycentric coordinate solver (shared by both variants).

    Solves the linear system to find barycentric coordinates w_1, ..., w_n such that:
        query_relative = sum(w_i * relative_vectors[i])

    Then computes w_0 = 1 - sum(w_i) and returns all coordinates [w_0, w_1, ..., w_n].

    Args:
        relative_vectors: Edge vectors from first vertex to others,
            shape (..., n_manifold_dims, n_spatial_dims)
        query_relative: Query point relative to first vertex,
            shape (..., n_spatial_dims)

    Returns:
        Barycentric coordinates, shape (..., n_vertices_per_cell) where
        n_vertices_per_cell = n_manifold_dims + 1

    Algorithm:
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

    else:
        ### Over-determined or under-determined system: use least squares
        A = relative_vectors.transpose(-2, -1)
        b = query_relative.unsqueeze(-1)
        weights_1_to_n = torch.linalg.lstsq(A, b).solution.squeeze(-1)

    ### Compute w_0 = 1 - sum(w_i for i=1..n)
    w_0 = 1.0 - weights_1_to_n.sum(dim=-1, keepdim=True)

    ### Concatenate to get all barycentric coordinates
    barycentric_coords = torch.cat([w_0, weights_1_to_n], dim=-1)

    return barycentric_coords


def compute_barycentric_coordinates(
    query_points: torch.Tensor,
    cell_vertices: torch.Tensor,
) -> torch.Tensor:
    """Compute barycentric coordinates of query points with respect to simplices.

    For each query point and each simplex, computes the barycentric coordinates.
    A point is inside a simplex if all barycentric coordinates are non-negative.

    Args:
        query_points: Query point locations, shape (n_queries, n_spatial_dims)
        cell_vertices: Vertices of cells to test, shape (n_cells, n_vertices_per_cell, n_spatial_dims)

    Returns:
        Barycentric coordinates, shape (n_queries, n_cells, n_vertices_per_cell).
        For each query-cell pair, the coordinates sum to 1.

    Algorithm:
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
    barycentric_coords = _solve_barycentric_system(
        relative_vectors_expanded, query_relative
    )

    return barycentric_coords


def compute_barycentric_coordinates_pairwise(
    query_points: torch.Tensor,
    cell_vertices: torch.Tensor,
) -> torch.Tensor:
    """Compute barycentric coordinates for paired queries and cells.

    Unlike compute_barycentric_coordinates which computes all O(n_queries × n_cells)
    combinations, this computes only n_pairs diagonal elements where each query point
    is paired with exactly one cell. This uses O(n) memory instead of O(n²).

    This is critical for performance when processing BVH candidate pairs, where we may
    have thousands of pairs but don't need the full cartesian product.

    Args:
        query_points: Query point locations, shape (n_pairs, n_spatial_dims)
        cell_vertices: Vertices of cells, shape (n_pairs, n_vertices_per_cell, n_spatial_dims)
            where cell_vertices[i] is paired with query_points[i]

    Returns:
        Barycentric coordinates, shape (n_pairs, n_vertices_per_cell).
        For each pair, the coordinates sum to 1.

    Example:
        >>> import torch
        >>> # For BVH results: each query has specific candidate cells
        >>> n_pairs = 1000
        >>> query_points = torch.randn(n_pairs, 3)
        >>> cell_vertices = torch.randn(n_pairs, 3, 3)  # Triangles in 3D
        >>> bary = compute_barycentric_coordinates_pairwise(query_points, cell_vertices)
        >>> assert bary.shape == (1000, 3)  # instead of (1000, 1000, 3) from full version
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
    barycentric_coords = _solve_barycentric_system(relative_vectors, query_relative)

    return barycentric_coords


def find_containing_cells(
    mesh: "Mesh",
    query_points: torch.Tensor,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find which cells contain each query point.

    Args:
        mesh: The mesh to query.
        query_points: Query point locations, shape (n_queries, n_spatial_dims)
        tolerance: Tolerance for considering a point inside a cell.
            A point is inside if all barycentric coordinates >= -tolerance.

    Returns:
        Tuple of (cell_indices, barycentric_coords):
        - cell_indices: Cell index for each query point, shape (n_queries,).
            Value is -1 if no cell contains the point, or the first containing cell index.
        - barycentric_coords: Barycentric coordinates for each query point in its
            containing cell, shape (n_queries, n_vertices_per_cell).
            Values are NaN if no containing cell exists.

    Note:
        If multiple cells contain a point, only the first is returned.
        Use find_all_containing_cells() to get all containing cells.
    """
    n_queries = query_points.shape[0]
    n_vertices_per_cell = mesh.n_manifold_dims + 1

    ### Get cell vertices: (n_cells, n_vertices_per_cell, n_spatial_dims)
    cell_vertices = mesh.points[mesh.cells]

    ### Compute barycentric coordinates for all query-cell pairs
    # Shape: (n_queries, n_cells, n_vertices_per_cell)
    bary_coords = compute_barycentric_coordinates(query_points, cell_vertices)

    ### Determine which query-cell pairs have the point inside
    # A point is inside if all barycentric coordinates are >= -tolerance
    # Shape: (n_queries, n_cells)
    is_inside = (bary_coords >= -tolerance).all(dim=-1)

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
) -> list[torch.Tensor]:
    """Find all cells that contain each query point.

    Args:
        mesh: The mesh to query.
        query_points: Query point locations, shape (n_queries, n_spatial_dims)
        tolerance: Tolerance for considering a point inside a cell.

    Returns:
        List of length n_queries, where each element is a tensor of cell indices
        that contain that query point. Empty tensor if no cells contain the point.
    """
    ### Get cell vertices: (n_cells, n_vertices_per_cell, n_spatial_dims)
    cell_vertices = mesh.points[mesh.cells]

    ### Compute barycentric coordinates for all query-cell pairs
    bary_coords = compute_barycentric_coordinates(query_points, cell_vertices)

    ### Determine which query-cell pairs have the point inside
    is_inside = (bary_coords >= -tolerance).all(dim=-1)

    ### For each query, collect all containing cells
    containing_cells = []
    for i in range(len(query_points)):
        containing = torch.where(is_inside[i])[0]
        containing_cells.append(containing)

    return containing_cells


def project_point_onto_cell(
    query_point: torch.Tensor,
    cell_vertices: torch.Tensor,
) -> tuple[torch.Tensor, float | torch.Tensor]:
    """Project a query point onto a simplex (cell).

    Args:
        query_point: Point to project, shape (n_spatial_dims,)
        cell_vertices: Vertices of the simplex, shape (n_vertices, n_spatial_dims)

    Returns:
        Tuple of (projected_point, squared_distance):
        - projected_point: Closest point on the simplex, shape (n_spatial_dims,)
        - squared_distance: Squared distance from query to projection, scalar
    """
    ### This is a complex optimization problem. For now, use a simple approach:
    # 1. Project onto the affine hull of the simplex
    # 2. If the projection is inside, return it
    # 3. Otherwise, recursively project onto lower-dimensional faces

    # Compute barycentric coordinates
    bary = (
        compute_barycentric_coordinates(
            query_point.unsqueeze(0),
            cell_vertices.unsqueeze(0),
        )
        .squeeze(0)
        .squeeze(0)
    )  # (n_vertices,)

    ### If all barycentric coords are non-negative, point projects inside the simplex
    if (bary >= 0).all():
        projected = (bary.unsqueeze(-1) * cell_vertices).sum(dim=0)
        dist_sq = ((query_point - projected) ** 2).sum()
        return projected, dist_sq

    ### Otherwise, find the closest face
    # For simplicity, check all faces (subsets of vertices)
    n_vertices = cell_vertices.shape[0]
    best_projected = None
    best_dist_sq = float("inf")

    # Try all (n-1)-dimensional faces
    for i in range(n_vertices):
        # Face is all vertices except vertex i
        face_vertices = torch.cat([cell_vertices[:i], cell_vertices[i + 1 :]], dim=0)
        if len(face_vertices) == 1:
            # Single vertex
            projected = face_vertices[0]
            dist_sq = ((query_point - projected) ** 2).sum()
        else:
            # Recursively project onto face
            projected, dist_sq = project_point_onto_cell(query_point, face_vertices)

        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_projected = projected

    return best_projected, best_dist_sq


def find_nearest_cells(
    mesh: "Mesh",
    query_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest cell for each query point.

    This is a simplified implementation that finds the cell whose centroid is nearest.
    A more accurate projection onto cell surfaces would require complex optimization.

    Args:
        mesh: The mesh to query.
        query_points: Query point locations, shape (n_queries, n_spatial_dims)

    Returns:
        Tuple of (cell_indices, projected_points):
        - cell_indices: Nearest cell index for each query point, shape (n_queries,)
        - projected_points: Centroids of nearest cells (approximation of projection),
            shape (n_queries, n_spatial_dims)

    Note:
        This is a simplified version using centroid distances. Full projection onto
        simplices would require iterative optimization and is complex to vectorize.
    """
    ### Compute all cell centroids
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)

    ### Compute distances from all queries to all cell centroids
    # query_points: (n_queries, n_spatial_dims)
    # cell_centroids: (n_cells, n_spatial_dims)
    # Broadcast to (n_queries, n_cells, n_spatial_dims)
    diffs = query_points.unsqueeze(1) - cell_centroids.unsqueeze(0)
    distances_sq = (diffs**2).sum(dim=-1)  # (n_queries, n_cells)

    ### Find nearest cell for each query
    cell_indices = distances_sq.argmin(dim=1)  # (n_queries,)

    ### Return centroids of nearest cells as approximation of projection
    projected_points = cell_centroids[cell_indices]  # (n_queries, n_spatial_dims)

    return cell_indices, projected_points


def sample_data_at_points(
    mesh: "Mesh",
    query_points: torch.Tensor,
    data_source: Literal["cells", "points"] = "cells",
    multiple_cells_strategy: Literal["mean", "nan"] = "mean",
    project_onto_nearest_cell: bool = False,
    tolerance: float = 1e-6,
) -> TensorDict:
    """Sample mesh data at query points in space.

    For each query point, finds the containing cell and returns interpolated data.

    Args:
        mesh: The mesh to sample from.
        query_points: Query point locations, shape (n_queries, n_spatial_dims)
        data_source: How to sample data:
            - "cells": Use cell data directly (no interpolation)
            - "points": Interpolate point data using barycentric coordinates
        multiple_cells_strategy: How to handle query points contained in multiple cells:
            - "mean": Return arithmetic mean of values from all containing cells
            - "nan": Return NaN for ambiguous points
        project_onto_nearest_cell: If True, projects each query point onto the
            nearest cell before sampling. This is useful for codimension != 0 manifolds
            where picking a point exactly on the manifold is difficult due to
            floating-point precision.
        tolerance: Tolerance for considering a point inside a cell (for barycentric coords).

    Returns:
        TensorDict containing sampled data for each query point, with the same keys
        as mesh.cell_data (if data_source="cells") or mesh.point_data (if data_source="points").
        Values are NaN for query points outside the mesh (unless project_onto_nearest_cell=True).

    Raises:
        ValueError: If data_source is invalid.

    Example:
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> mesh.cell_data["pressure"] = torch.tensor([1.0, 2.0])
        >>> # Sample cell data at specific points
        >>> query_pts = torch.tensor([[0.3, 0.3], [0.8, 0.5]])
        >>> sampled_data = sample_data_at_points(mesh, query_pts, data_source="cells")
        >>> assert "pressure" in sampled_data.keys()
    """
    if data_source not in ["cells", "points"]:
        raise ValueError(f"Invalid {data_source=}. Must be 'cells' or 'points'.")

    if multiple_cells_strategy not in ["mean", "nan"]:
        raise ValueError(
            f"Invalid {multiple_cells_strategy=}. Must be 'mean' or 'nan'."
        )

    n_queries = query_points.shape[0]

    ### Handle projection onto nearest cell if requested
    if project_onto_nearest_cell:
        _, projected_points = find_nearest_cells(mesh, query_points)
        query_points = projected_points

    ### Find containing cells for each query point
    # Get cell vertices and compute all barycentric coordinates
    cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_vertices, n_spatial_dims)
    bary_coords_all = compute_barycentric_coordinates(query_points, cell_vertices)

    # Determine which query-cell pairs have containment
    is_inside = (bary_coords_all >= -tolerance).all(dim=-1)  # (n_queries, n_cells)

    ### Get flat arrays of query and cell indices for all containments
    query_indices, cell_indices_containing = torch.where(is_inside)

    ### Count how many cells contain each query point
    # Use scatter to count containments per query
    query_containment_count = torch.zeros(
        n_queries, dtype=torch.long, device=mesh.points.device
    )
    query_containment_count.scatter_add_(
        0, query_indices, torch.ones_like(query_indices)
    )

    ### Initialize result TensorDict
    source_data = mesh.cell_data if data_source == "cells" else mesh.point_data
    result = TensorDict(
        {},
        batch_size=torch.Size([n_queries]),
        device=mesh.points.device,
    )

    ### Sample each field in the source_data
    for key, values in source_data.exclude("_cache").items():
        # Determine output shape
        if values.ndim == 1:
            output_shape = (n_queries,)
        else:
            output_shape = (n_queries,) + values.shape[1:]

        # Initialize output with NaN
        output = torch.full(
            output_shape,
            float("nan"),
            dtype=values.dtype,
            device=mesh.points.device,
        )

        if data_source == "cells":
            ### Use cell data directly - vectorized with scatter
            # Get cell data for all query-cell pairs that have containment
            cell_data_for_pairs = values[cell_indices_containing]  # (n_pairs, ...)

            if multiple_cells_strategy == "mean":
                # Sum up contributions using scatter_add
                if values.ndim == 1:
                    # Scalar case
                    output_sum = torch.zeros(
                        n_queries, dtype=values.dtype, device=mesh.points.device
                    )
                    output_sum.scatter_add_(0, query_indices, cell_data_for_pairs)
                    # Divide by count (avoiding division by zero)
                    valid_mask = query_containment_count > 0
                    output[valid_mask] = output_sum[
                        valid_mask
                    ] / query_containment_count[valid_mask].to(values.dtype)
                else:
                    # Multi-dimensional case
                    output_sum = torch.zeros(
                        output_shape, dtype=values.dtype, device=mesh.points.device
                    )
                    expanded_indices = query_indices.view(
                        -1, *([1] * (values.ndim - 1))
                    ).expand_as(cell_data_for_pairs)
                    output_sum.scatter_add_(0, expanded_indices, cell_data_for_pairs)
                    # Divide by count with broadcasting
                    valid_mask = query_containment_count > 0
                    output[valid_mask] = output_sum[
                        valid_mask
                    ] / query_containment_count[valid_mask].to(values.dtype).view(
                        -1, *([1] * (values.ndim - 1))
                    )
            else:  # "nan" strategy
                # Only assign for queries with exactly one containing cell
                single_cell_mask = query_containment_count == 1
                if single_cell_mask.any():
                    # Find which pairs correspond to single-cell queries
                    query_has_single_cell = single_cell_mask[query_indices]
                    single_cell_query_idx = query_indices[query_has_single_cell]
                    single_cell_values = cell_data_for_pairs[query_has_single_cell]
                    output[single_cell_query_idx] = single_cell_values

        else:  # data_source == "points"
            ### Interpolate point data using barycentric coordinates (vectorized)
            # Get barycentric coords for all query-cell pairs with containment
            bary_for_pairs = bary_coords_all[
                query_indices, cell_indices_containing
            ]  # (n_pairs, n_vertices)

            # Get point indices for all cells with containment
            point_indices_for_pairs = mesh.cells[
                cell_indices_containing
            ]  # (n_pairs, n_vertices)

            # Get point data values for these vertices
            point_values_for_pairs = values[
                point_indices_for_pairs
            ]  # (n_pairs, n_vertices, ...)

            # Interpolate using barycentric coordinates
            if values.ndim == 1:
                # Scalar: (n_pairs, n_vertices) * (n_pairs, n_vertices) -> sum over vertices
                interpolated = (bary_for_pairs * point_values_for_pairs).sum(
                    dim=1
                )  # (n_pairs,)
            else:
                # Multi-dimensional: broadcast barycentric coords
                bary_expanded = bary_for_pairs.view(
                    bary_for_pairs.shape[0],
                    bary_for_pairs.shape[1],
                    *([1] * (values.ndim - 1)),
                )
                interpolated = (bary_expanded * point_values_for_pairs).sum(
                    dim=1
                )  # (n_pairs, ...)

            if multiple_cells_strategy == "mean":
                # Average interpolated values using scatter
                if values.ndim == 1:
                    output_sum = torch.zeros(
                        n_queries, dtype=values.dtype, device=mesh.points.device
                    )
                    output_sum.scatter_add_(0, query_indices, interpolated)
                    valid_mask = query_containment_count > 0
                    output[valid_mask] = output_sum[
                        valid_mask
                    ] / query_containment_count[valid_mask].to(values.dtype)
                else:
                    output_sum = torch.zeros(
                        output_shape, dtype=values.dtype, device=mesh.points.device
                    )
                    expanded_indices = query_indices.view(
                        -1, *([1] * (values.ndim - 1))
                    ).expand_as(interpolated)
                    output_sum.scatter_add_(0, expanded_indices, interpolated)
                    valid_mask = query_containment_count > 0
                    output[valid_mask] = output_sum[
                        valid_mask
                    ] / query_containment_count[valid_mask].to(values.dtype).view(
                        -1, *([1] * (values.ndim - 1))
                    )
            else:  # "nan" strategy
                # Only assign for queries with exactly one containing cell
                single_cell_mask = query_containment_count == 1
                if single_cell_mask.any():
                    query_has_single_cell = single_cell_mask[query_indices]
                    single_cell_query_idx = query_indices[query_has_single_cell]
                    single_cell_values = interpolated[query_has_single_cell]
                    output[single_cell_query_idx] = single_cell_values

        result[key] = output

    return result
