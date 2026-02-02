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

"""Hierarchical spatial data sampling using BVH acceleration.

This module provides BVH-accelerated data sampling at query points, achieving
O(M*log(N)) complexity instead of O(M*N) for large meshes.
"""

from typing import TYPE_CHECKING, Literal

import torch
from tensordict import TensorDict

from physicsnemo.mesh.sampling.sample_data import (
    compute_barycentric_coordinates_pairwise,
)
from physicsnemo.mesh.spatial import BVH

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def sample_data_at_points(
    mesh: "Mesh",
    query_points: torch.Tensor,
    bvh: BVH | None = None,
    data_source: Literal["cells", "points"] = "cells",
    multiple_cells_strategy: Literal["mean", "nan"] = "mean",
    project_onto_nearest_cell: bool = False,
    tolerance: float = 1e-6,
) -> TensorDict:
    """Sample mesh data at query points using BVH acceleration.

    This function has the same API as physicsnemo.mesh.sampling.sample_data.sample_data_at_points
    but uses a Bounding Volume Hierarchy for O(log N) spatial queries instead of O(N).

    For meshes with many cells (>10,000), this can be significantly faster. For small
    meshes, the overhead of BVH traversal may make it slower than brute-force.

    Args:
        mesh: The mesh to sample from.
        query_points: Query point locations, shape (n_queries, n_spatial_dims)
        bvh: Pre-computed BVH for the mesh. If None, one will be built automatically.
            For multiple queries on the same mesh, pre-building the BVH is recommended.
        data_source: How to sample data:
            - "cells": Use cell data directly (no interpolation)
            - "points": Interpolate point data using barycentric coordinates
        multiple_cells_strategy: How to handle query points in multiple cells:
            - "mean": Return arithmetic mean of values from all containing cells
            - "nan": Return NaN for ambiguous points
        project_onto_nearest_cell: If True, projects each query point onto the
            nearest cell before sampling. Useful for codimension != 0 manifolds.
            Note: Projection is not yet BVH-accelerated and may be slow.
        tolerance: Tolerance for considering a point inside a cell.

    Returns:
        TensorDict containing sampled data for each query point. Values are NaN
        for query points outside the mesh (unless project_onto_nearest_cell=True).

    Raises:
        ValueError: If data_source or multiple_cells_strategy is invalid.
        NotImplementedError: If project_onto_nearest_cell=True (not yet implemented
            with BVH acceleration).

    Example:
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> from physicsnemo.mesh.spatial import BVH
        >>> mesh = two_triangles_2d.load()
        >>> mesh.cell_data["pressure"] = torch.tensor([1.0, 2.0])
        >>> # Build BVH once, reuse for many queries
        >>> bvh = BVH.from_mesh(mesh)
        >>> query_pts = torch.tensor([[0.3, 0.3]])
        >>> result = sample_data_at_points(mesh, query_pts, bvh=bvh)
        >>> assert "pressure" in result.keys()
    """
    if data_source not in ["cells", "points"]:
        raise ValueError(f"Invalid {data_source=}. Must be 'cells' or 'points'.")

    if multiple_cells_strategy not in ["mean", "nan"]:
        raise ValueError(
            f"Invalid {multiple_cells_strategy=}. Must be 'mean' or 'nan'."
        )

    if project_onto_nearest_cell:
        raise NotImplementedError(
            "project_onto_nearest_cell is not yet implemented with BVH acceleration. "
            "Use the non-hierarchical sample_data_at_points for this feature."
        )

    n_queries = query_points.shape[0]

    ### Build BVH if not provided
    if bvh is None:
        bvh = BVH.from_mesh(mesh)

    ### Find candidate cells for each query point using BVH
    # Use same tolerance for AABB checks as for barycentric coordinate checks
    candidate_cells_list = bvh.find_candidate_cells(
        query_points, aabb_tolerance=tolerance
    )

    ### Flatten all query-candidate pairs for batch processing (vectorized)
    # Convert list of tensors to a format suitable for batching
    # Each element in candidate_cells_list has variable length
    query_indices_list = []
    cell_indices_list = []

    for i, candidates in enumerate(candidate_cells_list):
        if len(candidates) > 0:
            query_indices_list.append(
                torch.full(
                    (len(candidates),), i, dtype=torch.long, device=mesh.points.device
                )
            )
            cell_indices_list.append(candidates)

    if len(query_indices_list) == 0:
        # No candidates at all
        query_indices_candidates = torch.tensor(
            [], dtype=torch.long, device=mesh.points.device
        )
        cell_indices_candidates = torch.tensor(
            [], dtype=torch.long, device=mesh.points.device
        )
    else:
        # Concatenate all pairs
        query_indices_candidates = torch.cat(query_indices_list)
        cell_indices_candidates = torch.cat(cell_indices_list)

    if len(query_indices_candidates) > 0:
        ### Batch compute barycentric coordinates for all candidates
        # Get query points and cell vertices for each pair
        candidate_query_points = query_points[
            query_indices_candidates
        ]  # (n_pairs, n_spatial_dims)
        candidate_cell_vertices = mesh.points[
            mesh.cells[cell_indices_candidates]
        ]  # (n_pairs, n_vertices, n_spatial_dims)

        ### Use pairwise barycentric computation (O(n) instead of O(n²))
        # This computes only the diagonal elements we need, avoiding massive memory allocation
        bary_coords_candidates = compute_barycentric_coordinates_pairwise(
            candidate_query_points,
            candidate_cell_vertices,
        )  # (n_pairs, n_vertices)

        ### Check which candidates actually contain their query point
        is_inside = (bary_coords_candidates >= -tolerance).all(dim=-1)  # (n_pairs,)

        ### Filter to only the containing pairs
        query_indices = query_indices_candidates[is_inside]
        cell_indices_containing = cell_indices_candidates[is_inside]
        bary_coords_for_containing = bary_coords_candidates[
            is_inside
        ]  # (n_containing, n_vertices)
    else:
        query_indices = torch.tensor([], dtype=torch.long, device=mesh.points.device)
        cell_indices_containing = torch.tensor(
            [], dtype=torch.long, device=mesh.points.device
        )
        bary_coords_for_containing = None

    ### Count how many cells contain each query point
    query_containment_count = torch.zeros(
        n_queries, dtype=torch.long, device=mesh.points.device
    )
    if len(query_indices) > 0:
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

    ### Sample each field in the source_data (vectorized with scatter operations)
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

        if len(query_indices) == 0:
            # No containments - all NaN
            result[key] = output
            continue

        if data_source == "cells":
            ### Use cell data directly - vectorized with scatter
            cell_data_for_pairs = values[cell_indices_containing]

            if multiple_cells_strategy == "mean":
                # Sum and average using scatter
                if values.ndim == 1:
                    output_sum = torch.zeros(
                        n_queries, dtype=values.dtype, device=mesh.points.device
                    )
                    output_sum.scatter_add_(0, query_indices, cell_data_for_pairs)
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
                    ).expand_as(cell_data_for_pairs)
                    output_sum.scatter_add_(0, expanded_indices, cell_data_for_pairs)
                    valid_mask = query_containment_count > 0
                    output[valid_mask] = output_sum[
                        valid_mask
                    ] / query_containment_count[valid_mask].to(values.dtype).view(
                        -1, *([1] * (values.ndim - 1))
                    )
            else:  # "nan" strategy
                single_cell_mask = query_containment_count == 1
                if single_cell_mask.any():
                    query_has_single_cell = single_cell_mask[query_indices]
                    single_cell_query_idx = query_indices[query_has_single_cell]
                    single_cell_values = cell_data_for_pairs[query_has_single_cell]
                    output[single_cell_query_idx] = single_cell_values

        else:  # data_source == "points"
            ### Interpolate point data using barycentric coordinates (vectorized)
            # Get point indices for all containing cells
            point_indices_for_pairs = mesh.cells[
                cell_indices_containing
            ]  # (n_pairs, n_vertices)

            # Get point data values
            point_values_for_pairs = values[
                point_indices_for_pairs
            ]  # (n_pairs, n_vertices, ...)

            # Interpolate using barycentric coordinates
            if values.ndim == 1:
                interpolated = (
                    bary_coords_for_containing * point_values_for_pairs
                ).sum(dim=1)
            else:
                bary_expanded = bary_coords_for_containing.view(
                    bary_coords_for_containing.shape[0],
                    bary_coords_for_containing.shape[1],
                    *([1] * (values.ndim - 1)),
                )
                interpolated = (bary_expanded * point_values_for_pairs).sum(dim=1)

            if multiple_cells_strategy == "mean":
                # Average using scatter
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
                single_cell_mask = query_containment_count == 1
                if single_cell_mask.any():
                    query_has_single_cell = single_cell_mask[query_indices]
                    single_cell_query_idx = query_indices[query_has_single_cell]
                    single_cell_values = interpolated[query_has_single_cell]
                    output[single_cell_query_idx] = single_cell_values

        result[key] = output

    return result
