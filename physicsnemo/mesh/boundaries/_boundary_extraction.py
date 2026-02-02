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

"""Boundary mesh extraction for simplicial meshes.

This module extracts boundary facets - i.e., codimension-1 facets that appear in
exactly one parent cell. This produces the watertight boundary surface of a mesh.

Key difference from facet extraction:
- Facet mesh: ALL facets (interior + boundary)
- Boundary mesh: ONLY facets that appear in exactly 1 cell
"""

from typing import TYPE_CHECKING, Literal

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def extract_boundary_mesh_data(
    parent_mesh: "Mesh",
    data_source: Literal["points", "cells"] = "cells",
    data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
) -> tuple[torch.Tensor, TensorDict]:
    """Extract boundary mesh data from parent mesh.

    Extracts only the codimension-1 facets that lie on the boundary (appear in
    exactly one parent cell). This produces the watertight boundary surface.

    Args:
        parent_mesh: The parent mesh to extract boundary from
        data_source: Whether to inherit data from "cells" or "points"
        data_aggregation: How to aggregate data (only applies when data_source="cells")
            Note: For boundary facets, each facet has exactly one parent cell,
            so aggregation only matters if the same boundary facet appears multiple
            times (which shouldn't happen in a valid mesh).

    Returns:
        boundary_cells: Connectivity for boundary mesh, shape (n_boundary_facets, n_vertices_per_facet)
        boundary_cell_data: Aggregated TensorDict for boundary mesh cells

    Example:
        >>> from physicsnemo.mesh.primitives.procedural import lumpy_ball
        >>> from physicsnemo.mesh import Mesh
        >>> # Extract surface of a volume mesh
        >>> vol_mesh = lumpy_ball.load(n_shells=2, subdivisions=1)
        >>> boundary_cells, boundary_data = extract_boundary_mesh_data(vol_mesh)
        >>> boundary_mesh = Mesh(points=vol_mesh.points, cells=boundary_cells, cell_data=boundary_data)
        >>> assert boundary_mesh.n_manifold_dims == 2  # Surface triangles
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        _aggregate_point_data_to_facets,
        categorize_facets_by_count,
        compute_aggregation_weights,
        extract_candidate_facets,
    )

    ### Extract all candidate codimension-1 facets
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        parent_mesh.cells,
        manifold_codimension=1,  # Always codimension-1 for boundaries
    )

    ### Filter to boundary facets (appear exactly once)
    boundary_facets, inverse_indices, _ = categorize_facets_by_count(
        candidate_facets, target_counts="boundary"
    )
    n_boundary_facets = len(boundary_facets)

    ### Extract parent cells for boundary facets
    # inverse_indices maps candidate facets to filtered unique facets
    # We need only the candidates that map to valid filtered facets (not -1)
    boundary_facet_mask = inverse_indices >= 0
    boundary_parent_indices = parent_cell_indices[boundary_facet_mask]
    boundary_facets_candidates = candidate_facets[boundary_facet_mask]

    ### Get mapping from boundary candidates to unique boundary facets
    # Since we already filtered, we just need to get the inverse mapping
    _, boundary_inverse = torch.unique(
        boundary_facets_candidates,
        dim=0,
        return_inverse=True,
    )

    ### Initialize empty output TensorDict
    boundary_cell_data = TensorDict(
        {},
        batch_size=torch.Size([n_boundary_facets]),
        device=parent_mesh.points.device,
    )

    ### Aggregate data based on source
    if data_source == "cells":
        ### Aggregate data from parent cells
        if len(parent_mesh.cell_data.keys()) > 0:
            ### Filter out cached properties
            filtered_cell_data = parent_mesh.cell_data.exclude("_cache")

            if len(filtered_cell_data.keys()) > 0:
                ### Compute facet centroids if needed for inverse_distance
                facet_centroids = None
                if data_aggregation == "inverse_distance":
                    # Compute centroid of each boundary candidate facet
                    facet_points = parent_mesh.points[boundary_facets_candidates]
                    facet_centroids = facet_points.mean(dim=1)

                ### Prepare parent cell areas and centroids if needed
                parent_cell_areas = None
                parent_cell_centroids = None

                if data_aggregation == "area_weighted":
                    parent_cell_areas = parent_mesh.cell_areas
                if data_aggregation == "inverse_distance":
                    parent_cell_centroids = parent_mesh.cell_centroids

                ### Compute aggregation weights
                weights = compute_aggregation_weights(
                    aggregation_strategy=data_aggregation,
                    parent_cell_areas=parent_cell_areas,
                    parent_cell_centroids=parent_cell_centroids,
                    facet_centroids=facet_centroids,
                    parent_cell_indices=boundary_parent_indices,
                )

                ### Aggregate data from parent cells to boundary facets
                # Since boundary facets appear in exactly 1 cell, aggregation is simpler
                from physicsnemo.mesh.boundaries._facet_extraction import (
                    _aggregate_tensor_data,
                )

                boundary_cell_data = filtered_cell_data.apply(
                    lambda tensor: _aggregate_tensor_data(
                        tensor,
                        boundary_parent_indices,
                        boundary_inverse,
                        n_boundary_facets,
                        weights,
                    ),
                    batch_size=torch.Size([n_boundary_facets]),
                )

    elif data_source == "points":
        ### Aggregate data from boundary points of each facet
        if len(parent_mesh.point_data.keys()) > 0:
            ### Average point data over facet vertices
            boundary_cell_data = _aggregate_point_data_to_facets(
                point_data=parent_mesh.point_data,
                candidate_facets=boundary_facets_candidates,
                inverse_indices=boundary_inverse,
                n_unique_facets=n_boundary_facets,
            )

    else:
        raise ValueError(f"Invalid {data_source=}. Must be one of: 'points', 'cells'")

    return boundary_facets, boundary_cell_data
