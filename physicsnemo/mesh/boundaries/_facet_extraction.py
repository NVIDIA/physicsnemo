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

"""High-performance facet extraction for simplicial meshes.

This module extracts k-codimension simplices from n-simplicial meshes. For example:
- Triangle meshes (2-simplices) → edge meshes (1-simplices) [codimension 1]
- Tetrahedral meshes (3-simplices) → triangular facets (2-simplices) [codimension 1]
- Tetrahedral meshes (3-simplices) → edge meshes (1-simplices) [codimension 2]
- Triangle meshes (2-simplices) → point meshes (0-simplices) [codimension 2]

Note: Originally designed to use Triton kernels, but Triton requires all array sizes
to be powers of 2, which doesn't work for triangles (3 vertices) or tets (4 vertices).
The pure PyTorch implementation here is highly optimized and performs excellently.
"""

from typing import TYPE_CHECKING, Literal

import torch
from tensordict import TensorDict

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _generate_combination_indices(n: int, k: int) -> torch.Tensor:
    """Generate all combinations of k elements from n elements.

    This is a vectorized implementation similar to itertools.combinations(range(n), k).

    Parameters
    ----------
    n : int
        Total number of elements
    k : int
        Number of elements to choose

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_choose_k, k) containing all combinations

    Examples
    --------
    >>> _generate_combination_indices(4, 2)
    tensor([[0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3]])
    """
    from itertools import combinations

    ### Use standard library for correctness
    # For small values of n and k (which is always the case for simplicial meshes),
    # this is fast enough and avoids reinventing the wheel
    combos = list(combinations(range(n), k))
    return torch.tensor(combos, dtype=torch.int64)


def categorize_facets_by_count(
    candidate_facets: torch.Tensor,  # shape: (n_candidate_facets, n_vertices_per_facet)
    target_counts: list[int] | Literal["boundary", "shared", "interior", "all"] = "all",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate facets and optionally filter by occurrence count.

    This utility consolidates the common pattern of deduplicating facets using
    torch.unique and filtering based on how many times each facet appears.

    Parameters
    ----------
    candidate_facets : torch.Tensor
        All candidate facets (may contain duplicates), already sorted
    target_counts : list[int] | {"boundary", "shared", "interior", "all"}, optional
        How to filter the results:
        - "all": Return all unique facets with their counts (no filtering)
        - "boundary": Return facets appearing exactly once (counts == 1)
        - "interior": Return facets appearing exactly twice (counts == 2)
        - "shared": Return facets appearing 2+ times (counts >= 2)
        - list[int]: Return facets with counts in the specified list

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of (unique_facets, inverse_indices, counts):
        - unique_facets: Deduplicated facets, possibly filtered by count
        - inverse_indices: Mapping from candidate facets to unique facet indices
        - counts: How many times each unique facet appears

        If filtering is applied, only the matching facets and their data are returned.

    Examples
    --------
    >>> import torch
    >>> # Create candidate facets from a simple mesh (edges from 2 triangles)
    >>> candidate_facets = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 2], [1, 3], [2, 3]])
    >>> # Find boundary facets (appear exactly once)
    >>> boundary_facets, _, counts = categorize_facets_by_count(
    ...     candidate_facets, target_counts="boundary"
    ... )
    >>> assert boundary_facets.shape[0] == 4  # 4 boundary edges
    """
    ### Deduplicate and count occurrences
    unique_facets, inverse_indices, counts = torch.unique(
        candidate_facets,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    ### Apply filtering based on target_counts
    if target_counts == "all":
        # Return everything, no filtering
        return unique_facets, inverse_indices, counts

    elif target_counts == "boundary":
        # Facets appearing exactly once (on boundary)
        mask = counts == 1

    elif target_counts == "interior":
        # Facets appearing exactly twice (interior of watertight mesh)
        mask = counts == 2

    elif target_counts == "shared":
        # Facets appearing 2+ times (shared by multiple cells)
        mask = counts >= 2

    elif isinstance(target_counts, list):
        # Custom list of target counts
        mask = torch.zeros_like(counts, dtype=torch.bool)
        for target_count in target_counts:
            mask |= counts == target_count

    else:
        raise ValueError(
            f"Invalid {target_counts=}. "
            f"Must be 'all', 'boundary', 'interior', 'shared', or a list of integers."
        )

    ### Filter facets and update inverse indices
    filtered_facets = unique_facets[mask]
    filtered_counts = counts[mask]

    # Update inverse indices to point to filtered facets
    # Create mapping from old unique indices to new filtered indices
    # For facets that don't pass the filter, map to -1
    old_to_new = torch.full(
        (len(unique_facets),), -1, dtype=torch.int64, device=unique_facets.device
    )
    old_to_new[mask] = torch.arange(
        mask.sum(), dtype=torch.int64, device=unique_facets.device
    )

    # Remap inverse indices
    filtered_inverse = old_to_new[inverse_indices]

    return filtered_facets, filtered_inverse, filtered_counts


def extract_candidate_facets(
    cells: torch.Tensor,  # shape: (n_cells, n_vertices_per_cell)
    manifold_codimension: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract all candidate k-codimension simplices from n-simplicial mesh.

    Each n-simplex generates C(n+1, n+1-k) candidate sub-simplices, where k is the
    manifold codimension. Sub-simplices are sorted to canonical form but may contain
    duplicates (sub-simplices shared by multiple parent cells).

    This uses vectorized PyTorch operations for high performance.

    Parameters
    ----------
    cells : torch.Tensor
        Parent mesh connectivity, shape (n_cells, n_vertices_per_cell)
    manifold_codimension : int, optional
        Codimension of the extracted mesh relative to parent.
        - 1: Extract (n-1)-facets (default, e.g., triangular faces from tets)
        - 2: Extract (n-2)-facets (e.g., edges from tets, vertices from triangles)
        - k: Extract (n-k)-facets

    Returns
    -------
    candidate_facets : torch.Tensor
        All sub-simplices with duplicates,
        shape (n_cells * n_combinations, n_vertices_per_subsimplex)
    parent_cell_indices : torch.Tensor
        Parent cell index for each sub-simplex,
        shape (n_cells * n_combinations,)

    Raises
    ------
    ValueError
        If manifold_codimension is invalid for the given cells

    Examples
    --------
    >>> import torch
    >>> # Extract edges (codim 1) from triangles
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> facets, parents = extract_candidate_facets(cells, manifold_codimension=1)
    >>> assert facets.shape == (3, 2)  # three edges with 2 vertices each

    >>> # Extract vertices (codim 2) from triangles
    >>> facets, parents = extract_candidate_facets(cells, manifold_codimension=2)
    >>> assert facets.shape == (3, 1)  # three vertices
    """
    n_cells, n_vertices_per_cell = cells.shape
    n_vertices_per_subsimplex = n_vertices_per_cell - manifold_codimension

    ### Validate codimension
    if manifold_codimension < 1:
        raise ValueError(
            f"{manifold_codimension=} must be >= 1. "
            "Use codimension=1 to extract immediate boundary facets."
        )
    if n_vertices_per_subsimplex < 1:
        raise ValueError(
            f"{manifold_codimension=} is too large for {n_vertices_per_cell=}. "
            f"Would result in {n_vertices_per_subsimplex=} < 1. "
            f"Maximum allowed codimension is {n_vertices_per_cell - 1}."
        )

    ### Generate combination indices for selecting vertices
    # Shape: (n_combinations, n_vertices_per_subsimplex)
    combination_indices = _generate_combination_indices(
        n_vertices_per_cell,
        n_vertices_per_subsimplex,
    ).to(cells.device)
    n_combinations = len(combination_indices)

    ### Extract sub-simplices using combination indices
    # Use advanced indexing to gather the correct vertex IDs
    # Shape: (n_cells, n_combinations, n_vertices_per_subsimplex)
    candidate_facets = torch.gather(
        cells.unsqueeze(1).expand(-1, n_combinations, -1),
        dim=2,
        index=combination_indices.unsqueeze(0).expand(n_cells, -1, -1),
    )

    ### Sort vertices within each sub-simplex to canonical form for deduplication
    # Shape remains (n_cells, n_combinations, n_vertices_per_subsimplex)
    candidate_facets = torch.sort(candidate_facets, dim=-1)[0]

    ### Reshape to (n_cells * n_combinations, n_vertices_per_subsimplex)
    candidate_facets = candidate_facets.reshape(-1, n_vertices_per_subsimplex)

    ### Create parent cell indices
    # Each cell contributes n_combinations sub-simplices
    # Shape: (n_cells * n_combinations,)
    parent_cell_indices = torch.arange(
        n_cells,
        device=cells.device,
        dtype=torch.int64,
    ).repeat_interleave(n_combinations)

    return candidate_facets, parent_cell_indices


def _aggregate_tensor_data(
    parent_data: torch.Tensor,  # shape: (n_parent_cells, *data_shape)
    parent_cell_indices: torch.Tensor,  # shape: (n_candidate_facets,)
    inverse_indices: torch.Tensor,  # shape: (n_candidate_facets,)
    n_unique_facets: int,
    aggregation_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Aggregate tensor data from parent cells to unique facets.

    Parameters
    ----------
    parent_data : torch.Tensor
        Data from parent cells
    parent_cell_indices : torch.Tensor
        Which parent cell each candidate facet came from
    inverse_indices : torch.Tensor
        Mapping from candidate facets to unique facets
    n_unique_facets : int
        Number of unique facets
    aggregation_weights : torch.Tensor | None
        Optional weights for aggregation

    Returns
    -------
    torch.Tensor
        Aggregated data for unique facets
    """
    from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate

    ### Gather parent cell data for each candidate facet
    # Shape: (n_candidate_facets, *data_shape)
    candidate_data = parent_data[parent_cell_indices]

    ### Use unified scatter aggregation utility
    return scatter_aggregate(
        src_data=candidate_data,
        src_to_dst_mapping=inverse_indices,
        n_dst=n_unique_facets,
        weights=aggregation_weights,
        aggregation="mean",
    )


def deduplicate_and_aggregate_facets(
    candidate_facets: torch.Tensor,  # shape: (n_candidate_facets, n_vertices_per_facet)
    parent_cell_indices: torch.Tensor,  # shape: (n_candidate_facets,)
    parent_cell_data: TensorDict,  # shape: (n_parent_cells, *data_shape)
    aggregation_weights: torch.Tensor | None = None,  # shape: (n_candidate_facets,)
) -> tuple[torch.Tensor, TensorDict, torch.Tensor]:
    """Deduplicate facets and aggregate data from parent cells.

    Finds unique facets (topologically, based on vertex indices) and aggregates
    associated data from all parent cells that share each facet.

    Parameters
    ----------
    candidate_facets : torch.Tensor
        All candidate facets including duplicates
    parent_cell_indices : torch.Tensor
        Which parent cell each candidate facet came from
    parent_cell_data : TensorDict
        TensorDict with data to aggregate from parent cells
    aggregation_weights : torch.Tensor | None, optional
        Weights for aggregating data (optional, defaults to uniform)

    Returns
    -------
    unique_facets : torch.Tensor
        Deduplicated facets, shape (n_unique_facets, n_vertices_per_facet)
    aggregated_data : TensorDict
        Aggregated TensorDict for each unique facet
    facet_to_parents : torch.Tensor
        Inverse mapping from candidate facets to unique facets, shape (n_candidate_facets,)
    """
    ### Find unique facets and inverse mapping
    unique_facets, inverse_indices = torch.unique(
        candidate_facets,
        dim=0,
        return_inverse=True,
    )

    ### Aggregate data using TensorDict.apply() (handles nested TensorDicts automatically)
    n_unique_facets = len(unique_facets)
    aggregated_data = parent_cell_data.apply(
        lambda tensor: _aggregate_tensor_data(
            tensor,
            parent_cell_indices,
            inverse_indices,
            n_unique_facets,
            aggregation_weights,
        ),
        batch_size=torch.Size([n_unique_facets]),
    )

    return unique_facets, aggregated_data, inverse_indices


def compute_aggregation_weights(
    aggregation_strategy: Literal["mean", "area_weighted", "inverse_distance"],
    parent_cell_areas: torch.Tensor | None,  # shape: (n_parent_cells,)
    parent_cell_centroids: torch.Tensor
    | None,  # shape: (n_parent_cells, n_spatial_dims)
    facet_centroids: torch.Tensor | None,  # shape: (n_candidate_facets, n_spatial_dims)
    parent_cell_indices: torch.Tensor,  # shape: (n_candidate_facets,)
) -> torch.Tensor:
    """Compute weights for aggregating parent cell data to facets.

    Parameters
    ----------
    aggregation_strategy : {"mean", "area_weighted", "inverse_distance"}
        How to weight parent contributions
    parent_cell_areas : torch.Tensor | None
        Areas of parent cells (required for area_weighted)
    parent_cell_centroids : torch.Tensor | None
        Centroids of parent cells (required for inverse_distance)
    facet_centroids : torch.Tensor | None
        Centroids of candidate facets (required for inverse_distance)
    parent_cell_indices : torch.Tensor
        Which parent cell each candidate facet came from

    Returns
    -------
    torch.Tensor
        Aggregation weights, shape (n_candidate_facets,)
    """
    n_candidate_facets = len(parent_cell_indices)
    device = parent_cell_indices.device

    if aggregation_strategy == "mean":
        return torch.ones(n_candidate_facets, device=device)

    elif aggregation_strategy == "area_weighted":
        if parent_cell_areas is None:
            raise ValueError("parent_cell_areas required for area_weighted aggregation")
        # Weight by parent cell area
        return parent_cell_areas[parent_cell_indices]

    elif aggregation_strategy == "inverse_distance":
        if parent_cell_centroids is None or facet_centroids is None:
            raise ValueError(
                "parent_cell_centroids and facet_centroids required for inverse_distance aggregation"
            )
        # Weight by inverse distance from facet centroid to parent cell centroid
        parent_centroids_for_facets = parent_cell_centroids[parent_cell_indices]
        distances = torch.norm(facet_centroids - parent_centroids_for_facets, dim=-1)
        # Avoid division by zero (facets exactly at parent centroid get high weight)
        distances = distances.clamp(min=safe_eps(distances.dtype))
        return 1.0 / distances

    else:
        raise ValueError(
            f"Invalid {aggregation_strategy=}. "
            f"Must be one of: 'mean', 'area_weighted', 'inverse_distance'"
        )


def extract_facet_mesh_data(
    parent_mesh: "Mesh",
    manifold_codimension: int = 1,
    data_source: Literal["points", "cells"] = "cells",
    data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
    target_counts: list[int] | Literal["boundary", "shared", "interior", "all"] = "all",
) -> tuple[torch.Tensor, TensorDict]:
    """Extract facet mesh data from parent mesh.

    Main entry point that orchestrates facet extraction, deduplication, and data
    aggregation. Optionally filters facets by occurrence count before aggregation.

    Parameters
    ----------
    parent_mesh : Mesh
        The parent mesh to extract facets from
    manifold_codimension : int, optional
        Codimension of extracted mesh relative to parent (default 1)
    data_source : {"points", "cells"}, optional
        Whether to inherit data from "cells" or "points"
    data_aggregation : {"mean", "area_weighted", "inverse_distance"}, optional
        How to aggregate data from multiple sources
    target_counts : list[int] | {"boundary", "shared", "interior", "all"}, optional
        Which facets to keep based on how many parent cells share them:

        - ``"all"`` (default): keep every unique facet
        - ``"boundary"``: keep facets appearing in exactly 1 cell
        - ``"interior"``: keep facets appearing in exactly 2 cells
        - ``"shared"``: keep facets appearing in 2+ cells
        - ``list[int]``: keep facets whose count is in the list

    Returns
    -------
    facet_cells : torch.Tensor
        Connectivity for facet mesh, shape (n_unique_facets, n_vertices_per_facet)
    facet_cell_data : TensorDict
        Aggregated TensorDict for facet mesh cells

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.procedural import lumpy_ball
    >>> vol_mesh = lumpy_ball.load(n_shells=2, subdivisions=1)
    >>> # Extract ALL codim-1 facets (interior + boundary)
    >>> all_facets, all_data = extract_facet_mesh_data(vol_mesh)
    >>> # Extract only the boundary surface
    >>> bnd_facets, bnd_data = extract_facet_mesh_data(
    ...     vol_mesh, target_counts="boundary"
    ... )
    >>> assert bnd_facets.shape[0] <= all_facets.shape[0]
    """
    ### Extract candidate facets from parent cells
    candidate_facets, parent_cell_indices = extract_candidate_facets(
        parent_mesh.cells,
        manifold_codimension=manifold_codimension,
    )

    ### Deduplicate, optionally filtering by occurrence count
    if target_counts == "all":
        unique_facets, inverse_indices = torch.unique(
            candidate_facets,
            dim=0,
            return_inverse=True,
        )
    else:
        unique_facets, inverse_indices, _ = categorize_facets_by_count(
            candidate_facets,
            target_counts=target_counts,
        )
        # Discard candidates that were filtered out (inverse == -1)
        keep_mask = inverse_indices >= 0
        candidate_facets = candidate_facets[keep_mask]
        parent_cell_indices = parent_cell_indices[keep_mask]
        inverse_indices = inverse_indices[keep_mask]

    n_unique_facets = len(unique_facets)

    ### Initialize empty output TensorDict
    facet_cell_data = TensorDict(
        {},
        batch_size=torch.Size([n_unique_facets]),
        device=parent_mesh.points.device,
    )

    if data_source == "cells":
        ### Aggregate data from parent cells
        filtered_cell_data = parent_mesh.cell_data
        if len(filtered_cell_data.keys()) > 0:
            ### Compute facet centroids if needed for inverse_distance
            facet_centroids = None
            if data_aggregation == "inverse_distance":
                facet_points = parent_mesh.points[candidate_facets]
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
                parent_cell_indices=parent_cell_indices,
            )

            ### Aggregate data from parent cells to unique facets
            facet_cell_data = filtered_cell_data.apply(
                lambda tensor: _aggregate_tensor_data(
                    tensor,
                    parent_cell_indices,
                    inverse_indices,
                    n_unique_facets,
                    weights,
                ),
                batch_size=torch.Size([n_unique_facets]),
            )

    elif data_source == "points":
        ### Aggregate data from facet vertices
        if len(parent_mesh.point_data.keys()) > 0:
            facet_cell_data = _aggregate_point_data_to_facets(
                point_data=parent_mesh.point_data,
                candidate_facets=candidate_facets,
                inverse_indices=inverse_indices,
                n_unique_facets=n_unique_facets,
            )

    else:
        raise ValueError(f"Invalid {data_source=}. Must be one of: 'points', 'cells'")

    return unique_facets, facet_cell_data


def _aggregate_point_data_to_facets(
    point_data: TensorDict,
    candidate_facets: torch.Tensor,
    inverse_indices: torch.Tensor,
    n_unique_facets: int,
) -> TensorDict:
    """Aggregate point data to facets by averaging over facet vertices.

    Parameters
    ----------
    point_data : TensorDict
        Data at points
    candidate_facets : torch.Tensor
        Candidate facet connectivity
    inverse_indices : torch.Tensor
        Mapping from candidate to unique facets
    n_unique_facets : int
        Number of unique facets

    Returns
    -------
    TensorDict
        Facet cell data (averaged from points)
    """

    def _aggregate_point_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Aggregate a single tensor from points to facets."""
        from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate

        ### Gather point data for vertices of each candidate facet
        # Shape: (n_candidate_facets, n_vertices_per_facet, *data_shape)
        facet_point_data = tensor[candidate_facets]

        ### Average over vertices to get candidate facet data
        # Shape: (n_candidate_facets, *data_shape)
        candidate_facet_data = facet_point_data.mean(dim=1)

        ### Aggregate to unique facets using scatter_aggregate
        return scatter_aggregate(
            src_data=candidate_facet_data,
            src_to_dst_mapping=inverse_indices,
            n_dst=n_unique_facets,
            aggregation="mean",
        )

    ### Use TensorDict.apply() to handle nested structure automatically
    return point_data.apply(
        _aggregate_point_tensor,
        batch_size=torch.Size([n_unique_facets]),
    )
