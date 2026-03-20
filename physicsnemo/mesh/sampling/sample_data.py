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

"""Spatial sampling of data at query points in a mesh.

All containment queries use BVH-accelerated O(M*log(N)) search. A ``BVH``
can be supplied to amortise construction cost across repeated calls; if
omitted, one is built automatically.
"""

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float, Int
from tensordict import TensorDict

from physicsnemo.mesh.neighbors._adjacency import Adjacency, build_adjacency_from_pairs
from physicsnemo.mesh.spatial import BVH
from physicsnemo.nn.functional.knn import knn

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_bvh(mesh: "Mesh", bvh: BVH | None) -> BVH:
    """Return the given BVH, or build one from *mesh* if ``None``."""
    if bvh is not None:
        return bvh
    return BVH.from_mesh(mesh)


# ---------------------------------------------------------------------------
# Barycentric coordinate solvers
# ---------------------------------------------------------------------------


def _solve_barycentric_system(
    relative_vectors: Float[torch.Tensor, "*batch n_manifold_dims n_spatial_dims"],
    query_relative: Float[torch.Tensor, "*batch n_spatial_dims"],
) -> tuple[
    Float[torch.Tensor, "*batch n_vertices_per_cell"],
    Float[torch.Tensor, " *batch"],
]:
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
        - barycentric_coords: shape (..., n_vertices_per_cell)
            where n_vertices_per_cell = n_manifold_dims + 1
        - reconstruction_error: L2 distance from query point to its projection
            onto the simplex's affine hull, shape (...). Zero for codimension-0.

    Notes
    -----
    For square systems (n_spatial_dims == n_manifold_dims): uses direct solve.
    For over/under-determined systems: uses least squares.
    """
    n_manifold_dims = relative_vectors.shape[-2]
    n_spatial_dims = relative_vectors.shape[-1]

    A = relative_vectors.transpose(-2, -1)
    b = query_relative.unsqueeze(-1)

    if n_spatial_dims == n_manifold_dims:
        ### Square system: use torch.linalg.solve
        try:
            weights_1_to_n = torch.linalg.solve(A, b).squeeze(-1)
        except torch.linalg.LinAlgError:
            weights_1_to_n = torch.linalg.lstsq(A, b).solution.squeeze(-1)

        reconstruction_error = torch.zeros(
            weights_1_to_n.shape[:-1],
            dtype=query_relative.dtype,
            device=query_relative.device,
        )

    else:
        ### Over-determined or under-determined system: use least squares
        weights_1_to_n = torch.linalg.lstsq(A, b).solution.squeeze(-1)

        reconstructed = torch.einsum(
            "...m,...ms->...s", weights_1_to_n, relative_vectors
        )
        residual = query_relative - reconstructed
        reconstruction_error = torch.linalg.vector_norm(residual, dim=-1)

    ### w_0 = 1 - sum(w_i for i=1..n)
    w_0 = 1.0 - weights_1_to_n.sum(dim=-1, keepdim=True)
    barycentric_coords = torch.cat([w_0, weights_1_to_n], dim=-1)

    return barycentric_coords, reconstruction_error


def compute_barycentric_coordinates(
    query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
    cell_vertices: Float[torch.Tensor, "n_cells n_vertices_per_cell n_spatial_dims"],
) -> tuple[
    Float[torch.Tensor, "n_queries n_cells n_vertices_per_cell"],
    Float[torch.Tensor, "n_queries n_cells"],
]:
    """Compute barycentric coordinates of query points with respect to simplices.

    Computes the full O(n_queries x n_cells) cartesian product. For BVH-pruned
    candidate pairs, use :func:`compute_barycentric_coordinates_pairwise` instead.

    Parameters
    ----------
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims)
    cell_vertices : torch.Tensor
        Vertices of cells, shape (n_cells, n_vertices_per_cell, n_spatial_dims)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (barycentric_coords, reconstruction_error):
        - barycentric_coords: shape (n_queries, n_cells, n_vertices_per_cell)
        - reconstruction_error: shape (n_queries, n_cells). Zero for codimension-0.
    """
    v0 = cell_vertices[:, 0:1, :]  # (n_cells, 1, n_spatial_dims)
    relative_vectors = cell_vertices[:, 1:, :] - v0
    query_relative = query_points.unsqueeze(1) - v0.squeeze(1).unsqueeze(0)
    relative_vectors_expanded = relative_vectors.unsqueeze(0)

    return _solve_barycentric_system(relative_vectors_expanded, query_relative)


def compute_barycentric_coordinates_pairwise(
    query_points: Float[torch.Tensor, "n_pairs n_spatial_dims"],
    cell_vertices: Float[torch.Tensor, "n_pairs n_vertices_per_cell n_spatial_dims"],
) -> tuple[
    Float[torch.Tensor, "n_pairs n_vertices_per_cell"],
    Float[torch.Tensor, " n_pairs"],
]:
    """Compute barycentric coordinates for paired queries and cells.

    Unlike :func:`compute_barycentric_coordinates` which computes all
    O(n_queries x n_cells) combinations, this computes only n_pairs diagonal
    elements where each query is paired with exactly one cell. O(n) memory.

    Parameters
    ----------
    query_points : torch.Tensor
        Query point locations, shape (n_pairs, n_spatial_dims)
    cell_vertices : torch.Tensor
        Vertices of cells, shape (n_pairs, n_vertices_per_cell, n_spatial_dims).
        ``cell_vertices[i]`` is paired with ``query_points[i]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (barycentric_coords, reconstruction_error):
        - barycentric_coords: shape (n_pairs, n_vertices_per_cell)
        - reconstruction_error: shape (n_pairs,). Zero for codimension-0.

    Examples
    --------
    >>> import torch
    >>> n_pairs = 1000
    >>> query_points = torch.randn(n_pairs, 3)
    >>> cell_vertices = torch.randn(n_pairs, 3, 3)  # Triangles in 3D
    >>> bary, err = compute_barycentric_coordinates_pairwise(query_points, cell_vertices)
    >>> assert bary.shape == (1000, 3)
    """
    v0 = cell_vertices[:, 0, :]  # (n_pairs, n_spatial_dims)
    relative_vectors = cell_vertices[:, 1:, :] - v0.unsqueeze(1)
    query_relative = query_points - v0

    return _solve_barycentric_system(relative_vectors, query_relative)


# ---------------------------------------------------------------------------
# Containment queries
# ---------------------------------------------------------------------------


def _find_containing_pairs(
    mesh: "Mesh",
    query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
    bvh: BVH,
    tolerance: float,
) -> tuple[
    Int[torch.Tensor, " n_containing"],
    Int[torch.Tensor, " n_containing"],
    Float[torch.Tensor, "n_containing n_vertices_per_cell"] | None,
]:
    """Find (query_idx, cell_idx, bary_coords) via BVH-accelerated search.

    Parameters
    ----------
    mesh : Mesh
        Source mesh.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims).
    bvh : BVH
        Bounding Volume Hierarchy for the mesh.
    tolerance : float
        Containment tolerance.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        (query_indices, cell_indices, bary_coords):
        - query_indices: shape (n_containing,)
        - cell_indices: shape (n_containing,)
        - bary_coords: shape (n_containing, n_vertices_per_cell) or None if empty
    """
    device = mesh.points.device

    ### Get candidate pairs from BVH (AABB overlap test)
    candidate_adj = bvh.find_candidate_cells(query_points, aabb_tolerance=tolerance)

    if candidate_adj.n_total_neighbors == 0:
        return (
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
            None,
        )

    query_idx_cand, cell_idx_cand = candidate_adj.expand_to_pairs()

    ### Refine candidates with exact barycentric test
    cand_query_pts = query_points[query_idx_cand]
    cand_cell_verts = mesh.points[mesh.cells[cell_idx_cand]]

    bary_cand, recon_cand = compute_barycentric_coordinates_pairwise(
        cand_query_pts, cand_cell_verts
    )

    is_inside = (bary_cand >= -tolerance).all(dim=-1) & (recon_cand <= tolerance)

    ### Filter to confirmed containments
    query_indices = query_idx_cand[is_inside]
    cell_indices = cell_idx_cand[is_inside]
    bary_coords = bary_cand[is_inside] if len(query_indices) > 0 else None

    return query_indices, cell_indices, bary_coords


def find_containing_cells(
    mesh: "Mesh",
    query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
    tolerance: float = 1e-6,
    bvh: BVH | None = None,
) -> tuple[
    Int[torch.Tensor, " n_queries"],
    Float[torch.Tensor, "n_queries n_vertices_per_cell"],
]:
    """Find which cell contains each query point (first match).

    Parameters
    ----------
    mesh : Mesh
        The mesh to query.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims).
    tolerance : float
        Tolerance for considering a point inside a cell.
    bvh : BVH or None, optional
        Pre-built BVH. Auto-built from *mesh* if ``None``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (cell_indices, barycentric_coords):
        - cell_indices: shape (n_queries,). Value is -1 if no cell contains
          the point, otherwise the first containing cell index.
        - barycentric_coords: shape (n_queries, n_vertices_per_cell).
          NaN if no containing cell.

    Notes
    -----
    If multiple cells contain a point, only the first is returned.
    Use :func:`find_all_containing_cells` to get all containing cells.
    """
    n_queries = query_points.shape[0]
    n_verts = mesh.n_manifold_dims + 1
    device = mesh.points.device
    bvh = _ensure_bvh(mesh, bvh)

    query_idx, cell_idx, bary = _find_containing_pairs(
        mesh, query_points, bvh, tolerance
    )

    ### Initialise outputs (default: not found)
    cell_indices = torch.full((n_queries,), -1, dtype=torch.long, device=device)
    result_bary = torch.full(
        (n_queries, n_verts), float("nan"), dtype=query_points.dtype, device=device
    )

    if len(query_idx) == 0:
        return cell_indices, result_bary

    ### For each query, keep only the first containing cell
    is_first = torch.cat(
        [
            torch.tensor([True], device=device),
            query_idx[1:] != query_idx[:-1],
        ]
    )
    first_pos = torch.where(is_first)[0]
    hit_queries = query_idx[first_pos]
    hit_cells = cell_idx[first_pos]

    cell_indices[hit_queries] = hit_cells
    if bary is not None:
        result_bary[hit_queries] = bary[first_pos]

    return cell_indices, result_bary


def find_all_containing_cells(
    mesh: "Mesh",
    query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
    tolerance: float = 1e-6,
    bvh: BVH | None = None,
) -> Adjacency:
    """Find all cells that contain each query point.

    Parameters
    ----------
    mesh : Mesh
        The mesh to query.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims).
    tolerance : float
        Tolerance for considering a point inside a cell.
    bvh : BVH or None, optional
        Pre-built BVH. Auto-built from *mesh* if ``None``.

    Returns
    -------
    Adjacency
        Adjacency where containing cells for query *i* are at
        ``result.indices[result.offsets[i]:result.offsets[i+1]]``.
    """
    bvh = _ensure_bvh(mesh, bvh)
    query_indices, cell_indices, _ = _find_containing_pairs(
        mesh, query_points, bvh, tolerance
    )

    return build_adjacency_from_pairs(
        source_indices=query_indices,
        target_indices=cell_indices,
        n_sources=len(query_points),
    )


# ---------------------------------------------------------------------------
# Nearest-cell helpers
# ---------------------------------------------------------------------------


def find_nearest_cells(
    mesh: "Mesh",
    query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
) -> tuple[
    Int[torch.Tensor, " n_queries"],
    Float[torch.Tensor, "n_queries n_spatial_dims"],
]:
    """Find the nearest cell for each query point (by centroid distance).

    Uses :func:`~physicsnemo.nn.functional.knn.knn` which auto-dispatches
    to the optimal backend (cuML on GPU, scipy KDTree on CPU).

    Parameters
    ----------
    mesh : Mesh
        The mesh to query.
    query_points : torch.Tensor
        Query point locations, shape ``(n_queries, n_spatial_dims)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(cell_indices, projected_points)``:
        - cell_indices: shape ``(n_queries,)``
        - projected_points: centroids of nearest cells, shape
          ``(n_queries, n_spatial_dims)``
    """
    cell_centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)
    cell_indices, _ = knn(cell_centroids, query_points, k=1)
    cell_indices = cell_indices.squeeze(1)
    projected_points = cell_centroids[cell_indices]
    return cell_indices, projected_points


# ---------------------------------------------------------------------------
# Shared accumulation logic
# ---------------------------------------------------------------------------


def _accumulate_sampled_data(
    mesh: "Mesh",
    n_queries: int,
    query_indices: Int[torch.Tensor, " n_containing"],
    cell_indices: Int[torch.Tensor, " n_containing"],
    bary_coords: Float[torch.Tensor, "n_containing n_vertices_per_cell"] | None,
    data_source: Literal["cells", "points"],
    multiple_cells_strategy: Literal["mean", "nan"],
) -> TensorDict:
    """Accumulate sampled data from containing-pair arrays into a TensorDict.

    This is the shared kernel that handles scalar/multidimensional data,
    mean/nan strategies, and cell/point data sources.
    """
    device = mesh.points.device

    ### Count how many cells contain each query point
    query_containment_count = torch.zeros(n_queries, dtype=torch.long, device=device)
    if len(query_indices) > 0:
        query_containment_count.scatter_add_(
            0, query_indices, torch.ones_like(query_indices)
        )

    source_data = mesh.cell_data if data_source == "cells" else mesh.point_data
    cells = mesh.cells  # captured for point-data interpolation below

    def _accumulate_field(values: torch.Tensor) -> torch.Tensor:
        """Scatter-accumulate a single data field across query points."""
        output_shape = (n_queries,) + values.shape[1:]
        output = torch.full(
            output_shape, float("nan"), dtype=values.dtype, device=device
        )

        if len(query_indices) == 0:
            return output

        ### Compute per-pair values
        if data_source == "cells":
            pair_values = values[cell_indices]
        elif data_source == "points":
            if (
                bary_coords is None
            ):  # pragma: no cover — guaranteed when len(query_indices) > 0
                raise RuntimeError(
                    "bary_coords is unexpectedly None for non-empty query set."
                )
            point_idx = cells[cell_indices]
            point_vals = values[point_idx]

            bary_expanded = bary_coords.view(
                bary_coords.shape[0],
                bary_coords.shape[1],
                *([1] * (values.ndim - 1)),
            )
            pair_values = (bary_expanded * point_vals).sum(dim=1)
        else:
            raise ValueError(f"Invalid {data_source=!r}. Must be 'cells' or 'points'.")

        ### Scatter-accumulate into output
        if multiple_cells_strategy == "mean":
            output_sum = torch.zeros(output_shape, dtype=values.dtype, device=device)
            idx_expanded = query_indices.view(-1, *([1] * (values.ndim - 1))).expand_as(
                pair_values
            )
            output_sum.scatter_add_(0, idx_expanded, pair_values)

            valid = query_containment_count > 0
            output[valid] = output_sum[valid] / query_containment_count[valid].to(
                values.dtype
            ).view(-1, *([1] * (values.ndim - 1)))

        elif multiple_cells_strategy == "nan":
            single_cell_mask = query_containment_count == 1
            if single_cell_mask.any():
                has_single = single_cell_mask[query_indices]
                output[query_indices[has_single]] = pair_values[has_single]
        else:
            raise ValueError(
                f"Invalid {multiple_cells_strategy=!r}. Must be 'mean' or 'nan'."
            )

        return output

    # apply() always returns a TensorDict here (our fn never returns None),
    # but the generic return type is TensorDict | None.
    result = source_data.apply(
        _accumulate_field,
        batch_size=torch.Size([n_queries]),
    )
    if not isinstance(
        result, TensorDict
    ):  # pragma: no cover — apply() returns TensorDict here
        raise TypeError(f"Expected TensorDict from apply(), got {type(result)}")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_data_at_points(
    mesh: "Mesh",
    query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
    data_source: Literal["cells", "points"] = "cells",
    multiple_cells_strategy: Literal["mean", "nan"] = "mean",
    project_onto_nearest_cell: bool = False,
    tolerance: float = 1e-6,
    bvh: BVH | None = None,
) -> TensorDict:
    """Extract or interpolate mesh data at specified query points.

    For each query point, the function:

    1. Finds which cell(s) contain the point (BVH-accelerated, O(log N))
    2. Extracts cell data directly (``data_source="cells"``) or interpolates
       point data using barycentric coordinates (``data_source="points"``)

    Parameters
    ----------
    mesh : Mesh
        The mesh to extract data from.
    query_points : torch.Tensor
        Query point locations, shape (n_queries, n_spatial_dims).
    data_source : {"cells", "points"}, optional
        - "cells": Use cell data directly (no interpolation).
        - "points": Interpolate point data using barycentric coordinates.
    multiple_cells_strategy : {"mean", "nan"}, optional
        How to handle query points contained in multiple cells:
        - "mean": Return arithmetic mean of values from all containing cells.
        - "nan": Return NaN for ambiguous points.
    project_onto_nearest_cell : bool, optional
        If True, snaps each query point to the centroid of the nearest cell
        before performing containment testing. Useful for codimension != 0
        manifolds where exact on-surface points are hard to construct.
    tolerance : float, optional
        Tolerance for considering a point inside a cell. A point is inside if
        all barycentric coordinates >= -tolerance AND reconstruction error
        <= tolerance.
    bvh : BVH or None, optional
        Pre-built Bounding Volume Hierarchy. If ``None`` (default), one is
        built automatically. For repeated queries on the same mesh, pre-build
        with ``BVH.from_mesh(mesh)`` and pass it here to avoid redundant work.

    Returns
    -------
    TensorDict
        Sampled data for each query point, with the same keys as
        ``mesh.cell_data`` or ``mesh.point_data`` (depending on *data_source*).
        Values are NaN for query points outside the mesh.

    Raises
    ------
    ValueError
        If *data_source* or *multiple_cells_strategy* is invalid.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> mesh.cell_data["pressure"] = torch.tensor([1.0, 2.0])
    >>> query_pts = torch.tensor([[0.3, 0.3], [0.8, 0.5]])
    >>> sampled = sample_data_at_points(mesh, query_pts, data_source="cells")
    >>> assert "pressure" in sampled.keys()

    Pre-build a BVH for repeated queries:

    >>> from physicsnemo.mesh.spatial import BVH
    >>> bvh = BVH.from_mesh(mesh)
    >>> sampled = sample_data_at_points(mesh, query_pts, bvh=bvh)
    """
    if data_source not in ("cells", "points"):
        raise ValueError(f"Invalid {data_source=}. Must be 'cells' or 'points'.")
    if multiple_cells_strategy not in ("mean", "nan"):
        raise ValueError(
            f"Invalid {multiple_cells_strategy=}. Must be 'mean' or 'nan'."
        )

    n_queries = query_points.shape[0]

    ### Ensure BVH is available (shared across projection and containment)
    bvh = _ensure_bvh(mesh, bvh)

    ### Handle projection onto nearest cell
    if project_onto_nearest_cell:
        _, projected_points = find_nearest_cells(mesh, query_points)
        query_points = projected_points
    query_indices, cell_indices, bary_coords = _find_containing_pairs(
        mesh, query_points, bvh, tolerance
    )

    ### Accumulate sampled data
    return _accumulate_sampled_data(
        mesh=mesh,
        n_queries=n_queries,
        query_indices=query_indices,
        cell_indices=cell_indices,
        bary_coords=bary_coords,
        data_source=data_source,
        multiple_cells_strategy=multiple_cells_strategy,
    )
