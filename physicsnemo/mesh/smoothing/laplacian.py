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

"""Laplacian mesh smoothing with feature preservation.

Implements geometry-aware smoothing using cotangent weights, with options for
preserving boundaries and sharp features.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def smooth_laplacian(
    mesh: "Mesh",
    n_iter: int = 20,
    relaxation_factor: float = 0.01,
    convergence: float = 0.0,
    feature_angle: float = 45.0,
    boundary_smoothing: bool = True,
    feature_smoothing: bool = False,
    inplace: bool = False,
) -> "Mesh":
    """Smooth mesh using Laplacian smoothing with cotangent weights.

    Applies iterative Laplacian smoothing to adjust point positions, making cells
    better shaped and vertices more evenly distributed. Uses geometry-aware
    cotangent weights that respect the mesh structure.

    Args:
        mesh: Input mesh to smooth
        n_iter: Number of smoothing iterations. More iterations produce smoother
            results but take longer. Default: 20
        relaxation_factor: Controls displacement per iteration. Lower values are
            more stable but require more iterations. Range: (0, 1]. Default: 0.01
        convergence: Convergence criterion relative to bounding box diagonal.
            Stops early if max vertex displacement < convergence * bbox_diagonal.
            Set to 0.0 to disable early stopping. Default: 0.0
        feature_angle: Angle threshold (degrees) for sharp edge detection.
            Edges with dihedral angle > feature_angle are considered sharp features.
            Only used for codimension-1 manifolds. Default: 45.0
        boundary_smoothing: If True, boundary vertices remain fixed during smoothing.
            If False, boundary vertices are smoothed like interior vertices. Default: True
        feature_smoothing: If True, vertices on sharp features remain fixed.
            If False, feature vertices are smoothed. Default: False
        inplace: If True, modifies mesh in place. If False, creates a copy. Default: False

    Returns:
        Smoothed mesh. Same object as input if inplace=True, otherwise a new mesh.

    Raises:
        ValueError: If n_iter < 0 or relaxation_factor <= 0

    Example:
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> mesh = sphere_icosahedral.load(subdivisions=2)
        >>> # Basic smoothing
        >>> smoothed = smooth_laplacian(mesh, n_iter=10, relaxation_factor=0.1)
        >>> assert smoothed.n_points == mesh.n_points
        >>> # Preserve boundaries and sharp edges
        >>> smoothed = smooth_laplacian(
        ...     mesh,
        ...     n_iter=50,
        ...     feature_angle=45.0,
        ...     boundary_smoothing=True,
        ...     feature_smoothing=True,
        ... )
        >>>
        >>> # With convergence criterion
        >>> smoothed = smooth_laplacian(
        ...     mesh,
        ...     n_iter=1000,
        ...     convergence=0.001,  # Stop if change < 0.1% of bbox
        ... )

    Note:
        - Cotangent weights are used for codimension-1 manifolds (surfaces, curves)
        - Uniform weights are used for higher codimension or volumetric meshes
        - Feature detection only works for codimension-1 manifolds where normals exist
        - Cell connectivity and all data fields are preserved (only points move)
    """
    ### Validate parameters
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter=}")
    if relaxation_factor <= 0:
        raise ValueError(f"relaxation_factor must be > 0, got {relaxation_factor=}")
    if convergence < 0:
        raise ValueError(f"convergence must be >= 0, got {convergence=}")

    ### Handle empty mesh or zero iterations
    if mesh.n_points == 0 or mesh.n_cells == 0 or n_iter == 0:
        if inplace:
            return mesh
        else:
            return mesh.clone()

    ### Create working copy if not inplace
    if not inplace:
        mesh = mesh.clone()

    device = mesh.points.device
    dtype = mesh.points.dtype
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims

    ### Extract unique edges and compute weights
    from physicsnemo.mesh.subdivision._topology import extract_unique_edges

    edges, _ = extract_unique_edges(mesh)  # (n_edges, 2)

    # Compute cotangent weights for edges
    edge_weights = _compute_edge_weights(mesh, edges)  # (n_edges,)

    ### Save original positions for constrained vertices
    original_points = mesh.points.clone()

    ### Identify constrained vertices (boundaries and features)
    constrained_vertices = torch.zeros(n_points, dtype=torch.bool, device=device)

    if boundary_smoothing:
        # Boundary vertices should not move
        boundary_vertex_mask = _get_boundary_vertices(mesh, edges)
        constrained_vertices |= boundary_vertex_mask

    if feature_smoothing:
        # Feature vertices should not move
        feature_vertex_mask = _get_feature_vertices(mesh, edges, feature_angle)
        constrained_vertices |= feature_vertex_mask

    ### Compute convergence threshold
    convergence_threshold = 0.0
    if convergence > 0:
        # Threshold relative to bounding box diagonal
        bbox_min = mesh.points.min(dim=0).values
        bbox_max = mesh.points.max(dim=0).values
        bbox_diagonal = torch.norm(bbox_max - bbox_min)
        convergence_threshold = convergence * bbox_diagonal

    ### Iterative smoothing
    for iteration in range(n_iter):
        # Save old positions for convergence check
        if convergence > 0:
            old_points = mesh.points.clone()

        ### Compute Laplacian at each vertex: L(p_i) = Σ_j w_ij (p_j - p_i)
        laplacian = torch.zeros((n_points, n_spatial_dims), dtype=dtype, device=device)
        weight_sum = torch.zeros(n_points, dtype=dtype, device=device)

        # For each edge (i, j) with weight w:
        #   laplacian[i] += w * (p_j - p_i)
        #   laplacian[j] += w * (p_i - p_j)
        #   weight_sum[i] += w
        #   weight_sum[j] += w

        # Edge vectors: p_j - p_i
        edge_vectors = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
        weighted_vectors = edge_vectors * edge_weights.unsqueeze(-1)

        # Accumulate contributions from edges
        # For vertex edges[:,0]: add weighted_vectors
        laplacian.scatter_add_(
            0,
            edges[:, 0].unsqueeze(-1).expand(-1, n_spatial_dims),
            weighted_vectors,
        )
        # For vertex edges[:,1]: subtract weighted_vectors
        laplacian.scatter_add_(
            0,
            edges[:, 1].unsqueeze(-1).expand(-1, n_spatial_dims),
            -weighted_vectors,
        )

        # Accumulate weight sums
        weight_sum.scatter_add_(0, edges[:, 0], edge_weights)
        weight_sum.scatter_add_(0, edges[:, 1], edge_weights)

        ### Normalize by total weight per vertex
        # Avoid division by zero for isolated vertices
        weight_sum = weight_sum.clamp(min=1e-10)
        laplacian = laplacian / weight_sum.unsqueeze(-1)

        ### Apply relaxation
        mesh.points = mesh.points + relaxation_factor * laplacian

        ### Restore constrained vertices to original positions
        if torch.any(constrained_vertices):
            mesh.points[constrained_vertices] = original_points[constrained_vertices]

        ### Check convergence
        if convergence > 0:
            max_displacement = torch.norm(mesh.points - old_points, dim=-1).max()
            if max_displacement < convergence_threshold:
                break

    return mesh


def _compute_edge_weights(mesh: "Mesh", edges: torch.Tensor) -> torch.Tensor:
    """Compute weights for each edge based on mesh geometry.

    For codimension-1 manifolds with n_manifold_dims >= 2: uses cotangent weights
    Otherwise: uses uniform weights

    Args:
        mesh: Input mesh
        edges: Edge connectivity, shape (n_edges, 2)

    Returns:
        Edge weights, shape (n_edges,)
    """
    n_edges = len(edges)
    device = mesh.points.device
    dtype = mesh.points.dtype

    if mesh.codimension == 1 and mesh.n_manifold_dims >= 2:
        ### Use cotangent weights (geometry-aware)
        from physicsnemo.mesh.curvature._laplacian import compute_cotangent_weights

        weights = compute_cotangent_weights(mesh, edges)

        ### Clamp weights for numerical stability
        # Negative cotangents occur for obtuse angles - treat as zero (no contribution)
        # Very large cotangents occur for nearly degenerate triangles - cap for stability
        weights = weights.clamp(min=0.0, max=10.0)

    else:
        ### Use uniform weights for 1D manifolds or higher codimension
        weights = torch.ones(n_edges, dtype=dtype, device=device)

    return weights


def _get_boundary_vertices(
    mesh: "Mesh",
    edges: torch.Tensor,
) -> torch.Tensor:
    """Identify vertices on mesh boundaries.

    Args:
        mesh: Input mesh
        edges: All unique edges, shape (n_edges, 2)

    Returns:
        Boolean mask, shape (n_points,), True for boundary vertices
    """
    device = mesh.points.device
    n_points = mesh.n_points

    # For 1D manifolds (edges), boundary detection is different
    # Boundary vertices are those that appear in only one edge
    if mesh.n_manifold_dims == 1:
        # Count edge occurrences per vertex
        vertex_edge_count = torch.zeros(n_points, dtype=torch.long, device=device)
        vertex_edge_count.scatter_add_(
            0, edges[:, 0], torch.ones(len(edges), dtype=torch.long, device=device)
        )
        vertex_edge_count.scatter_add_(
            0, edges[:, 1], torch.ones(len(edges), dtype=torch.long, device=device)
        )
        # Boundary vertices appear in only 1 edge
        boundary_mask = vertex_edge_count == 1
        return boundary_mask

    # For higher dimensional manifolds, use boundary edge detection
    from physicsnemo.mesh.boundaries import get_boundary_edges

    boundary_edges = get_boundary_edges(mesh)  # (n_boundary_edges, 2)

    if len(boundary_edges) == 0:
        # No boundaries
        return torch.zeros(n_points, dtype=torch.bool, device=device)

    # Mark all vertices in boundary edges
    boundary_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    boundary_mask[boundary_edges[:, 0]] = True
    boundary_mask[boundary_edges[:, 1]] = True

    return boundary_mask


def _get_feature_vertices(
    mesh: "Mesh",
    edges: torch.Tensor,
    feature_angle: float,
) -> torch.Tensor:
    """Identify vertices on sharp feature edges.

    Only applicable for codimension-1 manifolds where normals exist.

    Args:
        mesh: Input mesh
        edges: All unique edges, shape (n_edges, 2)
        feature_angle: Dihedral angle threshold (degrees) for sharp features

    Returns:
        Boolean mask, shape (n_points,), True for feature vertices
    """
    device = mesh.points.device
    n_points = mesh.n_points

    # Feature detection only works for codimension-1
    if mesh.codimension != 1:
        return torch.zeros(n_points, dtype=torch.bool, device=device)

    # Detect sharp edges
    sharp_edges = _detect_sharp_edges(mesh, edges, feature_angle)  # (n_sharp_edges, 2)

    if len(sharp_edges) == 0:
        return torch.zeros(n_points, dtype=torch.bool, device=device)

    # Mark all vertices in sharp edges
    feature_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    feature_mask[sharp_edges[:, 0]] = True
    feature_mask[sharp_edges[:, 1]] = True

    return feature_mask


def _detect_sharp_edges(
    mesh: "Mesh",
    edges: torch.Tensor,
    feature_angle: float,
) -> torch.Tensor:
    """Detect edges with dihedral angle exceeding threshold.

    Fully vectorized implementation using scatter operations.

    Args:
        mesh: Input mesh (must be codimension-1)
        edges: All unique edges, shape (n_edges, 2)
        feature_angle: Dihedral angle threshold in degrees

    Returns:
        Sharp edges, shape (n_sharp_edges, 2)
    """
    from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

    device = mesh.points.device
    n_manifold_dims = mesh.n_manifold_dims

    ### Extract candidate edges with parent cell info
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=n_manifold_dims - 1,
    )

    ### Map candidate edges to unique edges using hashing
    sorted_candidate_edges = torch.sort(candidate_edges, dim=1).values
    sorted_edges = torch.sort(edges, dim=1).values

    # Hash edges for fast lookup: hash = v0 * (n_points + 1) + v1
    def edge_to_hash(e: torch.Tensor) -> torch.Tensor:
        """Convert sorted edge (v0, v1) to unique hash."""
        return e[:, 0] * (mesh.n_points + 1) + e[:, 1]

    unique_edge_hashes = edge_to_hash(sorted_edges)
    candidate_edge_hashes = edge_to_hash(sorted_candidate_edges)

    # Build hash-to-index mapping for unique edges
    max_hash = unique_edge_hashes.max().item()
    edge_hash_to_idx = torch.full((max_hash + 1,), -1, dtype=torch.long, device=device)
    edge_hash_to_idx[unique_edge_hashes] = torch.arange(
        len(unique_edge_hashes), device=device
    )
    candidate_to_unique = edge_hash_to_idx[candidate_edge_hashes]

    ### Count cells per edge
    edge_cell_counts = torch.zeros(len(edges), dtype=torch.long, device=device)
    edge_cell_counts.scatter_add_(
        0,
        candidate_to_unique,
        torch.ones_like(candidate_to_unique),
    )

    ### Find interior edges (exactly 2 adjacent cells)
    interior_edge_mask = edge_cell_counts == 2

    if not torch.any(interior_edge_mask):
        return torch.empty((0, 2), dtype=edges.dtype, device=device)

    interior_edge_indices = torch.where(interior_edge_mask)[0]

    ### For each interior edge, collect its two adjacent cells (vectorized)
    # Strategy: Sort candidates by edge index, then use cumulative counting

    # Sort candidates by their unique edge index
    sorted_order = torch.argsort(candidate_to_unique)
    sorted_edge_ids = candidate_to_unique[sorted_order]

    # For each position in sorted array, compute how many times we've seen this edge before
    # This is the "occurrence index" (0 for first, 1 for second, etc.)
    # Vectorized approach: use cumsum on a binary indicator of edge boundaries

    # Mark boundaries: True where edge_id changes (first occurrence of each edge)
    edge_changes = torch.cat(
        [
            torch.tensor([True], device=device),
            sorted_edge_ids[1:] != sorted_edge_ids[:-1],
        ]
    )

    # Cumsum to get group numbers for each unique edge in the sorted array
    group_numbers = torch.cumsum(edge_changes.long(), dim=0) - 1

    # For each group, compute running index within that group
    # Start indices for each group
    group_starts = torch.where(edge_changes)[0]

    # Broadcast group_starts to all positions in that group
    group_start_for_each_pos = torch.zeros(
        len(sorted_order), dtype=torch.long, device=device
    )
    group_start_for_each_pos.scatter_(
        0, torch.arange(len(sorted_order), device=device), group_starts[group_numbers]
    )

    # Occurrence index = position - group_start
    occurrence_indices = (
        torch.arange(len(sorted_order), device=device) - group_start_for_each_pos
    )

    # Map back to original candidate order
    occurrence_in_original_order = torch.zeros(
        len(candidate_edges), dtype=torch.long, device=device
    )
    occurrence_in_original_order[sorted_order] = occurrence_indices

    # Split into first (0) and second (1) occurrences
    is_first = occurrence_in_original_order == 0
    is_second = occurrence_in_original_order == 1

    # Build edge-to-cells mapping
    edge_first_cell = torch.full((len(edges),), -1, dtype=torch.long, device=device)
    edge_second_cell = torch.full((len(edges),), -1, dtype=torch.long, device=device)

    # Use scatter to assign (will keep last value if multiple, but we expect exactly one)
    edge_first_cell.scatter_(
        0, candidate_to_unique[is_first], parent_cell_indices[is_first]
    )
    edge_second_cell.scatter_(
        0, candidate_to_unique[is_second], parent_cell_indices[is_second]
    )

    ### Compute dihedral angles for interior edges (vectorized)
    interior_first_cells = edge_first_cell[interior_edge_indices]
    interior_second_cells = edge_second_cell[interior_edge_indices]

    # Get normals for both cells
    normals_first = mesh.cell_normals[
        interior_first_cells
    ]  # (n_interior, n_spatial_dims)
    normals_second = mesh.cell_normals[
        interior_second_cells
    ]  # (n_interior, n_spatial_dims)

    # Compute angles
    cos_angles = (normals_first * normals_second).sum(dim=-1)
    cos_angles = cos_angles.clamp(-1.0, 1.0)
    angles_rad = torch.acos(cos_angles)
    angles_deg = angles_rad * 180.0 / torch.pi

    ### Filter for sharp edges
    sharp_mask = angles_deg > feature_angle
    sharp_edge_indices = interior_edge_indices[sharp_mask]

    if len(sharp_edge_indices) == 0:
        return torch.empty((0, 2), dtype=edges.dtype, device=device)

    sharp_edges = edges[sharp_edge_indices]
    return sharp_edges
