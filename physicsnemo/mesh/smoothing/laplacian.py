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

"""Laplacian smoothing for mesh geometry and point-associated fields.

Geometry smoothing supports boundary and sharp-feature preservation. Point-field
smoothing applies normalized edge averaging to every connected point.
"""

import math
from numbers import Real
from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.mesh.boundaries import get_boundary_vertices
from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets
from physicsnemo.mesh.calculus.laplacian import _apply_cotan_laplacian_operator
from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
from physicsnemo.mesh.utilities._tolerances import safe_eps
from physicsnemo.mesh.utilities._topology import extract_unique_edges

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def smooth_point_field(
    mesh: "Mesh",
    field: Float[torch.Tensor, "n_points ..."],
    n_iter: int = 1,
    relaxation_factor: float = 0.2,
) -> Float[torch.Tensor, "n_points ..."]:
    """Smooth a point field with the mesh's normalized edge Laplacian.

    The edge-weight construction matches :func:`smooth_laplacian`, but this
    function does not apply its boundary or feature-preservation masks.
    Cotangent weights are used for codimension-one manifolds of dimension two
    or greater and uniform weights are used otherwise. Unlike
    :meth:`~physicsnemo.mesh.mesh.Mesh.laplacian`, the update is normalized by
    the incident edge-weight sum, so a dimensionless ``relaxation_factor`` has
    consistent averaging semantics across mesh scales.

    Parameters
    ----------
    mesh : Mesh
        Mesh providing point adjacency and geometric edge weights.
    field : Float[torch.Tensor, "n_points ..."]
        Point-associated scalar, vector, or tensor field with leading shape
        ``(mesh.n_points, ...)``. It must share the mesh point dtype and device;
        float32 and float64 are supported.
    n_iter : int, optional
        Number of smoothing iterations. Default is ``1``.
    relaxation_factor : float, optional
        Fraction of the normalized neighbor update applied per iteration, in
        the closed interval ``[0, 1]``. Default is ``0.2``.

    Returns
    -------
    Float[torch.Tensor, "n_points ..."]
        Smoothed field with the same shape, dtype, and device as ``field``.

    Raises
    ------
    TypeError
        If ``field`` is not a tensor, its dtype differs from the mesh point
        dtype, the shared dtype is not float32 or float64, or an iteration or
        relaxation argument has an unsupported type.
    ValueError
        If the leading field dimension does not equal ``mesh.n_points``, the
        field and mesh are on different devices, ``n_iter`` is negative, or
        ``relaxation_factor`` is outside ``[0, 1]``.

    Notes
    -----
    The input tensor is not modified. Isolated points and vertices whose
    usable cotangent weights sum to zero retain their original values. The
    operation is differentiable with respect to both ``field`` and the mesh
    geometry used to construct cotangent weights.
    """
    ### Validate arguments
    if not isinstance(field, torch.Tensor):
        raise TypeError(f"field must be a torch.Tensor, got {type(field).__name__}")
    if field.ndim < 1 or field.shape[0] != mesh.n_points:
        raise ValueError(
            "field must have leading shape "
            f"({mesh.n_points}, ...), got {tuple(field.shape)}"
        )
    if field.device != mesh.points.device:
        raise ValueError(
            "field and mesh points must be on the same device, got "
            f"{field.device} and {mesh.points.device}"
        )
    if field.dtype != mesh.points.dtype:
        raise TypeError(
            "field and mesh points must have the same dtype, got "
            f"{field.dtype} and {mesh.points.dtype}"
        )
    if field.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "field and mesh points must have dtype torch.float32 or "
            f"torch.float64, got {field.dtype}"
        )
    if not isinstance(n_iter, int) or isinstance(n_iter, bool):
        raise TypeError(f"n_iter must be an integer, got {type(n_iter).__name__}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter=}")
    if not isinstance(relaxation_factor, Real) or isinstance(relaxation_factor, bool):
        raise TypeError(
            "relaxation_factor must be a finite real scalar, got "
            f"{type(relaxation_factor).__name__}"
        )
    if not math.isfinite(float(relaxation_factor)):
        raise ValueError(f"relaxation_factor must be finite, got {relaxation_factor=}")
    if not 0.0 <= float(relaxation_factor) <= 1.0:
        raise ValueError(
            f"relaxation_factor must be in [0, 1], got {relaxation_factor=}"
        )

    ### Handle meshes and settings with no smoothing work
    if (
        n_iter == 0
        or relaxation_factor == 0.0
        or mesh.n_points == 0
        or mesh.n_cells == 0
        or mesh.n_manifold_dims == 0
    ):
        return field

    ### Recompute geometric weights from the current points
    # Geometry caches may belong to an autograd graph that a caller has already
    # consumed. A cache-free view avoids reusing those stale graph tensors while
    # retaining gradients to ``mesh.points``.
    geometry_mesh = mesh.strip_caches()
    edge_weights, edges = _compute_edge_weights(geometry_mesh)

    trailing_singletons = (1,) * (field.ndim - 1)
    weight_sum = torch.zeros(mesh.n_points, dtype=field.dtype, device=field.device)
    weight_sum = weight_sum.index_add(0, edges[:, 0], edge_weights)
    weight_sum = weight_sum.index_add(0, edges[:, 1], edge_weights)
    safe_weight_sum = weight_sum.clamp_min(safe_eps(field.dtype)).reshape(
        (mesh.n_points, *trailing_singletons)
    )

    ### Apply normalized weighted-neighbor iterations
    result = field
    for _ in range(n_iter):
        laplacian = _apply_cotan_laplacian_operator(
            mesh.n_points, edges, edge_weights, result
        )
        result = result + relaxation_factor * laplacian / safe_weight_sum

    return result


def smooth_laplacian(
    mesh: "Mesh",
    n_iter: int = 20,
    relaxation_factor: float = 0.01,
    convergence: float = 0.0,
    feature_angle: float = 45.0,
    preserve_boundaries: bool = True,
    preserve_features: bool = False,
    inplace: bool = False,
) -> "Mesh":
    """Smooth mesh using Laplacian smoothing with cotangent weights.

    Applies iterative Laplacian smoothing to adjust point positions, making cells
    better shaped and vertices more evenly distributed. Uses geometry-aware
    cotangent weights that respect the mesh structure.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to smooth
    n_iter : int, optional
        Number of smoothing iterations. More iterations produce smoother
        results but take longer. Default: 20
    relaxation_factor : float, optional
        Controls displacement per iteration. Lower values are
        more stable but require more iterations. Range: (0, 1]. Default: 0.01
    convergence : float, optional
        Convergence criterion relative to bounding box diagonal.
        Stops early if max vertex displacement < convergence * bbox_diagonal.
        Set to 0.0 to disable early stopping. Default: 0.0
    feature_angle : float, optional
        Angle threshold (degrees) for sharp edge detection.
        Edges with dihedral angle > feature_angle are considered sharp features.
        Only used for codimension-1 manifolds. Default: 45.0
    preserve_boundaries : bool, optional
        If True (default), boundary vertices are fixed and
        will not move during smoothing, preserving the original boundary shape.
        If False, boundary vertices are smoothed like interior vertices.
    preserve_features : bool, optional
        If True, vertices on sharp feature edges (with dihedral
        angle > feature_angle) are fixed and will not move. If False (default),
        feature vertices are smoothed normally.
    inplace : bool, optional
        If True, modifies mesh in place. If False, creates a copy. Default: False

    Returns
    -------
    Mesh
        Smoothed mesh. Same object as input if inplace=True, otherwise a new mesh.

    Raises
    ------
    ValueError
        If n_iter < 0 or relaxation_factor <= 0

    Examples
    --------
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
    ...     preserve_boundaries=True,
    ...     preserve_features=True,
    ... )
    >>>
    >>> # With convergence criterion
    >>> smoothed = smooth_laplacian(
    ...     mesh,
    ...     n_iter=1000,
    ...     convergence=0.001,  # Stop if change < 0.1% of bbox
    ... )

    Notes
    -----
    - Cotangent weights are used for codimension-1 manifolds of dimension >= 2
    - Uniform weights are used for curves, higher codimension, or volumetric meshes
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

    ### Compute edge weights (also extracts unique edges)
    edge_weights, edges = _compute_edge_weights(mesh)  # (n_edges,), (n_edges, 2)

    ### Save original positions for constrained vertices
    original_points = mesh.points.clone()

    ### Identify constrained vertices (boundaries and features)
    constrained_vertices = torch.zeros(n_points, dtype=torch.bool, device=device)

    if preserve_boundaries:
        # Boundary vertices should not move
        boundary_vertex_mask = get_boundary_vertices(mesh)
        constrained_vertices |= boundary_vertex_mask

    if preserve_features:
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

    ### Pre-allocate buffers for iterative smoothing (avoid per-iteration allocation)
    laplacian = torch.zeros((n_points, n_spatial_dims), dtype=dtype, device=device)
    weight_sum = torch.zeros(n_points, dtype=dtype, device=device)

    ### Iterative smoothing
    for iteration in range(n_iter):
        # Save old positions for convergence check
        if convergence > 0:
            old_points = mesh.points.clone()

        ### Compute Laplacian at each vertex: L(p_i) = Σ_j w_ij (p_j - p_i)
        laplacian.zero_()
        weight_sum.zero_()

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

        ### Normalize by total weight per vertex (in place, to actually reuse the
        # pre-allocated buffers across iterations rather than reallocating each step).
        weight_sum.clamp_(min=safe_eps(dtype))
        laplacian /= weight_sum.unsqueeze(-1)

        ### Apply relaxation
        mesh.points = mesh.points + relaxation_factor * laplacian

        ### Restore constrained vertices to original positions
        # Written unconditionally to avoid a torch.compile graph break;
        # masked assignment on an all-False mask is a no-op.
        mesh.points[constrained_vertices] = original_points[constrained_vertices]

        ### Check convergence
        if convergence > 0:
            max_displacement = torch.norm(mesh.points - old_points, dim=-1).max()
            if max_displacement < convergence_threshold:
                break

    ### The loop mutated mesh.points in place, so geometry caches derived from
    ### point positions (cell_areas / cell_normals / cell_centroids, point_normals,
    ### curvature, ...) are now stale. Invalidate them so callers recompute from the
    ### smoothed geometry. Topology/adjacency caches depend only on `cells` (which
    ### smoothing does not change), so they remain valid and are preserved.
    mesh._cache["cell"] = mesh._cache["cell"].empty()
    mesh._cache["point"] = mesh._cache["point"].empty()
    return mesh


def _compute_edge_weights(
    mesh: "Mesh",
) -> tuple[Float[torch.Tensor, " n_edges"], Int[torch.Tensor, "n_edges 2"]]:
    """Compute weights for each edge based on mesh geometry.

    For codimension-1 manifolds with n_manifold_dims >= 2: uses cotangent weights
    Otherwise: uses uniform weights

    Parameters
    ----------
    mesh : Mesh
        Input mesh

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (edge_weights, edges) where edge_weights has shape (n_edges,)
        and edges has shape (n_edges, 2).
    """
    device = mesh.points.device
    dtype = mesh.points.dtype

    if mesh.codimension == 1 and mesh.n_manifold_dims >= 2:
        ### Use cotangent weights (geometry-aware) via FEM stiffness matrix
        weights, edges = compute_cotan_weights_fem(mesh)

        ### Clamp weights for numerical stability
        # Negative cotangents occur for obtuse angles - treat as zero (no contribution)
        # Very large cotangents occur for nearly degenerate triangles - cap for stability
        weights = weights.clamp(min=0.0, max=10.0)

    else:
        ### Use uniform weights for 1D manifolds or higher codimension
        edges, _ = extract_unique_edges(mesh)
        weights = torch.ones(len(edges), dtype=dtype, device=device)

    return weights, edges


def _get_feature_vertices(
    mesh: "Mesh",
    edges: Int[torch.Tensor, "n_edges 2"],
    feature_angle: float,
) -> Bool[torch.Tensor, " n_points"]:
    """Identify vertices on sharp feature edges.

    Only applicable for codimension-1 manifolds where normals exist.

    Parameters
    ----------
    mesh : Mesh
        Input mesh
    edges : torch.Tensor
        All unique edges, shape (n_edges, 2)
    feature_angle : float
        Dihedral angle threshold (degrees) for sharp features

    Returns
    -------
    torch.Tensor
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
    edges: Int[torch.Tensor, "n_edges 2"],
    feature_angle: float,
) -> Int[torch.Tensor, "n_sharp_edges 2"]:
    """Detect edges with dihedral angle exceeding threshold.

    Fully vectorized implementation using :func:`find_edges_in_reference`
    for O(n log n) edge matching and O(n) memory.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must be codimension-1)
    edges : torch.Tensor
        All unique edges, shape (n_edges, 2)
    feature_angle : float
        Dihedral angle threshold in degrees

    Returns
    -------
    torch.Tensor
        Sharp edges, shape (n_sharp_edges, 2)
    """
    from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

    device = mesh.points.device
    n_manifold_dims = mesh.n_manifold_dims

    ### Extract candidate edges with parent cell info
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=n_manifold_dims - 1,
    )

    ### Map candidate edges to unique edges via searchsorted (O(m log n), O(n) memory)
    candidate_to_unique, matched = find_edges_in_reference(
        edges,
        candidate_edges,
        index_bound=mesh.n_points,
    )

    ### Count cells per edge (only matched candidates contribute)
    edge_cell_counts = torch.bincount(
        candidate_to_unique[matched], minlength=len(edges)
    )

    ### Find interior edges (exactly 2 adjacent cells)
    interior_edge_mask = edge_cell_counts == 2

    if not torch.any(interior_edge_mask):
        return torch.empty((0, 2), dtype=edges.dtype, device=device)

    interior_edge_indices = torch.where(interior_edge_mask)[0]

    ### Collect the two adjacent cells per interior edge
    # Sort matched candidates by their unique edge index; within each group of
    # size 2, the first element becomes cell_a and the second becomes cell_b.
    valid_mask = matched
    valid_candidate_to_unique = candidate_to_unique[valid_mask]
    valid_parent_cells = parent_cell_indices[valid_mask]

    sorted_order = torch.argsort(valid_candidate_to_unique, stable=True)
    sorted_edge_ids = valid_candidate_to_unique[sorted_order]
    sorted_parent_cells = valid_parent_cells[sorted_order]

    # Detect group boundaries (where the edge index changes)
    # is_first: True at positions where the edge id differs from the predecessor
    is_first_in_group = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=device),
            sorted_edge_ids[1:] != sorted_edge_ids[:-1],
        ]
    )
    # is_second: True at positions where the edge id equals the predecessor
    is_second_in_group = torch.cat(
        [
            torch.zeros(1, dtype=torch.bool, device=device),
            sorted_edge_ids[1:] == sorted_edge_ids[:-1],
        ]
    )

    # Build per-edge cell arrays (scatter first/second occurrence to edge index)
    edge_first_cell = torch.full((len(edges),), -1, dtype=torch.long, device=device)
    edge_second_cell = torch.full((len(edges),), -1, dtype=torch.long, device=device)

    edge_first_cell.scatter_(
        0, sorted_edge_ids[is_first_in_group], sorted_parent_cells[is_first_in_group]
    )
    edge_second_cell.scatter_(
        0, sorted_edge_ids[is_second_in_group], sorted_parent_cells[is_second_in_group]
    )

    ### Compute dihedral angles for interior edges (vectorized)
    interior_first_cells = edge_first_cell[interior_edge_indices]
    interior_second_cells = edge_second_cell[interior_edge_indices]

    normals_first = mesh.cell_normals[
        interior_first_cells
    ]  # (n_interior, n_spatial_dims)
    normals_second = mesh.cell_normals[
        interior_second_cells
    ]  # (n_interior, n_spatial_dims)

    cos_angles = (normals_first * normals_second).sum(dim=-1)
    cos_angles = cos_angles.clamp(-1.0, 1.0)
    angles_deg = torch.acos(cos_angles) * (180.0 / torch.pi)

    ### Filter for sharp edges
    sharp_mask = angles_deg > feature_angle
    sharp_edge_indices = interior_edge_indices[sharp_mask]

    if len(sharp_edge_indices) == 0:
        return torch.empty((0, 2), dtype=edges.dtype, device=device)

    return edges[sharp_edge_indices]
