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

"""Circumcentric dual mesh computation for Discrete Exterior Calculus.

This module computes circumcenters and dual cell volumes, which are essential for
the Hodge star operator in DEC. Unlike barycentric duals, circumcentric (Voronoi)
duals preserve geometric properties like orthogonality and normals.

Reference: Desbrun et al., "Discrete Exterior Calculus", Section 2
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._cache import get_cached, set_cached
from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_circumcenters(
    vertices: torch.Tensor,  # (n_simplices, n_vertices_per_simplex, n_spatial_dims)
) -> torch.Tensor:
    """Compute circumcenters of simplices using perpendicular bisector method.

    The circumcenter is the unique point equidistant from all vertices of the simplex.
    It lies at the intersection of perpendicular bisector hyperplanes.

    Parameters
    ----------
    vertices : torch.Tensor
        Vertex positions for each simplex.
        Shape: (n_simplices, n_vertices_per_simplex, n_spatial_dims)

    Returns
    -------
    torch.Tensor
        Circumcenters, shape (n_simplices, n_spatial_dims)

    Notes
    -----
    Algorithm:
        For simplex with vertices v₀, v₁, ..., vₙ, the circumcenter c satisfies:
            ||c - v₀||² = ||c - v₁||² = ... = ||c - vₙ||²

        This gives n linear equations in n_spatial_dims unknowns:
            2(v_i - v₀)·c = ||v_i||² - ||v₀||²  for i=1,...,n

        In matrix form: A·c = b where:
            A = 2[(v₁-v₀)^T, (v₂-v₀)^T, ...]^T
            b = [||v₁||²-||v₀||², ||v₂||²-||v₀||², ...]^T

        For over-determined systems (embedded manifolds), use least-squares.
    """
    n_simplices, n_vertices, n_spatial_dims = vertices.shape
    n_manifold_dims = n_vertices - 1

    ### Handle special cases
    if n_vertices == 1:
        # 0-simplex: circumcenter is the vertex itself
        return vertices.squeeze(1)

    if n_vertices == 2:
        # 1-simplex (edge): circumcenter is the midpoint
        # This avoids numerical issues with underdetermined lstsq for edges in higher dimensions
        return vertices.mean(dim=1)

    ### Build linear system for circumcenter
    # Reference vertex (first one)
    v0 = vertices[:, 0, :]  # (n_simplices, n_spatial_dims)

    # Relative vectors from v₀ to other vertices
    # Shape: (n_simplices, n_manifold_dims, n_spatial_dims)
    relative_vecs = vertices[:, 1:, :] - v0.unsqueeze(1)

    # Matrix A = 2 * relative_vecs (each row is an equation)
    # Shape: (n_simplices, n_manifold_dims, n_spatial_dims)
    A = 2 * relative_vecs

    # Right-hand side: ||v_i||² - ||v₀||²
    # Shape: (n_simplices, n_manifold_dims)
    vi_squared = (vertices[:, 1:, :] ** 2).sum(dim=-1)
    v0_squared = (v0**2).sum(dim=-1, keepdim=True)
    b = vi_squared - v0_squared

    ### Solve for circumcenter
    # Need to solve: A @ (c - v₀) = b for each simplex
    # This is: 2*(v_i - v₀) @ (c - v₀) = ||v_i||² - ||v₀||²

    if n_manifold_dims == n_spatial_dims:
        ### Square system: use direct solve
        # A is (n_simplices, n_dims, n_dims)
        # b is (n_simplices, n_dims)
        try:
            # Solve A @ x = b
            c_minus_v0 = torch.linalg.solve(
                A,  # (n_simplices, n_dims, n_dims)
                b.unsqueeze(-1),  # (n_simplices, n_dims, 1)
            ).squeeze(-1)  # (n_simplices, n_dims)
        except torch.linalg.LinAlgError:
            # Singular matrix - fall back to least squares
            c_minus_v0 = torch.linalg.lstsq(
                A,
                b.unsqueeze(-1),
            ).solution.squeeze(-1)
    else:
        ### Over-determined system (manifold embedded in higher dimension)
        # Use least-squares: (A^T A)^-1 A^T b
        # A is (n_simplices, n_manifold_dims, n_spatial_dims)
        # We need A^T @ A which is (n_simplices, n_spatial_dims, n_spatial_dims)

        # Use torch.linalg.lstsq which handles batched least-squares
        c_minus_v0 = torch.linalg.lstsq(
            A,  # (n_simplices, n_manifold_dims, n_spatial_dims)
            b.unsqueeze(-1),  # (n_simplices, n_manifold_dims, 1)
        ).solution.squeeze(-1)  # (n_simplices, n_spatial_dims)

    ### Circumcenter = v₀ + solution
    circumcenters = v0 + c_minus_v0

    return circumcenters


def compute_cotan_weights_triangle_mesh(
    mesh: "Mesh",
    edges: torch.Tensor | None = None,
    return_edges: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute cotangent Laplacian weights for edges in a mesh.

    For each edge, computes the cotangent weights using the standard formula from
    discrete differential geometry (Meyer et al. 2003, Desbrun et al. 2005).

    For 2D manifolds (triangles):
        w_ij = (1/2) × Σ cot(α) over adjacent triangles

        This gives the proper ratio |⋆e|/|e| where |⋆e| is the dual 1-cell volume
        (length of segment from edge midpoint through triangle circumcenters).

    For 3D manifolds (tets):
        Uses an inverse-edge-length approximation rather than true
        dihedral-angle cotangent weights. This is acceptable for
        well-shaped tetrahedra but degrades on slivers. Not
        implemented for manifold dimensions > 3.

    For 1D manifolds (edges):
        Uses uniform weights

    Parameters
    ----------
    mesh : Mesh
        Input mesh
    edges : torch.Tensor | None
        Edge connectivity, shape (n_edges, 2). If None, extracts edges from mesh.
    return_edges : bool
        If True, returns (weights, edges). If False, returns weights only.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        If return_edges=True: Tuple of (cotan_weights, edges)
        If return_edges=False: Just cotan_weights
        where cotan_weights has shape (n_edges,) and edges has shape (n_edges, 2)

    Notes
    -----
    Mathematical Background:
        The cotangent weight formula comes from the circumcentric dual construction in DEC.
        For an edge e shared by triangles with opposite angles α and β, the dual 1-cell
        volume is |⋆e| = (|e|/2)(cot α + cot β), giving |⋆e|/|e| = (1/2)(cot α + cot β).

        The factor of 1/2 is GEOMETRIC, arising from the distance from edge midpoints
        to triangle circumcenters. This is rigorously derived in Desbrun et al. (2005)
        "Discrete Exterior Calculus" and Meyer et al. (2003).

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> # Standard usage
    >>> weights, edges = compute_cotan_weights_triangle_mesh(mesh)
    >>> # Get weights only
    >>> weights = compute_cotan_weights_triangle_mesh(mesh, return_edges=False)
    """
    n_manifold_dims = mesh.n_manifold_dims
    device = mesh.points.device

    ### Extract edges if not provided
    if edges is None:
        from physicsnemo.mesh.boundaries._facet_extraction import extract_unique_edges

        sorted_edges, _ = extract_unique_edges(mesh)
    else:
        sorted_edges, _ = torch.sort(edges, dim=-1)

    n_edges = len(sorted_edges)

    ### Initialize weights
    cotan_weights = torch.zeros(n_edges, dtype=mesh.points.dtype, device=device)

    ### Compute weights based on manifold dimension
    if n_manifold_dims == 1:
        ### 1D: Use uniform weights (no cotangent defined)
        cotan_weights = torch.ones(n_edges, dtype=mesh.points.dtype, device=device)

    elif n_manifold_dims == 2:
        ### 2D triangles: Cotangent of opposite angles (fully vectorized)
        # Use facet extraction to get candidate edges with parent tracking
        from physicsnemo.mesh.boundaries import extract_candidate_facets

        candidate_edges, parent_cell_indices = extract_candidate_facets(
            mesh.cells,
            manifold_codimension=1,
        )

        ### For each candidate edge, compute cotangent in parent triangle
        # Shape: (n_candidates, 3)
        all_triangles = mesh.cells[parent_cell_indices]

        ### Find opposite vertices for all candidate edges
        is_v0 = all_triangles == candidate_edges[:, 0].unsqueeze(1)
        is_v1 = all_triangles == candidate_edges[:, 1].unsqueeze(1)
        opposite_mask = ~(is_v0 | is_v1)

        opposite_idx = torch.argmax(opposite_mask.int(), dim=1)
        opposite_verts = torch.gather(
            all_triangles, dim=1, index=opposite_idx.unsqueeze(1)
        ).squeeze(1)

        ### Compute cotangents for all candidates
        p_opp = mesh.points[opposite_verts]
        p_v0 = mesh.points[candidate_edges[:, 0]]
        p_v1 = mesh.points[candidate_edges[:, 1]]

        vec_to_v0 = p_v0 - p_opp
        vec_to_v1 = p_v1 - p_opp

        dot_products = (vec_to_v0 * vec_to_v1).sum(dim=-1)

        if mesh.n_spatial_dims == 2:
            cross_z = (
                vec_to_v0[:, 0] * vec_to_v1[:, 1] - vec_to_v0[:, 1] * vec_to_v1[:, 0]
            )
            cross_mag = torch.abs(cross_z)
        else:
            cross_vec = torch.linalg.cross(vec_to_v0, vec_to_v1)
            cross_mag = torch.norm(cross_vec, dim=-1)

        # Compute cotangent = dot / |cross|
        # For near-degenerate triangles (collinear vertices), cross_mag ~ 0
        # Use a relative tolerance based on edge lengths to handle this robustly
        edge_scale = torch.norm(vec_to_v0, dim=-1) * torch.norm(vec_to_v1, dim=-1)
        min_cross = edge_scale * 1e-6  # Relative tolerance for degeneracy detection
        min_cross = torch.clamp(min_cross, min=safe_eps(min_cross.dtype))  # Absolute minimum

        # For degenerate triangles (cross_mag < min_cross), set cotangent to 0
        # This effectively excludes degenerate triangles from contributing
        is_degenerate = cross_mag < min_cross
        safe_cross_mag = torch.where(is_degenerate, torch.ones_like(cross_mag), cross_mag)
        cotans = dot_products / safe_cross_mag
        cotans = torch.where(is_degenerate, torch.zeros_like(cotans), cotans)

        ### Map candidate edges to sorted_edges and accumulate (vectorized)
        from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

        indices_in_original, valid_matches = find_edges_in_reference(
            sorted_edges, candidate_edges
        )

        # Only accumulate cotangents for edges that actually matched
        valid_cotans = torch.where(valid_matches, cotans, torch.zeros_like(cotans))
        cotan_weights.scatter_add_(0, indices_in_original, valid_cotans)

        ### Apply the REQUIRED factor of 1/2 from the geometric derivation
        # |⋆e|/|e| = (1/2) × Σ cot(opposite angles)
        cotan_weights = cotan_weights / 2.0

    elif n_manifold_dims == 3:
        ### 3D tetrahedra: Geometric approximation (inverse edge length weighting)
        # Full dihedral angle cotangents would require complex face-based structures
        # For now use simplified formula (divide by 2 for consistency with 2D case)
        edge_vectors = mesh.points[sorted_edges[:, 1]] - mesh.points[sorted_edges[:, 0]]
        edge_lengths = torch.norm(edge_vectors, dim=-1)
        cotan_weights = (1.0 / edge_lengths.clamp(min=safe_eps(edge_lengths.dtype))) / 2.0

    else:
        raise NotImplementedError(
            f"Cotangent weights not implemented for {n_manifold_dims=}."
        )

    ### Return based on return_edges flag
    if return_edges:
        return cotan_weights, sorted_edges
    else:
        return cotan_weights


def compute_cotan_weights_fem(
    mesh: "Mesh",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute cotangent weights for all edges using the FEM stiffness matrix.

    This is the dimension-general approach that works for simplicial meshes of
    any manifold dimension (1D edges, 2D triangles, 3D tetrahedra, etc.). It
    derives the cotangent weights from the Finite Element Method (FEM) stiffness
    matrix with piecewise-linear basis functions.

    For an n-simplex with vertices v_0, ..., v_n and barycentric coordinate
    functions lambda_i, the stiffness matrix entry for edge (i, j) is:

        K_ij = |sigma| * (grad lambda_i . grad lambda_j)

    The cotangent weight is w_ij = -K_ij, accumulated over all cells sharing
    the edge. This is mathematically equivalent to the classical cotangent
    formula in 2D: w_ij = (1/2)(cot alpha + cot beta).

    The gradient dot products are computed efficiently via the Gram matrix:

        E = [v_1 - v_0, ..., v_n - v_0]  (n x d edge matrix)
        G = E @ E^T                        (n x n Gram matrix)
        grad lambda_k . grad lambda_l = (G^{-1})_{k-1, l-1}   for k, l >= 1

    For pairs involving vertex 0, the constraint sum(grad lambda_i) = 0 is used.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh of any manifold dimension.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (cotan_weights, unique_edges):
        - cotan_weights: Cotangent weight for each unique edge, shape (n_edges,)
        - unique_edges: Sorted edge vertex indices, shape (n_edges, 2)

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> weights, edges = compute_cotan_weights_fem(mesh)
    >>> # weights[i] is the cotangent weight for edges[i]
    """
    from itertools import combinations

    from physicsnemo.mesh.boundaries._facet_extraction import extract_unique_edges

    device = mesh.points.device
    dtype = mesh.points.dtype
    n_cells = mesh.n_cells
    n_manifold_dims = mesh.n_manifold_dims
    n_verts_per_cell = n_manifold_dims + 1  # n+1 vertices in an n-simplex

    ### Extract unique edges and the inverse mapping from candidate edges
    unique_edges, inverse_indices = extract_unique_edges(mesh)
    n_unique_edges = len(unique_edges)

    ### Handle empty mesh
    if n_cells == 0:
        return (
            torch.zeros(n_unique_edges, dtype=dtype, device=device),
            unique_edges,
        )

    ### Compute edge vectors from reference vertex (vertex 0 of each cell)
    # cell_vertices: (n_cells, n_verts_per_cell, n_spatial_dims)
    cell_vertices = mesh.points[mesh.cells]
    # E: (n_cells, n_manifold_dims, n_spatial_dims) - rows are e_k = v_k - v_0
    E = cell_vertices[:, 1:, :] - cell_vertices[:, [0], :]

    ### Compute Gram matrix G = E @ E^T
    # G: (n_cells, n_manifold_dims, n_manifold_dims)
    G = E @ E.transpose(-1, -2)

    ### Handle degenerate cells by regularizing singular Gram matrices
    # Degenerate cells (collinear/coplanar vertices) have det(G) ~ 0.
    # We regularize these so that torch.linalg.inv doesn't produce NaN,
    # then zero out their contributions via the cell volume (which is also ~0).
    det_G = torch.linalg.det(G)  # (n_cells,)
    # Scale-aware degeneracy threshold: compare det against typical edge length
    # raised to the 2n power (since det(G) has units of length^{2n})
    edge_length_scale = E.norm(dim=-1).mean(dim=-1).clamp(min=1e-30)  # (n_cells,)
    det_threshold = (edge_length_scale ** (2 * n_manifold_dims)) * 1e-12
    is_degenerate = det_G.abs() < det_threshold  # (n_cells,)

    if is_degenerate.any():
        # Add identity to make degenerate Gram matrices invertible.
        # The contribution from these cells will be zeroed by cell_volumes ~ 0.
        eye = torch.eye(n_manifold_dims, dtype=dtype, device=device)
        G = G.clone()
        G[is_degenerate] += eye

    ### Invert Gram matrix
    # G_inv: (n_cells, n_manifold_dims, n_manifold_dims)
    G_inv = torch.linalg.inv(G)

    ### Build the gradient dot product matrix C = H @ G_inv @ H^T
    # H: (n_verts_per_cell, n_manifold_dims) = [[-1,...,-1]; I_n]
    # This encodes the relationship: grad lambda_0 = -sum(grad lambda_k for k>=1)
    H = torch.zeros(n_verts_per_cell, n_manifold_dims, dtype=dtype, device=device)
    H[0, :] = -1.0
    H[1:, :] = torch.eye(n_manifold_dims, dtype=dtype, device=device)

    # C: (n_cells, n_verts_per_cell, n_verts_per_cell)
    # C[c, i, j] = grad lambda_i . grad lambda_j in cell c
    C = H.unsqueeze(0) @ G_inv @ H.T.unsqueeze(0)

    ### Extract gradient dot products for each local edge pair
    # Local edge pairs in combinations order (matches extract_candidate_facets)
    local_pairs = list(combinations(range(n_verts_per_cell), 2))
    pair_i = torch.tensor([p[0] for p in local_pairs], device=device)
    pair_j = torch.tensor([p[1] for p in local_pairs], device=device)
    n_pairs = len(local_pairs)

    # grad_dots: (n_cells, n_pairs) - one value per cell per local edge
    grad_dots = C[:, pair_i, pair_j]

    ### Compute cotangent weight contributions per cell per edge
    # w = -|sigma| * (grad lambda_i . grad lambda_j)
    cell_volumes = mesh.cell_areas  # (n_cells,)
    weights_per_cell = -cell_volumes[:, None] * grad_dots  # (n_cells, n_pairs)

    ### Accumulate contributions to unique edges via scatter_add
    cotan_weights = torch.zeros(n_unique_edges, dtype=dtype, device=device)
    # inverse_indices maps each candidate edge to its unique edge index.
    # For 1D: shape (n_cells,); for nD>1: shape (n_cells * n_pairs,)
    # weights_per_cell.reshape(-1) aligns with inverse_indices in both cases.
    cotan_weights.scatter_add_(0, inverse_indices, weights_per_cell.reshape(-1))

    return cotan_weights, unique_edges


def compute_dual_volumes_1(mesh: "Mesh") -> torch.Tensor:
    """Compute dual 1-cell volumes (dual to edges).

    For triangle meshes, uses the circumcentric dual construction from DEC.
    The dual 1-cell for an edge consists of segments from the edge midpoint
    to the circumcenters of adjacent triangles.

    For an edge shared by triangles with opposite angles α and β:
        |⋆e| = (|e|/2)(cot α + cot β) = |e| × w_ij
    where w_ij are the cotangent weights.

    For boundary edges (shared by only one triangle), the dual volume is half
    of an interior edge with the same geometry, since only one triangle contributes.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh

    Returns
    -------
    torch.Tensor
        Dual 1-cell volumes for each edge, shape (n_edges,)

    Notes
    -----
    Dual volumes are guaranteed to be non-negative. For degenerate or
    near-degenerate triangles, volumes may be zero or very small.
    """
    if mesh.n_manifold_dims == 2:
        ### Use cotangent weights for triangles
        # The cotangent weights already encode the ratio |⋆e|/|e|
        # So to get |⋆e|, we multiply by |e|
        cotan_weights, edges = compute_cotan_weights_triangle_mesh(mesh)
        edge_lengths = torch.norm(
            mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]],
            dim=-1,
        )

        # |⋆e| = |e| × (|⋆e|/|e|) = |e| × w_ij
        # where w_ij = (1/2)(cot α + cot β) is the cotangent weight
        dual_volumes_1 = cotan_weights * edge_lengths

        # Ensure non-negative values (cotangent can be negative for obtuse angles,
        # but the sum over adjacent triangles should be positive for valid meshes)
        # Clamp to zero as a safety measure for numerical edge cases
        dual_volumes_1 = torch.clamp(dual_volumes_1, min=0.0)

    else:
        ### For other dimensions, use simplified approximation
        edge_mesh = mesh.get_facet_mesh(manifold_codimension=1)
        edges = edge_mesh.cells
        sorted_edges, _ = torch.sort(edges, dim=-1)

        edge_lengths = torch.norm(
            mesh.points[sorted_edges[:, 1]] - mesh.points[sorted_edges[:, 0]],
            dim=-1,
        )
        dual_volumes_1 = edge_lengths

    return dual_volumes_1


def get_or_compute_dual_volumes_0(mesh: "Mesh") -> torch.Tensor:
    """Get cached dual 0-cell volumes or compute if not present.

    Parameters
    ----------
    mesh : Mesh
        Input mesh

    Returns
    -------
    torch.Tensor
        Dual volumes for vertices, shape (n_points,)
    """
    from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0

    cached = get_cached(mesh.point_data, "dual_volumes_0")
    if cached is None:
        cached = compute_dual_volumes_0(mesh)
        set_cached(mesh.point_data, "dual_volumes_0", cached)
    return cached


def get_or_compute_circumcenters(mesh: "Mesh") -> torch.Tensor:
    """Get cached circumcenters or compute if not present.

    Parameters
    ----------
    mesh : Mesh
        Input mesh

    Returns
    -------
    torch.Tensor
        Circumcenters for all cells, shape (n_cells, n_spatial_dims)
    """
    cached = get_cached(mesh.cell_data, "circumcenters")
    if cached is None:
        parent_cell_vertices = mesh.points[mesh.cells]
        cached = compute_circumcenters(parent_cell_vertices)
        set_cached(mesh.cell_data, "circumcenters", cached)
    return cached
