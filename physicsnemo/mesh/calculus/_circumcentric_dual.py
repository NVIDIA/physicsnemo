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

    # Add identity to degenerate Gram matrices to make them invertible.
    # The contribution from these cells will be zeroed by cell_volumes ~ 0.
    # Written branchlessly so torch.compile can trace through without graph breaks.
    eye = torch.eye(n_manifold_dims, dtype=dtype, device=device)
    G = G + is_degenerate.float().unsqueeze(-1).unsqueeze(-1) * eye

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
    pair_i = torch.as_tensor([p[0] for p in local_pairs], device=device)
    pair_j = torch.as_tensor([p[1] for p in local_pairs], device=device)
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


def compute_dual_volumes_1(
    mesh: "Mesh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dual 1-cell volumes (dual to edges).

    The dual 1-cell of an edge is the portion of the circumcentric dual mesh
    associated with that edge. For a 2D triangle mesh, it consists of segments
    from the edge midpoint to the circumcenters of adjacent triangles:

        |⋆e| = |e| × w_ij

    where w_ij is the FEM cotangent weight for the edge. This relationship
    holds for any manifold dimension; the FEM stiffness matrix approach
    (see :func:`compute_cotan_weights_fem`) derives these weights from the
    gradient dot products of barycentric basis functions.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh of any manifold dimension.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of ``(dual_volumes, edges)``:

        - ``dual_volumes``: Dual 1-cell volume for each edge, shape ``(n_edges,)``.
          May be negative for edges in non-Delaunay configurations (obtuse
          angles exceeding pi/2 at both adjacent cells).
        - ``edges``: Canonically sorted edge connectivity, shape ``(n_edges, 2)``,
          with ``edges[:, 0] < edges[:, 1]``.

    Notes
    -----
    Negative dual volumes are geometrically meaningful: they indicate that the
    circumcentric dual edge crosses the primal edge. Clamping them to zero (as
    some implementations do) silently degrades accuracy on non-Delaunay meshes.
    """
    ### Derive cotangent weights from the FEM stiffness matrix (works for any dimension)
    cotan_weights, edges = compute_cotan_weights_fem(mesh)

    ### |⋆e| = w_ij × |e|
    edge_vectors = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
    edge_lengths = torch.norm(edge_vectors, dim=-1)
    dual_volumes_1 = cotan_weights * edge_lengths

    return dual_volumes_1, edges


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
