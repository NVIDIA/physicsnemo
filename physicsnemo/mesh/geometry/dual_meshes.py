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

"""Dual mesh (circumcentric/Voronoi) volume computation and DEC dual operators.

This module provides the unified implementation of dual cell volumes (Voronoi regions),
circumcenters, and cotangent weights for n-dimensional simplicial meshes. These are
fundamental to both:
- Discrete Exterior Calculus (DEC) operators (Hodge star, Laplacian, etc.)
- Discrete differential geometry (curvature computations)

Dual 0-cell volumes follow Meyer et al. (2003) for 2D manifolds, using the mixed
Voronoi area approach that handles both acute and obtuse triangles correctly.
For higher dimensions, barycentric approximation is used as rigorous circumcentric
dual volumes require well-centered meshes (Desbrun et al. 2005, Hirani 2003).

Circumcenters and cotangent weights are computed using the perpendicular bisector
method and FEM stiffness matrix approach, respectively, following Desbrun et al.
"Discrete Exterior Calculus", Section 2.

References:
    Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
    "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds". VisMath.

    Desbrun, M., Hirani, A. N., Leok, M., & Marsden, J. E. (2005).
    "Discrete Exterior Calculus". arXiv:math/0508341.

    Hirani, A. N. (2003). "Discrete Exterior Calculus". PhD thesis, Caltech.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._cache import get_cached, set_cached
from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _scatter_add_cell_contributions_to_vertices(
    dual_volumes: torch.Tensor,  # shape: (n_points,)
    cells: torch.Tensor,  # shape: (n_selected_cells, n_vertices_per_cell)
    contributions: torch.Tensor,  # shape: (n_selected_cells,)
) -> None:
    """Scatter cell volume contributions to all cell vertices.

    This is a common pattern in dual volume computation where each cell
    contributes a fraction of its volume to each of its vertices.

    Parameters
    ----------
    dual_volumes : torch.Tensor
        Accumulator for dual volumes (modified in place)
    cells : torch.Tensor
        Cell connectivity for selected cells
    contributions : torch.Tensor
        Volume contribution from each cell to its vertices

    Examples
    --------
        >>> import torch
        >>> # Add 1/3 of each triangle area to each vertex
        >>> dual_volumes = torch.zeros(4)
        >>> triangle_cells = torch.tensor([[0, 1, 2], [1, 2, 3]])
        >>> triangle_areas = torch.tensor([0.5, 0.5])
        >>> _scatter_add_cell_contributions_to_vertices(
        ...     dual_volumes, triangle_cells, triangle_areas / 3.0
        ... )
    """
    n_vertices_per_cell = cells.shape[1]
    for vertex_idx in range(n_vertices_per_cell):
        dual_volumes.scatter_add_(
            0,
            cells[:, vertex_idx],
            contributions,
        )


def _compute_meyer_mixed_voronoi_areas(
    cell_vertices: torch.Tensor,  # (n_cells, 3, n_spatial_dims)
    cell_areas: torch.Tensor,  # (n_cells,)
) -> torch.Tensor:
    """Compute per-(cell, local_vertex) mixed Voronoi areas (Meyer et al. 2003).

    This implements the branchless mixed Voronoi area formula for triangular meshes,
    handling both acute and obtuse triangles correctly. For acute triangles, uses the
    circumcentric Voronoi formula (Meyer Eq. 7). For obtuse triangles, uses the
    mixed area subdivision (Meyer Fig. 4).

    Parameters
    ----------
    cell_vertices : torch.Tensor
        Vertex positions for each triangle cell.
        Shape: (n_cells, 3, n_spatial_dims)
    cell_areas : torch.Tensor
        Area of each triangle cell.
        Shape: (n_cells,)

    Returns
    -------
    torch.Tensor
        Per-(cell, local_vertex) Voronoi areas, shape (n_cells * 3,).
        Ordered as [cell0_v0, cell0_v1, cell0_v2, cell1_v0, ...], i.e.
        the flattened (n_cells, 3) tensor where column j corresponds to
        local vertex j.

    References
    ----------
    Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
    "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds".
    Section 3.3 (Equation 7) and Section 3.4 (Figure 4).
    """
    from physicsnemo.mesh.curvature._utils import compute_triangle_angles

    n_cells = cell_vertices.shape[0]
    device = cell_vertices.device
    dtype = cell_vertices.dtype

    ### Compute all 3 angles in a single vectorized call (E6 optimization)
    # Stack the 3 vertex permutations so compute_triangle_angles is called once
    # instead of three times. Each permutation computes the angle at a different
    # vertex of the triangle.
    #   Permutation 0: angle at vertex 0 -> (v0, v1, v2)
    #   Permutation 1: angle at vertex 1 -> (v1, v2, v0)
    #   Permutation 2: angle at vertex 2 -> (v2, v0, v1)
    stacked_p0 = torch.cat(
        [
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
        ],
        dim=0,
    )  # (3 * n_cells, n_spatial_dims)
    stacked_p1 = torch.cat(
        [
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
        ],
        dim=0,
    )  # (3 * n_cells, n_spatial_dims)
    stacked_p2 = torch.cat(
        [
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
        ],
        dim=0,
    )  # (3 * n_cells, n_spatial_dims)

    stacked_angles = compute_triangle_angles(
        stacked_p0, stacked_p1, stacked_p2
    )  # (3 * n_cells,)

    # Unstack into (n_cells, 3) where column j = angle at local vertex j
    all_angles = stacked_angles.reshape(3, n_cells).T  # (n_cells, 3)

    # Check if triangle is obtuse (any angle > pi/2)
    is_obtuse = torch.any(all_angles > torch.pi / 2, dim=1)  # (n_cells,)

    ### Branchless computation of mixed Voronoi areas
    # Computes both acute (Eq. 7) and obtuse (Fig. 4) formulas for all cells,
    # then selects per-cell via torch.where. This avoids data-dependent branching
    # that would break torch.compile.
    eps = safe_eps(all_angles.dtype)
    voronoi_per_vertex = torch.zeros(n_cells, 3, dtype=dtype, device=device)

    for local_v_idx in range(3):
        next_idx = (local_v_idx + 1) % 3
        prev_idx = (local_v_idx + 2) % 3

        ### Voronoi contribution (Eq. 7) - computed for ALL cells
        edge_to_next = (
            cell_vertices[:, next_idx, :] - cell_vertices[:, local_v_idx, :]
        )  # (n_cells, n_spatial_dims)
        edge_to_prev = (
            cell_vertices[:, prev_idx, :] - cell_vertices[:, local_v_idx, :]
        )  # (n_cells, n_spatial_dims)

        edge_to_next_sq = (edge_to_next**2).sum(dim=-1)  # (n_cells,)
        edge_to_prev_sq = (edge_to_prev**2).sum(dim=-1)  # (n_cells,)

        cot_prev = torch.cos(all_angles[:, prev_idx]) / torch.sin(
            all_angles[:, prev_idx]
        ).clamp(min=eps)
        cot_next = torch.cos(all_angles[:, next_idx]) / torch.sin(
            all_angles[:, next_idx]
        ).clamp(min=eps)

        voronoi_contribution = (
            edge_to_next_sq * cot_prev + edge_to_prev_sq * cot_next
        ) / 8.0  # (n_cells,)

        ### Mixed-area contribution (Figure 4) - computed for ALL cells
        is_obtuse_at_vertex = all_angles[:, local_v_idx] > torch.pi / 2
        mixed_contribution = torch.where(
            is_obtuse_at_vertex,
            cell_areas / 2.0,
            cell_areas / 4.0,
        )  # (n_cells,)

        ### Select per cell: Voronoi for acute, mixed for obtuse
        voronoi_per_vertex[:, local_v_idx] = torch.where(
            is_obtuse, mixed_contribution, voronoi_contribution
        )

    return voronoi_per_vertex.reshape(-1)  # (n_cells * 3,)


def compute_dual_volumes_0(mesh: "Mesh") -> torch.Tensor:
    """Compute circumcentric dual 0-cell volumes (Voronoi regions) at mesh vertices.

    This is the unified, mathematically rigorous implementation used by both DEC
    operators and curvature computations. It replaces the previous buggy
    `compute_dual_volumes_0()` in `calculus/_circumcentric_dual.py` which failed
    on obtuse triangles (giving up to 513% conservation error).

    The dual 0-cell (also called Voronoi cell or circumcentric dual) of a vertex
    is the region of points closer to that vertex than to any other. In DEC, these
    volumes appear in the Hodge star operator and normalization of the Laplacian.

    **Note**: In the curvature/differential geometry literature, these are often
    called "Voronoi areas" (for 2D) or "Voronoi volumes". In DEC literature, they
    are called "dual 0-cell volumes" (denoted |⋆v|). These are identical concepts.

    Dimension-specific algorithms:

    **1D manifolds (edges)**:
        Each vertex receives half the length of each incident edge.
        Formula: V(v) = Σ_{edges ∋ v} |edge|/2

    **2D manifolds (triangles)**:
        Uses Meyer et al. (2003) mixed area approach:
        - **Acute triangles** (all angles ≤ π/2): Circumcentric Voronoi formula (Eq. 7)
          V(v) = (1/8) Σ (||e_i||² cot(α_i) + ||e_j||² cot(α_j))
          where e_i, e_j are edges from v, α_i, α_j are opposite angles

        - **Obtuse triangles**: Mixed area subdivision (Figure 4)
          - If obtuse at vertex v: V(v) = area(T)/2
          - Otherwise: V(v) = area(T)/4

        This ensures perfect tiling and optimal error bounds.

    **3D+ manifolds (tetrahedra, etc.)**:
        Barycentric approximation (standard practice):
        V(v) = Σ_{cells ∋ v} |cell| / (n_manifold_dims + 1)

        Note: Rigorous circumcentric dual volumes in 3D require "well-centered"
        meshes where all circumcenters lie inside their simplices (Desbrun 2005).
        Mixed volume formulas for obtuse tetrahedra do not exist in the literature.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_points,) containing dual 0-cell volume for each vertex.
        For isolated vertices, volume is 0.

        Property: Σ dual_volumes = total_mesh_volume (perfect tiling)

    Raises
    ------
    NotImplementedError
        If n_manifold_dims > 3

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> dual_vols = compute_dual_volumes_0(mesh)
        >>> # Use in Hodge star: ⋆f(⋆v) = f(v) × dual_vols[v]
        >>> # Use in Laplacian: Δf(v) = (1/dual_vols[v]) × Σ w_ij(f_j - f_i)

    Mathematical Properties:
        1. Conservation: Σ_v |⋆v| = |mesh|  (perfect tiling)
        2. Optimality: Minimizes spatial averaging error (Meyer Section 3.2)
        3. Gauss-Bonnet: Enables Σ K_i × |⋆v_i| = 2πχ(M) to hold exactly

    References:
        - Meyer Eq. 7 (circumcentric Voronoi, acute triangles)
        - Meyer Fig. 4 (mixed area, obtuse triangles)
        - Desbrun Def. of circumcentric dual (lines 333-352 in umich_dec.tex)
        - Hirani Def. 2.4.5 (dual cell definition, lines 884-896 in Hirani03.txt)
    """
    device = mesh.points.device
    n_points = mesh.n_points
    n_manifold_dims = mesh.n_manifold_dims

    ### Initialize dual volumes
    dual_volumes = torch.zeros(n_points, dtype=mesh.points.dtype, device=device)

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return dual_volumes

    ### Get cell volumes (reuse existing computation)
    cell_volumes = mesh.cell_areas  # (n_cells,) - "areas" is volumes in nD

    ### Dimension-specific computation
    if n_manifold_dims == 1:
        ### 1D: Each vertex gets half the length of each incident edge
        # This is exact for piecewise linear 1-manifolds
        _scatter_add_cell_contributions_to_vertices(
            dual_volumes, mesh.cells, cell_volumes / 2.0
        )

    elif n_manifold_dims == 2:
        ### 2D: Mixed Voronoi area for triangles using Meyer et al. 2003 algorithm
        # Reference: Section 3.3 (Equation 7) and Section 3.4 (Figure 4)
        #
        # CRITICAL: This correctly handles BOTH acute and obtuse triangles.
        # The previous buggy implementation in _circumcentric_dual.py assumed
        # circumcenters were always inside triangles, which is only true for acute.

        cell_vertices = mesh.points[mesh.cells]  # (n_cells, 3, n_spatial_dims)
        voronoi_areas = _compute_meyer_mixed_voronoi_areas(
            cell_vertices, cell_volumes
        )  # (n_cells * 3,)

        ### Scatter to global dual volumes
        voronoi_areas_2d = voronoi_areas.reshape(mesh.n_cells, 3)  # (n_cells, 3)
        for local_v_idx in range(3):
            vertex_indices = mesh.cells[:, local_v_idx]
            dual_volumes.scatter_add_(
                0, vertex_indices, voronoi_areas_2d[:, local_v_idx]
            )

    elif n_manifold_dims >= 3:
        ### 3D and higher: Barycentric subdivision
        # Each vertex gets equal share of each incident cell's volume
        #
        # NOTE: This is an APPROXIMATION, not rigorous like 2D.
        # Rigorous circumcentric dual volumes in 3D+ require "well-centered"
        # meshes where all circumcenters lie inside simplices (Desbrun 2005).
        # Mixed volume formulas for obtuse tetrahedra do NOT exist in literature.
        n_vertices_per_cell = n_manifold_dims + 1
        _scatter_add_cell_contributions_to_vertices(
            dual_volumes, mesh.cells, cell_volumes / n_vertices_per_cell
        )

    else:
        raise NotImplementedError(
            f"Dual volume computation not implemented for {n_manifold_dims=}. "
            f"Currently supported: 1D (edges), 2D (triangles), 3D+ (tetrahedra, etc.)."
        )

    return dual_volumes


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

        Substituting d = c - v₀ gives n linear equations:
            2(v_i - v₀)·d = ||v_i - v₀||²  for i=1,...,n

        In matrix form: A·d = b where:
            A = 2[(v₁-v₀)^T, (v₂-v₀)^T, ...]^T
            b = [||v₁-v₀||², ||v₂-v₀||², ...]^T

        Then c = v₀ + d. For over-determined systems (embedded manifolds),
        use least-squares.
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

    # Right-hand side: ||v_i - v₀||²
    # Shape: (n_simplices, n_manifold_dims)
    b = (relative_vecs**2).sum(dim=-1)

    ### Solve for circumcenter
    # Need to solve: A @ (c - v₀) = b for each simplex
    # This is: 2*(v_i - v₀) @ (c - v₀) = ||v_i - v₀||²

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

    from physicsnemo.mesh.utilities._topology import extract_unique_edges

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
