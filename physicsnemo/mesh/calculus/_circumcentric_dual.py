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

    Args:
        vertices: Vertex positions for each simplex.
            Shape: (n_simplices, n_vertices_per_simplex, n_spatial_dims)

    Returns:
        Circumcenters, shape (n_simplices, n_spatial_dims)

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
        Uses geometric approximation (inverse edge length weighting)

    For 1D manifolds (edges):
        Uses uniform weights

    Args:
        mesh: Input mesh
        edges: Edge connectivity, shape (n_edges, 2). If None, extracts edges from mesh.
        return_edges: If True, returns (weights, edges). If False, returns weights only.

    Returns:
        If return_edges=True: Tuple of (cotan_weights, edges)
        If return_edges=False: Just cotan_weights
        where cotan_weights has shape (n_edges,) and edges has shape (n_edges, 2)

    Mathematical Background:
        The cotangent weight formula comes from the circumcentric dual construction in DEC.
        For an edge e shared by triangles with opposite angles α and β, the dual 1-cell
        volume is |⋆e| = (|e|/2)(cot α + cot β), giving |⋆e|/|e| = (1/2)(cot α + cot β).

        The factor of 1/2 is GEOMETRIC, arising from the distance from edge midpoints
        to triangle circumcenters. This is rigorously derived in Desbrun et al. (2005)
        "Discrete Exterior Calculus" and Meyer et al. (2003).

    Example:
        >>> # Standard usage
        >>> weights, edges = compute_cotan_weights_triangle_mesh(mesh)
        >>> # Get weights only
        >>> weights = compute_cotan_weights_triangle_mesh(mesh, return_edges=False)
    """
    n_manifold_dims = mesh.n_manifold_dims
    device = mesh.points.device

    ### Extract edges if not provided
    if edges is None:
        if n_manifold_dims == 1:
            # For 1D manifolds, cells ARE edges
            sorted_cells = torch.sort(mesh.cells, dim=1)[0]
            sorted_edges = torch.unique(sorted_cells, dim=0)
        else:
            # For higher dimensions, extract edges from facets
            edge_mesh = mesh.get_facet_mesh(manifold_codimension=1, data_source="cells")
            sorted_edges, _ = torch.sort(edge_mesh.cells, dim=-1)
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

        cotans = dot_products / cross_mag.clamp(min=1e-10)

        ### Map candidate edges to sorted_edges and accumulate (vectorized)
        # Build hash for quick lookup
        edge_hash = candidate_edges[:, 0] * (mesh.n_points + 1) + candidate_edges[:, 1]
        sorted_hash = sorted_edges[:, 0] * (mesh.n_points + 1) + sorted_edges[:, 1]

        # Sort sorted_hash to enable binary search via searchsorted
        sorted_hash_argsort = torch.argsort(sorted_hash)
        sorted_hash_sorted = sorted_hash[sorted_hash_argsort]

        # Find index of each edge_hash in the sorted sorted_hash
        indices_in_sorted = torch.searchsorted(sorted_hash_sorted, edge_hash)

        # Clamp indices to valid range (handles any edge_hash not found)
        indices_in_sorted = torch.clamp(indices_in_sorted, 0, n_edges - 1)

        # Map back to original sorted_edges indices
        indices_in_original = sorted_hash_argsort[indices_in_sorted]

        # Accumulate cotans using scatter_add (vectorized)
        cotan_weights.scatter_add_(0, indices_in_original, cotans)

        ### Apply the REQUIRED factor of 1/2 from the geometric derivation
        # |⋆e|/|e| = (1/2) × Σ cot(opposite angles)
        cotan_weights = cotan_weights / 2.0

    elif n_manifold_dims == 3:
        ### 3D tetrahedra: Geometric approximation (inverse edge length weighting)
        # Full dihedral angle cotangents would require complex face-based structures
        # For now use simplified formula (divide by 2 for consistency with 2D case)
        edge_vectors = mesh.points[sorted_edges[:, 1]] - mesh.points[sorted_edges[:, 0]]
        edge_lengths = torch.norm(edge_vectors, dim=-1)
        cotan_weights = (1.0 / edge_lengths.clamp(min=1e-10)) / 2.0

    else:
        raise NotImplementedError(
            f"Cotangent weights not implemented for {n_manifold_dims=}."
        )

    ### Return based on return_edges flag
    if return_edges:
        return cotan_weights, sorted_edges
    else:
        return cotan_weights


def compute_dual_volumes_1(mesh: "Mesh") -> torch.Tensor:
    """Compute dual 1-cell volumes (dual to edges).

    For triangle meshes, uses the circumcentric dual construction from DEC.
    The dual 1-cell for an edge consists of segments from the edge midpoint
    to the circumcenters of adjacent triangles.

    For an edge shared by triangles with opposite angles α and β:
        |⋆e| = (|e|/2)(cot α + cot β) = |e| × w_ij
    where w_ij are the cotangent weights.

    Args:
        mesh: Input simplicial mesh

    Returns:
        Dual 1-cell volumes for each edge, shape (n_edges,)
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

    Args:
        mesh: Input mesh

    Returns:
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

    Args:
        mesh: Input mesh

    Returns:
        Circumcenters for all cells, shape (n_cells, n_spatial_dims)
    """
    cached = get_cached(mesh.cell_data, "circumcenters")
    if cached is None:
        parent_cell_vertices = mesh.points[mesh.cells]
        cached = compute_circumcenters(parent_cell_vertices)
        set_cached(mesh.cell_data, "circumcenters", cached)
    return cached
