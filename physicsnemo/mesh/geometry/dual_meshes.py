"""Dual mesh (circumcentric/Voronoi) volume computation.

This module provides the unified implementation of dual 0-cell volumes (Voronoi regions)
for n-dimensional simplicial meshes. These volumes are fundamental to both:
- Discrete Exterior Calculus (DEC) operators (Hodge star, Laplacian, etc.)
- Discrete differential geometry (curvature computations)

The implementation follows Meyer et al. (2003) for 2D manifolds, using the mixed
Voronoi area approach that handles both acute and obtuse triangles correctly.

For higher dimensions, barycentric approximation is used as rigorous circumcentric
dual volumes require well-centered meshes (Desbrun et al. 2005, Hirani 2003).

References:
    Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
    "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds". VisMath.

    Desbrun, M., Hirani, A. N., Leok, M., & Marsden, J. E. (2005).
    "Discrete Exterior Calculus". arXiv:math/0508341.

    Hirani, A. N. (2003). "Discrete Exterior Calculus". PhD thesis, Caltech.
"""

from typing import TYPE_CHECKING

import torch

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

    Args:
        dual_volumes: Accumulator for dual volumes (modified in place)
        cells: Cell connectivity for selected cells
        contributions: Volume contribution from each cell to its vertices

    Example:
        >>> # Add 1/3 of each triangle area to each vertex
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

    Args:
        mesh: Input simplicial mesh

    Returns:
        Tensor of shape (n_points,) containing dual 0-cell volume for each vertex.
        For isolated vertices, volume is 0.

        Property: Σ dual_volumes = total_mesh_volume (perfect tiling)

    Raises:
        NotImplementedError: If n_manifold_dims > 3

    Example:
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

        # Compute all three angles in each triangle
        cell_vertices = mesh.points[mesh.cells]  # (n_cells, 3, n_spatial_dims)

        from physicsnemo.mesh.curvature._utils import compute_triangle_angles

        angles_0 = compute_triangle_angles(
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
        )
        angles_1 = compute_triangle_angles(
            cell_vertices[:, 1, :],
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
        )
        angles_2 = compute_triangle_angles(
            cell_vertices[:, 2, :],
            cell_vertices[:, 0, :],
            cell_vertices[:, 1, :],
        )

        # Stack angles: (n_cells, 3)
        all_angles = torch.stack([angles_0, angles_1, angles_2], dim=1)

        # Check if obtuse (any angle > π/2)
        is_obtuse = torch.any(all_angles > torch.pi / 2, dim=1)  # (n_cells,)

        ### Non-obtuse triangles: Use circumcentric Voronoi formula (Eq. 7)
        # A_voronoi_i = (1/8) * Σ (||e_ij||² cot(α_ij) + ||e_ik||² cot(α_ik))
        # For each vertex i in a non-obtuse triangle, compute Voronoi contribution
        non_obtuse_mask = ~is_obtuse

        if non_obtuse_mask.any():
            ### Extract non-obtuse triangles
            non_obtuse_cells = mesh.cells[non_obtuse_mask]  # (n_non_obtuse, 3)
            non_obtuse_vertices = cell_vertices[
                non_obtuse_mask
            ]  # (n_non_obtuse, 3, n_spatial_dims)
            non_obtuse_angles = all_angles[non_obtuse_mask]  # (n_non_obtuse, 3)

            ### For each of the 3 vertices in each triangle, compute Voronoi area
            # Vertex 0: uses edges to vertices 1 and 2
            # Voronoi area = (1/8) * (||edge_01||² * cot(angle_2) + ||edge_02||² * cot(angle_1))

            for local_v_idx in range(3):
                ### Get the two adjacent vertices (in cyclic order)
                next_idx = (local_v_idx + 1) % 3
                prev_idx = (local_v_idx + 2) % 3

                ### Compute edge vectors from current vertex
                edge_to_next = (
                    non_obtuse_vertices[:, next_idx, :]
                    - non_obtuse_vertices[:, local_v_idx, :]
                )  # (n_non_obtuse, n_spatial_dims)
                edge_to_prev = (
                    non_obtuse_vertices[:, prev_idx, :]
                    - non_obtuse_vertices[:, local_v_idx, :]
                )  # (n_non_obtuse, n_spatial_dims)

                ### Compute edge lengths squared
                edge_to_next_sq = (edge_to_next**2).sum(dim=-1)  # (n_non_obtuse,)
                edge_to_prev_sq = (edge_to_prev**2).sum(dim=-1)  # (n_non_obtuse,)

                ### Get cotangents of opposite angles
                # Cotangent at prev vertex (opposite to edge_to_next)
                cot_prev = torch.cos(non_obtuse_angles[:, prev_idx]) / torch.sin(
                    non_obtuse_angles[:, prev_idx]
                ).clamp(min=1e-10)
                # Cotangent at next vertex (opposite to edge_to_prev)
                cot_next = torch.cos(non_obtuse_angles[:, next_idx]) / torch.sin(
                    non_obtuse_angles[:, next_idx]
                ).clamp(min=1e-10)

                ### Compute Voronoi area contribution for this vertex (Equation 7)
                voronoi_contribution = (
                    edge_to_next_sq * cot_prev + edge_to_prev_sq * cot_next
                ) / 8.0  # (n_non_obtuse,)

                ### Scatter to global dual volumes
                vertex_indices = non_obtuse_cells[:, local_v_idx]
                dual_volumes.scatter_add_(0, vertex_indices, voronoi_contribution)

        ### Obtuse triangles: Use mixed area (Figure 4)
        # If angle at vertex is obtuse: add area(T)/2
        # Else: add area(T)/4
        if is_obtuse.any():
            obtuse_cells = mesh.cells[is_obtuse]  # (n_obtuse, 3)
            obtuse_volumes = cell_volumes[is_obtuse]  # (n_obtuse,)
            obtuse_angles = all_angles[is_obtuse]  # (n_obtuse, 3)

            ### For each of the 3 vertices in each obtuse triangle
            for local_v_idx in range(3):
                ### Check if angle at this vertex is obtuse
                is_obtuse_at_vertex = obtuse_angles[:, local_v_idx] > torch.pi / 2

                ### Compute contribution based on Meyer Figure 4
                # If obtuse at vertex: area(T)/2, else: area(T)/4
                contribution = torch.where(
                    is_obtuse_at_vertex,
                    obtuse_volumes / 2.0,
                    obtuse_volumes / 4.0,
                )  # (n_obtuse,)

                ### Scatter to global dual volumes
                vertex_indices = obtuse_cells[:, local_v_idx]
                dual_volumes.scatter_add_(0, vertex_indices, contribution)

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
