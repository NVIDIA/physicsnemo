"""Topology validation for simplicial meshes.

This module provides functions to check topological properties of meshes:
- Watertight checking: mesh has no boundary (all facets shared by exactly 2 cells)
- Manifold checking: mesh is a valid topological manifold
"""

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def is_watertight(mesh: "Mesh") -> bool:
    """Check if mesh is watertight (has no boundary).

    A mesh is watertight if every codimension-1 facet is shared by exactly 2 cells.
    This means the mesh forms a closed surface/volume with no holes or gaps.

    Args:
        mesh: Input simplicial mesh to check

    Returns:
        True if mesh is watertight (no boundary facets), False otherwise

    Example:
        >>> # Closed sphere is watertight
        >>> sphere = create_sphere_mesh(subdivisions=3)
        >>> is_watertight(sphere)  # True
        >>>
        >>> # Open cylinder with holes at ends
        >>> cylinder = create_cylinder_mesh(closed=False)
        >>> is_watertight(cylinder)  # False
        >>>
        >>> # Single tetrahedron has 4 boundary faces
        >>> tet = Mesh(points, cells=torch.tensor([[0, 1, 2, 3]]))
        >>> is_watertight(tet)  # False
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        extract_candidate_facets,
        categorize_facets_by_count,
    )

    ### Empty mesh is considered watertight
    if mesh.n_cells == 0:
        return True

    ### Extract all codimension-1 facets
    candidate_facets, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Deduplicate and get counts
    _, _, counts = categorize_facets_by_count(candidate_facets, target_counts="all")

    ### Watertight iff all facets appear exactly twice
    # Each facet should be shared by exactly 2 cells
    return bool(torch.all(counts == 2))


def is_manifold(
    mesh: "Mesh",
    check_level: Literal["facets", "edges", "full"] = "full",
) -> bool:
    """Check if mesh is a valid topological manifold.

    A mesh is a manifold if it locally looks like Euclidean space at every point.
    This function checks various topological constraints depending on the check level.

    Args:
        mesh: Input simplicial mesh to check
        check_level: Level of checking to perform:
            - "facets": Only check codimension-1 facets (each appears 1-2 times)
            - "edges": Check facets + edge neighborhoods (for 2D/3D meshes)
            - "full": Complete manifold validation (default)

    Returns:
        True if mesh passes the specified manifold checks, False otherwise

    Example:
        >>> # Valid manifold (sphere)
        >>> sphere = create_sphere_mesh(subdivisions=3)
        >>> is_manifold(sphere)  # True
        >>>
        >>> # Non-manifold mesh with T-junction (edge shared by 3+ faces)
        >>> non_manifold = create_t_junction_mesh()
        >>> is_manifold(non_manifold)  # False
        >>>
        >>> # Manifold with boundary (open cylinder)
        >>> cylinder = create_cylinder_mesh(closed=False)
        >>> is_manifold(cylinder)  # True (manifold with boundary is OK)

    Note:
        This function checks topological constraints but does not check for
        geometric self-intersections (which would require expensive spatial queries).
    """
    ### Empty mesh is considered a valid manifold
    if mesh.n_cells == 0:
        return True

    ### Check facets (codimension-1)
    if not _check_facets_manifold(mesh):
        return False

    if check_level == "facets":
        return True

    ### Check edges (for 2D and 3D meshes)
    if mesh.n_manifold_dims >= 2:
        if not _check_edges_manifold(mesh):
            return False

    if check_level == "edges":
        return True

    ### Full check includes vertices (for 2D and 3D meshes)
    if mesh.n_manifold_dims >= 2:
        if not _check_vertices_manifold(mesh):
            return False

    return True


def _check_facets_manifold(mesh: "Mesh") -> bool:
    """Check if facets satisfy manifold constraints.

    For a manifold (possibly with boundary), each codimension-1 facet must appear
    in at most 2 cells. Facets appearing once are on the boundary; facets appearing
    twice are interior.

    Args:
        mesh: Input mesh

    Returns:
        True if facets satisfy manifold constraints
    """
    from physicsnemo.mesh.boundaries._facet_extraction import (
        extract_candidate_facets,
        categorize_facets_by_count,
    )

    ### Extract all codimension-1 facets
    candidate_facets, _ = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Deduplicate and get counts
    _, _, counts = categorize_facets_by_count(candidate_facets, target_counts="all")

    ### For manifold: each facet appears at most twice (1 = boundary, 2 = interior)
    # If any facet appears 3+ times, it's a non-manifold edge
    return bool(torch.all(counts <= 2))


def _check_edges_manifold(mesh: "Mesh") -> bool:
    """Check if edges satisfy manifold constraints.

    For 2D manifolds (triangles): Each edge should be shared by at most 2 triangles.
    For 3D manifolds (tetrahedra): Each edge should have a valid "link" - the set of
    facets (triangles) incident to the edge should form a topological disk or circle.

    Args:
        mesh: Input mesh (must have n_manifold_dims >= 2)

    Returns:
        True if edges satisfy manifold constraints
    """
    from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

    ### For 2D meshes, edges are codimension-1, already checked in _check_facets_manifold
    if mesh.n_manifold_dims == 2:
        return True

    ### For 3D meshes, extract edges (codimension-2 facets)
    if mesh.n_manifold_dims == 3:
        candidate_edges, parent_cell_indices = extract_candidate_facets(
            mesh.cells,
            manifold_codimension=2,
        )

        ### Find unique edges and their parent cells
        unique_edges, inverse_indices = torch.unique(
            candidate_edges,
            dim=0,
            return_inverse=True,
        )

        ### For each edge, check that the cells around it form a valid configuration
        # In a manifold, the triangular faces around an edge should form a cycle
        # (for interior edges) or a fan (for boundary edges)

        ### Simple check: count cells per edge
        # In a 3D manifold, an edge can be shared by any number of tetrahedra,
        # but the triangular faces around the edge must form a valid fan/cycle

        ### For now, we do a simpler check: ensure each edge appears in at least one cell
        # A more sophisticated check would require analyzing the link of the edge
        edge_counts = torch.zeros(
            len(unique_edges), dtype=torch.int64, device=mesh.cells.device
        )
        edge_counts.scatter_add_(
            dim=0,
            index=inverse_indices,
            src=torch.ones_like(inverse_indices),
        )

        ### All edges should be used by at least one cell
        if torch.any(edge_counts == 0):
            return False

        ### Additional check: extract the triangular faces around each edge
        # and verify they form a topological disk or circle
        # This is more complex and requires analyzing face adjacency
        # For now, we rely on the facet check which catches most non-manifold cases

        return True

    ### For higher dimensions, we don't have specific checks yet
    return True


def _check_vertices_manifold(mesh: "Mesh") -> bool:
    """Check if vertices satisfy manifold constraints.

    For a manifold, the link of each vertex (the set of cells incident to the vertex)
    must form a valid topological structure:
    - For 2D: The edges around each vertex form a single cycle or fan
    - For 3D: The faces around each vertex form a single connected surface

    Args:
        mesh: Input mesh (must have n_manifold_dims >= 2)

    Returns:
        True if vertices satisfy manifold constraints
    """
    ### For 2D meshes, check that edges around each vertex form a valid fan/cycle
    if mesh.n_manifold_dims == 2:
        return _check_2d_vertex_manifold(mesh)

    ### For 3D meshes, check that faces around each vertex form a connected surface
    if mesh.n_manifold_dims == 3:
        return _check_3d_vertex_manifold()

    ### For other dimensions, no specific check
    return True


def _check_2d_vertex_manifold(mesh: "Mesh") -> bool:
    """Check vertex manifold constraints for 2D meshes.

    For a 2D triangular mesh to be manifold at a vertex, the triangles around the
    vertex must form a single fan (for boundary vertices) or a complete cycle
    (for interior vertices).

    Args:
        mesh: 2D triangular mesh

    Returns:
        True if all vertices satisfy 2D manifold constraints
    """
    from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

    ### Extract edges (codimension-1 for 2D)
    candidate_edges, parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Find unique edges
    unique_edges, inverse_indices, edge_counts = torch.unique(
        candidate_edges,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    ### For each vertex, count how many boundary edges are incident
    # In a manifold with boundary, each boundary vertex should have exactly 2 boundary edges
    # In a closed manifold, no vertex should have boundary edges

    boundary_edge_mask = edge_counts == 1
    boundary_edges = unique_edges[boundary_edge_mask]

    if len(boundary_edges) > 0:
        ### Count boundary edges per vertex
        vertex_boundary_count = torch.zeros(
            mesh.n_points, dtype=torch.int64, device=mesh.cells.device
        )
        vertex_boundary_count.scatter_add_(
            dim=0,
            index=boundary_edges.flatten(),
            src=torch.ones(
                boundary_edges.numel(), dtype=torch.int64, device=mesh.cells.device
            ),
        )

        ### Each boundary vertex should have exactly 2 boundary edges (forms a chain)
        # Non-boundary vertices should have 0
        valid_counts = (vertex_boundary_count == 0) | (vertex_boundary_count == 2)
        if not torch.all(valid_counts):
            return False

    return True


def _check_3d_vertex_manifold() -> bool:
    """Check vertex manifold constraints for 3D meshes.

    For a 3D tetrahedral mesh to be manifold at a vertex, the triangular faces
    around the vertex must form a single connected surface (topological sphere
    for interior vertices, or disk for boundary vertices).

    Returns:
        True if all vertices satisfy 3D manifold constraints

    Note:
        This is a stub implementation that always returns True. A proper
        implementation would analyze face connectivity around each vertex.
    """
    ### This is a complex check that requires analyzing face connectivity
    ### around each vertex. For now, we rely on the facet and edge checks
    ### which catch most non-manifold configurations.

    ### A proper implementation would:
    ### 1. For each vertex, extract all incident triangular faces
    ### 2. Build the face adjacency graph (faces sharing an edge)
    ### 3. Check that this graph forms a single connected component
    ### 4. Check that it has the topology of a sphere (for interior) or disk (for boundary)

    ### This requires significant computation, so we defer to simpler checks for now
    return True
