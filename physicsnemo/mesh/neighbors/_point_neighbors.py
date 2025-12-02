"""Compute point-based adjacency relationships in simplicial meshes.

This module provides functions to compute:
- Point-to-cells adjacency (star of each vertex)
- Point-to-points adjacency (graph edges)
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.neighbors._adjacency import Adjacency

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def get_point_to_cells_adjacency(mesh: "Mesh") -> Adjacency:
    """Compute the star of each vertex (all cells containing each point).

    For each point in the mesh, finds all cells that contain that point. This
    is the graph-theoretic "star" operation on vertices.

    Args:
        mesh: Input simplicial mesh.

    Returns:
        Adjacency where adjacency.to_list()[i] contains all cell indices that
        contain point i. Isolated points (not in any cells) have empty lists.

    Example:
        >>> # Triangle mesh with 4 points, 2 triangles
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> adj = get_point_to_cells_adjacency(mesh)
        >>> adj.to_list()
        [[0], [0, 1], [0, 1], [1]]  # Point 0 in cell 0, point 1 in cells 0&1, etc.
    """
    ### Handle empty mesh
    if mesh.n_cells == 0 or mesh.n_points == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_points + 1, dtype=torch.int64, device=mesh.points.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.points.device),
        )

    from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs

    ### Create (point_id, cell_id) pairs for all vertices in all cells
    n_cells, n_vertices_per_cell = mesh.cells.shape

    # Flatten cells to get all point indices
    # Shape: (n_cells * n_vertices_per_cell,)
    point_ids = mesh.cells.reshape(-1)

    # Create corresponding cell indices for each point
    # Shape: (n_cells * n_vertices_per_cell,)
    cell_ids = torch.arange(
        n_cells, dtype=torch.int64, device=mesh.cells.device
    ).repeat_interleave(n_vertices_per_cell)

    ### Build adjacency using shared utility
    return build_adjacency_from_pairs(
        source_indices=point_ids,
        target_indices=cell_ids,
        n_sources=mesh.n_points,
    )


def get_point_to_points_adjacency(mesh: "Mesh") -> Adjacency:
    """Compute point-to-point adjacency (graph edges of the mesh).

    For each point, finds all other points that share a cell with it. In simplicial
    meshes, this is equivalent to finding all points connected by an edge, since
    all vertices in a simplex are pairwise connected.

    Args:
        mesh: Input simplicial mesh.

    Returns:
        Adjacency where adjacency.to_list()[i] contains all point indices that
        share a cell (edge) with point i. Isolated points have empty lists.

    Example:
        >>> # Three points forming a single triangle
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]])
        >>> cells = torch.tensor([[0, 1, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> adj = get_point_to_points_adjacency(mesh)
        >>> adj.to_list()
        [[1, 2], [0, 2], [0, 1]]  # Each point connected to the other two
    """
    from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

    ### Handle empty mesh
    if mesh.n_cells == 0 or mesh.n_points == 0:
        return Adjacency(
            offsets=torch.zeros(
                mesh.n_points + 1, dtype=torch.int64, device=mesh.points.device
            ),
            indices=torch.zeros(0, dtype=torch.int64, device=mesh.points.device),
        )

    ### Extract all edges (1-simplices) from cells
    # Special case: For 1D meshes, cells ARE edges already
    if mesh.n_manifold_dims == 1:
        # For 1D meshes, cells are already edges, just deduplicate them
        # Sort each edge's vertices to canonical form
        sorted_cells = torch.sort(mesh.cells, dim=1)[0]
        unique_edges = torch.unique(sorted_cells, dim=0)
    else:
        # For n-simplices with n > 1, edges are (n-1)-dimensional facets
        # manifold_codimension = n_manifold_dims - 1 gives us 1-simplices (edges)
        candidate_edges, _ = extract_candidate_facets(
            mesh.cells,
            manifold_codimension=mesh.n_manifold_dims - 1,
        )

        ### Deduplicate edges using torch.unique
        # Each edge appears only once after deduplication
        # Shape: (n_unique_edges, 2)
        unique_edges = torch.unique(candidate_edges, dim=0)

    from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs

    ### Create bidirectional edges
    # For each edge [a, b], create both [a, b] and [b, a]
    # Shape: (2 * n_unique_edges, 2)
    bidirectional_edges = torch.cat(
        [
            unique_edges,
            unique_edges.flip(dims=[1]),  # Reverse the edge direction
        ],
        dim=0,
    )

    ### Build adjacency from bidirectional edge pairs
    return build_adjacency_from_pairs(
        source_indices=bidirectional_edges[:, 0],
        target_indices=bidirectional_edges[:, 1],
        n_sources=mesh.n_points,
    )
