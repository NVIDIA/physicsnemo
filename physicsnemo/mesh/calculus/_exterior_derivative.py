"""Discrete exterior derivative operators for DEC.

The exterior derivative d maps k-forms to (k+1)-forms. In the discrete setting,
d is the coboundary operator, dual to the boundary operator ∂.

Fundamental property: d² = 0 (applying d twice always gives zero)

This implements the discrete Stokes theorem exactly:
    ⟨dα, c⟩ = ⟨α, ∂c⟩  (true by definition)

Reference: Desbrun et al., "Discrete Exterior Calculus", Section 3
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def exterior_derivative_0(
    mesh: "Mesh",
    vertex_0form: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exterior derivative of 0-form (function on vertices).

    Maps Ω⁰(K) → Ω¹(K): takes vertex values to edge values.

    For an oriented edge [v_i, v_j]:
        df([v_i, v_j]) = f(v_j) - f(v_i)

    This is the discrete gradient, represented as a 1-form on edges.

    Args:
        mesh: Simplicial mesh
        vertex_0form: Values at vertices, shape (n_points,) or (n_points, ...)

    Returns:
        Tuple of (edge_values, edge_connectivity):
        - edge_values: 1-form values on edges, shape (n_edges,) or (n_edges, ...)
        - edge_connectivity: Edge vertex indices, shape (n_edges, 2)

    Example:
        For a triangle mesh with scalar field f at vertices:
        >>> edge_df, edges = exterior_derivative_0(mesh, f)
        >>> # edge_df[i] = f[edges[i,1]] - f[edges[i,0]]
    """
    ### Extract edges from mesh
    # Get 1-skeleton (edge mesh) from the full mesh
    # For triangle mesh: edges are 1-simplices (codimension 1 of 2-simplex)
    # For tet mesh: edges are also needed

    # Use get_facet_mesh to extract edges (codimension = n_manifold_dims - 1)
    # This gives us (n-1)-dimensional facets, but we want 1-simplices (edges)
    # So we need codimension to get to dimension 1

    if mesh.n_manifold_dims >= 1:
        # Extract 1-simplices (edges)
        codim_to_edges = mesh.n_manifold_dims - 1
        edge_mesh = mesh.get_facet_mesh(
            manifold_codimension=codim_to_edges,
            data_source="cells",
        )
        edges = edge_mesh.cells  # (n_edges, 2)
    else:
        # 0-manifold (point cloud): no edges
        edges = torch.empty((0, 2), dtype=torch.long, device=mesh.cells.device)

    ### Compute oriented difference along each edge
    # df(edge) = f(v₁) - f(v₀)
    # Edge ordering: we use canonical ordering (sorted vertices)

    # Ensure edges are canonically ordered (smaller index first)
    # This is important for consistent orientation
    sorted_edges, sort_indices = torch.sort(edges, dim=-1)

    # Compute differences
    if vertex_0form.ndim == 1:
        # Scalar case
        edge_values = (
            vertex_0form[sorted_edges[:, 1]] - vertex_0form[sorted_edges[:, 0]]
        )
    else:
        # Tensor case: apply to each component
        edge_values = (
            vertex_0form[sorted_edges[:, 1]] - vertex_0form[sorted_edges[:, 0]]
        )

    return edge_values, sorted_edges


def exterior_derivative_1(
    mesh: "Mesh",
    edge_1form: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exterior derivative of 1-form (values on edges).

    Maps Ω¹(K) → Ω²(K): takes edge values to face values (2-cells or higher).

    For a 2-simplex (triangle) with boundary edges [v₀,v₁], [v₁,v₂], [v₂,v₀]:
        dα(triangle) = α([v₁,v₂]) - α([v₀,v₂]) + α([v₀,v₁])

    This implements the discrete curl in 2D, or the circulation around faces.

    Args:
        mesh: Simplicial mesh
        edge_1form: Values on edges, shape (n_edges,) or (n_edges, ...)
        edges: Edge connectivity, shape (n_edges, 2)

    Returns:
        Tuple of (face_values, face_connectivity):
        - face_values: 2-form values on 2-simplices, shape (n_faces,) or (n_faces, ...)
        - face_connectivity: Face vertex indices

    Note:
        For n_manifold_dims = 2 (triangle mesh), faces are the triangles themselves.
        For n_manifold_dims = 3 (tet mesh), faces are the triangular facets.
    """
    if mesh.n_manifold_dims < 2:
        # Cannot compute d₁ for manifolds of dimension < 2
        raise ValueError(
            f"exterior_derivative_1 requires n_manifold_dims >= 2, got {mesh.n_manifold_dims=}"
        )

    ### Get 2-skeleton (faces)
    if mesh.n_manifold_dims == 2:
        # For triangle mesh, the 2-cells are the triangles themselves
        faces = mesh.cells  # (n_cells, 3)
        n_faces = mesh.n_cells
    else:
        # For higher-dimensional meshes, extract 2-simplices
        codim_to_faces = mesh.n_manifold_dims - 2
        face_mesh = mesh.get_facet_mesh(
            manifold_codimension=codim_to_faces,
            data_source="cells",
        )
        faces = face_mesh.cells  # (n_faces, 3)
        n_faces = face_mesh.n_cells

    ### Extract all boundary edges from all faces (vectorized)
    # For each triangular face [v₀, v₁, v₂], extract edges [v₀,v₁], [v₁,v₂], [v₂,v₀]
    # Shape: (n_faces, 3, 2) where 3 is the number of edges per triangle
    boundary_edges = torch.stack(
        [
            faces[:, [0, 1]],  # edge from v₀ to v₁
            faces[:, [1, 2]],  # edge from v₁ to v₂
            faces[:, [2, 0]],  # edge from v₂ to v₀
        ],
        dim=1,
    )  # (n_faces, 3, 2)

    # Flatten to (n_faces*3, 2) for easier processing
    boundary_edges_flat = boundary_edges.reshape(-1, 2)  # (n_faces*3, 2)

    ### Create canonical edge representations (sorted vertices) for fast matching
    # Sort vertices within each edge to get canonical form (lower vertex first)
    boundary_edges_sorted, _ = boundary_edges_flat.sort(dim=1)
    edges_sorted, _ = edges.sort(dim=1)

    # Convert each edge to a unique integer ID for efficient lookup
    # Formula: edge_id = min_vertex * (max_vertex + 1) + max_vertex
    # This creates a unique mapping assuming vertices are non-negative integers
    max_vertex_id = max(edges.max().item(), faces.max().item()) + 1
    boundary_edge_ids = (
        boundary_edges_sorted[:, 0] * max_vertex_id + boundary_edges_sorted[:, 1]
    )
    edge_ids = edges_sorted[:, 0] * max_vertex_id + edges_sorted[:, 1]

    ### Use searchsorted for efficient vectorized lookup
    # Sort edge_ids and keep track of original indices
    edge_ids_sorted, sort_indices = torch.sort(edge_ids)

    # Find where each boundary edge ID would fit in the sorted edge list
    positions = torch.searchsorted(edge_ids_sorted, boundary_edge_ids)

    # Clamp positions to valid range to avoid index errors
    positions = positions.clamp(max=len(edge_ids_sorted) - 1)

    # Check if the found positions are exact matches
    matches = edge_ids_sorted[positions] == boundary_edge_ids  # (n_faces*3,)

    # Get the original edge indices
    edge_indices = sort_indices[positions]  # (n_faces*3,)

    ### Determine orientation of each boundary edge
    # If edge is [v_i, v_j] with v_i < v_j, orientation is +1
    # If edge is [v_i, v_j] with v_i > v_j, orientation is -1 (reversed)
    orientations = torch.where(
        boundary_edges_flat[:, 0] < boundary_edges_flat[:, 1],
        torch.ones(
            boundary_edges_flat.shape[0],
            dtype=edge_1form.dtype,
            device=edge_1form.device,
        ),
        -torch.ones(
            boundary_edges_flat.shape[0],
            dtype=edge_1form.dtype,
            device=edge_1form.device,
        ),
    )  # (n_faces*3,)

    ### Compute contributions from each edge, respecting orientation
    # Get the edge values for all boundary edges
    edge_values = edge_1form[edge_indices]  # (n_faces*3,) or (n_faces*3, ...)

    # Broadcast orientations and matches to match the shape of edge_values
    # Add singleton dimensions to the right to match any trailing dimensions
    orientations_broadcast = orientations.reshape(
        -1, *([1] * (edge_values.ndim - 1))
    )  # (n_faces*3, 1, 1, ...)
    matches_broadcast = matches.reshape(
        -1, *([1] * (edge_values.ndim - 1))
    )  # (n_faces*3, 1, 1, ...)

    # Apply orientation and mask out non-matches (set to 0 contribution)
    edge_contributions = torch.where(
        matches_broadcast,
        orientations_broadcast * edge_values,
        torch.zeros_like(edge_values),
    )  # (n_faces*3,) or (n_faces*3, ...)

    ### Sum contributions from the 3 edges of each face to get circulation
    # Reshape to (n_faces, 3, ...) and sum over the 3 edges
    edge_contributions = edge_contributions.reshape(n_faces, 3, *edge_1form.shape[1:])
    face_values = edge_contributions.sum(dim=1)  # (n_faces,) or (n_faces, ...)

    return face_values, faces
