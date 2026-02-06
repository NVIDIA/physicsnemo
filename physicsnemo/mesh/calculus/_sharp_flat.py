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

"""Sharp and flat operators for converting between forms and vector fields.

These operators relate 1-forms (edge-based) to vector fields (vertex-based):
- Flat (♭): Converts vector fields to 1-forms
- Sharp (♯): Converts 1-forms to vector fields

These are metric-dependent operators crucial for DEC gradient and divergence.

Reference: Desbrun et al., "Discrete Exterior Calculus", Section 5
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def sharp(
    mesh: "Mesh",
    edge_1form: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    r"""Apply sharp operator to convert 1-form to primal vector field (rigorous DEC).

    Maps :math:`\sharp: \Omega^1(K) \to \mathfrak{X}(K)`

    Converts edge-based 1-form values to vectors at vertices:

    .. math::

        \alpha^\sharp(v) = \sum_{\text{edges } [v,\sigma^0]}
            \langle \alpha, [v,\sigma^0] \rangle
            \sum_{\sigma^n \supset \text{edge}}
            \frac{|{\star}v \cap \sigma^n|}{|{\star}v|}
            \,\nabla\varphi_{\sigma^0, \sigma^n}

    Where:

    - :math:`\langle \alpha, [v,\sigma^0] \rangle` is the 1-form value on the
      oriented edge from *v* to :math:`\sigma^0`
    - :math:`|{\star}v \cap \sigma^n|` is the portion of vertex *v*'s Voronoi
      cell within cell :math:`\sigma^n`
    - :math:`|{\star}v|` is the total dual 0-cell volume of vertex *v*
    - :math:`\nabla\varphi_{\sigma^0, \sigma^n}` is the gradient of the
      barycentric interpolation function for :math:`\sigma^0` in cell
      :math:`\sigma^n`

    The weights :math:`|{\star}v \cap \sigma^n| / |{\star}v|` sum to 1.0 for
    each vertex, guaranteeing exact reproduction of constant gradients.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh (2D or 3D).
    edge_1form : torch.Tensor
        1-form values on edges, shape ``(n_edges,)`` or ``(n_edges, ...)``.
    edges : torch.Tensor
        Edge connectivity, shape ``(n_edges, 2)``.

    Returns
    -------
    torch.Tensor
        Vector field at vertices, shape ``(n_points, n_spatial_dims)`` or
        ``(n_points, n_spatial_dims, ...)`` for tensor-valued 1-forms.

    References
    ----------
    Hirani (2003) Definition 5.8.1, Corollary 6.1.8.
    """
    n_points = mesh.n_points
    n_spatial_dims = mesh.n_spatial_dims

    ### Initialize output
    if edge_1form.ndim == 1:
        vector_field = torch.zeros(
            (n_points, n_spatial_dims),
            dtype=edge_1form.dtype,
            device=mesh.points.device,
        )
    else:
        vector_field = torch.zeros(
            (n_points, n_spatial_dims) + edge_1form.shape[1:],
            dtype=edge_1form.dtype,
            device=mesh.points.device,
        )

    ### Get barycentric gradients for all cells
    from physicsnemo.mesh.geometry.interpolation import compute_barycentric_gradients

    bary_grads = compute_barycentric_gradients(
        mesh
    )  # (n_cells, n_verts_per_cell, n_spatial_dims)

    ### Get support volume fractions |⋆v ∩ cell| / |⋆v| (sum to 1.0 per vertex)
    from physicsnemo.mesh.geometry.support_volumes import (
        compute_vertex_support_volume_cell_fractions,
    )

    fractions, cell_vertex_pairs = compute_vertex_support_volume_cell_fractions(mesh)
    # fractions: (n_pairs,)
    # cell_vertex_pairs: (n_pairs, 2) - [cell_idx, local_vertex_idx]

    ### Build mapping from edges to cells containing them
    from physicsnemo.mesh.boundaries import extract_candidate_facets

    candidate_edges, parent_cells = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=1,
    )

    ### Match candidates to input edges to get 1-form values
    # Implements Hirani Eq. 5.8.1 (FULLY VECTORIZED)
    # Challenge: This is complex to vectorize due to variable vertex valence
    # Strategy: Process all (edge, cell) pairs, then scatter to vertices

    edge_indices_for_candidates, matches = find_edges_in_reference(
        edges, candidate_edges
    )

    ### Filter to only matched candidates
    matched_mask = matches
    matched_edge_indices = edge_indices_for_candidates[matched_mask]  # Which input edge
    matched_cell_indices = parent_cells[matched_mask]  # Which cell
    matched_candidate_edges = candidate_edges[matched_mask]  # (n_matched, 2)

    ### For each matched triple, process both vertices of the edge
    # We'll create contributions for v0 and v1 separately
    for vertex_position in [0, 1]:  # Process v0, then v1
        ### Get global vertex indices
        vertex_indices = matched_candidate_edges[:, vertex_position]  # (n_matched,)

        ### Get the OTHER vertex (for ∇φ)
        other_vertex_position = 1 - vertex_position
        other_vertex_indices = matched_candidate_edges[:, other_vertex_position]

        ### Find local indices in cells
        # For each matched triple, find where vertex appears in cell
        cells_expanded = mesh.cells[
            matched_cell_indices
        ]  # (n_matched, n_verts_per_cell)

        # Find local index of current vertex
        local_v_mask = cells_expanded == vertex_indices.unsqueeze(1)
        local_v_idx = torch.argmax(local_v_mask.int(), dim=1)  # (n_matched,)

        # Find local index of other vertex
        local_other_mask = cells_expanded == other_vertex_indices.unsqueeze(1)
        local_other_idx = torch.argmax(local_other_mask.int(), dim=1)  # (n_matched,)

        ### Get weights: |⋆v ∩ cell| / |⋆v|
        pair_indices = matched_cell_indices * (mesh.n_manifold_dims + 1) + local_v_idx
        weights = fractions[pair_indices]  # (n_matched,)

        ### Get barycentric gradients ∇φ_{other,cell}
        grad_phi = bary_grads[
            matched_cell_indices, local_other_idx, :
        ]  # (n_matched, n_spatial_dims)

        ### Get 1-form values (with orientation)
        # Orientation: +1 if vertex is first in canonical edge order, -1 if second
        # Canonical order has smaller index first
        canonical_v0 = torch.minimum(
            matched_candidate_edges[:, 0], matched_candidate_edges[:, 1]
        )
        is_first_in_canonical = vertex_indices == canonical_v0
        orientations = torch.where(is_first_in_canonical, 1.0, -1.0)  # (n_matched,)

        alpha_values = edge_1form[
            matched_edge_indices
        ]  # (n_matched,) or (n_matched, ...)

        ### Compute contributions
        if edge_1form.ndim == 1:
            # Scalar case: (n_matched,) * (n_matched,) * (n_matched, n_spatial_dims)
            contributions = (
                orientations.unsqueeze(-1)
                * alpha_values.unsqueeze(-1)
                * weights.unsqueeze(-1)
                * grad_phi
            )  # (n_matched, n_spatial_dims)

            ### Scatter-add to vector_field
            vector_field.scatter_add_(
                0,
                vertex_indices.unsqueeze(-1).expand(-1, n_spatial_dims),
                contributions,
            )
        else:
            # Tensor case: more complex broadcasting
            # alpha_values: (n_matched, features...)
            # Need: (n_matched, n_spatial_dims, features...)
            contrib_spatial = (
                orientations.unsqueeze(-1) * weights.unsqueeze(-1) * grad_phi
            )  # (n_matched, n_spatial_dims)
            contrib_spatial_expanded = contrib_spatial.unsqueeze(
                -1
            )  # (n_matched, n_spatial_dims, 1)
            alpha_expanded = alpha_values.unsqueeze(1)  # (n_matched, 1, features...)

            contributions = (
                contrib_spatial_expanded * alpha_expanded
            )  # (n_matched, n_spatial_dims, features...)

            # Flatten and scatter
            contributions_flat = contributions.reshape(len(matched_edge_indices), -1)
            vector_field_flat = vector_field.reshape(n_points, -1)

            vertex_indices_expanded = vertex_indices.unsqueeze(-1).expand(
                -1, contributions_flat.shape[1]
            )
            vector_field_flat.scatter_add_(
                0, vertex_indices_expanded, contributions_flat
            )

            vector_field = vector_field_flat.reshape(vector_field.shape)

    return vector_field


def flat(
    mesh: "Mesh",
    vector_field: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Apply PDP-flat operator to convert primal vector field to primal 1-form (rigorous DEC).

    Maps ♭: 𝔛(K) → Ω¹(K)

    Converts vectors at vertices (primal vector field) to edge-based 1-form values.
    Uses the PDP-flat formula from Hirani Section 5.6 (line 2456):

        ⟨X♭, edge⟩ = X(v0) · edge⃗/2 + X(v1) · edge⃗/2 = (X(v0) + X(v1))/2 · edge⃗

    This is the simplest flat operator for primal fields and is exact for
    linearly interpolated vector fields along edges.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    vector_field : torch.Tensor
        Vectors at vertices, shape (n_points, n_spatial_dims) or
        (n_points, n_spatial_dims, ...) for tensor fields
    edges : torch.Tensor
        Edge connectivity, shape (n_edges, 2)

    Returns
    -------
    torch.Tensor
        1-form values on edges, shape (n_edges,) or (n_edges, ...)

    References
    ----------
    Hirani (2003) Section 5.6, PDP-flat (lines 2456-2465)

    Notes
    -----
    Note on flat operator variants:
        Hirani defines 8 different flat operators depending on:
        - Source: primal vs dual vector field
        - Interpolation: constant in cells vs barycentric
        - Destination: primal vs dual 1-form

        This implements PDP-flat (Primal-Dual-Primal): primal vectors, constant
        in Voronoi regions, to primal 1-form. This is compatible with PP-sharp.

    Algorithm:
        For edge [v0, v1]:
        1. Average vectors: (X(v0) + X(v1))/2
        2. Project onto edge direction
        3. Multiply by edge length for proper units
    """
    ### Get edge vectors
    edge_vectors = (
        mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
    )  # (n_edges, n_spatial_dims)

    ### Get vectors at edge endpoints
    v0_vectors = vector_field[edges[:, 0]]  # (n_edges, n_spatial_dims, ...)
    v1_vectors = vector_field[edges[:, 1]]  # (n_edges, n_spatial_dims, ...)

    ### Average vectors (PDP-flat: constant in Voronoi regions, average at boundary)
    avg_vectors = (v0_vectors + v1_vectors) / 2  # (n_edges, n_spatial_dims, ...)

    ### Project onto edge direction: X̄ · edge⃗
    # Dot product along spatial dimension
    if vector_field.ndim == 2:
        # Scalar field case
        projection = (avg_vectors * edge_vectors).sum(dim=-1)  # (n_edges,)
    else:
        # Tensor field case
        projection = (avg_vectors * edge_vectors.unsqueeze(-1)).sum(
            dim=1
        )  # (n_edges, ...)

    return projection
