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

r"""Sharp and flat operators for converting between forms and vector fields.

These operators relate 1-forms (edge-based) to vector fields (vertex-based):

- Flat (:math:`\flat`): converts vector fields to 1-forms.
- Sharp (:math:`\sharp`): converts 1-forms to vector fields.

These are metric-dependent operators crucial for DEC gradient and divergence.

Reference: Desbrun et al. (2005), *Discrete Exterior Calculus*, §7
(Maps between 1-Forms and Vector Fields).
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def sharp(
    mesh: "Mesh",
    edge_1form: Float[torch.Tensor, "n_edges ..."],
    edges: Int[torch.Tensor, "n_edges 2"],
) -> Float[torch.Tensor, "n_points n_spatial_dims ..."]:
    r"""Apply sharp operator to convert 1-form to primal vector field (rigorous DEC).

    Maps :math:`\sharp: \Omega^1(K) \to \mathfrak{X}(K)`

    Converts edge-based 1-form values to vectors at vertices,

    .. math::

        \alpha^\sharp(v) = \sum_{[v, \sigma^0]}
            \langle \alpha, [v, \sigma^0] \rangle
            \sum_{\sigma^n \supset [v, \sigma^0]}
            \frac{|{\star}v \cap \sigma^n|}{|{\star}v|}
            \, \nabla \varphi_{\sigma^0,\, \sigma^n},

    where the outer sum is over edges :math:`[v, \sigma^0]` incident to
    :math:`v`, the inner sum is over cells :math:`\sigma^n` containing each
    edge, and:

    - :math:`\langle \alpha, [v, \sigma^0] \rangle` is the 1-form value on the
      oriented edge from :math:`v` to :math:`\sigma^0`,
    - :math:`|{\star}v \cap \sigma^n|` is the portion of vertex :math:`v`'s
      Voronoi cell within cell :math:`\sigma^n`,
    - :math:`|{\star}v|` is the total dual 0-cell volume of vertex :math:`v`,
    - :math:`\nabla \varphi_{\sigma^0,\, \sigma^n}` is the gradient of the
      barycentric interpolation function for :math:`\sigma^0` in cell
      :math:`\sigma^n`.

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
        Vector field at vertices, shape ``(n_points, n_spatial_dims, ...)``.

    References
    ----------
    Hirani (2003), *Discrete Exterior Calculus* (PhD thesis), Definition 5.8.1
    and Corollary 6.1.8.
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
    # Implements Hirani (2003), Discrete Exterior Calculus, Eq. 5.8.1 (FULLY VECTORIZED)
    # Challenge: This is complex to vectorize due to variable vertex valence
    # Strategy: Process all (edge, cell) pairs, then scatter to vertices

    edge_indices_for_candidates, matches = find_edges_in_reference(
        edges,
        candidate_edges,
        index_bound=mesh.n_points,
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
    vector_field: Float[torch.Tensor, "n_points n_spatial_dims ..."],
    edges: Int[torch.Tensor, "n_edges 2"],
) -> Float[torch.Tensor, "n_edges ..."]:
    r"""Apply PDP-flat operator to convert primal vector field to primal 1-form (rigorous DEC).

    Maps :math:`\flat : \mathfrak{X}(K) \to \Omega^1(K)`.

    Converts vectors at vertices (primal vector field) to edge-based 1-form
    values. For an oriented edge :math:`e = [v_0, v_1]` with direction
    vector :math:`\vec{e} = v_1 - v_0`, the PDP-flat formula from Hirani
    (2003), *Discrete Exterior Calculus* (PhD thesis), §5.6, gives

    .. math::

        \langle X^\flat, e \rangle
            = \tfrac{1}{2} X(v_0) \cdot \vec{e}
              + \tfrac{1}{2} X(v_1) \cdot \vec{e}
            = \tfrac{1}{2} \bigl(X(v_0) + X(v_1)\bigr) \cdot \vec{e}.

    This is the simplest flat operator for primal fields and is exact for
    linearly interpolated vector fields along edges.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    vector_field : Float[torch.Tensor, "n_points n_spatial_dims ..."]
        Vectors at vertices.
    edges : Int[torch.Tensor, "n_edges 2"]
        Edge connectivity.

    Returns
    -------
    Float[torch.Tensor, "n_edges ..."]
        1-form values on edges.

    References
    ----------
    Hirani (2003), *Discrete Exterior Calculus* (PhD thesis), §5.6
    (Other Discrete Flats), PDP-flat operator.

    Notes
    -----
    Hirani defines 8 different flat operators depending on:

    - Source: primal vs dual vector field.
    - Interpolation: constant in cells vs barycentric.
    - Destination: primal vs dual 1-form.

    This implements PDP-flat (primal-dual-primal): primal vectors, constant
    in Voronoi regions, to primal 1-form. This is compatible with PP-sharp.

    Algorithm: for edge :math:`[v_0, v_1]`,

    1. average vectors :math:`\bigl(X(v_0) + X(v_1)\bigr) / 2`,
    2. project onto edge direction,
    3. multiply by edge length for proper units.
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
