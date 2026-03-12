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

"""Divergence operator for vector fields.

Implements divergence using both DEC and LSQ methods.

The DEC divergence is the composition ⋆₀⁻¹ d* ⋆₁ ♭(X), where ♭ is the
PDP-flat operator (Hirani 2003, Section 5.6) that converts a vertex vector
field to a primal 1-form via midpoint averaging along edges. The composition
reduces to a weighted sum over edges:

    div(X)(v) = (1/|⋆v|) Σ_{edges [v,w]} w_ij × (X(v) + X(w))/2 · (w - v)

where w_ij are the FEM cotangent weights (= |⋆e|/|e|) and |⋆v| is the dual
0-cell (Voronoi) volume. This is exact for linear vector fields at interior
vertices and first-order convergent on smooth fields.

Physical interpretation: net flux through the dual cell boundary per unit
volume, with the PDP-flat providing the edge flux estimate.
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.utilities._tolerances import safe_eps

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_divergence_points_dec(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    r"""Compute divergence at vertices using DEC: div = ⋆₀⁻¹ d* ⋆₁ ♭(X).

    For a vertex vector field X, the DEC divergence at vertex v is:

    .. math::

        \operatorname{div}(X)(v) = \frac{1}{|{\star}v|}
        \sum_{\text{edges } [v,w]} w_{vw}\;
        \frac{X(v) + X(w)}{2} \cdot (w - v)

    where :math:`w_{vw} = |{\star}e|/|e|` is the FEM cotangent weight and
    :math:`|{\star}v|` is the dual 0-cell volume (Voronoi area).

    The edge-length factors from the Hodge star and the PDP-flat cancel
    algebraically: :math:`|{\star}e| \times (X \cdot \hat{e})
    = w \times |e| \times (X \cdot \vec{e}/|e|) = w \times (X \cdot \vec{e})`,
    so only cotangent weights and full edge vectors are needed.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh of any manifold dimension.
    vector_field : torch.Tensor
        Vectors at vertices, shape ``(n_points, n_spatial_dims)``.

    Returns
    -------
    torch.Tensor
        Divergence at vertices, shape ``(n_points,)``.
    """
    from physicsnemo.mesh.geometry.dual_meshes import (
        compute_cotan_weights_fem,
        get_or_compute_dual_volumes_0,
    )

    n_points = mesh.n_points

    ### Get FEM cotangent weights and canonical edges (one consistent source)
    cotan_weights, edges = compute_cotan_weights_fem(mesh)  # (n_edges,), (n_edges, 2)

    ### Get dual 0-cell volumes |⋆v| at vertices
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)  # (n_points,)

    ### Edge vectors: (w - v) for each canonical edge [v, w] with v < w
    edge_vectors = (
        mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
    )  # (n_edges, n_spatial_dims)

    v0_indices = edges[:, 0]  # (n_edges,)
    v1_indices = edges[:, 1]  # (n_edges,)

    ### PDP-flat 1-form value: <X_flat, e> = (X(v) + X(w))/2 . (w - v)
    v_edge = (
        vector_field[v0_indices] + vector_field[v1_indices]
    ) / 2  # (n_edges, n_spatial_dims)
    flat_1form = (v_edge * edge_vectors).sum(dim=-1)  # (n_edges,)

    ### Weighted flux: w_ij × <X_flat, e>
    weighted_flux = cotan_weights * flat_1form  # (n_edges,)

    ### Scatter-add to vertices with orientation signs
    # v0 (smaller index): edge points outward from v0's dual cell → positive
    # v1 (larger index):  edge points inward to v1's dual cell   → negative
    divergence = torch.zeros(
        n_points, dtype=vector_field.dtype, device=mesh.points.device
    )
    divergence.scatter_add_(0, v0_indices, weighted_flux)
    divergence.scatter_add_(0, v1_indices, -weighted_flux)

    ### Normalize by dual 0-cell volumes
    divergence = divergence / dual_volumes_0.clamp(min=safe_eps(dual_volumes_0.dtype))

    return divergence


def compute_divergence_points_lsq(
    mesh: "Mesh",
    vector_field: torch.Tensor,
) -> torch.Tensor:
    """Compute divergence at vertices using LSQ gradient of each component.

    For vector field v = [vₓ, vᵧ, v_z]:
        div(v) = ∂vₓ/∂x + ∂vᵧ/∂y + ∂v_z/∂z

    Computes the full Jacobian via a single batched LSQ solve, then takes
    the trace. This is more efficient than solving each component separately,
    because the adjacency construction, neighbor grouping, A-matrix assembly,
    and batched lstsq are all performed once instead of ``n_spatial_dims``
    times.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    vector_field : torch.Tensor
        Vectors at vertices, shape (n_points, n_spatial_dims)

    Returns
    -------
    torch.Tensor
        Divergence at vertices, shape (n_points,)
    """
    from physicsnemo.mesh.calculus._lsq_reconstruction import compute_point_gradient_lsq

    ### Single call computes full Jacobian: J[i, j, k] = ∂v_j/∂x_k
    # Shape: (n_points, n_spatial_dims, n_spatial_dims)
    jacobian = compute_point_gradient_lsq(mesh, vector_field)

    ### Divergence = trace of Jacobian = Σ_k ∂v_k/∂x_k
    return torch.einsum("...ii", jacobian)
