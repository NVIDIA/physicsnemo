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

r"""Divergence operator for vector fields.

Implements divergence using both DEC and LSQ methods.

The DEC divergence is the composition :math:`\star_0^{-1}\, d^{\ast}\, \star_1\, \flat(X)`,
where :math:`\flat` is the PDP-flat operator (Hirani 2003, *Discrete Exterior
Calculus*, §5.6) that converts a vertex vector field to a primal 1-form via
midpoint averaging along edges. The composition reduces to a weighted sum
over edges:

.. math::

    \operatorname{div}(X)(v) = \frac{1}{|{\star}v|}
    \sum_{\text{edges } [v,w]} w_{vw}\;
    \frac{X(v) + X(w)}{2} \cdot (w - v)

where :math:`w_{vw} = |{\star}e|/|e|` are the FEM cotangent weights and
:math:`|{\star}v|` is the dual 0-cell (Voronoi) volume. This is exact for
linear vector fields at interior vertices and first-order convergent on
smooth fields.

Physical interpretation: net flux through the dual cell boundary per unit
volume, with the PDP-flat providing the edge flux estimate.
"""

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_divergence_points_dec(
    mesh: "Mesh",
    vector_field: Float[torch.Tensor, "n_points n_spatial_dims"],
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, " n_points"]:
    r"""Compute divergence at vertices using DEC.

    Implements :math:`\operatorname{div} = \star_0^{-1}\, d^{\ast}\, \star_1\, \flat(X)`.
    For a vertex vector field :math:`X`, the DEC divergence at vertex
    :math:`v` is:

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
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    torch.Tensor
        Divergence at vertices, shape ``(n_points,)``.
    """
    from physicsnemo.mesh.geometry.dual_meshes import (
        get_or_compute_cotan_weights_fem,
        get_or_compute_dual_volumes_0,
    )
    from physicsnemo.nn.functional.derivatives.mesh_cotan_divergence import (
        mesh_cotan_divergence,
    )

    ### Get FEM cotangent weights and canonical edges (one consistent source)
    cotan_weights, edges = get_or_compute_cotan_weights_fem(mesh)

    ### Get dual 0-cell volumes |⋆v| at vertices
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)  # (n_points,)

    return mesh_cotan_divergence(
        points=mesh.points,
        edges=edges,
        cotan_weights=cotan_weights,
        dual_volumes=dual_volumes_0,
        vector_field=vector_field,
        implementation=implementation,
    )


def compute_divergence_points_lsq(
    mesh: "Mesh",
    vector_field: Float[torch.Tensor, "n_points n_spatial_dims"],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, " n_points"]:
    r"""Compute divergence at vertices using LSQ gradient of each component.

    For a vector field :math:`v = (v_x, v_y, v_z)`:

    .. math::

        \operatorname{div}(v) = \frac{\partial v_x}{\partial x}
            + \frac{\partial v_y}{\partial y}
            + \frac{\partial v_z}{\partial z}

    Computes the full derivative-first Jacobian
    ``jacobian[i, k, j] = ∂v_j/∂x_k`` via a single batched LSQ solve,
    then takes the trace. This is more efficient than solving each component
    separately, because the adjacency construction, neighbor grouping,
    A-matrix assembly, and batched lstsq are all performed once instead of
    ``n_spatial_dims`` times.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    vector_field : torch.Tensor
        Vectors at vertices, shape ``(n_points, n_spatial_dims)``.
    weight_power : float, optional
        Exponent for inverse-distance weighting.
    min_neighbors : int, optional
        Points with fewer than this many neighbors receive zero divergence.
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    Float[torch.Tensor, " n_points"]
        Divergence at vertices, shape ``(n_points,)``.
    """
    from physicsnemo.nn.functional.derivatives.mesh_lsq_divergence import (
        mesh_lsq_divergence,
    )

    adjacency = mesh.get_point_to_points_adjacency()
    return mesh_lsq_divergence(
        points=mesh.points,
        vector_field=vector_field,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation=implementation,
    )


def compute_divergence_cells_lsq(
    mesh: "Mesh",
    vector_field: Float[torch.Tensor, "n_cells n_spatial_dims"],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, " n_cells"]:
    r"""Compute divergence at cell centers using the LSQ Jacobian trace.

    Cell-centered analogue of :func:`compute_divergence_points_lsq`: computes
    the full derivative-first Jacobian via a single batched cell-neighbour LSQ
    solve, then takes the trace.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    vector_field : torch.Tensor
        Vectors at cell centers, shape ``(n_cells, n_spatial_dims)``.
    weight_power : float, optional
        Exponent for inverse-distance weighting.
    min_neighbors : int, optional
        Cells with fewer than this many neighbors receive zero divergence.
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    Float[torch.Tensor, " n_cells"]
        Divergence at cell centers, shape ``(n_cells,)``.
    """
    from physicsnemo.nn.functional.derivatives.mesh_lsq_divergence import (
        mesh_lsq_divergence,
    )

    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
    return mesh_lsq_divergence(
        points=mesh.cell_centroids,
        vector_field=vector_field,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation=implementation,
    )
