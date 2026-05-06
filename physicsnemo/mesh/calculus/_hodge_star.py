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

r"""Hodge star operator for Discrete Exterior Calculus.

The Hodge star :math:`\star` maps :math:`k`-forms to :math:`(n-k)`-forms,
where :math:`n` is the manifold dimension. It's used for defining inner
products on forms and building higher-level DEC operators.

Key property: :math:`\star \star = (-1)^{k(n-k)}` on :math:`k`-forms.

The discrete Hodge star preserves averages between primal and dual cells:

.. math::

    \frac{\langle \alpha, \sigma \rangle}{|\sigma|}
        = \frac{\langle \star \alpha, \star \sigma \rangle}{|\star \sigma|}.

Reference: Desbrun et al. (2005), *Discrete Exterior Calculus*, §6
(Hodge Star and Codifferential).
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def hodge_star_0(
    mesh: "Mesh",
    primal_0form: Float[torch.Tensor, "n_points ..."],
) -> Float[torch.Tensor, "n_points ..."]:
    r"""Apply Hodge star to 0-form (vertex values).

    Maps :math:`\star_0 : \Omega^0(K) \to \Omega^n(\star K)`.

    Takes values at vertices (0-simplices) to values at dual :math:`n`-cells.
    In the dual mesh, each vertex corresponds to a dual :math:`n`-cell
    (Voronoi region).

    .. math::

        \frac{\langle \star f, \star v \rangle}{|\star v|}
            = \frac{\langle f, v \rangle}{|v|} = f(v),

    since :math:`|v| = 1` for a 0-simplex; therefore
    :math:`\star f(\star v) = f(v) \, |\star v|`.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    primal_0form : Float[torch.Tensor, "n_points ..."]
        Values at vertices.

    Returns
    -------
    Float[torch.Tensor, "n_points ..."]
        Dual :math:`n`-form values (one per cell in the dual mesh, i.e. one
        per vertex in the primal).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> f = torch.randn(mesh.n_points)  # function at vertices
    >>> star_f = hodge_star_0(mesh, f)
    >>> # star_f[i] = f[i] * dual_volume[i]
    """
    from physicsnemo.mesh.geometry.dual_meshes import (
        get_or_compute_dual_volumes_0,
    )

    dual_volumes = get_or_compute_dual_volumes_0(mesh)  # (n_points,)

    ### Apply Hodge star: multiply by dual volume
    # This preserves the average: f(v)/|v| = ⋆f(⋆v)/|⋆v|
    # Since |v| = 1 for a vertex (0-dimensional), we get: ⋆f(⋆v) = f(v) × |⋆v|

    if primal_0form.ndim == 1:
        return primal_0form * dual_volumes
    else:
        # Tensor case: broadcast dual volumes
        return primal_0form * dual_volumes.view(-1, *([1] * (primal_0form.ndim - 1)))


def hodge_star_1(
    mesh: "Mesh",
    primal_1form: Float[torch.Tensor, "n_edges ..."],
    edges: Int[torch.Tensor, "n_edges 2"],
) -> Float[torch.Tensor, "n_edges ..."]:
    r"""Apply Hodge star to 1-form (edge values).

    Maps :math:`\star_1 : \Omega^1(K) \to \Omega^{n-1}(\star K)`.

    Takes values at edges (1-simplices) to values at dual
    :math:`(n-1)`-cells. From

    .. math::

        \frac{\langle \star \alpha, \star e \rangle}{|\star e|}
            = \frac{\langle \alpha, e \rangle}{|e|}

    we obtain
    :math:`\star \alpha(\star e) = \alpha(e) \, |\star e| / |e| = \alpha(e) \, w_{ij}`,
    where :math:`w_{ij}` is the FEM cotangent weight for the edge.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh of any manifold dimension.
    primal_1form : Float[torch.Tensor, "n_edges ..."]
        Values on edges.
    edges : Int[torch.Tensor, "n_edges 2"]
        Edge connectivity.

    Returns
    -------
    Float[torch.Tensor, "n_edges ..."]
        Dual :math:`(n-1)`-form values.
    """
    from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
    from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

    ### Get FEM cotangent weights w_ij = |⋆e|/|e| in canonical edge order
    canonical_weights, canonical_edges = compute_cotan_weights_fem(mesh)

    ### Map the caller's edges to the canonical ordering
    indices, matched = find_edges_in_reference(
        canonical_edges,
        edges,
        index_bound=mesh.n_points,
    )

    if not matched.all():
        n_unmatched = (~matched).sum().item()
        raise ValueError(
            f"hodge_star_1: {n_unmatched} of {len(edges)} input edges were not found "
            "in the mesh's canonical edge set. Ensure edges are valid mesh edges."
        )

    cotan_weights = canonical_weights[indices]  # (n_edges,)

    ### Apply Hodge star: ⋆α(⋆e) = α(e) × w_ij
    if primal_1form.ndim == 1:
        return primal_1form * cotan_weights
    else:
        return primal_1form * cotan_weights.view(-1, *([1] * (primal_1form.ndim - 1)))
