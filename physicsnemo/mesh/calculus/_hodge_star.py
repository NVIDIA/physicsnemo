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

"""Hodge star operator for Discrete Exterior Calculus.

The Hodge star ⋆ maps k-forms to (n-k)-forms, where n is the manifold dimension.
It's used for defining inner products on forms and building higher-level DEC operators.

Key property: ⋆⋆ = (-1)^(k(n-k)) on k-forms

The discrete Hodge star preserves averages between primal and dual cells:
    ⟨α, σ⟩/|σ| = ⟨⋆α, ⋆σ⟩/|⋆σ|

Reference: Desbrun et al., "Discrete Exterior Calculus", Section 4
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def hodge_star_0(
    mesh: "Mesh",
    primal_0form: torch.Tensor,
) -> torch.Tensor:
    """Apply Hodge star to 0-form (vertex values).

    Maps ⋆₀: Ω⁰(K) → Ωⁿ(⋆K)

    Takes values at vertices (0-simplices) to values at dual n-cells.
    In the dual mesh, each vertex corresponds to a dual n-cell (Voronoi region).

    Formula: ⟨⋆f, ⋆v⟩/|⋆v| = ⟨f, v⟩/|v| = f(v) (since |v|=1 for 0-simplex)
    Therefore: ⋆f(⋆v) = f(v) × |⋆v|

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh
    primal_0form : torch.Tensor
        Values at vertices, shape (n_points,) or (n_points, ...)

    Returns
    -------
    torch.Tensor
        Dual n-form values (one per cell in dual mesh = one per vertex in primal),
        shape (n_points,) or (n_points, ...)

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
    primal_1form: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Apply Hodge star to 1-form (edge values).

    Maps ⋆₁: Ω¹(K) → Ω^(n-1)(⋆K)

    Takes values at edges (1-simplices) to values at dual (n-1)-cells.

    Formula: ⟨⋆α, ⋆e⟩/|⋆e| = ⟨α, e⟩/|e|
    Therefore: ⋆α(⋆e) = α(e) × |⋆e|/|e| = α(e) × w_ij

    where w_ij is the FEM cotangent weight for the edge.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh of any manifold dimension.
    primal_1form : torch.Tensor
        Values on edges, shape (n_edges,) or (n_edges, ...)
    edges : torch.Tensor
        Edge connectivity, shape (n_edges, 2)

    Returns
    -------
    torch.Tensor
        Dual (n-1)-form values, shape (n_edges,) or (n_edges, ...)
    """
    from physicsnemo.mesh.geometry.dual_meshes import compute_cotan_weights_fem
    from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

    ### Get FEM cotangent weights w_ij = |⋆e|/|e| in canonical edge order
    canonical_weights, canonical_edges = compute_cotan_weights_fem(mesh)

    ### Map the caller's edges to the canonical ordering
    indices, matched = find_edges_in_reference(canonical_edges, edges)

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
