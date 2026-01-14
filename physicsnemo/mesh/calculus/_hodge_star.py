"""Hodge star operator for Discrete Exterior Calculus.

The Hodge star ⋆ maps k-forms to (n-k)-forms, where n is the manifold dimension.
It's essential for defining the codifferential δ and inner products on forms.

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

    Args:
        mesh: Simplicial mesh
        primal_0form: Values at vertices, shape (n_points,) or (n_points, ...)

    Returns:
        Dual n-form values (one per cell in dual mesh = one per vertex in primal),
        shape (n_points,) or (n_points, ...)

    Example:
        For a function f on triangle mesh vertices:
        >>> star_f = hodge_star_0(mesh, f)
        >>> # star_f[i] = f[i] * dual_volume[i]
    """
    from physicsnemo.mesh.calculus._circumcentric_dual import (
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
    Therefore: ⋆α(⋆e) = α(e) × |⋆e|/|e|

    Args:
        mesh: Simplicial mesh
        primal_1form: Values on edges, shape (n_edges,) or (n_edges, ...)
        edges: Edge connectivity, shape (n_edges, 2)

    Returns:
        Dual (n-1)-form values, shape (n_edges,) or (n_edges, ...)
    """
    from physicsnemo.mesh.calculus._circumcentric_dual import compute_dual_volumes_1

    ### Compute edge lengths (primal 1-cell volumes)
    edge_vectors = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
    edge_lengths = torch.norm(edge_vectors, dim=-1)  # |e|, shape (n_edges,)

    ### Get dual volumes
    dual_volumes = compute_dual_volumes_1(mesh)  # |⋆e|, shape (n_edges,)

    ### Apply Hodge star: multiply by ratio of dual to primal volumes
    volume_ratio = dual_volumes / edge_lengths  # |⋆e|/|e|

    if primal_1form.ndim == 1:
        return primal_1form * volume_ratio
    else:
        # Tensor case
        return primal_1form * volume_ratio.view(-1, *([1] * (primal_1form.ndim - 1)))


def codifferential(
    k: int,
    **kwargs,
) -> torch.Tensor:
    """Compute codifferential δ of a (k+1)-form.

    The codifferential is the adjoint of the exterior derivative:
        δ = (-1)^(nk+1) ⋆ d ⋆

    Maps Ω^(k+1)(K) → Ω^k(K).

    Fundamental property: δ² = 0 (applying δ twice gives zero)

    For k=0 (acting on 1-forms): δ = (-1)^(n×0+1) ⋆₀ d₀ ⋆₁ = -⋆₀ d₀ ⋆₁
    This gives the divergence operator.

    Args:
        k: Degree of the output form (input is (k+1)-form)
        **kwargs: Additional arguments needed for specific k values (e.g., 'edges' for k=0)

    Returns:
        k-form values after applying codifferential

    Example:
        For divergence of a vector field (represented as 1-form on edges):
        >>> div_f = codifferential(k=0, edges=edges)
    """
    if k == 0:
        ### δ: Ω¹ → Ω⁰ (divergence)
        # δ = -⋆₀ d₀ ⋆₁ (for n odd) or +⋆₀ d₀ ⋆₁ (for n even)
        edges = kwargs.get("edges")
        if edges is None:
            raise ValueError("Must provide 'edges' argument for k=0 codifferential")

        # Step 2: Apply d₀ on dual mesh (this requires dual mesh structure)
        # For now, we'll implement this directly using the divergence formula
        # from the DEC paper (lines 1610-1654)

        # This is complex to implement fully, so let's return a placeholder
        # The full implementation requires dual mesh construction
        raise NotImplementedError(
            "Codifferential requires full dual mesh implementation. "
            "Use explicit divergence formula instead."
        )

    else:
        raise NotImplementedError(f"Codifferential for k={k} not yet implemented")
