"""Discrete calculus operators for simplicial meshes.

This module implements discrete differential operators using both:
1. Discrete Exterior Calculus (DEC) - rigorous differential geometry framework
2. Weighted Least-Squares (LSQ) reconstruction - standard CFD approach

The DEC implementation follows Desbrun, Hirani, Leok, and Marsden's seminal work
on discrete exterior calculus (arXiv:math/0508341v2).

Key operators:
- Gradient: ∇φ (scalar → vector)
- Divergence: div(v) (vector → scalar)
- Curl: curl(v) (vector → vector, 3D only)
- Laplacian: Δφ (scalar → scalar, Laplace-Beltrami operator)

Both intrinsic (manifold tangent space) and extrinsic (ambient space) derivatives
are supported for manifolds embedded in higher-dimensional spaces.
"""

from physicsnemo.mesh.calculus.derivatives import (
    compute_point_derivatives,
    compute_cell_derivatives,
)

__all__ = [
    "compute_point_derivatives",
    "compute_cell_derivatives",
]
