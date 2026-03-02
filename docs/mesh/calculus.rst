Discrete Calculus
=================

.. currentmodule:: physicsnemo.mesh.calculus

This module implements discrete differential operators on simplicial meshes
using two complementary approaches:

1. **Discrete Exterior Calculus (DEC)** -- a rigorous differential-geometry
   framework based on Desbrun, Hirani, Leok, and Marsden's work
   (`arXiv:math/0508341 <https://arxiv.org/abs/math/0508341>`_). DEC operators
   use the primal/dual mesh structure (circumcentric dual volumes, Hodge stars)
   and produce results that satisfy discrete analogues of Stokes' theorem.

2. **Weighted Least-Squares (LSQ)** -- a standard CFD/FEM approach that
   reconstructs derivatives by fitting polynomials to local neighborhoods.
   LSQ methods are more flexible (they work for any manifold/codimension) and
   are generally the recommended default.

Both intrinsic (manifold tangent space) and extrinsic (ambient space)
derivatives are supported for manifolds embedded in higher-dimensional spaces.

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.calculus import (
        compute_gradient_points_lsq,
        compute_divergence_points_lsq,
        compute_curl_points_lsq,
    )

    # Linear scalar field T = x + 2y on a mesh
    mesh.point_data["T"] = mesh.points[:, 0] + 2 * mesh.points[:, 1]

    # Gradient via the Mesh method (wraps compute_gradient_points_lsq)
    mesh = mesh.compute_point_derivatives(keys="T", method="lsq")
    grad_T = mesh.point_data["T_gradient"]  # (n_points, n_spatial_dims)

    # Divergence and curl via standalone functions
    mesh.point_data["velocity"] = mesh.points.clone()
    div_v = compute_divergence_points_lsq(mesh, mesh.point_data["velocity"])
    curl_v = compute_curl_points_lsq(mesh, mesh.point_data["velocity"])  # 3D only

Key Operators
-------------

- **Gradient**: :math:`\nabla\varphi` (scalar :math:`\to` vector)
- **Divergence**: :math:`\operatorname{div}(\mathbf{v})` (vector :math:`\to` scalar)
- **Curl**: :math:`\operatorname{curl}(\mathbf{v})` (vector :math:`\to` vector, 3D only)
- **Laplacian**: :math:`\Delta\varphi` (scalar :math:`\to` scalar, Laplace-Beltrami)

API Reference
-------------

.. automodule:: physicsnemo.mesh.calculus
   :members:
   :show-inheritance:
