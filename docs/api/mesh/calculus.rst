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

Effective measures and integration
----------------------------------

The reserved ``_effective_measure`` field stores one complete integration
measure per cell or point in ``cell_data`` or ``point_data``. Read these with
``cell_measures(mesh)`` or ``point_measures(mesh)``; they already include any
geometric contribution and sampling correction.

* ``mesh.integrate(field, data_source="cells")`` integrates piecewise-constant
  cell values; ``data_source="points"`` integrates piecewise-linear vertex
  values over the same cells. Both use cell measures, which default to the
  geometric simplex measures.
* ``mesh.integrate_samples(field)`` sums independent point samples times their
  point measures, regardless of connectivity. Explicit point measures are
  required. For counting measure, use an ordinary sum.

Use ``set_cell_measures`` or ``set_point_measures`` to assign measures. Point
measures require their represented ``dimension``: 0 for counting, 1 for length,
2 for area, or 3 for volume. ``scale_measures`` multiplies existing measures by
a scalar or per-entity factor, such as an inverse sampling probability. Raw
slicing does not apply a sampling correction.

Converting cells to centroid samples transfers their measures automatically:

.. code:: python

    queries = mesh.to_point_cloud(point_source="cell_centroids")
    values = queries.points[:, 0]  # Integrate f(x, ...) = x.
    integral = queries.integrate_samples(values)

For vertex quadrature, ``lumped_point_measures(mesh)`` distributes each cell's
measure equally among its vertices without modifying the mesh. For finite
fields, these weights reproduce piecewise-linear integration.

See :doc:`transformations` for supported geometric changes and explicit
preservation of reference measures.

API Reference
-------------

.. automodule:: physicsnemo.mesh.calculus
   :members:
   :show-inheritance:
