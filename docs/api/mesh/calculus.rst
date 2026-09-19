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

The reserved ``_effective_measure`` field stores the complete measure associated
with each cell or point sample. It has shape ``(n_cells,)`` in ``cell_data`` or
``(n_points,)`` in ``point_data``. It is never a dimensionless correction that
must still be multiplied by a geometric area or volume.

``cell_measures(mesh)`` returns the explicit cell measures, or the geometric
simplex measures when the field is absent. ``point_measures(mesh)`` requires
explicit point measures: it never changes its interpretation based on whether
``mesh.cells`` is empty. An ordinary sum is counting measure; it can also be
represented explicitly by installing ones with ``dimension=0``.

.. code:: python

    from physicsnemo.mesh.calculus import (
        cell_measures,
        point_measures,
        scale_measures,
        set_point_measures,
    )

    # A sampling stage retains k of N cells. Prior corrections are preserved.
    sampled = mesh.slice_cells(indices)
    scale_measures(sampled, mesh.n_cells / sampled.n_cells)

    # Transfer complete measures when cells become independent point samples.
    queries = Mesh(points=sampled.cell_centroids)
    set_point_measures(
        queries, cell_measures(sampled), dimension=sampled.n_manifold_dims
    )
    integral = queries.integrate_samples(predictions)

There are two distinct integration operations:

* ``mesh.integrate(field, data_source="cells")`` integrates piecewise-constant
  cell values over the cells. ``data_source="points"`` integrates a
  piecewise-linear vertex field over those same cells. Both use effective
  **cell** measures. Vertex fields keep the existing rule that a NaN vertex
  invalidates its incident cell's contribution when ``nan_policy="omit"``.
* ``mesh.integrate_samples(field)`` sums independent point samples times explicit
  **point** measures, with no dependence on connectivity. NaN omission applies
  independently to each sample contribution. Missing point measures raise an
  error; there is no implicit counting or geometric fallback.

``lumped_point_measures(mesh)`` explicitly constructs vertex quadrature by
sharing each cell's effective measure equally among its vertices. For finite
nodal fields it reproduces piecewise-linear integration. Calling it does not
modify the mesh or change either integration rule.

Measure lifecycle
~~~~~~~~~~~~~~~~~

All storage and reweighting helpers live in ``physicsnemo.mesh.calculus.measure``.
``set_cell_measures`` and ``set_point_measures`` assign complete measures;
``scale_measures`` multiplies existing measures by a scalar or per-entity factor.
Sampling uses the latter with inverse inclusion probabilities. Raw slicing is a
restriction and does not apply a sampling correction. Serialization and device
transfers preserve the fields.

``mesh.to_point_cloud(point_source="cell_centroids")`` and ``mesh.to_dual_graph``
transfer each cell's complete measure to its centroid, including the represented
dimension. Dual-graph edges retain their own geometric length measure; the
original surface or volume measure is associated with the graph's points.

Point measures carry a scalar ``_point_measure_dimension`` in their mesh's
``global_data``. ``set_point_measures`` writes this metadata: 0 for counting, 1
for length, 2 for area, and 3 for volume. It describes the represented measure,
not the point cloud's topological dimension. Merging point quadrature requires
matching measure dimensions and preserves this scalar metadata.

Rigid transformations preserve measures. Uniform scaling by ``s`` multiplies
measures of dimension ``d`` by ``abs(s)**d``. Cell geometry changes preserve the
ratio of represented to geometric measure. Subdivision transfers that ratio to
children; linear subdivision therefore conserves each parent's total measure.
A nonzero measure on a geometrically degenerate cell needs explicit replacement
measures when its geometry changes.

For points, full-dimensional measures also support general square linear maps
through their absolute determinant. Anisotropic transformations of embedded
surface/curve quadrature require support geometry that a point cloud does not
contain, and are rejected. Transform the source cells before creating those
samples, or explicitly retain reference measures with
``mesh.with_points(new_points, preserve_measures=True)``. This policy also serves
coordinate normalization where measures deliberately remain in reference units.

Generic field interpolation excludes effective measures. Subdivision of explicit
point quadrature and remeshing of explicit cell/point quadrature require an
explicit conservative transfer or replacement measures; ordinary field
interpolation cannot provide one.

This pre-release API replaces ``_measure_weights`` and the datapipes-specific
target quadrature key. Files using the old cell multiplier must be regenerated
or converted to complete measures before use. Existing geometric areas remain
geometric: producers must not override area caches to store represented areas.

For a legacy mesh, convert the stored multiplier once and remove its old key:

.. code:: python

    from physicsnemo.mesh.calculus import set_cell_measures, set_point_measures

    if "_measure_weights" in mesh.cell_data:
        weights = mesh.cell_data.pop("_measure_weights")
        set_cell_measures(mesh, mesh.cell_areas * weights)

    # Legacy centroid query measures already contain the geometric contribution.
    # Use the dimension of the source cells (2 here for a surface), not the
    # zero-dimensional topology of the query cloud.
    if "_target_quadrature_measure" in mesh.point_data:
        measures = mesh.point_data.pop("_target_quadrature_measure")
        set_point_measures(mesh, measures, dimension=2)

Update field mappings to ``cell_data._effective_measure`` or
``point_data._effective_measure`` as appropriate. Do not multiply these complete
measures by geometric areas a second time.

API Reference
-------------

.. automodule:: physicsnemo.mesh.calculus
   :members:
   :show-inheritance:
