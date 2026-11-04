Discrete Calculus
=================

.. currentmodule:: physicsnemo.mesh.calculus

``physicsnemo.mesh.calculus`` computes derivatives of fields on a mesh and
integrates fields over it. Each operator takes point or cell values, by name or
as a tensor, and returns a tensor.

:bdg-success:`GPU` :bdg-info:`autograd` :bdg-secondary:`any dimension` :bdg-warning:`curl in 3D only`

.. figure:: ../../img/mesh/calculus_lsq_vs_dec.png
   :alt: Two square maps of the gradient error, one for least squares (LSQ)
         and one for discrete exterior calculus (DEC),
         dark inside with a bright rim on the boundary. Beside them, a log-log
         plot of the error against the mesh spacing: four parallel lines (LSQ
         and DEC, at interior and boundary vertices) next to a slope-1
         reference line. The boundary lines lie five to eight times higher
         than the interior lines.
   :align: center
   :width: 100%

   Error of the least squares (LSQ) and discrete exterior calculus (DEC)
   vertex gradients of a smooth field on a jittered grid. Both methods
   converge at first order on this grid, and the error at boundary vertices
   stays five to eight times larger.

.. rubric:: Most Used

.. grid:: 1 2 4 4
   :gutter: 2

   .. grid-item-card:: Gradient
      :link: physicsnemo.mesh.mesh.Mesh.gradient
      :link-type: any
      :shadow: md
      :class-card: sd-card-hover

      :math:`\nabla\varphi`, scalar to vector

      ``mesh.gradient("T")``

   .. grid-item-card:: Divergence
      :link: physicsnemo.mesh.mesh.Mesh.divergence
      :link-type: any
      :shadow: md
      :class-card: sd-card-hover

      :math:`\nabla\cdot\mathbf{v}`, vector to scalar

      ``mesh.divergence("v")``

   .. grid-item-card:: Curl
      :link: physicsnemo.mesh.mesh.Mesh.curl
      :link-type: any
      :shadow: md
      :class-card: sd-card-hover

      :math:`\nabla\times\mathbf{v}`, vector to vector, in 3D

      ``mesh.curl("v")``

   .. grid-item-card:: Laplacian
      :link: physicsnemo.mesh.mesh.Mesh.laplacian
      :link-type: any
      :shadow: md
      :class-card: sd-card-hover

      :math:`\Delta\varphi = \nabla\cdot\nabla\varphi`, scalar to scalar

      ``mesh.laplacian("T")``

Example
-------

The following example computes derivatives of fields on a tetrahedral cube.
Select a method tab. The rest of the example is the same for both:

.. tab-set::
   :sync-group: method

   .. tab-item:: LSQ
      :sync: lsq

      .. code:: python

          from physicsnemo.mesh.primitives.volumes import cube_volume

          mesh = cube_volume.load(subdivisions=4)  # tetrahedra in 3D

          # The gradient of T = x + 2y is (1, 2, 0). The Mesh method stores it as "T_gradient".
          mesh.point_data["T"] = mesh.points[:, 0] + 2 * mesh.points[:, 1]
          mesh = mesh.compute_point_derivatives(keys="T", method="lsq")
          print(mesh.point_data["T_gradient"][0])  # about (1, 2, 0)

   .. tab-item:: DEC
      :sync: dec

      .. code:: python

          from physicsnemo.mesh.primitives.volumes import cube_volume

          mesh = cube_volume.load(subdivisions=4)  # tetrahedra in 3D

          # The gradient of T = x + 2y is (1, 2, 0). The Mesh method stores it as "T_gradient".
          mesh.point_data["T"] = mesh.points[:, 0] + 2 * mesh.points[:, 1]
          mesh = mesh.compute_point_derivatives(keys="T", method="dec")
          print(mesh.point_data["T_gradient"][0])  # about (1, 2, 0)

.. code:: python

    # v = (x, y, z) has divergence 3 and curl 0.
    mesh.point_data["v"] = mesh.points.clone()
    div_v = mesh.divergence("v")  # (n_points,), about 3
    curl_v = mesh.curl("v")  # (n_points, 3), about 0

    # The DEC Laplacian of u = x^2 + y^2 + z^2 is 6 at interior points.
    mesh.point_data["u"] = (mesh.points**2).sum(dim=-1)
    lap_u = mesh.laplacian("u")  # (n_points,)

    # The functions in this module take tensors and return the same values.
    from physicsnemo.mesh.calculus import compute_divergence_points_lsq

    div_v = compute_divergence_points_lsq(mesh, mesh.points)

On a curved surface or curve, an intrinsic gradient lies in the tangent space
of the mesh, and an extrinsic gradient lies in the ambient space. Select one
with ``gradient_type="intrinsic"`` (the default), ``"extrinsic"``, or
``"both"``.

Choosing a Method
-----------------

Two methods compute the operators:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: LSQ, the Default
      :shadow: md

      Weighted least squares fits a linear function to each point's
      neighborhood, as in finite-volume computational fluid dynamics (CFD).
      It supports the following:

      - Gradient, divergence, and curl
      - At points and at cells
      - ``method="lsq"``

      **Watch out:** LSQ has no Laplacian.

   .. grid-item-card:: DEC
      :shadow: md

      Discrete exterior calculus uses the circumcentric dual mesh and the
      Hodge star, after Desbrun, Hirani, Leok, and Marsden
      (`arXiv:math/0508341 <https://arxiv.org/abs/math/0508341>`_). It
      supports the following:

      - Gradient, divergence, and Laplacian
      - At points only
      - ``method="dec"``

      **Watch out:** DEC has no curl and no cell derivatives.

.. figure:: ../../img/mesh/dec_primal_dual.png
   :alt: Two panels on the same jittered triangle mesh. Left, labeled
         method="lsq", a vermillion vertex with blue arrows to its six blue
         neighbors. Right, labeled method="dec", black dots mark the triangle
         circumcenters. A shaded hexagon around the vertex joins the six
         nearest circumcenters. A blue primal edge runs from the vertex to one
         neighbor, and a short green dual edge crosses it, labeled w_ij = 0.50.
   :align: center
   :width: 100%

   The stencils behind the two methods. LSQ fits a linear function to the
   values at a vertex's neighbors. DEC works on the circumcentric dual mesh.
   The dual cell of a vertex joins the circumcenters of its triangles, and its
   area is the dual volume. Each primal edge has a dual edge that joins the
   circumcenters of its two triangles, and the cotangent weight is the ratio of
   their lengths.

Use the following guidelines to choose a function:

- Use :meth:`~physicsnemo.mesh.mesh.Mesh.compute_point_derivatives` or
  :meth:`~physicsnemo.mesh.mesh.Mesh.compute_cell_derivatives` to add
  gradients of named fields to the mesh data.
- Use ``mesh.gradient``, ``mesh.divergence``, ``mesh.curl``, and
  ``mesh.laplacian``, or the ``compute_*`` functions, for a tensor result
  without changing the mesh.
- For the Laplacian, use ``mesh.laplacian`` or
  :func:`compute_laplacian_points_dec`.
- Expect larger errors at boundary points, where the neighborhood is
  one-sided, as the error figure shows. The `User Guide
  <https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh/using_mesh.html>`__
  compares the two methods in detail.

.. warning::

   ``compute_cell_derivatives(..., method="dec")`` raises
   ``NotImplementedError``. Use ``method="lsq"`` for derivatives of
   ``cell_data``.

.. dropdown:: Support by Dimension

   The following table shows where each operator runs, by mesh type:

   .. list-table::
      :header-rows: 1
      :widths: 28 14 14 14 14 16

      * - Operator
        - ``Mesh[1, 2]``
        - ``Mesh[1, 3]``
        - ``Mesh[2, 2]``
        - ``Mesh[2, 3]``
        - ``Mesh[3, 3]``
      * - Gradient, LSQ, points
        - yes
        - yes
        - yes
        - yes
        - yes
      * - Gradient, LSQ, cells
        - yes
        - extrinsic only
        - yes
        - yes
        - yes
      * - Gradient, DEC, points
        - yes
        - yes
        - yes
        - yes
        - yes
      * - Divergence, LSQ, points and cells
        - yes
        - yes
        - yes
        - yes
        - yes
      * - Divergence, DEC, points
        - yes
        - yes
        - yes
        - yes
        - yes
      * - Curl, LSQ, points and cells
        - no
        - yes
        - no
        - yes
        - yes
      * - Laplacian, DEC, points
        - yes
        - yes
        - yes
        - yes
        - yes

Effective Measures and Integration
----------------------------------

The reserved ``_effective_measure`` field stores one complete integration
measure for each cell or point in ``cell_data`` or ``point_data``. Read these
with ``cell_measures(mesh)`` or ``point_measures(mesh)``. They already include
any geometric contribution and sampling correction. Two integrals use them:

* ``mesh.integrate(field, data_source="cells")`` integrates piecewise-constant
  cell values. ``data_source="points"`` integrates piecewise-linear vertex
  values over the same cells. Both use cell measures, which default to the
  geometric simplex measures.
* ``mesh.integrate_samples(field)`` sums independent point samples times their
  point measures, regardless of connectivity. It requires explicit point
  measures. For counting measure, use an ordinary sum.

Use ``set_cell_measures`` or ``set_point_measures`` to assign measures. Point
measures require their represented ``dimension``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``dimension``
     - Measure
   * - 0
     - Counting
   * - 1
     - Length
   * - 2
     - Area
   * - 3
     - Volume

``scale_measures`` multiplies existing measures by a scalar or a factor for
each cell or point, such as an inverse sampling probability. Raw slicing does
not apply a sampling correction.

Converting cells to centroid samples transfers their measures automatically:

.. code:: python

    from physicsnemo.mesh.primitives.planar import unit_square

    square = unit_square.load(subdivisions=3)
    queries = square.to_point_cloud(point_source="cell_centroids")
    values = queries.points[:, 0]  # Integrate f(x, y) = x over the unit square.
    print(queries.integrate_samples(values))  # tensor(0.5000)

For vertex quadrature, ``lumped_point_measures(mesh)`` distributes each cell's
measure equally among its vertices without modifying the mesh. For finite
fields, these weights reproduce piecewise-linear integration.

Refer to :doc:`transformations` for the geometric changes that keep reference
measures.

Summary
-------

.. rubric:: Derivatives

.. autosummary::
   :nosignatures:

   compute_point_derivatives
   compute_cell_derivatives
   compute_gradient_points_lsq
   compute_gradient_points_dec
   compute_gradient_cells_lsq
   compute_divergence_points_lsq
   compute_divergence_points_dec
   compute_divergence_cells_lsq
   compute_curl_points_lsq
   compute_curl_cells_lsq
   compute_laplacian_points_dec

.. rubric:: Integration and Measures

.. autosummary::
   :nosignatures:

   integrate
   integrate_cell_data
   integrate_point_data
   integrate_samples
   integrate_flux
   integrate_moment
   cell_measures
   point_measures
   set_cell_measures
   set_point_measures
   scale_measures
   lumped_point_measures

Related Topics
--------------

For more information, refer to the following resources:

- :doc:`curvature` and :doc:`geometry`.
- User Guide: `Using Mesh
  <https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh/using_mesh.html>`_
- Tutorial: `Discrete Calculus and Curvature
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_5_calculus.ipynb>`_

API Reference
-------------

Derivatives
^^^^^^^^^^^

.. autofunction:: compute_point_derivatives

.. autofunction:: compute_cell_derivatives

.. autofunction:: compute_gradient_points_lsq

.. autofunction:: compute_gradient_points_dec

.. autofunction:: compute_gradient_cells_lsq

.. autofunction:: compute_divergence_points_lsq

.. autofunction:: compute_divergence_points_dec

.. autofunction:: compute_divergence_cells_lsq

.. autofunction:: compute_curl_points_lsq

.. autofunction:: compute_curl_cells_lsq

.. autofunction:: compute_laplacian_points_dec

Integration and Measures
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: integrate

.. autofunction:: integrate_cell_data

.. autofunction:: integrate_point_data

.. autofunction:: integrate_samples

.. autofunction:: integrate_flux

.. autofunction:: integrate_moment

.. autofunction:: cell_measures

.. autofunction:: point_measures

.. autofunction:: set_cell_measures

.. autofunction:: set_point_measures

.. autofunction:: scale_measures

.. autofunction:: lumped_point_measures

.. py:data:: EFFECTIVE_MEASURE_KEY
   :value: "_effective_measure"

   The reserved ``cell_data`` and ``point_data`` key that stores measures.

.. py:data:: POINT_MEASURE_DIMENSION_KEY
   :value: "_point_measure_dimension"

   The reserved ``global_data`` key that stores the dimension of point
   measures.
