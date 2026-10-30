Validation and Quality
======================

.. currentmodule:: physicsnemo.mesh.validation

``physicsnemo.mesh.validation`` checks the integrity of a mesh and measures
the quality of its cells. It returns a report of the problems it finds, one
quality value for each cell, and summary statistics.

:bdg-success:`GPU` :bdg-secondary:`any dimension`

.. figure:: ../../img/mesh/mesh_defects.png
   :alt: A grid of ten panels in two rows, each a small square triangle mesh.
         Top row, from left: blue rings on five points where the right half of
         the mesh has its own copies, labeled n_duplicate_vertices: 5. A
         vermillion line through three collinear points, labeled
         n_degenerate_cells: 1. Three vermillion dots outside the mesh, labeled
         n_isolated_vertices: 3. Circular arrows in every cell, one of them
         vermillion and reversed, labeled 1 normal points down. Vermillion
         outlines on the outer edge and on a square hole, labeled 20 boundary
         edges. The bottom row shows the repaired meshes. clean_mesh, n_points
         30 to 25. remove_degenerate_cells, n_cells 33 to 32.
         remove_isolated_points, n_points 28 to 25. fix_orientation,
         n_cells_flipped: 1. fill_holes, one hole filled in light blue and the
         outer loop skipped.
   :align: center
   :width: 100%

   Five defects, each with the call that finds it (top) and the call that
   repairs it (bottom). :func:`validate` finds duplicate points and zero-area
   cells. :func:`compute_mesh_statistics` counts unused points. The normals in
   ``cell_normals`` show a reversed cell, and ``get_boundary_mesh`` shows the
   loops around holes. ``fill_holes`` fills the hole and skips the outer loop,
   which has more edges than ``max_hole_edges``.

.. rubric:: Most Used

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Validate
      :link: physicsnemo.mesh.validation.validate
      :link-type: any
      :shadow: md
      :class-card: sd-card-hover

      ``mesh.validate()`` returns a ``dict`` with ``"valid"`` and a count for
      each problem.

   .. grid-item-card:: Quality Metrics
      :link: physicsnemo.mesh.validation.compute_quality_metrics
      :link-type: any
      :shadow: md
      :class-card: sd-card-hover

      ``mesh.quality_metrics`` gives a ``TensorDict`` with one value for each
      cell.

   .. grid-item-card:: Statistics
      :link: physicsnemo.mesh.validation.compute_mesh_statistics
      :link-type: any
      :shadow: md
      :class-card: sd-card-hover

      ``mesh.statistics`` gives counts and summaries of cell sizes and
      quality.

   .. grid-item-card:: Watertight and Manifold
      :link: boundaries
      :link-type: doc
      :shadow: md
      :class-card: sd-card-hover

      ``mesh.is_watertight()`` and ``mesh.is_manifold()`` check the topology.

Example
-------

The following example validates a sphere and reads its quality metrics and
statistics:

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)

    report = mesh.validate(check_manifoldness=True)
    print(report)

.. code-block:: text
   :caption: Output

   {'valid': True, 'n_out_of_bounds_cells': 0, 'n_duplicate_vertices': 0, 'n_degenerate_cells': 0,
    'is_manifold': True, 'n_non_manifold_edges': 0}

.. code:: python

    quality = mesh.quality_metrics  # TensorDict, batch size (n_cells,)
    print(list(quality.keys()))

.. code-block:: text
   :caption: Output

   ['aspect_ratio', 'edge_length_ratio', 'min_angle', 'max_angle',
    'min_edge_length', 'max_edge_length', 'quality_score']

.. code:: python

    stats = mesh.statistics
    print(stats["n_cells"], stats["n_degenerate_cells"])  # 320 0

The statistics summarize cell areas, edge lengths, aspect ratios, and quality
scores, each as a tuple of minimum, mean, maximum, and standard deviation.

Choosing a Check
----------------

:func:`validate` runs the following checks. Options that are off by default
cost more time:

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Option
     - Default
     - Finds
   * - ``check_out_of_bounds``
     - on
     - Cells whose point indices are out of range
   * - ``check_duplicate_vertices``
     - on
     - Points that coincide within ``tolerance``
   * - ``check_degenerate_cells``
     - on
     - Cells whose area or volume is below ``tolerance`` raised to the
       manifold dimension
   * - ``check_inverted_cells``
     - off
     - Cells with a negative signed volume, on meshes of codimension 0 with a
       manifold dimension of 2 or more. The sign depends on the vertex order
       (see :ref:`mesh-orientation-conventions`). Other meshes report ``-1``.
   * - ``check_manifoldness``
     - off
     - Edges that break manifold topology, on meshes with a manifold
       dimension of 2

With ``raise_on_error=True``, :func:`validate` raises ``ValueError`` at the
first problem instead of returning a report. For watertight and manifold
checks of any mesh, refer to :doc:`boundaries`.

.. note::

   ``validate_mesh`` is a compatibility name for :func:`validate`. It is
   pending deprecation.

Summary
-------

.. autosummary::
   :nosignatures:

   validate
   compute_quality_metrics
   compute_mesh_statistics
   validate_mesh

Related Topics
--------------

For more information, refer to the following resources:

- :doc:`repair` to fix the problems that validation finds.
- :doc:`boundaries` for watertight and manifold checks.
- Tutorial: `Quality, Validation, and Repair
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_7_quality_repair.ipynb>`_

API Reference
-------------

.. autofunction:: validate

.. autofunction:: compute_quality_metrics

.. autofunction:: compute_mesh_statistics

.. autofunction:: validate_mesh
