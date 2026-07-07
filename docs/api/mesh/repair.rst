Repair
======

.. currentmodule:: physicsnemo.mesh.repair

Tools for fixing common mesh problems. Individual repair operations are
available as standalone functions, and :func:`repair_mesh` chains them
into a single pipeline.

Available operations:

- **Merge duplicate points**: collapse vertices within a tolerance
- **Remove duplicate cells**: eliminate cells with identical vertex sets
- **Remove degenerate cells**: remove cells with zero area/volume
- **Remove unused/isolated points**: clean up unreferenced vertices
- **Fix orientation**: ensure consistent face winding
- **Fill holes**: close open boundaries

The all-in-one :func:`clean_mesh` function (also accessible as
``mesh.clean()``) applies the most common subset of these operations.
For full control, use :func:`repair_mesh` or call individual functions.

.. code:: python

    from physicsnemo.mesh.repair import clean_mesh, repair_mesh

    # Quick cleanup through the Mesh convenience API returns only the mesh
    clean = mesh.clean()

    # Standalone cleanup and the full repair pipeline also return statistics
    clean, clean_stats = clean_mesh(mesh)
    repaired, repair_stats = repair_mesh(mesh)

    # Individual operations
    from physicsnemo.mesh.repair import (
        remove_degenerate_cells,
        fix_orientation,
        fill_holes,
    )
    mesh, degenerate_stats = remove_degenerate_cells(mesh)
    mesh, orientation_stats = fix_orientation(mesh)
    mesh, hole_stats = fill_holes(mesh)

``merge_duplicate_points`` is a lower-level tensor API accepting separate
``points``, ``cells``, and ``point_data`` arguments. Use ``mesh.clean()`` or
``clean_mesh()`` for mesh-level duplicate-point cleanup.

API Reference
-------------

.. automodule:: physicsnemo.mesh.repair
   :members:
   :show-inheritance:
