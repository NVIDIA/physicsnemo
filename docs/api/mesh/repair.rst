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

    from physicsnemo.mesh.repair import repair_mesh, clean_mesh

    # Quick cleanup
    clean = mesh.clean()

    # Full repair pipeline
    repaired = repair_mesh(mesh)

    # Individual operations
    from physicsnemo.mesh.repair import (
        merge_duplicate_points,
        remove_degenerate_cells,
        fix_orientation,
        fill_holes,
    )
    mesh = merge_duplicate_points(mesh)
    mesh = remove_degenerate_cells(mesh)
    mesh = fix_orientation(mesh)
    mesh = fill_holes(mesh)

API Reference
-------------

.. automodule:: physicsnemo.mesh.repair
   :members:
   :show-inheritance:
