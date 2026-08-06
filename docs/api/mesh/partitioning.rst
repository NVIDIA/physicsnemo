Cell Partitioning
=================

.. currentmodule:: physicsnemo.mesh.remeshing

Cell partitioning assigns each existing mesh cell to its nearest seed and
aggregates area, normal, and centroid data for every cluster. It preserves the
input topology. For topology reconstruction with new vertices and cells, use
:func:`physicsnemo.geometry.remeshing.remesh`.

API Reference
-------------

.. autoclass:: CellPartition
   :members:

.. autofunction:: partition_cells
