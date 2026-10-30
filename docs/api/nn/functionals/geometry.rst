Geometry Functionals
====================

Point deformation, deformation energies, and surface remeshing moved into
:doc:`PhysicsNeMo-Mesh <../../mesh/index>`, as operations on a ``Mesh``.

Signed Distance Field
---------------------

.. autofunction:: physicsnemo.nn.functional.signed_distance_field

.. rubric:: Visualization

This visualization shows signed-distance values on a 2D slice through the
domain, with the zero level-set contour indicating the implicit surface. The
animation shows a sweep plane through the mesh (left) and corresponding SDF
slice image (right).

.. figure:: /img/nn/functional/geometry/sdf/sdf_slice_overview.png
   :alt: Signed distance field 2D slice visualization
   :width: 90%

.. figure:: /img/nn/functional/geometry/sdf/sdf_slice_sweep.gif
   :alt: Signed distance field z-slice sweep animation
   :width: 70%
