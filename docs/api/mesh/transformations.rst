Transformations and Projections
===============================

Geometric Transformations
-------------------------

.. currentmodule:: physicsnemo.mesh.transformations.geometric

Linear and affine transformations on mesh geometry. Each function returns a new
:class:`~physicsnemo.mesh.mesh.Mesh` with transformed point coordinates and
appropriately invalidated caches. (Any cached quantities, such as normals and
areas, are automatically recomputed on next access.)

All transformations are also available as methods on
:class:`~physicsnemo.mesh.mesh.Mesh`.

.. code:: python

    import numpy as np
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)

    # Via Mesh methods
    translated = mesh.translate([1.0, 0.0, 0.0])
    rotated = mesh.rotate(axis=[0, 0, 1], angle=np.pi / 4)
    scaled = mesh.scale(2.0)
    scaled_aniso = mesh.scale([2.0, 1.0, 0.5])

    # Arbitrary linear transform
    import torch
    matrix = torch.eye(3) * 2
    transformed = mesh.transform(matrix)

.. automodule:: physicsnemo.mesh.transformations.geometric
   :members:
   :show-inheritance:

Projections
-----------

.. currentmodule:: physicsnemo.mesh.projections

Spatial dimension manipulation -- changing the embedding dimension of a mesh
without altering its manifold dimension.

- :func:`embed` -- add spatial dimensions (non-destructive; for example, 2D mesh to 3D
  by appending zero coordinates)
- :func:`extrude` -- sweep a manifold to create a mesh one dimension higher
  (for example, a triangle mesh extruded to a prism mesh)
- :func:`project` -- reduce spatial dimensions (lossy; drops coordinate axes)

.. automodule:: physicsnemo.mesh.projections
   :members:
   :show-inheritance:
