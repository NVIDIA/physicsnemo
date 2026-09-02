Smoothing
=========

.. currentmodule:: physicsnemo.mesh.smoothing

Mesh smoothing algorithms for improving mesh regularity while preserving
geometric features.

Provides
`Laplacian smoothing <https://en.wikipedia.org/wiki/Laplacian_smoothing>`_,
which iteratively moves each vertex toward a weighted average of its neighbors.
Codimension-one manifolds of dimension at least 2 use cotangent weights; other
supported mesh types use uniform weights. Boundary vertices are held fixed by
default.

.. code:: python

    from physicsnemo.mesh.smoothing import smooth_laplacian
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)
    smoothed = smooth_laplacian(mesh, n_iter=10)

API Reference
-------------

.. automodule:: physicsnemo.mesh.smoothing
   :members:
   :show-inheritance:
