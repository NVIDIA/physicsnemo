Smoothing
=========

.. currentmodule:: physicsnemo.mesh.smoothing

Mesh smoothing algorithms for improving mesh regularity while preserving
geometric features.

Provides geometry smoothing with
`Laplacian smoothing <https://en.wikipedia.org/wiki/Laplacian_smoothing>`_,
which iteratively moves each vertex toward a weighted average of its neighbors.
Codimension-one manifolds of dimension at least 2 use cotangent weights; other
supported mesh types use uniform weights. Boundary vertices are held fixed by
default by :func:`smooth_laplacian`. :func:`smooth_point_field` applies
normalized edge averaging to scalar, vector, or tensor data without moving the
mesh; it updates every connected point, including boundary and feature points.

.. code:: python

    import torch
    from physicsnemo.mesh.smoothing import smooth_laplacian, smooth_point_field
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)
    smoothed = smooth_laplacian(mesh, n_iter=10)

    noisy_sensitivity = torch.randn(mesh.n_points, device=mesh.points.device)
    smooth_sensitivity = smooth_point_field(mesh, noisy_sensitivity, n_iter=5)

.. figure:: /img/mesh/smooth_point_field.png
   :alt: A noisy scalar point field before and after normalized mesh smoothing.
   :width: 100%

   Normalized edge-Laplacian smoothing suppresses point-scale oscillations
   while retaining the broader field structure.

API Reference
-------------

.. automodule:: physicsnemo.mesh.smoothing
   :members:
   :show-inheritance:
