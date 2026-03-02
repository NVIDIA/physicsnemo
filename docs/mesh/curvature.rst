Curvature
=========

.. currentmodule:: physicsnemo.mesh.curvature

Discrete differential geometry tools for computing intrinsic and extrinsic
curvatures on simplicial manifolds.

Two kinds of curvature are provided:

**Gaussian Curvature** (intrinsic)
    Computed through the
    `angle defect <https://en.wikipedia.org/wiki/Angular_defect>`_ method:
    :math:`K_i = (\theta_{\text{full}} - \sum \theta_{ij}) / A_i^*`, where
    :math:`A_i^*` is the dual (Voronoi) area around vertex :math:`i`.
    Gaussian curvature is an intrinsic property (Gauss's Theorema Egregium),
    so it works for any codimension. Available at both vertices and cells.

**Mean Curvature** (extrinsic)
    Computed through the
    `cotangent Laplacian <https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Mesh_Laplacians>`_
    method: :math:`H_i = \|L \mathbf{x}\|_i / (2 A_i^*)`. Requires
    codimension 1 (the mesh must have well-defined normal vectors).

Both curvatures are also accessible as cached properties on the
:class:`~physicsnemo.mesh.mesh.Mesh` class (``mesh.gaussian_curvature_vertices``,
``mesh.mean_curvature_vertices``).

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3)

    # Via standalone functions
    from physicsnemo.mesh.curvature import gaussian_curvature_vertices, mean_curvature_vertices
    K = gaussian_curvature_vertices(mesh)
    H = mean_curvature_vertices(mesh)

    # Or via Mesh properties (equivalent, cached)
    K = mesh.gaussian_curvature_vertices
    H = mesh.mean_curvature_vertices

API Reference
-------------

.. automodule:: physicsnemo.mesh.curvature
   :members:
   :show-inheritance:
