Remeshing
=========

.. currentmodule:: physicsnemo.mesh.remeshing

Uniform remeshing via the ACVD (Approximate Centroidal Voronoi Diagram)
clustering algorithm. Given a target number of clusters, the algorithm
redistributes mesh vertices to produce a more uniform cell distribution.

The approach is dimension-agnostic and works for any simplicial manifold:

1. Weight vertices by incident cell areas
2. Initialize clusters via area-based region growing
3. Remove spatially isolated cluster regions
4. Reconstruct a simplified mesh from cluster adjacency

.. note::

   The output mesh may contain a small percentage (~0.5-1%) of non-manifold
   edges, which is inherent to the face-mapping approach. Higher cluster
   counts relative to mesh resolution produce better manifold quality.

.. code:: python

    from physicsnemo.mesh.remeshing import remesh
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3)
    remeshed = remesh(mesh, n_clusters=100)
    print(remeshed.n_cells)  # approximately 200 triangles

API Reference
-------------

.. automodule:: physicsnemo.mesh.remeshing
   :members:
   :show-inheritance:
