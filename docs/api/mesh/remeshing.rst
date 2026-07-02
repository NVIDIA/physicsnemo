Remeshing
=========

.. currentmodule:: physicsnemo.mesh.remeshing

Uniform remeshing via the ACVD (Approximate Centroidal Voronoi Diagram)
clustering algorithm. Given a target number of clusters, the algorithm
redistributes mesh vertices to produce a more uniform cell distribution.

The current implementation wraps ``pyacvd`` and therefore supports only
triangle surfaces (2D manifolds) embedded in 3D:

1. Weight vertices by incident cell areas
2. Initialize clusters via area-based region growing
3. Remove spatially isolated cluster regions
4. Reconstruct a simplified mesh from cluster adjacency

``n_clusters`` controls the approximate number of output vertices (one per
cluster), not the number of triangles. The output cell count follows from the
surface topology and is commonly close to twice the vertex count for a closed
triangular surface.

.. code:: python

    from physicsnemo.mesh.remeshing import remesh
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3)
    remeshed = remesh(mesh, n_clusters=100)
    print(remeshed.n_points)  # approximately 100 cluster centroids

API Reference
-------------

.. automodule:: physicsnemo.mesh.remeshing
   :members:
   :show-inheritance:
