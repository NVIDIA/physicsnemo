Tessellation
============

.. currentmodule:: physicsnemo.mesh.tessellation

Decompose non-simplicial cells into the simplices that the
:class:`~physicsnemo.mesh.mesh.Mesh` data structure stores. Currently this
provides triangulation of a *polygon soup* (:func:`triangulate`): a vectorized
vertex-0 fan for convex polygons and `ear clipping
<https://en.wikipedia.org/wiki/Polygon_triangulation>`_ for the rare non-convex
ones.

A *polygon soup* is a flat list of polygonal cells, each given only by its
vertex ring, with no shared-edge connectivity between cells -- the form a
surface mesh takes when read straight from a polygonal face list (e.g. a VTP or
STL file). Equivalently, it is a surface mesh whose cells may be arbitrary
polygons rather than triangles.

.. note::

   This is unrelated to the (now deprecated) ``Tessellation`` *geometry* class
   in PhysicsNeMo-Sym (``physicsnemo.sym.geometry.tessellation``), which
   represents a tessellated STL surface as a solid for sampling
   surface/interior points and signed-distance values. The utilities here
   instead split the polygonal cells of an existing mesh into simplices.

Handling non-convex polygons correctly matters for any unsigned-area-weighted
quantity (wall-shear / viscous force integration, or total wetted area): the
signed *vector* area of a vertex-0 fan telescopes to the polygon's total area regardless
of convexity, but the sum of the *unsigned* triangle areas does not.

Every ``k``-gon yields exactly ``k - 2`` triangles, so per-polygon data is
broadcast to the output identically in both paths using the returned
``parent_index``.

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.neighbors import Adjacency
    from physicsnemo.mesh.tessellation import triangulate

    # A polygon soup as a cell-to-vertex Adjacency (CSR): one quad (vertices 0-3).
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    polygons = Adjacency(offsets=torch.tensor([0, 4]),
                         indices=torch.tensor([0, 1, 2, 3]))

    # Low-level: triangle connectivity plus the parent-polygon index.
    cells, parent_index = triangulate(points, polygons)
    cells          # tensor([[0, 1, 2], [0, 2, 3]])
    parent_index   # tensor([0, 0]); broadcast data via cell_data[parent_index]

    # High-level: build a Mesh directly, broadcasting per-polygon cell data.
    mesh = Mesh.from_polygons(
        points, polygons, cell_data={"pressure": torch.tensor([2.5])}
    )

A :class:`~physicsnemo.mesh.mesh.Mesh` can also be constructed in one step with
:meth:`~physicsnemo.mesh.mesh.Mesh.from_polygons`.

API Reference
-------------

.. automodule:: physicsnemo.mesh.tessellation
   :members:
   :show-inheritance:
