Tessellation
============

.. currentmodule:: physicsnemo.mesh.tessellation

Decompose non-simplicial cells into the simplices that the
:class:`~physicsnemo.mesh.mesh.Mesh` data structure stores. Currently this
provides polygon-soup triangulation (:func:`triangulate_polygons`): a
vectorized vertex-0 fan for convex polygons and `ear clipping
<https://en.wikipedia.org/wiki/Polygon_triangulation>`_ for the rare non-convex
ones.

Handling non-convex polygons correctly matters for any unsigned-area-weighted
quantity (wall-shear / viscous force integration, or total wetted area): the
signed *vector* area of a vertex-0 fan telescopes to the polygon's regardless
of convexity, but the sum of *unsigned* triangle areas does not.

Every ``k``-gon yields exactly ``k - 2`` triangles, so per-polygon data is
broadcast to the output identically in both paths using the returned
``parent_index``.

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.tessellation import triangulate_polygons

    # A polygon soup in the flat VTK layout: one quad (vertices 0-3).
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    connectivity = torch.tensor([0, 1, 2, 3])
    offsets = torch.tensor([4])

    # Low-level: triangle connectivity plus the parent-polygon index.
    result = triangulate_polygons(points, connectivity, offsets)
    result.cells          # tensor([[0, 1, 2], [0, 2, 3]])
    result.parent_index   # tensor([0, 0]); broadcast data via cell_data[parent_index]

    # High-level: build a Mesh directly, broadcasting per-polygon cell data.
    mesh = Mesh.from_polygons(
        points, connectivity, offsets, cell_data={"pressure": torch.tensor([2.5])}
    )

A :class:`~physicsnemo.mesh.mesh.Mesh` can also be constructed in one step with
:meth:`~physicsnemo.mesh.mesh.Mesh.from_polygons`.

API Reference
-------------

.. automodule:: physicsnemo.mesh.tessellation
   :members:
   :show-inheritance:
