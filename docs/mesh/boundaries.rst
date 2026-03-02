Boundaries and Topology
========================

.. currentmodule:: physicsnemo.mesh.boundaries

This module provides three groups of functionality:

**Boundary detection**
    Identify which vertices, edges, or cells lie on the boundary of a mesh.
    A boundary facet is one that appears in exactly one cell (interior facets
    are shared by two cells).

**Facet extraction**
    Extract the (n-1)-dimensional sub-simplices (facets) from an n-dimensional
    mesh. For a triangle mesh, facets are edges; for a tetrahedral mesh, facets
    are triangles. The extraction pipeline deduplicates facets and can aggregate
    per-cell data onto the resulting facet mesh.

**Topology checking**
    Test whether a mesh is
    `watertight <https://en.wikipedia.org/wiki/Watertight_(3D_modeling)>`_
    (no boundary facets) or
    `manifold <https://en.wikipedia.org/wiki/Manifold>`_
    (every facet is shared by at most two cells).

These are also accessible as methods on :class:`~physicsnemo.mesh.mesh.Mesh`:
``get_boundary_mesh()``, ``get_facet_mesh()``, ``is_watertight()``,
``is_manifold()``.

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)

    # Topology checks
    print(mesh.is_watertight())  # True for a closed sphere
    print(mesh.is_manifold())    # True

    # Extract boundary (empty for a closed mesh)
    boundary = mesh.get_boundary_mesh()

    # Extract all edges (facets of a triangle mesh)
    edge_mesh = mesh.get_facet_mesh()

API Reference
-------------

.. automodule:: physicsnemo.mesh.boundaries
   :members:
   :show-inheritance:
