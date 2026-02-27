Geometry
========

.. currentmodule:: physicsnemo.mesh.geometry

Fundamental geometric primitives shared across the codebase. These functions
compute quantities that underlie both the discrete calculus operators and the
curvature computations:

- **Interior angles** of each cell at each vertex
- **Circumcenters** of each cell (used by DEC for the dual mesh)
- **Dual volumes** (Voronoi areas/volumes around vertices and edges)
- **Cotangent weights** (the standard FEM/DEC edge weighting)

Most users will not call these functions directly -- they are invoked
internally by the calculus and curvature modules, and their results are
cached on the :class:`~physicsnemo.mesh.mesh.Mesh` object. They are
exposed here for advanced use cases such as custom DEC operators or
specialized FEM assembly.

.. code:: python

    from physicsnemo.mesh.geometry import (
        compute_vertex_angles,
        compute_circumcenters,
        compute_dual_volumes_0,
        compute_cotan_weights_fem,
    )
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)

    angles = compute_vertex_angles(mesh)           # (n_cells, n_manifold_dims + 1)
    circumcenters = compute_circumcenters(mesh)    # (n_cells, n_spatial_dims)
    dual_areas = compute_dual_volumes_0(mesh)      # (n_points,)
    cotan_w = compute_cotan_weights_fem(mesh)      # (n_edges,)

API Reference
-------------

.. automodule:: physicsnemo.mesh.geometry
   :members:
   :show-inheritance:
