Sampling and Interpolation
==========================

.. currentmodule:: physicsnemo.mesh.sampling

This module provides two categories of functionality:

**Random point sampling**
    Generate random points distributed over the surface (or volume) of a mesh.
    Points are sampled using the
    `Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_
    to produce uniform random barycentric coordinates within each cell. Sampling
    can be uniform per cell or weighted by cell area/volume.

**Data interpolation at query points**
    Given a set of arbitrary query points, find which mesh cell contains each
    point (via BVH-accelerated search), compute
    `barycentric coordinates <https://en.wikipedia.org/wiki/Barycentric_coordinate_system>`_,
    and interpolate point-level or cell-level data to the query locations.

Both capabilities are also accessible as methods on
:class:`~physicsnemo.mesh.mesh.Mesh`:
``sample_random_points_on_cells()`` and ``sample_data_at_points()``.

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3)

    # Sample 10000 random points on the surface
    sampled_mesh = mesh.sample_random_points_on_cells(n_points=10000)
    print(sampled_mesh.points.shape)  # (10000, 3)

    # Interpolate data at arbitrary query points
    import torch
    query_points = torch.randn(500, 3)
    mesh.point_data["height"] = mesh.points[:, 2]
    result = mesh.sample_data_at_points(query_points, data_source="points")

API Reference
-------------

.. automodule:: physicsnemo.mesh.sampling
   :members:
   :show-inheritance:
