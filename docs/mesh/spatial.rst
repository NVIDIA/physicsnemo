Spatial Queries
===============

.. currentmodule:: physicsnemo.mesh.spatial

This module provides spatial acceleration structures for efficient geometric
queries on large meshes.

The :class:`BVH` (Bounding Volume Hierarchy) is an axis-aligned bounding box
tree built over mesh cells. It accelerates two key operations:

- **Point containment**: given a set of query points, find which mesh cell
  (if any) contains each point
- **Nearest-cell search**: find the closest cell to each query point

The BVH is used internally by the sampling module
(:func:`~physicsnemo.mesh.sampling.sample_data_at_points`,
:func:`~physicsnemo.mesh.sampling.find_containing_cells`) to avoid brute-force
search over all cells.

.. code:: python

    import torch
    from physicsnemo.mesh.spatial import BVH
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3)
    bvh = BVH.from_mesh(mesh)

    query_points = torch.randn(1000, 3)
    candidate_cells = bvh.find_candidate_cells(query_points)

API Reference
-------------

.. automodule:: physicsnemo.mesh.spatial
   :members:
   :show-inheritance:
