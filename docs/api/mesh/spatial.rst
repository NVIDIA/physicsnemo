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

Signed Distance Field
---------------------

:func:`signed_distance_field_mesh` computes the signed distance from a set of
query points to a triangle surface mesh, together with the closest point on the
surface for each query. It is a mesh-native, pure-PyTorch implementation that
reuses the spatial acceleration structures in this module, so it runs
identically on CPU and GPU.

The unsigned distance and closest point come from a bounded-stack
depth-first nearest-triangle search over the :class:`BVH` (a single-kernel
Triton traversal is used on CUDA when available, otherwise a pure-PyTorch DFS).
The **sign** is determined by one of two methods, selected with
``use_sign_winding_number``:

- ``False`` (default): the angle-weighted pseudo-normal of the nearest face.
  This is fast and robust for **watertight** meshes.
- ``True``: the generalized winding number (Jacobson et al., 2013), evaluated
  with a :class:`ClusterTree` dual-tree Barnes-Hut summation over the mesh. This
  is robust for **non-watertight / self-intersecting** ("soup") geometry and
  scales as :math:`O(N_\text{query} \log N_\text{faces})`.

.. code:: python

    import torch
    from physicsnemo.mesh.spatial import signed_distance_field_mesh

    # A triangle surface mesh: (n_vertices, 3) coords + flattened connectivity.
    vertices = torch.randn(500, 3)
    faces = torch.randint(0, 500, (3000,))  # (3 * n_faces,) or (n_faces, 3)

    query = torch.randn(10000, 3)
    sdf, hit_points = signed_distance_field_mesh(
        vertices, faces, query, use_sign_winding_number=True
    )
    # sdf: (10000,) signed distances; hit_points: (10000, 3) closest surface points.

API Reference
-------------

.. automodule:: physicsnemo.mesh.spatial
   :members:
   :show-inheritance:
