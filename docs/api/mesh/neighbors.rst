Neighbors and Adjacency
=======================

.. currentmodule:: physicsnemo.mesh.neighbors

This module computes topological adjacency relationships between mesh
elements. Rather than using spatial proximity, it indicates the mesh connectivity for which points/cells are connected to which other
points/cells.

Four adjacency types are supported:

- **Point-to-points**: which vertices share an edge (graph neighbors)
- **Point-to-cells**: which cells contain a given vertex (the vertex star)
- **Cell-to-cells**: which cells share a facet
- **Cell-to-points**: which vertices belong to a given cell

All adjacency relationships are returned as :class:`Adjacency` objects, which
encode ragged arrays using an ``(indices, offsets)`` pair. This is the same
sparse format used by PyTorch Geometric's ``edge_index`` and is efficient for
GPU computation. For debugging or interoperability, call ``.to_list()`` to
convert to a Python list of lists.

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)

    # Compute adjacency (also available as Mesh methods)
    p2p = mesh.get_point_to_points_adjacency()
    c2c = mesh.get_cell_to_cells_adjacency()

    # Inspect the ragged structure
    print(p2p.indices.shape)   # (total_neighbor_pairs,)
    print(p2p.offsets.shape)   # (n_points + 1,)

    # Convert to list-of-lists for inspection
    neighbors_of_vertex_0 = p2p.to_list()[0]

API Reference
-------------

.. automodule:: physicsnemo.mesh.neighbors
   :members:
   :show-inheritance:
