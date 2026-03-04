Mesh
====

.. currentmodule:: physicsnemo.mesh.mesh

The :class:`Mesh` class is the central data structure of PhysicsNeMo-Mesh. It is
a `tensorclass <https://pytorch.org/tensordict/stable/reference/tensorclass.html>`_
built on TensorDict, representing an n-dimensional simplicial manifold embedded in
m-dimensional Euclidean space.

A ``Mesh`` stores vertex coordinates (``points``), cell connectivity (``cells``),
and three ``TensorDict`` containers for attaching arbitrary tensor data at the
vertex, cell, and global levels. All tensors move together under ``.to(device)``
calls, and expensive geometric quantities -- centroids, normals, areas, curvature
-- are computed lazily on first access and cached internally.

Most mesh operations (subdivision, derivatives, transformations) are
available both as ``Mesh`` methods and as standalone functions in the
corresponding submodules. The methods are thin wrappers that pass ``self`` to
the standalone functions.

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    cells = torch.tensor([[0, 1, 2]])
    mesh = Mesh(points=points, cells=cells)

    # Geometric properties (lazily computed, cached)
    print(mesh.cell_centroids)   # shape (1, 2)
    print(mesh.cell_areas)       # shape (1,)

    # Attach data and compute derivatives
    mesh.point_data["T"] = torch.tensor([1.0, 2.0, 3.0])
    mesh = mesh.compute_point_derivatives(keys="T", method="lsq")
    print(mesh.point_data["T_gradient"])  # shape (3, 2)

.. autoclass:: Mesh
   :members:
   :show-inheritance:
