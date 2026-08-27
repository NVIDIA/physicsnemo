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
corresponding submodules. Each pair shares one canonical function, and
normal Python descriptor binding supplies the instance as the standalone
function's ``mesh`` argument.

To construct a triangle mesh from a surface mesh whose cells are arbitrary
polygons -- a "polygon soup" (see :doc:`tessellation`) -- use
:meth:`Mesh.from_polygons`.

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

Cache-Aware Functional Updates
------------------------------

Use the cache-aware replacement methods when creating a mesh that keeps the
same point or cell indexing. Each method returns a new ``Mesh`` with independent
``TensorDict`` containers while sharing their tensor leaves.

.. list-table::
   :header-rows: 1

   * - Method
     - Intended change
     - Default cache policy
   * - :meth:`Mesh.with_data`
     - Replace point, cell, or global field data
     - Retain every geometry and topology cache
   * - :meth:`Mesh.with_points`
     - Replace coordinates without changing point indexing or connectivity
     - Retain topology and clear geometry caches
   * - :meth:`Mesh.with_cells`
     - Replace connectivity without changing cell indexing or simplex type
     - Clear every cache
   * - :meth:`Mesh.strip_caches`
     - Remove cached values without changing the mesh
     - Clear every cache

The ``keep`` override accepts any of the following:

- A top-level string such as ``"topology"``
- A single nested-key tuple such as ``("cell", "areas")``
- A sequence of keys

Use a list when retaining multiple keys, for example
``["topology", ("cell", "areas")]``. Retaining a cache that depends on replaced
geometry or connectivity is an expert operation. You are responsible for the
validity of the retained cache.

.. autoclass:: Mesh
   :members:
   :show-inheritance:

DomainMesh
----------

.. currentmodule:: physicsnemo.mesh.domain_mesh

The :class:`DomainMesh` class groups an interior mesh with named boundary
meshes and domain-level data. Operations such as
:meth:`~physicsnemo.mesh.domain_mesh.DomainMesh.morph` and
:meth:`~physicsnemo.mesh.domain_mesh.DomainMesh.radial_basis_function_deform`
apply one consistent geometry change to every component and return a new
domain.

.. autoclass:: DomainMesh
   :members:
   :show-inheritance:
