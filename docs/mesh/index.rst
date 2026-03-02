PhysicsNeMo Mesh
================

.. currentmodule:: physicsnemo.mesh

GPU-accelerated mesh processing for physical simulation and scientific
computing in any dimension.

Overview
--------

The word "mesh" means different things to different communities:

- **CFD/FEM engineers** think "volume mesh" (3D tetrahedra filling a 3D domain)
- **Graphics programmers** think "surface mesh" (2D triangles in 3D space)
- **Computer vision researchers** think "point cloud" (0D vertices in 3D space)
- **Robotics engineers** think "curves" (1D edges in 2D or 3D space)

PhysicsNeMo-Mesh handles all of these in a unified, dimensionally-generic
framework. More precisely, it operates on arbitrary-dimensional
`simplicial complexes <https://en.wikipedia.org/wiki/Simplicial_complex>`_
embedded in arbitrary-dimensional
`Euclidean spaces <https://en.wikipedia.org/wiki/Euclidean_space>`_:

- 2D triangles in 2D space (planar meshes for 2D simulations)
- 2D triangles in 3D space (surface meshes for graphics and CFD)
- 3D tetrahedra in 3D space (volume meshes for FEM and CFD)
- 1D edges in 3D space (curve meshes for path planning)
- Any n-dimensional manifold in m-dimensional space (where n <= m)

The library depends only on `PyTorch <https://pytorch.org/>`_ and
`TensorDict <https://github.com/pytorch/tensordict>`_ (an official PyTorch
data structure).

What Does "Simplicial" Mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The building block of a ``Mesh`` is a **simplex** - a generalization of the
notion of a triangle or tetrahedron to arbitrary dimensions:

=========  ====================  =========================================
n           Common Name           Description
=========  ====================  =========================================
0-simplex  Point                 A single vertex
1-simplex  Line segment / Edge   Connects 2 points; boundary: 2 points
2-simplex  Triangle              Connects 3 points; boundary: 3 edges
3-simplex  Tetrahedron           Connects 4 points; boundary: 4 triangles
=========  ====================  =========================================

A PhysicsNeMo ``Mesh`` is a collection of simplices of the same dimension
that share vertices. Every simplex in the mesh has the same dimension,
called the **manifold dimension** (``n_manifold_dims``). The dimension of
the ambient space where vertex coordinates live is the **spatial dimension**
(``n_spatial_dims``). For example, a triangle mesh representing a 3D
surface has ``n_spatial_dims=3`` and ``n_manifold_dims=2``.

This restriction to simplicial meshes (as opposed to arbitrary polygonal
meshes) enables rigorous
`discrete exterior calculus <https://en.wikipedia.org/wiki/Discrete_exterior_calculus>`_,
dimension-generic algorithms, and significant performance benefits.


Design Properties
^^^^^^^^^^^^^^^^^

**GPU-accelerated**
    All operations are fully vectorized with PyTorch and run natively on
    CUDA. There are no Python loops over mesh elements. An entire mesh,
    including all attached data, moves between devices with a single
    ``.to("cuda")`` or ``.to("cpu")`` call.

**Autograd-differentiable**
    Most operations integrate seamlessly with
    `PyTorch autograd <https://pytorch.org/docs/stable/autograd.html>`_.
    Gradients flow through mesh construction, geometric computations, and
    field derivative operators, enabling end-to-end differentiable
    simulation and optimization pipelines.

**Dimensionally generic**
    Algorithms are written once for n-dimensional manifolds in m-dimensional
    spaces. The same code that computes normals on a triangle mesh in 3D
    also computes normals on a line mesh in 2D, or on a tetrahedral mesh
    in 3D. Manifold dimension, spatial dimension, and codimension are
    first-class concepts throughout the API.

**Arbitrary-rank tensor fields**
    Field data attached to a mesh is not limited to scalars or vectors. You
    can store tensors of any rank - scalar fields, vector fields, matrix
    fields (for example, stress tensors), or higher-order tensor fields - at any of
    the three levels (per-point, per-cell, or global).

**Nested, structured data via TensorDict**
    All field data is stored in ``TensorDict`` containers, which support
    arbitrarily nested string keys. This allows semantically rich, hierarchical
    data organization (e.g. ``mesh.point_data["boundary_conditions", "inlet",
    "velocity"]``) without flattening everything into a single namespace.


Data Model
----------

The central data structure is the :class:`~physicsnemo.mesh.mesh.Mesh`
tensorclass, defined by five fields:

.. code:: python

    Mesh(
        points: torch.Tensor,      # (n_points, n_spatial_dims)
        cells: torch.Tensor,       # (n_cells, n_manifold_dims + 1), integer dtype
        point_data: TensorDict,    # Per-vertex data
        cell_data: TensorDict,     # Per-cell data
        global_data: TensorDict,   # Mesh-level data
    )

Field data of any rank can be attached at each level:

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    cells = torch.tensor([[0, 1, 2]])
    mesh = Mesh(points=points, cells=cells)

    # Scalar field: shape (n_points,)
    mesh.point_data["temperature"] = torch.tensor([300.0, 350.0, 325.0])

    # Vector field: shape (n_points, n_spatial_dims)
    mesh.point_data["velocity"] = torch.tensor([[1.0, 0.5], [0.8, 1.2], [0.0, 0.9]])

    # Tensor field: shape (n_cells, n_spatial_dims, n_spatial_dims)
    mesh.cell_data["reynolds_stress"] = torch.tensor([[[2.1, 0.3], [0.3, 1.8]]])

The repr shows trailing dimensions for each field:

.. code-block:: text

    Mesh(manifold_dim=2, spatial_dim=2, n_points=3, n_cells=1)
        point_data : {temperature: (), velocity: (2,)}
        cell_data  : {reynolds_stress: (2, 2)}
        global_data: {}

All data moves together under ``.to(device)`` calls. Expensive geometric
quantities (centroids, normals, curvature) are computed lazily and cached
automatically.


Quick Start
-----------

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh

    # Create a triangle in 2D
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    cells = torch.tensor([[0, 1, 2]])
    mesh = Mesh(points=points, cells=cells)

    # Move to GPU and compute derivatives
    mesh_gpu = mesh.to("cuda")
    mesh_gpu.point_data["T"] = mesh_gpu.points[:, 0] + 2 * mesh_gpu.points[:, 1]
    mesh_gpu = mesh_gpu.compute_point_derivatives(keys="T", method="lsq")

    # Load from PyVista (supports STL, VTK, PLY, OBJ, ...)
    from physicsnemo.mesh.io import from_pyvista
    import pyvista as pv
    mesh = from_pyvista(pv.read("geometry.stl"))


Key Features
------------

- **Discrete calculus**: gradient, divergence, curl, and Laplacian via both
  Discrete Exterior Calculus (DEC) and least-squares (LSQ) methods, with
  intrinsic (tangent space) and extrinsic (ambient space) variants
- **Differential geometry**: Gaussian curvature, mean curvature, normals,
  tangent spaces
- **Mesh operations**: subdivision (linear, Loop, Butterfly), smoothing,
  remeshing, repair
- **Spatial queries**: BVH-accelerated point containment and nearest-cell search
- **Topology**: boundary detection, watertight/manifold checking, adjacency
  queries
- **I/O**: bidirectional conversion with PyVista (supporting all formats PyVista
  handles)
- **Visualization**: matplotlib and PyVista rendering backends


Tutorials
---------

Runnable Jupyter notebook tutorials are available in ``examples/minimal/mesh/``:

1. **Getting Started** -- mesh creation, data attachment, GPU usage, autograd
2. **Operations** -- transformations, subdivision, slicing, merging, boundaries
3. **Discrete Calculus** -- gradients, divergence, curl, curvature
4. **Neighbors & Spatial** -- adjacency queries, BVH, sampling, interpolation
5. **Quality & Repair** -- validation, quality metrics, repair pipeline
6. **ML Integration** -- batching, feature extraction, torch.compile, benchmarks


API Reference
-------------

.. toctree::
   :maxdepth: 2

   core
   io
   calculus
   curvature
   geometry
   boundaries
   neighbors
   spatial
   sampling
   transformations
   subdivision
   smoothing
   remeshing
   repair
   validation
   visualization
