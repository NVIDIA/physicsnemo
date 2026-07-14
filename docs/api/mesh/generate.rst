Generate
========

.. currentmodule:: physicsnemo.mesh.generate

Mesh generation from implicit representations: isosurface extraction and
dimension-generic volume meshing.

Choosing a Mesher
-----------------

PhysicsNeMo-Mesh has three generation tools with distinct contracts:

.. list-table::
   :header-rows: 1

   * - Tool
     - Input
     - Output
     - Use When
   * - ``physicsnemo.mesh.tessellation.fill_interior``
     - closed boundary ``Mesh`` (2D today)
     - volume ``Mesh``, **exact** boundary, guaranteed angles,
       deterministic
     - the boundary discretization is the contract (coupling, per-boundary
       BCs, guaranteed worst-element quality)
   * - :func:`mesh_implicit_domain`
     - implicit function :math:`\varphi(x)` (SDF, level set, neural field)
     - volume ``Mesh`` in **any dimension**, boundary approximated to
       :math:`O(h^2)`, GPU-native, differentiable refit
     - geometry is implicit, you need 3D/ND volumes, GPU scale, or
       gradients with regard to shape
   * - :func:`marching_cubes`
     - 3D scalar field on a grid
     - **surface** ``Mesh`` (the isosurface only, no interior)
     - the boundary surface itself is the deliverable (visualization,
       B-rep export)

Volume Meshing of Implicit Domains
----------------------------------

:func:`mesh_implicit_domain` generates a simplex volume mesh of
``{x : phi(x) < 0}`` for an arbitrary implicit function ``phi`` (a signed
distance function, or any level set with a usable gradient, including
neural implicit fields), in *any spatial dimension*, entirely in PyTorch
tensor ops on CPU or CUDA. The meshed set is ``{phi < 0}`` intersected
with the bounding box: where the domain reaches the box, its faces are
honored as boundary, so external-flow "box minus obstacle" domains work
directly. The generator is structurally robust. Every
optimization step is validity-gated, so it always returns a positively
oriented mesh with a closed-manifold boundary. Difficult inputs degrade
element quality (reported in diagnostics), but never prevent the mesh
from being generated. A coverage guard raises an error, rather than
silently dropping geometry, when the domain
has features below the target edge length ``h`` or when coverage cannot
be certified at all (a phi that is NaN inside the box, for example a neural
field queried outside its training range). You can interpolate sharp
corners exactly through ``feature_points``.

:func:`refit_mesh_to_implicit` is the differentiable companion: it
re-projects a mesh's boundary onto ``phi = 0`` with graph-preserving
Newton steps at fixed topology, so gradients flow from mesh coordinates to
shape parameters inside ``phi``. This enables meshing as a
differentiable layer for shape optimization.

.. code:: python

    import torch
    from physicsnemo.mesh.generate import (
        mesh_implicit_domain,
        refit_mesh_to_implicit,
        sdf_difference,
        sdf_sphere,
    )

    # A spherical shell, meshed with tetrahedra on the GPU:
    shell = sdf_difference(sdf_sphere([0.0] * 3, 0.8), sdf_sphere([0.0] * 3, 0.35))
    mesh = mesh_implicit_domain(shell, ([-1] * 3, [1] * 3), h=0.05, device="cuda")

    # Differentiable geometry at fixed topology:
    r = torch.tensor(0.8, requires_grad=True)
    refit = refit_mesh_to_implicit(mesh.to("cpu"), lambda x: x.norm(dim=-1) - r)

The following signed-distance building blocks are provided for
convenience:

- :func:`sdf_sphere`
- :func:`sdf_box`
- :func:`sdf_polygon_2d`
- :func:`sdf_union` (CSG combinator)
- :func:`sdf_intersection` (CSG combinator)
- :func:`sdf_difference` (CSG combinator)
- :func:`project_to_zero_set` (Newton projection onto ``phi = 0``)

Any callable with the signature ``phi(x: (..., d)) -> (...)`` also works.

Isosurface Extraction
---------------------

:func:`marching_cubes` extracts a triangle isosurface from a 3D scalar
field.

API Reference
-------------

.. automodule:: physicsnemo.mesh.generate
   :members:
   :show-inheritance:
