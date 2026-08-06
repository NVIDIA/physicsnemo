PhysicsNeMo Geometry
====================

.. py:module:: physicsnemo.geometry

Geometry modifiers have a public API separate from ``physicsnemo.mesh`` and
can operate on :class:`~physicsnemo.mesh.mesh.Mesh` and
:class:`~physicsnemo.mesh.domain_mesh.DomainMesh` objects. Tensor-level APIs
remain available for compiled, batched, and backend-specific workflows.

The package currently covers topology-preserving deformation, differentiable
deformation energies, and topology-changing surface remeshing. Future geometry
modifiers can live here without expanding the Mesh data-model API.

.. toctree::
   :maxdepth: 2

   deformation
   energies
   remeshing
