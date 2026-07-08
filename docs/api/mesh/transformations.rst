Transformations and Projections
===============================

Geometric Transformations
-------------------------

.. currentmodule:: physicsnemo.mesh.transformations.geometric

Linear, affine, and nonlinear transformations on mesh geometry. Each function
returns a new :class:`~physicsnemo.mesh.mesh.Mesh` with transformed point
coordinates and appropriately invalidated caches. Cached quantities such as
normals and areas are automatically recomputed on next access.

All transformations are also available as methods on
:class:`~physicsnemo.mesh.mesh.Mesh`.

.. code:: python

    import numpy as np
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3)

    # Via Mesh methods
    translated = mesh.translate([1.0, 0.0, 0.0])
    rotated = mesh.rotate(axis=[0, 0, 1], angle=np.pi / 4)
    scaled = mesh.scale(2.0)
    scaled_aniso = mesh.scale([2.0, 1.0, 0.5])

    # Arbitrary linear transform
    import torch
    matrix = torch.eye(3) * 2
    transformed = mesh.transform(matrix)

Dense and Sparse Morphing
-------------------------

The mesh methods wrap the tensor-level
:func:`~physicsnemo.nn.functional.displace_points` and
:func:`~physicsnemo.nn.functional.morph_points` operations.

Dense displacement accepts a tensor or a point-data key (including a nested
tuple key). The operation returns a new mesh without changing ``mesh.points``.
Assigning a point-data key, as in the second example below, is a separate,
explicit mutation of the source mesh's attached data.

.. code:: python

    displacement = torch.zeros_like(mesh.points)
    displacement[:, 2] = 0.05
    displaced = mesh.displace(0.5 * displacement)

    # Point-data fields can drive the same operation.
    mesh.point_data["design_displacement"] = displacement
    displaced_from_data = mesh.displace("design_displacement")

Sparse controls are useful when only a small set of design handles is known. A
control point is a location in world coordinates, and its control displacement
is a vector rather than a destination coordinate. Control points do not need to
be mesh vertices, although selecting vertices makes their prescribed movement
directly visible in the result.

Single-Control Morphing
^^^^^^^^^^^^^^^^^^^^^^^

Indexing one vertex produces a coordinate vector with shape ``(3,)``. The
``morph`` API instead expects ``(n_controls, n_spatial_dims)``, so
``unsqueeze(0)`` adds the control dimension and gives shape ``(1, 3)``. It is
not a batch dimension.

.. code:: python

    top_index = mesh.points[:, 2].argmax()
    control_points = mesh.points[top_index].unsqueeze(0)  # (1, 3)
    control_displacements = mesh.points.new_tensor(
        [[0.0, 0.0, 0.5]], requires_grad=True
    )
    single_morph = mesh.morph(
        control_points,
        control_displacements,
        radius=1.0,
    )

    # Autograd continues through the returned point coordinates.
    objective = single_morph.points.square().mean()
    objective.backward()

``control_displacements`` is differentiable. An optimizer can learn it from any
differentiable loss computed from ``single_morph.points``. A model can also
predict the displacements.

Without point weights, a mesh vertex exactly at a unique control moves by its
prescribed displacement. ``point_weights`` can scale or mask the final movement.
Duplicate controls at the same coordinate contribute their mean displacement.

Multiple-Control Morphing
^^^^^^^^^^^^^^^^^^^^^^^^^

Advanced indexing retains the control dimension when several vertices are
selected. Each row of ``control_points`` pairs with the same row of
``control_displacements`` and, when supplied, one entry of ``radius``.

.. code:: python

    bottom_index = mesh.points[:, 2].argmin()
    control_indices = torch.stack((top_index, bottom_index))
    control_points = mesh.points[control_indices]  # (2, 3)
    control_displacements = mesh.points.new_tensor(
        [[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]
    )
    radii = mesh.points.new_tensor([1.0, 1.0])

    multiple_morph = mesh.morph(
        control_points,
        control_displacements,
        radius=radii,
    )

The radius is a Euclidean support distance in the same coordinate units as the
mesh. A control's influence vanishes smoothly at its support boundary. Where
supports overlap, all active controls are evaluated together using a stationary
zero-displacement background; the result is not a simple sum or average. Points
outside every support remain unchanged. Put simultaneous controls in one call,
because applying several morphs sequentially evaluates later fields on already
modified coordinates and is therefore order-dependent.

The ``kernel`` keyword names the compact radial kernel used by the field;
``"wendland_c2"`` is currently the supported value and the default.

Every tensor-valued radius must remain finite and strictly positive; its values
are not validated at runtime. When a model learns the radius, use a positive
parameterization such as
``torch.nn.functional.softplus(raw_radius) + radius_epsilon`` rather than
optimizing an unconstrained radius directly. Floating ``point_weights`` are
used as supplied and may be signed or greater than one.

.. rubric:: Visualization

The panels compare the original sphere with the single-control and
multiple-control examples above. Green markers identify the displaced handle
locations, while arrows and labels show the prescribed displacement directions
and magnitudes.

.. figure:: /img/mesh/sphere_morphing.png
   :alt: Original sphere and single-control and multiple-control sphere morphing
   :width: 100%

Domain Meshes
^^^^^^^^^^^^^

:meth:`~physicsnemo.mesh.domain_mesh.DomainMesh.morph` evaluates one
world-coordinate control field on the interior and every named boundary. With
``point_weights=None``, coincident component points receive identical motion.
Domain point weights must instead be a point-data key (or nested tuple key)
present in every component; raw weight tensors are rejected because component
point counts can differ. Every resolved field must use one common dtype across
the domain: bool for a hard mask, or the same floating dtype as the mesh points.
Coincident points remain coincident under a point-weight key only when their
resolved values also match.

.. code:: python

    import torch
    from physicsnemo.mesh import DomainMesh, Mesh

    interior = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
        point_data={"design_weight": torch.tensor([1.0, 0.8, 0.5])},
    )
    wall = Mesh(
        points=interior.points[:2],
        cells=torch.tensor([[0, 1]]),
        point_data={"design_weight": torch.tensor([1.0, 0.8])},
    )
    domain = DomainMesh(interior=interior, boundaries={"wall": wall})

    domain_controls = interior.points[[0]]  # (1, 2)
    domain_displacements = interior.points.new_tensor([[0.0, 0.25]])
    morphed_domain = domain.morph(
        domain_controls,
        domain_displacements,
        radius=1.25,
        point_weights="design_weight",
        implementation="torch",
    )

    # Equal point weights keep the shared wall vertices coincident.
    assert torch.allclose(
        morphed_domain.interior.points[:2],
        morphed_domain.boundaries["wall"].points,
    )

Morphing preserves connectivity and attached point, cell, global, and domain
data. Attached vector and tensor fields are treated as Lagrangian data and are
not pushed forward. Geometry-dependent caches are discarded and recomputed
lazily; topology caches are retained.

.. warning::

   Displacement and morphing do not detect or repair inverted, degenerate, or
   self-intersecting output cells. Call
   :meth:`~physicsnemo.mesh.mesh.Mesh.validate` or
   :meth:`~physicsnemo.mesh.domain_mesh.DomainMesh.validate` explicitly when a
   deformation could compromise validity.

.. automodule:: physicsnemo.mesh.transformations.geometric
   :members:
   :show-inheritance:

Guarded Normal Updates
----------------------

.. currentmodule:: physicsnemo.mesh.transformations.normal_update

:func:`guarded_normal_update` turns an already-signed point direction into a
normal-only deformation of a positive-dimensional codimension-one mesh. It
smooths the scalar normal component,
reconstructs the vector field along point normals, applies design-region
point weights, globally clips the maximum step, and backtracks when a candidate
contains collapsed or normal-reversed cells.
The source must be locally manifold and consistently oriented so each point has
a usable normal.

.. figure:: /img/mesh/guarded_normal_update.png
   :alt: An original triangular surface and its guarded normal update.
   :width: 100%

   A masked, smoothed normal update; color on the updated surface shows the
   accepted displacement magnitude.

.. code:: python

    objective = model(mesh)
    gradient = torch.autograd.grad(objective, mesh.points)[0]
    step_size = 1.0e-3
    updated, diagnostics = mesh.guarded_normal_update(
        -step_size * gradient,  # sign and magnitude are explicit
        point_weights=design_region,
        smoothing_iterations=5,
        max_step=0.01,
    )

The direction is not implicitly negated or rescaled. Pass a scaled
``-gradient`` for ordinary minimization, but pass an already-improving
sensitivity such as ``d(-drag)/dX`` directly. Cell-centered fields can first
be mapped to points with
:meth:`~physicsnemo.mesh.mesh.Mesh.cell_data_to_point_data`; that conversion is
an unweighted incidence mean.

Backtracking makes a discrete, non-differentiable choice of scale. Once chosen,
the accepted mesh is reconstructed from the original tensors so autograd still
flows through that selected branch. This host-controlled routine is an eager
optimization-step boundary rather than a full-graph-compilable operator. The
guard checks local cell geometry; it does not detect global self-intersections
or perform an objective-value line search. This first version updates a
codimension-one ``Mesh`` rather than propagating boundary movement through a
volume ``DomainMesh``.

Floating ``point_weights`` are per-point multipliers applied after smoothing;
they may be signed or greater than one. Use values in ``[0, 1]`` for a
conventional design-region mask or falloff.

.. automodule:: physicsnemo.mesh.transformations.normal_update
   :members:
   :show-inheritance:

Projections
-----------

.. currentmodule:: physicsnemo.mesh.projections

Spatial dimension manipulation -- changing the embedding dimension of a mesh
without altering its manifold dimension.

- :func:`embed` -- add spatial dimensions (non-destructive; for example, 2D mesh to 3D
  by appending zero coordinates)
- :func:`extrude` -- sweep a manifold to create a mesh one dimension higher
  (for example, a triangle mesh extruded to a prism mesh)
- :func:`project` -- reduce spatial dimensions (lossy; drops coordinate axes)

.. automodule:: physicsnemo.mesh.projections
   :members:
   :show-inheritance:
