Geometry Functionals
====================

Point Displacement
------------------

.. autofunction:: physicsnemo.nn.functional.displace_points

.. code:: python

    import torch
    from physicsnemo.nn.functional import displace_points

    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], requires_grad=True
    )
    displacement = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
    )
    point_weights = torch.tensor([0.0, 0.5, 1.0])

    moved = displace_points(
        points,
        displacement,
        point_weights=point_weights,
    )
    moved.square().sum().backward()

Sparse Control-Point Morphing
-----------------------------

.. autofunction:: physicsnemo.nn.functional.morph_points

.. code:: python

    import torch
    from physicsnemo.nn.functional import morph_points

    x = torch.linspace(0.0, 1.0, 9)
    points = torch.stack((x, torch.zeros_like(x)), dim=-1).requires_grad_()
    control_points = points.detach()[[0, -1]].clone().requires_grad_()
    control_displacements = points.new_tensor(
        [[0.0, 0.25], [0.0, -0.15]], requires_grad=True
    )
    radii = points.new_tensor([0.8, 0.8])

    morphed = morph_points(
        points,
        control_points,
        control_displacements,
        radius=radii,
        kernel="wendland_c2",
    )
    morphed.square().mean().backward()

This allows an optimizer—or a model producing the control displacements—to
learn a deformation from a differentiable objective on ``morphed``.

Performance and Compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dense point displacement uses Torch on every device. Morphing uses Torch by
default on CPU and Warp by default on CUDA for sparse control sets. If Warp is
unavailable, automatic CUDA dispatch falls back to Torch, while explicitly
requesting ``implementation="warp"`` for morphing raises an ``ImportError``.
For a repeatedly evaluated, fixed-shape CUDA morph wrapped in
:func:`torch.compile`, benchmark ``implementation="torch"`` as well; compiler
fusion can make that path faster after its one-time compilation cost. Keep the
backend explicit when comparing compiled and eager runs.

Morphing evaluates every query/control pair and therefore has computational
cost proportional to ``batch_size * n_points * n_controls * n_spatial_dims``.
Pass all simultaneous controls in one call. For a
:class:`~physicsnemo.mesh.domain_mesh.DomainMesh`, the object API combines its
interior and boundary queries into one field evaluation before rebuilding the
individual component meshes.

For connectivity-preserving object APIs, use
:meth:`~physicsnemo.mesh.mesh.Mesh.displace`,
:meth:`~physicsnemo.mesh.mesh.Mesh.morph`, or
:meth:`~physicsnemo.mesh.domain_mesh.DomainMesh.morph`.

Lattice Free-Form Deformation
-----------------------------

.. autofunction:: physicsnemo.nn.functional.free_form_deform_points

.. code:: python

    import torch
    from physicsnemo.nn.functional import free_form_deform_points

    points = torch.rand(1024, 3)
    control_displacements = torch.zeros(4, 4, 4, 3, requires_grad=True)
    origin = points.new_zeros(3)
    extent = points.new_ones(3)

    deformed = free_form_deform_points(
        points,
        control_displacements,
        origin=origin,
        extent=extent,
        basis="bernstein",
    )
    deformed.square().mean().backward()

With zero control displacements, the operation is exactly the identity, so a
lattice initialized at zero is a well-behaved starting point for shape
optimization. An optimizer, or a model that produces the lattice
displacements, learns the deformation from a differentiable objective on
``deformed``.

For repeated GPU calls, create ``origin`` and ``extent`` once as device tensors,
as shown in the example. Python sequences are convenient for one-off calls.
Each invocation with sequence inputs creates and transfers new tensors.

Choosing a basis:

- ``"bernstein"`` provides classic free-form deformation. Every lattice node
  influences every point in the box, which suits coarse design lattices. The
  polynomial degree, global support, and evaluation cost grow with the
  resolution.
- ``"bspline"`` uses uniform cubic B-splines with local four-node-per-axis
  support. The per-point cost is independent of the lattice resolution, so it
  scales to fine lattices for local sculpting and registration-style
  deformation. Along an axis with ``n``
  coefficients, index ``i`` is associated with the Greville coordinate
  ``(i - 1) / (n - 3)``. The first and last coefficient planes therefore lie
  one knot spacing outside the evaluation box.
- ``"linear"``, ``"cubic_hermite"``, and ``"quintic_hermite"`` use the two
  neighboring lattice nodes per axis and exactly reproduce every control-node
  displacement. ``"linear"`` is piecewise multilinear and C0 across cell
  boundaries. The cubic and quintic Hermite variants are C1 and C2,
  respectively. These modes suit design parameters whose values must be
  attained at the lattice nodes.

For ``"bernstein"``, the evaluation cost is proportional to
``batch_size * n_points * prod(resolution) * n_spatial_dims``. For
``"bspline"``, it is proportional to
``batch_size * n_points * 4**n_spatial_dims * n_spatial_dims``. The
node-interpolating modes use ``2**n_spatial_dims`` controls per point. Points
outside the lattice box remain unchanged. A sufficient condition for continuity
with a fixed exterior is to zero the outermost coefficient plane on every
Bernstein or node-interpolating face. For cubic B-splines, zero the first and
last three coefficient planes on every axis because three planes have nonzero
weight at each box face.

Eager Torch evaluation chunks query points to keep estimated live FFD
temporaries within 256 MiB. Under :func:`torch.compile`, the Torch backend uses
one vectorized block because symbolic chunk loops cannot be unrolled. The eager
memory budget is therefore not enforced. Very large Bernstein workloads may
require substantially more peak memory when compiled.

For connectivity-preserving object APIs, use
:meth:`~physicsnemo.mesh.mesh.Mesh.free_form_deform` or
:meth:`~physicsnemo.mesh.domain_mesh.DomainMesh.free_form_deform`.

Mesh Poisson Disk Sample
------------------------

.. autofunction:: physicsnemo.nn.functional.mesh_poisson_disk_sample

.. rubric:: Visualization

This visualization compares Poisson samples generated by ``dart_throwing`` and
``weighted_sample_elimination`` on the same Stanford Bunny surface mesh.

.. figure:: /img/nn/functional/geometry/mesh_poisson_disk_sample/mesh_poisson_modes.gif
   :alt: Rotating Mesh Poisson disk sampling mode comparison
   :width: 100%

Mesh To Voxel Fraction
----------------------

.. autofunction:: physicsnemo.nn.functional.mesh_to_voxel_fraction

.. rubric:: Visualization

This visualization shows a side-by-side rotating view of the Stanford Bunny
mesh and the occupied voxels inferred by ``mesh_to_voxel_fraction``.

.. figure:: /img/nn/functional/geometry/mesh_to_voxel_fraction/mesh_to_voxel_rotation.gif
   :alt: Mesh to voxel fraction mesh and occupied-voxel rotation animation
   :width: 85%

Surface Remeshing
-----------------

.. autofunction:: physicsnemo.nn.functional.remeshing

Ray Mesh Intersect
------------------

.. autofunction:: physicsnemo.nn.functional.ray_mesh_intersect

.. rubric:: Visualization

This visualization shows a batch of rays intersecting a triangulated sphere,
with hits, misses, hit points, and surface normals.

.. figure:: /img/nn/functional/geometry/ray_mesh_intersect/ray_mesh_intersect_overview.png
   :alt: Ray mesh intersection overview with rays, hit points, and normals
   :width: 85%

Signed Distance Field
---------------------

.. autofunction:: physicsnemo.nn.functional.signed_distance_field

.. rubric:: Visualization

This visualization shows signed-distance values on a 2D slice through the
domain, with the zero level-set contour indicating the implicit surface. The
animation shows a sweep plane through the mesh (left) and corresponding SDF
slice image (right).

.. figure:: /img/nn/functional/geometry/sdf/sdf_slice_overview.png
   :alt: Signed distance field 2D slice visualization
   :width: 90%

.. figure:: /img/nn/functional/geometry/sdf/sdf_slice_sweep.gif
   :alt: Signed distance field z-slice sweep animation
   :width: 70%
