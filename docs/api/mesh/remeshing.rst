Remeshing
=========

.. currentmodule:: physicsnemo.mesh.remeshing

PhysicsNeMo provides Warp-based uniform remeshing on CPU and CUDA for 2D
triangle manifolds embedded in 3D. ``n_clusters`` is the target number of
output vertices, not triangles. Cleanup can produce slightly fewer vertices.
Point and cell data are discarded because their associations no longer match
the reconstructed topology. Global data, point dtype, and device are
preserved.

CPU and CUDA Example
--------------------

The output remains on the input device. The example below selects CUDA when it
is available and otherwise runs on CPU. The equivalent
:meth:`~physicsnemo.mesh.Mesh.remesh` convenience method accepts
``n_clusters`` and ``max_iterations``:

.. code:: python

   import torch

   from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
   from physicsnemo.mesh.remeshing import remesh

   device = "cuda" if torch.cuda.is_available() else "cpu"
   dense = sphere_icosahedral.load(subdivisions=6, device=device)
   coarse = remesh(dense, n_clusters=4_096)

   assert coarse.points.device == dense.points.device
   assert 0 < coarse.n_points <= 4_096

Warp Tuning
-----------

Advanced users can tune the backend search and initialization policy through
the tensor functional. These backend-specific parameters may change as the
implementation evolves:

.. code:: python

   from physicsnemo.nn.functional import remeshing

   tuned_points, tuned_cells = remeshing(
       dense.points,
       dense.cells,
       n_clusters=4_096,
       search_radius_scale=2.0,
       voxel_width_scale=1.0,
       hash_grid_resolution=192,
       farthest_point_threshold=512,
       farthest_point_oversampling=6,
   )

These values are host-side controls or runtime kernel arguments. Changing them
reuses the compiled Warp kernels rather than triggering JIT recompilation.

The Warp implementation uses area-weighted centroidal relaxation with a hash
grid, projects the relaxed vertices onto the source surface using a bounding
volume hierarchy (BVH), removes collapsed and duplicate faces, and compacts
unused vertices. Small targets use farthest-point initialization for mesh
quality. Large targets use a linearithmic spatially stratified initializer to
avoid quadratic setup cost.

.. image:: /img/mesh/remeshing_comparison.png
   :alt: Dense Stanford bunny beside its Warp-remeshed result
   :align: center
   :width: 72%

Performance
-----------

The checked-in ASV benchmark measures warmed, end-to-end GPU execution:

- clustering
- surface projection
- topology reconstruction
- cleanup

Timing includes an explicit CUDA synchronization.

On supported CUDA devices, remeshing can be up to 300× faster than a CPU
baseline.

.. code:: console

   ./benchmarks/run_benchmarks.sh -b remesh

The figure below is a representative run of
``docs/img/mesh/remeshing_performance.py`` on an NVIDIA RTX PRO 6000 Blackwell
Server Edition MIG 1g.24GB partition using Warp 1.14.0. Absolute timings depend
on hardware and software versions. Use the ASV benchmark above for measurements
in another environment.

.. image:: /img/mesh/remeshing_performance.png
   :alt: GPU remeshing runtime plot across increasing input sizes
   :align: center
   :width: 65%

Behavior and Limitations
------------------------

* Remeshing is non-differentiable. The implementation centers and scales
  geometry before computing in ``float32``, then restores the input coordinate
  frame and point dtype on return.
* Warp floating-point atomics can introduce small run-to-run differences in
  vertex positions and, near assignment ties, topology, even though centroid
  sampling uses a fixed random seed. Do not rely on bitwise reproducibility.
* Because clustering uses spatial distance rather than mesh connectivity,
  sheets or thin features separated by less than the mean cluster spacing can
  be assigned to a common cluster and welded together.
* Projection can map distinct cluster centroids to the same surface position.
  Output vertices are compacted by connectivity but are not welded by
  position.
* The optional ``max_iterations`` argument defaults to four centroid updates.

API Reference
-------------

.. automodule:: physicsnemo.mesh.remeshing
   :members:
   :show-inheritance:
