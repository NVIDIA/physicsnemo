Remeshing
=========

.. currentmodule:: physicsnemo.mesh.remeshing

PhysicsNeMo provides CUDA-accelerated uniform remeshing for 2D triangle
manifolds embedded in 3D. ``n_clusters`` is the target number of output
vertices, not triangles; cleanup can produce slightly fewer vertices. Point
and cell data are discarded because their associations no longer match the
reconstructed topology. Global data, point dtype, and device are preserved.

GPU example
-----------

Create or move the input mesh on CUDA before calling :func:`remesh`, or use the
equivalent :meth:`~physicsnemo.mesh.Mesh.remesh` convenience method:

.. code:: python

   from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
   from physicsnemo.mesh.remeshing import remesh

   dense = sphere_icosahedral.load(subdivisions=6, device="cuda")
   coarse = remesh(dense, n_clusters=4_096)

   assert coarse.points.is_cuda
   assert 0 < coarse.n_points <= 4_096

Warp tuning
-----------

Advanced users can tune the search and initialization policy without changing
the remeshing kernels:

.. code:: python

   from physicsnemo.mesh.remeshing import WarpRemeshOptions

   options = WarpRemeshOptions(
       search_radius_scale=2.0,
       voxel_width_scale=1.0,
       hash_grid_resolution=192,
       farthest_point_threshold=512,
       farthest_point_oversampling=6,
   )
   tuned = dense.remesh(4_096, warp_options=options)

These values are host-side controls or runtime kernel arguments. Changing them
reuses the compiled Warp kernels rather than triggering JIT recompilation.

The GPU implementation uses area-weighted centroidal relaxation with a Warp
hash grid, projects the relaxed vertices onto the source surface using a GPU
bounding volume hierarchy (BVH), removes collapsed and duplicate faces, and
compacts unused vertices. Small targets use farthest-point initialization for
mesh quality; large targets use a linearithmic spatially stratified initializer
to avoid quadratic setup cost.

.. image:: /img/mesh/remeshing_comparison.png
   :alt: Dense Stanford bunny beside its GPU-remeshed result
   :align: center
   :width: 72%

Performance
-----------

The checked-in ASV benchmark measures warmed, end-to-end GPU execution:
clustering, surface projection, topology reconstruction, and cleanup. Timing
includes an explicit CUDA synchronization.

.. code:: console

   ./benchmarks/run_benchmarks.sh -b remesh

The figure below is a representative run of
``docs/img/mesh/remeshing_performance.py`` on an NVIDIA RTX PRO 6000 Blackwell
Server Edition MIG 1g.24GB partition using Warp 1.14.0. Absolute timings depend
on hardware and software versions; use the ASV benchmark above for measurements
in another environment.

.. image:: /img/mesh/remeshing_performance.png
   :alt: GPU remeshing latency plot across increasing input sizes
   :align: center
   :width: 65%

Behavior and limitations
------------------------

* Remeshing is non-differentiable. The implementation centers and scales
  geometry before computing in ``float32``, then restores the input coordinate
  frame and point dtype on return.
* Warp floating-point atomics can introduce small run-to-run differences in
  vertex positions and, near assignment ties, topology. Do not rely on bitwise
  reproducibility.
* Because Warp clusters by spatial distance rather than mesh connectivity,
  extremely close disconnected sheets can share a cluster.
* The optional ``max_iterations`` argument defaults to four centroid updates.

API reference
-------------

.. automodule:: physicsnemo.mesh.remeshing
   :members:
   :show-inheritance:
