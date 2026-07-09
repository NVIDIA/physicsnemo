Remeshing
=========

.. currentmodule:: physicsnemo.mesh.remeshing

PhysicsNeMo provides uniform triangle-surface remeshing on both CUDA and CPU.
The public :func:`remesh` function dispatches from the input mesh device:

.. list-table:: Remeshing backends
   :header-rows: 1
   :widths: 18 24 58

   * - Backend
     - Selection
     - Implementation
   * - Warp
     - CUDA mesh (default)
     - GPU centroidal clustering, surface projection, and topology cleanup.
   * - PyACVD
     - CPU mesh (default)
     - Approximate Centroidal Voronoi Diagram (ACVD) clustering through the
       separately installed ``pyacvd`` package.

Both backends support 2D triangle manifolds embedded in 3D. ``n_clusters`` is
the target number of output vertices, not triangles; cleanup can produce
slightly fewer vertices. Point and cell data are discarded because their
associations no longer match the reconstructed topology. Global data, point
dtype, and device are preserved.

Install the optional CPU backend separately when it is needed:

.. code:: console

   pip install "pyacvd>=0.3.2" "pyvista>=0.47.0"

GPU example
-----------

Move the input mesh to CUDA before calling :func:`remesh`, or use the equivalent
:meth:`~physicsnemo.mesh.Mesh.remesh` convenience method:

.. code:: python

   from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
   from physicsnemo.mesh.remeshing import remesh

   dense = sphere_icosahedral.load(subdivisions=6, device="cuda")
   coarse = remesh(dense, n_clusters=4_096)

   assert coarse.points.is_cuda
   assert 0 < coarse.n_points <= 4_096

Select a backend explicitly when comparing implementations:

.. code:: python

   gpu_result = dense.remesh(4_096, implementation="warp")
   cpu_result = dense.to("cpu").remesh(4_096, implementation="pyacvd")

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
   :alt: Dense Stanford bunny beside Warp and PyACVD remeshes with the same target vertex count
   :align: center
   :width: 72%

Performance
-----------

The checked-in ASV benchmark compares warmed, end-to-end execution: clustering,
surface projection, topology reconstruction, and cleanup. Each implementation
receives data resident on its native device (CUDA for Warp and CPU for PyACVD),
and CUDA timing includes an explicit synchronization.

.. code:: console

   ./benchmarks/run_benchmarks.sh -b remesh

The figure below is a representative run of
``docs/img/mesh/remeshing_performance.py`` on an AMD EPYC 9B45 CPU and an
NVIDIA RTX PRO 6000 Blackwell Server Edition GPU, using PyACVD 0.4.0 and Warp
1.14.0. Absolute timings depend on hardware and software versions; use the ASV
benchmark above for measurements in another environment.

.. image:: /img/mesh/remeshing_performance.png
   :alt: Log-scale CPU PyACVD and GPU Warp remeshing latency plot
   :align: center
   :width: 65%

Behavior and limitations
------------------------

* Remeshing is non-differentiable. The Warp backend centers and scales geometry
  before computing in ``float32``, then restores the input coordinate frame and
  point dtype on return.
* Warp floating-point atomics can introduce small run-to-run differences in
  vertex positions and, near assignment ties, topology. Do not rely on bitwise
  reproducibility.
* Because Warp clusters by spatial distance rather than mesh connectivity,
  extremely close disconnected sheets can share a cluster. Use the
  topology-aware ``pyacvd`` implementation when strict component separation is
  required for such geometry.
* The optional ``max_iterations`` argument uses backend-tuned defaults when
  omitted: four centroid updates for Warp and up to 100 for PyACVD.

API reference
-------------

.. automodule:: physicsnemo.mesh.remeshing
   :members:
   :show-inheritance:
