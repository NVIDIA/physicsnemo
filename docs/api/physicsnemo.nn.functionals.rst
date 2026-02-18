PhysicsNeMo Functionals
=======================

PhysicsNeMo functionals follow the ``torch.nn.functional`` pattern: stateless
operations designed for direct use in model code, training loops, and
pre/post-processing pipelines. They are intended to be easy to compose and to
behave consistently across CPU and GPU execution paths.

Many functionals are optimized for NVIDIA GPUs and can dispatch to accelerated
implementations when those backends are installed. For operations with multiple
implementations, PhysicsNeMo selects a preferred implementation by default and
falls back to another supported one when needed, emitting a warning so behavior
is explicit. Functionals with multiple implementations have plots available
in the documentation for performance comparisons.

.. toctree::
   :maxdepth: 2
   :caption: PhysicsNeMo Functionals
   :name: PhysicsNeMo Functionals

   nn/functionals/neighbors
   nn/functionals/geometry
   nn/functionals/fourier_spectral
   nn/functionals/regularization_parameterization
   nn/functionals/resampling_interpolation
