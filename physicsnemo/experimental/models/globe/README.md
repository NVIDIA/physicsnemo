# GLOBE: Green's-function-Like Operator for Boundary Element PDEs

## Overview

This code contains an experimental architecture called GLOBE:
Green's-function-Like Operator for Boundary Element PDEs.

Paper: <https://arxiv.org/abs/2511.15856>

## PDE Problems Suitable for GLOBE

GLOBE is intended to be used for PDEs with the following mathematical
properties:

- The important behavior of the PDE is primarily driven by the boundary
  conditions - in particular, the *geometry* of the boundary.
- The PDE is elliptic, requiring global information propagation to solve
  correctly.
- The engineering quantities of interest are primarily located on or near the
  boundary.
- The PDE is either linear or can be roughly-approximated as linear for
  significant portions of the domain.

A GLOBE model represents the solution of a PDE as a linear combination of
learnable kernel functions evaluated from boundary source faces to target points.

## Mathematical Properties

The architecture itself has many invariant properties. So, *without retraining
the model*, you can do the following:

- **Translation-invariant**: Predictions do not change if you translate the
  entire problem.
- **Rotation-invariant**: Predictions do not change if you rotate the entire
  problem.
  - Also invariant to in-plane rotations of any given face on the boundary mesh.
- **Discretization-invariant**: If you decimate the boundary mesh uniformly, the
  predictions should be the same.
- **Parity-invariant**: If you reflect the problem across a plane, the
  predictions should be the same. Also, a problem with a plane of symmetry in
  the geometry will yield predictions that also have a plane of symmetry.
- **Units-invariant**: If you pose the same *physical* problem in a different
  scale (e.g., using different units), the predictions will be the same if you
  use the `reference_lengths` argument appropriately.

In addition:

- The model *bakes in* a physical prior that the influence of any individual
  small feature tends to fall off as 1/r^2 in 3D and 1/r in 2D. Model is
  globally interacting but with provably-diminishing influence as a function of
  distance -- this describes many elliptic PDEs of industrial relevance.
- Optionally, you can *mathematically force* any generated vector field to be
  exactly divergence-free. (This is not enabled by default.) This massively
  shrinks the space of possible field solutions, and therefore may boost model
  accuracy, generalizability, and data efficiency.
  - Author note on how to achieve this: in 2D, set the radial component of the
    basis vectors to zero. In 3D, set both the radial and polar components of
    the basis vectors to zero. (So, in both cases, you're only keeping the
    vortex-like components.)

## Other Notes

- Notably, the model is *not* invariant with respect to normal-vector-flips of
  the boundary mesh - a face with its normal facing inwards and a face with its
  normal facing outwards are encoded differently, and hence, will yield (in
  general) different predictions. You should ensure that the boundary mesh is
  consistently oriented between cases, and also that individual faces are
  consistently oriented within a single case. (This is by design, not by
  accident - if the model is later extended to handle oriented boundary
  conditions, like Neumann conditions, this is required.)
- Requires Python 3.12+ if you want to use the `torch.compile` decorator. This
  is because GLOBE extensively uses Python built-in dataclasses with
  `@cached_property` decorators - in previous versions of Python, these acquired
  a `threading.RLock` to guard first-time computation. This `RLock` is not
  traceable by Torch-Dynamo, and so, `torch.compile` will fail. This was fixed
  in Python 3.12, which made further thread-safety enhancements to
  `cached_property` that eliminated the need for the `RLock`.
