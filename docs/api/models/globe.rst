GLOBE
=====

GLOBE (Green's-function-Like Operator for Boundary Element PDEs) is meant for
PDEs whose solution is driven mainly by the boundary conditions, in particular
the geometry of the boundary. It suits elliptic problems that need global
information propagation, with quantities of interest on or near the boundary,
such as external aerodynamics. GLOBE represents the solution as a linear
combination of learnable kernel functions evaluated from boundary source faces
to arbitrary target points.

Without retraining, the model is translation-, rotation-, and
parity-equivariant, invariant to the units of the problem through its
``reference_lengths`` argument, and invariant to uniform refinement of the
boundary mesh in the fine-mesh limit. Its kernels bake in the prior that the
influence of a small boundary feature falls off as :math:`1/r^2` in 3D and
:math:`1/r` in 2D.

For more information on GLOBE, refer to the `GLOBE paper
<https://arxiv.org/abs/2511.15856>`__.

.. autoclass:: physicsnemo.models.globe.GLOBE
    :show-inheritance:
    :members:
    :exclude-members: forward
