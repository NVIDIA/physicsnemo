GLOBE
=====

GLOBE (Green's-function-Like Operator for Boundary Element PDEs) targets
partial differential equations (PDEs) whose solutions depend primarily on
boundary conditions, particularly boundary geometry. It supports elliptic
problems that require global information propagation and have quantities of
interest on or near the boundary. External aerodynamics is one example. GLOBE
represents the solution as a linear combination of learnable kernel functions
evaluated from boundary source faces to arbitrary target points.

Without retraining, the model is equivariant to translations, rotations, and
parity transformations. It is invariant to uniform refinement of the boundary
mesh in the fine-mesh limit. Predictions remain unchanged when you change the
problem units and express ``reference_lengths`` and ``reference_area`` in the
new units. Set ``reference_area`` during construction. GLOBE saves its value
in checkpoints.
The kernels incorporate the prior that the influence of a small boundary
feature falls off as :math:`1/r^2` in 3D and :math:`1/r` in 2D.

For more information about GLOBE, refer to the `GLOBE paper
<https://arxiv.org/abs/2511.15856>`__.

.. autoclass:: physicsnemo.models.globe.GLOBE
    :show-inheritance:
    :members:
    :exclude-members: forward
