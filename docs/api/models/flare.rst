FLARE
=====

FLARE (Fast Low-rank Attention Routing Engine) is a Transolver variant that
replaces the physics-attention slice mechanism with a low-rank attention routing
backend, reducing the cost of attention over large meshes while preserving the
physics-attention formulation.

For more information, please see the `FLARE paper <https://arxiv.org/abs/2508.12594>`_.

.. autoclass:: physicsnemo.models.flare.flare.FLARE
    :show-inheritance:
    :members:
    :exclude-members: forward
