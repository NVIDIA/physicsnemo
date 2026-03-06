Transolver
==========

The Transolver model adapts the transformer architecture with a physics-attention
mechanism for solving partial differential equations on structured and unstructured
meshes. It projects inputs onto physics-informed slices before applying attention,
enabling efficient learning of physical systems.

.. autoclass:: physicsnemo.models.transolver.transolver.Transolver
    :show-inheritance:
    :members:
    :exclude-members: forward

Building blocks
---------------

.. autoclass:: physicsnemo.models.transolver.transolver.TransolverBlock
    :show-inheritance:
    :members:
    :exclude-members: forward
