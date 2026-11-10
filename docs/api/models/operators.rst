Operator Models
===============

.. autoclass:: physicsnemo.models.domino.model.DoMINO
    :show-inheritance:
    :members:
    :exclude-members: forward

Domino Utils
------------

Utilities for working with the Domino model, including data processing and grid construction.
These utilities are essential for implementing and training Domino-based models.

.. automodule:: physicsnemo.models.domino.utils
    :members:
    :show-inheritance:

SMART (Experimental)
--------------------

SMART encodes sampled geometry into latent contexts and predicts surface or
volume fields at independent query coordinates. This implementation is based on
the `SMART paper <https://arxiv.org/abs/2601.18707>`_. Its configurable engineering
choices and CFD-mesh recipe are described in the `unified external aerodynamics recipe
<https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/unified_external_aero_recipe#smart-with-independent-geometry-and-query-sampling>`_.

.. autoclass:: physicsnemo.experimental.models.smart.SMART
    :show-inheritance:
    :members: forward, encode_geometry, decode_queries
