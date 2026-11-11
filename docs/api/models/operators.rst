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
the `SMART paper <https://arxiv.org/abs/2601.18707>`_. See the
`unified external aerodynamics recipe
<https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/unified_external_aero_recipe#smart>`_
for training and inference.

.. autoclass:: physicsnemo.experimental.models.smart.SMART
    :show-inheritance:
    :members: forward, encode_geometry, decode_queries

AB-UPT (Experimental)
---------------------

AB-UPT uses geometry supernodes and interacting surface/volume anchor streams
to define a query-independent field. The independent implementation follows
`the AB-UPT paper <https://arxiv.org/abs/2502.09692v4>`_, without research
implementation code, weights, or reference outputs. It supports joint or
single-branch prediction, reusable per-forward contexts, and finite-difference
vorticity with physical scaling. See the class notes for implementation choices
where the paper does not specify details.

Single-branch configurations are ablations of the coupled architecture:
setting ``volume_channels=0`` removes the volume branch and its anchor inputs,
while setting ``surface_channels=0`` removes the surface branch. Cross-branch
blocks become self-attention blocks to preserve depth. Joint configurations
retain both interacting streams. The public ``forward`` handles coordinate and
conditioning scaling, query chunking, and optional anchor supervision; low-level
encode/decode methods operate in the scaled model frame.

.. autoclass:: physicsnemo.experimental.models.ab_upt.ABUPT
    :show-inheritance:
    :members: encode_geometry, encode, decode_queries, decode_vorticity

.. autoclass:: physicsnemo.experimental.models.ab_upt.ABUPTContext
