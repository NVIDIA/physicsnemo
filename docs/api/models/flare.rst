FLARE
=====

The FLARE model adapts Transolver by replacing its physics-attention blocks with
:class:`~physicsnemo.nn.module.flare_attention.FLARE` attention. FLARE uses
learned global queries to aggregate and redistribute token information through
a low-rank attention mechanism, and supports structured and unstructured data.

For details of the attention mechanism, see the `FLARE paper
<https://arxiv.org/abs/2508.12594>`__.

.. autoclass:: physicsnemo.models.flare.flare.FLARE
    :show-inheritance:
    :members:
    :exclude-members: forward
