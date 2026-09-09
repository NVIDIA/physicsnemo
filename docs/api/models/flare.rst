FLARE
=====

The FLARE model adapts Transolver by replacing its physics-attention blocks with
:class:`~physicsnemo.nn.module.flare_attention.FLARE` attention. FLARE uses
learned global queries to aggregate and redistribute token information through
a low-rank attention mechanism, and supports structured and unstructured data.

For details of the attention mechanism, see the `FLARE paper
<https://arxiv.org/abs/2508.12594>`__.

Activation Checkpointing
------------------------

FLARE supports the same block-level activation-checkpointing policy as
Transolver. Set ``activation_checkpointing=True`` to enable checkpointing. By
default every FLARE block is checkpointed; set ``checkpointing_ratio`` to a
value in ``[0, 1]`` to checkpoint that fraction of blocks, distributed evenly
across the block stack. Checkpointing is disabled by default.

.. code-block:: python

    model = FLARE(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        structured_shape=None,
        activation_checkpointing=True,
        checkpointing_ratio=0.5,
    )

Checkpointing is active only during gradient-enabled training. The native
backend uses PyTorch's non-reentrant checkpoint implementation, while
``use_te=True`` uses Transformer Engine's checkpoint wrapper. The native
PyTorch checkpoint backend can also be combined with ``torch.compile``.

.. autoclass:: physicsnemo.models.flare.flare.FLARE
    :show-inheritance:
    :members:
    :exclude-members: forward
