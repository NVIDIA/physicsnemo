FLARE Attention
===============

FLARE (Fast Low-rank Attention Routing Engine) is a low-rank self-attention
mechanism that aggregates token features into learned global query slots before
routing information back to the tokens. It provides an alternative to
:class:`~physicsnemo.nn.module.physics_attention.PhysicsAttentionBase` and can
use either PyTorch scaled dot-product attention or Transformer Engine by setting
``use_te=True``.

For details of the method, see the `FLARE paper
<https://arxiv.org/abs/2508.12594>`__.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLARE
   :show-inheritance:
   :members:
   :exclude-members: forward

FLARE++
-------

FLARE++ replaces FLARE's fixed learned routing queries with routing queries
synthesized from the current input. It uses one attention pass to create those
queries and the usual FLARE gather/scatter pair to route information, preserving
linear complexity in the number of input tokens for a fixed query count.
With PhysicsNeMo domain parallelism, the token dimension may be sharded across
devices: both encoder calls perform globally normalized distributed attention,
while the decoder remains token-local, so no device gathers the full sequence.

For details, see the `FLARE++ paper
<https://arxiv.org/abs/2608.11519>`__.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLAREPlusPlus
   :show-inheritance:
   :members:
   :exclude-members: forward
