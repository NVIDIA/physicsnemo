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
