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

FLARE++ adds input-conditioned routing to FLARE. One attention pass synthesizes
queries from the current input, and a per-head gate adds them to a learned fixed
reference query. The usual FLARE gather/scatter pair then routes information,
preserving linear complexity in the number of input tokens for a fixed query
count. Head-wise normalization of the synthesis call and near-identity input
projections stabilize early training.

Like FLARE, FLARE++ accepts token-sharded
:class:`~physicsnemo.domain_parallel.ShardTensor` inputs. Each rank keeps its
shard of the tokens, and only latent-sized tensors are communicated.

For details, see the `FLARE++ paper
<https://arxiv.org/abs/2608.11519>`__.

.. note::

   :class:`~physicsnemo.nn.module.flare_attention.FLAREPlusPlus` does not yet
   support token-sharded (``ShardTensor``) inputs and raises
   ``NotImplementedError`` for them. Use replicated inputs with data
   parallelism.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLAREPlusPlus
   :show-inheritance:
   :members:
   :exclude-members: forward
