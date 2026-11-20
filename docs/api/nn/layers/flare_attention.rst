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

The :class:`~physicsnemo.nn.FLARE` layer is deprecated and will be removed in
the 27.3 release; constructing it emits a ``FutureWarning``. Use the
experimental :class:`~physicsnemo.experimental.nn.FLAREv2` or
:class:`~physicsnemo.experimental.nn.FLAREPlusPlusv2` layers (see below) for
new training.

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

For details, see the `FLARE++ paper
<https://arxiv.org/abs/2608.11519>`__.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLAREPlusPlus
   :show-inheritance:
   :members:
   :exclude-members: forward

FLAREv2 and FLAREPlusPlusv2 (Experimental)
------------------------------------------

In :class:`~physicsnemo.experimental.nn.FLAREv2`, keys and values come from
``dim x dim`` projections initialized to identity plus small noise;
``qk_norm=True`` (the default) normalizes queries and keys per head (RMSNorm by
default, LayerNorm with ``use_rmsnorm=False``).
:class:`~physicsnemo.experimental.nn.FLAREPlusPlusv2` (FLARE++) adds routing
queries synthesized from the current input: one attention pass from normalized
learned seeds creates dynamic queries, and a per-head sigmoid gate adds them to
learned fixed queries before the usual FLARE gather/scatter pair.
``no_weight_decay()`` names the latent parameters to train without weight
decay.

Both layers run their attention passes through
``torch.nn.functional.scaled_dot_product_attention`` (which also handles
token-sharded :class:`~physicsnemo.domain_parallel.ShardTensor` inputs), or,
with ``use_te=True``, Transformer Engine ``Linear``, norm and
``DotProductAttention`` modules.

.. autoclass:: physicsnemo.experimental.nn.flare_attention_v2.FLAREv2
   :show-inheritance:
   :members:
   :exclude-members: forward

.. autoclass:: physicsnemo.experimental.nn.flare_attention_v2.FLAREPlusPlusv2
   :show-inheritance:
   :members:
   :exclude-members: forward
