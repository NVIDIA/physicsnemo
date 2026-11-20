FLARE Attention
===============

FLARE (Fast Low-rank Attention Routing Engine) is a low-rank self-attention
mechanism that aggregates token features into learned latent queries before
routing information back to the tokens. It provides an alternative to
:class:`~physicsnemo.nn.module.physics_attention.PhysicsAttentionBase`.

For details of the method, see the `FLARE paper
<https://arxiv.org/abs/2508.12594>`__.

FLAREv2
-------

In :class:`~physicsnemo.nn.FLAREv2`, keys and values come from ``dim x dim``
projections initialized to identity plus small noise; ``qk_norm=True`` (the
default) normalizes queries and keys per head (RMSNorm by default, LayerNorm
with ``use_rmsnorm=False``). ``no_weight_decay()`` names the latent parameters
to train without weight decay.

FLAREv2 and FLAREPlusPlusv2 run their attention passes through shared kernels:
``torch.nn.functional.scaled_dot_product_attention`` (which also handles
token-sharded :class:`~physicsnemo.domain_parallel.ShardTensor` inputs), or,
with ``use_te=True``, Transformer Engine ``Linear``, norm and
``DotProductAttention`` modules.

.. autoclass:: physicsnemo.nn.module.flare_attention_v2.FLAREv2
   :show-inheritance:
   :members:
   :exclude-members: forward

FLAREPlusPlusv2
---------------

:class:`~physicsnemo.nn.FLAREPlusPlusv2` (FLARE++) adds routing queries
synthesized from the current input. One attention pass from normalized learned
seeds creates dynamic queries; a per-head sigmoid gate adds them to learned
fixed queries; the usual FLARE gather/scatter pair then routes information,
preserving linear complexity in the number of input tokens for a fixed query
count.

For details, see the `FLARE++ paper
<https://arxiv.org/abs/2608.11519>`__.

.. autoclass:: physicsnemo.nn.module.flare_attention_v2.FLAREPlusPlusv2
   :show-inheritance:
   :members:
   :exclude-members: forward

FLARE (deprecated)
------------------

:class:`~physicsnemo.nn.FLARE` is the original FLARE layer. It is deprecated
and will be removed in the 27.3 release; constructing it emits a
``FutureWarning``. Use :class:`~physicsnemo.nn.FLAREv2` or
:class:`~physicsnemo.nn.FLAREPlusPlusv2` for new training.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLARE
   :show-inheritance:
   :members:
   :exclude-members: forward

FLARE++
-------

:class:`~physicsnemo.nn.FLAREPlusPlus` is the original FLARE++ layer: learned
seeds synthesize routing queries from the current input with one attention
pass, and the usual FLARE gather/scatter pair routes information. It is the
attention used by GeoTransolver's ``"GALE_FPP"`` backend.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLAREPlusPlus
   :show-inheritance:
   :members:
   :exclude-members: forward
