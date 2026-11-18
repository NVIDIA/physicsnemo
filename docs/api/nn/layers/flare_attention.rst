FLARE Attention
===============

FLARE (Fast Low-rank Attention Routing Engine) is a low-rank self-attention
mechanism that aggregates token features into learned latent queries before
routing information back to the tokens. It provides an alternative to
:class:`~physicsnemo.nn.module.physics_attention.PhysicsAttentionBase`. Keys and
values come from ``dim x dim`` projections initialized to identity plus small
noise; ``qk_norm=True`` normalizes queries and keys per head (RMSNorm by
default, LayerNorm with ``use_rmsnorm=False``). ``no_weight_decay()`` names the
latent parameters to train without weight decay.

For details of the method, see the `FLARE paper
<https://arxiv.org/abs/2508.12594>`__.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLARE
   :show-inheritance:
   :members:
   :exclude-members: forward

FLARE++
-------

FLARE++ adds routing queries synthesized from the current input. One attention
pass from normalized learned seeds creates dynamic queries; a per-head sigmoid
gate adds them to learned fixed queries; the usual FLARE gather/scatter pair
then routes information, preserving linear complexity in the number of input
tokens for a fixed query count.

For details, see the `FLARE++ paper
<https://arxiv.org/abs/2608.11519>`__.

.. autoclass:: physicsnemo.nn.module.flare_attention.FLAREPlusPlus
   :show-inheritance:
   :members:
   :exclude-members: forward
