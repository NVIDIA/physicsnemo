FLARE, FLAREv2 and FLARE++
==========================

:class:`~physicsnemo.models.flare.FLAREv2` and
:class:`~physicsnemo.models.flare.FLAREPlusPlus` are standalone pre-norm
transformers for point clouds and meshes. Each block applies a token mixer and
an MLP as residual branches:

.. code-block:: text

    x = Mlp(cat(embedding, fx))      # C_in -> 2C -> C
    x = x + Mixer(Norm(x))           # FLAREv2 / FLARE++ attention
    x = x + Mlp(Norm(x))             # C -> mlp_ratio * C -> C
    y = Linear(Norm(x))              # C -> C_out

The mixer is :class:`~physicsnemo.nn.FLAREv2` (learned latent queries) or
:class:`~physicsnemo.nn.FLAREPlusPlus` (routing queries synthesized from the
block input). Norms are RMSNorm by default (``use_rmsnorm=True``) and LayerNorm
otherwise. Inputs are flattened to ``(B, N, C)``; structured grids and unified
position embeddings are not supported. ``use_te=True`` runs the linear layers,
norms (fused ``LayerNormMLP`` / ``LayerNormLinear``) and attention on
Transformer Engine. With ``use_te=False``, FLAREv2 and FLARE++ accept
token-sharded :class:`~physicsnemo.domain_parallel.ShardTensor` inputs (domain
parallelism); the output keeps the token sharding.

The latent queries and gate logits should be trained without weight decay:
``model.no_weight_decay()`` returns their fully qualified parameter names.

For details, see the `FLARE paper <https://arxiv.org/abs/2508.12594>`__ and
the `FLARE++ paper <https://arxiv.org/abs/2608.11519>`__.

:class:`~physicsnemo.models.flare.FLARE` is the original FLARE model: a
:class:`~physicsnemo.models.transolver.Transolver` whose physics-attention
blocks are replaced by :class:`~physicsnemo.nn.FLARE` attention blocks. It is
deprecated and will be removed in the 27.3 release; constructing it emits a
``FutureWarning``. Existing FLARE checkpoints still load. Use FLAREv2 or
FLARE++ for new training; FLARE checkpoints do not load into FLAREv2.

Activation Checkpointing
------------------------

FLARE, FLAREv2 and FLARE++ support the same block-level
activation-checkpointing policy as Transolver. Set
``activation_checkpointing=True`` to enable checkpointing. By default every
block is checkpointed; set ``checkpointing_ratio`` to a value in ``[0, 1]`` to
checkpoint that fraction of blocks, distributed evenly across the block stack.
Checkpointing is disabled by default.

.. code-block:: python

    model = FLAREv2(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        activation_checkpointing=True,
        checkpointing_ratio=0.5,
    )

Checkpointing is active only during gradient-enabled training. The native
backend uses PyTorch's non-reentrant checkpoint implementation, while
``use_te=True`` uses Transformer Engine's checkpoint wrapper. The native
PyTorch checkpoint backend can also be combined with ``torch.compile``.

.. autoclass:: physicsnemo.models.flare.FLAREv2
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.flare.FLAREPlusPlus
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.flare.FLARE
    :show-inheritance:
    :members:
    :exclude-members: forward
