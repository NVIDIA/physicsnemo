FLARE and FLARE++
=================

The FLARE model adapts Transolver by replacing its physics-attention blocks with
:class:`~physicsnemo.nn.module.flare_attention.FLARE` attention. FLARE uses
learned global queries to aggregate and redistribute token information through
a low-rank attention mechanism, and supports structured and unstructured data.

For details of the attention mechanism, see the `FLARE paper
<https://arxiv.org/abs/2508.12594>`__.

The standalone :class:`~physicsnemo.models.flare.FLAREPlusPlus` model keeps the
same Transolver-style residual backbone while synthesizing routing queries from
each block's current input. It is a complete model in its own right; the same
attention mechanism is also available to GeoTransolver through its
``"GALE_FPP"`` backend. See the `FLARE++ paper
<https://arxiv.org/abs/2608.11519>`__ for the dynamic-routing formulation.

The :class:`~physicsnemo.models.flare.FLARE` and
:class:`~physicsnemo.models.flare.FLAREPlusPlus` models are deprecated and will
be removed in the 27.3 release; constructing them emits a ``FutureWarning``.
Existing checkpoints still load. Use the experimental
:class:`~physicsnemo.experimental.models.flarev2.FLAREv2` and
:class:`~physicsnemo.experimental.models.flarev2.FLAREPlusPlusv2` models (see
below) for new training; checkpoints of the original models do not load into
them.

Activation Checkpointing
------------------------

All FLARE models support the same block-level activation-checkpointing policy
as Transolver. Set ``activation_checkpointing=True`` to enable checkpointing.
By default every block is checkpointed; set ``checkpointing_ratio`` to a value
in ``[0, 1]`` to checkpoint that fraction of blocks, distributed evenly across
the block stack. Checkpointing is disabled by default.

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

.. autoclass:: physicsnemo.models.flare.FLARE
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.flare.FLAREPlusPlus
    :show-inheritance:
    :members:
    :exclude-members: forward

FLAREv2 and FLAREPlusPlusv2 (Experimental)
------------------------------------------

:class:`~physicsnemo.experimental.models.flarev2.FLAREv2` and
:class:`~physicsnemo.experimental.models.flarev2.FLAREPlusPlusv2` are standalone
pre-norm transformers for point clouds and meshes. Each block applies a token
mixer and an MLP as residual branches:

.. code-block:: text

    x = Mlp(cat(embedding, fx))      # C_in -> 2C -> C
    x = x + Mixer(Norm(x))           # FLAREv2 / FLARE++ attention
    x = x + Mlp(Norm(x))             # C -> mlp_ratio * C -> C
    y = Linear(Norm(x))              # C -> C_out

The mixer is :class:`~physicsnemo.experimental.nn.FLAREv2` (learned latent
queries) or :class:`~physicsnemo.experimental.nn.FLAREPlusPlusv2` (FLARE++:
routing queries synthesized from the block input). Norms are RMSNorm by default
(``use_rmsnorm=True``) and LayerNorm otherwise. Inputs are flattened to
``(B, N, C)``; structured grids and unified position embeddings are not
supported. ``use_te=True`` runs the linear layers, norms (fused
``LayerNormMLP`` / ``LayerNormLinear``) and attention on Transformer Engine.
With ``use_te=False``, both models accept token-sharded
:class:`~physicsnemo.domain_parallel.ShardTensor` inputs (domain parallelism);
the output keeps the token sharding.

The latent queries and gate logits should be trained without weight decay:
``model.no_weight_decay()`` returns their fully qualified parameter names.

.. code-block:: python

    from physicsnemo.experimental.models.flarev2 import FLAREv2

    model = FLAREv2(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        activation_checkpointing=True,
        checkpointing_ratio=0.5,
    )

.. autoclass:: physicsnemo.experimental.models.flarev2.FLAREv2
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.experimental.models.flarev2.FLAREPlusPlusv2
    :show-inheritance:
    :members:
    :exclude-members: forward
