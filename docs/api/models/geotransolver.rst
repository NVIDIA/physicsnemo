GeoTransolver
==============

The GeoTransolver model extends Transolver with Geometry-Aware Latent Embeddings
(GALE) attention. It combines physics-aware self-attention over learned state
slices with cross-attention to geometry and global context, supporting both
unstructured meshes and structured 2D or 3D grids.

GALE layers use either
:class:`~physicsnemo.nn.module.physics_attention.PhysicsAttentionBase` (the default
setting) or
:class:`~physicsnemo.nn.module.flare_attention.FLARE` (with ``attention_type="GALE_FA"``)
as the self-attention backend.

For more information on GeoTransolver, refer to the `GeoTransolver paper
<https://arxiv.org/abs/2512.20399>`__.

Activation Checkpointing
------------------------

GeoTransolver supports configurable activation checkpointing during training.
Set ``activation_checkpointing=True`` to checkpoint every selected component,
and set ``checkpointing_ratio`` to a value in ``[0, 1]`` to checkpoint that
fraction of GALE blocks, distributed evenly across the block stack. The ratio
defaults to ``1.0``. This provides a model-depth-independent control for trading
activation memory against backward recomputation without specifying block indices.
Checkpointing is disabled by default.

The ``activation_checkpointing_components`` argument selects the checkpoint
boundaries. Supported values are ``"context"``, ``"preprocess"``, ``"blocks"``,
and ``"output"``. The default is ``("blocks",)`` for compatibility with
Transolver's block-only policy. For example, full-scope checkpointing is enabled
with:

.. code-block:: python

    model = GeoTransolver(
        functional_dim=8,
        out_dim=4,
        geometry_dim=3,
        use_te=False,
        activation_checkpointing=True,
        checkpointing_ratio=0.5,
        activation_checkpointing_components=(
            "context",
            "preprocess",
            "blocks",
            "output",
        ),
    )

Checkpointing is active only in training mode when gradients are enabled. The
native backend uses PyTorch's non-reentrant checkpoint implementation, while
``use_te=True`` uses Transformer Engine's checkpoint wrapper. The native
PyTorch checkpoint backend can be combined with ``torch.compile``.

.. autoclass:: physicsnemo.models.geotransolver.geotransolver.GeoTransolver
    :show-inheritance:
    :members:
    :exclude-members: forward

Building blocks
---------------

.. autoclass:: physicsnemo.models.geotransolver.context_projector.ContextProjector
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.geotransolver.context_projector.StructuredContextProjector
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.geotransolver.context_projector.GeometricFeatureProcessor
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.geotransolver.context_projector.MultiScaleFeatureExtractor
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.geotransolver.context_projector.GlobalContextBuilder
    :show-inheritance:
    :members:
    :exclude-members: forward

FLARE Attention Backend
-----------------------

For large meshes, setting ``attention_type="GALE_FA"`` swaps the
physics-attention slice mechanism for the `FLARE
<https://arxiv.org/abs/2508.12594>`__ (Fast Low-rank Attention Routing Engine)
backend. GALE_FA keeps GeoTransolver's geometry- and context-aware
cross-attention while using FLARE for the self-attention pass over learned
physical-state slices, reducing attention cost at scale. Refer also the
:doc:`FLARE model <flare>` documentation.

.. autoclass:: physicsnemo.nn.module.gale.GALE_FA
    :show-inheritance:
    :members:
    :exclude-members: forward
