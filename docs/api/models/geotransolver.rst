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

Activation checkpointing
------------------------

GeoTransolver supports configurable activation checkpointing during training.
Set ``activation_checkpointing=True`` to checkpoint every selected component,
or provide a fraction in ``(0, 1)`` to checkpoint that leading fraction of the
GALE block stack. Checkpointing is disabled by default.

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
        activation_checkpointing_components=(
            "context",
            "preprocess",
            "blocks",
            "output",
        ),
    )

Checkpointing is active only in training mode when gradients are enabled. The
native backend uses PyTorch's non-reentrant checkpoint implementation, while
``use_te=True`` uses Transformer Engine's checkpoint wrapper. The option can be
combined with ``torch.compile``.

With ShardTensor domain parallelism, checkpointing is validated for standard
GALE with token-sharded inputs and ``include_local_features=False``. GALE_FA
with mixed placements and gradients through distributed ball-query coordinates
remain limited by the underlying distributed operators.

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
