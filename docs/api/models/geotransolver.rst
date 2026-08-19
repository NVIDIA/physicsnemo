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
