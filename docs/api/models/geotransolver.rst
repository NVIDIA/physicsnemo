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
