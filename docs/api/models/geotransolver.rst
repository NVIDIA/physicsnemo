GeoTransolver
=============

GeoTransolver extends the Transolver physics-attention architecture with
geometry and global-context awareness. It constructs global context embeddings
from geometry and global features by projecting them onto physical state spaces,
then uses these embeddings throughout its GALE attention blocks via
cross-attention, letting geometric and global information guide the learned
physical-state dynamics.

For more information on GeoTransolver, please see the `GeoTransolver paper <https://arxiv.org/abs/2512.20399>`_.

.. autoclass:: physicsnemo.models.geotransolver.geotransolver.GeoTransolver
    :show-inheritance:
    :members:
    :exclude-members: forward

FLARE attention backend
-----------------------

GeoTransolver uses GALE attention by default. For large meshes, you can swap the
physics-attention slice mechanism for the
`FLARE <https://arxiv.org/abs/2508.12594>`_ (Fast Low-rank Attention Routing Engine)
backend by setting ``attention_type="GALE_FA"``. GALE_FA keeps GeoTransolver's
geometry- and context-aware cross-attention while using FLARE for the self-attention
pass over learned physical-state slices, reducing attention cost at scale. See also
the :doc:`FLARE model <flare>` documentation.

.. autoclass:: physicsnemo.nn.module.gale.GALE_FA
    :show-inheritance:
    :members:
    :exclude-members: forward

Context building
----------------

.. autoclass:: physicsnemo.models.geotransolver.context_projector.GlobalContextBuilder
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.models.geotransolver.context_projector.ContextProjector
    :show-inheritance:
    :members:
    :exclude-members: forward
