Local Geometry Encodings (experimental)
=======================================

Two layers that tell each query point about the geometry around it, for
surrogates on meshes and point clouds such as GeoTransolver. Both live in
:mod:`physicsnemo.experimental.nn`; their APIs may change.

Signed-distance stencil
-----------------------

The signed distance to a triangle surface is probed at a cubic stencil of points
around each query, with several spacings, and encoded by an MLP whose output
projection starts at zero. At a point in a thin member the finest probes leave
the material on both sides, so the encoding carries the local member thickness.
The triangles have to face outward consistently.

.. autoclass:: physicsnemo.experimental.nn.SignedDistanceStencil
   :show-inheritance:
   :members: probe_values

Radius-pooled local features
----------------------------

For each radius, candidate points are voxel-downsampled, neighbours within the
radius are found with :func:`~physicsnemo.nn.functional.radius_search` and drawn
at random up to a cap, and a shared MLP of their relative positions is
max-pooled.

.. autoclass:: physicsnemo.experimental.nn.RadiusPooledFeatures
   :show-inheritance:
   :members: neighbours
