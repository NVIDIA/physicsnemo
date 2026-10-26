Feature-based OOD scoring
=========================

``KNNScorer`` accepts reference and query features independently of the model
or application that produced them. Larger scores indicate greater novelty.
It does not emit warnings or choose an acceptance threshold.

.. code-block:: python

    import torch
    from physicsnemo.experimental.uq.ood import KNNScorer

    scorer = KNNScorer(k=10, normalize=True)
    scorer.fit(reference_embeddings)  # [reference samples, features]
    threshold = torch.quantile(scorer.score(validation_embeddings), 0.99)
    scores = scorer.score(query_embeddings)
    flagged = scores > threshold     # Application policy

With ``normalize=True``, each vector is divided by its L2 norm plus ``1e-8``.
The score is the mean **Euclidean** distance to the nearest normalized reference
vectors, matching the former embedding guard. This is not a cosine-distance
score. With ``normalize=False``, distances use the supplied feature units.
Neither score is a calibrated probability or a prediction interval.

Use separate reference-validation data to select a threshold. Scoring the fitted
reference against itself includes each sample's zero-distance self-neighbor.

Persistence
-----------

.. code-block:: python

    torch.save({
        "scorer": scorer.state_dict(),
        "threshold": threshold,
        "features": feature_metadata,
    }, "ood.pt")

    saved = torch.load("ood.pt", weights_only=True, map_location=device)
    if saved["features"] != feature_metadata:
        raise ValueError("The saved reference uses different features")
    restored = KNNScorer()
    restored.load_state_dict(saved["scorer"])
    flagged = restored.score(query_embeddings) > saved["threshold"]

The application supplies ``feature_metadata`` as a dictionary of serializable
values identifying the model checkpoint, pooling, preprocessing, descriptor
schema/version, and units as applicable. The scorer persists its own reference,
neighbor count, and normalization setting. Matching feature dimensions alone
does not establish compatibility. Query tensors must use the fitted device;
half-precision reference features are stored in float32.

.. autoclass:: physicsnemo.experimental.uq.ood.KNNScorer
    :members:

Density and geometry descriptors
--------------------------------

``DensityScorer`` extracts the existing GMM/PCE implementation without changing
the feature transforms or score definition. Seeded GMM initialization now
uses a private RNG, and deterministic PCE fitting does not reseed the caller;
constructing a scorer therefore does not reset a dropout prediction stream.
It consumes arbitrary feature matrices; geometry is optional.

.. code-block:: python

    from physicsnemo.experimental.uq.ood import DensityScorer

    scorer = DensityScorer(method="pce", pce_components=3, device="cuda")
    scorer.fit(reference_features)
    scores = scorer.score(query_features)
    percentiles = scorer.percentiles(scores)
    flagged = percentiles > 99.0  # Application policy

    saved = scorer.get_state()
    restored = DensityScorer()
    restored.set_state(saved, device="cpu")

``get_state`` / ``set_state`` retain the existing NumPy serialization format,
including fitted GMM parameters or PCE standardization, PCA, polynomial state,
and reference scores. New PCE fits accumulate the small polynomial covariance
and precision matrix in float64 to avoid invalid scores on collinear features.
Restoration preserves that dtype and still accepts existing float32 artifacts.
Artifacts with non-finite reference scores require refitting; percentile scoring
rejects non-finite values rather than treating them as low novelty. Persist feature
metadata and decision thresholds alongside this state, just as for the distance
scorer.

Mesh-specific descriptors are explicitly imported from
``physicsnemo.experimental.uq.ood.geometry``. Schema v1.1 corrects the PCA
feature names to match the existing interleaved column order (axis1_x, axis2_x,
axis1_y, axis2_y, axis1_z, axis2_z). Numerical descriptors are unchanged, but
``FEATURE_VERSION`` and the name hash change. Existing v1.0 metadata must be
explicitly reviewed and updated before use with the recipe; do not reorder
fitted model parameters. Generic scorer imports do not import these utilities
or require mesh loading. Use ``physicsnemo.mesh.io`` to convert meshes before extracting features.

GeoTransolver can expose detached geometry tokens with
``return_geometry_features=True``. This output is distinct from combined context
embeddings; the model completes pending domain reductions before returning it.
Pooling, checkpoint identity, scoring, and accept/reject decisions belong to the
recipe. See ``examples/minimal/uq/feature_scoring.py``.

.. autoclass:: physicsnemo.experimental.uq.ood.DensityScorer
    :members:

Migration from experimental guardrails
--------------------------------------

The experimental guardrail namespace and its model/geometry wrappers are removed.
Use ``KNNScorer`` or ``DensityScorer`` with caller-provided features. Geometry
descriptors remain available through ``experimental.uq.ood.geometry``. Recipes
own reference collection, input bounds, thresholds, warnings, and fallback policy.
See ``examples/minimal/uq`` and ``examples/minimal/guardrails`` for compositions.
