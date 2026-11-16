Feature-Based Out-of-Distribution Scoring
=========================================

``KNNScorer`` uses k-nearest neighbors (kNN) for out-of-distribution (OOD)
scoring. It accepts reference and query features independently of the model
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

With ``normalize=True``, the scorer divides each vector by its L2 norm plus
``1e-8``.
The score is the mean *Euclidean* distance to the nearest normalized reference
vectors, matching the former embedding guard. This is not a cosine-distance
score. With ``normalize=False``, distances use the supplied feature units.
Neither score represents a calibrated probability or a prediction interval.

Use validation data separate from the fitted reference data to select a
threshold. Scoring the fitted reference against itself includes each sample's
zero-distance self-neighbor.
``reference_scores()`` provides leave-one-out scores when a validation split is
unavailable, preserving the former guard's threshold workflow. The method
excludes only each row's self-neighbor. Distinct rows with identical features
remain valid neighbors.
The scorer caches reference normalization at fit time. Exact PyTorch distance
calculations use bounded query and reference blocks. ``query_chunk_size`` and
``reference_chunk_size`` control memory independently of the stored reference
population.

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
values that identify the following applicable metadata:

- Model checkpoint
- Pooling
- Preprocessing
- Descriptor schema and version
- Units

The scorer persists its own reference, neighbor count, and normalization
setting. Matching feature dimensions alone does not establish compatibility.
Query tensors must be on the same device as the fitted reference data.
The scorer stores half-precision reference features in ``float32``.

.. autoclass:: physicsnemo.experimental.uq.ood.KNNScorer
    :members:

Density and Geometry Descriptors
--------------------------------

``DensityScorer`` provides the existing Gaussian mixture model (GMM) and
polynomial chaos expansion (PCE) implementations without changing the feature
transforms or score definition. Seeded GMM initialization uses a private random
number generator (RNG). Deterministic PCE fitting does not reseed the caller.
Constructing a scorer therefore does not reset a dropout prediction stream.
The scorer accepts arbitrary feature matrices. Geometry is optional.

.. code-block:: python

    from physicsnemo.experimental.uq.ood import DensityScorer

    scorer = DensityScorer(method="pce", pce_components=3, device="cuda")
    scorer.fit(reference_features)
    scorer.calibrate(calibration_features)  # Independent in-distribution samples
    scores = scorer.score(query_features)
    percentiles = scorer.percentiles(scores)
    flagged = percentiles > 99.0  # Application policy

    torch.save(scorer.state_dict(), "density.pt")
    restored = DensityScorer()
    restored.load_state_dict(
        torch.load("density.pt", weights_only=True, map_location="cpu")
    )

Both scorers provide ``state_dict`` and ``load_state_dict`` interfaces for
versioned tensor state and weights-only persistence. ``DensityScorer`` also
retains explicit NumPy support and moves inputs to its configured device. The
kNN scorer requires tensors on the reference device.

``get_state`` and ``set_state`` retain the existing NumPy serialization format,
including:

- Fitted GMM parameters or PCE standardization
- Principal component analysis (PCA)
- Polynomial state
- Reference scores

New PCE fits accumulate the small polynomial covariance and precision matrix
in ``float64`` to avoid invalid scores on collinear features. Restoration
preserves that dtype and still accepts existing ``float32`` artifacts.
Artifacts with non-finite reference scores require refitting. Percentile
scoring rejects non-finite values rather than treating them as low novelty.
Persist feature metadata and decision thresholds alongside this state, as for
the distance scorer.

Training percentiles are optimistic ranks, not false-positive probabilities.
Use held-out calibration features for reference ranks and separate evaluation
data to measure the resulting policy. Percentile resolution is ``100 / N``,
where ``N`` is the calibration sample count. Thresholds must be distinguishable
with the available calibration sample count. Calibration does not fix an
overparameterized model. The PCE and GMM fitting methods warn for small samples
when the number of terms or feature width is large relative to the training
population.
PCE standardizes inputs but does not whiten PCA components or build polynomial
features in ``float64``. Its covariance calculation uses ``float64``. GMM
retains full covariance in raw feature units. Scale heterogeneous features
consistently before fitting and scoring. Whitening, shrinkage, and alternative
covariance models would change fitting behavior, so this migration does not
introduce them.

Explicitly import mesh-specific descriptors from
``physicsnemo.experimental.uq.ood.geometry``. Schema ``v1.1`` corrects the PCA
feature names to match the following existing interleaved column order:

- ``axis1_x``
- ``axis2_x``
- ``axis1_y``
- ``axis2_y``
- ``axis1_z``
- ``axis2_z``

Numerical descriptors are unchanged, but ``FEATURE_VERSION`` and the name hash
change. Explicitly review and update existing ``v1.0`` metadata before using it
with the recipe. Do not reorder fitted model parameters. Generic scorer imports
do not import these utilities or require mesh loading. Use
``physicsnemo.mesh.io`` to convert meshes before extracting features.

``GeoTransolver.forward_with_geometry_features`` returns predictions and
detached geometry tokens in one pass. This output is distinct from combined
context embeddings. The model completes pending domain reductions before
returning it.
The following decisions belong to the recipe:

- Pooling
- Checkpoint identity
- Scoring
- Acceptance and rejection

Refer to ``examples/minimal/uq/feature_scoring.py``.

.. autoclass:: physicsnemo.experimental.uq.ood.DensityScorer
    :members:

Migration from Experimental Guardrails
--------------------------------------

This migration removes the experimental guardrail namespace and its model and
geometry wrappers.
Use ``KNNScorer`` or ``DensityScorer`` with caller-provided features. Geometry
descriptors remain available through ``experimental.uq.ood.geometry``. The
recipes control:

- Reference collection
- Input bounds
- Thresholds
- Warnings
- Fallback policy

Refer to ``examples/minimal/uq`` and ``examples/minimal/guardrails`` for
examples that combine these components.
