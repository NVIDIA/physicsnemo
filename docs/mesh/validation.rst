Validation and Quality
======================

.. currentmodule:: physicsnemo.mesh.validation

Tools for assessing mesh integrity and element quality.

**Validation** (:func:`validate_mesh`)
    Checks structural correctness: valid index ranges, consistent dimensions,
    proper data types, and data shape compatibility. Returns a report of any
    errors found. Also accessible as ``mesh.validate()``.

**Quality metrics** (:func:`compute_quality_metrics`)
    Per-cell geometric quality indicators including aspect ratio, minimum/maximum
    angles, edge length ratios, and an overall quality score. Returned as a
    ``TensorDict``. Also accessible as ``mesh.quality_metrics``.

**Statistics** (:func:`compute_mesh_statistics`)
    Aggregate summaries (min, max, mean, std) of geometric quantities across
    the entire mesh: edge lengths, cell areas, angles, and quality scores.
    Also accessible as ``mesh.statistics``.

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)

    # Validate structural integrity
    report = mesh.validate()

    # Per-cell quality
    quality = mesh.quality_metrics
    print(quality["quality_score"].mean())

    # Aggregate statistics
    stats = mesh.statistics

API Reference
-------------

.. automodule:: physicsnemo.mesh.validation
   :members:
   :show-inheritance:
