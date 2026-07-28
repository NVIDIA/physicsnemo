Validation and Quality
======================

.. currentmodule:: physicsnemo.mesh.validation

Tools for assessing mesh integrity and element quality.

Validation
    :func:`validate` checks structural correctness: valid index ranges,
    consistent dimensions, proper data types, and data shape compatibility. It
    returns a report of any errors found and is also accessible as
    ``mesh.validate()``. ``validate_mesh`` remains available as a
    pending-deprecation compatibility name.

Quality metrics
    :func:`compute_quality_metrics` returns per-cell geometric quality
    indicators including aspect ratio, minimum and maximum angles, edge length
    ratios, and an overall quality score in a ``TensorDict``. It is also
    accessible as ``mesh.quality_metrics``.

Statistics
    :func:`compute_mesh_statistics` returns aggregate summaries (minimum,
    maximum, mean, and standard deviation) of geometric quantities across the
    entire mesh, including edge lengths, cell areas, angles, and quality scores.
    It is also accessible as ``mesh.statistics``.

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
