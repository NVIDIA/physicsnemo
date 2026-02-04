Guardrails User Guide
=====================

PhysicsNeMo Guardrails provide robust validation and safety checks for AI-driven physics simulations. This guide covers how to use guardrails to detect out-of-distribution inputs and ensure reliable model predictions.

Overview
--------

Guardrails are validation layers that protect physics models from unreliable predictions on unusual or out-of-specification inputs. They provide three levels of alerts:

- **OK**: Input is within the expected distribution
- **WARN**: Input is unusual but may be acceptable (investigate)
- **REJECT**: Input is highly anomalous (likely invalid or out-of-distribution)

Available Guardrails
--------------------

PhysicsNeMo currently provides guardrails for different stages of the inference pipeline:

Pre-Inference Guardrails
~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-inference guardrails validate inputs **before** they are passed to your physics model.

Geometry Guardrails (Available)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The geometry guardrail detects out-of-distribution geometric configurations in CAD models, simulation meshes, and 3D shape data.

**Key Features:**

- Density-based anomaly detection with multiple methods (GMM, PCE)
- Non-invariant features capturing position, orientation, and scale
- Parallel processing for efficient batch validation
- Optional GPU acceleration (2-10x speedup for GMM)
- Optional Rust-based STL reader (5-10x faster I/O)

**Use Cases:**

- Quality control in additive manufacturing
- Simulation mesh validation
- CAD model anomaly detection
- Design space exploration safety checks

See :ref:`geometry-guardrails-tutorial` for detailed examples.

Field Guardrails (Coming Soon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Placeholder: Future guardrails for validating physical field distributions (velocity, pressure, temperature, etc.) before model inference.*

**Planned Features:**

- Statistical distribution checks for field quantities
- Boundary condition validation
- Physical constraint verification (e.g., mass conservation)

Parameter Guardrails (Coming Soon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Placeholder: Future guardrails for validating simulation parameters and boundary conditions.*

**Planned Features:**

- Range checking for physical parameters
- Consistency validation across parameter sets
- Reynolds number, Mach number, and other dimensionless quantity checks

Post-Inference Guardrails
~~~~~~~~~~~~~~~~~~~~~~~~~~

Post-inference guardrails validate **model outputs** to ensure physical plausibility.

Physical Consistency Guardrails (Coming Soon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Placeholder: Future guardrails for checking if model predictions satisfy physical laws.*

**Planned Features:**

- Conservation law verification (mass, momentum, energy)
- Boundary condition satisfaction checks
- Temporal consistency validation
- Spatial smoothness checks

Uncertainty Guardrails (Coming Soon)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Placeholder: Future guardrails for flagging predictions with high uncertainty.*

**Planned Features:**

- Ensemble-based uncertainty quantification
- Confidence interval validation
- Epistemic vs. aleatoric uncertainty separation

.. _geometry-guardrails-tutorial:

Geometry Guardrails Tutorial
-----------------------------

This tutorial demonstrates how to use geometry guardrails for out-of-distribution detection in CAD/STL files.

Installation
~~~~~~~~~~~~

Install the required dependencies:

.. code-block:: bash

    pip install trimesh scikit-learn

For GPU acceleration (optional):

.. code-block:: bash

    pip install torch

Basic Usage
~~~~~~~~~~~

Here's a simple example of training and using a geometry guardrail:

.. code-block:: python

    import trimesh
    from pathlib import Path
    from physicsnemo.experimental.guardrails import GeometryGuardrail

    # Load training data (known-good geometries)
    train_meshes = [
        trimesh.load("part_001.stl", force="mesh"),
        trimesh.load("part_002.stl", force="mesh"),
        trimesh.load("part_003.stl", force="mesh"),
        # ... more training meshes
    ]

    # Create and fit guardrail
    guardrail = GeometryGuardrail(
        n_components=1,      # Single Gaussian (unimodal assumption)
        warn_pct=95.0,       # Warn above 95th percentile
        reject_pct=99.0,     # Reject above 99th percentile
        covariance_type="full",
        random_state=42,
    )
    guardrail.fit(train_meshes)

    # Save the fitted model
    guardrail.save(Path("geometry_guardrail.npz"))

    # Query new geometries
    test_meshes = [trimesh.load("new_part.stl", force="mesh")]
    results = guardrail.query(test_meshes)

    for res in results:
        status = res['status']      # 'OK', 'WARN', or 'REJECT'
        percentile = res['percentile']  # Anomaly percentile
        score = res['score']        # Raw anomaly score
        
        print(f"Status: {status}, Percentile: {percentile:.2f}%")

Batch Processing from Directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets, use directory-based processing with multiprocessing:

.. code-block:: python

    import multiprocessing as mp
    from pathlib import Path
    from physicsnemo.experimental.guardrails import GeometryGuardrail

    # Fit from directory of STL files
    train_dir = Path("validated_parts/")
    guardrail = GeometryGuardrail(warn_pct=95.0, reject_pct=99.0)
    
    guardrail.fit_from_dir(
        train_dir,
        n_workers=mp.cpu_count() - 1,  # Use all but one CPU
        chunksize=8,                    # Process files in batches
    )
    
    # Save the model
    guardrail.save(Path("guardrail_model.npz"))

    # Query a directory of test geometries
    test_dir = Path("new_parts/")
    results = guardrail.query_from_dir(
        test_dir,
        n_workers=mp.cpu_count() - 1,
    )

    # Process results
    for r in results:
        if r['status'] == 'REJECT':
            print(f"⚠ REJECT: {r['name']} (p={r['percentile']:.1f}%)")
        elif r['status'] == 'WARN':
            print(f"⚠ WARN: {r['name']} (p={r['percentile']:.1f}%)")

GPU Acceleration
~~~~~~~~~~~~~~~~

For large datasets (1000+ samples), GPU acceleration provides 2-10x speedup:

.. code-block:: python

    from physicsnemo.experimental.guardrails import GeometryGuardrail

    # Create guardrail with GPU support
    guardrail_gpu = GeometryGuardrail(
        n_components=2,
        warn_pct=95.0,
        reject_pct=99.0,
        device="cuda",  # Use GPU for density modeling
        random_state=42,
    )

    # Fit and query on GPU (faster for large datasets)
    guardrail_gpu.fit(train_meshes)
    results = guardrail_gpu.query(test_meshes)

**Device Options:**

- ``device="cpu"`` - CPU-only (default, always available)
- ``device="cuda"`` - Default GPU
- ``device="cuda:0"`` - Specific GPU device

**When to Use GPU:**

- ✓ Dataset size > 1000 samples
- ✓ Batch inference on 100+ geometries
- ✓ Latency-critical applications (<100ms queries)

Loading and Reusing Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save fitted models and load them later for inference:

.. code-block:: python

    from pathlib import Path
    from physicsnemo.experimental.guardrails import GeometryGuardrail

    # Train once and save
    guardrail = GeometryGuardrail(warn_pct=95.0, reject_pct=99.0)
    guardrail.fit_from_dir(train_dir)
    guardrail.save(Path("production_guardrail.npz"))

    # Later: load for inference (potentially on different device)
    guardrail_loaded = GeometryGuardrail.load(
        Path("production_guardrail.npz"),
        device="cuda",  # Can load on different device
    )
    
    results = guardrail_loaded.query(test_meshes)

Production Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete example for production deployment:

.. code-block:: python

    import multiprocessing as mp
    from pathlib import Path
    from physicsnemo.experimental.guardrails import GeometryGuardrail

    def validate_geometry_batch(stl_dir: Path, model_path: Path):
        """
        Validate a batch of geometries against a trained guardrail.
        
        Parameters
        ----------
        stl_dir : Path
            Directory containing STL files to validate.
        model_path : Path
            Path to saved guardrail model.
            
        Returns
        -------
        dict
            Statistics and results for the batch.
        """
        # Load the guardrail
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        
        guardrail = GeometryGuardrail.load(model_path, device=device)
        
        # Validate all geometries
        results = guardrail.query_from_dir(
            stl_dir,
            n_workers=mp.cpu_count() - 1,
            chunksize=8,
        )
        
        # Compute statistics
        ok_count = sum(1 for r in results if r['status'] == 'OK')
        warn_count = sum(1 for r in results if r['status'] == 'WARN')
        reject_count = sum(1 for r in results if r['status'] == 'REJECT')
        
        # Flag anomalies
        anomalies = [r for r in results if r['status'] == 'REJECT']
        
        return {
            'total': len(results),
            'ok': ok_count,
            'warn': warn_count,
            'reject': reject_count,
            'anomalies': anomalies,
        }

    # Usage
    stats = validate_geometry_batch(
        stl_dir=Path("production_batch/"),
        model_path=Path("guardrail_model.npz"),
    )
    
    print(f"Validated {stats['total']} geometries:")
    print(f"  OK: {stats['ok']}")
    print(f"  WARN: {stats['warn']}")
    print(f"  REJECT: {stats['reject']}")
    
    if stats['reject'] > 0:
        print("\nRejected geometries require review:")
        for anom in stats['anomalies']:
            print(f"  - {anom['name']} (percentile: {anom['percentile']:.2f}%)")

Configuration Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

**Choosing Density Estimation Method:**

Use **GMM (Gaussian Mixture Model)** when:

- Data has multi-modal distributions (multiple sub-populations)
- Features are relatively independent
- You need GPU acceleration
- Default choice for most applications

.. code-block:: python

    guardrail = GeometryGuardrail(
        method="gmm",
        n_components=2,  # Number of Gaussian components
        device="cuda",   # GPU acceleration
    )

Use **PCE (Polynomial Chaos Expansion with Hermite polynomials)** when:

- Features have strong correlations (common in physics)
- You want orthogonal basis functions (Hermite polynomials)
- Hermite coefficients can be interpreted as sensitivity indices
- Data is high-dimensional with smooth, Gaussian-like distributions
- You prefer avoiding GMM hyperparameter tuning

**Note**: PCE currently supports CPU computation only. For GPU acceleration, use GMM.

.. code-block:: python

    guardrail = GeometryGuardrail(
        method="pce",
        n_components=10,      # PCA components (None=auto-select)
        poly_degree=2,        # Quadratic expansion
        interaction_only=False,
    )

**Choosing n_components:**

For GMM:

- ``n_components=1``: Single Gaussian, fastest, assumes unimodal distribution
- ``n_components=2-5``: Captures multimodal distributions (e.g., multiple part families)
- ``n_components > 5``: Risk of overfitting on small datasets

For PCE:

- ``n_components=None``: Auto-select to explain 95% variance (recommended)
- ``n_components=5-15``: Explicit number of PCA components
- Higher values capture more variance but may include noise

**Setting Thresholds:**

- ``warn_pct=95.0, reject_pct=99.0``: Conservative (fewer false alarms)
- ``warn_pct=90.0, reject_pct=95.0``: Balanced
- ``warn_pct=80.0, reject_pct=90.0``: Aggressive (catches more anomalies, more false positives)

**Covariance Types (GMM only):**

- ``"full"``: Most flexible, slowest (recommended)
- ``"tied"``: Shared covariance, faster
- ``"diag"``: Diagonal covariance (assumes feature independence)
- ``"spherical"``: Fastest, least flexible

**Polynomial Degree (PCE with Hermite polynomials only):**

- ``poly_degree=1``: Linear Hermite basis (fastest, captures linear correlations only)
- ``poly_degree=2``: Quadratic Hermite expansion (good default, captures second-order effects)
- ``poly_degree=3``: Cubic Hermite expansion (more expressive, risk of overfitting)
- ``poly_degree >= 4``: Use with caution (likely to overfit, many terms)

**Note**: Hermite polynomials are orthogonal with respect to the Gaussian distribution,
making them the natural choice for PCA-transformed features. Higher degrees generate
more terms: for d=10 components, degree 2 → 66 terms, degree 3 → 286 terms.

Advanced Topics
---------------

Custom Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

*Coming soon: How to define custom feature extractors for domain-specific applications.*

Ensemble Guardrails
~~~~~~~~~~~~~~~~~~~

*Coming soon: Combining multiple guardrails for improved robustness.*

Guardrail Interpretability
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Coming soon: Understanding why geometries are flagged as anomalous.*

API Reference
-------------

For detailed API documentation, see:

- :class:`physicsnemo.experimental.guardrails.GeometryGuardrail`
- :class:`physicsnemo.experimental.guardrails.geometry.GeometryDensityModel`
- :func:`physicsnemo.experimental.guardrails.geometry.extract_features`

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue: "No valid STL files found"**

*Solution:* Ensure your STL files are valid and contain at least 50 vertices. Check file permissions and paths.

**Issue: Slow performance on large datasets**

*Solution:* Use GPU acceleration (``device="cuda"``) and increase ``n_workers`` for parallel processing.

**Issue: Too many false positives**

*Solution:* Increase training data diversity or adjust thresholds (lower ``warn_pct`` and ``reject_pct``).

**Issue: GPU out of memory**

*Solution:* Reduce batch size or fall back to CPU (``device="cpu"``).

Best Practices
--------------

1. **Representative Training Data**: Ensure training data covers the full range of expected geometries
2. **Sufficient Sample Size**: Use at least 100 training samples for reliable density estimation
3. **Threshold Tuning**: Validate thresholds on a held-out validation set
4. **Regular Retraining**: Update guardrails as your geometry distribution evolves
5. **Monitor Performance**: Track false positive/negative rates in production

References
----------

- Li, Z., et al. (2020). "Fourier Neural Operator for Parametric PDEs." ICLR 2021.
- Chandola, V., et al. (2009). "Anomaly Detection: A Survey." ACM Computing Surveys.
- Reynolds, D. A. (2009). "Gaussian Mixture Models." Encyclopedia of Biometrics.

Contributing
------------

To contribute new guardrails or improvements:

1. Follow the coding standards in ``CODING_STANDARDS/MODELS_IMPLEMENTATION.md``
2. Add comprehensive docstrings and tests
3. Submit a PR with examples and documentation

See ``CONTRIBUTING.md`` for detailed guidelines.

Support
-------

For issues, questions, or feature requests:

- File issues on the PhysicsNeMo GitHub repository
- Join the NVIDIA Developer Forums
- Consult the full documentation at https://docs.nvidia.com/physicsnemo
