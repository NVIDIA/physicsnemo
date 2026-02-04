# Geometry Guardrails Example

This example demonstrates how to use PhysicsNeMo's geometry guardrails for validating CAD/STL files against a distribution of known-good geometries.

## Overview

Geometry guardrails provide out-of-distribution (OOD) detection for 3D geometric data. They learn the distribution of "normal" geometries from training data and flag unusual or unexpected shapes at inference time.

**Key Features:**
- Density-based anomaly detection using Gaussian Mixture Models
- Three-level classification: OK, WARN, REJECT
- Parallel processing for efficient batch validation
- Optional GPU acceleration (2-10x speedup for large datasets)
- Model serialization for production deployment

## Prerequisites

Install the required dependencies:

```bash
pip install trimesh scikit-learn
```

For GPU acceleration (optional):

```bash
pip install torch
```

## Directory Structure

Create the following directory structure with your data:

```
examples/guardrails/
├── geometry_validation.py     # Main script
├── README.md                   # This file
└── data/                       # Your data (not included)
    ├── train_geometries/       # Known-good STL files for training
    │   ├── part_001.stl
    │   ├── part_002.stl
    │   └── ...
    └── test_geometries/        # STL files to validate
        ├── new_part_001.stl
        ├── new_part_002.stl
        └── ...
```

## Getting Started

### 1. Prepare Your Data

Place your known-good geometries in `data/train_geometries/` and geometries to validate in `data/test_geometries/`.

### 2. Run the Example

```bash
python geometry_validation.py
```

The script will:
1. Check for available optimizations (GPU, fast STL reader)
2. Train a guardrail model (or load existing model)
3. Validate test geometries
4. Report results with OK/WARN/REJECT classifications

### 3. Interpret Results

**Output Example:**

```
============================================================
Results: 100 geometries validated
  OK:      85 (85.0%)
  WARN:     12 (12.0%)
  REJECT:    3 (3.0%)
============================================================

✓ part_001.stl                            45.23%  [OK    ]
✓ part_002.stl                            62.18%  [OK    ]
⚠ part_015.stl                            96.42%  [WARN  ]
✗ part_089.stl                            99.87%  [REJECT]
```

- **OK**: Geometry is within the expected distribution (safe for inference)
- **WARN**: Geometry is unusual but may be acceptable (investigate)
- **REJECT**: Geometry is highly anomalous (likely invalid or OOD)

## Customization

### Adjust Detection Thresholds

Modify the `warn_pct` and `reject_pct` parameters in `geometry_validation.py`:

```python
guardrail = GeometryGuardrail(
    warn_pct=95.0,   # Flag top 5% as warnings (adjust for sensitivity)
    reject_pct=99.0, # Flag top 1% as rejections
)
```

**Guidelines:**
- **Conservative** (fewer false alarms): `warn_pct=95.0, reject_pct=99.0`
- **Balanced**: `warn_pct=90.0, reject_pct=95.0`
- **Aggressive** (catch more anomalies): `warn_pct=80.0, reject_pct=90.0`

### Enable GPU Acceleration

For large datasets (1000+ samples), GPU provides significant speedup:

```python
device = "cuda"  # or "cuda:0" for specific GPU
guardrail = GeometryGuardrail(device=device, ...)
```

**Performance:**
- CPU: ~100-500 ms per geometry (dense GMM)
- GPU: ~10-50 ms per geometry (2-10x faster)

### Handle Multiple Geometry Families

If your training data contains distinct families (e.g., brackets vs. gears), increase `n_components`:

```python
guardrail = GeometryGuardrail(
    n_components=3,  # Capture 3 distinct sub-populations
    ...
)
```

## Use Cases

### Quality Control in Manufacturing

Validate manufactured parts against design specifications:

```bash
# Train on validated production parts
python geometry_validation.py --train data/validated_parts/ --save qc_model.npz

# Check new batch
python geometry_validation.py --load qc_model.npz --test data/new_batch/
```

### Simulation Mesh Validation

Ensure simulation meshes are within expected ranges before expensive computations:

```python
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Train on validated meshes
guardrail = GeometryGuardrail.load("mesh_guardrail.npz")

# Check new mesh before simulation
result = guardrail.query([new_mesh])[0]

if result['status'] == 'REJECT':
    print(f"⚠ Mesh rejected (p={result['percentile']:.1f}%)")
    print("Skipping expensive simulation...")
else:
    run_simulation(new_mesh)
```

### Design Space Exploration

Flag novel designs during generative design or topology optimization:

```python
# Train on known-good designs
guardrail.fit(baseline_designs)

# Evaluate generated candidates
for candidate in generated_designs:
    result = guardrail.query([candidate])[0]
    
    if result['status'] != 'REJECT':
        candidates_to_simulate.append(candidate)
```

## Performance Tips

1. **Use GPU for large datasets**: Set `device="cuda"` for 1000+ samples
2. **Parallelize I/O**: Increase `n_workers` to match CPU cores
3. **Batch processing**: Use `fit_from_dir()` and `query_from_dir()` for directories
4. **Model caching**: Save and load models to avoid retraining

## Advanced Usage

### Programmatic Integration

```python
from pathlib import Path
from physicsnemo.experimental.guardrails import GeometryGuardrail

def validate_batch(stl_files, model_path):
    """Validate a batch of geometries."""
    guardrail = GeometryGuardrail.load(model_path)
    
    # Load and validate
    meshes = [trimesh.load(f, force="mesh") for f in stl_files]
    results = guardrail.query(meshes)
    
    # Filter out rejected geometries
    valid_files = [
        f for f, r in zip(stl_files, results)
        if r['status'] != 'REJECT'
    ]
    
    return valid_files

# Use in pipeline
valid_meshes = validate_batch(
    stl_files=list(Path("input/").glob("*.stl")),
    model_path=Path("guardrail.npz")
)
```

## Limitations

1. **Feature Design**: Uses hand-crafted geometric features (centroid, PCA, bounding box, etc.)
2. **Non-Invariance**: Sensitive to absolute position, orientation, and scale
3. **Training Data**: Performance depends on representative training samples
4. **Mesh Quality**: Requires valid, non-degenerate meshes with ≥50 vertices

## Troubleshooting

**Issue: "No valid STL files found"**

*Solution:* Verify STL files are valid, contain sufficient vertices, and paths are correct.

**Issue: Too many false positives**

*Solution:* Lower thresholds (`warn_pct`, `reject_pct`) or increase training data diversity.

**Issue: Slow performance**

*Solution:* Enable GPU (`device="cuda"`), increase `n_workers`, or reduce dataset size.

## References

- [Geometry Guardrails Documentation](../../docs/user_guide_guardrails.rst)
- [API Reference](../../physicsnemo/experimental/guardrails/)
- Reynolds, D. A. (2009). "Gaussian Mixture Models." Encyclopedia of Biometrics.

## Support

For questions or issues:
- File issues on the PhysicsNeMo GitHub repository
- Consult the full documentation at https://docs.nvidia.com/physicsnemo
