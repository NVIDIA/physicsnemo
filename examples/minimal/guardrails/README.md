# Geometry Guardrails Example

This example demonstrates how to use PhysicsNeMo's geometry guardrails for validating CAD/STL files against a distribution of known-good geometries.

## Overview

Geometry guardrails provide out-of-distribution (OOD) detection for 3D geometric data. They learn the distribution of "normal" geometries from training data and flag unusual or unexpected shapes at inference time.

## Prerequisites

Install the required dependencies:

```bash
pip install trimesh scikit-learn
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

Place your training geometries in `data/train_geometries/` and geometries to validate in `data/test_geometries/`.

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

### Handle Multiple Geometry Families

If your training data contains distinct families (e.g., brackets vs. gears), increase `n_components`:

```python
guardrail = GeometryGuardrail(
    n_components=3,  # Capture 3 distinct sub-populations
    ...
)
```

## Troubleshooting

**Issue: "No valid STL files found"**

*Solution:* Verify STL files are valid, contain sufficient vertices, and paths are correct.

**Issue: Too many false positives**

*Solution:* Lower thresholds (`warn_pct`, `reject_pct`).

**Issue: Slow performance**

*Solution:* Enable GPU (`device="cuda"`), increase `n_workers`, or reduce dataset size.

## Support

For questions or issues, file an issue on the PhysicsNeMo GitHub repository.
