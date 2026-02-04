# Geometry Guardrails

Out-of-distribution detection for geometric data using density-based anomaly detection.

## Overview

The geometry guardrails module provides tools for detecting anomalous geometric configurations in CAD models, simulation meshes, and other 3D shape data. It learns the distribution of "normal" geometries from training data and flags unusual or unexpected shapes at inference time.

**Key Features:**
- **Density-based anomaly detection** using Gaussian Mixture Models
- **Non-invariant features** that capture position, orientation, and scale
- **Three-level classification**: OK, WARN, REJECT based on configurable thresholds
- **Parallel processing** for efficient batch processing of STL files
- **Serialization support** for saving and loading fitted models
- **Comprehensive validation** with automatic schema compatibility checking

## Installation

This module requires optional dependencies:

```bash
pip install trimesh scikit-learn
```

**For GPU acceleration** (optional):

```bash
pip install trimesh scikit-learn torch
```

GPU acceleration requires NVIDIA CUDA and provides 2-10x speedup for large datasets (1000+ samples).

**For fast STL loading** (optional, experimental):

The geometry guardrails include an adapter for optional Rust-based STL readers that can provide 5-10x faster file I/O. The code will automatically detect and use any compatible reader installed in your environment, with graceful fallback to `trimesh`.

This is recommended for processing large batches of STL files (100+ files), but **not required**. See [Fast I/O](#fast-io-optional) below for implementation details if you want to build your own accelerator.

## Quick Start

### Basic Usage

```python
import trimesh
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Load or create training meshes
train_meshes = [
    trimesh.load("part_001.stl", force="mesh"),
    trimesh.load("part_002.stl", force="mesh"),
    # ... more training data
]

# Create and fit guardrail
guardrail = GeometryGuardrail(
    n_components=1,      # Number of Gaussian components (1 = single Gaussian)
    warn_pct=95.0,       # Flag geometries above 95th percentile as WARN
    reject_pct=99.0,     # Flag geometries above 99th percentile as REJECT
    covariance_type="full",
    random_state=42,
)
guardrail.fit(train_meshes)

# Query new geometries
test_meshes = [trimesh.load("new_part.stl", force="mesh")]
results = guardrail.query(test_meshes)

for res in results:
    print(f"Status: {res['status']}, Percentile: {res['percentile']:.2f}")
# Output: Status: OK, Percentile: 45.23
```

### Working with STL Directories

For large datasets stored as STL files, use the directory-based API with automatic parallel processing:

```python
from pathlib import Path
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Fit from directory of STL files
guardrail = GeometryGuardrail(n_components=2, warn_pct=95.0, reject_pct=99.0)
guardrail.fit_from_dir(
    Path("/path/to/training/stl/files"),
    n_workers=8,      # Use 8 CPU cores
    chunksize=16,     # Process 16 files per worker task
)

# Query entire directory
results = guardrail.query_from_dir(
    Path("/path/to/test/stl/files"),
    n_workers=8,
)

# Filter for flagged geometries
flagged = [r for r in results if r["status"] != "OK"]
print(f"Flagged {len(flagged)} / {len(results)} geometries:")
for r in flagged:
    print(f"  {r['name']}: {r['status']} (p={r['percentile']:.1f}%)")
```

**Fast STL Loading** (5-10x speedup):

If you've installed the optional Rust STL reader:

```python
from physicsnemo.experimental.guardrails.geometry import is_fast_reader_available

# Check if fast reader is available
if is_fast_reader_available():
    print("Using fast Rust-based STL reader")
else:
    print("Using trimesh (install stlreader for 5-10x faster I/O)")

# Use fast reader for batch loading
guardrail.fit_from_dir(
    Path("/path/to/stl/files"),
    n_workers=16,            # More workers benefit from faster I/O
    use_fast_reader=True,    # Enable fast Rust reader
)
```

### Saving and Loading Models

```python
from pathlib import Path
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Save fitted guardrail
guardrail.save(Path("guardrail.npz"))

# Load for inference (with automatic compatibility checking)
loaded_guardrail = GeometryGuardrail.load(Path("guardrail.npz"))
results = loaded_guardrail.query(test_meshes)
```

### GPU Acceleration

For improved performance on large datasets, use GPU acceleration:

```python
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Create guardrail with GPU support (requires PyTorch and CUDA)
guardrail_gpu = GeometryGuardrail(
    n_components=2,
    warn_pct=95.0,
    reject_pct=99.0,
    device="cuda",  # Use GPU
    random_state=42,
)

# Fit on GPU (faster for large datasets)
guardrail_gpu.fit(train_meshes)

# Fast batch inference
results = guardrail_gpu.query(test_meshes)
```

**When to use GPU:**
- ✅ Dataset size > 1000 samples
- ✅ Batch inference on 100+ geometries
- ✅ Latency-critical applications (<100ms queries)
- ✅ Iterative refinement workflows

**When to use CPU:**
- ✅ Dataset size < 100 samples (CPU may be faster)
- ✅ No GPU available
- ✅ Simplicity preferred over speed

**Device Options:**
```python
device="cpu"       # CPU-only (default, always available)
device="cuda"      # Default GPU
device="cuda:0"    # Specific GPU device
```

**Loading Models on Different Devices:**
```python
# Save on CPU
guardrail_cpu = GeometryGuardrail(device="cpu")
guardrail_cpu.fit(train_meshes)
guardrail_cpu.save(Path("model.npz"))

# Load on GPU for fast inference
guardrail_gpu = GeometryGuardrail.load(Path("model.npz"), device="cuda")
results = guardrail_gpu.query(test_meshes)  # Fast GPU inference
```

## How It Works

### Feature Extraction

The guardrail extracts **22 non-invariant geometric features** from each mesh:

| Feature Category | Description | Count |
|-----------------|-------------|-------|
| Centroid | 3D position of geometry center | 3 |
| PCA Axes | First two principal component directions | 6 |
| PCA Eigenvalues | Variance along principal axes | 3 |
| Bounding Box | Axis-aligned extents (width, height, depth) | 3 |
| Second Moments | Variance per coordinate axis | 3 |
| Total Surface Area | Sum of all face areas | 1 |
| Projected Areas | Area projections onto XY, XZ, YZ planes | 3 |

**Important**: Features are intentionally **not invariant** to transformations. This allows detection of geometries that differ in:
- **Translation** (absolute position in space)
- **Rotation** (absolute orientation)
- **Scale** (absolute size)

### Density Modeling

A **Gaussian Mixture Model (GMM)** learns the probability density \( p(\mathbf{x}) \) over the feature space:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \mu_k, \Sigma_k)
$$

where:
- \( K \) is the number of components (`n_components`)
- \( \pi_k \) are mixture weights
- \( \mu_k, \Sigma_k \) are mean and covariance for component \( k \)

For a new geometry with features \( \mathbf{x} \), the **anomaly score** is:

$$
s(\mathbf{x}) = -\log p(\mathbf{x} | \theta)
$$

Higher scores indicate lower likelihood (more anomalous).

### Classification

Anomaly scores are converted to **empirical percentiles** relative to the training distribution. Given percentile \( p \):

- **OK**: \( p < \text{warn\_pct} \) — Typical geometry
- **WARN**: \( \text{warn\_pct} \leq p < \text{reject\_pct} \) — Unusual geometry (investigate)
- **REJECT**: \( p \geq \text{reject\_pct} \) — Highly anomalous (likely OOD)

## API Reference

### Main Classes

#### `GeometryGuardrail`

Main user-facing API for geometry OOD detection.

**Constructor:**
```python
GeometryGuardrail(
    n_components=1,          # Number of GMM components
    warn_pct=95.0,          # Warning threshold percentile
    reject_pct=99.0,        # Rejection threshold percentile
    covariance_type="full", # GMM covariance type
    random_state=0,         # Random seed for reproducibility
)
```

**Methods:**
- `fit(meshes: list[trimesh.Trimesh])` — Fit from mesh objects
- `fit_from_dir(stl_dir: Path, **kwargs)` — Fit from STL directory
- `query(meshes: list[trimesh.Trimesh])` — Query mesh objects
- `query_from_dir(stl_dir: Path, **kwargs)` — Query STL directory
- `save(path: Path)` — Save fitted model
- `load(path: Path)` — Load fitted model (class method)

#### `GeometryDensityModel`

Low-level density estimation using GMM.

**Methods:**
- `fit(X: np.ndarray)` — Fit density model
- `score(X: np.ndarray)` — Compute anomaly scores
- `percentiles(scores: np.ndarray)` — Convert scores to percentiles

### Utility Functions

#### `extract_features(mesh: trimesh.Trimesh) -> np.ndarray`

Extract 22-dimensional feature vector from a mesh.

#### `validate_mesh(mesh: trimesh.Trimesh, min_verts: int = 50)`

Validate mesh integrity before feature extraction.

#### `load_features_from_dir(stl_dir: Path, n_workers=None, chunksize=8)`

Parallel feature extraction from STL directory.

## Configuration Guidelines

### GPU vs. CPU Performance

Choose the appropriate backend based on your dataset size and hardware:

| Dataset Size | CPU Time | GPU Time | Speedup | Recommendation |
|--------------|----------|----------|---------|----------------|
| < 100 samples | ~1s | ~2s | 0.5x | **Use CPU** (GPU overhead) |
| 100-1000 samples | 5-30s | 3-10s | 2-3x | **Either** (marginal benefit) |
| 1000-10000 samples | 1-5min | 15-60s | 3-5x | **Use GPU** |
| > 10000 samples | 10+ min | 1-3min | 5-10x | **Use GPU** |

**Performance Tips:**
- GPU shines for batch inference, not necessarily fitting
- Transfer overhead dominates for small datasets
- Use CPU for interactive single-query workflows
- Use GPU for production batch processing

### Choosing `n_components`

- **n_components=1**: Single Gaussian, fastest, assumes unimodal distribution
- **n_components=2-5**: Captures multimodal distributions (e.g., multiple part families)
- **n_components > 5**: Risk of overfitting on small datasets

**Recommendation**: Start with 1, increase if training data has distinct subgroups.

### Setting Thresholds

- **warn_pct=95.0, reject_pct=99.0**: Conservative (fewer false alarms)
- **warn_pct=90.0, reject_pct=95.0**: Balanced
- **warn_pct=80.0, reject_pct=90.0**: Aggressive (catches more anomalies, more false positives)

**Recommendation**: Tune thresholds based on your application's tolerance for false positives vs. missed anomalies.

### Covariance Types

- **"full"**: Each component has its own covariance matrix (most flexible, slowest)
- **"tied"**: All components share one covariance matrix (faster, less flexible)
- **"diag"**: Diagonal covariance (assumes feature independence)
- **"spherical"**: Isotropic covariance (fastest, least flexible)

**Recommendation**: Use "full" unless you have a very large dataset or need faster inference.

## Limitations

1. **Feature Design**: Features are hand-crafted and may not capture all relevant geometric properties for your application.

2. **Non-Invariance**: The guardrail is sensitive to absolute position, orientation, and scale. If you need invariance, consider pre-processing meshes (centering, alignment, normalization).

3. **Mesh Quality**: Requires valid, non-degenerate meshes with sufficient vertices (≥50 by default).

4. **Training Data**: Performance depends on having representative training data covering the expected distribution.

5. **Interpretability**: Anomaly scores don't explain *why* a geometry is flagged. Further analysis is needed to diagnose issues.

## Examples

### Example 1: Additive Manufacturing Quality Control

```python
from pathlib import Path
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Fit on known-good parts from production
guardrail = GeometryGuardrail(n_components=1, warn_pct=95.0, reject_pct=99.0)
guardrail.fit_from_dir(Path("production_parts/good/"), n_workers=16)

# Monitor new parts
results = guardrail.query_from_dir(Path("production_parts/new_batch/"), n_workers=16)

# Flag for manual inspection
for r in results:
    if r["status"] == "REJECT":
        print(f"REJECT: {r['name']} (p={r['percentile']:.1f}%) - inspect immediately")
    elif r["status"] == "WARN":
        print(f"WARN: {r['name']} (p={r['percentile']:.1f}%) - may need review")
```

### Example 2: Simulation Mesh Validation

```python
import trimesh
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Fit on validated simulation meshes
train_meshes = [trimesh.load(f"validated_mesh_{i:03d}.stl") for i in range(100)]
guardrail = GeometryGuardrail(n_components=2)  # Two mesh families
guardrail.fit(train_meshes)

# Check automatically generated meshes
generated_meshes = [trimesh.load(f"generated_mesh_{i:03d}.stl") for i in range(20)]
results = guardrail.query(generated_meshes)

# Statistics
ok_count = sum(1 for r in results if r["status"] == "OK")
print(f"Mesh quality: {ok_count}/{len(results)} passed validation")
```

## Fast I/O (Optional)

The geometry guardrails include an adapter (`fast_stl.py`) that can utilize optional Rust-based STL readers for 5-10x faster file I/O. This is useful for large-scale batch processing (1000+ files).

### Implementation Details

If you want to implement your own fast reader, it should provide a module with:

```python
def load_stl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load STL file and return (vertices, faces, normals, areas).
    
    Returns
    -------
    vertices : np.ndarray, shape (N, 3)
        Vertex coordinates
    faces : np.ndarray, shape (M, 3)
        Face indices (int64)
    normals : np.ndarray, shape (M, 3)
        Face normal vectors (unit vectors)
    areas : np.ndarray, shape (M,)
        Face areas (float64)
    """
```

The adapter will automatically detect and use any module named `stlreader` with this interface, falling back to `trimesh` if not found.

### Reference Implementation

A reference Rust implementation is available in the repository at `stlreader/` (not built by default). Key features:
- Uses `stl_io` crate for fast binary/ASCII parsing
- Precomputes normals and areas during load
- Integrates with NumPy via PyO3
- 5-10x faster than pure Python readers

To build (requires Rust toolchain):
```bash
cd stlreader
pip install maturin
maturin develop --release
```

This is **entirely optional** and intended for users with high-performance requirements.

## See Also

- **Mesh I/O**: `physicsnemo.mesh` for advanced mesh operations
- **Data Validation**: Other guardrails in `physicsnemo.experimental.guardrails`
- **Trimesh Docs**: https://trimsh.org/ for mesh manipulation

## Citation

If you use this module in your research, please cite the PhysicsNemo framework:

```bibtex
@software{physicsnemo,
  title = {PhysicsNemo: Physics-Informed Machine Learning Framework},
  author = {{NVIDIA Corporation}},
  year = {2026},
  url = {https://github.com/NVIDIA/physicsnemo}
}
```

## Support

For issues, questions, or contributions:
- File issues on the PhysicsNemo GitHub repository
- Consult the full documentation at https://docs.nvidia.com/physicsnemo
- Join the NVIDIA Developer Forums

## License

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0.
