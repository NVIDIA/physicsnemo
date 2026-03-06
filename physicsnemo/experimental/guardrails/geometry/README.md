# Geometry Guardrails

Out-of-distribution detection for geometric data using density-based anomaly detection.

## Overview

The geometry guardrails module provides tools for detecting anomalous geometric
configurations in CAD models, simulation meshes, and other 3D shape data. It
learns the distribution of "normal" geometries from training data and flags
unusual or unexpected shapes at inference time.

**Key Features:**

- **Density-based anomaly detection** using Gaussian Mixture Models (GMM) or
  Polynomial Chaos Expansion (PCE)
- **Non-invariant features** that capture position, orientation, and scale
- **Three-level classification**: OK, WARN, REJECT based on configurable
  thresholds
- **GPU acceleration** for both GMM and PCE methods (PyTorch-based)
- **Parallel processing** for efficient batch processing of STL files
- **Serialization support** for saving and loading fitted models
- **Comprehensive validation** with automatic schema compatibility checking

## Installation

This module requires optional dependencies:

```bash
pip install pyvista
```

## Quick Start

### Choosing a Method

The guardrail supports two density estimation methods:

- **GMM (Gaussian Mixture Model)**: Default method, best for unimodal or
  multimodal Gaussian-like distributions. Fast and interpretable.
- **PCE (Polynomial Chaos Expansion)**: Better for capturing non-Gaussian
  distributions and higher-order correlations. More computationally intensive
  but can model complex feature relationships.

Use `method="gmm"` (default) for most cases, or `method="pce"` when you need
to capture non-Gaussian patterns in your geometry distribution.

### Basic Usage

```python
import pyvista as pv
from physicsnemo.mesh.io import from_pyvista
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Load or create training meshes
train_meshes = [
    from_pyvista(pv.read("part_001.stl")),
    from_pyvista(pv.read("part_002.stl")),
    # ... more training data
]

# Create and fit guardrail
guardrail = GeometryGuardrail(
    method="gmm",        # Use GMM (or "pce" for Polynomial Chaos Expansion)
    n_components=1,      # Number of Gaussian components (1 = single Gaussian)
    warn_pct=99.0,       # Flag geometries above 99th percentile as WARN
    reject_pct=99.9,     # Flag geometries above 99.9th percentile as REJECT
)
guardrail.fit(train_meshes)

# Query new geometries
test_meshes = [from_pyvista(pv.read("new_part.stl"))]
results = guardrail.query(test_meshes)

for res in results:
    print(f"Status: {res['status']}, Percentile: {res['percentile']:.2f}")
# Output: Status: OK, Percentile: 45.23
```

### Working with STL Directories

For large datasets stored as STL files, use the directory-based API with
automatic parallel processing:

```python
from pathlib import Path
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Fit from directory of STL files
guardrail = GeometryGuardrail(
    method="gmm",
    n_components=2,
    warn_pct=99.0,
    reject_pct=99.9,
)
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

**Batch Processing**:

For large datasets, use the directory-based API with automatic parallel processing:

```python
guardrail.fit_from_dir(
    Path("/path/to/stl/files"),
    n_workers=16,            # Use 16 CPU cores for parallel processing
)
```

### Saving and Loading Guardrails

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

Both GMM and PCE methods support GPU acceleration via PyTorch:

```python
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Create guardrail with GPU support (requires PyTorch and CUDA)
guardrail_gpu = GeometryGuardrail(
    method="gmm",        # Both "gmm" and "pce" support GPU
    n_components=2,
    warn_pct=95.0,
    reject_pct=99.0,
    device="cuda",       # Use GPU
    random_state=42,
)

# Fit on GPU (faster for large datasets)
guardrail_gpu.fit(train_meshes)

# Fast batch inference on GPU
results = guardrail_gpu.query(test_meshes)
```

**Note**: Multiprocessing workers always run on CPU to avoid OOM issues. Features
are extracted on CPU in parallel, then moved to the specified device (CPU or GPU)
in the main process for density model training and inference.

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

**Important**: Features are intentionally **not invariant** to transformations.
This allows detection of geometries that differ in:

- **Translation** (absolute position in space)
- **Rotation** (absolute orientation)
- **Scale** (absolute size)

### Density Modeling

The guardrail supports two density estimation methods:

**1. Gaussian Mixture Model (GMM)** - Default method, learns the probability density
\( p(\mathbf{x}) \) over the feature space:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \mu_k, \Sigma_k)
$$

where \( K \) is the number of components (`n_components`), \( \pi_k \) are mixture
weights, and \( \mu_k, \Sigma_k \) are mean and covariance for component \( k \).

**2. Polynomial Chaos Expansion (PCE)** - Alternative method using polynomial
basis functions for density estimation. Useful for capturing non-Gaussian
distributions and higher-order correlations.

For a new geometry with features \( \mathbf{x} \), the **anomaly score** is:

$$
s(\mathbf{x}) = -\log p(\mathbf{x} | \theta) \quad \text{(GMM)}
$$

or Mahalanobis distance (PCE). Higher scores indicate lower likelihood (more anomalous).

Both methods use PyTorch and support GPU acceleration.

### Classification

Anomaly scores are converted to **empirical percentiles** relative to the
training distribution. Given percentile \( p \):

- **OK**: \( p < \text{warn\_pct} \) — Typical geometry
- **WARN**: \( \text{warn\_pct} \leq p < \text{reject\_pct} \) — Unusual
  geometry (investigate)
- **REJECT**: \( p \geq \text{reject\_pct} \) — Highly anomalous (likely OOD)

## TODO: Future Enhancements (Contributions Welcome!)

We welcome contributions to advance the geometry guardrails module. Key areas
for future work:

### 1. **Advanced Shape Descriptors**

Expand beyond basic geometric features to include spectral descriptors
(Laplacian eigenfunctions), topological features, curvature statistics, and
graph-based representations. Support configurable feature sets and custom
extractors.

### 2. **Optional Invariance**

Add user-configurable invariance to rotation, scale, and translation. Currently
all features are non-invariant.

### 3. **Enhanced Feature Extraction**

Optimize feature extraction for GPU acceleration. Currently, feature extraction
runs on CPU in multiprocessing workers, with features moved to GPU for density
modeling. Direct GPU feature extraction could further improve performance.

### 4. Advanced Anomaly Detection Methods

Implement additional density estimation methods: Kernel Density Estimation,
Variational Autoencoders, Normalizing Flows, and deep learning approaches.

**How to Contribute:** For guidance on contributing to PhysicsNeMo, please refer to the
[contributing guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CONTRIBUTING.md).

## Support

For issues, questions, or contributions:

- File issues on the PhysicsNemo GitHub repository
- Consult the full documentation at <https://docs.nvidia.com/physicsnemo>
