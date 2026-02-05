# Geometry Guardrails

Out-of-distribution detection for geometric data using density-based anomaly detection.

## Overview

The geometry guardrails module provides tools for detecting anomalous geometric
configurations in CAD models, simulation meshes, and other 3D shape data. It
learns the distribution of "normal" geometries from training data and flags
unusual or unexpected shapes at inference time.

**Key Features:**

- **Density-based anomaly detection** using Gaussian Mixture Models
- **Non-invariant features** that capture position, orientation, and scale
- **Three-level classification**: OK, WARN, REJECT based on configurable
  thresholds
- **Parallel processing** for efficient batch processing of STL files
- **Serialization support** for saving and loading fitted models
- **Comprehensive validation** with automatic schema compatibility checking

## Installation

This module requires optional dependencies:

```bash
pip install trimesh scikit-learn
```

**For fast STL loading** (optional, experimental):

The geometry guardrails include an adapter for optional Rust-based STL readers
that can provide faster file I/O. The code will automatically detect and use
any compatible reader installed in your environment, with graceful fallback to
`trimesh`.

This is recommended for processing large batches of STL files, but **not
required**. See [Fast I/O](#fast-io-optional) below for implementation details
if you want to build your own accelerator.

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
    warn_pct=99.0,       # Flag geometries above 99th percentile as WARN
    reject_pct=99.9,     # Flag geometries above 99.9th percentile as REJECT
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

For large datasets stored as STL files, use the directory-based API with
automatic parallel processing:

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

**Fast STL Loading**:

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

GPU acceleration is most beneficial for large datasets and batch inference. For
small datasets and batches, CPU may be faster due to transfer overhead.

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

A **Gaussian Mixture Model (GMM)** learns the probability density
\( p(\mathbf{x}) \) over the feature space:

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

Anomaly scores are converted to **empirical percentiles** relative to the
training distribution. Given percentile \( p \):

- **OK**: \( p < \text{warn\_pct} \) — Typical geometry
- **WARN**: \( \text{warn\_pct} \leq p < \text{reject\_pct} \) — Unusual
  geometry (investigate)
- **REJECT**: \( p \geq \text{reject\_pct} \) — Highly anomalous (likely OOD)

## Examples

### Additive Manufacturing Quality Control

```python
from pathlib import Path
from physicsnemo.experimental.guardrails import GeometryGuardrail

# Fit on known-good parts from production
guardrail = GeometryGuardrail(n_components=1, warn_pct=99.0, reject_pct=99.9)
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

## Fast I/O (Optional)

The geometry guardrails include an adapter (`fast_stl.py`) that can utilize
optional Rust-based STL readers for faster file I/O. This is useful for
large-scale batch processing or large STL files.

A reference Rust implementation is available in the repository at `stlreader/`
(not built by default). Key features:

- Uses `stl_io` crate for fast binary/ASCII parsing
- Precomputes normals and areas during load
- Integrates with NumPy via PyO3

To build (requires Rust toolchain):

```bash
cd stlreader
pip install maturin
maturin develop --release
```

This is **entirely optional** and intended for users with high-performance
requirements.

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

### 3. **Expanded GPU Support**

Extend GPU acceleration beyond GMM to cover feature extraction, PCE density
estimation, and batch STL loading.

### 4. **Advanced Anomaly Detection Methods**

Implement additional density estimation methods: Kernel Density Estimation,
Variational Autoencoders, Normalizing Flows, and deep learning approaches.

### 5. **Interpretability & Explainability**

Provide feature importance analysis and visual diagnostics to help users
understand why specific geometries were flagged as anomalous.

### 7. **Multi-Modal & Multi-Physics**

Extend guardrails to jointly model geometry, material properties, boundary
conditions, simulation results, and manufacturing metadata for comprehensive
anomaly detection.

---

**How to Contribute:** Fork the repository, implement enhancements with tests
and documentation following PhysicsNemo coding standards (`.cursor/rules/`),
and submit a pull request. For questions, open an issue on GitHub.

## Support

For issues, questions, or contributions:

- File issues on the PhysicsNemo GitHub repository
- Consult the full documentation at <https://docs.nvidia.com/physicsnemo>
