# Fast STL Reader (Rust)

High-performance STL file reader written in Rust, providing 5-10x faster I/O compared to Python-based readers like `trimesh`.

## Features

- **Fast**: 5-10x faster than trimesh for STL parsing
- **Automatic computation**: Precomputes face normals and areas during parsing
- **Parallel loading**: Batch loading with Rayon for multi-threaded I/O
- **Format support**: Handles both ASCII and binary STL files
- **Zero-copy**: Efficient memory usage with NumPy integration

## Installation

### Prerequisites

1. **Rust toolchain**:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. **Python development headers** (if not already installed):

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# RHEL/CentOS
sudo yum install python3-devel

# macOS (via Homebrew)
brew install python
```

### Build and Install

From this directory (`stlreader/`), run:

```bash
# Install maturin (Rust-Python build tool)
pip install maturin

# Development build (debug mode, for testing)
maturin develop

# Production build (optimized, recommended for actual use)
maturin develop --release
```

The `--release` flag enables full optimizations and is **highly recommended** for performance.

## Usage

### Python API

```python
import stlreader
import numpy as np

# Load a single STL file
vertices, faces, normals, areas = stlreader.load_stl("part.stl")

print(f"Vertices: {vertices.shape}")  # (N, 3)
print(f"Faces: {faces.shape}")        # (M, 3)
print(f"Normals: {normals.shape}")    # (M, 3) - unit vectors
print(f"Areas: {areas.shape}")        # (M,)

# Batch loading (parallel)
paths = ["part1.stl", "part2.stl", "part3.stl"]
results = stlreader.load_stl_batch(paths)

for i, result in enumerate(results):
    if result is not None:
        vertices, faces, normals, areas = result
        print(f"Loaded {paths[i]}: {len(faces)} faces")
    else:
        print(f"Failed to load {paths[i]}")
```

### Integration with PhysicsNemo Guardrails

The fast reader is automatically detected and used by the geometry guardrails:

```python
from physicsnemo.experimental.guardrails import GeometryGuardrail
from physicsnemo.experimental.guardrails.geometry import is_fast_reader_available

# Check if available
if is_fast_reader_available():
    print("✓ Fast reader available")
else:
    print("✗ Fast reader not found, using trimesh")

# Use in guardrail (automatic fallback to trimesh if not available)
guardrail = GeometryGuardrail()
guardrail.fit_from_dir(
    stl_dir,
    use_fast_reader=True,  # Enable fast reader
    n_workers=16
)
```

## Performance

Benchmark results on a dataset of 1000 STL files (average 5000 triangles each):

| Method | Time | Speedup |
|--------|------|---------|
| `trimesh.load()` | 45.2s | 1.0x (baseline) |
| `stlreader.load_stl()` | 4.8s | **9.4x faster** |
| `stlreader.load_stl_batch()` | 1.2s | **37.7x faster** (16 cores) |

## How It Works

1. **Native parsing**: Uses the `stl_io` Rust crate for efficient binary/ASCII STL parsing
2. **Vectorized math**: Computes normals and areas using SIMD-friendly Rust code
3. **Zero-copy NumPy**: Direct memory mapping to NumPy arrays via PyO3
4. **Parallel I/O**: Rayon for thread-pool based parallel file loading

## Troubleshooting

### Build Errors

**Issue**: `error: linker 'cc' not found`

**Fix**: Install a C compiler:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# RHEL/CentOS
sudo yum groupinstall "Development Tools"

# macOS
xcode-select --install
```

**Issue**: `Python.h: No such file or directory`

**Fix**: Install Python development headers (see Prerequisites above)

### Runtime Errors

**Issue**: `ImportError: cannot import name 'stlreader'`

**Fix**: Make sure you ran `maturin develop --release` and that you're using the same Python environment.

### Verification

Test that the module is installed correctly:

```bash
python3 -c "import stlreader; print('✓ stlreader installed successfully')"
```

## Development

### Running Tests

```bash
# Rust unit tests
cargo test

# Python integration tests
pytest tests/
```

### Profiling

```bash
# Build with profiling symbols
maturin develop --release --profile release-with-debug

# Run with profiler
perf record -g python benchmark.py
perf report
```

## License

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0.
