# PhysicsNeMo-Mesh Tutorials

This directory contains a series of progressive tutorials introducing
**PhysicsNeMo-Mesh** - NVIDIA's GPU-accelerated mesh processing library for
physics-AI applications.

## What is PhysicsNeMo-Mesh?

PhysicsNeMo-Mesh is a PyTorch-based library for working with simplicial meshes
(point clouds, curves, surfaces, volumes) in a unified, dimensionally-generic
framework. Key features include:

- **GPU-Accelerated**: All operations vectorized with PyTorch, run natively on CUDA
- **Dimensionally Generic**: Works with n-D manifolds embedded in m-D spaces
- **TensorDict Integration**: Structured data management with automatic device handling
- **Differentiable**: Seamless integration with PyTorch autograd
- **Flexible Data**: Arbitrary-rank tensor fields on points, cells, or globally

For the complete feature list, see the
[physicsnemo.mesh README](../../../physicsnemo/mesh/README.md).

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, not required)

## Installation

```bash
pip install nvidia-physicsnemo pyvista[all,trame] matplotlib jupyter
```

Or install from the repository:

```bash
pip install -e ".[mesh]"
```

## Tutorial Overview

<!-- markdownlint-disable MD013 -->
| Tutorial | Topic | What You'll Learn |
|----------|-------|-------------------|
| **1. Getting Started** | Core concepts | Mesh structure, data attachment, GPU acceleration |
| **2. Operations** | Mesh manipulation | Transformations, subdivision, slicing, merging |
| **3. Discrete Calculus** | Mathematical operators | Gradients, divergence, curl, curvature |
| **4. Neighbors & Spatial** | Queries | Adjacency, BVH, sampling, interpolation |
| **5. Quality & Repair** | Mesh health | Validation, quality metrics, repair |
| **6. ML Integration** | Production workflows | Performance, batching, torch.compile |
<!-- markdownlint-enable MD013 -->

## Running the Tutorials

### Option 1: Jupyter Notebook

```bash
cd examples/minimal/mesh
jupyter notebook
```

Then open any `tutorial_*.ipynb` file.

### Option 2: JupyterLab

```bash
cd examples/minimal/mesh
jupyter lab
```

### Option 3: VS Code / Cursor

Open the `.ipynb` files directly - they work with the built-in notebook support.

## Tutorial Contents

### Tutorial 1: Getting Started

**File**: `tutorial_1_getting_started.ipynb`

Learn the core concepts - a `Mesh` is just 5 fields: 2 for geometry, 3 for data.

- The 5-field data structure (points, cells, point_data, cell_data, global_data)
- Creating meshes from scratch
- Loading from PyVista and built-in primitives
- Attaching scalar, vector, and tensor data
- Visualization with `.draw()`
- GPU acceleration with `.to("cuda")`
- Autograd integration

### Tutorial 2: Operations and Transformations

**File**: `tutorial_2_operations.ipynb`

Learn mesh manipulation operations.

- Geometric transformations (translate, rotate, scale, transform)
- Subdivision schemes (linear, Loop, Butterfly)
- Slicing (slice_cells, slice_points)
- Merging multiple meshes
- Boundary and facet extraction
- Data conversion (cell_data_to_point_data, point_data_to_cell_data)
- Topology checks (is_watertight, is_manifold)

### Tutorial 3: Discrete Calculus and Differential Geometry

**File**: `tutorial_3_calculus.ipynb`

Learn mathematical operations on meshes.

- Computing gradients (LSQ and DEC methods)
- Divergence and curl
- Gaussian and mean curvature
- Intrinsic vs extrinsic derivatives
- Vector calculus identities
- Physics-informed feature extraction

### Tutorial 4: Neighbors, Adjacency, and Spatial Queries

**File**: `tutorial_4_neighbors_spatial.ipynb`

Learn about mesh queries for GNN-style processing.

- Topological neighbors (point-to-points, cell-to-cells)
- Sparse adjacency encoding
- BVH construction and queries
- Point containment
- Random point sampling
- Data interpolation at query points

### Tutorial 5: Quality, Validation, and Repair

**File**: `tutorial_5_quality_repair.ipynb`

Learn mesh quality assessment and repair.

- Quality metrics (aspect ratio, angles, quality score)
- Mesh statistics
- Validation (detect errors)
- Repair operations (clean, remove duplicates, fix orientation)
- Manifold and watertight checking

### Tutorial 6: Integration with ML Workflows

**File**: `tutorial_6_ml_integration.ipynb`

Learn to use PhysicsNeMo-Mesh in production ML pipelines.

- Performance comparison with PyVista/VTK
- Batching with padding for torch.compile
- Feature extraction for ML models
- Boundary condition handling
- End-to-end CAE preprocessing workflow
- torch.compile compatibility

## Assets

The `assets/` directory contains pre-saved meshes for use in tutorials:

- `bunny.pt` - Stanford bunny mesh (coarse, use `.subdivide()` for detail)

## Additional Resources

- [PhysicsNeMo Documentation](https://docs.nvidia.com/deeplearning/physicsnemo)
- [physicsnemo.mesh Module Reference](../../../physicsnemo/mesh/README.md)
- [PyTorch TensorDict](https://github.com/pytorch/tensordict)
- [PyVista](https://pyvista.org/)
