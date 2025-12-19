# PhysicsNeMo Code Structure

**Version:** 1.4.0a0 (v2.0-refactor branch)  
**Repository:** https://github.com/NVIDIA/physicsnemo

---

## Table of Contents

- [Overview](#overview)
- [Project Root Structure](#project-root-structure)
- [Core Module Structure](#core-module-structure)
- [Key Components](#key-components)
  - [Models](#models)
  - [Neural Network Layers](#neural-network-layers)
  - [Data Pipelines](#data-pipelines)
  - [Metrics](#metrics)
  - [Distributed Computing](#distributed-computing)
  - [Active Learning](#active-learning)
  - [Utils](#utils)
- [Examples](#examples)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Overview

PhysicsNeMo (Physics-informed Neural Models) is NVIDIA's comprehensive framework for building physics-informed machine learning models. It provides a modular architecture for scientific computing, weather forecasting, computational fluid dynamics (CFD), and other physics-based applications.

---

## Project Root Structure

```
physicsnemo/
├── physicsnemo/              # Main source code package
├── examples/                 # Example implementations and use cases
├── test/                     # Comprehensive test suite
├── docs/                     # Documentation and tutorials
├── CODING_STANDARDS/         # Coding guidelines and best practices
├── pyproject.toml           # Python project configuration
├── Dockerfile               # Container configuration
├── README.md                # Project overview
├── CHANGELOG.md             # Version history
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE.txt              # License information
├── SECURITY.md              # Security policies
├── FAQ.md                   # Frequently asked questions
└── v2.0-MIGRATION-GUIDE.md  # Migration guide for v2.0
```

---

## Core Module Structure

### `physicsnemo/` - Main Package

```
physicsnemo/
├── __init__.py
├── active_learning/         # Active learning workflows
├── core/                    # Core framework functionality
├── datapipes/              # Data loading and preprocessing
├── deploy/                 # Model deployment utilities
├── distributed/            # Distributed computing support
├── domain_parallel/        # Domain parallelism implementations
├── experimental/           # Experimental features
├── metrics/                # Evaluation metrics
├── models/                 # Neural network architectures
├── nn/                     # Neural network building blocks
└── utils/                  # Utility functions and helpers
```

---

## Key Components

### Models

**Location:** `physicsnemo/models/`

PhysicsNeMo provides 20+ state-of-the-art model architectures:

#### Weather & Climate Models
- **AFNO** (`afno/`) - Adaptive Fourier Neural Operator
  - `afno.py` - Base AFNO implementation
  - `modafno.py` - Mixture-of-Depths AFNO
  - `modembed.py` - Mixture-of-Depth embeddings
  - `distributed/` - Distributed AFNO variants

- **FengWu** (`fengwu/`) - FengWu weather forecasting model
- **Pangu** (`pangu/`) - Pangu weather prediction model
- **GraphCast** (`graphcast/`) - Google DeepMind's weather model
  - Graph-based neural network for weather forecasting
  - Icosahedral mesh utilities
  - Custom graph operations

- **DLWP** (`dlwp/`) - Deep Learning Weather Prediction
- **DLWP-HEALPix** (`dlwp_healpix/`) - HEALPix-based weather models
  - `HEALPixUNet.py` - U-Net on HEALPix grids
  - `HEALPixRecUNet.py` - Recurrent U-Net variant
  - `dlwp_healpix_layers/` - Specialized HEALPix layers

#### Generative Models
- **Diffusion** (`diffusion/`) - Diffusion models for physics
  - `dhariwal_unet.py` - Dhariwal's U-Net architecture
  - `song_unet.py` - Song's U-Net architecture
  - `unet.py` - Generic U-Net implementation
  - `preconditioning.py` - Noise preconditioning
  - `sampling/` - Deterministic and stochastic samplers
  - `training_utils/` - Training utilities

- **TopoDiff** (`topodiff/`) - Topology-aware diffusion
- **CorrDiff** - Correlation-aware diffusion (via `corrdiff_utils.py`)

#### CFD & Engineering Models
- **DOMINO** (`domino/`) - Deep Operator Network for Multiphysics
  - Geometry representations
  - Solution encodings
  - VTK file utilities

- **FIGConvNet** (`figconvnet/`) - Feature-Informed Graph Convolutional Network
  - Grid-based feature operations
  - Point cloud convolutions
  - Neighbor search algorithms

- **MeshGraphNet** (`meshgraphnet/`) - Graph networks for mesh-based simulation
  - `meshgraphnet.py` - Standard MeshGraphNet
  - `hybrid_meshgraphnet.py` - Hybrid variant
  - `meshgraphkan.py` - KAN-based variant
  - `bsms_mgn.py` - BSMS-specific implementation

- **MeshReduced** (`mesh_reduced/`) - Reduced-order mesh models
- **Transolver** (`transolver/`) - Transformer-based PDE solver
  - Physics-aware attention mechanisms
  - Custom embeddings

#### Operator Learning Models
- **FNO** (`fno/`) - Fourier Neural Operator
- **DPOT** (`dpot/`) - Deep Potential Operator Transform
  - `dpot.py` - 2D DPOT
  - `dpot3d.py` - 3D DPOT

#### Computer Vision Models
- **Pix2Pix** (`pix2pix/`) - Image-to-image translation
- **U-Net** (`unet/`) - Classic U-Net architecture
- **SRRN** (`srrn/`) - Super-Resolution Residual Network
- **VFGN** (`vfgn/`) - Vector Field Graph Network

#### Recurrent Models
- **RNN** (`rnn/`) - Recurrent neural networks
  - `rnn_one2many.py` - One-to-many prediction
  - `rnn_seq2seq.py` - Sequence-to-sequence models
  - `layers.py` - Custom RNN layers

- **SwinVRNN** (`swinvrnn/`) - Swin Transformer-based VRNN

#### General Models
- **MLP** (`mlp/`) - Multi-layer perceptrons

---

### Neural Network Layers

**Location:** `physicsnemo/nn/`

Building blocks for constructing models:

#### Core Layers
- `activations.py` - Custom activation functions
- `attention_layers.py` - Attention mechanisms
- `conv_layers.py` - Convolutional layers
- `fully_connected_layers.py` - Dense layers
- `mlp_layers.py` - MLP building blocks
- `transformer_layers.py` - Transformer components
- `transformer_decoder.py` - Transformer decoder

#### Specialized Layers
- `fourier_layers.py` - Fourier-based layers (for FNO, AFNO)
- `spectral_layers.py` - Spectral operations
- `fft.py` - Fast Fourier Transform utilities
- `kan_layers.py` - Kolmogorov-Arnold Network layers
- `siren_layers.py` - SIREN activations
- `dgm_layers.py` - Deep Galerkin Method layers

#### GNN Layers
**Location:** `physicsnemo/nn/gnn_layers/`
- Graph neural network components
- Message passing layers
- Edge and node feature updates
- 12+ specialized GNN layer types

#### Normalization & Regularization
- `layer_norm.py` - Layer normalization
- `weight_norm.py` - Weight normalization
- `weight_fact.py` - Weight factorization
- `drop.py` - Dropout variants

#### Geometry & Spatial Operations
- `ball_query.py` - Ball query for point clouds
- `interpolation.py` - Interpolation operations
- `sdf.py` - Signed distance functions
- `neighbors/` - Neighbor search algorithms (11 implementations)
  - K-NN, radius search, grid-based search

#### Sampling & Resampling
- `resample_layers.py` - Up/downsampling operations

---

### Data Pipelines

**Location:** `physicsnemo/datapipes/`

Comprehensive data loading and preprocessing:

#### Climate & Weather Data
**Location:** `datapipes/climate/`
- `era5_hdf5.py` - ERA5 data in HDF5 format
- `era5_netcdf.py` - ERA5 data in NetCDF format
- `climate.py` - General climate datasets
- `synthetic.py` - Synthetic climate data generation
- `utils/` - Climate data utilities

#### HEALPix Data
**Location:** `datapipes/healpix/`
- `timeseries_dataset.py` - Time series on HEALPix grids
- `coupledtimeseries_dataset.py` - Coupled time series
- `couplers.py` - Data coupling utilities
- `data_modules.py` - PyTorch Lightning data modules

#### CFD & Engineering Data
**Location:** `datapipes/gnn/`
- `ahmed_body_dataset.py` - Ahmed body aerodynamics
- `drivaernet_dataset.py` - DrivAerNet dataset
- `vortex_shedding_dataset.py` - Vortex shedding simulations
- `stokes_dataset.py` - Stokes flow problems
- `lagrangian_dataset.py` - Lagrangian particle systems
- `hydrographnet_dataset.py` - Hydrograph networks
- `bsms.py` - BSMS dataset utilities

#### CAE Data
**Location:** `datapipes/cae/`
- `cae_dataset.py` - General CAE datasets
- `mesh_datapipe.py` - Mesh data loading
- `domino_datapipe.py` - DOMINO-specific data
- `transolver_datapipe.py` - Transolver data loading
- `readers.py` - Various file format readers

#### Benchmark Data
**Location:** `datapipes/benchmarks/`
- `darcy.py` - Darcy flow benchmark
- `kelvin_helmholtz.py` - Kelvin-Helmholtz instability
- `kernels/` - Data generation kernels

---

### Metrics

**Location:** `physicsnemo/metrics/`

Domain-specific evaluation metrics:

#### Climate Metrics
**Location:** `metrics/climate/`
- `acc.py` - Anomaly Correlation Coefficient
- `efi.py` - Extreme Forecast Index
- `loss.py` - Climate-specific loss functions
- `healpix_loss.py` - HEALPix grid losses
- `reduction.py` - Reduction operations

#### Diffusion Metrics
**Location:** `metrics/diffusion/`
- `fid.py` - Fréchet Inception Distance
- `loss.py` - Diffusion model losses

#### General Metrics
**Location:** `metrics/general/`
- `mse.py` - Mean Squared Error variants
- `crps.py` - Continuous Ranked Probability Score
- `wasserstein.py` - Wasserstein distance
- `ensemble_metrics.py` - Ensemble forecasting metrics
- `calibration.py` - Calibration metrics
- `entropy.py` - Entropy-based metrics
- `histogram.py` - Histogram-based metrics
- `power_spectrum.py` - Spectral analysis
- `reduction.py` - Reduction utilities

#### CAE Metrics
**Location:** `metrics/cae/`
- `cfd.py` - CFD-specific metrics
- `integral.py` - Integral quantities

---

### Distributed Computing

**Location:** `physicsnemo/distributed/`

Distributed training and inference support:

- `manager.py` - Distributed process management
- `config.py` - Distributed configuration
- `autograd.py` - Distributed automatic differentiation
- `mappings.py` - Tensor mapping utilities
- `fft.py` - Distributed FFT operations
- `utils.py` - Distributed utilities

#### Domain Parallelism
**Location:** `physicsnemo/domain_parallel/`

Advanced domain decomposition:
- `shard_tensor.py` - Sharded tensor implementation
- `_shard_tensor_spec.py` - Tensor sharding specifications
- `_shard_redistribute.py` - Redistribution operations

**Custom Operations** (`custom_ops/`)
- `_reductions.py` - Distributed reductions
- `_tensor_ops.py` - Tensor operations

**Shard Utilities** (`shard_utils/`)
- `halo.py` - Halo exchange
- `ring.py` - Ring communication
- `attention_patches.py` - Patched attention for domain parallelism
- `conv_patches.py` - Patched convolutions
- `pooling_patches.py` - Patched pooling
- `normalization_patches.py` - Patched normalization
- `knn.py` - Distributed K-NN
- `mesh_ops.py` - Mesh operations
- `point_cloud_ops.py` - Point cloud operations
- `padding.py` - Padding utilities

---

### Active Learning

**Location:** `physicsnemo/active_learning/`

Active learning framework for efficient data acquisition:

- `driver.py` - Active learning driver
- `loop.py` - Training loop implementation
- `config.py` - Configuration management
- `logger.py` - Logging utilities
- `protocols.py` - Interface protocols
- `_registry.py` - Component registry

---

### Utils

**Location:** `physicsnemo/utils/`

General utilities and helpers:

#### Core Utilities
- `checkpoint.py` - Model checkpointing
- `capture.py` - CUDA graph capture
- `memory.py` - Memory management
- `insolation.py` - Solar insolation calculations
- `zenith_angle.py` - Zenith angle computations

#### Logging
**Location:** `utils/logging/`
- Comprehensive logging infrastructure (6 modules)
- Experiment tracking
- Metrics logging

#### Profiling
**Location:** `utils/profiling/`
- Performance profiling tools (5 modules)
- CUDA profiling utilities
- Memory profiling

#### Mesh Utilities
**Location:** `utils/mesh/`
- Mesh processing utilities (4 modules)
- Mesh I/O operations
- Mesh transformations

---

### Deployment

**Location:** `physicsnemo/deploy/`

Model deployment utilities:

- `onnx/` - ONNX export and optimization
  - `utils.py` - ONNX conversion utilities

---

### Experimental Features

**Location:** `physicsnemo/experimental/`

Cutting-edge research features:

- `metrics/diffusion/` - Experimental diffusion metrics
- `models/diffusion/` - Experimental diffusion models
- `models/dit/` - DiT (Diffusion Transformer) variants

---

## Examples

**Location:** `examples/`

Comprehensive examples organized by domain (776 files):

### Weather & Climate
**Location:** `examples/weather/`
- 298 files (139 YAML configs, 124 Python scripts, 15 markdown docs)
- ERA5 data processing
- FourCastNet examples
- GraphCast implementations
- StormCast examples

### CFD (Computational Fluid Dynamics)
**Location:** `examples/cfd/`
- 310 files (155 Python, 99 YAML, 24 markdown)
- Vortex shedding
- Ahmed body aerodynamics
- DrivAerNet examples
- Stokes flow
- MeshGraphNet tutorials

### Geophysics
**Location:** `examples/geophysics/`
- `diffusion_fwi/` - Diffusion-based Full Waveform Inversion
- Seismic modeling

### Healthcare
**Location:** `examples/healthcare/`
- Medical imaging applications
- Brain wave analysis
- Blood flow modeling

### Structural Mechanics
**Location:** `examples/structural_mechanics/`
- 42 files (18 Python, 18 YAML, 2 markdown)
- Deforming plate examples
- Crash simulations

### Additive Manufacturing
**Location:** `examples/additive_manufacturing/`
- `sintering_physics/` - Sintering process modeling

### Reservoir Simulation
**Location:** `examples/reservoir_simulation/`
- Oil & gas reservoir modeling
- Porous media flow

### Molecular Dynamics
**Location:** `examples/molecular_dynamics/`
- Lennard-Jones systems
- Molecular simulations

### Generative Models
**Location:** `examples/generative/`
- `corrdiff/` - CorrDiff examples
- `diffusion/` - General diffusion models
- `stormcast/` - StormCast weather generation
- `topodiff/` - Topology-aware diffusion

### Active Learning
**Location:** `examples/active_learning/`
- `moons/` - Active learning on synthetic datasets

### Minimal Examples
**Location:** `examples/minimal/`
- `neighbor_list/` - Neighbor list construction
- `ShardTensorExamples/` - Domain parallelism tutorials (29 files)

---

## Testing

**Location:** `test/`

Comprehensive test suite (281 files):

### Test Organization
```
test/
├── active_learning/        # Active learning tests (7 files)
├── core/                   # Core functionality tests (7 files)
├── datapipes/             # Data pipeline tests (23 files)
├── distributed/           # Distributed computing tests (6 files)
├── domain_parallel/       # Domain parallelism tests (23 files)
├── metrics/               # Metrics tests (9 files)
├── models/                # Model tests (151 files: 78 .py, 63 .pth, 10 .mdlus)
├── nn/                    # Neural network layer tests (26 files)
├── utils/                 # Utility tests (10 files)
├── common/                # Common test utilities (6 files)
├── ci_tests/              # CI/CD specific tests (4 files)
├── plugins/               # Test plugins (2 files)
├── conftest.py            # Pytest configuration
├── pytest_utils.py        # Test utilities
├── coverage.pytest.rc     # Coverage configuration
└── get_coverage.sh        # Coverage script
```

### Test Assets
- **63 `.pth` files** - Pre-trained model checkpoints for testing
- **10 `.mdlus` files** - Modulus checkpoint files
- **1 `.json` file** - Test configuration

---

## Documentation

**Location:** `docs/`

Comprehensive documentation (196 files):

### Documentation Structure
- **API Reference** (`api/`) - Auto-generated API docs
  - `models/` - Model documentation (10 RST files)
  - Module-specific API docs (RST format)

- **Examples Documentation** - Organized by domain:
  - `examples_weather.rst`
  - `examples_cfd.rst`
  - `examples_geophysics.rst`
  - `examples_healthcare.rst`
  - `examples_introductory.rst`
  - `examples_molecular_dynamics.rst`

- **Images & Figures** (`img/`) - 85 PNG, 30 GIF, various result visualizations
  - Architecture diagrams
  - Result visualizations
  - Training curves
  - Profiling screenshots

- **Test Scripts** (`test_scripts/`) - 36 documentation test scripts

- **Configuration**
  - `conf.py` - Sphinx configuration
  - `Makefile` - Documentation build system

---

## Configuration Files

### Project Configuration
- **`pyproject.toml`** - Python project metadata, dependencies, build configuration
- **`Dockerfile`** - Container definition for reproducible environments
- **`Makefile`** - Build automation and common tasks
- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration
- **`.importlinter`** - Import dependency rules
- **`.markdownlint.yaml`** - Markdown linting rules

### CI/CD
- **`.gitlab-ci.yml`** - GitLab CI/CD pipeline
- **`.github/`** - GitHub workflows and configurations
- **`sonar-project.properties`** - SonarQube configuration

---

## Coding Standards

**Location:** `CODING_STANDARDS/`

- `MODELS_IMPLEMENTATION.md` - Model implementation guidelines
- `EXTERNAL_IMPORTS.md` - External dependency guidelines

---

## Key Features

### 1. Modular Architecture
- Plug-and-play model components
- Extensible datapipe system
- Customizable metrics

### 2. Distributed Training
- Multi-GPU support
- Domain decomposition
- Efficient communication primitives

### 3. Multiple Domains
- Weather & climate forecasting
- Computational fluid dynamics
- Structural mechanics
- Molecular dynamics
- Geophysics
- Healthcare

### 4. State-of-the-Art Models
- 20+ pre-implemented architectures
- Latest research implementations
- Production-ready models

### 5. Comprehensive Testing
- 281 test files
- Pre-trained checkpoints
- CI/CD integration

### 6. Rich Examples
- 776 example files
- Domain-specific tutorials
- Production use cases

---

## Dependencies

### Core Dependencies (from `pyproject.toml`)
- **PyTorch** ≥ 2.4.0 - Deep learning framework
- **Warp-lang** - NVIDIA's high-performance Python framework
- **Hydra-core** ≥ 1.3.2 - Configuration management
- **OmegaConf** ≥ 2.3.0 - Configuration system
- **xarray** ≥ 2025.6.1 - N-D labeled arrays
- **zarr** ≥ 2.18.3 - Chunked array storage
- **h5py** ≥ 3.15.1 - HDF5 file I/O
- **einops** ≥ 0.8.1 - Tensor operations
- **timm** ≥ 1.0.22 - Vision models
- **onnx** ≥ 1.14.0 - Model export
- **pandas** - Data analysis
- **s3fs** ≥ 2023.5.0 - S3 filesystem
- **tqdm** ≥ 4.60.0 - Progress bars
- **requests** ≥ 2.32.2 - HTTP library

---

## Version Information

- **Current Version:** 1.4.0a0 (alpha)
- **Branch:** v2.0-refactor
- **Python:** 3.11+
- **License:** Apache 2.0 (see LICENSE.txt)

---

## Getting Started

### Installation
```bash
# Clone the repository
git clone -b v2.0-refactor https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo

# Install in development mode
pip install -e .
```

### Quick Example
```python
import physicsnemo
from physicsnemo.models import FNO

# Create a Fourier Neural Operator
model = FNO(
    in_channels=2,
    out_channels=1,
    decoder_layer_size=128,
    num_fno_layers=4,
    num_fno_modes=12,
)

# Model is ready for training or inference
print(f"PhysicsNeMo version: {physicsnemo.__version__}")
```

---

## Resources

- **Documentation:** `docs/`
- **Examples:** `examples/`
- **Tests:** `test/`
- **FAQ:** `FAQ.md`
- **Contributing:** `CONTRIBUTING.md`
- **Migration Guide:** `v2.0-MIGRATION-GUIDE.md`

---

## File Statistics

- **Total Python files:** 298 in `physicsnemo/`, 378 in `examples/`, 191 in `test/`
- **Total example files:** 776 (includes YAML configs, scripts, docs)
- **Total test files:** 281 (includes checkpoints and test data)
- **Documentation files:** 196 (RST, Python, images)
- **Lines of code:** ~100,000+ (estimated)

---

**Last Updated:** December 19, 2025

