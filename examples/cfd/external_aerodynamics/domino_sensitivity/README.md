# DoMINO Sensitivity Analysis for Aerodynamic Design

This directory contains a sensitivity analysis pipeline for the DoMINO
(Decomposable Multi-scale Iterative Neural Operator) model, using an example of
aerodynamic analysis. The pipeline computes gradient-based sensitivities that
indicate how geometric modifications to a vehicle or aircraft surface affect
aerodynamic performance metrics such as drag force. This is intended to serve as
an example template of how to compute geometry sensitivities for DoMINO
surrogates for any PDE it is applied to.

## Overview

The DoMINO sensitivity analysis pipeline leverages automatic differentiation to
compute gradients of aerodynamic quantities (e.g., drag force) with respect to
surface geometry coordinates. This enables:

- **Sensitivity Visualization**: Generate heat maps showing which parts of the
  geometry are most critical for aerodynamic performance  
- **Gradient Validation**: Verify gradient accuracy through finite-difference
  checking
- **Design Optimization**: Provide gradient information for gradient-based
  optimization algorithms

## Key Features

- **Automatic Differentiation**: Uses PyTorch's autograd to compute exact
  gradients efficiently
- **Surface Sensitivity Maps**: Generates sensitivity fields that can be
  visualized on the geometry surface
- **Gradient Smoothing**: Applies Laplacian smoothing to reduce noise in
  sensitivity fields
- **Validation Tools**: Includes finite-difference gradient checking for
  verification
- **Batch Processing**: Handles large geometries through efficient batching
  strategies
- **Multi-GPU Support**: Compatible with distributed inference for large-scale
  problems

## Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: This pipeline requires a pre-trained DoMINO model checkpoint. The
example uses `DoMINO.0.41.pt` which should be placed in the same directory as
the scripts. See the [main DoMINO example](../domino/) for details on how to
train your own model checkpoint.

## Pipeline Components

### Core Modules

- **`main.py`**: Main inference pipeline containing the `DoMINOInference` class
- **`design_datapipe.py`**: Data preprocessing pipeline (`DesignDatapipe`) for
  mesh processing
- **`main_gradient_checking.py`**: Gradient validation script using finite
  differences
- **`plot_gradient_checking.py`**: Visualization tools for gradient checking
  results

### DoMINOInference Class

The `DoMINOInference` class is the main interface for sensitivity analysis:

```python
from main import DoMINOInference
import pyvista as pv

# Initialize the inference pipeline
domino = DoMINOInference(
    cfg=config,
    model_checkpoint_path="DoMINO.0.41.pt",
    dist=distributed_manager
)

# Load geometry
mesh = pv.read("vehicle.stl")

# Compute sensitivities
results = domino(
    mesh=mesh,
    stream_velocity=38.889,  # m/s
    stencil_size=7,
    air_density=1.205        # kg/m³
)
```

## Usage Examples

### Basic Sensitivity Analysis

```python
import hydra
import pyvista as pv
from pathlib import Path
from main import DoMINOInference
from physicsnemo.distributed import DistributedManager

# Initialize configuration
with hydra.initialize(version_base="1.3", config_path="conf"):
    cfg = hydra.compose(config_name="config")

# Setup distributed computing
DistributedManager.initialize()
dist = DistributedManager()

# Create inference pipeline
domino = DoMINOInference(
    cfg=cfg,
    model_checkpoint_path="DoMINO.0.41.pt",
    dist=dist
)

# Load geometry
mesh = pv.read("car.stl")

# Run sensitivity analysis
results = domino(
    mesh=mesh,
    stream_velocity=30.0,    # Inlet velocity [m/s]
    stencil_size=7,          # Neighbor stencil size
    air_density=1.205        # Air density [kg/m³]
)

# Access results
print(f"Total drag force: {results['aerodynamic_force'][0]:.2f} N")
sensitivity_shape = results['geometry_sensitivity'].shape
print(f"Geometry sensitivity shape: {sensitivity_shape}")
```

### Post-processing and Smoothing

```python
# Apply post-processing to compute smoothed sensitivities
sensitivity_results = domino.postprocess_point_sensitivities(
    results=results,
    mesh=mesh,
    n_laplacian_iters=20  # Number of smoothing iterations
)

# Add results to mesh for visualization
for key, value in results.items():
    if len(value) == mesh.n_cells:
        mesh.cell_data[key] = value
    elif len(value) == mesh.n_points:
        mesh.point_data[key] = value

# Add smoothed sensitivities
for key, value in sensitivity_results.items():
    mesh[key] = value

# Save results
mesh.save("results_with_sensitivities.vtk")
```

### Gradient Validation

The pipeline includes finite-difference gradient checking to validate
sensitivity accuracy:

```bash
python main_gradient_checking.py  # Run validation
python plot_gradient_checking.py  # Visualize results
```

The validation compares analytical gradients from automatic differentiation
against finite-difference approximations across multiple perturbation scales.

## Output Data Structure

The sensitivity analysis returns a dictionary with the following keys:

<!-- markdownlint-disable -->

| Key | Description | Shape | Units |
|-----|-------------|-------|-------|
| `geometry_coordinates` | Surface mesh coordinates | `(n_cells, 3)` | `[m]` |
| `geometry_sensitivity` | Raw sensitivity vectors | `(n_cells, 3)` | `[N/m]` |
| `pred_surf_pressure` | Predicted surface pressure | `(n_cells,)` | `[Pa]` |
| `pred_surf_wall_shear_stress` | Wall shear stress components | `(n_cells, 3)` | `[Pa]` |
| `aerodynamic_force` | Total aerodynamic force | `(3,)` | `[N]` |

<!-- markdownlint-enable -->

After calling `postprocess_point_sensitivities()`, additional smoothed
sensitivity fields are available with `_point` and `_cell` suffixes for
different mesh representations.

## Configuration and Parameters

The pipeline uses Hydra configuration management. Key parameters include:

- **Model settings**: `model.interp_res` controls grid resolution `[128, 64,
  48]`
- **Bounding boxes**: `data.bounding_box` and `data.bounding_box_surface` define
  computational domains
- **Physics parameters**: `stream_velocity` (inlet velocity), `air_density`, and
  `stencil_size` (neighbor count)
- **Smoothing**: `n_laplacian_iters` controls sensitivity smoothing strength
  (default: 20)

## Limitations

- **Model accuracy dependency**: Sensitivity accuracy depends on the underlying
  DoMINO model quality
- **Model smoothness dependency**: If the underlying ML architecture does not
  produce solutions that are at least $C^1$ continuous, the sensitivity fields
  will be noisy
- **STL resolution**: Mesh resolution affects sensitivity field quality and
  smoothness

## References

1. [DoMINO: A Decomposable Multi-scale Iterative Neural
   Operator](https://arxiv.org/abs/2501.13350)
2. [Automatic Differentiation in Machine Learning: A
   Survey](https://arxiv.org/abs/1502.05767)
