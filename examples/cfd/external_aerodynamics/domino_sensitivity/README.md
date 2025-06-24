# DoMINO Sensitivity Analysis for Aerodynamic Design

This directory contains a sensitivity analysis pipeline for the DoMINO
(Decomposable Multi-scale Iterative Neural Operator) model, specifically
designed for aerodynamic analysis. The pipeline computes
gradient-based sensitivities that indicate how geometric modifications to a
vehicle or aircraft surface affect aerodynamic performance metrics such as drag
force.

## Overview

The DoMINO sensitivity analysis pipeline leverages automatic differentiation to
compute gradients of aerodynamic quantities (e.g., drag force) with respect to
surface geometry coordinates. This enables:

- **Design Optimization**: Identify surface regions where modifications will
  most effectively reduce drag
- **Sensitivity Visualization**: Generate heat maps showing which parts of the
  geometry are most critical for aerodynamic performance  
- **Gradient Validation**: Verify gradient accuracy through finite-difference
  checking
- **Shape Optimization**: Provide gradient information for gradient-based
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
the scripts.

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

```python
# Run gradient checking to validate sensitivity accuracy
python main_gradient_checking.py

# Plot validation results
python plot_gradient_checking.py
```

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

### Post-processed Sensitivities

After calling `postprocess_point_sensitivities()`, additional fields are
available:

<!-- markdownlint-disable -->

| Key | Description | Shape | Units |
|-----|-------------|-------|-------|
| `raw_sensitivity_cells` | Raw cell-centered sensitivities | `(n_cells, 3)` | `[N/m]` |
| `raw_sensitivity_normal_cells` | Normal component of raw sensitivities | `(n_cells,)` | `[N/m]` |
| `smooth_sensitivity_point` | Smoothed point sensitivities | `(n_points, 3)` | `[N/m]` |
| `smooth_sensitivity_normal_point` | Smoothed normal sensitivities | `(n_points,)` | `[N/m]` |
| `smooth_sensitivity_cell` | Smoothed cell sensitivities | `(n_cells, 3)` | `[N/m]` |

<!-- markdownlint-enable -->

## Configuration

The pipeline uses Hydra configuration management. Key configuration parameters
include:

### Model Parameters

- `model.interp_res`: Grid resolution for interpolation `[128, 64, 48]`
- `model_checkpoint_path`: Path to pre-trained DoMINO model

### Bounding Box Settings

- `data.bounding_box.min/max`: Volume bounding box coordinates
- `data.bounding_box_surface.min/max`: Surface bounding box coordinates

### Physics Parameters

- `stream_velocity`: Inlet flow velocity [m/s]
- `air_density`: Air density [kg/m³]
- `stencil_size`: Number of neighboring points for surface calculations

## Gradient Checking and Validation

The pipeline includes comprehensive gradient validation tools:

### Finite Difference Validation

```python
# Run gradient checking with multiple epsilon values
python main_gradient_checking.py
```

This script:

1. Computes baseline drag force
2. Perturbs geometry using computed sensitivities
3. Evaluates drag force at perturbed geometries
4. Compares finite-difference gradients with analytical gradients

### Visualization

```python
# Generate gradient checking plots
python plot_gradient_checking.py
```

Creates plots showing:

- Analytical gradient predictions vs. finite differences
- Validation across multiple perturbation scales
- Comparison between raw and smoothed sensitivities

## Performance Considerations

### Memory Management

- Large geometries are processed in batches to manage GPU memory
- Batch size can be adjusted in the DataLoader configuration
- Memory usage is monitored and reported during processing

### Computational Efficiency

- Automatic differentiation is more efficient than finite differences
- Gradient computation scales well with geometry complexity
- Multi-GPU support available for very large problems

### Smoothing Parameters

- `n_laplacian_iters`: Controls smoothing strength (default: 20)
- Higher values produce smoother but potentially less accurate sensitivities
- Lower values preserve sharp features but may be noisy

## Applications

### Design Optimization

- Integration with gradient-based optimizers (L-BFGS, Adam, etc.)
- Topology optimization for aerodynamic shapes
- Parameter optimization for vehicle styling

### Sensitivity Analysis

- Identify critical design regions
- Understand trade-offs between different geometric features
- Guide manual design modifications

### Validation and Verification

- Compare with finite-difference gradients
- Validate optimization algorithms
- Assess numerical accuracy

## Limitations and Considerations

1. **Model Accuracy**: Sensitivities are only as accurate as the underlying
   DoMINO model
2. **Geometry Resolution**: STL resolution affects sensitivity field quality
3. **Boundary Conditions**: Currently configured for external flow around
   vehicles
4. **Memory Requirements**: Large geometries may require distributed computing

## Troubleshooting

### Common Issues

**Out of Memory Errors**:

- Reduce batch size in DataLoader
- Use gradient checkpointing
- Enable distributed inference

**Noisy Sensitivities**:

- Increase Laplacian smoothing iterations
- Check STL mesh quality
- Verify model convergence

**Gradient Checking Failures**:

- Verify finite difference step sizes
- Check numerical precision settings
- Ensure mesh consistency

## References

1. [DoMINO: A Decomposable Multi-scale Iterative Neural
   Operator](https://arxiv.org/abs/2501.13350)
2. [Automatic Differentiation in Machine Learning: A
   Survey](https://arxiv.org/abs/1502.05767)

## Contributing

When extending this pipeline:

- Follow the existing code style and documentation standards
- Add appropriate type hints and docstrings
- Include gradient checking for new sensitivity computations
- Update this README with new functionality
