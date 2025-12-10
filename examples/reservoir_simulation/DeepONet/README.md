# U-FNO for Reservoir Simulation

This example is part of the implementation of Neural Operator Factory, it implements an FNO family of architectures for predicting pressure and saturation fields in reservoir simulations. The implementation leverages PhysicsNeMo's core FNO layers and UNet components.

**Current status**: The framework has been developed and tested on 2D CO2 sequestration datasets.
**Future work in progress**: DeepONet family of neural operators and proper support for 3D problems.

## Table of Contents

- [Overview](#overview)
- [Background](#background)
  - [Reservoir Simulation](#reservoir-simulation)
  - [Neural Operators](#neural-operators)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Results](#results)
- [References](#references)
- [Citation](#citation)

## Overview

Reservoir simulation is essential for predicting subsurface fluid flow behavior in applications such as:

- **Carbon storage**: CO2 sequestration in geological formations
- **Oil and gas recovery**: Enhanced oil recovery and production forecasting
- **Groundwater management**: Aquifer modeling and contamination transport

Traditional numerical simulators solve the governing PDEs but are computationally expensive, especially for uncertainty quantification requiring thousands of forward simulations. Neural operators provide a fast surrogate model that can accelerate these workflows by 3-4 orders of magnitude while maintaining high accuracy. Moreover, as opposed to numerical reduced order models, neural operators provide a full physics surrogates with straight forward support data assimilation.  

This example provides a flexible framework for training neural operator models on reservoir simulation data.

## Background

### Reservoir Simulation

Reservoir simulation models **multiphase flow** through porous media, governed by:

- **Darcy's law**: Relating fluid velocity to pressure gradients
- **Mass conservation**: For each fluid phase
- **Constitutive relations**: Relative permeability and capillary pressure curves

Key output variables typically include:
- **Pressure**: Pressure distribution or change from initial conditions
- **Saturation**: Phase saturation fractions (e.g., gas, oil, water)

### Neural Operators

Unlike standard neural networks that learn point-to-point mappings, **neural operators** learn mappings between function spaces. This enables:

- **Resolution invariance**: Train on coarse grids, evaluate on fine grids
- **Generalization**: Handle varying input functions (e.g., permeability fields)
- **Efficiency**: Single forward pass vs. iterative PDE solvers

The **Fourier Neural Operator (FNO)** parameterizes the kernel integral in Fourier space, enabling efficient global convolutions. **U-FNO** enhances FNO with U-Net skip connections to capture multi-scale features critical for heterogeneous reservoirs.

To be implemeted: The **Deep Operator Network (DeepONet)** allows for seperation of space and time variables, significantly improving effeciency. **U-DeepONet** enhances DeepONet with U-Net skip connections to capture multi-scale features critical for heterogeneous reservoirs. **Fourier-DeepONet** combines the best of the two architectures. **Fourier-MIONet** Allows for further seperation input features: scalar and field variables.

## Model Architectures

| Model | Description | Use Case |
|-------|-------------|----------|
| **U-FNO** | FNO + U-Net skip connections | Best accuracy for heterogeneous reservoirs |
| **Conv-FNO** | FNO + 3D convolutions | Balanced performance/computational cost |
| **Standard FNO** | Pure Fourier layers | Fast training, global patterns |
| **Standalone UNet** | PhysicsNeMo's UNet | Baseline comparison |

### U-FNO Architecture

```
Input (B, H, W, T, C)
    │
    ▼
┌─────────────────┐
│  Lifting Layer  │  (MLP or Conv: C → width)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Standard FNO Layers (×N)        │
│  ┌─────────────┐  ┌──────────────┐  │
│  │ SpectralConv│ +│ 1×1×1 Conv   │  │
│  └─────────────┘  └──────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   U-Net Enhanced FNO Layers (×M)    │
│  ┌─────────────┐  ┌──────────────┐  │
│  │ SpectralConv│ +│ 1×1×1 Conv   │  │
│  └──────┬──────┘  └──────────────┘  │
│         │                           │
│         ▼                           │
│  ┌──────────────┐                   │
│  │   3D U-Net   │ (skip connection) │
│  └──────────────┘                   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Decoder (MLP or CNN)  │  (width → out_channels)
└────────┬────────┘
         │
         ▼
Output (B, H, W, T)
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PhysicsNeMo (installed from source or pip)
- CUDA-capable GPU (recommended: 16GB+ VRAM)

### Install Dependencies

```bash
# Navigate to the example directory
cd examples/reservoir_simulation/DeepONet

# Install additional dependencies (if any)
pip install -r requirements.txt
```

## Data Requirements

### Input Data Format

The model expects input tensors of shape `(B, H, W, T, C)` where:
- `B`: Batch size
- `H`: Height (spatial dimension)
- `W`: Width (spatial dimension)
- `T`: Time steps
- `C`: Input channels (problem-dependent)

The number of input channels and their meaning depend on your specific reservoir simulation problem. Configure the model's `in_channels` parameter accordingly.

### Output Data

- **Pressure model**: Predicts pressure field of shape `(B, H, W, T)`
- **Saturation model**: Predicts saturation field of shape `(B, H, W, T)`

### Data Directory Structure

```
data/
├── train/
│   ├── inputs/          # Input tensors
│   └── outputs/         # Target tensors (pressure or saturation)
├── val/
│   ├── inputs/
│   └── outputs/
└── test/
    ├── inputs/
    └── outputs/
```

## Training

### Configuration

1. Edit the data path in `conf/training_config.yaml`:
   ```yaml
   data:
     data_path: /path/to/your/data
     variable: pressure  # or 'saturation'
   ```

2. Select the model architecture in `conf/model_config.yaml`:
   ```yaml
   arch:
     model_type: ufno  # Options: 'ufno', 'unet'
     ufno:
       in_channels: 12        # Adjust based on your data
       out_channels: 1
       num_fno_layers: 3
       num_unet_layers: 3     # Set > 0 for U-FNO
       num_conv_layers: 0     # Set > 0 for Conv-FNO (mutually exclusive with unet_layers)
   ```

### Single GPU Training

```bash
python train_fno3d.py
```

### Multi-GPU Training (DDP)

```bash
# 4 GPUs on a single node
torchrun --nproc_per_node=4 train_fno3d.py

# 8 GPUs on a single node
torchrun --nproc_per_node=8 train_fno3d.py
```

### Monitoring Training

Training progress is logged to:
- **Console**: Loss values and learning rate
- **TensorBoard**: `tensorboard --logdir=./tensorboard`
- **MLflow** (optional): Enable in config with `logging.use_mlflow: true`

## Evaluation

After training, evaluate the model on the test set:

```bash
# Evaluate pressure model
python evaluate_pressure.py --checkpoint checkpoints/best_model_pressure_ufno_physicsnemo.pth

# Evaluate saturation model
python evaluate_saturation.py --checkpoint checkpoints/best_model_saturation_ufno_physicsnemo.pth
```

### Evaluation Metrics

- **Relative L2 Error**: `||pred - target||_2 / ||target||_2`
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Per-timestep errors**: Track accuracy evolution over time

## Configuration

The configuration system uses two YAML files:

### `conf/model_config.yaml`
Model architecture settings (rarely changed between runs):
- Model type (U-FNO, Conv-FNO, UNet)
- Network dimensions (width, Fourier modes)
- Activation functions
- Loss function configuration

### `conf/training_config.yaml`
Training parameters (frequently tuned):
- Data paths and variable selection
- Batch size, learning rate, epochs
- Optimizer and scheduler settings
- Logging and checkpointing

### Example Configurations

**U-FNO (Default)**:
```yaml
arch:
  model_type: ufno
  ufno:
    num_fno_layers: 3
    num_unet_layers: 3
    num_conv_layers: 0
    unet_type: physicsnemo
```

**Conv-FNO**:
```yaml
arch:
  model_type: ufno
  ufno:
    num_fno_layers: 3
    num_unet_layers: 0
    num_conv_layers: 3
```

**Pure FNO**:
```yaml
arch:
  model_type: ufno
  ufno:
    num_fno_layers: 6
    num_unet_layers: 0
    num_conv_layers: 0
```

## Testing

Run the unit tests to verify the implementation:

```bash
# Navigate to the example directory
cd examples/reservoir_simulation/DeepONet

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_unet.py -v      # Test UNet models
pytest tests/test_losses.py -v    # Test loss functions
pytest tests/test_ufno.py -v      # Test U-FNO model

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run only fast tests (skip slow tests)
pytest tests/ -v -m "not slow"
```

### Test Coverage

The test suite covers:
- **UNet models**: Forward pass, gradient flow, parameter counting
- **Loss functions**: MSE, L1, Relative L2, masking, derivatives
- **U-FNO model**: Different configurations, lifting/decoder types, UNet types

## Results

Results from testing on a 2D CO2 sequestration dataset:

<!-- TODO: Add your experimental results here -->

| Model | Variable | Relative L2 Error | Training Time |
|-------|----------|-------------------|---------------|
| U-FNO | Pressure | TBD | TBD |
| U-FNO | Saturation | TBD | TBD |
| Conv-FNO | Pressure | TBD | TBD |
| Conv-FNO | Saturation | TBD | TBD |

### Computational Requirements

- **GPU Memory**: ~12-16 GB for batch_size=4
- **Training Time**: Varies based on dataset size and model configuration

## References

1. **U-FNO Paper**:
   > Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., & Benson, S. M. (2022).
   > U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow.
   > *Advances in Water Resources*, 163, 104180.
   > https://doi.org/10.1016/j.advwatres.2022.104180

2. **Original FNO Paper**:
   > Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021).
   > Fourier Neural Operator for Parametric Partial Differential Equations.
   > *ICLR 2021*.
   > https://arxiv.org/abs/2010.08895

3. **PhysicsNeMo**:
   > NVIDIA PhysicsNeMo: A deep learning framework for physics-ML applications.
   > https://github.com/NVIDIA/physicsnemo

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wen2022ufno,
  title={U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow},
  author={Wen, Gege and Li, Zongyi and Azizzadenesheli, Kamyar and Anandkumar, Anima and Benson, Sally M},
  journal={Advances in Water Resources},
  volume={163},
  pages={104180},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.advwatres.2022.104180}
}

@misc{physicsnemo,
  title={PhysicsNeMo: A deep learning framework for physics-ML applications},
  author={NVIDIA},
  howpublished={\url{https://github.com/NVIDIA/physicsnemo}},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](../../../LICENSE.txt) file for details.
