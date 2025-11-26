# U-FNO for CO2 Sequestration

Deep learning models for predicting pressure and saturation in CO2 sequestration reservoirs.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Training

```bash
# Edit configuration
vim conf/training_config.yaml

# Run training (single GPU)
python train_fno3d.py

# Run training (multi-GPU with DDP)
torchrun --nproc_per_node=4 train_fno3d.py
```

### 3. Evaluation

```bash
# Evaluate pressure model
python evaluate_pressure.py --checkpoint checkpoints/best_model_pressure_*.pth

# Evaluate saturation model
python evaluate_saturation.py --checkpoint checkpoints/best_model_saturation_*.pth
```

## Available Models

| Model Type | Description | Best For |
|-----------|-------------|----------|
| **U-FNO** | Fourier + U-Net (hybrid) | Best accuracy, spatiotemporal PDEs |
| **Conv-FNO** | Fourier + 3D Convolutions | Balanced performance/speed |
| **Standalone UNet** | Pure spatial convolutions | Baseline comparisons |
| **Standard FNO** | Pure Fourier layers | Global patterns, fast training |

All models support both **custom** and **PhysicsNemo** UNet implementations.

## Documentation

### Core Guides

- Configuration system (model_config.yaml + training_config.yaml)
- Model architectures and parameters
- Data format requirements and validation
- Model evaluation system

### Architecture Guides

- Conv-FNO architecture details

## File Structure

```text
U-FNO/
├── conf/
│   ├── model_config.yaml       # Model architecture & loss
│   └── training_config.yaml    # Training, data, optimizer settings
├── train_fno3d.py              # Training script
├── evaluate_pressure.py        # Pressure evaluation
├── evaluate_saturation.py      # Saturation evaluation
├── ufno.py                     # U-FNO model architectures
├── unet3d.py                   # Custom UNet implementations
├── physicsnemo_unet.py         # PhysicsNemo UNet wrapper
├── data_validation.py          # Data validation utilities
├── dataset.py                  # Data loading
├── losses.py                   # Loss functions
├── metrics.py                  # Evaluation metrics
└── checkpoints/                # Trained models
```

## Configuration

### Two-File System

1. **`model_config.yaml`** - Model architecture and loss (rarely changed)
2. **`training_config.yaml`** - Training parameters (frequently tuned)

### Example: Train U-FNO

```yaml
# training_config.yaml
data:
  variable: pressure  # or "saturation"

# model_config.yaml
arch:
  model_type: ufno
  ufno:
    num_fno_layers: 3
    num_unet_layers: 3
    num_conv_layers: 0
    unet_type: custom  # or "physicsnemo"
```

### Example: Train Conv-FNO

```yaml
# model_config.yaml
arch:
  model_type: ufno
  ufno:
    num_fno_layers: 3
    num_unet_layers: 0    # Disable U-Net
    num_conv_layers: 3    # Enable Conv
```

### Example: Train Standalone UNet

```yaml
# model_config.yaml
arch:
  model_type: unet
  unet:
    unet_type: physicsnemo  # NOTE: Only physicsnemo supported for standalone use
```

**Note:** Custom UNet3D is designed for U-FNO only (constant channel dimensions).  
For standalone UNet, always use `unet_type: physicsnemo`.

## Model Checkpoints

Models are automatically named based on architecture:

- `best_model_pressure_ufno_custom.pth`
- `best_model_pressure_convfno.pth`
- `best_model_saturation_unet_physicsnemo.pth`
- etc.

No manual naming needed - prevents accidental overwriting!

## Citation

If you use this code, please cite:

- PhysicsNemo: [NVIDIA PhysicsNemo](https://github.com/NVIDIA/physicsnemo)
- U-FNO: [Your paper/reference]
