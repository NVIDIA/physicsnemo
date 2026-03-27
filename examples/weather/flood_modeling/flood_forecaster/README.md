# FloodForecaster: A Domain-Adaptive Geometry-Informed Neural Operator Framework for Rapid Flood Forecasting

FloodForecaster is a deep learning framework for rapid, high-resolution flood forecasting that leverages a time-dependent Geometry-Informed Neural Operator (GINO) with domain adaptation capabilities. The framework enables accurate, real-time flood predictions by learning from source domain data and adapting to target domains through adversarial training, delivering predictions of water depth and velocity across unstructured spatial meshes.

## Problem Overview

Flooding is one of the most destructive and widespread natural hazards, causing significant socioeconomic damage worldwide. Rapid urbanization and climate change are intensifying flood risks, making vulnerable population centers even more susceptible. To effectively mitigate these escalating risks, rapid and high-resolution flood forecasts are essential for enabling timely public warnings, efficient emergency response, and strategic resource deployment.

Traditionally, flood forecasting relies on physically based numerical models that solve the shallow water equations (SWEs). While accurate, these models demand immense computational resources, especially when simulating large geographical areas at fine-scale resolutions (e.g., 1–10 m grid cells) needed to capture complex topographies and flow paths. This computational burden is prohibitive for many real-world applications, such as generating rapid inundation forecasts from meteorological predictions or running large ensembles needed to quantify forecast uncertainty within tight operational deadlines.

FloodForecaster addresses these challenges by offering a computationally efficient surrogate model that:
- **Processes complex geometries**: Leverages unstructured spatial meshes to handle irregular terrain
- **Enables domain transfer**: Incorporates domain adaptation to transfer knowledge from data-rich source domains to data-scarce target domains
- **Achieves superior performance**: Demonstrates superior accuracy and stability over state-of-the-art GNN baselines
- **Requires minimal data**: Successfully adapts with as few as 10 training simulations from a new domain, reducing prediction error by approximately 75% compared to standard fine-tuning

## Model Overview and Architecture

### Core Architecture

FloodForecaster uses a three-stage training pipeline to achieve domain-adaptive flood prediction:

1. **Pretraining**: Train a GINO model on source domain data to learn fundamental flood dynamics
2. **Domain Adaptation**: Fine-tune the model using adversarial training with a domain classifier to learn domain-invariant features
3. **Rollout Evaluation**: Perform autoregressive multi-step predictions and compute comprehensive evaluation metrics

### Key Components

#### GINO (Geometry-Informed Neural Operator)

The core predictive model synergizes:
- **Graph Neural Operators (GNO)**: Process irregular terrain and extract geometric features from unstructured meshes
- **Fourier Neural Operators (FNO)**: Efficiently capture global flood dynamics through spectral processing

GINO processes:
- **Static features**: Elevation, slope, curvature, roughness (Manning's n), cell area
- **Dynamic features**: Water depth, velocity components (Vx, Vy) over time
- **Boundary conditions**: Inflow hydrographs at upstream boundaries

#### Domain Adaptation with Gradient Reversal Layer

The framework integrates a domain adaptation technique using a gradient reversal layer (GRL), which encourages the model to learn domain-invariant physical features. This approach:
- **Prevents catastrophic forgetting**: Unlike standard fine-tuning, preserves the model's original expertise while learning hydraulics of new river segments
- **Enables data-efficient adaptation**: Achieves strong performance with minimal target domain data
- **Uses adversarial training**: A CNN-based domain classifier distinguishes between source and target domains, with gradients reversed during backpropagation to encourage domain-invariant representations

### Key Features

- **Autoregressive forecasting**: Multi-step predictions using a sliding window of historical states
- **Physics-informed evaluation**: Metrics including volume conservation, arrival time, and inundation duration
- **Comprehensive metrics**: RMSE, CSI (Critical Success Index), MAE for temporal characteristics, and FHCA (Flood Hazard Classification Accuracy)

## Data Generation

Data generation utilities for creating synthetic hydrographs and automating HEC-RAS simulations are available in a separate repository:

**Data Generation Repository**: [https://github.com/MehdiTaghizadehUVa/FloodForecaster](https://github.com/MehdiTaghizadehUVa/FloodForecaster)

This repository includes:
- Synthetic hydrograph generation utilities
- HEC-RAS simulation automation scripts
- Data preprocessing and formatting tools

Please refer to the [FloodForecaster repository](https://github.com/MehdiTaghizadehUVa/FloodForecaster) for data generation documentation and usage examples.

## Dataset

FloodForecaster expects data organized in a structured format with the following components:

### Spatial Mesh Files
- `M40_XY.txt`: Cell center coordinates (N × 2)
- `M40_CA.txt`: Cell area (N × 1)

### Static Attribute Files
- `M40_CE.txt`: Elevation
- `M40_CS.txt`: Slope
- `M40_CU.txt`: Curvature
- `M40_FA.txt`: Roughness (Manning's n)
- `M40_A.txt`: Additional static features

### Dynamic Time-Series Files (per timestep `t`)
- `M40_WD_{t}.txt`: Water depth (N × 1)
- `M40_VX_{t}.txt`: X-velocity component (N × 1)
- `M40_VY_{t}.txt`: Y-velocity component (N × 1)

### Boundary Condition Files (per timestep `t`)
- `M40_US_InF_{t}.txt`: Upstream inflow hydrograph

Data should be organized in folders with consistent naming patterns as specified in the configuration. The model supports both source and target domain datasets for domain adaptation training.

**Data Generation Scripts**: Scripts for generating the dataset used in the paper are available at: https://github.com/MehdiTaghizadehUVa/FloodForecaster. This repository provides utilities for creating synthetic hydrographs and automating HEC-RAS simulations to generate training data, but does not include the pre-generated dataset itself.

## Quick Start

### Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your dataset following the structure described above. Organize source and target domain data in separate directories.

3. Configure training parameters in `conf/config.yaml`:
   - Set `source_data.root` and `target_data.root` to your data paths
     - You can use environment variables: `DATA_ROOT`, `TARGET_DATA_ROOT`, `ROLLOUT_DATA_ROOT`
     - Or directly edit the paths in the config file
   - Adjust model parameters (channels, FNO modes, hidden dimensions)
   - Configure training epochs, learning rates, and batch sizes
   - Set domain adaptation hyperparameters (lambda_max, classifier architecture)

### Training

Run the training script:

```bash
python train.py
```

The training pipeline will:
- Pretrain on source domain data
- Perform domain adaptation on combined source and target data
- Save checkpoints after each stage

Training logs, model checkpoints, and metrics will be saved in the directory specified in `config.yaml`.

**Resuming Training:**

To resume training from a checkpoint, set the appropriate checkpoint path in `conf/config.yaml`:

- **Resume pretraining**: Set `checkpoint.resume_from_source` to the pretraining checkpoint directory (e.g., `"./checkpoints_flood_forecaster/pretrain"`). This will resume the source domain pretraining stage from the saved checkpoint.

- **Resume domain adaptation**: Set `checkpoint.resume_from_adapt` to the domain adaptation checkpoint directory (e.g., `"./checkpoints_flood_forecaster/adapt"`). This will resume the domain adaptation stage from the saved checkpoint.

- **For inference**: Set `checkpoint.resume_from_adapt` (preferred) or `checkpoint.resume_from_source` to load a trained model. The inference script will use `resume_from_adapt` if available, otherwise falls back to `resume_from_source`.

Checkpoints are saved in subdirectories under `checkpoint.save_dir`:
- `{save_dir}/pretrain/` - Contains pretraining checkpoints
- `{save_dir}/adapt/` - Contains domain adaptation checkpoints

### Inference

To perform autoregressive rollout and generate evaluation visualizations:

1. Configure your inference settings in `conf/config.yaml`:
   - Set `rollout_data.root` to your test dataset path
   - Configure `rollout.out_dir` for output directory
   - Adjust `rollout_length` and `skip_before_timestep` as needed

2. Run the inference script:

```bash
python inference.py
```

**Note**: The inference script currently does not support multi-GPU or multi-node distributed inference. It runs on a single GPU/device. For distributed inference, you would need to modify the rollout logic to split samples across ranks.

3. The script will output comprehensive visualizations and metrics:
   - **Publication maps**: Water depth and velocity comparisons at selected time steps (12, 24, 36, 48, 60, 72 hours)
   - **Maximum value maps**: Peak water depth and velocity across the entire event
   - **Combined analysis plots**: Temporal characteristics (arrival time, duration), hazard metrics (max momentum flux), and classification accuracy
   - **Volume conservation plots**: Total water volume over time for both predictions and ground truth
   - **Conditional error analysis**: Error distributions conditioned on water depth and velocity magnitude
   - **Rollout animations**: GIF animations showing temporal evolution of water depth and velocity components (3×2 grid: GT vs. Predicted)
   - **Aggregated metrics**: Time-series metrics (RMSE, CSI) and scalar metrics (MAE, FHCA) aggregated across all test events
   - **Event magnitude analysis**: RMSE vs. peak inflow and total volume relationships

All outputs are saved to the configured output directory with organized subdirectories for figures and animations.

### Example Results

The following animations demonstrate FloodForecaster's performance on source and target domain data:

**Source Domain Rollout:**
<p align="center">
  <img src="../../../../docs/img/floodforecaster_source_domain.gif" alt="Source domain rollout animation" width="80%" />
</p>

**Target Domain Rollout (T1):**
<p align="center">
  <img src="../../../../docs/img/floodforecaster_target_domain.gif" alt="Target domain rollout animation" width="80%" />
</p>

These animations show the temporal evolution of water depth and velocity components, comparing ground truth (left panels) with model predictions (right panels) across multiple time steps. The model demonstrates accurate flood forecasting capabilities in both source and target domains, with successful domain adaptation enabling transfer to new river segments.

## Dataset Loading

The dataset is handled via custom dataset classes defined in the `datasets/` module:

- **`FloodDatasetWithQueryPoints`**: Loads raw flood simulation data and generates query points for GINO's latent and output representations
- **`NormalizedDataset`**: Wraps the raw dataset with normalization using `UnitGaussianNormalizer` for static, dynamic, boundary, and target fields
- **`NormalizedRolloutTestDataset`**: Specialized dataset for rollout evaluation that preserves run IDs and cell area information

The datasets automatically:
- Load and concatenate static, dynamic, and boundary features
- Generate query point grids for GINO's coordinate-based processing
- Normalize features using statistics computed from training data
- Handle variable-length time series and multiple simulation runs

To use the datasets, they are instantiated through the training and inference pipelines, which handle data splitting, normalization fitting, and DataLoader creation automatically.

## Evaluation Metrics

FloodForecaster computes comprehensive evaluation metrics:

### Time-Series Metrics

- **RMSE (Root Mean Square Error)**: For water depth (WD) and velocity components (Vx, Vy) at each time step
- **CSI (Critical Success Index)**: Binary classification accuracy at thresholds of 0.05m and 0.3m water depth

### Scalar Hydrological Metrics

- **Arrival Time MAE**: Mean absolute error in flood arrival time (hours)
- **Inundation Duration MAE**: Mean absolute error in flood duration (hours)
- **Max Momentum Flux RMSE**: RMSE of maximum h·V² (m³/s²) across the event
- **FHCA (Flood Hazard Classification Accuracy)**: Classification accuracy for flood hazard categories

### Physics-Informed Metrics

- **Volume Conservation**: Total water volume over time, comparing predictions to ground truth
- **Conditional Error Analysis**: Error distributions conditioned on water depth and velocity magnitude

All metrics are aggregated across multiple test events and saved as both visualizations and numerical data (`.npz` files) for further analysis.

## Configuration

Key configuration sections in `conf/config.yaml`:

### Data Paths

```yaml
source_data:
  root: "${DATA_ROOT:/path/to/source/data}"
  xy_file: "M40_XY.txt"
  static_files: ["M40_XY.txt", "M40_CA.txt", ...]
  dynamic_patterns:
    WD: "M40_WD_{}.txt"
    VX: "M40_VX_{}.txt"
    VY: "M40_VY_{}.txt"
  boundary_patterns:
    inflow: "M40_US_InF_{}.txt"
```

### Model Settings

```yaml
model:
  model_arch: 'gino'  # Note: This codebase is hardcoded for GINO architecture
  data_channels: 20
  out_channels: 3
  fno_n_modes: [16, 16]
  fno_hidden_channels: 64
  gno_embed_channels: 32
```

**Note**: While `model_arch` is a parameter for neuralop's `get_model` function, the FloodForecaster codebase is specifically designed for the GINO (Geometry-Informed Neural Operator) architecture. The code includes GINO-specific wrappers (`GINOWrapper`), data processors (`FloodGINODataProcessor`), and domain adaptation logic that assumes GINO's architecture. Changing `model_arch` to a different model type would require significant code modifications to support other architectures.

### Training Settings

```yaml
training:
  n_epochs_source: 100
  n_epochs_adapt: 50
  learning_rate: 1e-4
  batch_size: 8
  da_lambda_max: 1.0
  da_class_loss_weight: 0.0
```

### Distributed Computing

FloodForecaster uses `physicsnemo`'s `DistributedManager` for distributed training, which automatically detects and configures the distributed environment. The framework supports:

- **torchrun**: For PyTorch-native distributed training
- **mpirun**: For OpenMPI-based distributed training  
- **SLURM**: For cluster-based distributed training

**Configuration:**

The `distributed` section in `config.yaml` contains minimal settings:

```yaml
distributed:
  seed: 123
  device: 'cuda:0'  # Fallback device for non-distributed execution
```

**Key Points:**

- **Device Assignment**: When running in distributed mode (via `torchrun` or `mpirun`), the device is automatically set to `cuda:{local_rank}` for each process. The `device` field in the config is only used as a fallback for single-GPU/CPU execution.

- **No Manual Configuration Needed**: `DistributedManager.initialize()` automatically detects:
  - Number of processes (`world_size`)
  - Process rank (`rank`)
  - Local rank (`local_rank`)
  - Appropriate device assignment

- **Example Distributed Training:**

  ```bash
  # Single node, multiple GPUs
  torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py
  
  # Multi-node (example)
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<master_ip> train.py
  ```

The framework handles all distributed setup automatically - you don't need to specify device lists or wireup configurations.

### Domain Adaptation

```yaml
da_classifier:
  conv_layers:
    - out_channels: 64
      kernel_size: 3
      pool_size: 2
  fc_dim: 1
```

## Logging

FloodForecaster supports logging via [Weights & Biases (W&B)](https://wandb.ai/):

- Training and validation losses for both pretraining and domain adaptation
- Domain classification loss during adversarial training
- Learning rate schedule
- Model checkpoints and training state

Set up W&B by modifying `wandb.log`, `wandb.project`, and `wandb.entity` in `config.yaml`. The framework also uses `physicsnemo`'s `PythonLogger` for distributed training and standard logging.

## Project Structure

```
FloodForecaster/
├── conf/
│   └── config.yaml          # Hydra configuration file
├── datasets/                # Dataset classes
│   ├── flood_dataset.py     # Raw dataset loader
│   ├── normalized_dataset.py # Normalized training dataset
│   └── rollout_dataset.py   # Rollout evaluation dataset
├── data_processing/         # Data preprocessing
│   └── data_processor.py    # GINO data processor and wrappers
├── training/                # Training modules
│   ├── pretraining.py       # Source domain pretraining
│   └── domain_adaptation.py # Domain adaptation fine-tuning
├── inference/               # Inference modules
│   └── rollout.py           # Autoregressive rollout and evaluation
├── utils/                   # Utility functions
│   ├── normalization.py     # Data normalization utilities
│   └── plotting.py          # Visualization and plotting functions
├── train.py                 # Main training script
├── inference.py             # Main inference script
└── README.md
```

## Notes

- **Batching**: GINO supports batching when geometry is shared across samples. For variable geometries, use `batch_size: 1`.
- **GPU Requirements**: GPU with 24GB+ VRAM recommended for larger meshes (>10,000 cells) and longer rollouts.
- **Domain Adaptation**: The model uses adversarial domain adaptation to improve generalization. The gradient reversal lambda can be scheduled during training for improved stability. This approach prevents catastrophic forgetting and enables data-efficient adaptation with as few as 10 training simulations from a new domain.
- **Autoregressive Error**: Long rollouts may accumulate prediction errors. The model uses a sliding window of historical states to mitigate this.
- **Framework Integration**: This example uses Hydra for configuration management and `physicsnemo` utilities for distributed training and logging, following NVIDIA Modulus (PhysicsNeMo) framework patterns.

## Citation

If you use FloodForecaster in your research, please cite:

```bibtex
@article{taghizadeh2025floodforecaster,
  title     = {FloodForecaster: A domain-adaptive geometry-informed neural operator framework for rapid flood forecasting},
  author    = {Taghizadeh, Mehdi and Zandsalimi, Zanko and Nabian, Mohammad Amin and Goodall, Jonathan L. and Alemazkoor, Negin},
  journal   = {Journal of Hydrology},
  volume    = {664},
  pages     = {134512},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.jhydrol.2025.134512},
  url       = {https://doi.org/10.1016/j.jhydrol.2025.134512}
}
```

## Contact

For questions, feedback, or collaborations:

- **Mehdi Taghizadeh** (Code Contributor and Maintainer) – <jrj6wm@virginia.edu>
- **Zanko Zandsalimi** – <mfx2uq@virginia.edu>
- **Mohammad Amin Nabian** – <mnabian@nvidia.com>
- **Jonathan L. Goodall** – <jlg7h@virginia.edu>
- **Negin Alemazkoor** (Corresponding Author) – <na7fp@virginia.edu>
