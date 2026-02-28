# GLOBE on AirFRANS (2D External Aerodynamics)

This example trains a [GLOBE](/physicsnemo/physicsnemo/models/globe/model.py)
([paper](https://arxiv.org/abs/2511.15856)) model to predict aerodynamic flow
fields around 2D airfoils using the
[AirFRANS](https://airfrans.readthedocs.io/) dataset.

## Problem Description

Given an airfoil boundary mesh and freestream conditions, the model predicts
volume fields including velocity, pressure coefficients, turbulent viscosity,
and surface forces at arbitrary query points. The architecture is
discretization-invariant, rotation-equivariant, and uses learnable Green's
function-like kernels evaluated from boundary faces to interior target points.

## Dataset

AirFRANS provides Reynolds-Averaged Navier-Stokes (RANS) simulation data for
~1,000 2D airfoil geometries at varying angles of attack and Reynolds numbers.

1. Download the dataset from the AirFRANS repository:
   <https://airfrans.readthedocs.io/>

2. Set the dataset location via one of:
   - The `--data-dir` CLI argument
   - The `AIRFRANS_DATA_DIR` environment variable (set automatically by `run.sh`)

The dataset root should contain `manifest.json` and the individual sample
directories.

## Running

### Training

Single-node:

```bash
uv run torchrun --nproc-per-node $NUM_GPUS train.py
```

Multi-node via SLURM (see `run.sh` for the full configuration):

```bash
sbatch run.sh
```

Key training arguments (see `uv run python train.py --help` for all options):

- `--airfrans-task`: Dataset split (`full`, `scarce`, `reynolds`, `aoa`)
- `--amp`: Enable automatic mixed precision
- `--use-compile`: Enable `torch.compile` (default: True)
- `--use-mlflow / --no-use-mlflow`: Toggle MLflow tracking

### MLflow

Training metrics are tracked via MLflow. The tracking URI is configured via
`MLFLOW_TRACKING_URI` (set in `run.sh`). To view the dashboard:

```bash
./mlflow_launch.sh
```

### Inference

```bash
uv run python inference.py
```

Set `GLOBE_OUTPUT_DIR` to point at a specific training run, or the script
automatically selects the most recent output directory.

## File Overview

| File | Purpose |
|---|---|
| `train.py` | Training loop, loss function, model construction |
| `dataset.py` | AirFRANS preprocessing, caching, DataLoader creation |
| `utilities.py` | Checkpointing, device transfer, distributed helpers |
| `inference.py` | Single-sample inference and visualization |
| `run.sh` | SLURM launch script for multi-node training |
| `mlflow_launch.sh` | Launches the MLflow UI |
