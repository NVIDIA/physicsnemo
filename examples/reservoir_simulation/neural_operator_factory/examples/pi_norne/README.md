# Physics-Informed Fourier-DeepONet for Norne Field Simulation

Physics-informed neural operator surrogate for the Norne reservoir
simulation dataset using a 4D Fourier-DeepONet with derivative
regularization, mass conservation losses, and autoregressive feedback.

## Overview

This example trains a Fourier-DeepONet on the Norne field dataset — a
real-world 3D reservoir model based on the publicly available
[Norne Field](https://github.com/OPM/opm-data/tree/master/norne) dataset.
Norne requires volumetric 4D operators
(3D spatial + time) and handles complex geological features including
numerous faults with Non-Neighbor Connections (NNCs), pinch-outs, and
39% inactive cells.

A Design of Experiment study identified fault transmissibility and
KVKH multipliers as key uncertain parameters, which were varied using
Latin Hypercube Sampling (LHS) to generate 500 realizations.  The
primary LHS variable is PERMZ (vertical permeability), controlling the
Kv/Kh ratio.  All simulations were generated using the open-source
[OPM](https://opm-project.org/) reservoir simulator.

### Pressure Architecture (Fourier-DeepONet)

| Component | Configuration |
|-----------|--------------|
| Branch1 | Linear encoder → 6 Fourier layers, gelu, modes 10×10×6 |
| Trunk | 12-layer tanh FNN, time input, linear output |
| Decoder | Temporal projection (width → K=1) |
| Feedback | Previous prediction appended as extra input channel |
| Width | 64 |
| Parameters | 118M |

### Saturation Architecture (Fourier-DeepONet — SWAT / SGAS)

| Component | Configuration |
|-----------|--------------|
| Branch1 | Linear encoder → 6 Fourier layers, gelu, modes 10×10×6 |
| Trunk | 12-layer tanh FNN, time input, linear output |
| Decoder | Temporal projection (width → K=1) |
| Feedback | Previous prediction appended as extra input channel |
| Width | 64 |
| Parameters | 118M |

### Losses

| Variable | Data Loss | Derivative | Mass Conservation |
|----------|----------|-----------|------------------|
| Pressure | Relative L2 (w=1.0) | dx, dy, dz (w=0.5) | Disabled |
| SWAT | L1 (w=1.0) | dx, dy, dz (w=0.5) | Enabled (w=0.5) |
| SGAS | L1 (w=1.0) | dx, dy, dz (w=0.5) | Enabled (w=0.5) |

### Training Configuration

| Setting | Pressure | Saturations (SWAT / SGAS) |
|---------|----------|--------------------------|
| Regime | AR: 10 TF + 90 rollout | AR: 10 TF + 90 rollout |
| Rollout mode | `detached` | `detached` |
| L / K | 3 / 1 | 3 / 1 |
| Feedback channel | enabled | enabled |
| Batch size | 1 per GPU × 8 GPUs | 1 per GPU × 8 GPUs |
| Optimizer | Adam, lr=1e-3, wd=1e-4 | Adam, lr=1e-3, wd=1e-4 |
| Scheduler | StepLR(10, 0.85) | StepLR(10, 0.85) |
| Masking | ACTNUM ch 5 (39.2%) | ACTNUM auto-detect |
| Normalize | true | false |

## Dataset

Norne field reservoir simulation (500 LHS realizations, OPM simulator):

- **Grid**: 46 × 112 × 22 (113,344 total cells, 44,431 active — 39.2%)
- **Time steps**: 65 (0 to 3,260 days, ~9 years of operation)
- **Wells**: 36 (producers and injectors, multi-layer horizontal completions)
- **LHS variable**: PERMZ (vertical permeability / Kv/Kh ratio)
- **Input channels**: 11
  - Static: PERMX (log10), PERMZ (log10), PORO, PORV, NTG, ACTNUM
  - Coordinates: grid_x, grid_y, grid_z (normalized)
  - Dynamic: grid_t (normalized), WCID (+1 injector / -1 producer / 0 none)
- **Output variables** (separate models):
  - `pressure` — cell pressure (bar)
  - `swat` — water saturation (fraction)
  - `sgas` — gas saturation (fraction)
- **Samples**: 400 train / 50 val / 50 test

### Comparison with CO2 Dataset

| Aspect | CO2 | Norne |
|--------|-----|-------|
| Spatial dims | 2D (96 × 200) | 3D (46 × 112 × 22) |
| Total cells | 19,200 | 113,344 |
| Active cells | variable (~53%) | 44,431 (39.2%) |
| Timesteps | 24 | 65 |
| Samples | 5,500 | 500 |
| Input channels | 12 | 11 |
| Simulator | Custom | OPM |

## Usage

All commands from `neural_operator_factory/`:

### Training

```bash
# Pressure
sbatch examples/pi_norne/train.sbatch pressure_training_config

# Water saturation (SWAT)
sbatch examples/pi_norne/train.sbatch swat_training_config

# Gas saturation (SGAS)
sbatch examples/pi_norne/train.sbatch sgas_training_config
```

### Evaluation

```bash
# Pressure (normalize + feedback)
NORMALIZE=1 FEEDBACK=1 sbatch examples/pi_norne/eval.sbatch pressure

# Saturations (feedback, no normalization)
FEEDBACK=1 sbatch examples/pi_norne/eval.sbatch swat
FEEDBACK=1 sbatch examples/pi_norne/eval.sbatch sgas

# With explicit checkpoint
NORMALIZE=1 FEEDBACK=1 \
  CHECKPOINT=checkpoints/best_model_pressure_deeponet3d_fourier_deeponet_linear.pth \
  sbatch examples/pi_norne/eval.sbatch pressure
```

## Results

### Pressure (Fourier-DeepONet)

| Metric | Value |
|--------|-------|
| MAE | 0.92 bar |
| RMSE | 1.54 bar |
| Relative L2 | 0.54% |
| R² | 0.999 |
| Parameters | 118M |
| Training time | 3 hr 24 min (8× H100, 100 epochs) |

### Water Saturation (SWAT)

| Metric | Value |
|--------|-------|
| MAE | 2.91e-3 |
| RMSE | 7.89e-3 |
| Relative L2 | 1.05% |
| R² | 0.9996 |
| Parameters | 118M |
| Training time | 3 hr 17 min (8× H100, 100 epochs) |

### Gas Saturation (SGAS)

| Metric | Value |
|--------|-------|
| MAE | 1.09e-2 |
| RMSE | 4.17e-2 |
| Relative L2 | 14.1% |
| R² | 0.977 |
| Parameters | 118M |
| Training time | 3 hr 15 min (8× H100, 100 epochs) |

## Files

```text
pi_norne/
├── README.md
├── train.sbatch                        # 8 GPU, 4 hour time limit
├── eval.sbatch                         # 1 GPU evaluation
└── conf/
    ├── pressure_model_config.yaml      # Pressure architecture + loss (no mass conservation)
    ├── swat_model_config.yaml          # SWAT architecture + loss (mass conservation enabled)
    ├── sgas_model_config.yaml          # SGAS architecture + loss (mass conservation enabled)
    ├── pressure_training_config.yaml
    ├── swat_training_config.yaml
    └── sgas_training_config.yaml

Each variable has its own model config. All three currently use the same
Fourier-DeepONet architecture with per-variable loss configuration.
```
