# Neural Operator Factory

A config-driven framework for training neural operator surrogates
for reservoir simulation, built on
[PhysicsNeMo](https://github.com/NVIDIA/physicsnemo).
Switch between 165 model architectures, 6 training regimes, and
physics-informed loss combinations — all from YAML, zero code changes.

## Why NOF

Pick a model, a training strategy, and a loss function.
The framework handles everything else: multi-GPU distribution,
autoregressive rollout, inactive-cell masking, checkpointing,
and metric tracking.

**165 architectures** from composable building blocks:

| Family | How it works | Count |
|--------|-------------|-------|
| **xFNO** | Fourier base +/- UNet +/- Conv | 5 |
| **DeepONet** | 1 branch (8 types) + trunk | 16 |
| **MIONet** | 2 branches + trunk | 16 |
| **TNO** | 2 independent branches + trunk | 128 |

Spatial branches are assembled from **three independent layer types**
(Fourier, UNet, Conv) that can be freely combined — a Fourier-UNet
branch, a Conv-only branch, or a triple Fourier-UNet-Conv hybrid
are all one config change apart.

**6 training regimes**, all model-agnostic:

| Regime | Description |
|--------|-------------|
| Full-mapping | Entire trajectory in one forward pass |
| AR: teacher forcing | Ground-truth input at each window |
| AR: TF + rollout (detached) | Free-running with per-window gradients |
| AR: TF + rollout (live) | Full-trajectory backprop through the chain |
| AR: TF + pushforward + rollout | Curriculum unroll bridging TF and rollout |

**Physics-informed losses** compose on top:

- **Data**: MSE, L1, relative L2, Huber (combinable with weights)
- **Derivative regularization**: central differences on any
  spatial axis, using actual cell widths from the grid
- **Mass conservation**: volume-weighted spatial integration
  constraint at each timestep

Together this gives **~46,500** unique configurations from YAML alone.

## What You Get for Free

Every configuration — from a basic FNO to a triple-hybrid TNO —
automatically inherits:

- **Multi-GPU DDP** with proper gradient sync, distributed
  sampling, and rank-0-only I/O
- **DDP-safe autoregressive rollout** (`no_sync` + manual
  AllReduce for multi-forward AR steps)
- **Automatic mask detection** (ACTNUM, non-zero fallback,
  per-sample) propagated through all losses and metrics
- **Dimension-agnostic pipeline** — the same code handles 2D and
  3D spatial data; losses, AR utilities, and masking adapt
  automatically
- **Lazy module initialization** — dummy forward pass materializes
  all layers before DDP wrapping, with correct branch2 handling
  for TNO and MIONet
- **Self-describing checkpoints** — architecture config saved
  alongside weights for one-line model reconstruction
- **Optimizer and scheduler resume** — seamless training
  continuation from any checkpoint
- **BatchNorm freeze** during live-gradient rollout to prevent
  autograd graph invalidation
- **Mixed precision** (AMP) for any model via a single flag (works with selected models)
- **Configurable validation metrics** (RMSE, MAE, MRE, MPE,
  relative L2) with automatic per-sample masking

## Quick Start

```bash
# Train a TNO on Norne (8-GPU DDP via SLURM)
sbatch examples/pi_norne/train.sbatch pressure_training_config

# Train a U-FNO on CO2 saturation
sbatch examples/ufno_co2/train.sbatch U-FNO saturation_training_config

# Evaluate
sbatch examples/tno_co2/eval.sbatch saturation
```

All commands run from the `neural_operator_factory/` directory.

## Reproduced Papers

Each example ships with configs that reproduce published results:

| Example | Paper | Architecture | Dataset |
|---------|-------|-------------|---------|
| [pi_norne](examples/pi_norne/) | — | Fourier-DeepONet + feedback | Norne field (3D) |
| [ufno_co2](examples/ufno_co2/) | Wen et al. 2022 | FNO, Conv-FNO, U-FNO | CO2 sequestration |
| [udeeponet_co2](examples/udeeponet_co2/) | Diab & Al Kobaisi 2024 | U-DeepONet | CO2 sequestration |
| [fourier_mionet_co2](examples/fourier_mionet_co2/) | Jiang et al. 2024 | MIONet, Fourier-MIONet | CO2 sequestration |
| [tno_co2](examples/tno_co2/) | Diab & Al Kobaisi 2025 | TNO | CO2 sequestration |

## Dataset Format

Input and output tensors in `.pt` format:

- **2D spatial**: input `(N, H, W, T, C)`, output `(N, H, W, T)`
- **3D spatial**: input `(N, X, Y, Z, T, C)`, output `(N, X, Y, Z, T)`

The last input channels must follow the NOF grid convention:

| 2D (last 3 channels) | 3D (last 4 channels) | Used by |
|-----------------------|----------------------|---------|
| grid\_x (W widths) | grid\_x (X widths) | Derivative loss |
| grid\_y (H widths) | grid\_y (Y widths) | Derivative loss |
| — | grid\_z (Z widths) | Derivative loss |
| grid\_t (time) | grid\_t (time) | DeepONet trunk |

Inactive cells must be zero in all channels.  The framework
auto-detects binary ACTNUM masks, falls back to non-zero
pattern detection, and supports per-sample masks when reservoir
geometry varies across realizations.

The CO2 sequestration dataset used by the included examples is
publicly available at:
<https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing>

## Project Structure

```text
neural_operator_factory/
├── models/              xfno.py, xdeeponet.py, unet.py, physicsnemo_unet.py
├── data/                dataloader.py, validation.py, scalar_utils.py
├── training/            train.py, losses.py, physics_losses.py, ar_utils.py, metrics.py
├── utils/               checkpoint.py, padding.py, co2_normalization.py
├── conf/                model_config.yaml, training_config.yaml
├── examples/            pi_norne/, ufno_co2/, udeeponet_co2/, fourier_mionet_co2/, tno_co2/
└── tests/               375 unit tests
```

## References

1. Wen, G. et al. (2022). "U-FNO — An enhanced Fourier neural
   operator-based deep-learning model for multiphase flow."
   *Advances in Water Resources*, 163, 104180.
2. Diab, W. & Al Kobaisi, M. (2024). "U-DeepONet: U-Net
   enhanced deep operator network for geologic carbon
   sequestration." *Scientific Reports*, 14, 21298.
3. Jiang, Z. et al. (2024). "Fourier-MIONet: Fourier-enhanced
   multiple-input neural operators for multiphase modeling of
   geological carbon sequestration."
   *Reliability Eng. & System Safety*, 251, 110392.
4. Diab, W. & Al Kobaisi, M. (2025). "Temporal neural operator
   for modeling time-dependent physical phenomena."
   *Scientific Reports*, 15.
5. Li, Z. et al. (2021). "Fourier Neural Operator for Parametric
   Partial Differential Equations." *ICLR 2021*.
6. Lu, L. et al. (2021). "Learning nonlinear operators via
   DeepONet." *Nature Machine Intelligence*, 3, 218-229.
7. Jin, P., Meng, S. & Lu, L. (2022). "MIONet: Learning
   multiple-input operators via tensor product."
   *SIAM J. Scientific Computing*, 44(6), A3490-A3514.
8. Zhu, M. et al. (2023). "Fourier-DeepONet: Fourier-enhanced
   deep operator networks for full waveform inversion."
   *arXiv:2305.17289*.
9. Chandra, A. et al. (2025). "Neural operators for
   accelerating scientific simulations and design."
   *arXiv:2503.11031*.

## License

Apache License 2.0. See [LICENSE](../../../LICENSE.txt).
