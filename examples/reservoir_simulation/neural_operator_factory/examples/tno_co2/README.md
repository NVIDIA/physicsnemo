# TNO for CO2 Sequestration

Reproduces the CO2 sequestration results from the Temporal Neural Operator
(TNO) paper:

> **Temporal Neural Operator for Modeling Time-Dependent Physical Phenomena**
> Diab, W. & Al Kobaisi, M. (2025).
> [arXiv:2504.20249](https://arxiv.org/abs/2504.20249)

## Overview

The Temporal Neural Operator (TNO) extends DeepONet with a temporal branch
(t-branch) that processes the previous solution state, enabling
autoregressive temporal predictions with temporal bundling.  The
architecture uses a Hadamard product to combine branch, t-branch, and
trunk outputs, followed by a temporal projection decoder that maps from
the latent space directly to K output timesteps.

### Architecture

| Component | Configuration |
|-----------|--------------|
| Branch1 | Lifting + 1 x UNet2D + ReLU (input fields) |
| Branch2 (t-branch) | Lifting + 1 x UNet2D + ReLU (previous solution) |
| Trunk | 14-layer tanh FNN, linear output (time only) |
| Decoder | Temporal projection: MLP + Linear(p → K) |
| Width (p) | 96 (saturation), 128 (pressure) |
| L / K | 1 / 3 (temporal bundling) |

### Training Configuration

| Setting | Value |
|---------|-------|
| Regime | Autoregressive: 90 TF + 40 rollout epochs |
| Rollout mode | `live_gradients` (full-trajectory loss) |
| Loss | Relative L2 |
| Optimizer | Adam, lr=6e-4, weight_decay=1e-4 |
| Scheduler | StepLR(step_size=2, gamma=0.92) |
| Batch size | 4 per GPU × 8 GPUs |
| Training timesteps | First 16 of 24 (up to 1.8 years) |

### Temporal Extrapolation

The model is trained on the first 16 timesteps (up to 1.8 years) and
tested on all 24 timesteps.  Timesteps 17-24 (2.6 to 30 years) are
temporal extrapolation — the model must predict dynamics it never saw
during training, while also generalizing to 500 unseen geological
realizations.

## Dataset

Same CO2 sequestration dataset as other NOF examples (Wen et al. 2022):

- **Grid**: 96 × 200 (variable thickness per realization)
- **Time steps**: 24 (1 day to 30 years, logarithmic spacing)
- **Input channels**: 12 (4 spatial fields + 5 scalars + grid coordinates)
- **Samples**: 4,500 train / 500 val / 500 test

The dataset is publicly available at:
<https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing>

## Usage

All commands from `neural_operator_factory/`:

### Training

```bash
# Gas saturation (p=96, ~2.7M params)
sbatch examples/tno_co2/train.sbatch saturation_training_config

# Pressure buildup (p=128, ~7.7M params)
sbatch examples/tno_co2/train.sbatch pressure_training_config
```

### Evaluation

```bash
sbatch examples/tno_co2/eval.sbatch saturation
sbatch examples/tno_co2/eval.sbatch pressure
```

Note: the pressure evaluation script applies `r_dnorm_dP` to convert the
test targets from physical bar units to normalized space before feeding
to the AR rollout, and `dnorm_dP` to convert predictions back to bar for
metric computation.

## Results

### Test Set (500 samples, 24 timesteps including 8 extrapolation)

| Variable | Metric | NOF | Paper |
|----------|--------|-----|-------|
| **Saturation** | MPE | 4.89% | ~3-5% |
| | MAE | 0.0079 | ~0.005-0.01 |
| | R² | 0.958 | ~0.96 |
| **Pressure** | MRE | 1.27% | ~1-2% |
| | MAE | 1.11 bar | ~1-2 bar |
| | R² | 0.985 | ~0.98 |

### Validation (16 timesteps, generalization only)

| Variable | Best Val Loss | Best Metric | Epoch |
|----------|--------------|-------------|-------|
| Saturation | 0.0798 | MPE = 1.81% | 130 |
| Pressure | 0.0577 | MRE = 0.65% | 130 |

## Key Implementation Details

- **Temporal projection decoder**: The trunk is queried once; a linear
  head projects from width to K=3 output timesteps directly (paper Eq. 10).
  Faster than per-timestep trunk queries.

- **Trunk output activation**: Set to `false` (linear output) to avoid
  squashing the Hadamard product's dynamic range.  Other DeepONet variants
  use activated trunk output (`true`).

- **Live-gradient rollout**: During the rollout training stage, predictions
  are collected with live gradients through the entire chain, and a single
  loss is computed on the concatenated trajectory.  DDP gradient sync is
  handled via `model.no_sync()` + manual AllReduce.

- **BatchNorm freeze**: During live-gradient rollout, BatchNorm layers are
  set to eval mode to prevent inplace running-stat updates from
  invalidating the autograd graph across chained forward passes.

## Files

```text
tno_co2/
├── README.md
├── train.sbatch
├── eval.sbatch
├── evaluate_pressure.py        # r_dnorm_dP + dnorm_dP for test data
├── evaluate_saturation.py
└── conf/
    ├── model_config.yaml               # TNO architecture + loss
    ├── saturation_training_config.yaml  # p=96, 90 TF + 40 rollout
    └── pressure_training_config.yaml   # p=128, 90 TF + 40 rollout
```

## Reference

```bibtex
@article{diab2025tno,
  title={Temporal Neural Operator for Modeling Time-Dependent
    Physical Phenomena},
  author={Diab, Waleed and Al Kobaisi, Mohammed},
  journal={arXiv preprint arXiv:2504.20249},
  year={2025},
  url={https://arxiv.org/abs/2504.20249}
}
```
