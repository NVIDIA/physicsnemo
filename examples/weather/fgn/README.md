<!-- markdownlint-disable -->
# Functional Generative Networks for Weather Forecasting

A PhysicsNeMo implementation of Functional Generative Networks (FGN) for
probabilistic global weather forecasting, following the approach of:

> Alet et al., "Skillful joint probabilistic weather forecasting from marginals"
> ([arXiv:2506.10772](https://arxiv.org/abs/2506.10772))

FGN generates ensemble weather forecasts by perturbing a deterministic backbone
with a low-dimensional latent noise vector `z ~ N(0, I_32)` injected through
conditional layer normalization (CLN) at every layer, producing globally coherent
ensemble spread from a marginal (fair-CRPS) training loss. Multiple independently
trained model seeds form a deep ensemble (J=4 in the paper) that captures both
aleatoric and epistemic uncertainty.

## Problem Overview

FGN autoregressively predicts the next 6-hour atmospheric state from the two
previous states (`X_{t-2}`, `X_{t-1}`), sampled from ERA5 (pre-training) and
HRES-fc0 (fine-tuning) at 0.25° global resolution. Each forward pass is
non-diffusive: one pass per forecast step, with a fresh `z` drawn per step per
ensemble member. AR fine-tuning with rollouts up to 8 steps (Table A.2) improves
temporal coherence without requiring a diffusion sampler.

This example implements:

- Latent-conditioned `FGNUNet` backbone (`utils/nn.py`) with AdaGN modulation
- ARCO-backed real dataset using `earth2studio.data.ARCO` (`datasets/arco.py`)
- Fair-CRPS training loss with paper-faithful per-variable and area weights (`utils/loss.py`)
- Autoregressive rollout training with BPTT (`utils/trainer.py`)
- Multi-stage AR schedule runner (Table A.2: `8k·1AR → 4k·2AR → 1k·{3..8}AR`)
- Validation metrics and plots: CRPS, RMSE, spread-skill, rank histograms,
  power spectra (`utils/metrics.py`)
- FSDP + ShardTensor distributed training via `ParallelHelper` (`utils/parallel.py`)
- Deep ensemble inference across multiple independently trained checkpoints
- Per-channel normalization stats with Welford online estimation

## Dataset

Training data is fetched live from the [ARCO ERA5](https://cloud.google.com/storage/docs/public-datasets/era5)
dataset via `earth2studio.data.ARCO`. No local download is required for training.

The dataset covers the full 83-channel Table A.1 schema: 78 atmospheric channels
(6 variables × 13 pressure levels: 50–1000 hPa) plus 5 input/predicted surface
channels (`t2m`, `u10m`, `v10m`, `msl`, `sst`) and `tp06` (6-h accumulated
precipitation, predicted-only). Static inputs (surface geopotential, land-sea mask)
and clock features (local time, year progress sin/cos) are added automatically.

All variables use compact Earth2Studio / PhysicsNeMo names: `u10m`, `v10m`, `t2m`,
`msl`, `sst`, `tp06`, `z{level}`, `q{level}`, `t{level}`, `u{level}`, `v{level}`,
`w{level}`.

### Normalization Stats

Pre-compute per-channel mean and standard deviation before training:

```bash
python scripts/compute_arco_stats.py \
    --start 2020-01-01 --end 2023-12-31 \
    --output rundir/fgn_2024_val/stats_2024.npz
```

Pass the resulting `.npz` file to the trainer via `dataset.stats_path`.

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

PyTorch 2.10 or higher is required for domain parallelism.

### Smoke Test

Run the self-contained synthetic test suite (no GPU, no network access):

```bash
pytest test_training.py
```

Multi-GPU tests require `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 --no-python pytest test_training.py
```

## Configuration

Training is configured with [Hydra](https://hydra.cc) and validated with Pydantic
(`utils/config.py`). Configs live under `config/`:

- `config/fgn.yaml` — base defaults (model, training, dataset structure)
- `config/fgn_arco.yaml` — ARCO real-data training config (inherits from `fgn.yaml`)
- `config/test_fgn.yaml` — fast synthetic smoke-test config

Key config knobs:

| Setting | Description |
|---|---|
| `model.hidden_channels` | U-Net channel width (64 for quick runs, 256+ for full scale) |
| `model.latent_dim` | Latent noise dimension (32, per paper) |
| `training.batch_size` | Global batch size; local per-GPU = `batch_size / data_parallel_size` |
| `training.ar_steps` | AR rollout length for loss (1 = single-step pre-training) |
| `training.loss.num_samples` | Ensemble members per training example (N=2 per paper) |
| `training.domain_parallel_size` | GPUs per sample for domain parallelism (1 = pure DDP) |
| `dataset.stats_path` | Path to `.npz` normalization stats |

Training outputs (checkpoints, logs, plots) are saved to:

```
rundir/{training.experiment_name}/{training.run_id}/
```

## Training

### Single GPU

```bash
python train.py --config-name fgn_arco \
    dataset.stats_path=rundir/fgn_2024_val/stats_2024.npz \
    training.experiment_name=fgn_run \
    training.batch_size=1
```

### Multi-GPU (torchrun)

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train.py --config-name fgn_arco \
    dataset.stats_path=rundir/fgn_2024_val/stats_2024.npz \
    training.experiment_name=fgn_run \
    training.batch_size=2
```

With 2 GPUs and `domain_parallel_size=1` (DDP), `batch_size` is the global batch
size — each GPU processes `batch_size / 2` samples.

### SLURM

```bash
sbatch scripts/train_fgn.sh
```

Override defaults via environment variables:

```bash
sbatch --export=ALL,EXP_NAME=fgn_2024,RUN_ID=1,STEPS=10000 scripts/train_fgn.sh
```

See `scripts/train_fgn.sh` for all overridable variables (`EXP_NAME`, `RUN_ID`,
`STEPS`, `CFG`, `STATS_PATH`, `NGPU`).

### Resuming

Set `training.resume_checkpoint=latest` (default) to automatically resume from
the most recent checkpoint in the run directory.

### Domain Parallelism

For models too large to fit one sample on a single GPU, enable domain parallelism:

```bash
torchrun --standalone --nproc_per_node=4 \
    train.py --config-name fgn_arco \
    training.domain_parallel_size=2 \
    training.batch_size=2
```

With `domain_parallel_size=2` and 4 GPUs: 2 domain-parallel pairs, each handling
1 sample (`batch_size / data_parallel_size = 2 / 2 = 1`).

### AR Fine-Tuning Schedule (Table A.2)

The trainer implements the paper's multi-stage AR schedule automatically when
`training.ar_steps` increases across runs. Start with single-step pre-training,
then resume with progressively longer rollouts:

| Stage | `ar_steps` | Steps | Notes |
|---|---|---|---|
| 1 | 1 | 8000 | Single-step pre-train |
| 2 | 2 | 4000 | Resume from stage 1 |
| 3–8 | 3–8 | 1000 each | Resume from previous |

## Inference

Run stochastic ensemble inference from a trained checkpoint:

```bash
python inference.py --config-name inference_fgn \
    inference.checkpoint=rundir/fgn_run/0/checkpoints/FGNUNet.mdlus
```

For deep ensemble inference across multiple independently trained seeds:

```bash
python inference.py --config-name inference_fgn \
    "inference.checkpoints=[seed0/FGNUNet.mdlus, seed1/FGNUNet.mdlus, seed2/FGNUNet.mdlus, seed3/FGNUNet.mdlus]"
```

Trajectories are distributed across checkpoints following paper §2.2.1.

### Bad-Seed Detection

Before including a checkpoint in a deep ensemble, check its spectral properties:

```bash
python scripts/check_spectra.py \
    --checkpoint rundir/fgn_run/0/checkpoints/FGNUNet.mdlus \
    --stats rundir/fgn_2024_val/stats_2024.npz
```

## Adding Custom Datasets

Implement the `FGNDataset` interface from `datasets/dataset.py`:

```python
class MyDataset(FGNDataset):
    def state_channels(self) -> list[str]: ...
    def background_channels(self) -> list[str]: ...
    def image_shape(self) -> tuple[int, int]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx): ...
    # Optional:
    def get_invariants(self) -> np.ndarray | None: ...
    def output_only_channels(self) -> list[int]: ...
```

`__getitem__` should return a dict with keys `history` (shape `(T, C, H, W)`),
`target` (shape `(K, C, H, W)`), and optionally `background`. Register your
dataset by placing it in `datasets/` — it is discovered automatically via
`pkgutil.iter_modules` at import time.

## Memory Management

At 0.25° (721×1440), each training sample is large. Recommended settings for an
80 GB H100:

- `training.batch_size=2` (1 per GPU) with 2 GPUs, `domain_parallel_size=1`
- bf16 AMP is enabled automatically (`torch.autocast(bfloat16)`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in `train_fgn.sh`)

For larger models (hidden_channels ≥ 256), use `domain_parallel_size=2` with
4+ GPUs, or enable gradient checkpointing via `model.checkpoint_level`.

## References

- [Skillful joint probabilistic weather forecasting from marginals](https://arxiv.org/abs/2506.10772)
- [Generative Ensemble Downscaling with Diffusion Models (CorrDiff)](https://arxiv.org/abs/2308.14453)
- [Kilometer-Scale Convection Allowing Model Emulation (StormCast)](https://arxiv.org/abs/2408.10958)
- [GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794)
