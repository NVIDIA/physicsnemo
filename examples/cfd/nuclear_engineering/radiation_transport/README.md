# Radiation Transport with Transolver

A PhysicsNeMo example that trains a [Transolver](https://arxiv.org/abs/2402.02366)
surrogate for the steady-state radiative-transfer equation on two canonical
2-D benchmarks from the thermal-radiation-transport literature: **lattice**
and **hohlraum**. The training pipeline uses a physics-informed loss that
combines region-weighted MSE on the scalar flux with a quantity-of-interest
(QoI) penalty.

---

## Table of contents

1. [The science](#1-the-science)
2. [Installation](#2-installation)
3. [Dataset](#3-dataset)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Interpreting model performance](#6-interpreting-model-performance)
7. [Configuration reference](#7-configuration-reference)
8. [File overview](#8-file-overview)
9. [References](#9-references)

---

## 1. The science

The model solves the steady-state radiative-transfer equation for a scalar
flux field `φ(x)` over a 2-D domain. Inputs to the surrogate are:

- **Coordinates** `(x, y)` per cell, normalized to `[-1, 1]` and augmented with
  Fourier features (3 frequencies × 2 axes × {sin, cos} = 12 extra channels).
- **Material properties** per cell: absorption coefficient `σ_a`, scattering
  coefficient `σ_s`, and total `σ_t = σ_a + σ_s`. Lattice cases additionally
  include a heat source `Q`.

The surrogate predicts the **z-score-of-log scalar flux**, which is then
inverted via `transforms.denormalize_flux` to recover the physical flux.

### 1.1 Lattice benchmark

A unit square partitioned into a 7×7 grid of material blocks. Each block is
either **absorber** (high `σ_a`, low `σ_s`), **scatterer** (low `σ_a`, high
`σ_s`), or **source** (interior `Q > 0`). The model has to capture sharp flux
discontinuities at material interfaces and reproduce the integrated
absorption rate in the central region.

QoI: integrated absorption `∫_Ω_c σ_a · φ dA` over the central source block.

### 1.2 Hohlraum benchmark

An axisymmetric cylindrical cavity with optional interior void regions,
representing a simplified inertial-confinement-fusion target. There is no
interior heat source — flux enters from boundary conditions and propagates
through the cavity. Geometry parameters (upper/lower laser-entry radii,
center offsets) vary across simulations.

QoI: per-region integrated absorption over `{center, vertical strip,
horizontal strip, total domain}`. By default `train.physics_loss.qoi_region=all`
averages all four region losses so every region contributes to the gradient;
set it to a single region (`center`, `vertical`, `horizontal`, `total`) to
backprop on that region alone. Either way, all four region losses are
logged each batch.

---

## 2. Installation

The example is in the PhysicsNeMo repo. From the example directory:

```bash
cd physicsnemo/examples/cfd/nuclear_engineering/radiation_transport
pip install -r requirements.txt
```

Prerequisites:

- **PyTorch ≥ 2.6** — `torch.optim.Muon` is built in. Earlier PyTorch versions
  work if you stick to the default `train.optimizer.type=adam`.
- **PhysicsNeMo** — install the host repo with `[model-extras,datapipes-extras]`
  to get `physicsnemo.models.transolver.Transolver` and the `tensordict`-based
  data utilities. We don't need `gnns` / `mesh-extras` / `uq-extras`.

Quick install via `uv` (PhysicsNeMo's recommended package manager):

```bash
cd <physicsnemo_root>
uv venv .venv --python 3.13
source .venv/bin/activate
# Pick the CUDA wheel that matches your driver:
#   driver supports CUDA 13.x  -> cu13   (default in pyproject)
#   driver only supports 12.x  -> cu12
uv pip install -e ".[cu12,model-extras,datapipes-extras]"
uv pip install tensorboard                                  # for TB logging
```

#### TransformerEngine (default `model.use_te=true`)

The Transolver model imports `transformer_engine.pytorch` at module load
time, even when `use_te=false`. You must have a TE wheel matching your
PyTorch CUDA version installed. For the cu12 venv:

```bash
uv pip install --reinstall --no-cache transformer-engine-cu12==2.14.1
uv pip install transformer-engine-torch transformer_engine
```

> **uv quirk:** The first `--reinstall --no-cache` is required. Without
> it, uv may silently drop the 855 MB `libtransformer_engine.so` from the
> wheel and the import will fail with `Could not find shared object file
> for Transformer Engine core lib`.

If your driver is on CUDA 13, swap `cu12` → `cu13` everywhere above.

Verify:

```bash
python -c "
from physicsnemo.models.transolver import Transolver
from torch.optim import Muon
import zarr, tensordict, torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('Transolver:', Transolver)
"
```

---

## 3. Dataset

### 3.1 Download

> **TODO:** HuggingFace dataset URL. Until then, raw simulation data has to
> be curated through the upstream RTE workshop pipeline.

### 3.2 Expected on-disk layout

```
<DATA_ROOT>/
├── lattice/
│   ├── lattice_abs<a>_scatter<s>_p<p>_q<q>.zarr/
│   └── ...
├── hohlraum/
│   ├── hohlraum_variable_cl<...>_q<...>_ulr<...>_llr<...>_<...>.zarr/
│   └── ...
├── splits/
│   ├── lattice_splits.json     # train/val/test split lists
│   └── hohlraum_splits.json
└── stats/
    ├── lattice_flux_stats.yaml
    ├── lattice_material_stats.yaml
    ├── hohlraum_flux_stats.yaml
    └── hohlraum_material_stats.yaml
```

### 3.3 What's in each zarr store

Each `*.zarr/` directory is one simulation. Keys (read by `dataset.ZarrDataReader`):

| Key | Shape | Notes |
|---|---|---|
| `scalar_flux` | `(T, N)` or `(N,)` | physical flux (W m⁻²·sr⁻¹). Steady-state stores have `T=1`. |
| `sigma_a`, `sigma_s` | `(N,)` | absorption / scattering coefficients per cell |
| `Q` | `(N,)` | heat source (lattice only; zeros in hohlraum) |
| `coordinates` | `(N, 2)` | cell-center positions in physical units |
| `cell_areas` | `(N,)` | per-cell areas — used by physics loss for surface integrals |
| `material_labels` | `(N,)` | integer region IDs (consumed by `LatticeMaterialMapper` / `HohlraumMaterialMapper`) |
| `metadata` | dict | timestep, sim_time, geometry params (hohlraum), filename |

`N` is the number of cells per simulation (~tens of thousands). Different
simulations may have different `N` — point-cloud collation handles this.

### 3.4 Splits file format

The dataset reader (`dataset._load_split_from_file`) expects a wrapped
JSON document with a `"splits"` key:

```json
{
  "case_type": "lattice",
  "split_name": "default",
  "total_samples": 707,
  "train_size": 494,
  "val_size": 106,
  "test_size": 107,
  "splits": {
    "train": ["lattice_abs52.5_scatter4.6_p0.015_q6", ...],
    "val":   ["lattice_abs85.0_scatter9.1_p0.015_q6", ...],
    "test":  ["lattice_abs77.5_scatter4.1_p0.015_q6", ...]
  }
}
```

Filenames in the splits arrays are zarr **basenames** without the `.zarr`
suffix; the reader appends it automatically when opening stores.

If the splits file is named with a suffix (e.g. `lattice_splits_default.json`,
`lattice_splits_overfit_1sample.json`), point at it explicitly:

```bash
... case.split_file=<DATA_ROOT>/splits/lattice_splits_overfit_1sample.json
```

### 3.5 Computing normalization stats

If `<DATA_ROOT>/stats/<case>_{flux,material}_stats.yaml` are missing (e.g. you
re-curated the data, or you started from a fresh download that only ships
flux stats), generate them with:

```bash
python src/compute_normalizations.py \
    --data_path <DATA_ROOT>/lattice \
    --case_type lattice \
    --output_dir <DATA_ROOT>/stats \
    --steady_state

python src/compute_normalizations.py \
    --data_path <DATA_ROOT>/hohlraum \
    --case_type hohlraum \
    --output_dir <DATA_ROOT>/stats \
    --steady_state
```

Pass `--split_file ...` to compute stats over the train split only (matches
what training will see). Drop `--steady_state` for time-resolved data.

The flux stats YAML contains the log-flux mean/std/min/max + `clip_threshold`,
used by `RTEFluxLogClip` and `denormalize_flux`. The material stats YAML
contains per-channel mean/std/min/max for `{σ_a, σ_s, σ_t, Q}`.

---

## 4. Training

### 4.1 Quick start

Lattice:

```bash
python src/train.py case=lattice data=lattice case.data_root=<DATA_ROOT>
```

Hohlraum:

```bash
python src/train.py case=hohlraum data=hohlraum case.data_root=<DATA_ROOT>
```

Single-process default: 501 epochs, AMP-bf16, cosine LR with 10 warmup epochs,
peak LR 3e-5, physics loss enabled at weight 0.005 (lattice) / 0.01 (hohlraum).

### 4.2 Multi-GPU

```bash
torchrun --nproc_per_node=N src/train.py \
    case=lattice data=lattice case.data_root=<DATA_ROOT>
```

DDP is auto-detected via `physicsnemo.distributed.DistributedManager`. Set
`data.preload_data=true` (default) so each rank loads its static arrays into
host RAM through a sequenced barrier; this is much faster than re-reading
zarr per epoch but uses ~`N_train × N × 4 bytes × num_static_fields` of memory
on rank 0.

### 4.3 Common overrides

| Override | Effect |
|---|---|
| `train.epochs=200` | Shorter run |
| `train.optimizer.type=muon` | Use `torch.optim.Muon` for 2-D weights, Adam for biases / norms |
| `train.amp=false` | Disable mixed precision (debug / numerical parity) |
| `train.physics_loss.qoi_region=center` | Hohlraum-only: backprop on a single region. Default `all` averages the four (center, vertical, horizontal, total). |
| `train.physics_loss.weight=0.0` | Pure MSE training (disables QoI penalty) |
| `train.dataloader.num_workers=4` | DataLoader workers |
| `model.num_spatial_points=8192` | Subsample cells per training step (–1 = use all) |
| `model.n_layers=12 model.n_hidden=384` | Bigger Transolver |
| `model.use_te=true` | Use NVIDIA TransformerEngine layers (requires `[model-extras]`) |
| `train.resume_checkpoint=path/to/checkpoint.0.0.pt` | Resume from a checkpoint |

### 4.4 Output structure

Per run, under `outputs/${project.name}/${case.type}/${exp_tag}/`:

```
outputs/RTE_Transolver/lattice/transolver/
├── hydra/
│   ├── config.yaml          # resolved Hydra config (canonical record of the run)
│   ├── hydra.yaml
│   └── overrides.yaml
├── checkpoints/
│   ├── checkpoint.0.0.pt              # latest training-state checkpoint (every train.checkpoint_interval)
│   ├── Transolver.0.0.mdlus           # latest model weights only
│   ├── best_model_epoch_<E>/          # snapshot of the lowest val_loss epoch
│   ├── best_qoi_model/                # snapshot of the lowest val_qoi epoch (use this for QoI eval)
│   └── top_model/                     # current top-1 by val_loss
├── tensorboard/             # TB event files (open with `tensorboard --logdir tensorboard/`)
└── train.log
```

`best_qoi_model/` is the checkpoint to feed `inference.py` when comparing
runs by QoI relative error. `best_model_epoch_<E>/` and `top_model/` track
val MSE.

---

## 5. Evaluation

### 5.1 Run inference

```bash
python src/inference.py \
    --checkpoint_dir outputs/RTE_Transolver/lattice/transolver/checkpoints/best_qoi_model \
    --data_path <DATA_ROOT> \
    --case_type lattice \
    --output_dir results/lattice
```

CLI options:

| Flag | Effect |
|---|---|
| `--checkpoint_dir DIR` | A directory containing `Transolver.0.0.mdlus` + `checkpoint.0.0.pt`. Pass either a `best_*/` snapshot dir or the run's `checkpoints/` root (where `find_best_checkpoint` will pick the latest). |
| `--data_path DIR` | The same `<DATA_ROOT>` you trained against. The script overrides `case.data_root`/`split_file`/`stats` paths from this. |
| `--case_type {lattice,hohlraum}` | Required. |
| `--output_dir DIR` | Where to write metrics + figures. Default: `<run_dir>/evaluation`. |
| `--num_samples N` | Limit to the first `N` test simulations (default: all). |
| `--num_workers N` | DataLoader workers. |
| `--device {cpu,cuda,cuda:0,...}` | Defaults to CUDA if available. |
| `--num_plot_samples N` | Number of `flux_panels_<idx>.png` figures to write (default: 4). |

### 5.2 Outputs

```
<output_dir>/
├── metrics.yaml          # field-level metrics over the whole test set
├── qoi_metrics.yaml      # per-region QoI relative error
└── figures/
    ├── flux_panels_<idx>.png   # target / prediction / error 3-panel for sample <idx>
    ├── true_vs_pred.png        # scatter of all (true, pred) flux values
    └── error_histogram.png     # distribution of pointwise (pred − true)
```

### 5.3 Metric definitions

`metrics.yaml::overall` is computed once over **all** evaluation samples
flattened together (denormalized to physical flux):

| Key | Definition |
|---|---|
| `mse` | `mean((pred − target)^2)` |
| `rmse` | `sqrt(mse)` |
| `mae` | `mean(|pred − target|)` |
| `l2_relative_error` | `‖pred − target‖₂ / ‖target‖₂` — the headline number |
| `relative_error` | `mean(|pred − target| / |target|)` — sensitive to near-zero target cells, often dominated by void regions |
| `max_error` | `max(|pred − target|)` |

`metrics.yaml::per_sample_aggregate` reports `{mean, std, min, max}` of each
metric across simulations — useful for catching outliers (one bad simulation
dominating the mean).

`qoi_metrics.yaml` reports per-region:

| Key | Definition |
|---|---|
| `mae` | mean absolute error of the integrated QoI scalar |
| `rmse` | RMSE of the integrated QoI scalar |
| `max_error` | worst single-simulation QoI error |
| `mean_relative_error_pct` | mean of `100 · |Q_pred − Q_true| / |Q_true|` |
| `median_relative_error_pct` | median of the same |
| `max_relative_error_pct` | worst single-simulation relative error |

For lattice, the only region is `cur_absorption` (central source block). For
hohlraum, you'll see entries keyed by whichever `qoi_region` was active
during training.

### 5.4 Comparing runs

The single most useful comparison is **`qoi_metrics.yaml::<region>::mean_relative_error_pct`**.
Below 5% on hohlraum-center is competitive with classical solvers on these
benchmarks; below 1% is the published Transolver target after full training.

For field-level comparisons, use `metrics.yaml::overall::l2_relative_error`.
Values below 5% indicate the model has learned the global flux structure;
below 1% means it's also picking up sharp interface features.

---

## 6. Interpreting model performance

### 6.1 What "good" looks like (after full training, ~500 epochs)

| Benchmark | l2_relative_error | QoI mean_relative_error_pct |
|---|---|---|
| Lattice (center QoI) | 1–3% | 0.5–2% |
| Hohlraum (center QoI) | 2–5% | 1–3% |

These targets assume the default Transolver size (`n_layers=8, n_hidden=256,
slice_num=128`) and the published 7×7 lattice / variable-geometry hohlraum
distribution.

### 6.2 Reading the training log

Per-epoch line you'll see in `train.log`:

```
Epoch 250: train_loss=4.23e-03, val_loss=5.91e-03,
           train_mse=4.18e-03, val_mse=5.84e-03,
           train_qoi=2.15e-02, val_qoi=2.43e-02,
           train_qoi_center=2.15e-02, val_qoi_center=2.43e-02,
           lr=2.81e-05
```

Key signals:

- **`val_loss` plateauing while `train_loss` keeps falling** → overfitting.
  Lower `model.dropout`, raise `train.region_weights.material_weight`, or
  use `--num_samples` per-rank subsetting to grow the effective training
  data.
- **`val_qoi` stuck near 1.0** while `val_mse` shrinks → model is learning
  the bulk flux but not preserving the integral. Increase
  `train.physics_loss.weight`, or extend `train.physics_loss.warmup_epochs`
  to let MSE settle first.
- **`val_loss` oscillates wildly** → reduce LR (`train.learning_rate=1e-5`)
  or shorten `train.warmup_epochs`.
- **`lr` dropping below `train.min_learning_rate`** late in training → cosine
  schedule has bottomed out; consider a longer `train.epochs` for a slower
  decay.

### 6.3 Reading the inference figures

- **`flux_panels_<idx>.png`** — three panels: target, prediction, signed
  error. Sharp interface features in the target should appear (slightly
  blurred) in the prediction; the error panel should be near-zero in
  homogeneous regions and concentrated along material interfaces. Persistent
  bias of one sign (all-positive or all-negative error) indicates a
  systematic offset — usually a normalization stat issue.
- **`true_vs_pred.png`** — points should lie close to the `y = x` diagonal
  across the full dynamic range. A "fan" near the origin is normal (low-flux
  void cells are hard); fans at the high end are not normal and usually
  indicate undertraining or saturation in the model's last layer.
- **`error_histogram.png`** — should be symmetric around zero with thin
  tails. Heavy-tailed asymmetric errors typically mean the QoI loss is
  off-balance with the MSE loss.

### 6.4 Common pitfalls

- **Hohlraum's `embedding_dim` mismatch.** With `case.include_q_in_embedding=false`
  the adapter produces 3 channels (no `Q`), so the model's `embedding_dim`
  must also be 3. The `case/hohlraum.yaml::embedding_dim_override: 3`
  handles this; if you override `model.embedding_dim` directly without
  matching the case config you'll get a silent shape mismatch at the first
  forward pass.
- **AMP underflow on the QoI integral.** `train.amp=true` casts the forward
  pass to bf16, but the physics loss runs in fp32 internally (denormalized
  flux is sensitive to log-domain spread). If you see `loss_qoi=NaN` early
  in training, check that your dataset's flux range fits inside
  `clip_threshold` correctly.
- **Stale `top_model/` after CLI override of `output:`.** The `top_model/`
  symlink is per-run; if you rerun with the same `${output}` path the new
  run will overwrite the old top model. Either change `exp_tag=...` or
  `output=...` to keep separate run trees.

---

## 7. Configuration reference

All training hyperparameters live under `src/conf/`, composed by Hydra:

```
src/conf/
├── config.yaml             # root: composes case / data / model / train
├── case/{lattice,hohlraum}.yaml
├── data/{lattice,hohlraum}.yaml
├── model/transolver.yaml
└── train/base.yaml
```

`config.yaml` defaults list:

```yaml
defaults:
  - case: lattice
  - data: lattice
  - model: transolver
  - train: base
  - _self_
```

CLI overrides follow Hydra's standard syntax:

```bash
python src/train.py \
    case=hohlraum data=hohlraum \
    case.data_root=/path/to/data \
    train.epochs=300 \
    train.optimizer.type=muon \
    train.physics_loss.weight=0.02 \
    model.n_layers=12 model.n_hidden=384
```

The Hydra group structure means `case=hohlraum` swaps the entire
`case/hohlraum.yaml` (including `physics_loss_weight`, `qoi_region`,
`include_q_in_embedding`, and `embedding_dim_override`). The downstream
`train/base.yaml` and `model/transolver.yaml` interpolate from `${case.*}`
so case-specific overrides propagate automatically.

---

## 8. File overview

| File | LOC | Purpose |
|---|---|---|
| `src/train.py` | ~500 | Hydra entry; inlined Transolver build/forward/loss_inputs; dispatches to trainer |
| `src/trainer.py` | ~700 | Training loop, gradient accumulation, DDP primitives, TB logging |
| `src/losses.py` | ~1080 | MSE / region-weighted / physics-informed losses, torch QoI helpers, schedulers |
| `src/checkpointing.py` | ~600 | Save/load checkpoints, resume, optimizer (Adam + `torch.optim.Muon`) and scheduler factory |
| `src/inference.py` | ~850 | Checkpoint inference, metrics, plots; argparse CLI |
| `src/compute_normalizations.py` | ~440 | One-shot CLI to compute flux + material statistics over a zarr root |
| `src/dataset.py` | ~860 | Zarr reader, PyTorch Dataset, stats loaders |
| `src/transforms.py` | ~740 | Transform framework + flux / coordinate / sampling / QoI transforms |
| `src/material.py` | ~530 | Lattice/hohlraum material mappers + material lookup transform |
| `src/loader.py` | ~1200 | TransolverAdapter, collate, datapipe orchestration, DataLoader factory |

Total: ~7.5 KLOC across 10 flat source files (no `__init__.py`, no
subpackages).

---

## 9. References

- **Transolver:** Wu, H. et al. ["Transolver: A Fast Transformer Solver for
  PDEs on General Geometries"](https://arxiv.org/abs/2402.02366), ICML 2024.
- **Lattice benchmark:** Brunner, T. A. (2002). *Forms of approximate
  radiation transport*, SAND2002-1778.
- **Hohlraum benchmark:** Tencer, J. et al. (2018). *A multifidelity Monte
  Carlo method for thermal radiation transport*, JCP 376.
- **PhysicsNeMo:** [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo).

---

## 10. Full-dataset commands (this workstation)

Tested layout for the local high-fidelity dataset at
`/home/carmelog/Projects/Datasets/RTE/high_fidel_zarr/new_zarr_stores_t0_tfinal/`
(709 lattice sims, 846 hohlraum sims, splits and pre-computed flux + material
stats already present). The default `case/{lattice,hohlraum}.yaml` paths
match this layout exactly — no `case.split_file` or
`data.flux_normalization_stats_file` overrides needed.

### Setup (once)

This workstation has two GPUs: GPU 0 is an RTX A400 (4 GB, too small for the
default model), GPU 1 is an RTX 6000 Ada (48 GB) — these are the indices
`nvidia-smi -L` reports. Pin training to GPU 1.

> **Note on GPU ordering.** PyTorch defaults to FASTEST_FIRST ordering, which
> swaps the two cards relative to `nvidia-smi`. Setting
> `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes the CUDA indices match
> `nvidia-smi`'s, so `CUDA_VISIBLE_DEVICES=1` reliably picks the 6000 Ada.

```bash
# Activate the cu12 venv (PyTorch with CUDA 12.8, matches the 570.x driver).
source /home/carmelog/Projects/Workshops/RTE/physicsnemo/.venv/bin/activate
cd /home/carmelog/Projects/Workshops/RTE/physicsnemo/examples/cfd/nuclear_engineering/radiation_transport

# Pin training to the RTX 6000 Ada (PCI index 1).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export DATA_ROOT=/home/carmelog/Projects/Datasets/RTE/high_fidel_zarr/new_zarr_stores_t0_tfinal
```

Sanity-check (should print `True NVIDIA RTX 6000 Ada Generation`):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Train — lattice (default config: 501 epochs, Muon, AMP-bf16)

Single line (paste-safe — no backslash continuations to break across lines):

```bash
python src/train.py case=lattice data=lattice case.data_root=$DATA_ROOT train.optimizer.type=muon exp_tag=lattice_full
```

### Train — hohlraum

```bash
python src/train.py case=hohlraum data=hohlraum case.data_root=$DATA_ROOT train.optimizer.type=muon exp_tag=hohlraum_full
```

> **Heads-up.** If you see Hydra's `LexerNoViableAltException` with a stray
> `^`, it means an override expanded to empty — usually `$DATA_ROOT` wasn't
> exported in the current shell. Run `echo "$DATA_ROOT"` to confirm it's
> non-empty before launching.

The hohlraum config picks up `physics_loss_weight=0.01`, `qoi_region=center`,
`include_q_in_embedding=false`, and `embedding_dim_override=3` automatically
from `case/hohlraum.yaml`.

### Monitor training

```bash
# Live log
tail -f outputs/RTE_Transolver/lattice/lattice_full/train.log
# Or hohlraum
tail -f outputs/RTE_Transolver/hohlraum/hohlraum_full/train.log

# TensorBoard
tensorboard --logdir outputs/RTE_Transolver --port 6006
```

### Evaluate — lattice

```bash
python src/inference.py --checkpoint_dir outputs/RTE_Transolver/lattice/lattice_full/checkpoints/best_qoi_model --data_path $DATA_ROOT --case_type lattice --output_dir results/lattice_full
```

### Evaluate — hohlraum

```bash
python src/inference.py --checkpoint_dir outputs/RTE_Transolver/hohlraum/hohlraum_full/checkpoints/best_qoi_model --data_path $DATA_ROOT --case_type hohlraum --output_dir results/hohlraum_full
```

After both runs you'll have:

```
results/
├── lattice_full/
│   ├── metrics.yaml          # field-level: l2_relative_error is the headline number
│   ├── qoi_metrics.yaml      # cur_absorption: target <2% mean_relative_error_pct
│   └── figures/{flux_panels_*.png, true_vs_pred.png, error_histogram.png}
└── hohlraum_full/
    ├── metrics.yaml
    ├── qoi_metrics.yaml      # qoi_<region>: target <3% mean_relative_error_pct on center
    └── figures/{...}
```

Compare runs by `qoi_metrics.yaml::<region>::mean_relative_error_pct` for QoI
fidelity, and `metrics.yaml::overall::l2_relative_error` for global flux
fidelity. See §6.1 for target ranges on each benchmark.

### Optional: shorter run for a first pass

If you want to confirm the pipeline before committing to the full 501-epoch
schedule, run 50 epochs first:

```bash
python src/train.py case=lattice data=lattice case.data_root=$DATA_ROOT train.optimizer.type=muon train.epochs=50 train.warmup_epochs=5 exp_tag=lattice_quick
```

Expect `val_loss` around 5e-2–1e-1 and `val_qoi` somewhere below 0.5 by
epoch 50 with the default model size on the full dataset.
