# Radiation Transport with Transolver

A PhysicsNeMo example that trains a [Transolver](https://arxiv.org/abs/2402.02366)
surrogate model for the 2-D linear radiation transport benchmark defined in
[Reference solutions for linear radiation transport: the Hohlraum and Lattice
benchmarks](https://arxiv.org/pdf/2505.17284). The pipeline learns the
steady-state mapping from the initial flux snapshot to the final scalar flux,
using a physics-informed loss that combines region-weighted MSE with a
quantity-of-interest (QoI) penalty based on absorption in key regions.

The datasets used for this example were generated using
[KiT-RT](https://github.com/KiT-RT) [^1].

---

## 1. The science

The model solves the steady-state radiative-transfer equation for a scalar
flux field `φ(x)` over a 2-D domain. Inputs to the surrogate are:

- **Coordinates** `(x, y)` per cell, normalized to `[-1, 1]` and augmented with
  Fourier features (3 frequencies × 2 axes × {sin, cos} = 12 extra channels).
- **Material properties** per cell: absorption coefficient `σ_a`, scattering
  coefficient `σ_s`, total cross-section `σ_t`, and, for lattice cases, heat
  source `Q`. Boundary input flux may be incorporated from upstream hohlraum
  data, but it is not used as a model input in this example.

The surrogate predicts the **z-score-of-log scalar flux**, which is then
inverted via `transforms.denormalize_flux` to recover the physical flux.

### 1.1 Lattice benchmark

A square domain partitioned into a 7×7 grid of material blocks. Each block is
either **absorber** (high `σ_a`, low `σ_s`), **scatterer** (low `σ_a`, high
`σ_s`), or **source** (interior `Q > 0`). The model has to capture sharp flux
discontinuities at material interfaces and reproduce the instantaneous
absorption rate in the absorbing regions.

QoI: instantaneous absorption `σ_a · φ · A` over the absorbing blocks.

### 1.2 Hohlraum benchmark

An axisymmetric cylindrical cavity with interior void regions,
representing a simplified inertial-confinement-fusion target. There is no
interior heat source — flux enters from boundary conditions and propagates
through the cavity. Geometry parameters (upper/lower laser-entry radii,
center offsets) vary across simulations.

QoI: per-region instantaneous absorption over `{center, vertical strip,
horizontal strip, total domain}`. By default `train.physics_loss.qoi_region=all`
averages all four region losses so every region contributes to the gradient;
set it to a single region (`center`, `vertical`, `horizontal`, `total`) to
backprop on that region alone. Either way, all four region losses are
logged each batch.

---

## 2. Installation

Prerequisites:

- **PyTorch ≥ 2.6** — `torch.optim.Muon` is built in. Earlier PyTorch versions
  work if you stick to the default `train.optimizer.type=adam`.
- **PhysicsNeMo** — install the host repo with `[model-extras,datapipes-extras]`
  to get `physicsnemo.models.transolver.Transolver` and the `tensordict`-based
  data utilities.

From the PhysicsNeMo repo root, install the example dependencies:

```bash
uv pip install -e ".[model-extras,datapipes-extras]" tensorboard
```

---

## 3. Dataset

### 3.1 Data source

**TODO:** HuggingFace dataset URL. Until then, raw simulation data may be
curated from the [KiT-RT repositories](https://github.com/KiT-RT).

### 3.2 Expected on-disk layout

The runtime data format is the PhysicsNeMo `Mesh` memmap layout. Each
simulation lives in a `<name>.mesh/` directory next to a `<name>.attrs.json`
sidecar, loaded via `physicsnemo.mesh.Mesh.load(<name>.mesh)`.

```text
<DATA_ROOT>/
├── lattice/
│   ├── lattice_abs<a>_scatter<s>_p<p>_q<q>.mesh/
│   ├── lattice_abs<a>_scatter<s>_p<p>_q<q>.attrs.json
│   └── ...
├── hohlraum/
│   ├── hohlraum_variable_cl<...>_q<...>_ulr<...>_llr<...>_<...>.mesh/
│   ├── hohlraum_variable_cl<...>_q<...>_ulr<...>_llr<...>_<...>.attrs.json
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

### 3.3 What's in each mesh store

Each `*.mesh/` directory is one simulation, written by
`physicsnemo.mesh.Mesh.save(...)`. The loader uses the first and final
`scalar_flux` snapshots and ignores intermediate snapshots. The fields are:

`Mesh.points` — `(N, 3)` float32 cell-center coordinates.

`Mesh.point_data` (per-cell tensors):

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `cell_areas` | `(N,)` | float32 | per-cell areas — used by physics loss for surface integrals |
| `sigma_a`, `sigma_s`, `sigma_t` | `(N,)` | float32 | absorption / scattering / total cross-section per cell |
| `Q` | `(N,)` | float32 | heat source (non-zero in lattice; zeros in hohlraum) |
| `geometric_features` | `(N, k)` | float32 | optional per-cell geometric features |
| `material_properties` | `(N,)` | int64 | integer region IDs (consumed by `LatticeMaterialMapper` / `HohlraumMaterialMapper`) |
| `scalar_flux` | `(N, T)` | float32 | physical flux, transposed to put cells first |

`Mesh.global_data` (per-simulation tensors):

| Key | Shape | Notes |
|---|---|---|
| `sim_times` | `(T,)` | simulation times for each flux snapshot |
| `attr__<key>` | scalar / `(...)` | numeric simulation attributes flattened from the source curator |

`<name>.attrs.json` (sidecar):
A JSON file holding `raw_attrs` (the verbatim source attrs dict — final
simulation time, geometry params, etc.) and `residue_attrs` (the
non-numeric attrs that don't fit in `global_data`). `MeshDataReader.load`
exposes `raw_attrs` as the `metadata` `NonTensorData` entry on the returned
`TensorDict`.

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

Filenames in the splits arrays are **basenames** without any format
suffix; the reader appends `.mesh` when opening stores.

If the splits file is named with a different suffix, point at it explicitly:

```bash
... case.split_file=<DATA_ROOT>/splits/my_split_file.json
```

### 3.5 Computing normalization stats

If `<DATA_ROOT>/stats/<case>_{flux,material}_stats.yaml` are missing (e.g. you
re-curated the data, or you started from a fresh download that only ships
flux stats), generate them with:

```bash
python src/compute_normalizations.py \
    --data_path /Datasets/lattice \
    --case_type lattice \
    --split_file /Datasets/splits/lattice_splits.json \
    --output_dir /Datasets/stats

python src/compute_normalizations.py \
    --data_path /Datasets/hohlraum \
    --case_type hohlraum \
    --split_file /Datasets/splits/hohlraum_splits.json \
    --output_dir /Datasets/stats
```

`--split_file` is required so stats are computed over the same train split
used by training.

The flux stats YAML contains the log-flux mean/std/min/max + `clip_threshold`,
used by `RTEFluxLogClip` and `denormalize_flux`. The material stats YAML
contains per-channel mean/std/min/max for `{σ_a, σ_s, σ_t, Q}`.

---

## 4. Training

### 4.1 Quick start

Full-mesh training used at least a 48 GB GPU during development (RTX6000 Ada).
By default `data.preload_data=true`, so the train and validation splits are
loaded into host RAM before training. Disable with `data.preload_data=false`
if RAM is tight, at the cost of slower per-epoch reads from disk.

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

Use `torchrun` for DDP. A plain `python src/train.py ...` launch runs as a
single process, even inside an allocated SLURM shell. Set `data.preload_data=true`
(default) so each rank loads static arrays through a sequenced barrier; this is
faster than re-reading the mesh stores per epoch but uses host RAM proportional to the
training split size.

### 4.3 Common overrides

| Override | Effect |
|---|---|
| `train.epochs=200` | Shorter run |
| `train.optimizer.type=muon` | Use `torch.optim.Muon` for 2-D weights, Adam for biases / norms |
| `train.amp=false` | Disable mixed precision (debug / numerical parity) |
| `train.physics_loss.qoi_region=center` | Hohlraum-only: backprop on a single region. Default `all` averages the four (center, vertical, horizontal, total). |
| `train.physics_loss.weight=0.0` | Pure MSE training (disables QoI penalty) |
| `train.dataloader.num_streams=4` | CUDA streams used by `physicsnemo.datapipes.DataLoader` for prefetch overlap (no CPU fork workers) |
| `train.dataloader.use_streams=false` | Disable CUDA-stream prefetching — useful for debugging or CPU-only runs |
| `train.dataloader.prefetch_factor=4` | How many batches to prefetch ahead |
| `model.num_spatial_points=8192` | Subsample cells per training step (–1 = use all) |
| `model.n_layers=12 model.n_hidden=384` | Bigger Transolver |
| `model.use_te=true` | Use NVIDIA TransformerEngine layers (requires `[model-extras]`) |
| `train.resume_checkpoint=.../checkpoints/latest_checkpoint` | Resume from a checkpoint directory |
| `train.latest_checkpoint_interval=0` | Disable the rolling `latest_checkpoint/` directory (`null` also works) |

### 4.4 Output structure

Per run, under `outputs/${project.name}/${case.type}/${exp_tag}/`:

```text
outputs/RTE_Transolver/lattice/transolver/
├── hydra/
│   ├── config.yaml          # resolved Hydra config (canonical record of the run)
│   ├── hydra.yaml
│   └── overrides.yaml
├── checkpoints/
│   ├── checkpoint.0.0.pt              # periodic training-state checkpoint (every train.checkpoint_interval)
│   ├── Transolver.0.0.mdlus           # periodic model weights
│   ├── latest_checkpoint/             # rolling full-state resume checkpoint (train.latest_checkpoint_interval)
│   ├── best_model_epoch_<E>/          # snapshot of the lowest val_loss epoch
│   ├── best_qoi_model/                # snapshot of the lowest validation QoI-loss epoch
│   └── top_model/                     # current top-1 by val_loss
├── tensorboard/             # TB event files (open with `tensorboard --logdir tensorboard/`)
└── train.log
```

When loading checkpoints, `best_model_epoch_<E>/` and `top_model/` track
validation loss, while `best_qoi_model/` tracks validation QoI loss.

---

## 5. Evaluation

### 5.1 Run inference

```bash
python src/inference.py \
    --checkpoint_dir outputs/RTE_Transolver/lattice/transolver/checkpoints/top_model \
    --data_path <DATA_ROOT> \
    --case_type lattice \
    --split_file <DATA_ROOT>/splits/lattice_splits.json \
    --output_dir results/lattice
```

By default, `--flux_stats_file` is read from the checkpoint's saved hydra
config — pass `--flux_stats_file <PATH>` to override.

CLI options:

| Flag | Effect |
|---|---|
| `--checkpoint_dir DIR` | A directory containing `Transolver.0.0.mdlus` + `checkpoint.0.0.pt`. Pass either a `best_*/` snapshot dir or the run's `checkpoints/` root (where inference will use `top_model`). |
| `--data_path DIR` | The dataset root containing the per-case mesh stores (e.g. `<DATA_ROOT>/lattice/*.mesh`). |
| `--case_type {lattice,hohlraum}` | Required. |
| `--split_file FILE` | Required explicit split JSON. |
| `--flux_stats_file FILE` | Optional override for the flux-normalization YAML recorded in the checkpoint's hydra config. If omitted, the training-time path is reused. The matching `<case>_material_stats.yaml` is read from the same directory. |
| `--output_dir DIR` | Where to write metrics + figures. Default: `<run_dir>/evaluation`. |
| `--num_samples N` | Limit to the first `N` test simulations (default: all). |
| `--device {cpu,cuda,cuda:0,...}` | Defaults to CUDA if available. |
| `--num_plot_samples N` | Number of `flux_panels_<idx>.png` figures to write (default: 3). |

### 5.2 Outputs

```text
<output_dir>/
├── metrics.yaml          # field-level metrics over the whole test set
├── qoi_metrics.yaml      # per-region QoI relative error
└── figures/
    ├── flux_panels_<idx>.png   # target / prediction / error 3-panel for sample <idx>
    ├── true_vs_pred.png        # scatter of all (true, pred) flux values
    └── error_histogram.png     # distribution of pointwise absolute error
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

For lattice, the only region is `cur_absorption`. For hohlraum, inference
reports center, vertical, horizontal, and total QoI components when metadata is
available.

### 5.4 Comparing runs

The single most useful comparison is
**`qoi_metrics.yaml::<region>::mean_relative_error_pct`**. On the default
randomized splits, a well-trained surrogate should reach low single-digit
percent QoI error.

For field-level comparisons, use `metrics.yaml::overall::l2_relative_error`,
which helps interpret global flux structure and sharp interface features.

---

## 6. Interpreting model performance

### 6.1 What "good" looks like (after full training, ~500 epochs)

| Benchmark | l2_relative_error | QoI mean_relative_error_pct |
|---|---|---|
| Lattice (absorption QoI) | 0.60% | 0.23% |
| Hohlraum (regional QoI) | 2.06% | 0.52–0.73% |

These observed values come from the default full-training runs with defaults
configs. Training logs converged to final validation losses of about
`2.10e-05` for lattice and `1.51e-05` for hohlraum.

### 6.2 Reading the training log

Each epoch logs train/validation MSE, QoI loss, learning rate, and checkpoint
updates. A typical completed epoch looks like:

```text
Epoch 500: train_loss=1.7081e-05, val_loss=2.0973e-05,
    train_mse=1.7032e-05, val_mse=2.0900e-05,
    train_qoi=9.8040e-06,  val_qoi=1.4658e-05, lr=1.00e-06
[checkpoint][INFO] - Saved model state dictionary:
    ./checkpoints/Transolver.0.500.mdlus
Training completed!
Top validation losses: ['0.000021', '0.000021', '0.000021']
Best QoI loss: 5.887844e-06
```

### 6.3 Reading the inference figures

- **`flux_panels_<idx>.png`** — three panels: target, prediction, absolute
  error.
- **`true_vs_pred.png`** — points should lie close to the `y = x` diagonal
  across the full dynamic range.
- **`error_histogram.png`** — distribution of pointwise absolute error; lower
  and thinner tails are better.

---

## 7. Configuration reference

All training hyperparameters live under `src/conf/`, composed by Hydra:

```text
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

## 8. Tests

Pipeline regression tests live under `tests/test_pipeline.py` and exercise the
dataset / dataloader contract end-to-end against a small dev split. They skip
cleanly when the dataset isn't present.

```bash
python -m pytest examples/cfd/nuclear_engineering/radiation_transport/tests/ -v
```

The tests expect a converted dev dataset at
`/home/<user>/Projects/Datasets/RTE/devset/mesh/{lattice,hohlraum}/` (12 mesh
stores per case) plus the matching `splits/` and `stats/` directories. If your
layout differs, adjust the `_DATASET_ROOT` constant at the top of the test
file.

---

## References

[^1]: Kusch, J., Schotthöfer, S., Stammer, P., Wolters, J., & Xiao, T. (2023).
"KiT-RT: An extendable framework for radiative transfer and therapy."
*ACM Transactions on Mathematical Software*, **49**(4), 1–24.

```bibtex
@article{kitrt2023,
  title     = {KiT-RT: An extendable framework for radiative transfer and therapy},
  author    = {Kusch, Jonas and Schotth{\"o}fer, Steffen and Stammer, Pia
               and Wolters, Jannick and Xiao, Tianbai},
  journal   = {ACM Transactions on Mathematical Software},
  volume    = {49},
  number    = {4},
  pages     = {1--24},
  year      = {2023},
  publisher = {ACM New York, NY}
}
```
