# Radiation Transport with Transolver

A PhysicsNeMo example that trains a [Transolver](https://arxiv.org/abs/2402.02366)
surrogate model for the 2-D linear radiation transport benchmark defined in
[Reference solutions for linear radiation transport: the Hohlraum and Lattice
benchmarks](https://arxiv.org/pdf/2505.17284).[^2] The pipeline learns the
final-time mapping from the initial flux snapshot to the final scalar flux,
using a physics-informed training objective that can combine void/material-
weighted MSE with a quantity-of-interest (QoI) penalty based on absorption in
key regions.

The dataset used for this example was generated using
[KiT-RT](https://github.com/KiT-RT),[^1] curated into the
PhysicsNeMo `Mesh` format, and published on Hugging Face:
[Linear Radiation Transport][hf-rte].

[hf-rte]: https://huggingface.co/datasets/nvidia/Linear-Radiation-Transport

---

## 1. The science

The model approximates the final-time scalar flux `φ(x)` of the 2-D linear
radiative-transfer equation. The simulator is run forward in time and the
training target is the last snapshot — the underlying transport problem
is not run to convergence. Inputs to the surrogate are:

- **Coordinates** `(x, y)` per cell, normalized to `[-1, 1]` and augmented with
  Fourier features (3 frequencies × 2 axes × {sin, cos} = 12 extra channels).
- **Material properties** per cell: absorption coefficient `σ_a`, scattering
  coefficient `σ_s`, total cross-section `σ_t`, and particle source `Q`.
  `Q` is non-zero for source cells in the lattice case and zero in the
  hohlraum case. Boundary input flux may be present in upstream simulation
  data, but it is not used as a model input in this example.

The surrogate predicts the **z-score-of-log scalar flux**, which is then
inverted via `transforms.denormalize_flux` to recover the physical flux.

### 1.1 Lattice benchmark

A square domain partitioned into a 7×7 grid of material blocks. The material
layout contains blue absorbing blocks, red scattering/source blocks, and a
white scattering background. The model has to capture sharp flux
discontinuities at material interfaces and reproduce the integrated
absorption in the absorbing regions.

**QoI** — matches **QoI-3** of the reference paper (Schotthöfer et al. 2025, §3.1):
the final-time radiation absorption over the absorbing blocks `B`:

$$\mathrm{QoI}_{\mathrm{Lattice}} = \int_{B} \sigma_a(x)\,\phi(x, T)\,dx.$$

In code this is `cur_absorption`, computed as
`Σ_{c ∈ B} σ_a,c · φ_c · A_c` over absorber cells.

![Lattice: target, prediction, absolute error of final-time flux][lattice-fig]

[lattice-fig]: ../../../docs/img/radiation_transport/transolver_lattice.png

### 1.2 Hohlraum benchmark

A symmetric 2-D hohlraum-style cavity benchmark with interior void regions,
red wall strips, and a center insert/capsule region. There is no interior
particle source — flux enters from boundary conditions and propagates through
the cavity. Geometry varies across simulations through eight scalar
parameters:

- `ulr`, `llr`: upper/lower extent of the left red wall strip
- `urr`, `lrr`: upper/lower extent of the right red wall strip
- `hlr`, `hrr`: left/right horizontal wall-strip positions
- `cx`, `cy`: center insert/capsule offsets

**QoI** — variation of **QoI-2** of the reference paper (Schotthöfer et al.
2025, §3.2): final-time absorption evaluated separately over three regions:

$$\mathrm{QoI}_{\mathrm{Hohlraum}, S} = \int_{S} \sigma_a(x)\,\phi(x, T)\,dx.$$

In the PhysicsNeMo evaluator, the three regions are labeled
`cur_absorption_{center, vertical, horizontal}` and each is computed as
`Σ_{c ∈ S} σ_a,c · φ_c · A_c`.

The training-time physics loss evaluates relative-squared-error losses for
the three component absorptions and adds a fourth `total` loss on the sum
of those three component absorptions. The QoI penalty is the mean of these
four loss terms. Inference reports the three component QoIs only.

![Hohlraum: target, prediction, absolute error of final-time flux][hohlraum-fig]

[hohlraum-fig]: ../../../docs/img/radiation_transport/transolver_hohlraum.png

---

## 2. Installation

Prerequisites:

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

The curated dataset is available on Hugging Face:
[Linear Radiation Transport][hf-rte]. Raw simulations can be regenerated or
curated using the [KiT-RT](https://github.com/KiT-RT/kitrt_code) solver and
the [CharmKiT](https://github.com/KiT-RT/charm_kit) workflow scripts.

### 3.2 Expected on-disk layout

The runtime data format is the PhysicsNeMo `Mesh` memmap layout. Each
simulation lives in a `<name>.pmsh/` directory next to a `<name>.attrs.json`
sidecar, loaded via `physicsnemo.mesh.Mesh.load(<name>.pmsh)`.

Set `<DATA_ROOT>` to the directory that directly contains the `lattice/`,
`hohlraum/`, `splits/`, and `stats/` directories. If using the Hugging Face
tarballs exactly as published, this may be the extracted `mesh/` directory
rather than the parent download directory.

```text
<DATA_ROOT>/
├── lattice/
│   ├── lattice_abs<a>_scatter<s>_p<p>_q<q>.pmsh/
│   ├── lattice_abs<a>_scatter<s>_p<p>_q<q>.attrs.json
│   └── ...
├── hohlraum/
│   ├── hohlraum_variable_cl<...>_q<...>_ulr<...>_llr<...>_<...>.pmsh/
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

Each `*.pmsh/` directory is one simulation written via
`physicsnemo.mesh.Mesh.save(...)`. The flux series is stored as just
the first and final snapshots (`T = 2`); only those are used.

Cell-center coordinates and per-cell areas are derived from the mesh topology
via `mesh.cell_centroids` and `mesh.cell_areas`.

`Mesh.cell_data` (per-cell tensors the loader requires):

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `scalar_flux` | `(N, 2)` | float32 | flux at first / final snapshot, cells-first |
| `material_id` | `(N,)` | int64 | region IDs mapped by the material-property transforms |
| `sigma_a`, `sigma_s`, `sigma_t` | `(N,)` | float32 | absorption / scattering / total cross-section |
| `Q` | `(N,)` | float32 | particle source; non-zero in lattice source cells, zero in hohlraum |

`Mesh.global_data`: the loader consumes only `sim_time` (shape `(2,)`,
simulation time of each flux snapshot).
Other simulation diagnostics shipped with the data (`cur_absorption`,
`total_absorption`, `mass`, ...) are ignored at training time, but may be
useful for other downstream tasks.

`<name>.attrs.json` (sidecar): JSON with `case_type`,
`simulation_params`, `solver_config`, and `mesh_info`. The dataset exposes
sidecar-derived metadata alongside each loaded sample.

`N` is the number of cells per simulation (~tens of thousands). In the
published dataset, lattice samples use a fixed cell count, while hohlraum
samples may have different `N`; point-cloud collation handles variable-size
meshes.

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
    "train": ["lattice_abs52.5_scatter4.6_p0.015_q6", "..."],
    "val":   ["lattice_abs85.0_scatter9.1_p0.015_q6", "..."],
    "test":  ["lattice_abs77.5_scatter4.1_p0.015_q6", "..."]
  }
}
```

Filenames in the splits arrays may be basenames with no suffix or filenames
ending in `.pmsh`; the reader normalizes entries to `.pmsh` when opening
stores.

If the splits file is named with a different suffix, point at it explicitly:

```bash
... case.split_file=<DATA_ROOT>/splits/my_split_file.json
```

### 3.5 Computing normalization stats

The Hugging Face dataset includes both flux and material-property statistics
under `stats/`. If `<DATA_ROOT>/stats/<case>_{flux,material}_stats.yaml` are
missing after custom curation or relocation, regenerate them with:

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

Lattice:

```bash
python src/train.py case=lattice data=lattice \
    case.data_root=<DATA_ROOT> \
    case.split_file=./path/to/lattice_splits.json
```

Hohlraum:

```bash
python src/train.py case=hohlraum data=hohlraum \
    case.data_root=<DATA_ROOT> \
    case.split_file=./path/to/hohlraum_splits.json
```

Single-process default: 500 epochs, AMP-bf16, cosine LR with 10 warmup epochs,
peak LR 3e-5, physics loss enabled at weight 0.005 (lattice) / 0.01 (hohlraum).

### 4.2 Multi-GPU

```bash
torchrun --nproc_per_node=N src/train.py \
    case=lattice data=lattice case.data_root=<DATA_ROOT>
```

Use `torchrun` for DDP. A plain `python src/train.py ...` launch runs as a
single process.

### 4.3 Common overrides

| Override | Effect |
|---|---|
| `train.epochs=200` | Shorter run |
| `train.optimizer.type=muon` | Use `torch.optim.Muon` for 2-D weights, Adam for biases / norms |
| `train.amp=false` | Disable mixed precision (debug / numerical parity) |
| `train.physics_loss.weight=0.0` | Pure MSE training (disables QoI penalty) |
| `train.max_grad_norm=1.0` | Tighter gradient L2-norm clip (default `10.0`) |
| `train.dataloader.num_streams=4` | CUDA streams used by `physicsnemo.datapipes.DataLoader` for prefetch overlap (no CPU fork workers) |
| `train.dataloader.use_streams=false` | Disable CUDA-stream prefetching — useful for debugging or CPU-only runs |
| `train.dataloader.prefetch_factor=4` | How many batches to prefetch ahead |
| `model.num_spatial_points=8192` | Subsample cells per training step (`-1` = use all) |
| `model.n_layers=12 model.n_hidden=384` | Bigger Transolver |
| `model.use_te=true` | Use NVIDIA TransformerEngine layers (requires `[model-extras]`) |
| `train.resume_checkpoint=.../checkpoints/best_model` | Resume from a checkpoint directory |

### 4.4 Output structure

Per run, under `outputs/${project.name}/${case.type}/${exp_tag}/`:

```text
outputs/RTE_Transolver/lattice/transolver/
├── hydra/
│   ├── config.yaml          # resolved Hydra config (canonical record of the run)
│   ├── hydra.yaml
│   └── overrides.yaml
├── checkpoints/
│   └── best_model/                 # the lowest-val_loss snapshot to date
│       ├── checkpoint.0.0.pt       # training state (optimizer, scheduler, scaler, metadata)
│       └── Transolver.0.0.mdlus    # model state dict
├── tensorboard/             # TB event files (open with `tensorboard --logdir tensorboard/`)
└── train.log
```

Inference defaults to `checkpoints/best_model/` — the single
best-by-val_loss checkpoint maintained during training. No periodic,
rolling, or per-epoch snapshots are kept.

---

## 5. Evaluation

### 5.1 Run inference

Inference is Hydra-driven; supply the checkpoint path, data root, and split
file as standard Hydra overrides:

```bash
RUN=outputs/RTE_Transolver/lattice/transolver
python src/inference.py \
    case=lattice data=lattice \
    case.data_root=/path/to/data_root \
    case.split_file=/path/to/splits.json \
    inference.checkpoint_path=$RUN/checkpoints/best_model \
    inference.output_dir=$RUN/evaluation
```

The flux normalization stats file is read from
`cfg.data.flux_normalization_stats_file` (interpolated from `case.data_root`
by default); override it directly via
`data.flux_normalization_stats_file=<PATH>` if you keep stats elsewhere.

Inference-specific config keys (under `inference.*`):

| Key | Effect |
|---|---|
| `inference.checkpoint_path` | Required. Directory containing `Transolver.0.0.mdlus` + `checkpoint.0.0.pt`. Point at the `best_model/` directory under the run's `checkpoints/`. |
| `inference.output_dir` | Required. Where to write `metrics.yaml`, `qoi_metrics.yaml`, and `figures/`. |
| `inference.num_samples` | Cap on the number of test simulations (default: `null` = all). |
| `inference.num_plot_samples` | Number of `flux_panels_<idx>.png` figures to write (default: 3, evenly sampled across the test set). |
| `inference.device` | Override torch device (default: `null` = CUDA if available). |
| `inference.use_amp` | Autocast in eval; bf16 on CUDA, off on CPU (default: `true`). |

The case (`lattice` / `hohlraum`) is selected the same way as in training:
`case=<name> data=<name>`. The dataset root, split file, and material/flux
stats paths interpolate from `case.data_root` exactly as during training.

### 5.2 Outputs

```text
<output_dir>/
├── metrics.yaml             # field-level metrics over the whole test set
├── qoi_metrics.yaml         # per-region QoI relative error
└── figures/
    ├── flux_panels_0000.png # target / prediction / error 3-panel per plotted sample
    ├── ...
    └── qoi_true_vs_pred.png # predicted vs ground-truth QoI scatter (one panel per region)
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
reports `cur_absorption_{center, vertical, horizontal}` when geometry
metadata is available on the sample. The training-time physics loss
additionally includes a synthesized `total` loss on the sum of those three
component absorptions; inference does not report this total term.

### 5.4 Comparing runs

The single most useful comparison is
**`qoi_metrics.yaml::<region>::mean_relative_error_pct`**. On the default
randomized splits, a well-trained surrogate should reach low single-digit
percent QoI error.

For field-level comparisons, use `metrics.yaml::overall::l2_relative_error`,
which helps interpret global flux structure and sharp interface features.

---

## 6. Interpreting model performance

### 6.1 What "good" looks like

A converged model on either benchmark typically reaches `l2_relative_error`
in the **1–2%** range and per-region QoI `mean_relative_error_pct` **below
1%**.

### 6.2 Reading the training log

Each epoch logs train/validation loss and any per-component sub-losses
present (`mse`, `qoi`, `qoi_<region>`, ...) followed by the current
learning rate. A typical line looks like:

```text
Epoch 500: train_loss=1.7081e-05, val_loss=2.0973e-05,
    train_mse=1.7032e-05, val_mse=2.0900e-05,
    train_qoi=9.8040e-06, val_qoi=1.4658e-05, lr=1.00e-06
```

A `best_model/` checkpoint is written whenever `val_loss` improves; no
periodic per-epoch snapshots are kept.

### 6.3 Reading the inference figures

- **`flux_panels_<idx>.png`** — three panels per sample: target,
  prediction, absolute error.
- **`qoi_true_vs_pred.png`** — predicted vs ground-truth QoI scatter, one
  panel per region. Points should lie close to the `y = x` diagonal
  across the full test set.

---

## 7. Configuration reference

All training hyperparameters live under `src/conf/`, composed by Hydra:

```text
src/conf/
├── config.yaml             # root: composes case / data / model / train / inference
├── case/{lattice,hohlraum}.yaml
├── data/{lattice,hohlraum}.yaml
├── model/transolver.yaml
├── train/base.yaml
└── inference/default.yaml
```

`config.yaml` defaults list:

```yaml
defaults:
  - case: lattice
  - data: lattice
  - model: transolver
  - train: base
  - inference: default
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
`case/hohlraum.yaml` (including `physics_loss_weight`,
`include_q_in_embedding`, and `embedding_dim_override`). The downstream
`train/base.yaml` and `model/transolver.yaml` interpolate from `${case.*}`
so case-specific overrides propagate automatically.

---

## References

[^1]: Kusch, J., Schotthöfer, S., Stammer, P., Wolters, J., & Xiao, T. (2023).
"KiT-RT: An extendable framework for radiative transfer and therapy."
*ACM Transactions on Mathematical Software*, **49**(4), 1–24.

[^2]: Schotthoefer, S., & Hauck, C. (2025).
"Reference solutions for linear radiation transport: the Hohlraum and Lattice benchmarks."
*arXiv preprint arXiv:2505.17284*.

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

@misc{schotthoefer2025reference,
  title         = {Reference solutions for linear radiation transport:
                   the Hohlraum and Lattice benchmarks},
  author        = {Schotthoefer, Steffen and Hauck, Cory},
  year          = {2025},
  eprint        = {2505.17284},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  url           = {https://arxiv.org/abs/2505.17284}
}
```
