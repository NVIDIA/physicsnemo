<!-- markdownlint-disable -->
# Machine Learning Surrogates for Automotive Crash Dynamics 🧱💥🚗

## Problem Overview

Automotive crashworthiness assessment is a critical step in vehicle design. Traditionally, engineers rely on high-fidelity finite element (FE) simulations (e.g., LS-DYNA) to predict structural deformation and crash responses. While accurate, these simulations are computationally expensive and limit the speed of design iterations.

Machine Learning (ML) surrogates provide a promising alternative by learning mappings directly from simulation data, enabling rapid prediction of deformation histories across thousands of design candidates.

In this recipe, we demonstrate a unified pipeline for crash dynamics modeling. The implementation supports GeoTransolver, Transolver, MeshGraphNet, FIGConvUNet, MeshTransolver, MeshGeoTransolver, and MeshGeoFLARE architectures with multiple rollout schemes. It supports VTP and Zarr formats (preprocessed from LS-DYNA d3plot via PhysicsNeMo-Curator). The design is highly modular, enabling users to write their own readers, bring their own architectures, or implement custom rollout/transient schemes. Multiple experiments (different datasets, models, or feature sets) are managed via Hydra experiment configs without touching the core code.

For an in-depth comparison between the Transolver and MeshGraphNet models and the transient schemes for crash dynamics, see [this paper](https://arxiv.org/pdf/2510.15201). DeFormer adds structural mesh message passing around a geometry-aware FLARE backbone, with optional predictive node-to-face contact for closed-loop autoregressive dynamics.

### DeFormer with geodesic-filtered surface contact

The reference contact configuration is
`crash_deformer_contact_autoregressive`.
It combines the corrected latent-space mesh processor with original point-space
FLARE (`GALE_FA`), predictive node-to-face messages, reference-geodesic exclusions,
and fixed four-step truncated BPTT. Its default budget is 500 epochs without
teacher forcing or early stopping; validation rolls out all 24 predicted steps
after two observed frames. Node inputs are velocity and thickness, with position
used by the geometry and contact pathways. The paired baselines are
`crash_deformer_autoregressive` (no contact) and `crash_geoflare_autoregressive`
(no structural or contact message passing).

See [reference-geodesic exclusions](SURFACE_CONTACT_REFERENCE_GEODESIC.md).
The 5 mm geodesic gap floor is an explicit modeling assumption, separate from
the message-activation band, not a recovered solver-deck setting. Dataset paths
and splits must be supplied by the user; no proprietary datasets or trained
checkpoints are included.

The code retains the historical names `MeshGeoFLARE` (reusable model) and
`MeshGeoFLAREAutoregressive` (crash rollout wrapper) for checkpoint and Hydra
compatibility. These implement the two **DeFormer** experiments above; the
GeoFLARE baseline uses `GeoTransolverAutoregressive`. All three configs explicitly
select original FLARE (`GALE_FA`), not FLARE++.
Other hybrid classes, nearest-node/legacy contact backends, and analytic bumper
obstacle options are compatibility interfaces, not alternative reference recipes.
They are not used by the surface-contact experiment.

### Experimental: DeFormer with static kNN neighborhood attention

`crash_deformer_knn_autoregressive` replaces DeFormer's two MeshGraphNet
processors with attention over a **static reference neighborhood**: each node
attends to itself plus its `k` nearest neighbors selected once on the initial
mesh, with reference displacement/distance entering the attention logits and
values. Everything else is the no-contact DeFormer recipe (FLARE backbone,
zero-initialized residual gates, FiLM conditioning, BPTT-4, Muon, 500 epochs),
so the comparison isolates the local operator. Contact is off in this entry
point; the contact block is unchanged and can be re-enabled later.

Two neighbor metrics share one switch, resolved by the datapipe and checked by
the model:

```bash
# Geodesic: shortest path along structural edges, weighted by reference lengths.
torchrun --nproc_per_node=<NUM_GPUS> train.py \
  --config-name=crash_deformer_knn_autoregressive \
  model.local_neighborhood=geodesic hydra.run.dir=./outputs/deformer_knn_geodesic \
  training.raw_data_dir=/data/crash/train \
  training.raw_data_dir_validation=/data/crash/validation

# Euclidean: straight-line reference distance; can connect stacked, unconnected shells.
torchrun --nproc_per_node=<NUM_GPUS> train.py \
  --config-name=crash_deformer_knn_autoregressive \
  model.local_neighborhood=euclidean hydra.run.dir=./outputs/deformer_knn_euclidean \
  training.raw_data_dir=/data/crash/train \
  training.raw_data_dir_validation=/data/crash/validation
```

Neighborhoods are computed from the frozen physical initial geometry (never a
sampled window or future frame), once per distinct initial mesh, and cached under
`datapipe.local_knn_cache_dir` by a content hash of geometry, topology, `k`,
metric and algorithm source. Geodesic selection runs a bounded Dijkstra per node
in Python (minutes for a 385k-node mesh); Euclidean selection uses a KD-tree.
Features are standardized with train-only statistics (`knn_edge_stats.json`),
like structural edges. Checkpoints from the kNN variant, the MPNN DeFormer and
the two metrics are not interchangeable. Peak memory per block scales with
`N x (k+1) x width`; `model.local_knn_k=16` halves it.

Suggested protocol: compare the full 24-step validation MSE at matched epochs
against `crash_deformer_autoregressive` trained under the same world size and
seed, and record seconds per update and peak memory. This is an experimental
option with no accuracy claim; it does not alter the validated reference recipes.

`crash_deformer_knn_v2_autoregressive` is the second design of that block
(un-normalized difference values, concatenate-then-MLP update with the
attention log-sum-exp, float32 geometric bias, split key/value, small matrices
on AdamW); see `conf/README.md` for the key table. It exists because the first
design tracked the MPNN DeFormer until epoch 300 and then plateaued near
1.0e-3 while the MPNN reached 6.7e-4.

### Experimental: time-conditional DeFormer

`crash_deformer_knn_time_conditional` predicts each future frame independently
from the initial mesh with the kNN-attention local processors before and after
FLARE, a Fourier time code (node inputs and a global token), a zero-initialized
provisional position head after post stage A and an independent post stage B
for the final correction (`time_conditional.py`). Its matched baseline is
`crash_geoflare_time_conditional`; see `conf/README.md`.
`crash_deformer_knn_time_conditional_adaln` adds time-conditioned LayerNorms
throughout the network (GenCast norm conditioning, adaLN-zero): function
preserving at initialisation, so it isolates the value of letting the time code
modulate every layer. `crash_deformer_knn_time_conditional_contact` discovers
surface contact on the provisional shape from stage A and feeds its gated
messages to stage B, giving the time-conditional model the contact mechanism of
the autoregressive DeFormer without a rollout.

### Experimental: full-resolution Yaris with kNN neighborhood attention

`crash_deformer_knn_yaris_autoregressive` runs the kNN-attention DeFormer on the
1.48M-node Yaris Zarr stores at full resolution, without mesh coarsening and
without contact. The `yaris_full` reader/datapipe pair reads one window per
access (no trajectory preloading), uses the dataset creator's `splits.csv` as
the split authority, and drops the nodes flagged in an outlier mask (rigid-wall
nodes and fly-away nodes) from inputs, targets and neighborhoods. All stores
share one reference mesh, so the Euclidean neighborhoods, their standardized
features and the frozen TRAIN statistics (per-axis position std, normalized
velocity/acceleration moments, thickness) are built once. Because the store
edge list covers shells only, `model.local_neighborhood=euclidean` is the
default; geodesic selection would isolate every solid and beam node.

Memory policy: FLARE blocks and the geometry context are activation
checkpointed (`model.activation_checkpointing=true`) inside each checkpointed
rollout step, so one block's activations are live at a time. Prepare the
statistics and caches on a CPU node, then train:

```bash
python yaris_full_prepare.py --config-name=crash_deformer_knn_yaris_autoregressive \
  yaris.data_root=/data/yaris_zarr yaris.split_manifest=/data/splits.csv \
  yaris.outlier_mask=/data/outlier_nodes.npz yaris.work_dir=/runs/yaris_knn

torchrun --nproc_per_node=<NUM_GPUS> train.py \
  --config-name=crash_deformer_knn_yaris_autoregressive \
  yaris.data_root=/data/yaris_zarr yaris.split_manifest=/data/splits.csv \
  yaris.outlier_mask=/data/outlier_nodes.npz yaris.work_dir=/runs/yaris_knn
```

The frozen statistics carry an identity (split manifest, outlier mask, kept
node count, stride, dt); a changed mask or split invalidates them. Validation
uses the generic normalized-MSE path over every remaining transition of each
held-out case.

### Body-in-White Crash Modeling

<p align="center">
  <img src="../../../docs/img/crash/crash.gif" alt="Crash animation" width="60%" />

</p>

### Crushcan Modeling

<p align="center">
  <img src="../../../docs/img/crash/crushcan.gif" alt="Crushcan animation" width="60%" />

</p>

### Bumper Beam modeling

<p align="center">
  <img src="../../../docs/img/crash/bumper_beam.gif" alt="Bumper beam animation" width="80%" />

</p>


## Prerequisites

**Data:** LS-DYNA crash data preprocessed to VTP or Zarr format using [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/cae/crashsim). See [Data Preprocessing](#data-preprocessing) below for setup instructions.

**Code dependencies:**

```bash
pip install -r requirements.txt
```

For graph-based models (e.g., MeshGraphNet) or the graph datapipe, install the PhysicsNeMo `gnns` extra:

```bash
pip install "nvidia-physicsnemo[gnns]"
# or with uv:
uv sync --extra gnns
```

Reference-geodesic preprocessing also requires SciPy, declared in the existing
`nn-extras` optional dependency group. Install that extra, or install SciPy
directly in an existing graph-model environment:

```bash
pip install scipy
```

SciPy is loaded only when reference-geodesic preprocessing is called; unrelated
models and contact geometry remain importable without it.

## Data Preprocessing

Using `PhysicsNeMo-Curator`, crash simulation data from LS-DYNA can be processed into training-ready formats easily.
PhysicsNeMo-Curator can preprocess d3plot files into **VTP** (for visualization and smaller datasets) or **Zarr** (for large-scale ML training).

Install PhysicsNeMo-Curator following
[these instructions](https://github.com/NVIDIA/physicsnemo-curator?tab=readme-ov-file#installation-and-usage).

Process your LS-DYNA data to **VTP format**:

```bash
export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl                                         \
    --config-dir=examples/structural_mechanics/crash/config     \
    --config-name=crash_etl                                     \
    serialization_format=vtp                                    \
    etl.source.input_dir=/data/crash_sims/                      \
    serialization_format.sink.output_dir=/data/crash_vtp/       \
    etl.processing.num_processes=4
```

Or process to **Zarr format** for large-scale training:

```bash
export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl                                         \
    --config-dir=examples/structural_mechanics/crash/config     \
    --config-name=crash_etl                                     \
    serialization_format=zarr                                   \
    etl.source.input_dir=/data/crash_sims/                      \
    serialization_format.sink.output_dir=/data/crash_zarr/      \
    etl.processing.num_processes=4
```

The Curator expects your LS-DYNA data organized as:

```
crash_sims/
├── Run100/
│   ├── d3plot          # Required: binary mesh/displacement data
│   └── run100.k        # Optional: part thickness definitions
├── Run101/
│   ├── d3plot
│   └── run101.k
└── ...
```

### Output Formats

#### VTP Format

Produces single VTP file per run with all timesteps as displacement fields:

```
crash_processed_vtp/
├── Run100.vtp
├── Run101.vtp
└── ...
```

Each VTP contains:
- Reference coordinates at t=0
- Displacement fields: `displacement_t0.000`, `displacement_t0.005`, etc.
- Node thickness and other point data features

This format is directly compatible with the VTP reader in this example.

#### Zarr Format

Produces one Zarr store per run with pre-computed graph structure:

```
crash_processed_zarr/
├── Run100.zarr/
│   ├── mesh_pos       # (timesteps, nodes, 3) - temporal positions
│   ├── thickness      # (nodes,) - node features
│   └── edges          # (num_edges, 2) - pre-computed graph connectivity
├── Run101.zarr/
└── ...
```

Each Zarr store contains:
- `mesh_pos`: Full temporal trajectory (no displacement reconstruction needed)
- `thickness`: Per-node features
- `edges`: Pre-computed edge connectivity (no edge rebuilding during training)

**NOTE:** All heavy preprocessing (node filtering, edge building, thickness computation) is done once during curation using PhysicsNeMo-Curator. The reader simply loads pre-computed arrays.

This format is directly compatible with the Zarr reader in this recipe.

## Training

Training is managed via Hydra configurations located in `conf/`.
The main script is `train.py`.

### Config Structure

The existing bumper and body-in-white one-shot/time-conditional experiments remain
available. This contribution adds three autoregressive entry points:

| Experiment | Model |
|------------|-------|
| `crash_geoflare_autoregressive` | GeoTransolver with original point-space FLARE |
| `crash_deformer_autoregressive` | DeFormer without contact |
| `crash_deformer_contact_autoregressive` | DeFormer with predictive geodesic-filtered surface contact |

The DeFormer experiment inherits the GeoFLARE data/training recipe and switches to
a graph datapipe and mesh model. The contact experiment inherits DeFormer and adds
contact, reproducible sampling, and memory settings. Two model definitions live in
`conf/model/`; reader, datapipe, training, and inference defaults are shared with
the existing examples. See [configuration layout](conf/README.md) for overrides.

### Launch Training

Single GPU:

```bash
python train.py --config-name=bumper_geotransolver_oneshot
```

Multi-GPU (Distributed Data Parallel):

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py --config-name=bumper_geotransolver_oneshot
```

### DeFormer autoregressive comparison

Run one of the three entry points with your dataset paths:

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py \
  --config-name=crash_deformer_contact_autoregressive \
  training.raw_data_dir=/data/crash/train \
  training.raw_data_dir_validation=/data/crash/validation
```

Replace the config name with `crash_deformer_autoregressive` or
`crash_geoflare_autoregressive` for the baselines. These are fresh model runs,
not interchangeable checkpoints. Use separate output directories when launching
comparisons concurrently, for example `hydra.run.dir=./outputs/deformer_contact`.

The reference defaults assume 127 training cases, 8 validation cases, and 26
frames spaced 5 ms apart. By default, all models use velocity and thickness inputs,
predict acceleration, and train with random four-transition closed-loop BPTT windows.
The two initial observed frames initialize velocity; validation and inference
roll out the remaining 24 transitions. The common budget is 500 epochs with
Muon, BF16, a cosine learning-rate schedule from `2e-4` to `1e-6`, and no
teacher forcing or early stopping. Supply sample counts and time-step settings
appropriate for a different dataset.

DeFormer can optionally include current XYZ alongside velocity and thickness:

```bash
python train.py --config-name=crash_deformer_contact_autoregressive \
  model.include_position_features=true \
  training.raw_data_dir=/data/crash/train \
  training.raw_data_dir_validation=/data/crash/validation
```

This switch defaults to `false` and also works with `crash_deformer_autoregressive`.
When enabled, the input order is **normalized current XYZ, normalized velocity XYZ,
normalized thickness**. The wrapper automatically expands the input projection
from four to seven channels; leave `model.functional_dim=4` and
`model.node_input_mode=velocity` unchanged. XYZ is the observed current position
at initialization and the model's own predicted position during subsequent
closed-loop steps, never a future target. Geometry context, physical-unit contact,
and the residual velocity/position integration are unchanged.

Use the same switch for training and inference, and a separate checkpoint/output
directory for each mode: four- and seven-channel checkpoints are not interchangeable.
Legacy explicit `node_input_mode=position_velocity` with `functional_dim=7`
remains supported with the new switch disabled; do not combine both mechanisms.

Penetration regularization is a separate, default-off training option:

```bash
python train.py --config-name=crash_deformer_contact_penetration_autoregressive \
  training.raw_data_dir=/data/crash/train \
  training.raw_data_dir_validation=/data/crash/validation
```

This inherits the reference contact architecture and training settings, adding
loss-only initial physical geometry and a soft shell-overlap penalty. For each
node, the loss penalizes its largest excess overlap with a nonexcluded face,
after subtracting that pair's initial overlap and a 0.25 mm tolerance. Squared
excess is normalized by a 1 mm length scale, averaged over all nodes and rollout
steps, and added to position MSE. The weight is zero for the first 50 epochs,
ramps linearly for 50 epochs, and then stays at `1e-5`. Geometry is evaluated in
physical units, with gradients through all predicted BPTT steps. Validation
remains position MSE. These are experimental settings, not a solver contact law.

Add `model.include_position_features=true` to combine both options. The ordinary
contact recipe has neither penetration metadata nor penetration-loss evaluation.
The penalty adds no network parameters and is not needed for inference. It is
an endpoint regularizer, so it cannot guarantee nonpenetration or detect a full
crossing between saved frames. Use fresh output directories when changing the
objective: resume guards reject mismatched penetration settings.

The contact recipe requires valid triangle/quad connectivity and nodal shell
thickness in physical units (mm for the reference data). It rebuilds live
node-to-face contact from the predicted geometry, uses barycentric surface
features rather than face centroids, and retains gradients through the selected
geometry and contact messages. Candidate membership is discrete. The geometry is
denormalized before contact computation; reference-geodesic exclusions use
initial physical geometry. This is solver-inspired learned messaging, not a
contact-force solver or a nonpenetration guarantee.

Optional face-area quadrature (default off) multiplies each active pair's
activation weight by the face's center-fan area over a reference area, the
activation-band disk `pi * 5^2` mm^2 unless overridden. It makes the contact
aggregate and its activation total approximately independent of mesh
resolution, which matters when training on meshes with different element sizes
or on coarsened meshes. Enable it with `+model.contact_surface_area_weighting=true`
(optionally `+model.contact_surface_reference_area=<mm^2>`); it changes the
learned function, so use a fresh output directory and compare against the
reference recipe.

The contact recipe additionally enables deterministic sampling, RNG-isolated
contact initialization, nested contact checkpointing, and activation offloading.
Those execution/reproducibility settings are explicit; they are not silently
assumed identical to the historical no-contact recipe.

Small ablations are overrides instead of separate YAML files:

```bash
# Incidence-only exclusions; keep the same model and training budget.
python train.py --config-name=crash_deformer_contact_autoregressive \
  datapipe.contact_surface_exclusion=incidence \
  training.raw_data_dir=/data/crash/train \
  training.raw_data_dir_validation=/data/crash/validation

# Print the recipe without starting training.
python train.py --config-name=crash_deformer_contact_autoregressive --cfg job
```

Use `model.checkpoint_offloading=false` for an execution-only memory tradeoff.
Use `model.enable_contact=false` to disable contact messages while retaining the
contact parameters and initialization. The dedicated no-contact DeFormer
experiment omits the contact module entirely. See
[reference-geodesic exclusions](SURFACE_CONTACT_REFERENCE_GEODESIC.md) for filter,
gap-floor, and cache options.

## Inference

Use `inference.py` to evaluate trained models on test crash runs. Outputs are written under `output_dir_pred/rank{N}/{run_name}/`.

**Note:** Inference currently supports only the VTP format.

Single GPU:

```bash
python inference.py --config-name=bumper_geotransolver_oneshot
```

Multi-GPU (Distributed Data Parallel):

```bash
torchrun --nproc_per_node=<NUM_GPUS> inference.py --config-name=bumper_geotransolver_oneshot
```

Runs are sharded across ranks: rank `r` processes `run_items[r::world_size]`.
Predicted meshes are written as .vtp files under `./predicted_vtps/`, and can be opened using ParaView.

## Guardrails (OOD detection)

An optional out-of-distribution (OOD) guardrail calibrates during training and
emits warnings at inference when inputs drift outside the training distribution.
It watches two surfaces:

- **Global parameters** — per-channel bounding box on the global embedding
  (e.g. `velocity_x`, `thickness_scale`).
- **Geometry** — k-nearest-neighbour distance on a pooled geometry latent.

To enable the guard, wrap a constructed `GeoTransolver` with
`GuardedGeoTransolver` (from `physicsnemo.experimental.guardrails.embedded`).
The wrapper observes the two surfaces above through a forward hook and delegates the
forward pass unchanged:

```python
from physicsnemo.experimental.guardrails.embedded import (
    GuardedGeoTransolver,
    OODGuardConfig,
)

model = instantiate(cfg.model)  # your GeoTransolver (or rollout subclass)
model = GuardedGeoTransolver(
    model,
    OODGuardConfig(
        buffer_size=121,  # FIFO buffer; typically = num_training_samples
        knn_k=10,  # k for geometry kNN distance
        sensitivity=1.5,  # threshold multiplier on 99th-percentile kNN dist
    ),
)
```

Call `model.train()` / `model.eval()` as usual; the wrapper collects during
training and checks during inference.

What each parameter controls:

- **`buffer_size`** — Capacity of the FIFO ring buffer that stores pooled
  geometry latents during training; the kNN threshold is computed over its
  contents. Set it to at least the training-set size so calibration sees every
  sample. Under DDP each rank keeps its own buffer and the distributed sampler
  shuffles, so after a few epochs each rank's FIFO covers most of the data.
  Memory cost is `buffer_size × head_dim × 4` bytes — typically well under 1 MB.
- **`knn_k`** — Number of nearest neighbours used in the kNN distance. Smaller
  `k` is more sensitive to isolated training-set outliers; larger `k` is
  smoother but can blur multi-modal cluster boundaries. The default of `10`
  works for buffer sizes from ~100 up to several thousand.
- **`sensitivity`** — Multiplier on the 99th-percentile training kNN distance
  used as the OOD threshold. **Higher = less sensitive** (fewer warnings).
  Raise it if known in-distribution validation data triggers warnings; lower
  it if known-OOD inputs are being missed.

Recommended starting points by training-set size:

| Training samples | `buffer_size`           | `knn_k` | `sensitivity` |
|------------------|-------------------------|---------|---------------|
| ~100             | 100–200                 | 5–10    | 1.5           |
| ~500             | 500–1000                | 10      | 1.5           |
| 5000+            | = dataset size          | 10–15   | 1.5           |

If validation data trips warnings, raise `sensitivity` toward 2.0–3.0; if
known-OOD inputs slip through, lower it toward 1.0.

Once the model is wrapped, no further changes are required: during training the
guard silently collects calibration statistics, and during inference it emits
warnings of the form `OOD Guard: geometry sample ...` or
`OOD Guard: global_embedding dim ...` to the Python logger whenever a sample
falls outside the calibrated training envelope.  Warnings do not halt inference.

## Experiments

Each YAML entry point in `conf/` selects shared component defaults and any experiment-specific overrides. The autoregressive comparison recipes inherit common settings rather than copying them.

### Anatomy of an experiment config

Data paths must be set either in the config file or via CLI overrides. For training: `raw_data_dir`, `raw_data_dir_validation`. For inference: `raw_data_dir_test`. Use `???` in the config to make them mandatory overrides, or set concrete paths directly.

```yaml
# conf/my_experiment.yaml

hydra:
  job:
    chdir: True
  run:
    dir: ./outputs/

experiment_name: "My-Experiment"

defaults:
  - reader: vtp
  - datapipe: point_cloud
  - model: geotransolver_one_shot
  - training: default
  - inference: default
  - _self_

# ┌───────────────────────────────────────────┐
# │                   Data                    │
# └───────────────────────────────────────────┘

training:
  raw_data_dir: ???              # set in config or via CLI
  raw_data_dir_validation: ???   # set in config or via CLI
  global_features_filepath: ???  # or null if not using global features
  num_time_steps: 51
  num_training_samples: 121
  num_validation_samples: 5

inference:
  raw_data_dir_test: ???         # set in config or via CLI

# ┌───────────────────────────────────────────┐
# │            Datapipe features              │
# └───────────────────────────────────────────┘

datapipe:
  static_features: []  # per-node static features (e.g., thickness)
  dynamic_targets:    # per-node time-series targets (e.g., strain, stress)
    - effective_plastic_strain
    - stress_vm
  global_features:   # per-run scalar features (loaded from JSON)
    - velocity_x
    - thickness_scale
    - rwall_origin_y
  sample_type: all_time_steps  # or one_time_step for time-conditional models
```

### Provided experiments

| File | Dataset | Model | Launch command |
|------|---------|-------|----------------|
| `bumper_geotransolver_oneshot.yaml` | Bumper beam (VTP) | GeoTransolver one-shot | `python train.py --config-name=bumper_geotransolver_oneshot` |
| `bumper_geotransolver_time_conditional.yaml` | Bumper beam (VTP) | GeoTransolver time-conditional | `python train.py --config-name=bumper_geotransolver_time_conditional` |
| `crash_geotransolver_oneshot.yaml` | Car body-in-white crash (VTP) | GeoTransolver one-shot | `python train.py --config-name=crash_geotransolver_oneshot` |
| `bumper_geoflare_oneshot.yaml` | Bumper beam (VTP) | GeoFLARE one-shot | `python train.py --config-name=bumper_geoflare_oneshot` |
| `crash_geoflare_oneshot.yaml` | Car body-in-white crash (VTP) | GeoFLARE one-shot | `python train.py --config-name=crash_geoflare_oneshot` |
| `crash_geoflare_autoregressive.yaml` | Full-car crash (VTP) | GeoFLARE BPTT-4 baseline | `python train.py --config-name=crash_geoflare_autoregressive` |
| `crash_deformer_autoregressive.yaml` | Full-car crash (VTP) | DeFormer BPTT-4, no contact | `python train.py --config-name=crash_deformer_autoregressive` |
| `crash_deformer_contact_autoregressive.yaml` | Full-car crash (VTP) | DeFormer BPTT-4, geodesic surface contact | `python train.py --config-name=crash_deformer_contact_autoregressive` |
| `crash_deformer_knn_autoregressive.yaml` | Full-car crash (VTP) | Experimental DeFormer BPTT-4, static kNN neighborhood attention instead of MPNNs, no contact | `python train.py --config-name=crash_deformer_knn_autoregressive` |

### Choosing a time scheme

Three rollout schemes have premade experiment configurations:

| Scheme | Model | Datapipe `sample_type` | Behavior |
|--------|-------|------------------------|----------|
| **One-shot** | `geotransolver_one_shot` | `all_time_steps` | One sample per run. Model predicts the full trajectory `[N, T-1, Fo]` from t0 in a single forward pass. Lower training cost, competitive accuracy. |
| **Time-conditional** | `geotransolver_time_conditional` | `one_time_step` | One sample per run per timestep. Model predicts a single step `[N, Fo]` conditioned on normalized time `t/(T-1)`. Best accuracy for long horizons; higher training cost. Inference always rolls out the full trajectory. |
| **Autoregressive** | `geoflare_autoregressive` / `deformer_autoregressive` | `random_time_window` | Closed-loop BPTT during training; full-trajectory validation and inference. The contact experiment additionally rebuilds contact from predicted geometry. |

Use **one-shot** when you need fast iteration or have limited compute. Use
**time-conditional** when validation quality matters most. Use **autoregressive
contact** when deployment requires closed-loop dynamics and geometry-dependent
interactions. Legacy `geotransolver_autoregressive_rollout_training` and
`geotransolver_one_step_rollout` components also remain available for custom
experiments. See [Development tips](#development-tips) for the legacy comparison.

### Adding a new experiment

1. Create `conf/<my_experiment>.yaml` following the template above.
2. Set defaults for reader, datapipe, model, training, and inference in the `defaults` section.
3. Set all required fields: `raw_data_dir`, `raw_data_dir_validation` (training), `raw_data_dir_test` (inference), `num_time_steps`, `num_training_samples`. Either set concrete paths in the config or use `???` and pass them via CLI when launching `train.py` or `inference.py` as appropriate.
4. Set `datapipe.sample_type` to match your model: `all_time_steps` for one-shot or autoregressive rollout, `one_time_step` for time-conditional.
5. If using global features, set `global_features_filepath`; otherwise use `null`.
6. Optionally override any model or training hyperparameter directly in the experiment file (e.g., `model.out_dim: 150`, `training.epochs: 5000`), or add a new model config under `conf/model/` and select it in the defaults.
7. Run: `python train.py --config-name=<my_experiment>`

You can also override the model in the `defaults` section. For time-conditional training, use the dedicated experiment config:
```yaml
# Use bumper_geotransolver_time_conditional.yaml, or in your experiment:
defaults:
  - reader: vtp
  - model: geotransolver_time_conditional  # Override model
  - training: default
  - _self_
```

## Datapipe: how inputs are constructed and normalized

The datapipe converts reader output into model-ready `SimSample`s. Each sample is one crash run (or one timestep of a run, depending on `sample_type`).

Inputs `x` contain `x['coords']` (`[N, 3]` at t0) and `x['features']` (`[N, F]` static plus flattened dynamic features in config order). Targets `y` have shape `[N, T-1, Fo]` where Fo=3 for positions only, or 3+sum(C_k) if `dynamic_targets` is set (e.g. strain, stress).

Config lives under `conf/datapipe/`. Two variants exist: `graph` for MeshGraphNet and `point_cloud` for point cloud-based models (e.g., GeoTransolver and Transolver). Both use `static_features` (e.g. `[thickness]`), `dynamic_features`, `dynamic_targets`, and `global_features`. Use `static_features: []` for no node features. Position and feature stats are computed on the train split, saved to `./stats/` (`node_stats.json`, `feature_stats.json`, `edge_stats.json` for graph), and reused at eval and inference.

### Sample type: all_time_steps vs one_time_step

The datapipe `sample_type` (set in each experiment's `datapipe` block) controls how samples are constructed to match the model's rollout scheme:

- **`all_time_steps`** (one-shot): One sample per run. `x` has `coords` and `features`; `y` is the full trajectory `[N, T-1, Fo]`. Dataset length = `num_samples`.

- **`one_time_step`** (time-conditional): One sample per run per timestep. Each sample corresponds to `(batch_idx, time_idx)`; `x` adds `x['time']` = `time_idx/(T-1)` (normalized); `y` is a single step `[N, Fo]`. Dataset length = `num_samples * (T-1)`. Inference always uses `all_time_steps`, so it receives full trajectories.

Readers return `(srcs, dsts, point_data)`; see the [Reader](#reader-built-in-vtp-and-zarr-readers-and-how-to-add-your-own) section. The graph datapipe builds PyG `Data` with edge features from t0 positions; the point-cloud datapipe ignores connectivity. Models consume `x['coords']` and `x['features']` directly. The `_feature_slices` map associates feature names with column ranges for diagnostics. With `static_features: []`, the features tensor has width zero and the pipeline handles it correctly.

### Global features

Global features are per-run scalar values (e.g., impact velocity, thickness scale factor, wall position) that do not vary across mesh nodes. They are passed to the model as a single global conditioning vector and are distinct from the per-node `features` described above.

#### JSON file format

Global features are stored in a single JSON file shared across all splits (train, validation, test). The file is a flat dictionary keyed by **run ID**, where each value is a dictionary of `{feature_name: scalar_float}`:

```json
{
  "Run100": {
    "velocity_x": -5.0,
    "thickness_scale": 1.0,
    "rwall_origin_y": 0.0
  },
  "Run101": {
    "velocity_x": -5.0,
    "thickness_scale": 0.7,
    "rwall_origin_y": 120.0
  },
  ...
}
```

**Run ID convention:** the run ID must match the **filename stem** of the corresponding data file. For the VTP reader, a file named `Run100.vtp` maps to run ID `"Run100"`. For the Zarr reader, a store named `Run100.zarr/` maps to run ID `"Run100"`. All values must be Python-serializable floats (or ints, which are cast to float).

#### Configuration

Point to the JSON file and declare which keys to use in your experiment config. Set `global_features_filepath` in the config file or via CLI (`training.global_features_filepath=/path/to/global_features.json`):

```yaml
# conf/my_experiment.yaml
training:
  global_features_filepath: ???  # or a concrete path

datapipe:
  global_features:       # subset of keys to extract; order defines the global vector
    - velocity_x
    - thickness_scale
    - rwall_origin_y
```

Every run in the dataset must have **all listed keys** present in the JSON; a missing key raises a `KeyError` at dataset construction time. Keys present in the JSON but not listed in `global_features` are silently ignored, so you can store extra metadata in the file without affecting training.

To disable global features entirely, omit `global_features_filepath` (or leave it `null`) and set `global_features: null` in the datapipe block.

#### How the datapipe and model consume global features

Training computes mean and standard deviation from the training split and saves them
with the other dataset statistics. Validation, test, and inference reuse those values
without recomputing them. At `__getitem__` time, the datapipe standardizes each
selected scalar and attaches it to the `SimSample`:

```python
sample.global_features = {
    "velocity_x": tensor(z_velocity_x),
    "thickness_scale": tensor(z_thickness_scale),
    "rwall_origin_y": tensor(z_rwall_origin_y),
}
```

In the model forward pass, these are stacked into a single global embedding vector and passed to the network. The **`global_dim`** parameter in the model config must equal the number of global features selected:

```yaml
# conf/model/geotransolver_one_shot.yaml (or in experiment model overrides)
global_dim: 3   # must match len(datapipe.global_features)
```

If `global_features` is `null`, `sample.global_features` is `None` and the selected
model must support unconditioned execution. GeoTransolver and the geometry-aware
mesh hybrids consume the global token directly; the mesh-attention crash wrappers
also broadcast configured global values into their node inputs.

## Reader: built-in VTP and Zarr readers and how to add your own

The reader opens preprocessed simulation data and produces the arrays the datapipe consumes. Raw LS-DYNA d3plot files must be preprocessed to VTP or Zarr using [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/cae/crashsim) before use. The reader is swappable via Hydra so you can adapt the pipeline to different formats or add your own.

### Built‑in VTP reader (PolyData)

A lightweight VTP reader is provided in `vtp_reader.py`. It treats each `.vtp` file in a directory as a separate run and expects point displacements to be stored as vector arrays in `poly.point_data` with names like `displacement_t0.000`, `displacement_t0.005`, … (a more permissive fallback of any `displacement_t*` is also supported). The reader:

- loads the reference coordinates from `poly.points`
- builds absolute positions per timestep as `[t0: coords, t>0: coords + displacement_t]`
- extracts cell connectivity from the PolyData faces and converts it to unique edges
- extracts all point data fields dynamically (e.g., thickness, modulus)
- returns `(srcs, dsts, point_data)` where `point_data` contains `'coords': [T, N, 3]` and all feature arrays

The VTP reader dynamically extracts all non-displacement point data fields from the VTP file and makes them available to the datapipe. If your `.vtp` files include additional per‑point arrays (e.g., thickness or modulus), simply add their names to the `features` list in your datapipe config.

Example Hydra configuration for the VTP reader:

```yaml
# conf/reader/vtp.yaml
_target_: vtp_reader.Reader
```

Select it in your experiment config defaults:

```yaml
# conf/my_experiment.yaml
defaults:
  - reader: vtp
  - datapipe: point_cloud
  - model: geotransolver_one_shot
  - training: default
  - inference: default
  - _self_
```

And configure features in the experiment's `datapipe` block:

```yaml
# conf/my_experiment.yaml
datapipe:
  static_features: [thickness]  # or [] for no features
```

### Built‑in Zarr reader

A Zarr reader provided in `zarr_reader.py`. It reads pre-processed Zarr stores created by PhysicsNeMo-Curator, where all heavy computation (node filtering, edge building, thickness computation) has already been done during the ETL pipeline. The reader:

- loads pre-computed temporal positions directly from `mesh_pos` (no displacement reconstruction)
- loads pre-computed edges (no connectivity-to-edge conversion needed)
- dynamically extracts all point data fields (thickness, etc.) from the Zarr store
- returns `(srcs, dsts, point_data)` similar to VTP reader

Data layout expected by Zarr reader:
- `<DATA_DIR>/*.zarr/` (each `.zarr` directory is treated as one run)
- Each Zarr store must contain:
  - `mesh_pos`: `[T, N, 3]` temporal positions
  - `edges`: `[E, 2]` pre-computed edge connectivity
  - Feature arrays (e.g., `thickness`): `[N]` or `[N, K]` per-node features

Example Hydra configuration for the Zarr reader:

```yaml
# conf/reader/zarr.yaml
_target_: zarr_reader.Reader
```

Select it in your experiment config defaults:

```yaml
# conf/my_experiment.yaml
defaults:
  - reader: zarr
  - datapipe: point_cloud
  - model: geotransolver_one_shot
  - training: default
  - inference: default
  - _self_
```

And configure features in the experiment's `datapipe` block:

```yaml
# conf/my_experiment.yaml
datapipe:
  static_features: [thickness]  # Must match fields stored in Zarr
```

### Data layout expected by readers

- VTP reader (`vtp_reader.py`):
  - `<DATA_DIR>/*.vtp` (each `.vtp` is treated as one run)
  - Displacements stored as 3‑component arrays in point_data with names like `displacement_t0.000`, `displacement_t0.005`, ... (fallback accepts any `displacement_t*`).

- Zarr reader (`zarr_reader.py`):
  - `<DATA_DIR>/*.zarr/` (each `.zarr` directory is treated as one run)
  - Contains pre-computed `mesh_pos`, `edges`, and feature arrays

### Write your own reader

To write your own reader, implement a Hydra‑instantiable function or class whose call returns a three‑tuple `(srcs, dsts, point_data)`. The first two entries are lists of integer arrays describing edges per run (they can be empty lists if you are not producing a graph), and `point_data` is a list of Python dicts with one dict per run. Each dict must contain `'coords'` as a `[T, N, 3]` array and one array per feature name listed in `conf/datapipe/*.yaml` under `features`. Feature arrays can be `[N]` or `[N, K]` and should use the same node indexing as `'coords'`. For convenience, a simple class reader can accept the Hydra `split` argument (e.g., "train" or "test") and decide whether to save VTP frames, but this is optional.

As a starting point, your YAML can point to a class by dotted path:

```yaml
# conf/reader/my_reader.yaml
_target_: my_reader.MyReader
# any constructor kwargs here, e.g. thresholds or unit conversions
```

Then, in your experiment config, select the reader by adding `- reader: my_reader` to the `defaults` block. The datapipe will call your reader with `data_dir`, `num_samples`, `split`, and an optional `logger`, and will expect the tuple described above. Provided you populate `'coords'` and the configured feature arrays per run, the rest of the pipeline—normalization, batching, graph construction, and model rollout—will work without code changes.

A note on reader signatures and future‑proofing: the datapipe currently passes `data_dir`, `num_samples`, `split`, and `logger` when invoking the reader, and may pass additional keys in the future. To stay resilient, implement your reader with optional parameters and a catch‑all `**kwargs`.

For a class reader, use this signature in `__call__`:

```python
class MyReader:
    def __init__(self, some_option: float = 1.0):
        self.some_option = some_option

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: str | None = None,
        logger=None,
        **kwargs,
    ): ...
```

With this pattern, your reader will keep working even if the framework adds new optional arguments later.

## Postprocessing and Evaluation

The postprocessing/ folder provides scripts for quantitative and qualitative evaluation:

- Relative $L^2$ Error (compute_l2_error.py): Computes
per-timestep relative position error across runs.
Produces plots and optional CSVs.

Example:

```bash
python postprocessing/compute_l2_error.py \
    --predicted_parent ./predicted_vtps \
    --exact_parent ./exact_vtps \
    --output_plot rel_error.png \
    --output_csv rel_error.csv
```

- Probe Kinematics (Driver vs Passenger Toe Pan)(compute_probe_kinematics.py):
Extracts displacement/velocity/acceleration histories at selected probe nodes.
Generates comparison plots (GT vs predicted).

Example:

```bash
python postprocessing/compute_probe_kinematics.py \
    --pred_dir ./predicted_vtps/run_001 \
    --exact_dir ./exact_vtps/run_001 \
    --driver_points "70658-70659,70664" \
    --passenger_points "70676-70679" \
    --dt 0.005 \
    --output_plot probe_kinematics.png
```

- Cross-Sectional Plots (plot_cross_section.py): Plots 2D slices
of predicted vs ground truth deformations at specified cross-sections.

Example:

```bash
python postprocessing/plot_cross_section.py \
    --pred_dir ./predicted_vtps/run_001 \
    --exact_dir ./exact_vtps/run_001 \
    --output_file cross_section.png
```

run_post_processing.sh can automate all evaluation tasks across runs.

## Development tips

### Dynamics prediction

1. **Time-conditional** gives the best accuracy for long-horizon dynamics; prefer it when validation quality matters most.
2. **One-shot** offers competitive accuracy with much lower training cost; consider it when you need fast iteration or have few state variables to predict.
3. **AR-rollout** can work well for short-horizon prediction but tends to become unstable when training for longer rollouts.
4. **Teacher-forcing** yields low training loss but typically generalizes poorly at inference; avoid it for deployment.


| Bumper(T=50)      | t/epoch (sec) | Validation MSE | Car crash (T=14)  | t/epoch (sec) | Validation MSE |
|:------------------|:--------:|:--------:|:------------------|:--------:|:--------:|
| One-shot          | 1        | 5.42e-3  | One-shot          | 6.4      | 2.32e-4  |
| Time-conditional  | 29       | 4.12e-3  | Time-conditional  | 40.8     | 2.54e-4  |
| AR-rollout        | 37       | unstable | AR-rollout        | 61.6     | 2.27e-4  |
| Teacher-forcing   | 29       | 0.3      |                   |          |          |

<p align="center">
  <img src="../../../docs/img/crash/Time_integraton_val_loss.png" alt="Time integration validation loss" width="60%" />

</p>

### Models

5. **Accuracy ranking (one-shot):** GeoFlare > GeoTransolver > Transolver > MeshGraphNet. Use GeoFlare when best accuracy is the priority.
6. **Muon** generally outperforms Adam on validation MSE but can overfit; monitor validation loss and consider early stopping or regularization.

**One-shot comparison:**

| Test Relative L^2 | Bumper(Adam)   | Bumper(Muon) | Car crash(Adam)   | Car crash(Muon)|
|:------------------|:--------:|:--------:|:--------:|:--------:|
| Transolver        | 9.37e-3  | 9.12e-3  | 1.60e-2  | 1.60e-2  |
| GeoTransolver     | 9.40e-3  | 7.32e-3  | 1.40e-2  | 1.33e-2  |
| GeoFlare          | **8.73e-3**  | **6.80e-3**  | **1.16e-2**  | **8.95e-3**  |

Adam: Car-crash test MSE at probe location (Driver, Passenger):

| Driver            | position   | velocity | acceleration | Passenger   | position   | velocity | acceleration | 
|:------------------|:--------:|:--------:|:--------:|:------------------|:--------:|:--------:|:--------:|
| Transolver        | 2.21e-3  | 8.21e-1  | 5.60e+3  | Transolver        | 2.43e-3  | 9.31e-1  | 6.81e+3  |
| GeoTransolver     | 1.51e-3  | 5.74e-1  | 3.99e+3  | GeoTransolver     | 1.92e-3  | 7.03e-1  | 5.53e+3  |
| GeoFlare          | **1.01e-3**  | **4.38e-1**  | **2.99e+3**  | GeoFlare          | **1.19e-3**  | **5.16e-1**  | **3.93e+3**  |


Muon: Car-crash test MSE at probe location (Driver, Passenger):

| Driver            | position   | velocity | acceleration | Passenger   | position   | velocity | acceleration | 
|:------------------|:--------:|:--------:|:--------:|:------------------|:--------:|:--------:|:--------:|
| Transolver        | 2.63e-3  | 8.41e-1  | 2.14e+3  | Transolver        | 2.21e-3  | 7.25e-1  | 2.24e+3  |
| GeoTransolver     | 1.84e-3  | 6.09e-1  | 1.71e+3  | GeoTransolver     | 1.72e-3  | 5.53e-1  | 1.80e+3  |
| GeoFlare          | **7.18e-4**  | **2.71e-1**  | **1.27e+3**  | GeoFlare          | **6.52e-4**  | **2.53e-1**  | **1.45e+3**  |


<p align="center">
  <img src="../../../docs/img/crash/Test_MSE_models.png" alt="Time integration validation loss" width="60%" />

</p>

## TODO

- [x] **Normalize global features**: Training-only mean/std statistics are saved and reused for validation, test, inference, and physical-value reconstruction during autoregressive rollout.
- [ ] **Normalize dynamic targets**: Dynamic targets (e.g., effective_plastic_strain, stress_vm) are currently passed in the target `y` without normalization, while positions are normalized. Add per-target mean/std and denormalize at inference when exporting to VTP.
- [ ] **Support batch_size > 1**: The pipeline currently uses `batch_size=1` due to variable node counts per sample. Add padding or batching logic to enable larger batch sizes for improved throughput.

## Troubleshooting / FAQ

- My `.vtp` has no displacement fields.
  - Ensure point_data contains vector arrays named like `displacement_t0.000`, `displacement_t0.005`, ...; the reader falls back to any `displacement_t*` pattern.

- I want no node features.
  - Set `features: []`. The datapipe will return `x['features']` with shape `[N, 0]`, and the rollout will still concatenate velocity (and time if configured) for the model input.

- Can functional_dim be 0 for Transolver?
  - It can be 0 only if the total MLP input dimension remains > 0: e.g., you provide an embedding (required for unstructured) and/or time. In this pipeline, rollout always supplies an embedding (positions), so you are safe with `features: []`.

- My custom reader doesn’t accept `split` or `logger`.
  - Implement `__call__(..., split: str | None = None, logger=None, **kwargs)` to remain forward‑compatible with optional arguments.

## References

- [Automotive Crash Dynamics Modeling Accelerated with Machine Learning](https://arxiv.org/pdf/2510.15201)
- [Crash Assessment via Mesh-Based Graph Neural Networks and Physics-Aware Attention](https://arxiv.org/pdf/2605.11784)
- [GeoTransolver: Learning Physics on Irregular Domains Using Multi-scale Geometry Aware Physics Attention Transformer](https://arxiv.org/pdf/2512.20399)
- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)
- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/pdf/2010.03409)
