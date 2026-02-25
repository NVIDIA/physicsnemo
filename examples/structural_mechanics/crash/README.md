<!-- markdownlint-disable -->
# Machine Learning Surrogates for Automotive Crash Dynamics 🧱💥🚗

## Problem Overview

Automotive crashworthiness assessment is a critical step in vehicle design.   Traditionally, engineers rely on high-fidelity finite element (FE) simulations (e.g., LS-DYNA) to predict structural deformation and crash responses. While accurate, these simulations are computationally expensive and limit the speed of design iterations.

Machine Learning (ML) surrogates provide a promising alternative by learning mappings directly from simulation data, enabling:

- **Rapid prediction** of deformation histories across thousands of design candidates.
- **Scalability** to large structural models without rerunning costly FE simulations.
- **Flexibility** in experimenting with different model architectures (GNNs, Transformers).

In this example, we demonstrate a unified pipeline for crash dynamics modeling. The implementation supports GeoTransolver, Transolver, FIGConvUNet, and MeshGraphNet architectures with multiple rollout schemes. It supports VTP and Zarr formats (preprocessed from LS-DYNA d3plot via PhysicsNeMo-Curator). The design is highly modular, enabling users to write their own readers, bring their own architectures, or implement custom rollout/transient schemes. Multiple experiments (different datasets, models, or feature sets) are managed via Hydra experiment configs without touching the core code.

For an in-depth comparison between the Transolver and MeshGraphNet models and the transient schemes for crash dynamics, see [this paper](https://arxiv.org/pdf/2510.15201).

### Bumper Beam modeling

<p align="center">
  <img src="../../../docs/img/crash/bumper_beam.gif" alt="Bumper beam animation" width="80%" />

</p>

### Body-in-White Crash Modeling

<p align="center">
  <img src="../../../docs/img/crash/crash_case4_reduced.gif" alt="Crash animation" width="60%" />

</p>

### Crushcan Modeling

<p align="center">
  <img src="../../../docs/img/crash/crushcan.gif" alt="Crushcan animation" width="60%" />

</p>

## Quickstart

This pipeline uses **Hydra configs** to manage different datasets, models, and feature sets from a single codebase. Each experiment is a self-contained config file in `conf/` with the naming pattern `experiment_*.yaml`.

1) Pick or create an experiment config. Ready-to-use configs are provided:

```
conf/experiment_bumper_geotransolver.yaml
conf/experiment_crash_transolver.yaml
```

2) Preprocess your data (see [Data Preprocessing](#data-preprocessing) below).

3) **Launch training and inference.** Data paths are not hardcoded—edit the experiment YAML to replace `???` placeholders, or pass overrides on the command line. Training needs `training.raw_data_dir`, `training.raw_data_dir_validation` (and `training.global_features_filepath` for experiments with global features). Inference needs `inference.raw_data_dir_test`.

   **Training:**

   ```bash
   # Single GPU
   python train.py --config-name=experiment_bumper_geotransolver

   # Multi-GPU (DDP)
   torchrun --nproc_per_node=4 train.py --config-name=experiment_bumper_geotransolver
   ```

   **Inference:**

   ```bash
   # Single GPU
   python inference.py --config-name=experiment_bumper_geotransolver

   # Multi-GPU
   torchrun --nproc_per_node=4 inference.py --config-name=experiment_bumper_geotransolver
   ```

   Predictions are saved under `output_dir_pred` (default `./predicted_vtps/`). Normalization stats are written to `./stats/` during training and reused for inference.

You can override any individual config value on the command line without editing any file:

```bash
python train.py --config-name=experiment_bumper_geotransolver training.epochs=500 training.start_lr=1e-3
```

## Prerequisites

This example requires:
- LS-DYNA crash data preprocessed to VTP or Zarr via [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator).
- A GPU-enabled environment with PyTorch.

Install dependencies:

```bash
pip install -r requirements.txt
```

To use graph-based models (e.g., MeshGraphNet) or the graph datapipe, install the PhysicsNeMo `gnns` extra: `pip install "nvidia-physicsnemo[gnns]"` or `uv sync --extra gnns`.

## Data Preprocessing

`PhysicsNeMo` has a related project to help with data processing, called
[PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator).
Using `PhysicsNeMo-Curator`, crash simulation data from LS-DYNA can be processed into training-ready formats easily.

PhysicsNeMo-Curator can preprocess d3plot files into **VTP** (for visualization and smaller datasets) or **Zarr** (for large-scale ML training).

### Quick Start

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

### Input Data Structure

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

This format is directly compatible with the Zarr reader in this example.

## Training

Training is managed via Hydra configurations located in `conf/`.
The main script is `train.py`.

### Config Structure

```
conf/
├── experiment_bumper_geotransolver.yaml  # ← self-contained experiment configs
├── experiment_crash_transolver.yaml
├── datapipe/                              # dataset configs (generic defaults)
│   ├── graph.yaml
│   └── point_cloud.yaml
├── model/                                 # model configs
│   ├── geotransolver_one_shot_training.yaml
│   ├── transolver_autoregressive_rollout_training.yaml
│   ├── transolver_one_step_rollout.yaml
│   ├── transolver_time_conditional.yaml
│   ├── figconvunet_autoregressive_rollout_training.yaml
│   ├── figconvunet_one_step_rollout.yaml
│   ├── figconvunet_time_conditional.yaml
│   ├── mgn_autoregressive_rollout_training.yaml
│   ├── mgn_one_step_rollout.yaml
│   └── mgn_time_conditional.yaml
├── reader/                                # reader configs
│   ├── vtp.yaml
│   └── zarr.yaml
├── training/default.yaml                  # generic training hyperparameters
└── inference/default.yaml                 # generic inference options
```

Each experiment config is self-contained with its own defaults for reader, datapipe, model, training, and inference. All experiment-specific settings (data paths, dataset sizes, feature lists) are defined directly in the experiment config file.

### Launch Training

Single GPU:

```bash
python train.py --config-name=experiment_bumper_geotransolver
```

Multi-GPU (Distributed Data Parallel):

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py --config-name=experiment_bumper_geotransolver
```

## Inference

Use `inference.py` to evaluate trained models on test crash runs.

**Note:** For inference, place all `.vtp` files directly in `raw_data_dir_test` (flat layout).
Each file is treated as one run; outputs are written under `output_dir_pred/rank{N}/{run_name}/`.

Example directory structure:
```
data/
├── Run0.vtp
├── Run1.vtp
└── Run2.vtp
```

Single GPU:

```bash
python inference.py --config-name=experiment_bumper_geotransolver
```

Multi-GPU (Distributed Data Parallel):

```bash
torchrun --nproc_per_node=<NUM_GPUS> inference.py --config-name=experiment_bumper_geotransolver
```

Runs are sharded across ranks: rank `r` processes `run_items[r::world_size]`.
Predicted meshes are written as .vtp files under `./predicted_vtps/`, and can be opened using ParaView.

## Experiments

Each experiment is a self-contained YAML file in `conf/` with the naming pattern `experiment_*.yaml`. Each config file includes all defaults and experiment-specific settings.

### Anatomy of an experiment config

Data paths must be set either in the config file or via CLI overrides. For training: `raw_data_dir`, `raw_data_dir_validation`. For inference: `raw_data_dir_test`. Use `???` in the config to make them mandatory overrides, or set concrete paths directly.

```yaml
# conf/experiment_my_experiment.yaml

hydra:
  job:
    chdir: True
  run:
    dir: ./outputs/

experiment_name: "My-Experiment"
experiment_desc: "Description of the experiment"
run_desc: "Run description"

defaults:
  - reader: vtp
  - datapipe: point_cloud
  - model: geotransolver_one_shot_training
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
```

### Provided experiments

| File | Dataset | Model | Launch command |
|------|---------|-------|----------------|
| `experiment_bumper_geotransolver.yaml` | Bumper beam (VTP) | GeoTransolver one-shot | `python train.py --config-name=experiment_bumper_geotransolver` |
| `experiment_crash_transolver.yaml` | Car body-in-white crash (VTP) | Transolver autoregressive | `python train.py --config-name=experiment_crash_transolver` |

### Adding a new experiment

1. Create `conf/experiment_<my_experiment>.yaml` following the template above.
2. Set defaults for reader, datapipe, model, training, and inference in the `defaults` section.
3. Set all required fields: `raw_data_dir`, `raw_data_dir_validation` (training), `raw_data_dir_test` (inference), `num_time_steps`, `num_training_samples`. Either set concrete paths in the config or use `???` and pass them via CLI when launching `train.py` or `inference.py` as appropriate.
4. If using global features, set `global_features_filepath`; otherwise use `null`.
5. Optionally override any model or training hyperparameter directly in the experiment file (e.g., `model.out_dim: 150`, `training.epochs: 5000`), or add a new model config under `conf/model/` and select it in the defaults.
6. Run: `python train.py --config-name=experiment_<my_experiment>`

You can also override lower-level defaults in the `defaults` section:
```yaml
defaults:
  - reader: vtp
  - override /model: transolver_time_conditional  # Override model
  - override /training: default
  - _self_
```

## Datapipe: how inputs are constructed and normalized

The datapipe is responsible for turning raw LS-DYNA/Abaqus or other crash runs into model-ready tensors and statistics. It does three things in a predictable, repeatable way: it reads and filters the raw data, it constructs inputs and targets with a stable interface, and it computes the statistics required to normalize both positions and features. This section explains what the datapipe returns, how to configure it, and what models should expect to receive at training and inference time.

At a high level, each sample corresponds to one crash run. The datapipe loads the full deformation trajectory for that run, and emits exactly two items: inputs x and targets y. Inputs are a dictionary with two entries. The first entry, 'coords', is a [N, 3] tensor that contains the positions at the first timestep (t0) for all retained nodes. The second entry, 'features', is a [N, F] tensor that contains the concatenation of all node-wise features configured for this experiment. The order of columns in 'features' matches the order you provide in the configuration. This means if your configuration lists features as [thickness, Y_modulus], then column 0 will always be thickness and column 1 will always be Y_modulus. Targets y are the remaining positions from t1 to tT flattened along the feature dimension, so y has shape [N, (T-1)*3].

Configuration lives under `conf/datapipe/`. There are two datapipe variants: one for graph-based models and one for point-cloud models. Both accept the same core options, and both expose a `features` list. The `features` list is the single source of truth for what goes into the 'features' tensor and in which order. If you do not want any features, set `features: []` and the datapipe will return an empty [N, 0] tensor for 'features' while keeping 'coords' intact. If you add more features later, the datapipe will preserve their order and update the per-dimension statistics automatically.

Under the hood the datapipe reads node positions over time via the configured reader (VTP or Zarr). For each run it constructs a fixed number of time steps, selects and reindexes the active nodes, and optionally builds graph connectivity. It also computes statistics necessary for normalization. Position statistics include per-axis means and standard deviations, as well as normalized velocity and acceleration statistics used by autoregressive rollouts. Feature statistics are computed column-wise on the concatenated 'features' tensor. During dataset creation the datapipe normalizes the position trajectory using position means and standard deviations and normalizes every column of 'features' using feature means and standard deviations. The resulting tensors are numerically stable and consistent across training and evaluation. The statistics are written under `./stats/` as `node_stats.json` and `feature_stats.json` during training, and then read back in evaluation or inference.

Readers are configurable through Hydra. A reader is any callable that returns `(srcs, dsts, point_data)`, where `point_data` is a list of records—one per run. Each record must include 'coords' as a [T, N, 3] array and one array per configured feature name. Arrays for features can be [N] or [N, K]; the datapipe will promote [N] to [N, 1] and then concatenate all feature arrays in the order declared in the configuration to form 'features'. If you are using graph-based models, the `srcs` and `dsts` arrays will be used to build a PyG `Data` object with symmetric edges and self-loops, and initial edge features are computed from positions at t0 (displacements and distances). If you are using point-cloud models, graph connectivity is ignored but the remainder of the pipeline is identical.

Models should consume the two-part input without guessing column indices. Positions are always available in `x['coords']` and every node-wise feature is already concatenated in `x['features']`. If you need to separate features later—for example to log per-feature metrics—you can do so deterministically because the order of columns in `x['features']` exactly matches the `features` list in the configuration. For time-conditional models, you can pass the full `x['features']` to your functional input; for autoregressive models, you can concatenate `x['features']` to the normalized velocity (and time, if used) to form the model input at each rollout step.

Finally, the datapipe is designed to be resilient to the “no features” case. If you set `features: []`, the 'features' tensor simply has width zero. Statistics are computed correctly (zero-length mean and unit standard deviation) and concatenations degrade gracefully to the original position-only behavior. This makes it easy to start simple and then scale up to richer feature sets without revisiting model-side code or the data normalization logic.

For completeness, the datapipe also records a lightweight name-to-column map called `_feature_slices`. It associates each configured feature name with its [start, end) slice in `x['features']`. You typically won’t need it if you just consume the full `features` tensor, but it enables reliable, reproducible slicing by name for diagnostics or logging.

### Model I/O at a glance (what models receive)

- Inputs `x` (dictionary):
  - `x['coords']`: `[N, 3]` positions at `t0`
  - `x['features']`: `[N, F]` concatenated node features in the config‑specified order (can be width 0)

- Targets `y`: `[N, (T-1)*3]` positions from `t1..tT` flattened along the feature dimension.

- Rollout input construction (high level):
  - Autoregressive: per step, the model consumes normalized velocity, optionally time, and `x['features']`; positions are fed as embeddings/state.
  - Time‑conditional one‑step: time index is provided once per call along with `x['features']` and the positional embedding.

- Transolver specifics: for unstructured data, the embedding tensor is required; in this pipeline it is the current positions over the rollout. If you set `features: []`, the functional input still includes velocity (and optionally time), so the overall functional dimension remains > 0.

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
# conf/experiment_my_experiment.yaml
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

At `__getitem__` time, the datapipe converts the selected scalars to a dict of scalar tensors and attaches them to the `SimSample`:

```python
sample.global_features = {
    "velocity_x":      tensor(-5.0),
    "thickness_scale": tensor(1.0),
    "rwall_origin_y":  tensor(0.0),
}
```

In the model forward pass, these are stacked into a single global embedding vector and passed to the network. The **`global_dim`** parameter in the model config must equal the number of global features selected:

```yaml
# conf/model/geotransolver_one_shot_rollout_training.yaml
global_dim: 3   # must match len(datapipe.global_features)
```

If `global_features` is `null`, `sample.global_features` is `None` and the model must handle this case (currently only `GeoTransolverOneShotTraining` uses global features; other models ignore them).

## Reader: built-in VTP and Zarr readers and how to add your own

The reader opens preprocessed simulation data and produces the arrays the datapipe consumes. Raw LS-DYNA d3plot files must be preprocessed to VTP or Zarr using [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/structural_mechanics/crash) before use. The reader is swappable via Hydra so you can adapt the pipeline to different formats or add your own.

### Built-in VTP reader (PolyData)

A lightweight VTP reader is provided in `vtp_reader.py`. It treats each `.vtp` file in a directory as a separate run. For each run it opens the `d3plot` with `lasso.dyna.D3plot` and extracts node coordinates, time-varying displacements, element connectivity, and part identifiers. If a LS‑DYNA keyword (`.k`) file is present, it parses the shell section definitions to obtain per-part thickness values, then converts those into per-node thickness by averaging the values of incident elements. To avoid contaminating the training with rigid content, the reader classifies nodes as structural or wall based on a displacement variation threshold and drops wall nodes. After filtering, it builds a compact node index, remaps connectivity, and—if you are training a graph model—collects undirected edges from the remapped shell elements. It can optionally save one VTP file per time step to help you visually inspect the trajectories, or write the predictions to those files in inference.

The reader then assembles the per-run record expected by the datapipe. Positions are returned under the key `'coords'` as a float array of shape `[T, N, 3]`, where T is the number of time steps and N is the number of retained nodes after filtering and remapping. Feature arrays are returned one per configured feature name; for example, if your datapipe configuration lists `features: [thickness, Y_modulus]`, the reader should provide a `'thickness'` array with shape `[N]` or `[N, 1]` and a `'Y_modulus'` array with shape `[N]` or `[N, K]`. The datapipe promotes 1D arrays to 2D and concatenates all provided feature arrays in the order given by the configuration to form the final `'features'` block supplied to the model.

If you use the graph datapipe, the edge list is produced by walking the filtered shell elements and collecting unique boundary pairs, then symmetrized and augmented with self-loops inside the datapipe when constructing the PyG `Data` object. If you use the point‑cloud datapipe, the edge outputs are ignored but the rest of the record shape is the same, so you can swap between model families by changing configuration only.

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
# conf/experiment_my_experiment.yaml
defaults:
  - reader: vtp
  - datapipe: point_cloud
  - model: transolver_time_conditional
  - training: default
  - inference: default
  - _self_
```

And configure features in the experiment's `datapipe` block:

```yaml
# conf/experiment/my_experiment.yaml
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
# conf/experiment_my_experiment.yaml
defaults:
  - reader: zarr
  - datapipe: point_cloud
  - model: transolver_autoregressive_rollout_training
  - training: default
  - inference: default
  - _self_
```

And configure features in the experiment's `datapipe` block:

```yaml
# conf/experiment/my_experiment.yaml
datapipe:
  static_features: [thickness]  # Must match fields stored in Zarr
```

**Recommended workflow:**
1. Use PhysicsNeMo-Curator to preprocess d3plot → VTP or Zarr once
2. Use corresponding reader for all training/validation

### Data layout expected by readers

- VTP reader (`vtp_reader.py`):
  - `<DATA_DIR>/*.vtp` (each `.vtp` is treated as one run)
  - Displacements stored as 3‑component arrays in point_data with names like `displacement_t0.000`, `displacement_t0.005`, ... (fallback accepts any `displacement_t*`).

- Zarr reader (`zarr_reader.py`):
  - `<DATA_DIR>/*.zarr/` (each `.zarr` directory is treated as one run)
  - Contains pre-computed `mesh_pos`, `edges`, and feature arrays

### Write your own reader

To write your own reader, implement a Hydra‑instantiable function or class whose call returns a three‑tuple `(srcs, dsts, point_data)`. The first two entries are lists of integer arrays describing edges per run (they can be empty lists if you are not producing a graph), and `point_data` is a list of Python dicts with one dict per run. Each dict must contain `'coords'` as a `[T, N, 3]` array and one array per feature name listed in `conf/datapipe/*.yaml` under `features`. Feature arrays can be `[N]` or `[N, K]` and should use the same node indexing as `'coords'`. For convenience, a simple class reader can accept the Hydra `split` argument (e.g., "train" or "test") and decide whether to save VTP frames, but this is optional.

As a starting point, your YAML can point to a class by dotted path. For a class:

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
    ):
        ...
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

## Performance tips

- AMP is enabled by default in training; it reduces memory and accelerates matmuls on modern GPUs.
- For multi-GPU training, use `torchrun --nproc_per_node=<NUM_GPUS> train.py`.
- For DDP, prefer `torchrun --nproc_per_node=<NUM_GPUS> train.py`.

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

- [ ] **Normalize global features**: Global features (e.g., velocity_x, thickness_scale, rwall_origin_y) are currently passed to the model without normalization. Add support for computing and applying per-feature mean/std (or similar) so global inputs are normalized consistently with node features and positions.
- [ ] **Normalize dynamic targets**: Dynamic targets (e.g., effective_plastic_strain, stress_vm) are currently passed in the target `y` without normalization, while positions are normalized. Add per-target mean/std and denormalize at inference when exporting to VTP.
- [ ] **Support batch_size > 1**: The pipeline currently uses `batch_size=1` due to variable node counts per sample. Add padding or batching logic to enable larger batch sizes for improved throughput.

## Troubleshooting / FAQ

- I want to run without an experiment config file.
  - You can still override required fields directly, but you'll need to specify a base config or create a minimal experiment config. For recurring setups, creating an experiment config file is recommended.

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
- [GeoTransolver: Learning Physics on Irregular Domains Using Multi-scale Geometry Aware Physics Attention Transformer](https://arxiv.org/pdf/2512.20399)]
- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)
- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/pdf/2010.03409)
