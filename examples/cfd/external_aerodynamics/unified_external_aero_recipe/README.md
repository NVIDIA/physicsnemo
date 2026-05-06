<!-- markdownlint-disable -->
# Unified External Aerodynamics Recipe

> This unified recipe is still under some final polishing but nearly
> completed.  Feel free to used it and experiment.  In the meantime,
> be wary of sharp edges!

## Introduction

External Aerodynamic recipes in physicsnemo have proliferated: we have
a number of recipes, across a range of models, all working on different models
with unique data handling, pipelines, model architectures, metrics, training
paradigms, etc.  While there is nothing wrong with that, it does make comparison
challenging and development of new models somewhat challenging.  In this folder,
we have unified the external aerodynamic recipes for most of our best models (notably
missing is our newest model, still in development for large 3D use cases: GLOBE).

Here, you're able to train the following models:
- [Transolver](https://arxiv.org/abs/2402.02366)
- [GeoTransolver](https://arxiv.org/abs/2512.20399)
- [Flare](https://arxiv.org/abs/2508.12594)
- GeoTransolver also supports using the FLARE attention mechanism backend
- DoMINO is coming shortly

We currently support the following datasets:
- DrivaerML

Support for these datasets is coming imminently, with pre-processing support from
physicsnemo curator:
- ShiftSUV Estate
- ShiftSUV Fastback
- ShiftWING
- HiftliftAeroML

## Dataset Handling

The data processing pipeline in this example explicitly performs non dimensionalization
of input data to unitless fields for model inputs.  Check out the yaml configurations
in `conf/dataset/` to see examples: the metadata section describes the reference
parameters for each data.  Because datasets are non-dimensionalized, and are loaded
with the physicsnemo datapipes which support a MultiDataset abstraction, it's 
possible to merge datasets on-the-fly during training to perform multi-dataset
training.  We at PhysicsNeMo haven't extensively explored all of the parameters
of this multi-dataset training yet, but the infrastructure can support it and
we welcome you to try it if you're interested in it.

Dataset non dimensionalization is handled in the `nondim.py` transformation, which
is part of the data transformation pipeline.  See `src/nondim.py` in this example
for the source code.

## Quick start

```bash
cd examples/cfd/external_aerodynamics/unified_external_aero_recipe

# 1. Train (single GPU, default GeoTransolver surface config)
python src/train.py

# 1b. Train with a specific config
python src/train.py --config-name train_transolver_automotive_surface

# 1c. Train (multi-GPU)
torchrun --nproc_per_node=N src/train.py

# 2. Override config values
python src/train.py precision=bfloat16 training.num_epochs=100
```

## Pipeline architecture

Each dataset gets its own `MeshDataset` or `DomainMeshDataset` with an ordered chain of
`MeshTransform` steps defined in YAML.  Multiple datasets are then
merged via `MultiDataset`.

```
          ┌─────────────────────────────────────────────────────────────┐
          │  Per-dataset pipeline (one per YAML config)                │
          │                                                            │
          │  MeshReader / DomainMeshReader                              │
          │       │              Load raw Mesh from .pdmsh/.pmsh files  │
          │       │                                                    │
          │  (metadata injection)     Write U_inf, rho_inf, p_inf, nu  │
          │       │                   from YAML metadata into          │
          │       │                   global_data (done by builder)     │
          │       │                                                    │
          │ (DropMeshFields)          Remove unwanted fields           │
          │       │                   (e.g. TimeValue; drivaer only)    │
          │       │                                                    │
          │ (CenterMesh)              Translate center of mass         │
          │       │                   to origin                        │
          │       │                                                    │
          │ (RandomRotateMesh)        Random yaw around vertical axis  │
          │       │                   (inserted after CenterMesh when   │
          │       │                    augment=true)                    │
          │       │                                                    │
          │ (RandomTranslateMesh)     Random horizontal shift           │
          │       │                   (inserted after CenterMesh when   │
          │       │                    augment=true)                    │
          │       │                                                    │
          │ (NonDimensionalizeByMeta) Convert to Cp/Cf/nondim velocity │
          │       │                   using q_inf = ½ρ|U∞|²            │
          │       │                                                    │
          │ (ComputeSDFFromBoundary)  Compute signed distance field    │
          │       │                   from STL geometry (volume only)   │
          │       │                                                    │
          │ (DropBoundary)            Remove auxiliary STL boundary     │
          │       │                   after SDF (volume only)           │
          │       │                                                    │
          │  RenameMeshFields         Map dataset-specific names to    │
          │       │                   canonical names (pressure, wss)  │
          │       │                                                    │
          │ (NormalizeMeshFields)     z-score normalize using          │
          │       │                   inline stats from YAML            │
          │       │                                                    │
          │ (ComputeSurfaceNormals)   Compute per-cell surface normals │
          │       │                   (surface pipelines only)          │
          │       │                                                    │
          │ (SubsampleMesh)           Downsample to fixed point/cell   │
          │       │                   count (surface pipelines only;    │
          │       │                   volume uses reader subsampling)   │
          │       │                                                    │
          │  MeshToTensorDict         Convert Mesh → TensorDict        │
          │       │                                                    │
          │  (ComputeCellCentroids)   Compute cell centers from        │
          │       │                   connectivity (cell-based only)    │
          │       │                                                    │
          │  RestructureTensorDict    Remap flat TensorDict into       │
          │       │                   input/output groups for the      │
          │       │                   collate function                  │
          └───────┼────────────────────────────────────────────────────┘
                  │
                  ▼
          ┌──────────────┐
          │ MultiDataset │  Concatenates index spaces,
          │              │  adds dataset_index to metadata
          └──────────────┘
                  │
                  ▼
          ┌──────────────┐
          │   Collate    │  Stacks samples into batched tensors
          │              │  via model-specific mapping
          └──────────────┘
```

### Why each step exists

- **Metadata injection** — The dataset builder writes freestream conditions
  (`U_inf`, `rho_inf`, `p_inf`, `nu`) from the YAML config's `metadata:`
  block into each mesh's `global_data`. This makes physical reference
  quantities available to downstream transforms without hardcoding them
  in Python.

- **DropMeshFields** — Removes fields that are not needed for training
  (e.g. `TimeValue` in DrivaerML) to reduce memory and avoid schema
  mismatches when merging datasets.

- **CenterMesh** — Centers each geometry at the origin so that
  rotations happen around a sensible point.  DrivaerML uses point-mean
  centering (`use_area_weighting: false`); SHIFT SUV uses area-weighted
  cell centroid centering (`use_area_weighting: true`).

- **RandomRotateMesh / RandomTranslateMesh** — Data augmentation,
  defined in the `augmentations:` block of each dataset config and
  activated at runtime by setting `augment: true` (default `false`).
  Augmentations are inserted after `CenterMesh` by the dataset builder.
  Rotation is restricted to the vertical axis.  Translation is restricted
  to horizontal axes by setting the vertical component of the offset
  distribution to zero.

- **NonDimensionalizeByMetadata** — Converts raw physical fields into
  non-dimensional coefficients using the injected freestream metadata:
    - Pressure → Cp: `(p - p_inf) / q_inf` where `q_inf = 0.5 * rho_inf * |U_inf|²`
    - Wall shear stress → Cf: `tau / q_inf`
    - Velocity → `U / |U_inf|`
  
  Also supports temperature, density, and identity (pass-through) field
  types.  Provides an `inverse()` method for re-dimensionalizing
  predictions.

  Note that for input points, we non-dimensionalize by a reference scalar `L_ref`.
  In some recipes, the x/y/z axes are all scaled to unit-scale independently.
  Here, we've made a conscious decision to maintain the aspect ratios of the input
  positions and vectors deliberately use a scalar parameter for coordinate
  non-dimensionalization.  To disable, set `L_ref` to 1.0.

- **ComputeSDFFromBoundary** — Volume pipelines only.  Computes a
  signed distance field (and surface normals) from an auxiliary STL
  boundary mesh loaded via the reader's `extra_boundaries` option.
  The SDF and normals are stored into `point_data` and used as
  geometry-aware input features for the model.

- **DropBoundary** — Removes the auxiliary STL boundary mesh after
  `ComputeSDFFromBoundary` has consumed it, keeping only the interior
  volume and the original surface boundary.

- **RenameMeshFields** — Maps dataset-specific field names to canonical
  names (`pressure`, `wss`, `velocity`, etc.) so all downstream code
  uses a single naming convention.

- **NormalizeMeshFields** — Applies z-score normalization using
  inline statistics declared in the YAML config or loaded from a `.pt`
  file.  Handles scalar and vector fields differently.  The normalization
  stats are saved alongside model checkpoints for use at inference time.

  Note that not all fields are normalized, in fact most are not.  Only fields
  that are particularly far from unit mean or standard deviation are normalized.

- **ComputeSurfaceNormals** — Computes per-cell (or per-point) surface
  normals from the mesh connectivity.  Used in surface pipelines to
  provide normal vectors as part of the model's local embedding.

- **SubsampleMesh** — Randomly downsamples each mesh to a fixed size
  (controlled by `sampling_resolution` in the training config) so that
  samples can be batched.  Different samples in the same dataset get
  different random subsets each epoch.

- **MeshToTensorDict** — Terminal transform that converts the `Mesh`
  object into a flat `TensorDict`. After this step, further mesh
  transforms are invalid.

- **ComputeCellCentroids** — For cell-based datasets, computes the
  centroid of each cell from the connectivity and vertex positions.
  These centroids serve as the "point positions" for the model.

- **RestructureTensorDict** — Reorganizes the flat TensorDict into
  `input/` and `output/` groups expected by the collate function.  Maps
  point positions (or cell centroids), normals, and freestream velocity
  into `input`, and target fields into `output`.

## Non-dimensionalization and normalization

The pipeline applies two layers of field conditioning:

1. **Physics-based non-dimensionalization** (`NonDimensionalizeByMetadata`)
   converts raw simulation outputs to standard aerodynamic coefficients
   (Cp, Cf) or non-dimensional velocity.  This is essential when
   combining datasets that may use different freestream conditions, fluid
   properties, or unit conventions.  The freestream metadata (`U_inf`,
   `rho_inf`, `p_inf`) is declared per-dataset in the YAML config.

2. **Statistical normalization** (`NormalizeMeshFields`) applies z-score
   scaling so that all field values fed to the model have roughly zero
   mean and unit variance.  Statistics are specified inline in the dataset
   YAML config or loaded from a `.pt` file.

## Model and training

The default model is **GeoTransolver**, a transformer-based architecture
for point-cloud regression that uses multi-scale local attention with
geometric embeddings.

### Default settings (GeoTransolver automotive surface)

| Setting | Default |
|---|---|
| Model | `GeoTransolver` (12 layers, 256 hidden, 8 heads) |
| Attention type | `GALE` (also supports `GALE_FA` for FLARE-based self-attention) |
| State mixing | `weighted` (learnable sigmoid gate; also supports `concat_project`) |
| Input | Cell centroids (N×3) + surface normals (N×3) + freestream velocity (1×3) |
| Output | Pressure (1) + wall shear stress (3) = 4 channels |
| Loss | Huber (smooth L1), normalized by total channels |
| Optimizer | Muon (2D params) + AdamW (other params) |
| Scheduler | StepLR (step=100, gamma=0.1) |
| Precision | bfloat16 (float16/float32/float8 also supported) |
| Batch size | 1 |

### Data-to-model mapping

The **data-to-model mapping** (`src/collate.py`) converts datapipe
outputs into the model's forward signature.  Mappings are registered by
name in `MODEL_MAPPINGS`; the active mapping is selected via the
`data_mapping` config key (default: `"geotransolver_automotive_surface"`):

```python
# geotransolver_automotive_surface mapping produces:
{
    "geometry":         (B, N, 3),  # cell centroids / point positions
    "local_embedding":  (B, N, 6),  # cat(points, normals) via ["input/points", "input/normals"]
    "local_positions":  (B, N, 3),  # point positions (for local feature builder)
    "global_embedding": (B, 1, 3),  # freestream velocity
    "fields":           (B, N, 4),  # cat(pressure, wss) = prediction target
}
```

All available mappings:

| Mapping name | Model | Domain |
|---|---|---|
| `geotransolver_automotive_surface` | GeoTransolver | Automotive surface (Cp, Cf) |
| `geotransolver_automotive_volume` | GeoTransolver | Automotive volume (U, p, nut) |
| `geotransolver_highlift_surface` | GeoTransolver | HighLift surface (P, T, rho, U, tau_wall) |
| `geotransolver_highlift_volume` | GeoTransolver | HighLift volume (P, T, rho, U) |
| `transolver_automotive_surface` | Transolver | Automotive surface (Cp, Cf) |
| `transolver_automotive_volume` | Transolver | Automotive volume (U, p, nut) |
| `flare_automotive_surface` | FLARE | Automotive surface (Cp, Cf) |
| `flare_automotive_volume` | FLARE | Automotive volume (U, p, nut) |
| `domino_automotive_surface` | DoMINO | Automotive surface (Cp, Cf) |
| `domino_automotive_volume` | DoMINO | Automotive volume (U, p, nut) |

To add a new model, register a mapping in `MODEL_MAPPINGS` and set
`data_mapping` in your config.

The **loss calculator** (`src/loss.py`) and **metric calculator**
(`src/metrics.py`) are both driven by the same target config
(e.g. `pressure: scalar`, `wss: vector`), so adding a new field is a
config-only change.  Supported loss types: Huber, MSE, relative MSE.
Supported metrics: relative L1, relative L2, MAE.

## Scripts

All scripts are run from the recipe root directory:

```bash
cd examples/cfd/external_aerodynamics/unified_external_aero_recipe
```

### Train

```bash
# Single GPU (default: GeoTransolver automotive surface)
python src/train.py

# Explicit config selection
python src/train.py --config-name train_transolver_automotive_surface

# Multi-GPU
torchrun --nproc_per_node=N src/train.py

# Override config values
python src/train.py precision=float32 training.num_epochs=100 training.batch_size=1
```

Supports checkpointing (auto-resume), MLflow logging, mixed precision
(float16/bfloat16/float8 via Transformer Engine), `torch.compile`, and
NVIDIA profiling.

### Benchmark datapipe throughput

```bash
python src/train.py benchmark_io=true
python src/train.py benchmark_io=true +training.benchmark_max_steps=20
```

> NOTE: If you want to profile, we recommend you set the number of epochs to 2.

Measures per-sample load time and throughput without running the model.

## Configuration

The recipe uses a two-level config structure:

- **`conf/train_*.yaml`** — Top-level training configs.  Each specifies
  the model, optimizer, scheduler, precision, and which dataset configs
  to load.  Six are provided (see [Training configurations](#training-configurations)).
- **`conf/dataset/*.yaml`** — Per-dataset configs.  Each declares the
  reader, transform pipeline, freestream metadata, target field types,
  and metrics.

### Dataset config anatomy

```yaml
name: drivaer_ml_surface

train_datadir: /path/to/your/PhysicsNeMo-DrivaerML/

# Freestream conditions (injected into global_data by the dataset builder)
metadata:
  U_inf: [30.0, 0.0, 0.0]
  p_inf: 0.0
  rho_inf: 1.225
  nu: 1
  L_ref: 5.0

# Transform pipeline — each entry is Hydra-instantiated
pipeline:
  reader:
    _target_: ${dp:MeshReader}
    path: ${train_datadir}
    pattern: "**/*.pdmsh/_tensordict/boundaries/surface"
    subsample_n_cells: ${sampling_resolution}
  augmentations:
    - _target_: ${dp:RandomRotateMesh}
      axes: ["z"]
      transform_cell_data: true
      transform_global_data: true
    - _target_: ${dp:RandomTranslateMesh}
      distribution:
        _target_: torch.distributions.Uniform
        low: [-1.0, -1.0, 0.0]
        high: [1.0, 1.0, 0.0]
  transforms:
    - _target_: ${dp:DropMeshFields}
      global_data: [TimeValue]
    - _target_: ${dp:CenterMesh}
      use_area_weighting: false
    - _target_: ${dp:NonDimensionalizeByMetadata}
      fields:
        pMeanTrim: pressure
        wallShearStressMeanTrim: stress
      section: cell_data
    - _target_: ${dp:RenameMeshFields}
      cell_data:
        pMeanTrim: pressure
        wallShearStressMeanTrim: wss
    - _target_: ${dp:NormalizeMeshFields}
      section: cell_data
      fields:
        wss: {type: vector, mean: [0.0, 0.0, 0.0], std: 0.00313}
    - _target_: ${dp:ComputeSurfaceNormals}
      store_as: cell_data
      field_name: normals
    - _target_: ${dp:SubsampleMesh}
      n_cells: ${sampling_resolution}
    - _target_: ${dp:MeshToTensorDict}
    - _target_: ${dp:ComputeCellCentroids}
    - _target_: ${dp:RestructureTensorDict}
      groups:
        input:
          points: cell_centroids
          normals: cell_data.normals
          U_inf: global_data.U_inf
        output:
          pressure: cell_data.pressure
          wss: cell_data.wss

targets:
  pressure: scalar
  wss: vector

metrics: [l1, l2, mae]
```

The `${dp:ComponentName}` syntax is an OmegaConf resolver registered by
PhysicsNeMo's datapipe registry.  It maps short class names to fully
qualified import paths, so Hydra can instantiate them.  Each transform
entry's keys are passed directly as constructor kwargs.

The `${sampling_resolution}` interpolation is resolved from the
top-level training config's `dataset.sampling_resolution` value.

### Manifest-based data splitting

DrivaerML datasets use a `manifest.json` file to define train/val/test
splits.  The manifest path and split names are declared in the top-level
training config:

```yaml
data:
  drivaer_ml:
    config: conf/dataset/drivaer_ml_surface.yaml
    manifest: /path/to/PhysicsNeMo-DrivaerML/manifest.json
    train_split: train
    val_split: val
```

The `ManifestSampler` in `src/datasets.py` resolves manifest entries to
dataset indices and handles distributed sampling across ranks.

For datasets without a manifest (e.g. SHIFT SUV), separate
`train_datadir` / `val_datadir` paths are specified in the dataset YAML.

### Adding a new dataset

1. Create a new YAML config in `conf/dataset/` following the pattern above.
2. Set `reader.path` and `reader.pattern` for your data files.
   Use `MeshReader` for single-mesh files or `DomainMeshReader` for
   domain meshes that contain both interior and boundary sub-meshes.
3. Declare the correct `metadata:` block with freestream conditions.
4. Choose the right `section:` (`point_data` or `cell_data`) in
   `NonDimensionalizeByMetadata`, `RenameMeshFields`, and
   `NormalizeMeshFields`.
5. For cell-based surface data, add `ComputeSurfaceNormals` and
   `ComputeCellCentroids` and use `cell_centroids` as the point source
   in `RestructureTensorDict`.
6. Add inline normalization stats to `NormalizeMeshFields` (or point
   `stats_file` at a `.pt` file with precomputed statistics).
7. Add an entry in the appropriate `conf/train_*.yaml` under `data:`
   pointing to your new config.

No Python code changes are needed.

### MLflow experiment tracking

Training metrics are logged to MLflow.  By default, experiments are
stored in a local `./mlruns` directory.  To use a remote tracking
server, set `mlflow.tracking_uri` in the training config:

```yaml
mlflow:
  tracking_uri: "http://YOUR_MLFLOW_SERVER:5000"
  experiment_name: "unified_external_aero"
  log_every_n_steps: 10
```

## Source modules

| Module | Purpose |
|---|---|
| `src/datasets.py` | Factory functions: `build_dataset`, `build_multi_surface_dataset`, `load_dataset_config`.  Hydra-instantiates readers and transforms from YAML; injects metadata into `global_data`.  Also provides `load_manifest`, `resolve_manifest_indices`, and `ManifestSampler` for manifest-based splitting. |
| `src/nondim.py` | Recipe-local transform: `NonDimensionalizeByMetadata`.  Registered into the global datapipe registry.  Supports pressure, stress, velocity, temperature, density, and identity field types. |
| `src/collate.py` | Data-to-model mapping: converts datapipe `(TensorDict, metadata)` tuples into batched model inputs via a registry of 10 named mapping specs (see [Data-to-model mapping](#data-to-model-mapping)). |
| `src/loss.py` | `LossCalculator` — config-driven loss for mixed scalar/vector fields.  Supports Huber, MSE, relative MSE.  Normalizes total loss by number of output channels. |
| `src/metrics.py` | `MetricCalculator` — config-driven metrics (relative L1, relative L2, MAE) with optional distributed all-reduce.  Reports per-field and per-component (x/y/z) metrics for vector fields. |
| `src/utils.py` | `build_muon_optimizer` (Muon+AdamW via `CombinedOptimizer`), `parse_target_config`, `FieldSpec` dataclass, `set_seed`. |
| `src/train.py` | Training loop with DDP, mixed precision, checkpointing, MLflow logging, I/O benchmarking (`benchmark_io=true`), and profiling. |

## Design decisions

**Why cell-based representation for surfaces?**
Both DrivaerML and SHIFT SUV surface data use triangulated meshes with
fields stored in `cell_data`.  The pipeline computes cell centroids as
the model's point positions and cell-based surface normals for the local
embedding.  For volume data, fields live in `point_data` and vertex
positions are used directly.

**Why two-stage field conditioning (non-dim then normalize)?**
Non-dimensionalization is physics: it removes dependence on freestream
conditions and produces standard aerodynamic coefficients (Cp, Cf) that are
comparable across datasets. Statistical normalization is numerics: it
rescales those coefficients so the model sees inputs with zero mean and unit
variance, improving training stability. Separating them means you can
change normalization strategy without touching the physics, and vice versa.

**Why inject metadata from YAML instead of storing it in the mesh files?**
The freestream conditions are not always stored in the converted mesh
files.  Rather than modifying the data conversion pipeline, we inject
them at runtime from the config.  This keeps the mesh files
format-agnostic and makes it trivial to change conditions without
reconverting data.  The dataset builder reads the `metadata:` block and
prepends an injection step automatically.

**Why Hydra instantiation for the pipeline?**
The entire pipeline is expressed in YAML with no conditional Python logic.
Adding a new dataset, changing augmentation parameters, or swapping
transform order is a YAML-only change. The factory code in `src/datasets.py`
is compact and generic. The configs are self-documenting: you can read a
single YAML file and see exactly what transforms run and in what order.

**Why inline normalization stats?**
Specifying normalization statistics directly in the YAML config (or in a
`.pt` file) keeps the pipeline self-contained and avoids a separate
statistics collection step. The values are easy to inspect, update, and
version-control alongside the rest of the configuration.
