<!-- markdownlint-disable -->
# Unified External Aerodynamics Recipe

NOTE THIS README IS AI GENERATED AND YOU SHOULD USE IT WITH EXTREME CAUTION.
I WILL GO THROUGH IT CAREFULLY FOR ACCURACY BEFORE FINAL REVIEW OR MERGE. 


Train surface-field prediction models (pressure coefficient and wall shear
stress) on multiple car-geometry datasets simultaneously. The recipe currently
supports **DrivaerML** (point-cloud surface) and **SHIFT SUV Estate**
(triangulated surface), merged into a single training stream via
`MultiDataset`. A Hydra/YAML-driven transform pipeline normalizes every
dataset into a common schema — handling differences in mesh representation,
field names, units, and storage conventions — so the downstream model and
loss code never need to know which dataset a sample came from.

## Quick start

```bash
cd examples/cfd/external_aerodynamics/unified_external_aero_recipe

# 1. Inspect raw data (sanity-check fields, coordinate ranges, vertical axis)
python -m src.inspect_data

# 2. Train (single GPU)
python src/train.py

# 2b. Train (multi-GPU)
torchrun --nproc_per_node=N src/train.py
```

## Data layout

Each dataset lives on disk as a directory of simulation runs. Each run
contains PhysicsNeMo `Mesh` objects saved as `.pt` files (TensorDict memmap
format).

```
drivaer_ml_pnm_mesh/
  run_1/
    boundary_1.vtp.pt/      <-- surface mesh (point cloud, no cells)
    drivaer_1.stl.pt/        <-- STL geometry
    volume_1.vtu.pt/         <-- volume mesh
  run_2/
    ...

shift_suv_pnm_mesh/estate/
  run_00001/
    merged_surfaces.vtp.pt/  <-- surface mesh (triangulated, has cells)
    merged_surfaces.stl.pt/  <-- STL geometry
    merged_volumes.vtu.pt/   <-- volume mesh
  run_00002/
    ...
```

The two surface datasets store the same physical quantities but in
structurally different ways:

| Property | DrivaerML | SHIFT SUV Estate |
|---|---|---|
| Mesh type | Point cloud (~8.8M pts, no cells) | Triangulated surface (~2.5M pts, ~5M cells) |
| Pressure field | `point_data["pMeanTrim"]` | `cell_data["pressure_average"]` |
| Wall shear stress | `point_data["wallShearStressMeanTrim"]` (vec3) | `cell_data["wall_shear_stress_average"]` (vec3) |
| Global data | `TimeValue` (scalar) | (empty) |
| File pattern | `**/boundary*.vtp.pt` | `**/merged_surfaces.vtp.pt` |

## Pipeline architecture

Each dataset gets its own `MeshDataset` with an ordered chain of
`MeshTransform` steps defined in YAML. The two datasets are then merged
via `MultiDataset`.

```
          ┌─────────────────────────────────────────────────────────────┐
          │  Per-dataset pipeline (one per YAML config)                │
          │                                                            │
          │  MeshReader               Load raw Mesh from .pt files     │
          │       │                                                    │
          │  (metadata injection)     Write U_inf, rho_inf, p_inf, nu  │
          │       │                   from YAML metadata into          │
          │       │                   global_data (done by builder)     │
          │       │                                                    │
          │  DropMeshFields           Remove unwanted fields           │
          │       │                   (e.g. TimeValue)                 │
          │       │                                                    │
          │  CenterMesh               Translate center of mass         │
          │       │                   to origin                        │
          │       │                                                    │
          │ (RandomRotateMesh)        Random yaw around vertical axis  │
          │       │                   (training only)                   │
          │       │                                                    │
          │ (RandomTranslateMesh)     Random horizontal shift           │
          │       │                   (training only)                   │
          │       │                                                    │
          │  NonDimensionalizeByMeta  Convert to Cp and Cf using       │
          │       │                   q_inf = ½ρ|U∞|²                  │
          │       │                                                    │
          │  RenameMeshFields         Map dataset-specific names to    │
          │       │                   canonical names (pressure, wss)  │
          │       │                                                    │
          │  NormalizeMeshFields      z-score normalize using          │
          │       │                   inline stats from YAML            │
          │       │                                                    │
          │  SubsampleMesh            Downsample to fixed point/cell   │
          │       │                   count for batching                │
          │       │                                                    │
          │  MeshToTensorDict         Convert Mesh → TensorDict        │
          │       │                                                    │
          │  (ComputeCellCentroids)   Compute cell centers from        │
          │       │                   connectivity (SHIFT SUV only)     │
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
          │   Collate    │  Stacks samples into batched tensors:
          │              │  geometry (B,N,3), U_inf (B,1,3),
          │              │  fields (B,N,4)
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

- **CenterMesh** — Centers each car geometry at the origin so that
  rotations happen around a sensible point. DrivaerML uses the point mean
  (no cells available); SHIFT SUV uses area-weighted cell centroids.

- **RandomRotateMesh / RandomTranslateMesh** — Data augmentation.
  Currently commented out in both dataset configs; uncomment to enable.
  Rotation is restricted to the vertical axis. Translation is restricted
  to horizontal axes by setting the vertical component of `max_offset` to
  zero.

- **NonDimensionalizeByMetadata** — Converts raw physical fields into
  non-dimensional coefficients using the injected freestream metadata:
    - Pressure → Cp: `(p - p_inf) / q_inf` where `q_inf = 0.5 * rho_inf * |U_inf|²`
    - Wall shear stress → Cf: `tau / q_inf`
  
  This transform also supports velocity non-dimensionalization (`U / |U_inf|`)
  and provides an `inverse()` method for re-dimensionalizing predictions.

- **RenameMeshFields** — Maps dataset-specific field names to canonical
  names (`pressure`, `wss`) so all downstream code uses a single naming
  convention.

- **NormalizeMeshFields** — Applies z-score normalization using
  inline statistics declared in the YAML config or loaded from a `.pt`
  file. Handles scalar and vector fields differently.  The normalization
  stats are saved alongside model checkpoints for use at inference time.

- **SubsampleMesh** — Randomly downsamples each mesh to a fixed size
  (200k points for DrivaerML, 50k cells for SHIFT SUV) so that samples
  can be batched. Different samples in the same dataset get different
  random subsets each epoch.

- **MeshToTensorDict** — Terminal transform that converts the `Mesh`
  object into a flat `TensorDict`. After this step, further mesh
  transforms are invalid.

- **ComputeCellCentroids** — For cell-based datasets (SHIFT SUV),
  computes the centroid of each cell from the connectivity and vertex
  positions. These centroids serve as the "point positions" for the model.

- **RestructureTensorDict** — Reorganizes the flat TensorDict into
  `input/` and `output/` groups expected by the collate function. Maps
  point positions and freestream velocity into `input`, and target fields
  (pressure, wss) into `output`.

- **MultiDataset with output_strict=False** — The two datasets produce
  TensorDicts with different internal structure (one has `point_data`
  fields, the other has `cell_data` fields). Strict output validation is
  disabled because the keys differ. The training loop uses
  `metadata["dataset_index"]` to distinguish samples if needed.

## Non-dimensionalization and normalization

The pipeline applies two layers of field conditioning:

1. **Physics-based non-dimensionalization** (`NonDimensionalizeByMetadata`)
   converts raw simulation outputs to standard aerodynamic coefficients
   (Cp, Cf). This is essential when combining datasets that may use
   different freestream conditions, fluid properties, or unit conventions.
   The freestream metadata (`U_inf`, `rho_inf`, `p_inf`) is declared
   per-dataset in the YAML config.

2. **Statistical normalization** (`NormalizeMeshFields`) applies z-score
   scaling so that all field values fed to the model have roughly zero
   mean and unit variance.  Statistics are specified inline in the dataset
   YAML config or loaded from a `.pt` file.

## Model and training

The default model is **GeoTransolver**, a transformer-based architecture
for point-cloud regression that uses multi-scale local attention with
geometric embeddings.

| Setting | Default |
|---|---|
| Model | `GeoTransolver` (8 layers, 128 hidden, 8 heads) |
| Input | Point positions (N×3) + freestream velocity (1×3) |
| Output | Pressure (1) + wall shear stress (3) = 4 channels |
| Loss | Huber (smooth L1), normalized by total channels |
| Optimizer | Muon (2D params) + AdamW (other params) |
| Scheduler | StepLR (step=100, gamma=0.1) |
| Precision | float32 (float16/bfloat16/float8 supported) |
| Batch size | 1 |

The **collate function** (`src/collate.py`) stacks datapipe outputs into
the model's forward signature:

```python
{
    "geometry":         (B, N, 3),  # point positions
    "local_embedding":  (B, N, 3),  # same as geometry
    "global_embedding": (B, 1, 3),  # freestream velocity
    "fields":           (B, N, 4),  # [pressure, wss_x, wss_y, wss_z]
}
```

The **loss calculator** (`src/loss.py`) and **metric calculator**
(`src/metrics.py`) are both driven by the same target config
(`pressure: scalar`, `wss: vector`), so adding a new field is a
config-only change.

## Scripts

All scripts are run from the recipe root directory:

```bash
cd examples/cfd/external_aerodynamics/unified_external_aero_recipe
```

### Inspect data

```bash
python -m src.inspect_data
python -m src.inspect_data --configs conf/dataset/drivaer_ml_surface.yaml
```

Prints coordinate ranges (min/max/mean per axis), field names, shapes, and
value statistics for one sample from each dataset. Use the output to confirm
which axis is vertical and to understand the raw data layout.

### Benchmark datapipe throughput

```bash
python -m src.benchmark
python -m src.benchmark --max-samples 20
python -m src.benchmark --configs conf/dataset/drivaer_ml_surface.yaml
```

Measures per-sample load time, throughput, and prints the output TensorDict
layout with per-component value statistics for each pipeline and the
combined MultiDataset.

### Train

```bash
# Single GPU
python src/train.py

# Multi-GPU
torchrun --nproc_per_node=N src/train.py

# Override config values
python src/train.py precision=bfloat16 training.num_epochs=100
```

Trains a GeoTransolver on surface pressure and wall shear stress. Supports
checkpointing (auto-resume), TensorBoard logging, mixed precision
(float16/bfloat16/float8 via Transformer Engine), `torch.compile`, and
NVIDIA profiling.

## Configuration

The recipe uses a two-level config structure:

- **`conf/train_surface.yaml`** — Top-level training config. Specifies
  the model, optimizer, scheduler, precision, and which dataset configs
  to load.
- **`conf/dataset/*.yaml`** — Per-dataset configs. Each declares the
  reader, transform pipeline, freestream metadata, target field types,
  and metrics.

### Dataset config anatomy

```yaml
name: drivaer_ml_surface

train_datadir: /path/to/train/data
val_datadir: /path/to/val/data

# Freestream conditions (injected into global_data by the dataset builder)
metadata:
  U_inf: [30.0, 0.0, 0.0]
  p_inf: 0.0
  rho_inf: 1.225
  nu: 1

# Transform pipeline — each entry is Hydra-instantiated
pipeline:
  reader:
    _target_: ${dp:MeshReader}
    path: ${train_datadir}
    pattern: "**/boundary*.vtp.pt"
  transforms:
    - _target_: ${dp:DropMeshFields}
      global_data: [TimeValue]
    - _target_: ${dp:CenterMesh}
      use_area_weighting: false
    - _target_: ${dp:NonDimensionalizeByMetadata}
      fields:
        pMeanTrim: pressure
        wallShearStressMeanTrim: stress
      section: point_data
    - _target_: ${dp:RenameMeshFields}
      point_data:
        pMeanTrim: pressure
        wallShearStressMeanTrim: wss
    - _target_: ${dp:NormalizeMeshFields}
      section: point_data
      fields:
        wss: {type: vector, mean: [0.0, 0.0, 0.0], std: 0.00313}
    - _target_: ${dp:SubsampleMesh}
      n_points: 200000
    - _target_: ${dp:MeshToTensorDict}
    - _target_: ${dp:RestructureTensorDict}
      groups:
        input:
          points: points
          U_inf: global_data.U_inf
        output:
          pressure: point_data.pressure
          wss: point_data.wss

targets:
  pressure: scalar
  wss: vector

metrics: [l1, l2, mae]
```

The `${dp:ComponentName}` syntax is an OmegaConf resolver registered by
PhysicsNeMo's datapipe registry. It maps short class names to fully
qualified import paths, so Hydra can instantiate them. Each transform
entry's keys are passed directly as constructor kwargs.

### Adding a new dataset

1. Create a new YAML config in `conf/dataset/` following the pattern above.
2. Set `reader.path` and `reader.pattern` for your data files.
3. Declare the correct `metadata:` block with freestream conditions.
4. Choose the right `section:` (`point_data` or `cell_data`) in
   `NonDimensionalizeByMetadata` and `RenameMeshFields`.
5. For cell-based data, add `ComputeCellCentroids` after `MeshToTensorDict`
   and use `cell_centroids` as the point source in `RestructureTensorDict`.
6. Add inline normalization stats to `NormalizeMeshFields` (or point
   `stats_file` at a `.pt` file with precomputed statistics).
7. Add an entry in `conf/train_surface.yaml` under `data:` pointing to
   your new config.

No Python code changes are needed.

## Source modules

| Module | Purpose |
|---|---|
| `src/datasets.py` | Factory functions: `build_surface_dataset`, `build_multi_surface_dataset`, `load_dataset_config`. Hydra-instantiates readers and transforms from YAML; injects metadata into `global_data`. |
| `src/nondim.py` | Recipe-local transform: `NonDimensionalizeByMetadata`. Registered into the global datapipe registry. |
| `src/collate.py` | `surface_collate` — stacks datapipe `(TensorDict, metadata)` tuples into batched model inputs. |
| `src/loss.py` | `LossCalculator` — config-driven loss for mixed scalar/vector fields. Supports Huber, MSE, relative MSE. |
| `src/metrics.py` | `MetricCalculator` — config-driven metrics (relative L1, relative L2, MAE) with optional distributed all-reduce. |
| `src/utils.py` | `build_muon_optimizer` (Muon+AdamW), `parse_target_config`, `FieldSpec` dataclass. |
| `src/train.py` | Training loop with DDP, mixed precision, checkpointing, TensorBoard, and profiling. |
| `src/inspect_data.py` | Loads one sample per dataset and prints geometry/field summaries. |
| `src/benchmark.py` | Measures datapipe throughput and prints output shapes with value statistics. |

## Design decisions

**Why keep native point/cell representation?**
DrivaerML is a point cloud; SHIFT SUV is a triangulated surface. Converting
one to match the other would lose information (cell connectivity or point
precision). Instead, each dataset keeps its native structure and the
`SubsampleMesh` + `RestructureTensorDict` transforms produce a uniform
`(N, 3)` point array for the model — from vertex positions for point
clouds, or from cell centroids for triangulated meshes.

**Why two-stage field conditioning (non-dim then normalize)?**
Non-dimensionalization is physics: it removes dependence on freestream
conditions and produces standard aerodynamic coefficients (Cp, Cf) that are
comparable across datasets. Statistical normalization is numerics: it
rescales those coefficients so the model sees inputs with zero mean and unit
variance, improving training stability. Separating them means you can
change normalization strategy without touching the physics, and vice versa.

**Why inject metadata from YAML instead of storing it in the `.pt` files?**
The freestream conditions are not stored in the converted mesh files.
Rather than modifying the data conversion pipeline, we inject them at
runtime from the config. This keeps the `.pt` files format-agnostic and
makes it trivial to change conditions without reconverting data. The
dataset builder reads the `metadata:` block and prepends an injection
step automatically.

**Why output_strict=False in MultiDataset?**
With strict mode, MultiDataset checks that all sub-datasets produce
TensorDicts with identical keys. Since one dataset has `point_data.pressure`
and the other has `cell_data.pressure`, the pre-restructured keys differ.
After `RestructureTensorDict` both datasets produce matching `input/` and
`output/` groups, but the intermediate keys still differ.

**Why Muon + AdamW?**
Muon is used for 2D weight matrices (linear layers, attention projections)
where it acts as a steepest-descent optimizer on the Stiefel manifold.
AdamW handles everything else (biases, layer norms, embeddings). The two
are combined via `CombinedOptimizer`. If all parameters happen to be one
type, the optimizer gracefully falls back to a single optimizer.

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
