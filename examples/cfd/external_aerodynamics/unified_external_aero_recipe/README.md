<!-- markdownlint-disable -->
# Unified External Aerodynamics Recipe

Surface-mesh datapipes for training external aerodynamics models on multiple
car datasets simultaneously. Currently supports DrivaerML and SHIFT SUV Estate,
merged into a single training stream via `MultiDataset`. The pipeline handles
the fact that these datasets have different mesh representations, different
field names, and different storage conventions -- the transforms normalize
everything into a common schema while preserving each dataset's native
point/cell structure.

## Data layout

Each dataset lives on disk as a directory of runs, where each run contains
physicsnemo `Mesh` objects saved as `.pt` files (tensordict memmap format).

```yaml
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
| Mesh type | Point cloud (8.8M points, no cells) | Triangulated surface (2.5M points, 5M cells) |
| Pressure | `point_data["pMeanTrim"]` | `cell_data["pressure_average"]` |
| Wall shear stress | `point_data["wallShearStressMeanTrim"]` (vec3) | `cell_data["wall_shear_stress_average"]` (vec3) |
| Global data | `TimeValue` (scalar) | (empty) |
| File pattern | `**/boundary*.vtp.pt` | `**/merged_surfaces.vtp.pt` |

## Pipeline architecture

Each dataset gets its own `MeshDataset` with an ordered chain of
`MeshTransform` steps. The two datasets are then merged via `MultiDataset`.

```
          ┌─────────────────────────────────────────────────────────────┐
          │  Per-dataset pipeline (one per YAML config)                │
          │                                                            │
          │  MeshReader          Load raw Mesh from .pt files          │
          │       │                                                    │
          │  SetGlobalField      Inject inlet velocity into            │
          │       │              global_data (constant per dataset)    │
          │       │                                                    │
          │  CenterMesh          Translate mesh center of mass         │
          │       │              to origin                             │
          │       │                                                    │
          │  RandomRotateMesh    Random yaw around vertical axis.      │
          │       │              Rotates WSS vectors and inlet         │
          │       │              velocity along with geometry.         │
          │       │              Pressure (scalar) is unchanged.       │
          │       │                                                    │
          │  RandomTranslateMesh Random horizontal shift (up to 1m).   │
          │       │              Vertical component is zero.           │
          │       │              Does not affect global_data.          │
          │       │                                                    │
          │  RenameMeshFields    Map dataset-specific field names       │
          │       │              to canonical names                    │
          │       │              (pressure, wss)                       │
          │       │                                                    │
          │  MeshToTensorDict    Convert Mesh -> TensorDict            │
          │       │              (terminal transform)                  │
          └───────┼────────────────────────────────────────────────────┘
                  │
                  ▼
          ┌──────────────┐
          │ MultiDataset │  Concatenates index spaces,
          │              │  adds dataset_index to metadata
          └──────────────┘
```

**Why each step exists:**

- **SetGlobalField** -- The inlet velocity is a physical vector that must
  rotate with the car when we apply random yaw. Placing it in `global_data`
  before `RandomRotateMesh` (with `transform_global_data=True`) ensures
  consistent co-rotation. Existing scalar fields like `TimeValue` are
  automatically left alone (the rotation code detects scalars by tensor shape
  and skips them).

- **CenterMesh** -- Each car geometry is centered at the origin so that
  rotations happen around a sensible point. DrivaerML uses the point mean
  (no cells available); SHIFT SUV uses area-weighted centroids.

- **RandomRotateMesh / RandomTranslateMesh** -- Data augmentation. Rotation
  is restricted to the vertical axis (configurable). Translation is restricted
  to horizontal axes by setting the vertical component of `max_offset` to zero.

- **RenameMeshFields** -- Each dataset uses its own naming convention.
  This transform maps them to canonical names (`pressure`, `wss`) so that
  downstream code doesn't need to know which dataset a sample came from.

- **MultiDataset with output_strict=False** -- The two datasets produce
  TensorDicts with different internal structure (one has `point_data` fields,
  the other has `cell_data` fields). Strict output validation is disabled
  because the keys won't match. Use `metadata["dataset_index"]` in the
  training loop to distinguish samples if needed.

## How to run

All scripts are run from the recipe root directory:

```bash
cd examples/cfd/external_aerodynamics/unified_external_aero_recipe
```

### Inspect data (determine vertical axis, check fields)

```bash
python -m src.inspect_data
python -m src.inspect_data --configs conf/dataset/drivaer_ml_surface.yaml
```

Prints coordinate ranges (min/max/mean per axis), field names, shapes, and
value statistics for one sample from each dataset. Use the output to confirm
which axis is vertical and update the YAML configs accordingly.

### Collect field statistics (for future normalization)

```bash
python -m src.collect_stats
python -m src.collect_stats --output stats/ --force
python -m src.collect_stats --configs conf/dataset/drivaer_ml_surface.yaml
```

Iterates over raw meshes (no augmentation) using `FieldStatisticsCollector`
and writes per-sample, per-field statistics to parquet files (one per dataset).
Columns: `field_key`, `mean`, `std`, `min`, `max`, `abs_mean`, `abs_max`, etc.
Import into pandas or any dashboard tool to decide on normalization parameters.

The collector caches results: if the dataset length and tracked fields haven't
changed, subsequent runs skip computation. Use `--force` to recompute.

### Benchmark datapipe throughput

```bash
python -m src.benchmark
python -m src.benchmark --max-samples 20
python -m src.benchmark --configs conf/dataset/drivaer_ml_surface.yaml
```

Measures per-sample load time, throughput, and prints the output TensorDict
layout for each pipeline and the combined MultiDataset.

## Configuration

Each dataset has a YAML config in `conf/dataset/`. The pipeline is declared
using Hydra's `_target_` syntax with the `${dp:ComponentName}` resolver,
which maps short names to full class paths in the datapipes registry.

```yaml
name: drivaer_ml_surface

train_datadir: /path/to/data
val_datadir: /path/to/data

pipeline:
  reader:
    _target_: ${dp:MeshReader}
    path: ${train_datadir}
    pattern: "**/boundary*.vtp.pt"
  transforms:
    - _target_: ${dp:SetGlobalField}
      fields:
        inlet_velocity: [30.0, 0.0, 0.0]
    - _target_: ${dp:CenterMesh}
      use_area_weighting: false
    - _target_: ${dp:RandomRotateMesh}
      axes: ["z"]
      transform_point_data: true
      transform_global_data: true
    - _target_: ${dp:RandomTranslateMesh}
      max_offset: [1.0, 1.0, 0.0]
    - _target_: ${dp:RenameMeshFields}
      point_data:
        pMeanTrim: pressure
        wallShearStressMeanTrim: wss
    - _target_: ${dp:MeshToTensorDict}

targets:
  pressure: scalar
  wss: vector

metrics: [l1, l2, mae]
```

Each entry under `transforms:` is instantiated via `hydra.utils.instantiate()`.
The keys under each transform entry are passed directly as constructor kwargs.

To add a new dataset, create a new YAML config following this pattern. No
Python code changes are needed -- just point `reader.path` at the data,
declare the transform chain, and pass the config path to the scripts.

For datasets with cell-based fields (like SHIFT SUV), use `transform_cell_data`
instead of `transform_point_data` in `RandomRotateMesh`, and put field mappings
under `cell_data:` in `RenameMeshFields`.

## Design decisions

**Why keep native point/cell representation?**
DrivaerML is a point cloud; SHIFT SUV is a triangulated surface. Converting
one to match the other would lose information (cell connectivity or point
precision). Instead, each dataset keeps its native structure and the model
must handle both -- which it will, since the final mapping is to all points.

**Why inject inlet velocity as a global field?**
The inlet velocity is not stored in the converted mesh files. Rather than
modifying the data conversion pipeline, we inject it at runtime from the
config. This keeps the `.pt` files format-agnostic and makes it trivial to
change the velocity without reconverting data.

**Why output_strict=False in MultiDataset?**
With strict mode, MultiDataset checks that all sub-datasets produce
TensorDicts with identical keys. Since one dataset has `point_data.pressure`
and the other has `cell_data.pressure`, the keys differ. Strict mode would
reject this. The training loop uses `metadata["dataset_index"]` to know
which structure to expect.

**Why Hydra instantiation for the pipeline?**
The entire pipeline is expressed in YAML with no conditional Python logic.
Adding a new dataset, changing augmentation parameters, or swapping transform
order is a YAML-only change. The factory code in `src/datasets.py` is under
40 lines. This also makes the configs self-documenting: you can read a single
YAML file and see exactly what transforms run and in what order.

**Why parquet for statistics?**
Parquet is columnar, compressed, and readable by pandas, polars, DuckDB,
and most dashboard tools without any custom parsing code. The
`FieldStatisticsCollector` embeds dataset metadata (sample count, tracked
keys, timestamp) in Arrow file-level metadata for automatic cache
invalidation.
