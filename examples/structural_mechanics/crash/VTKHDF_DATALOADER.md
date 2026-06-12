# VTKHDF dataloader for the crash example

This adds a **VTKHDF reader** to the crash example so it can train directly on the
MIT/TUM bumper-beam crash dataset (Harvard Dataverse
[doi:10.7910/DVN/VTMLVB](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VTMLVB)),
which is distributed in [VTKHDF](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format)
format.

The example already ships pluggable readers for **VTP** (`vtp_reader.py`) and **Zarr**
(`zarr_reader.py`), selected via Hydra. `vtkhdf_reader.py` implements the same callable contract,
so it drops into the existing `datapipe.py` and the bumper experiment config with a one-line
`reader=vtkhdf` override — no changes to the datapipe, model, or training loop.

## Why a new reader was needed

The bumper data isn't a single flat mesh like the VTP/Zarr inputs. Each `sim_<id>.vtkhdf` file is a
VTKHDF v2.2 **`PartitionedDataSetCollection`**: a multi-part assembly of ~23 transient
`UnstructuredGrid` parts (bumper-beam segments, crash boxes, a rigid wall/pole, spotweld connectors,
body-in-white). The reader's job is to turn that into the flat `(coords[T,N,3], edges, per-step
targets)` the datapipe expects.

## What the reader does

For each file it:

1. **Selects the structural parts** — the 16 parts that carry both `Von Mises_2D` and
   `Plastic Strain_2D` cell fields (bumper beams, crash boxes, `RWALL`, `NewSECT1`). Detection is
   data-driven; spotwelds / body-in-white (no targets) are dropped. Override with
   `reader.exclude_parts` (e.g. `["RWALL_1"]`).
2. **Merges them into one mesh** (~15k nodes): concatenates per-step `Points`, offsets each part's
   node indices, and unions their connectivity into undirected edges.
3. **Reads 101 timesteps** of geometry (`Points` indexed by `Steps/PointOffsets`; topology is
   constant across steps).
4. **Maps the cell-centered targets to points** (`Von Mises_2D → stress_vm`,
   `Plastic Strain_2D → effective_plastic_strain`) by averaging over incident cells, emitting the
   per-step keys `stress_vm_t<j>` / `effective_plastic_strain_t<j>` the datapipe groups by prefix.
5. **Attaches per-run global features** from the master CSV (see below).

### Reader contract (matches `vtp_reader.Reader`)

```python
srcs, dsts, point_data_all, global_features_all = Reader()(
    data_dir, num_samples, split=None, global_features_filepath=None, logger=None
)
# point_data_all[i] = {"coords": [T, N, 3],
#                      "point_data": {"stress_vm_t0": [N], ..., "effective_plastic_strain_t0": [N], ...}}
# global_features_all[i] = {"velocity_x": ..., "thickness_scale": ..., "rwall_origin_y": ...}
```

## Global features & splits — the master CSV

Per-run scalars are **not** in the VTKHDF files; they live in `bumper_beam_master_with_split.csv`
(keyed by `sim_name`, one row per simulation, plus a `split` column for train/val/test). The reader
uses the CSV only for the split filter; the global scalars are read from a JSON in the exact format
`utils.load_global_features` expects.

This mirrors the shipped bumper contract — the same 3 keys every `bumper_*` config uses
(`global_dim: 3`):

| global feature    | CSV column          |
| ----------------- | ------------------- |
| `velocity_x`      | `velocity_mm_ms`    |
| `thickness_scale` | `thickness_bb_mm`   |
| `rwall_origin_y`  | `pole_offset_y_mm`  |

Build the JSON once:

```bash
python make_global_features.py \
    --master-csv ./bumper_beam_master_with_split.csv \
    --out ./global_features.json
# --thickness-col thickness_cb_mm  # to map thickness_scale to crash-box thickness instead
```

## Usage

```bash
# 1) build the global-features JSON from the master CSV (once)
python make_global_features.py

# 2) train (GeoTransolver, one-shot) on the VTKHDF data
python train.py --config-name=bumper_vtkhdf_geotransolver_oneshot \
    training.raw_data_dir=./simulations \
    training.raw_data_dir_validation=./simulations \
    training.global_features_filepath=./global_features.json
```

`raw_data_dir` and `raw_data_dir_validation` can point at the same directory — the reader filters by
the CSV `split`. The config sets `num_time_steps: 101` and `model.out_dim: 500`
(`= (101-1) * (3 displacement + 2 targets)`).

## Files added

| File | Purpose |
| ---- | ------- |
| `vtkhdf_reader.py` | the reader (merges structural parts → datapipe contract) |
| `conf/reader/vtkhdf.yaml` | Hydra reader selector (`_target_: vtkhdf_reader.Reader`) |
| `conf/bumper_vtkhdf_geotransolver_oneshot.yaml` | experiment config for the VTKHDF bumper data |
| `make_global_features.py` | builds `global_features.json` from the master CSV |
| `tests/test_vtkhdf_reader.py` | contract test (pure h5py/numpy, no torch) |
| `visualize_bumper.py` | quick PNG of the crush over time, colored by von Mises stress |

## Verification

`tests/test_vtkhdf_reader.py` asserts the reader output against the contract on a local copy of the
data: `coords [101, N, 3]`, edge indices in `[0, N)`, both target series present with 101 steps each,
and global features populated. The reader has also been driven through the real
`CrashPointCloudDataset`, producing `SimSample` `node_target` of shape `[N, 100, 5]`
(`→ out_dim 500`).

```bash
python tests/test_vtkhdf_reader.py        # or: pytest tests/test_vtkhdf_reader.py
```

## Notes & caveats

- **Timesteps:** the VTKHDF export has **101** frames (the CSV's `n_timesteps=1000` is the solver
  step count). `num_time_steps` truncates, so set it ≤ 101; `out_dim` must equal
  `(num_time_steps - 1) * 5`.
- **`RWALL` is a rigid wall** — its stress/strain are ~0. It's included by default (it has the target
  fields); exclude via `reader.exclude_parts=[RWALL_1]` if you'd rather not feed it to the model.
- **Disconnected graph:** dropping the spotweld connectors leaves the 16 structural parts as
  separate components. This is fine for `datapipe: point_cloud` + GeoTransolver (attention-based, the
  `srcs/dsts` edges aren't used). It would matter for an edge-message-passing model (MeshGraphNet).
- **torch / PhysicsNeMo version:** the reader itself needs only `h5py` + `numpy`. Running the full
  datapipe needs a torch matching PhysicsNeMo (HEAD uses `torch.amp.GradScaler` and tensordict
  `tensor_only`).
