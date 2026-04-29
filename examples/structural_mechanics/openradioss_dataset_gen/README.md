<!-- markdownlint-disable -->
# OpenRadioss Dataset Generation

End-to-end recipe for generating parameterized OpenRadioss simulation datasets
and converting the output to the LS-DYNA-style `d3plot` layout expected by
[PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator) — the
upstream step for training structural dynamics surrogates in
`examples/structural_mechanics/crash/` (for bumper beam) and
`examples/structural_mechanics/drop_test` (for drop test).

Two flows are supported out of the box:

| Flow           | Base case                    | Varied parameters                              | Default DoE size |
|----------------|------------------------------|------------------------------------------------|------------------|
| **bumper_beam**| Front-impact bumper beam     | geometry scale, shell thickness, impact velocity, rigid-wall diameter/origin | 135 runs         |
| **drop_test**  | Cell-phone drop onto a plane | per-material Young's modulus (E), rigid-wall plane orientation (rx, ry, rz) | 192 runs         |

Both flows share the same runner, d3plot-renaming step, and summary helpers;
only the parameter-mutation logic differs.


## Datasets

Each flow is a Design-of-Experiments sweep: the generator takes the Cartesian
product of the configured axes, writes one case folder per combination, and
applies the mutations directly to the `_0000.rad` starter deck. The defaults
below live in `DEFAULT_EXPERIMENT_SETUP` near the bottom of each flow's
`generate_dataset.py` — edit or replace that dict to resize or reshape the sweep.

### Bumper beam

Front-impact bumper-beam setup driven by an impactor on a rigid wall. The
sweep targets the crash-response envelope by perturbing *geometry and loading*:
bumper shape, wall-thickness mass distribution, impact speed, and the
impactor's relative location on the beam.

Default: `5 × 3 × 3 × 1 × 3 = 135` runs.

| # | Parameter | Deck target | Default values | Count |
|---|-----------|-------------|----------------|------:|
| 1 | **Geometry scale `(sx, sy, sz)`** — per-axis multiplier applied to every node | `/NODE` block | `(1,1,1)`, `(1,0.5,1)`, `(1,1,0.5)`, `(1,2,1)`, `(1,1,2)` | 5 |
| 2 | **Initial velocity `(vx, vy, vz)`** of the impactor (mm/ms) | `/INIVEL` line 2 | `(-5,0,0)`, `(-3,0,0)`, `(-7,0,0)` | 3 |
| 3 | **Shell thickness multiplier** | `/PROP/SHELL` line 4, col 3 | `1.0`, `0.7`, `1.3` | 3 |
| 4 | **Rigid-wall diameter** (mm) | `/RWALL` line 3, col 3 | `254.0` | 1 |
| 5 | **Rigid-wall origin `(x, y, z)`** — `M` is replaced; `M1` shifts by the same delta so the normal is preserved | `/RWALL` lines 4 + 5 | `(-170,0,0)`, `(-170,120,0)`, `(-170,240,0)` | 3 |

Global-feature columns emitted for training:
`geo_scale_{x,y,z}`, `velocity_{x,y,z}`, `thickness_scale`, `rwall_diameter`,
`rwall_origin_{x,y,z}`.

### Drop test

Cell-phone drop onto a plane. The sweep targets *material stiffness × impact
pose* — perturbing Young's modulus for every deformable component while
tilting the contact plane — to cover the stiffness/orientation interactions
that drive peak stress and deflection.

Default: `2⁵ × 6 = 192` runs.

| # | Parameter | Deck target | Default values | Combinations |
|---|-----------|-------------|----------------|-------------:|
| 1 | **Young's modulus scale per material** — scales the `E` value on the line after the `# E Nu` comment in each `/MAT/ELAST` or `/MAT/PLAS_TAB` block | mat 1 (polymer), mat 4 (battery), mat 5 (glass), mat 8 (PCB), mat 9 (composites) | each mat ∈ `{0.8, 1.2}` (±20%) | 2⁵ = 32 |
| 2 | **Rigid-wall plane orientation `(rx, ry, rz)`** — rotates the `M→M1` normal about `M` by the given XYZ Euler angles (degrees); `M1` is rewritten | `/RWALL` line 5 | `(0,0,0)`, `(±10,0,0)`, `(0,±10,0)`, `(0,0,10)` | 6 |

Global-feature columns emitted for training:
`e_scale_mat{1,4,5,8,9}`, `rwall_orientation_{rx,ry,rz}`.


## Directory layout

```
openradioss_dataset_gen/
├── README.md
├── common/
│   ├── radioss_runner.py        # Starter / Engine / VTK / D3PLOT batch runner
│   ├── rename_d3plot.py         # Rename <BASE>.d3plot* -> d3plot*
│   └── summary_utils.py         # summary.json + summary.csv writer
├── bumper_beam/
│   ├── generate_dataset.py      # DoE generator (see DEFAULT_EXPERIMENT_SETUP)
│   ├── run_simulations.py       # Thin wrapper over common.radioss_runner
│   ├── restructure_global_features.py   # summary.json -> global_features.json
│   └── templates/               # place Bumper_Beam_AP_meshed_0000.rad / _0001.rad here
└── drop_test/
    ├── generate_dataset.py
    ├── run_simulations.py
    ├── restructure_global_features.py
    └── templates/               # place Cell_Phone_Drop_0000.rad / _0001.rad here
```


## Prerequisites

- **OpenRadioss** built with the GFortran-Linux executables (`starter_linux64_gf`,
  `engine_linux64_gf`, `anim_to_vtk_linux64_gf`). Point the runner at your build
  via the `OPENRADIOSS_ROOT` env var.
- **`vortex_radioss`** for the `anim -> d3plot` conversion step:
  ```bash
  pip install vortex-radioss
  ```

### Base case inputs

The `.rad` starter and engine files are **not** shipped in the recipe because
the bumper deck is ~1.8 MB and the drop-test deck is ~42 MB. Fetch them from
the upstream OpenRadioss sources and place them under the corresponding
`templates/` directory:

| Flow        | Files to place under `templates/`                          | Source |
|-------------|------------------------------------------------------------|--------|
| bumper_beam | `Bumper_Beam_AP_meshed_0000.rad`, `Bumper_Beam_AP_meshed_0001.rad` | https://openradioss.atlassian.net/wiki/spaces/OPENRADIOSS/pages/11075585/Bumper+Beam |
| drop_test   | `Cell_Phone_Drop_0000.rad`, `Cell_Phone_Drop_0001.rad`     | OpenRadioss sample models (Cell Phone Drop example) |


## Workflow (same for both flows)

All commands below assume you are in the flow directory (`bumper_beam/` or
`drop_test/`).

### 1. Generate the parameterised dataset

```bash
python generate_dataset.py
```

Creates:
- `dataset/run1/`, `dataset/run2/`, … each containing a mutated `_0000.rad`
  starter, a copy of the `_0001.rad` engine deck, and a per-run `<run>.json`
  metadata file.
- `dataset/summary.json` — hierarchical log of every run.
- `dataset/summary.csv` — flat, ML-loader-friendly view.

To customise the DoE, edit `DEFAULT_EXPERIMENT_SETUP` near the bottom of
`generate_dataset.py`, or import `generate_dataset(...)` from another driver
script.

### 2. Run the simulations

```bash
export OPENRADIOSS_ROOT=/path/to/OpenRadioss   # required
export MAX_PARALLEL_JOBS=4                     # optional
export OMP_NUM_THREADS=8                       # optional
python run_simulations.py
```

For each `run*/` folder, the runner executes:
1. `starter_linux64_gf -i <BASE>_0000.rad -nt $OMP_NUM_THREADS` (log: `starter.log`)
2. `engine_linux64_gf  -i <BASE>_0001.rad`                       (log: `engine.log`)
3. `anim_to_vtk_linux64_gf <BASE>A### > <BASE>A###.vtk` per animation frame
4. `vortex_radioss.animtod3plot.Anim_to_D3plot.readAndConvert(<BASE>)` — emits
   `<BASE>.d3plot`, `<BASE>.d3plot01`, … (log: `d3plot_conv.log`)

Total wall time scales with `MAX_PARALLEL_JOBS * OMP_NUM_THREADS`; do not
oversubscribe CPU cores.

Set `DEBUG_MODE=1` to run only the first case.

### 3. Rename d3plot files to the LS-DYNA convention

```bash
python ../common/rename_d3plot.py \
    --dataset-dir ./dataset \
    --base-name Bumper_Beam_AP_meshed      # or Cell_Phone_Drop
```

PhysicsNeMo-Curator expects the plain `d3plot`, `d3plot01`, … layout; the
runner produces `<BASE>.d3plot*`. This step renames them in place.

### 4. Restructure global features for the datapipe

The recipes' datapipe expects a `global_features.json` keyed by run ID
(see `examples/structural_mechanics/crash/README.md`, *Global features*
section). Convert `dataset/summary.json` into that format:

```bash
python restructure_global_features.py                 # writes ./global_features.json
```

Bumper-beam keys: `geo_scale_{x,y,z}`, `velocity_{x,y,z}`, `thickness_scale`,
`rwall_diameter`, `rwall_origin_{x,y,z}`.

Drop-test keys: `e_scale_mat{1,4,5,8,9}`,
`rwall_orientation_{rx,ry,rz}`.


## One-command end-to-end

After placing the `.rad` templates in `templates/`:

```bash
cd bumper_beam   # or: cd drop_test
python generate_dataset.py && \
    python run_simulations.py && \
    python ../common/rename_d3plot.py --dataset-dir ./dataset \
        --base-name Bumper_Beam_AP_meshed && \
    python restructure_global_features.py
```


## Handing off to PhysicsNeMo-Curator

After step 3, the dataset layout matches what PhysicsNeMo-Curator expects:

```
dataset/
├── run1/
│   ├── d3plot
│   ├── d3plot01
│   └── ...
├── run2/
│   └── ...
└── ...
```

Point the curator's `etl.source.input_dir` at this `dataset/` folder and
follow the VTP/Zarr export instructions in
[`examples/structural_mechanics/crash/README.md`](../crash/README.md#data-preprocessing).
Once curated, the `global_features.json` produced in step 4 slots directly
into the training experiment configs via `training.global_features_filepath`.


## Troubleshooting

- **`starter.log` reports "BAD CARD" or zero E on a `/MAT` block.**
  OpenRadioss parses materials with fixed-width 20-char columns. If you add
  new materials with very large or very small E scales, inspect the rewritten
  `_0000.rad` to confirm the scaled value still fits the field.
- **`anim_to_vtk` or `vortex_radioss` cannot find shared libraries.**
  The runner sets `LD_LIBRARY_PATH` from `OPENRADIOSS_ROOT/extlib/{hm_reader,h3d}/lib/linux64`.
  If your OpenRadioss layout differs, edit `common/radioss_runner.build_radioss_env`.
- **`run_simulations.py` reports "No animation files (A001) found".**
  The Engine step did not finish a first animation checkpoint — check
  `engine.log`. Common cause: timestep collapse after an over-aggressive E
  reduction on the drop-test flow.
- **`restructure_global_features.py` fails with `Duplicate run_id`.**
  Indicates `summary.json` was appended to twice. Regenerate from a clean
  `dataset/` folder.


## Assumptions & OpenRadioss version notes

- Assumed Radioss deck format: block-style `.rad` files with whitespace- or
  comma-separated fixed-width fields, matching the Altair docs shipped with
  OpenRadioss 2024.x. No `/INCLUDE` or `/SUBMODEL` support.
- Material E detection in the drop-test flow relies on a preceding comment
  line that mentions `E` and `nu`/`Nu`. Remove this comment in the starter
  and the scaling silently becomes a no-op — intended, since the deck would
  then have an unknown field layout.
- The bumper flow scales `/PROP/SHELL` thickness at line 4, column 3. Decks
  that place thickness elsewhere (e.g. `/PROP/TYPE1`) need the mutation logic
  adapted in `bumper_beam/generate_dataset.py`.
