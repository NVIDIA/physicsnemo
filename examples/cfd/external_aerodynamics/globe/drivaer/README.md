# GLOBE for DrivAerML (3D Car Aerodynamics)

This example trains a GLOBE model to predict surface pressure coefficient
(C_p) and skin friction coefficient (C_f) on parametrically-varied 3D car
body geometries from the DrivAerML dataset.

## Problem Description

The DrivAerML dataset contains 500 CFD simulations of a parametric DrivAer car
in a wind tunnel.  Each sample varies the car body geometry (length, frontal
area, shape details) while keeping freestream conditions constant.

Of the 500 simulations, this example uses 484 (436 train + 48 validation),
inherited verbatim from the upstream [DoMINO DrivAerML
splits](https://github.com/NVIDIA/physicsnemo-cfd/blob/main/workflows/bench_example/drivaer_ml_files/train.csv)
so cross-applicability between recipes is preserved.  The 16 missing run IDs
are: `167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403,
473`.

GLOBE learns to map from car body geometry (represented as a triangulated
surface mesh) to nondimensional surface fields:

- **C_p** - Pressure coefficient (scalar)
- **C_f** - Skin friction coefficient (3D vector)

These can be integrated over the car body surface to obtain aerodynamic
force coefficients (Cd, Cl, Cs) for engineering evaluation.

## Dataset

[The DrivAerML dataset is available from
HuggingFace](https://huggingface.co/datasets/neashton/drivaerml). Once
downloaded, it should be available at a path like:

```text
drivaer_data_full/
  run_1/
    boundary_1.vtp       # Boundary surface mesh (~600 MB)
    volume_1.vtu         # Volume mesh (~45 GB, not used by this example)
    geo_ref_1.csv        # Geometry reference lengths and areas
    force_mom_1.csv      # Ground-truth force/moment coefficients
    drivaer_1.stl        # CAD geometry
    ...
  run_2/
    ...
```

This example uses only the VTP boundary files, geometry CSVs, and force
coefficient CSVs.  Volume VTU files are not loaded.

Set the dataset location via one of:

- The `--data-dir` CLI argument
- The `DRIVAER_DATA_DIR` environment variable (set automatically by `run.sh`)

## Usage

### Training

Multi-node via SLURM (edit `DRIVAER_DATA_DIR` in `run.sh` to point to your
dataset root, then):

```bash
sbatch run.sh
```

Or run locally on a single GPU:

```bash
export DRIVAER_DATA_DIR=/path/to/drivaer_data_full
uv run torchrun --nproc-per-node 1 train.py
```

### Inference

```bash
export DRIVAER_DATA_DIR=/path/to/drivaer_data_full
uv run python inference.py
```

Optionally set `GLOBE_OUTPUT_DIR` to point to a specific training output
directory.  Otherwise, the most recent output is used.

### MLflow Tracking

```bash
export MLFLOW_TRACKING_URI="sqlite:///output/mlflow.db"
uv run mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

## Architecture

GLOBE represents the PDE solution as a boundary integral with learnable
Green's function-like kernels.  Key architectural properties:

- **Discretization-invariant**: The boundary mesh can be decimated without
  changing predictions in the fine-mesh limit.
- **Rotation-equivariant**: Predictions follow rotations of the input.
- **Translation-equivariant**: Predictions follow translations.
- **Parity-equivariant**: Reflections are handled correctly.

For 3D, the model uses `n_spatial_dims=3` with 4 spherical harmonic terms
and multiscale kernels parameterized by per-sample reference lengths
(car length, sqrt of frontal area).

## Preprocessing Pipeline

1. Load VTP car body surface (~8.8M quad-dominant cells) and triangulate
2. Compute nondimensional fields: C_p (already nondimensional),
   C_f = wallShearStress / q_inf
3. Interpolate cell-centered data to mesh vertices
4. Parse reference lengths and force coefficients from CSV files
5. Cache preprocessed samples as .pt files for fast subsequent loading
6. At load time, randomly subsample cells for the GLOBE boundary mesh
   (default 80K faces, configurable via `--n-faces-per-boundary`)

## Expected Training Behavior

This section describes what a healthy training run with reference settings (as
of 2026-05-05) looks like, so you can sanity-check your own runs.

- Reference hardware: 4 nodes x 4 B200 192GB
- Wall time per epoch: 217 seconds
- Peak VRAM per rank: ~179 GB
- At epoch 20, train loss ~0.07, validation loss ~0.07; both still decreasing.
- At epoch 100, train loss ~0.0150, validation loss ~0.0155; both still decreasing.
- At epoch 500, train loss ~0.0080, validation loss ~0.0085; both still decreasing.
