# GLOBE for DrivAerML (3D Car Aerodynamics)

This example trains a GLOBE model to predict surface pressure coefficient
(C_p) and skin friction coefficient (C_f) on parametrically-varied 3D car
body geometries from the DrivAerML dataset.

## Problem Description

The DrivAerML dataset contains ~207 scale-resolving simulations (SRS) of a
parametric DrivAer car in a virtual wind tunnel at 140 km/h.  Each sample
varies the car body geometry (length, frontal area, shape details) while
keeping freestream conditions constant.

GLOBE learns to map from car body geometry (represented as a triangulated
surface mesh) to nondimensional surface fields:

- **C_p** - Pressure coefficient (scalar)
- **C_f** - Skin friction coefficient (3D vector)

These can be integrated over the car body surface to obtain aerodynamic
force coefficients (Cd, Cl, Cs) for engineering evaluation.

## Dataset

The DrivAerML dataset should be available at a path like:

```
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

## Usage

### Training

Set `DRIVAER_DATA_DIR` to point to your dataset root, then launch via SLURM:

```bash
export DRIVAER_DATA_DIR=/path/to/drivaer_data_full
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

1. Load VTP boundary mesh (~8.8M surface cells)
2. Extract car body cells (~83% of total) by filtering out tunnel walls
   (inlet, outlet, sides, ceiling, ground plane)
3. Interpolate cell-centered data to mesh vertices
4. Decimate car body to ~20K faces for GLOBE boundary input
5. Compute nondimensional fields: C_p (already nondimensional),
   C_f = wallShearStress / q_inf
6. Parse reference lengths and force coefficients from CSV files
7. Cache preprocessed samples as .pt files for fast subsequent loading
