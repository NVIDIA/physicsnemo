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
splits](https://github.com/NVIDIA/physicsnemo-cfd/blob/main/workflows/benchmarking/drivaer_ml_files/train.csv)
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

To train with SLURM on one node of 4 GPUs (the reference setup), set
`DRIVAER_DATA_DIR` in `run.sh` to your dataset root. A full run takes about
2.5 days on 4 GB300 GPUs as one job. If your cluster allows a job that long,
set `--time` in `run.sh` to 3 days (`--time=3-00:00:00`), and submit it:

```bash
sbatch run.sh
```

Otherwise, queue the run as a chain of shorter jobs. Each job saves a
checkpoint before its time limit, and the next job with the same name resumes
from it (`--dependency=singleton`). Each job spends about 75 minutes
compiling, so a 4-hour job trains about 19 epochs. A full run then takes
about 24 jobs and 4 days. Queue a few spare jobs:

```bash
for i in $(seq 26); do sbatch run.sh; done
```

If the run stops before epoch 450, queue more jobs. A job that starts after
the last epoch exits right after it loads the checkpoint.

Checkpoints from earlier versions of this example do not resume. Start a new
run in a new output directory, for example with a new job name (`-J`) in
`run.sh`.

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
and multiscale kernels parameterized by two per-sample reference lengths:
the car's reference length `lRef` (its wheelbase) and a flat-plate estimate
of the turbulent boundary-layer thickness.

The boundary input has three partitions: the car surface (`vehicle`) and
the wind-tunnel floor, split into an inviscid `slip_floor` upstream of
x = -2.339 m and a viscous `no_slip_floor` downstream.

The defaults follow the model and training configuration of the DrivAerML
results in the [GLOBE paper](https://arxiv.org/abs/2511.15856):

- 2 communication hyperlayers, kernel networks with 3 hidden layers of 512,
  16 scalar and 8 vector latent channels, and a Barnes-Hut opening angle of 1.5
- 50,000 car-surface faces and 50,000 prediction points sampled per car and
  step
- Huber loss
- Muon for matrix-shaped parameters and AdamW for the rest, with learning rate
  3e-3 and weight decay 1e-4
- 450 epochs, with the learning rate dropping 10x every 200 epochs (3e-3,
  3e-4, then 3e-5). The paper's run stopped after about 430 epochs.
- bfloat16 autocast and `torch.compile`, on one node of 4 GPUs (one car per
  GPU, so 4 cars per step)

The data pipeline is this example's own, so its scores are not directly
comparable with the paper's. It trains on 436 cars and validates on 48 (the
paper: 400 and 34), predicts at mesh vertices with equal weights (the paper:
at surface cells, with an area-weighted loss), and uses each car's own
reference lengths.

## Preprocessing Pipeline

1. Load VTP car body surface (~8.8M quad-dominant cells) and triangulate
2. Compute nondimensional fields: C_p (already nondimensional),
   C_f = wallShearStress / q_inf
3. Interpolate cell-centered data to mesh vertices
4. Parse reference lengths and force coefficients from CSV files
5. Create the two floor patches at the height of the tire contact
6. Cache preprocessed samples as .pt files for fast subsequent loading
   (about 295 GB for all 484 cars, written during the first epoch)
7. At load time, randomly subsample cells for the GLOBE boundary mesh
   (default 50K faces, configurable via `--n-faces-per-boundary`)

## Expected Training Behavior

This section describes a healthy run with the default settings, so you can
sanity-check your own runs.

- Reference hardware: 1 node of 4 NVIDIA GB300 GPUs
- Wall time per epoch: about 454 seconds (7.6 minutes), at 3.9 seconds per
  training step. The first epoch of each job takes about 60 to 75 minutes,
  almost all of it `torch.compile`. A few of the next epochs take 12 to 14
  minutes each, while one GPU recompiles the training step. These times were
  measured with the preprocessing cache already built. A new run also writes
  the cache during its first epoch, which then takes longer.
- Peak memory per GPU: about 118 GiB in `nvidia-smi` (about 117 GiB in
  MLflow's `system/vram_gb`)
- Losses:

  | Epoch | Train loss | Validation loss |
  | --- | --- | --- |
  | 20 | 0.0118 | 0.0114 |
  | 100 | 0.00478 | 0.00541 |
  | 200 | 0.00352 | 0.00401 |
  | 300 | 0.00193 | 0.00270 |
  | 400 | 0.00176 | 0.00265 |
  | 450 | 0.00165 | 0.00255 |

  Both losses step down when the learning rate drops at epoch 201 (MLflow's
  `lr` metric shows the drop one epoch early). The lowest validation loss,
  0.00252, came at epoch 414, and `best_model.mdlus` holds that epoch's model.

  These are one run's values. Expect yours to agree to about 10%. From epoch
  20 to epoch 200, the validation loss changes from one epoch to the next by
  3% (median) and by up to 13%. A second run was 10% higher at epoch 19.
