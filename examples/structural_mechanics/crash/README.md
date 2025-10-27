<!-- markdownlint-disable -->
# Machine Learning Surrogates for Automotive Crash Dynamics

## Problem Overview

Automotive crashworthiness assessment is a critical step in vehicle design.  
Traditionally, engineers rely on high-fidelity finite element (FE)
simulations (e.g., LS-DYNA) to predict structural deformation and crash responses.  
While accurate, these simulations are computationally expensive and
limit the speed of design iterations.

Machine Learning (ML) surrogates provide a promising alternative by learning
mappings directly from simulation data, enabling:

- **Rapid prediction** of deformation histories across thousands of design candidates.
- **Scalability** to large structural models without rerunning costly FE simulations.
- **Flexibility** in experimenting with different model architectures (GNNs, Transformers).

In this example, we demonstrate a unified pipeline for crash dynamics modeling.
The implementation supports both:

- **Mesh-based Graph Neural Networks (MeshGraphNet)** – leverage connectivity from FE meshes.
- **Point-cloud Transformers (Transolver)** – avoid explicit mesh dependency.

## Prerequisites

This example requires:
- Access to LS-DYNA crash datasets (with `d3plot` and `.k` keyword files).
- A GPU-enabled environment with PyTorch.

Install dependencies:

```bash
pip install -r requirements.txt
```

This will install:

- lasso-python (for LS-DYNA file parsing),
- torch_geometric and torch_scatter (for GNN operations),

## Dataset Preprocessing

Crash simulation data is parsed from LS-DYNA d3plot files using the d3plot_reader.py utility.

Key steps:

- Load node coordinates, displacements, element connectivity, and part IDs.
- Parse .k keyword files to assign part thickness values.
- Filter out rigid wall nodes using displacement thresholds.
- Build edges (for graphs) and store per-node features (e.g., thickness).
- Optionally export time-stepped meshes as .vtp for visualization.

Run preprocessing automatically via the dataset class (CrashGraphDataset or CrashPointCloudDataset) when launching training or inference.

## Training

Training is managed via Hydra configurations located in conf/.
The main script is train.py.

Config Structure

```bash
conf/
├── config.yaml              # master config (sets datapipe, model, training)
├── datapipe/                # dataset configs
│   ├── graph.yaml
│   └── point_cloud.yaml
├── model/                   # model configs
│   ├── mgn_autoregressive_rollout_training.yaml
│   ├── mgn_one_step_rollout.yaml
│   ├── mgn_time_conditional.yaml
│   ├── transolver_autoregressive_rollout_training.yaml
│   ├── transolver_one_step_rollout.yaml
│   └── transolver_time_conditional.yaml
├── training/default.yaml    # training hyperparameters
└── inference/default.yaml   # inference options
```

Launch Training
Single GPU:

```bash
python train.py
```

Multi-GPU (Distributed Data Parallel):

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> train.py
```

## Inference

Use inference.py to evaluate trained models on test crash runs.

```bash
python inference.py
```

Predicted meshes are written as .vtp files under
./predicted_vtps/, and can be opened using ParaView.

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

## References

- Automotive Crash Dynamics Modeling Accelerated with Machine Learning (https://arxiv.org/pdf/2510.15201)