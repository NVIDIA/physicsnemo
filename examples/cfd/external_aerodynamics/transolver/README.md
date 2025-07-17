# Transolver CFD Example: Code Overview

This directory contains the core components for training and evaluating a Transolver
model for external aerodynamics CFD problems.  The Transolver model is an adaptation
of the Attention mechanism to encourage models to learn meaningful representations.
In each PhysicsAttention layer, the input points are projected onto state vectors via
learnable transformations and weights.  These learnable transformations are in turn
used to calculate self-attention between all state vectors, and the weights are reused
to project states back to each input point.

Through a series of PhysicsAttention layers, the Transolver model learns high quality
projections from functional input space to output space.  The PhysicsNeMo implementation
of Transolver is an equivalent implementation of the original model architecture
([https://github.com/thuml/Transolver](https://github.com/thuml/Transolver)), with
modifications to improve numerical stability and support NVIDIA TransformerEngine.

The Transolver training recipe reuses the same input datasets as DoMINO - please see
the external_aerodynamics example for DoMINO for more information, as well as
[PhysicsNeMo Curator](https://github.com/NVIDIA/physicsnemo-curator) for information
on producing the data if needed.  The Transolver model does not require the neighbor
points as input to the model, nor does it require any structure to the input points
(graph connections, edges, etc.).  Consequently, the datapipe is significantly simpler
than for DoMINO - the entire datapipe is encapulated here in the examples.

Below is a high-level overview of the main files and their roles in the workflow.

---

## 1. `conf/train.yaml`

**Purpose:**  
Configuration file for training runs.  
**Contents:**  

- **Output and run settings:** Output directory, run ID, random seed, and precision.
- **Training parameters:** Number of epochs, checkpoint intervals, and whether to use
  compilation.
- **Model configuration:** Architecture details (input/output dimensions, number of
  layers, embedding size, attention heads, activation, etc.).
- **Optimizer settings:** Type (AdamW), learning rate, weight decay, and optimizer
  hyperparameters.
- **Data settings:** Paths to training/validation datasets, worker/thread settings,
  memory pinning, and which data keys to load.
- **Logging:** Logging level and format.

---

## 2. `src/train.py`

**Purpose:**  
Main training script for the DoMINO/Transolver model using distributed data parallelism.

**Key Features:**

- **Distributed Training:** Uses PyTorch DistributedDataParallel for multi-GPU training.
- **Data Loading:** Loads datasets using custom datapipe, supports distributed sampling.
- **Model Instantiation:** Builds the model based on config, supports both surface and
  volume predictions.
- **Loss Calculation:** Computes losses for volume, surface, and integral quantities
  (lift/drag).
- **Mixed Precision:** Supports mixed-precision training with gradient scaling.
- **Checkpointing:** Automatically loads/saves checkpoints and tracks best validation
  loss.
- **Logging:** Logs metrics to TensorBoard and console, including GPU memory usage.
- **Validation:** Evaluates model on validation set after each epoch.

---

## 3. `loss.py`

**Purpose:**  
Defines loss functions for training the Transolver model.

**Key Features:**

- **Surface Loss:** Computes MSE or RMSE for both pressure (scalar) and wall shear
  (vector) components, handling them separately and combining the results.
- **Integral Losses:** Implements physics-based losses for lift and drag, using surface
  integrals of predicted and true values, weighted by area, normals, and stream
  velocity.
- **Modularity:** Loss functions are modular and can be combined or extended for
  different training objectives.

---

## 4. `metrics.py`

**Purpose:**  
Defines evaluation metrics for model predictions.

**Key Features:**

- **Distributed Reduction:** Aggregates metrics across distributed processes.
- **Surface Metrics:** Computes normalized L2 errors for pressure and shear components,
  after unnormalizing predictions and targets.
- **Extensibility:** Can be extended for additional metrics or domains.

---

## 5. `datapipe.py`

**Purpose:**  
Implements a PyTorch dataset for efficient loading of large CFD datasets stored in Zarr
format.

**Key Features:**

- **Chunk-Aligned I/O:** Reads large arrays in chunk-aligned fashion for efficiency,
  using threads for parallel I/O.
- **Flexible Key Loading:** Allows specifying which data keys to load and which are
  considered "large" (for chunked reading).
- **Pinned Memory:** Optionally allocates pinned memory for faster GPU transfers.
- **Prefetching:** Supports asynchronous preloading of samples to overlap I/O and
  computation.

---

## Summary

- **`train.yaml`**: All configuration for model, data, optimizer, and logging.
- **`train.py`**: Orchestrates distributed training, validation, checkpointing, and
  logging.
- **`loss.py`**: Physics-informed and standard loss functions for model training.
- **`metrics.py`**: Evaluation metrics for model performance, with distributed support.
- **`datapipe.py`**: High-performance, domain-parallel data loading from Zarr files for
  large-scale CFD datasets.

---

For more details, refer to the docstrings and comments within each file.
