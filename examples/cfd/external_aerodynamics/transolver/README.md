# Transolver CFD Example: Code Overview

This directory contains the essential components for training and evaluating a
Transolver model tailored to external aerodynamics CFD problems. The Transolver model
adapts the Attention mechanism, encouraging the learning of meaningful representations.
In each PhysicsAttention layer, input points are projected onto state vectors through
learnable transformations and weights. These transformations are then used to compute
self-attention among all state vectors, and the same weights are reused to project
states back to each input point.

By stacking multiple PhysicsAttention layers, the Transolver model learns to map from
the functional input space to the output space with high fidelity. The PhysicsNeMo
implementation closely follows the original Transolver architecture
([https://github.com/thuml/Transolver](https://github.com/thuml/Transolver)), but
introduces modifications for improved numerical stability and compatibility with NVIDIA
TransformerEngine.

The training workflow for Transolver leverages the same input datasets as DoMINO. For
more information on the datasets, refer to the external_aerodynamics example for DoMINO
and the [PhysicsNeMo Curator](https://github.com/NVIDIA/physicsnemo-curator) for data
preparation guidance. Unlike DoMINO, Transolver does not require neighbor points or any
explicit structure (such as graph connections or edges) in the input data, resulting in
a much simpler datapipe that is fully encapsulated within these examples.

Below, we provide a high-level overview of the main files and their roles in the
workflow.

---

## 1. `src/train.py`

This script serves as the main entry point for training the DoMINO/Transolver model,
utilizing distributed data parallelism. The training loop processes the entire dataset,
computing a simple point-wise relative L2 loss for either surface or volume data, and
includes downsampling to handle the high native mesh resolutions.

The script is designed for multi-GPU training using PyTorch’s DistributedDataParallel,
and it loads datasets through a custom datapipe that supports distributed sampling.
Model instantiation is flexible, supporting either surface or volume predictions
as specified in the configuration.

The script supports mixed-precision training with gradient
scaling, and it manages checkpointing with the physicsnemo checkpointing utils.
Throughout training, metrics are logged to both TensorBoard and the console and
validation is performed after each epoch.

The script can be launched on a single GPU with

```bash
python train.py --config-name train_surface
```

or, for multi-GPU training, use `torchrun` or other distributed job launch tools.

Example output for one epoch of the script, in an 8 GPU run, looks like:

```default
[2025-07-17 14:27:36,040][training][INFO] - Epoch 47 [0/54] Loss: 0.117565 Duration: 0.78s
[2025-07-17 14:27:36,548][training][INFO] - Epoch 47 [1/54] Loss: 0.109625 Duration: 0.51s
[2025-07-17 14:27:37,048][training][INFO] - Epoch 47 [2/54] Loss: 0.122574 Duration: 0.50s
[2025-07-17 14:27:37,556][training][INFO] - Epoch 47 [3/54] Loss: 0.125667 Duration: 0.51s
[2025-07-17 14:27:38,063][training][INFO] - Epoch 47 [4/54] Loss: 0.101863 Duration: 0.51s
[2025-07-17 14:27:38,547][training][INFO] - Epoch 47 [5/54] Loss: 0.113324 Duration: 0.48s
[2025-07-17 14:27:39,054][training][INFO] - Epoch 47 [6/54] Loss: 0.115478 Duration: 0.51s
...[remove for brevity]...
[2025-07-17 14:28:00,662][training][INFO] - Epoch 47 [49/54] Loss: 0.107935 Duration: 0.49s
[2025-07-17 14:28:01,178][training][INFO] - Epoch 47 [50/54] Loss: 0.100087 Duration: 0.52s
[2025-07-17 14:28:01,723][training][INFO] - Epoch 47 [51/54] Loss: 0.097733 Duration: 0.55s
[2025-07-17 14:28:02,194][training][INFO] - Epoch 47 [52/54] Loss: 0.116489 Duration: 0.47s
[2025-07-17 14:28:02,605][training][INFO] - Epoch 47 [53/54] Loss: 0.104865 Duration: 0.41s

Epoch 47 Average Metrics:
+-------------+---------------------+
|   Metric    |    Average Value    |
+-------------+---------------------+
| l2_pressure | 0.20262257754802704 |
| l2_shear_x  | 0.2623567283153534  |
| l2_shear_y  | 0.35603201389312744 |
| l2_shear_z  | 0.38965049386024475 |
+-------------+---------------------+

[2025-07-17 14:28:02,834][training][INFO] - Val [0/6] Loss: 0.114801 Duration: 0.22s
[2025-07-17 14:28:03,074][training][INFO] - Val [1/6] Loss: 0.111632 Duration: 0.24s
[2025-07-17 14:28:03,309][training][INFO] - Val [2/6] Loss: 0.105342 Duration: 0.23s
[2025-07-17 14:28:03,537][training][INFO] - Val [3/6] Loss: 0.111033 Duration: 0.23s
[2025-07-17 14:28:03,735][training][INFO] - Val [4/6] Loss: 0.099963 Duration: 0.20s
[2025-07-17 14:28:03,903][training][INFO] - Val [5/6] Loss: 0.092340 Duration: 0.17s

Epoch 47 Validation Average Metrics:
+-------------+---------------------+
|   Metric    |    Average Value    |
+-------------+---------------------+
| l2_pressure | 0.19346082210540771 |
| l2_shear_x  | 0.26041051745414734 |
| l2_shear_y  | 0.3589216470718384  |
| l2_shear_z  |  0.370105117559433  |
+-------------+---------------------+
```

---

## 2. `conf/train_volume.yaml` and `conf/train_surface.yaml`

These configuration files define all the settings required for a training run. They
specify output directories, run identifiers, random seeds, and precision settings, as
well as the number of epochs, checkpoint intervals, and whether to use compilation. The
model architecture is described here, including input and output dimensions, the number
of layers, embedding size, attention heads, and activation functions. Optimizer
settings such as the type (AdamW), learning rate, weight decay, and other
hyperparameters are also included. Data-related settings cover paths to the training and
validation datasets, worker and thread counts, memory pinning, and which data keys to
load. Finally, logging preferences are set, controlling the level and format of output.

Note that, to use TransformerEngine, you must enable it directly in the model settings.
TransformerEngine is incompatible with `torch.compile`.  

---

## 3. `loss.py`

This file defines the loss functions used during Transolver training, primarily
focusing on a relative L2 loss. For surface data, it computes mean squared error (MSE)
or root mean squared error (RMSE) for both pressure (a scalar) and wall shear (a
vector), handling each component separately before combining the results.

---

## 4. `metrics.py`

Evaluation metrics for model predictions are defined here. The metrics are designed to
aggregate results across distributed processes, ensuring consistency in multi-GPU
setups. For surface predictions, the script computes normalized L2 errors for both
pressure and shear components, after unnormalizing the predictions and targets. The
structure of the code allows for straightforward extension to additional metrics or
application domains as needed.

---

## 5. `datapipe.py`

Efficient loading of large CFD datasets stored in Zarr format is handled by this file,
which implements like a PyTorch dataset. The datapipe reads large arrays in a chunk-aligned
manner for optimal performance, leveraging threads for parallel I/O. It offers
flexibility in specifying which data keys to load and which are considered “large” (and
thus read in chunks, vs. small arrays which are directly read in their entirety).
For faster GPU transfers, it can allocate directly to pinned memory, then share buffers
with numpy for streaming from disk directly to pinned memory.  The dataload has a
prefetch utility, as well, enabling the dataloader to queue the next batch to GPU.

CPU to GPU transfers are performed in a separate CUDA stream from the main computation,
enabling async transfers overlapping with model training.  The main parameter to tune
is the number of workers for the threading: too many, and you will introduce
CPU overhead limiting the model performance.  Too few, and the dataload won't acheive
peak throughput of dataloading.
