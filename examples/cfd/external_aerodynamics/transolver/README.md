# Transolver for External Aerodynamics on Irregular Meshes

## Transolver CFD Example: Overview

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

The training example for Transolver uses the [DrivaerML dataset](https://caemldatasets.org/drivaerml/).

> **Note:** Currently, training transolver in this example supports **surface** data only.
> Volumetric data is still in development.

## Requirements

Transolver requires TransformerEngine from NVIDIA, as well as Zarr >= 3.0 and `zarrs`
for the data pipeline.  Install them with `pip install -r requirements.txt`

> For the Transolver datapipe, zarr > 3.0 is required.  If you are using an older
> container, you may need to `unset PIP_CONSTRAINTS` to allow zarr 3.0 or higher.

## Using Transolver for External Aerodynamics

1. Prepare the Dataset.  Transolver uses the same Zarr outputs as other models with DrivaerML.
`PhysicsNeMo` has a related project to help with data processing, called [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator).
Using `PhysicsNeMo-Curator`, the data needed to train Transolver can be setup easily.
Please refer to [these instructions on getting started](https://github.com/NVIDIA/physicsnemo-curator?tab=readme-ov-file#what-is-physicsnemo-curator)
with `PhysicsNeMo-Curator`.  For specifics of preparing the dataset for this example,
see the [download](https://github.com/NVIDIA/physicsnemo-curator/blob/main/examples/external_aerodynamics/domino/README.md#download-drivaerml-dataset)
and [preprocessing](https://github.com/NVIDIA/physicsnemo-curator/blob/main/examples/external_aerodynamics/domino/README.md)
instructions from `physicsnemo-curator`.  Users should apply the
preprocessing steps locally to produce `zarr` output files.

2. Train your model.  The model and training configuration is set in
`conf/train_surface.yaml`, where you can control both network properties
and training properties. See below for an overview and explanation of key
parameters that may be of special interest.

3. Use the trained model to perform inference.  This example contains two
inference examples: one for inference on the validation set, already in
Zarr format, and a second example for inference directly on .vtp files.

The following sections contain further details on the training and inference
recipe.

## Model Training

To train the model, first we compute normalization factors on the dataset to
make the predictive quantities output in a well defined range.  The include
script, `compute_normalizations.py`, will compute either the normalization
factors.  Once run, it should save to an output file similar to
"surface_fields_normalization.npz".  This will get loaded during training.
Check the training script to ensure the right path is used for your normalization
factors - it's not a configuration parameter but directly encoded in the script.

> By default, the normalization sets the mean to 0.0 and std to 1.0 of all labels
> in the dataset, computing the mean across the train dataset.  You could adapt
> this to a different normalization, however take care to update both the
> preprocessing as well as inference scripts.  Min/Max is another popular strategy.

To configure your training run, use `hydra` and `conf/train_surface.yaml`.  The
config contains sections for the model, data, optimizer, and training settings.
For details on the model parameters, see the API for `physicsnemo.models.transolver`.
The data is processed with a custom Zarr dataloader, designed to use zarr 3.0 and
`zarrs` rust implementation for an optimized Codec.  It also uses python's `threading`
module to open parallel reads of multiple zarr keys.  You can control the number
of parallel python threads via `data.max_workers`.

Additionally, the Zarr dataloader optimizes CPU->GPU transfers by directly
allocating pinned memory on the CPU, reading the Zarr data into that
memory buffer via a 0-copy to numpy, and moving the data to GPU via a separate
stream with non-blocking transfers.  In short: you can completely overlap IO
and GPU processing as long as the IO file system can provide the next data example
fast enough.  In reality, the IO latency has some variance but is not a bottleneck.

You can disable memory pinning with `data.pin_memory=False`.  Further, to fit
the training into memory, you can apply on-the-fly downsampling to the data
with `data.resolution=N`, where `N` is how many points per GPU to use.  This dataloader
will yield the full data examples in shapes of `[1, K, f]` where `K` is the resolution
of the mesh, and `f` is the feature space (3 for points, normals, etc.  4 for surface
fields).  Downsampling happens in the preprocessing pipeline.

> The pipeline has the ability to optimally load data from disk into `physicsnemo.ShardTensor`
> for domain parallelism - however the model support is still in development.

During training, the configuration uses the OneCycle learning rate (similar to the
original Transolver publication), and float32 format.  The scheduler and learning rate
may be configured - note that the scheduler is updated every training step.  For
schedulers that update every epoch, modification of the training script may be required.

### Training Precision

Transolver, as a transformer-like architecture, has support for NVIDIA's
[TransformerEngine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
built in.  You can enable/disable the transformer engine path in the model with
`model.use_te=[True | False]`.  Available precisions for training with `transformer_engine`
are `training.precision=["float32" | "float16" | "bfloat16" | "float8" ]`.  In `float8`
precision, the TransformerEngine Hybrid recipe is used for casting weights and inputs
in the forward and backwards passes.  For more details on `float8` precision, see
the fp8 guide from
[TransformerEngine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).
When using fp8, the training script will automatically pad and unpad the input and output,
respectively, to use the fp8 hardware correctly.

> **Float8** precisions are only available on GPUs with fp8 tensorcore support, such
> as Hopper, Blackwell, Ada Lovelace, and others.

### Other Configuration Settings

Several other important configuration settings are available:

- `training.compile` will use `torch.compile` for optimized performance.  It is not
compatible with `transformer_engine` (`model.use_te=True`).  If TransformerEngine is
not used, and half precision is, `torch.compile` is recommended for improved performance.
- `training.num_epochs` controls the total number of epochs used during training.
- `training.save_interval` will dictate how often the model weights and training
tools are checkpointed.

> **Note** Like other parameters of the model, changing the value of `model.use_te`
> will make checkpoints incompatible.

The training script supports data-parallel training via PyTorch DDP.  In a future
update, we may enable domain parallelism via FSDP and ShardTensor.

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

## Dataset Inference

There are two scripts provided as inference examples - it's expected that every user's
inference workloads are different, so these aim to cover common scenarios as examples.

First, the validation dataset in Zarr format can be loaded, processed, and the L2
metrics summarized in `inference_on_zarr.py`.  Alternatively, the model can be used
directly on `.vtp` or `.stl` files as shown in `inference_on_vtp.py`.  Note that the
script contains several parameters from the DrivaerML dataset as hardcoded variable
names: `CpMeanTrim`, `pMeanTrim`, `wallShearStressMeanTrim`, which are used to
compute the L2 metrics on the inference outputs.

In `inference_on_zarr.py`, the dataset examples are downsampled and preprocessed
exactly as in the training script.  In `inference_on_vtp.py`, however, the entire
mesh is processed.  To enable the mesh to fit into GPU memory, the mesh is chunked
into pieces that are then processed, and recombined to form the prediction on the
entire mesh.  The outputs are then saved to .vtp files for downstream analysis.

## Future work

The Transolver model is a promising, transformer-based model that produces high
quality predictions for CFD surrogate simulations.  In the future, we may update
the example to include domain parallelism and Transolver++ enhancements,
as well as volumetric data examples.  If you
have issues, requests, or other items please feel free to open an issue and discuss!
