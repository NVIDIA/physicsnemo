<!-- markdownlint-disable -->
# `Transolver` and `Typhon` for External Aerodynamics on Irregular Meshes

This example is an end to end training recipe for two models, both of which can
be run on surface or volume data.

1. `Transolver` is a high-performance surrogate model for CFD solvers. The Transolver model
adapts the Attention mechanism, encouraging the learning of meaningful representations.
In each PhysicsAttention layer, input points are projected onto state vectors through
learnable transformations and weights. These transformations are then used to compute
self-attention among all state vectors, and the same weights are reused to project
states back to each input point.

2. `Typhon` is an extension of the PhysicsAttention of Transolver with geometric
or global state enhancements.  We call this layer "Geometry-Aware Latent Embeddings",
or `GALE`, and the model - built from sequential `GALE`
layers - is called `typhon` as a reference to the mythological Greek god of storms.

As `typhon` is an extension to the Transolver PhysicsAttention mechanism, the training recipes
for these two models are integrated into one example.  You may train either model
on surface or volume data, as described below.  A publication for the `typhon`
will be released soon.

## External Aerodynamics CFD Example: Overview

This directory contains the essential components for training and evaluating a
model tailored to external aerodynamics CFD problems. Two models are supported,
because they are so closely related: `Transolver` and `Typhon`.  

By stacking multiple PhysicsAttention layers, the `Transolver` model learns to map from
the functional input space to the output space with high fidelity. The PhysicsNeMo
implementation closely follows the original Transolver architecture
([https://github.com/thuml/Transolver](https://github.com/thuml/Transolver)), but
introduces modifications for improved numerical stability and compatibility with NVIDIA
TransformerEngine.

The training example for Transolver uses the [DrivaerML dataset](https://caemldatasets.org/drivaerml/).

`Typhon` is an extension to Transolver that particularly focuses on geometrical encodings
and enabling each attention layer to leverage the geometrical and global properties
of the system.  As a concrete example, in this example we are training external
aerodynamics surrogate models for automobiles.  `Transolver` takes as input
a point cloud on the surface or surrounding the surface, and iteratively processes
it with PhysicsAttention - and produces excellent results.  

`Typhon` also takes as inputs the STL mesh of the car, and maps it to the same latent space
that PhysicsAttention layers use.  This geometrical state is combined with the
self-attention state via cross-attention, allowing each attention layer to incorporate
self-attention between successive layers as well as attend to the overall geometry
of the problem, at every stage.

## Requirements

Transolver can use TransformerEngine from NVIDIA, as well as tensorstore (for IO),
zarr, einops and a few other python packages.  Install them with `pip install -r requirements.txt`
as well as physicsnemo 25.11 or higher.  The `typhon` model is a prerelease model
and not available in 25.11 - please install physicsnemo from source to use it.

## Using Transolver and Typhon for External Aerodynamics

1. Prepare the Dataset.  Both models uses the same Zarr outputs as other models with DrivaerML.
`PhysicsNeMo` has a related project to help with data processing, called [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator).
Using `PhysicsNeMo-Curator`, the data needed to train can be setup easily.
Please refer to [these instructions on getting started](https://github.com/NVIDIA/physicsnemo-curator?tab=readme-ov-file#what-is-physicsnemo-curator)
with `PhysicsNeMo-Curator`.  For specifics of preparing the dataset for this example,
see the [download](https://github.com/NVIDIA/physicsnemo-curator/blob/main/examples/external_aerodynamics/domino/README.md#download-drivaerml-dataset)
and [preprocessing](https://github.com/NVIDIA/physicsnemo-curator/blob/main/examples/external_aerodynamics/domino/README.md)
instructions from `physicsnemo-curator`.  Users should apply the
preprocessing steps locally to produce `zarr` output files.

2. Train your model.  The model and training configuration is configured with
`hydra`, and four configurations are available: [`transolver`, `typhon`] x [surface, volume].
Find configurations in `src/conf`, where you can control both network properties
and training properties. See below for an overview and explanation of key
parameters that may be of special interest.

3. Use the trained model to perform inference.  This example contains two
inference examples: one for inference on the validation set, already in
Zarr format.  The `.vtp` inference pipeline is being updated to accomodate both models.

The following sections contain further details on the training and inference
recipe.

## Model Training

To train the model, first we compute normalization factors on the dataset to
make the predictive quantities output in a well-defined range. The included
script, `compute_normalizations.py`, will compute the normalization
factors.  Once run, it should save to an output file similar to
"surface_fields_normalization.npz".  This will get loaded during training.
The normalization file location can be configured via `data.normalization_dir`
in the training configuration (defaults to current directory).

> By default, the normalization sets the mean to 0.0 and std to 1.0 of all labels
> in the dataset, computing the mean across the train dataset.  You could adapt
> this to a different normalization, however take care to update both the
> preprocessing as well as inference scripts.  Min/Max is another popular strategy.

To configure your training run, use `hydra`.  The
config contains sections for the model, data, optimizer, and training settings.
For details on the model parameters, see the API for `physicsnemo.models.transolver`.

To fit the training into memory, you can apply on-the-fly downsampling to the data
with `data.resolution=N`, where `N` is how many points per GPU to use.  This dataloader
will yield the full data examples in shapes of `[1, K, f]` where `K` is the resolution
of the mesh, and `f` is the feature space (3 for points, normals, etc.  4 for surface
fields).  Downsampling happens in the preprocessing pipeline.

During training, the configuration uses a flat learning rate that decays every 100
epochs, and bfloat16 format by default.  The scheduler and learning rate
may be configured.  

The Optimizer for this training is the `Muon` optimizer - available only in
`pytorch>=2.9.0`.  While not strictly required, we have found the `muon` optimizer
performs substantially better on these architectures than standard `AdamW` and
a oneCycle schedule.

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

- `checkpoint_dir` sets the directory for saving model checkpoints (defaults to `output_dir`
if not specified), allowing separation of checkpoints from other outputs.
- `compile` will use `torch.compile` for optimized performance.  It is not
compatible with `transformer_engine` (`model.use_te=True`).  If TransformerEngine is
not used, and half precision is, `torch.compile` is recommended for improved performance.
- `training.num_epochs` controls the total number of epochs used during training.
- `training.save_interval` will dictate how often the model weights and training
tools are checkpointed.

> **Note** Like other parameters of the model, changing the value of `model.use_te`
> will make checkpoints incompatible.

The training script supports data-parallel training via PyTorch DDP.  In a future
update, we may enable domain parallelism via FSDP and ShardTensor.

The script can be launched on a single GPU with, for example,

```bash
python train.py --config-name transolver_surface
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

<!-- There are two scripts provided as inference examples - it's expected that every user's
inference workloads are different, so these aim to cover common scenarios as examples. -->

The validation dataset in Zarr format can be loaded, processed, and the L2
metrics summarized in `inference_on_zarr.py`.  For surface data, this script will also
compute the drag and lift coefficients and the R^2 correlation of the predictions.

To ensure correct calculation of drag and lift, and accurate overall metrics,
the inference script will chunk a full-resolution training example into batches,
and stitch the outputs together at the end.  Output will appear as a table
with all metrics for that mode, for example:

```
|   Batch |   Loss |   L2 Pressure |   L2 Shear X |   L2 Shear Y |   L2 Shear Z |   Predicted Drag Coefficient |   Pred Lift Coefficient |   True Drag Coefficient |   True Lift Coefficient |   Elapsed (s) |
|---------|--------|---------------|--------------|--------------|--------------|------------------------------|-------------------------|-------------------------|-------------------------|---------------|
|       0 | 0.0284 |        0.0583 |       0.0904 |       0.1013 |       0.1159 |                       3.8533 |                  3.5871 |                  3.9506 |                  3.4867 |       11.2388 |
|       1 | 0.023  |        0.0558 |       0.0758 |       0.0985 |       0.1056 |                       3.9865 |                  2.0918 |                  3.9827 |                  2.1996 |       10.2538 |
|       2 | 0.0457 |        0.0726 |       0.106  |       0.12   |       0.1801 |                       3.9847 |                  1.9193 |                  3.8824 |                  1.8598 |        9.859  |
|       3 | 0.045  |        0.0675 |       0.1098 |       0.1252 |       0.1391 |                       6.4476 |                  3.03   |                  6.3828 |                  2.9734 |       11.6881 |
|       4 | 0.0367 |        0.0624 |       0.1068 |       0.1152 |       0.1263 |                       4.6706 |                  2.2905 |                  4.6637 |                  2.2301 |       13.5494 |
|       5 | 0.0228 |        0.0499 |       0.0785 |       0.0941 |       0.0981 |                       6.1097 |                  0.7497 |                  6.1664 |                  0.7472 |       12.4388 |
|       6 | 0.0285 |        0.0589 |       0.0909 |       0.1059 |       0.1312 |                       3.9335 |                  0.8309 |                  3.9136 |                  0.8324 |        8.8262 |
|       7 | 0.0376 |        0.0717 |       0.1095 |       0.1236 |       0.1276 |                       4.7873 |                  1.9045 |                  4.8402 |                  2.1894 |       11.8797 |
|       8 | 0.0284 |        0.0548 |       0.0863 |       0.107  |       0.1215 |                       4.229  |                  1.1434 |                  4.4872 |                  1.0741 |       10.7116 |
|       9 | 0.0461 |        0.0767 |       0.1125 |       0.1246 |       0.139  |                       5.1331 |                  1.2558 |                  5.0711 |                  1.2379 |       12.562  |
|      10 | 0.0536 |        0.0849 |       0.1178 |       0.129  |       0.1548 |                       5.0147 |                  3.5343 |                  5.1289 |                  3.4116 |       10.4503 |
|      11 | 0.0333 |        0.0634 |       0.0965 |       0.1104 |       0.1147 |                       6.5021 |                  2.64   |                  6.4    |                  2.7209 |       13.1643 |
|      12 | 0.0238 |        0.0537 |       0.0804 |       0.0958 |       0.1032 |                       5.8751 |                  1.8963 |                  5.945  |                  1.7916 |       10.5001 |
|      13 | 0.0343 |        0.0651 |       0.1027 |       0.1093 |       0.1278 |                       5.562  |                  0.994  |                  5.5422 |                  1.0268 |        9.7037 |
|      14 | 0.0329 |        0.0717 |       0.0938 |       0.1113 |       0.124  |                       5.1604 |                  2.535  |                  5.3534 |                  2.6942 |       10.454  |
|      15 | 0.0231 |        0.0529 |       0.0807 |       0.1003 |       0.1121 |                       4.646  |                  2.0366 |                  4.701  |                  1.9473 |       10.0764 |
|      16 | 0.0311 |        0.0575 |       0.1021 |       0.1079 |       0.1166 |                       5.6578 |                  1.6703 |                  5.3935 |                  1.7436 |       12.5273 |
|      17 | 0.0284 |        0.0629 |       0.0897 |       0.1079 |       0.1172 |                       5.1796 |                  1.5146 |                  5.3012 |                  1.4919 |       11.4887 |
|      18 | 0.0302 |        0.0668 |       0.0929 |       0.1106 |       0.1157 |                       5.9201 |                  1.0449 |                  5.9403 |                  0.8958 |       11.4593 |
|      19 | 0.0248 |        0.054  |       0.0962 |       0.1046 |       0.1182 |                       5.3302 |                  2.3861 |                  5.3644 |                  2.4404 |       11.6885 |
|      20 | 0.0232 |        0.0537 |       0.0834 |       0.0981 |       0.1009 |                       5.2209 |                  2.1628 |                  5.2129 |                  2.1078 |       11.1264 |
|      21 | 0.0237 |        0.0609 |       0.0793 |       0.1    |       0.0977 |                       5.5532 |                  1.7551 |                  5.6004 |                  1.6219 |       12.1105 |
|      22 | 0.0252 |        0.0568 |       0.0813 |       0.0996 |       0.103  |                       4.8141 |                  2.8054 |                  4.7433 |                  2.8088 |       11.5489 |
|      23 | 0.0327 |        0.0627 |       0.0911 |       0.108  |       0.1273 |                       5.9474 |                 -0.3203 |                  5.8594 |                 -0.2171 |       12.2077 |
|      24 | 0.0313 |        0.0611 |       0.0923 |       0.1085 |       0.1096 |                       5.565  |                  1.667  |                  5.4902 |                  1.9746 |       14.0795 |
|      25 | 0.0357 |        0.0752 |       0.1021 |       0.1211 |       0.1489 |                       4.4083 |                  1.6014 |                  4.2135 |                  1.7296 |        9.5865 |
|      26 | 0.0321 |        0.0703 |       0.0933 |       0.1098 |       0.1208 |                       5.6937 |                  3.166  |                  5.6328 |                  3.6129 |       13.1783 |
|      27 | 0.0247 |        0.0575 |       0.0845 |       0.1051 |       0.1129 |                       4.1762 |                  1.453  |                  4.2148 |                  1.4453 |       11.8495 |
|      28 | 0.0318 |        0.0609 |       0.0965 |       0.1104 |       0.1173 |                       5.2632 |                  3.2019 |                  5.2519 |                  3.1841 |       13.0749 |
|      29 | 0.0368 |        0.061  |       0.0992 |       0.115  |       0.1278 |                       6.585  |                  1.755  |                  6.4859 |                  1.6068 |       12.7777 |
|      30 | 0.0289 |        0.0577 |       0.0871 |       0.1062 |       0.1169 |                       5.0937 |                  3.2484 |                  5.3222 |                  3.2723 |       11.8586 |
|      31 | 0.0369 |        0.0671 |       0.0994 |       0.1129 |       0.1618 |                       5.5144 |                  2.3549 |                  5.5315 |                  2.3749 |       10.1762 |
|      32 | 0.0356 |        0.0753 |       0.0981 |       0.1176 |       0.1373 |                       5.5471 |                  0.2552 |                  5.5173 |                  0.4556 |        8.9759 |
|      33 | 0.02   |        0.0478 |       0.0775 |       0.0956 |       0.0944 |                       4.6799 |                  1.9003 |                  4.7444 |                  1.9062 |       14.5507 |
|      34 | 0.0226 |        0.0487 |       0.0789 |       0.0922 |       0.0963 |                       5.6734 |                  3.4928 |                  5.7506 |                  3.5023 |       13.0373 |
|      35 | 0.0213 |        0.0512 |       0.0804 |       0.0959 |       0.1018 |                       6.0567 |                  0.9755 |                  6.0311 |                  0.8335 |       12.5048 |
|      36 | 0.0273 |        0.0548 |       0.0844 |       0.1004 |       0.1263 |                       5.1413 |                  1.308  |                  5.2466 |                  1.2221 |        9.8688 |
|      37 | 0.0325 |        0.0621 |       0.0895 |       0.1121 |       0.1271 |                       3.2417 |                  0.9704 |                  3.3774 |                  1.0713 |        9.3579 |
|      38 | 0.043  |        0.0661 |       0.1029 |       0.1173 |       0.1312 |                       6.1339 |                  3.4028 |                  6.1527 |                  3.423  |       11.6961 |
|      39 | 0.0279 |        0.0573 |       0.0905 |       0.1034 |       0.1118 |                       6.7051 |                  1.803  |                  6.6982 |                  1.7571 |       12.1701 |
|      40 | 0.0453 |        0.07   |       0.0986 |       0.1161 |       0.1526 |                       6.9221 |                  2.9829 |                  6.734  |                  3.0083 |       12.1203 |
|      41 | 0.0303 |        0.0638 |       0.0931 |       0.1113 |       0.1403 |                       4.4597 |                  0.6138 |                  4.3089 |                  0.772  |        9.2097 |
|      42 | 0.0219 |        0.0505 |       0.0802 |       0.0988 |       0.0977 |                       6.1306 |                  1.9262 |                  6.1526 |                  1.6924 |       12.1102 |
|      43 | 0.0274 |        0.0604 |       0.0823 |       0.1083 |       0.1113 |                       3.8352 |                  2.3836 |                  3.9082 |                  2.5251 |       11.412  |
|      44 | 0.0338 |        0.0694 |       0.0923 |       0.1073 |       0.1273 |                       6.3879 |                  0.8005 |                  6.1468 |                  0.8924 |       10.3299 |
|      45 | 0.0271 |        0.0553 |       0.0888 |       0.1016 |       0.1065 |                       7.5199 |                  1.7131 |                  7.4145 |                  1.6298 |       12.9432 |
|      46 | 0.0258 |        0.0526 |       0.0834 |       0.0984 |       0.117  |                       4.7368 |                  0.6839 |                  4.7965 |                  0.7926 |        9.1506 |
|      47 | 0.0295 |        0.0564 |       0.0896 |       0.1123 |       0.1127 |                       5.1667 |                  2.7415 |                  5.2693 |                  2.779  |       12.7713 |
[2025-11-19 07:02:38,387][training][INFO] - R2 score for lift: 0.9807
[2025-11-19 07:02:38,387][training][INFO] - R2 score for drag: 0.9844
[2025-11-19 07:02:38,387][training][INFO] - Summary:
| Batch   |   Loss |   L2 Pressure |   L2 Shear X |   L2 Shear Y |   L2 Shear Z |   Predicted Drag Coefficient |   Pred Lift Coefficient |   True Drag Coefficient |   True Lift Coefficient |   Elapsed (s) |
|---------|--------|---------------|--------------|--------------|--------------|------------------------------|-------------------------|-------------------------|-------------------------|---------------|
| Mean    | 0.0311 |        0.0614 |       0.0921 |        0.108 |       0.1214 |                       5.2949 |                  1.9137 |                  5.2962 |                  1.9329 |       11.4647 |
```

  <!-- Alternatively, the model can be used
directly on `.vtp` or `.stl` files as shown in `inference_on_vtp.py`.  Note that the
script contains several parameters from the DrivaerML dataset as hardcoded variable
names: `CpMeanTrim`, `pMeanTrim`, `wallShearStressMeanTrim`, which are used to
compute the L2 metrics on the inference outputs. -->

<!-- In `inference_on_zarr.py`, the dataset examples are downsampled and preprocessed
exactly as in the training script.  In `inference_on_vtp.py`, however, the entire
mesh is processed.  To enable the mesh to fit into GPU memory, the mesh is chunked
into pieces that are then processed, and recombined to form the prediction on the
entire mesh.  The outputs are then saved to .vtp files for downstream analysis. -->

## Transolver++

Transolver++ is supported in both models with the `plus` flag to the model.  In
our experiments, we did not see gains and have focused on `transolver` and `typhon`.
You are welcome to try it and share your results with us on GitHub!
