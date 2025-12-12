<!-- markdownlint-disable -->
# `Transolver` for External Aerodynamics on Irregular Meshes

This example is an end to end training recipe for the `Transolver` model, which can
be run on surface or volume data.

`Transolver` is a high-performance surrogate model for CFD solvers. The Transolver model
adapts the Attention mechanism, encouraging the learning of meaningful representations.
In each PhysicsAttention layer, input points are projected onto state vectors through
learnable transformations and weights. These transformations are then used to compute
self-attention among all state vectors, and the same weights are reused to project
states back to each input point.

## External Aerodynamics CFD Example: Overview

This directory contains the essential components for training and evaluating a
model tailored to external aerodynamics CFD problems built on `Transolver`.

By stacking multiple PhysicsAttention layers, the `Transolver` model learns to map from
the functional input space to the output space with high fidelity. The PhysicsNeMo
implementation closely follows the original Transolver architecture
([https://github.com/thuml/Transolver](https://github.com/thuml/Transolver)), but
introduces modifications for improved numerical stability and compatibility with NVIDIA
TransformerEngine.

The training example for Transolver uses the [DrivaerML dataset](https://caemldatasets.org/drivaerml/).

As a concrete example, we are training external aerodynamics surrogate models for automobiles.
`Transolver` takes as input a point cloud on the surface or surrounding the surface,
iteratively processing it with PhysicsAttention to produce high-fidelity predictions.

## Requirements

Transolver can use TransformerEngine from NVIDIA, as well as tensorstore (for IO),
zarr, einops and a few other python packages.  Install them with `pip install -r requirements.txt`
as well as physicsnemo 25.11 or higher.

## Using Transolver for External Aerodynamics

1. Prepare the Dataset.  Transolver uses the same Zarr outputs as other models with DrivaerML.
`PhysicsNeMo` has a related project to help with data processing, called [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator).
Using `PhysicsNeMo-Curator`, the data needed to train can be setup easily.
Please refer to [these instructions on getting started](https://github.com/NVIDIA/physicsnemo-curator?tab=readme-ov-file#what-is-physicsnemo-curator)
with `PhysicsNeMo-Curator`.  For specifics of preparing the dataset for this example,
see the [download](https://github.com/NVIDIA/physicsnemo-curator/blob/main/examples/external_aerodynamics/README.md#download-drivaerml-dataset)
and [preprocessing](https://github.com/NVIDIA/physicsnemo-curator/blob/main/examples/external_aerodynamics/README.md)
instructions from `physicsnemo-curator`.  Users should apply the
preprocessing steps locally to produce `zarr` output files.

2. Train your model.  The model and training configuration is configured with
`hydra`, and two configurations are available: `transolver_surface` and `transolver_volume`.
Find configurations in `src/conf`, where you can control both network properties
and training properties. See below for an overview and explanation of key
parameters that may be of special interest.

3. Use the trained model to perform inference.  This example contains two
inference examples: one for inference on the validation set, already in
Zarr format.  The `.vtp` inference pipeline is being updated to accomodate Transolver.

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

The validation dataset in Zarr format can be loaded, processed, and the L2
metrics summarized in `inference_on_zarr.py`.  For surface data, this script will also
compute the drag and lift coefficients and the R^2 correlation of the predictions.

To run inference on surface data, it's necessary to add a line to your launch command:

```
python src/inference_on_zarr.py --config-name transolver_surface run_id=/path/to/model/

```

The `data.return_mesh_features` flag can also be set in the config file.  It is
disabled for training but necessary for inference.  The model path should be
the folder containing your saved checkpoints.


To ensure correct calculation of drag and lift, and accurate overall metrics,
the inference script will chunk a full-resolution training example into batches,
and stitch the outputs together at the end.  Output will appear as a table
with all metrics for that mode, for example:

```
|   Batch |   Loss |   L2 Pressure |   L2 Shear X |   L2 Shear Y |   L2 Shear Z |   Predicted Drag Coefficient |   Pred Lift Coefficient |   True Drag Coefficient |   True Lift Coefficient |   Elapsed (s) |
|---------|--------|---------------|--------------|--------------|--------------|------------------------------|-------------------------|-------------------------|-------------------------|---------------|
|       0 | 0.0188 |        0.0491 |       0.0799 |       0.1023 |       0.1174 |                      488.075 |                140.365  |                 475.534 |                135.944  |        8.1281 |
|       1 | 0.0144 |        0.045  |       0.0659 |       0.0955 |       0.107  |                      404.472 |                 21.8897 |                 406.484 |                 35.6202 |        0.7348 |
|       2 | 0.0239 |        0.0505 |       0.0835 |       0.1101 |       0.1592 |                      383.219 |                 41.973  |                 373.999 |                 43.7198 |        1.6722 |
|       3 | 0.0255 |        0.0526 |       0.088  |       0.1151 |       0.1305 |                      576.671 |                230.185  |                 579.655 |                210.01   |        1.4369 |
|       4 | 0.0214 |        0.0498 |       0.0849 |       0.109  |       0.1229 |                      451.478 |                -45.3076 |                 447.109 |                -36.7298 |        1.8973 |
|       5 | 0.0147 |        0.0402 |       0.0671 |       0.0923 |       0.0992 |                      419.76  |                -87.7945 |                 424.63  |                -83.8417 |        1.7255 |
|       6 | 0.0171 |        0.0463 |       0.0742 |       0.1016 |       0.126  |                      350.877 |                -32.1908 |                 338.721 |                -25.5008 |        1.3738 |
|       7 | 0.0248 |        0.0596 |       0.0989 |       0.123  |       0.1299 |                      420.122 |                -42.3073 |                 420.772 |                -16.9301 |        1.9126 |
|       8 | 0.0178 |        0.0453 |       0.0736 |       0.1021 |       0.118  |                      380.704 |                -90.6937 |                 374.134 |                -87.2395 |        1.8081 |
|       9 | 0.0297 |        0.0629 |       0.1004 |       0.1245 |       0.1418 |                      400.315 |               -149.927  |                 396.178 |               -147.33   |        1.6693 |
|      10 | 0.0303 |        0.0674 |       0.0978 |       0.1233 |       0.1455 |                      602.585 |                249.985  |                 588.987 |                237.999  |        1.6581 |
|      11 | 0.0188 |        0.0514 |       0.0772 |       0.1006 |       0.1114 |                      593.366 |                155.859  |                 590.833 |                167.067  |        1.6914 |
|      12 | 0.0147 |        0.0436 |       0.0681 |       0.0929 |       0.1009 |                      457.252 |                 77.7093 |                 449.866 |                 77.2836 |        1.734  |
|      13 | 0.0226 |        0.0529 |       0.0902 |       0.1092 |       0.1319 |                      374.561 |                -88.923  |                 372.675 |               -101.469  |        1.3918 |
|      14 | 0.0186 |        0.0591 |       0.0758 |       0.1056 |       0.1199 |                      516.445 |                275.197  |                 512.238 |                274.633  |        1.7587 |
|      15 | 0.0145 |        0.0443 |       0.0691 |       0.0974 |       0.1083 |                      397.664 |                 44.4129 |                 395.376 |                 31.417  |        1.6531 |
|      16 | 0.019  |        0.0502 |       0.0828 |       0.1028 |       0.1145 |                      502.079 |                 75.96   |                 501.056 |                 77.4457 |        1.6815 |
|      17 | 0.0155 |        0.0459 |       0.0721 |       0.1003 |       0.1064 |                      472.191 |                138.568  |                 460.808 |                139.42   |        1.7288 |
|      18 | 0.0186 |        0.0549 |       0.0783 |       0.1074 |       0.1162 |                      482.58  |                 37.7236 |                 482.344 |                 37.2805 |        1.7915 |
|      19 | 0.0148 |        0.0425 |       0.078  |       0.1004 |       0.113  |                      448.504 |                157.548  |                 446.845 |                173.68   |        1.8042 |
|      20 | 0.0144 |        0.0424 |       0.072  |       0.0946 |       0.0993 |                      500.781 |                 81.4317 |                 490.024 |                 85.8991 |        1.7812 |
|      21 | 0.0142 |        0.0462 |       0.0669 |       0.0983 |       0.0982 |                      483.057 |                134.258  |                 473.958 |                121.551  |        1.8255 |
|      22 | 0.0149 |        0.0432 |       0.0671 |       0.0964 |       0.1004 |                      510.518 |                162.651  |                 504.159 |                164.953  |        1.8021 |
|      23 | 0.0182 |        0.05   |       0.074  |       0.101  |       0.116  |                      388.014 |               -223.932  |                 393.797 |               -229.571  |        2.6297 |
|      24 | 0.0188 |        0.0486 |       0.0774 |       0.1049 |       0.1064 |                      477.557 |                -11.9395 |                 494.446 |                  7.5967 |        0.8668 |
|      25 | 0.0229 |        0.0608 |       0.0867 |       0.1211 |       0.1507 |                      348.804 |                  5.3412 |                 341.955 |                 30.8778 |        1.5065 |
|      26 | 0.019  |        0.0544 |       0.0814 |       0.1063 |       0.119  |                      467.791 |                170.149  |                 466.67  |                186.732  |        1.8434 |
|      27 | 0.0154 |        0.047  |       0.0734 |       0.1014 |       0.1102 |                      426.202 |                -78.8968 |                 417.572 |                -78.867  |        1.8177 |
|      28 | 0.0159 |        0.0455 |       0.0724 |       0.0983 |       0.1051 |                      523.8   |                165.693  |                 512.567 |                150.064  |        1.7851 |
|      29 | 0.0243 |        0.0498 |       0.0873 |       0.112  |       0.1309 |                      481.491 |                 55.202  |                 483.593 |                 59.5569 |        1.7285 |
|      30 | 0.021  |        0.054  |       0.0808 |       0.1097 |       0.1232 |                      508.089 |                200.01   |                 496.295 |                194.816  |        1.7602 |
|      31 | 0.0186 |        0.0479 |       0.0771 |       0.1047 |       0.1351 |                      422.298 |                 80.0045 |                 421.175 |                 97.6633 |        1.532  |
|      32 | 0.0205 |        0.0589 |       0.0793 |       0.1129 |       0.1308 |                      395.582 |                -12.36   |                 400.106 |                  6.3091 |        1.5378 |
|      33 | 0.0129 |        0.0396 |       0.0679 |       0.0923 |       0.0953 |                      431.082 |                  7.8286 |                 428.801 |                  8.6182 |        1.8789 |
|      34 | 0.0144 |        0.0412 |       0.0662 |       0.0893 |       0.0979 |                      530.599 |                179.193  |                 532.033 |                158.92   |        1.8429 |
|      35 | 0.0139 |        0.0424 |       0.0716 |       0.0945 |       0.1006 |                      430.982 |                  7.3476 |                 428.805 |                 -4.3425 |        1.711  |
|      36 | 0.0167 |        0.043  |       0.0702 |       0.0975 |       0.1217 |                      381.859 |                -45.0215 |                 376.432 |                -65.0582 |        1.4227 |
|      37 | 0.021  |        0.0516 |       0.0772 |       0.1106 |       0.1302 |                      348.402 |                -84.0741 |                 347.672 |                -69.1513 |        1.5184 |
|      38 | 0.029  |        0.0585 |       0.0895 |       0.1188 |       0.1347 |                      596.764 |                287.068  |                 586.433 |                236.509  |        1.6109 |
|      39 | 0.0176 |        0.0472 |       0.0758 |       0.1006 |       0.1115 |                      470.259 |                 25.2451 |                 468.965 |                 38.1292 |        1.7815 |
|      40 | 0.0309 |        0.0583 |       0.0827 |       0.1163 |       0.1649 |                      579.514 |                186.451  |                 587.644 |                177.782  |        1.6365 |
|      41 | 0.0188 |        0.0516 |       0.0776 |       0.1084 |       0.1369 |                      349.04  |               -106.107  |                 341.44  |                -94.3054 |        1.4013 |
|      42 | 0.014  |        0.0424 |       0.0673 |       0.0964 |       0.0977 |                      477.916 |                120.4    |                 474.075 |                116.718  |        1.8973 |
|      43 | 0.0171 |        0.0476 |       0.071  |       0.1054 |       0.1116 |                      423.233 |                 50.4327 |                 420.448 |                 69.2674 |        1.8893 |
|      44 | 0.0247 |        0.0613 |       0.0799 |       0.1171 |       0.141  |                      426.292 |                 -2.5913 |                 422.69  |                 20.4068 |        1.4871 |
|      45 | 0.0161 |        0.0431 |       0.0736 |       0.0959 |       0.1007 |                      538.835 |                 71.1159 |                 544.14  |                 89.5933 |        1.7929 |
|      46 | 0.017  |        0.0442 |       0.0722 |       0.0986 |       0.1175 |                      361.974 |               -136.836  |                 359.692 |               -151.266  |        1.4659 |
|      47 | 0.0186 |        0.046  |       0.0778 |       0.1076 |       0.1114 |                      502.144 |                 80.8261 |                 499.45  |                102.07   |        1.9431 |
[2025-12-01 08:19:42,350][training][INFO] - R2 score for lift: 0.9824
[2025-12-01 08:19:42,350][training][INFO] - R2 score for drag: 0.9904
[2025-12-01 08:19:42,351][training][INFO] - Summary:
| Batch   |   Loss |   L2 Pressure |   L2 Shear X |   L2 Shear Y |   L2 Shear Z |   Predicted Drag Coefficient |   Pred Lift Coefficient |   True Drag Coefficient |   True Lift Coefficient |   Elapsed (s) |
|---------|--------|---------------|--------------|--------------|--------------|------------------------------|-------------------------|-------------------------|-------------------------|---------------|
| Mean    | 0.0191 |        0.0496 |       0.0775 |       0.1047 |       0.1191 |                      456.371 |                 51.6484 |                 453.193 |                  53.624 |        1.8114 |
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

Transolver++ is supported with the `plus` flag to the model.  In
our experiments, we did not see gains, but you are welcome to try it and share
your results with us on GitHub!
