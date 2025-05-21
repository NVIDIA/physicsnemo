# DoMINO: Decomposable Multi-scale Iterative Neural Operator for External Aerodynamics

DoMINO is a local, multi-scale, point-cloud based model architecture to model large-scale
physics problems such as external aerodynamics. The DoMINO model architecture takes STL
geometries as input and evaluates flow quantities such as pressure and
wall shear stress on the surface of the car as well as velocity fields and pressure
in the volume around it. The DoMINO architecture is designed to be a fast, accurate
and scalable surrogate model for large-scale industrial simulations.

DoMINO uses local geometric information to predict solutions on discrete points. First,
a global geometry encoding is learnt from point clouds using a multi-scale, iterative
approach. The geometry representation takes into account both short- and long-range
depdencies that are typically encountered in elliptic PDEs. Additional information
as signed distance field (SDF), positional encoding are used to enrich the global encoding.
Next, discrete points are randomly sampled, a sub-region is constructed around each point
and the local geometry encoding is extracted in this region from the global encoding.
The local geometry information is learnt using dynamic point convolution kernels.
Finally, a computational stencil is constructed dynamically around each discrete point
by sampling random neighboring points within the same sub-region. The local-geometry
encoding and the computational stencil are aggregrated to predict the solutions on the
discrete points.

A preprint describing additional details about the model architecture can be found here
[paper](https://arxiv.org/abs/2501.13350).

## Dataset

In this example, the DoMINO model is trained using DrivAerML dataset from the
[CAE ML Dataset collection](https://caemldatasets.org/drivaerml/).
This high-fidelity, open-source (CC-BY-SA) public dataset is specifically designed
for automotive aerodynamics research. It comprises 500 parametrically morphed variants
of the widely utilized DrivAer notchback generic vehicle. Mesh generation and scale-resolving
computational fluid dynamics (CFD) simulations were executed using consistent and validated
automatic workflows that represent the industrial state-of-the-art. Geometries and comprehensive
aerodynamic data are published in open-source formats. For more technical details about this
dataset, please refer to their [paper](https://arxiv.org/pdf/2408.11969).

## Training the DoMINO model

To train and test the DoMINO model on AWS dataset, follow these steps:

1. Download the DrivAer ML dataset using the provided `download_aws_dataset.sh` script or
   using the [Hugging Face repo](https://huggingface.co/datasets/neashton/drivaerml).

2. Specify the configuration settings in `conf/config.yaml`.

3. Run `process_data.py`. This will process VTP/VTU files and save them as npy for faster
 processing in DoMINO datapipe. Modify data_processor key in config file. Additionally, run
 `cache_data.py` to save outputs of DoMINO datapipe in the `.npy` files. The DoMINO datapipe
 is set up to calculate Signed Distance Field and Nearest Neighbor interpolations
 on-the-fly during training. Caching will save these as a preprocessing step and should
 be used in cases where the STL surface meshes are upwards of 30 million cells.
 The final processed dataset should be divided and saved into 2 directories, for training
 and validation. Specify these directories in `conf/config.yaml`.

4. Run `train.py` to start the training. Modify data, train and model keys in config file.
  If using cached data then use `conf/cached.yaml` instead of `conf/config.yaml`.

5. Run `test.py` to test on `.vtp` / `.vtu`. Predictions are written to the same file.
  Modify eval key in config file to specify checkpoint, input and output directory.
  Important to note that the data used for testing is in the raw simulation format and
  should not be processed to `.npy`.

6. Download the validation results (saved in form of point clouds in `.vtp` / `.vtu` format),
   and visualize in Paraview.

### Training with Domain Parallelism

DoMINO has support for training and inference using domain parallelism in PhysicsNeMo,
via the `ShardTensor` mechanisms and pytorch's FSDP tools.  `ShardTensor`, built on
PyTorch's `DTensor` object, is a domain-parallel-aware tensor that can live on multiple
GPUs and perform operations in a numerically consistent way.  For more information
about the techniques of domain parallelism and `ShardTensor`, refer to PhysicsNeMo
tutorials such as [`ShardTensor`](shard_tensor_tutorial.html).

In DoMINO specifically, domain parallelism has been abled in two ways, which
can be used concurrently or separately.  First, the input sampled volumetric
and surface points can be sharded to accomodate higher resolution point sampling
Second, the latent space of the model - typically a regularlized grid - can be
sharded to reduce computational complexity of the latent processing.  When training
with sharded models in DoMINO, the primary objective is to enable higher
resolution inputs and larger latent spaces without sacrificing substantial compute time.

When configuring DoMINO for sharded training, adjust the following parameters
from `src/conf/config.yaml`:

```yaml
domain_parallelism:
  domain_size: 2
  shard_grid: True
  shard_points: True
```

The `domain_size` represents the number of GPUs used for each batch - setting
`domain_size: 1` is not advised since that is the standard training regime,
but with extra overhead.  `shard_grid` and `shard_points` will enable domain
parallelism over the latent space and input/output points, respectively.

Please see `src/train_sharded.py` for more details regarding the changes
from the standard training script required for domain parallel DoMINO training.

As one last note regarding domain-parallel training: in the phase of the DoMINO
where the output solutions are calculated, the model can used two different
techniques (numerically identical) to calculate the output.  Due to the
overhead of potential communication at each operation, it's recommended to
use the `one-loop` mode with `model.solution_calculation_mode` when doing
sharded training.  This technique launches vectorized kernels with less
launch overhead at the cost of more memory use.  For non-sharded
training, the `two-loop` setting is more optimal. The difference in `one-loop`
or `two-loop` is purely computational, not algorithmic.

## Retraining recipe for DoMINO model

To enable retraining the DoMINO model from a pre-trained checkpoint, follow the steps:

1. Add the pre-trained checkpoints in the resume_dir defined in `conf/config.yaml`.

2. Add the volume and surface scaling factors to the output dir defined in  `conf/config.yaml`.

3. Run `retraining.py` for specified number of epochs to retrain model at a small
 learning rate starting from checkpoint.

4. Run `test.py` to test on `.vtp` / `.vtu`. Predictions are written to the same file.
 Modify eval key in config file to specify checkpoint, input and output directory.

5. Download the validation results (saved in form of point clouds in `.vtp` / `.vtu` format),
   and visualize in Paraview.

## DoMINO model inference on STLs

The DoMINO model can be evaluated directly on unknown STLs using the pre-trained
 checkpoint. Follow the steps outlined below:

1. Run the `inference_on_stl.py` script to perform inference on an STL.

2. Specify the STL paths, velocity inlets, stencil size and model checkpoint
 path in the script.

3. The volume predictions are carried out on points sampled in a bounding box around STL.

4. The surface predictions are carried out on the STL surface. The drag and lift
 accuracy will depend on the resolution of the STL.

## Guidelines for training DoMINO model

1. The DoMINO model allows for training both volume and surface fields using a single model
 but currently the recommendation is to train the volume and surface models separately. This
  can be controlled through the config file.

2. MSE loss for both volume and surface model gives the best results.

3. The surface and volume variable names can change but currently the code only
 supports the variables in that specific order. For example, Pressure, wall-shear
  and turb-visc for surface and velocity, pressure and turb-visc for volume.

4. Bounding box is configurable and will depend on the usecase. The presets are
 suitable for the DriveAer-ML dataset.

The DoMINO model architecture is used to support the
[Real Time Digital Twin Blueprint](https://github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation)
and the
[DoMINO-Automotive-Aero NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero).

Some of the results are shown below.

![Results from DoMINO for RTWT SC demo](../../../../docs/img/domino_result_rtwt.jpg)

## References

1. [DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations](https://arxiv.org/abs/2501.13350)
