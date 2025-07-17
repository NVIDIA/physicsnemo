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

## Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

## Getting started with the DrivAerML example

### Configuration basics

DoMINO data processing, training and testing is managed through YAML configuration files
powered by Hydra. The base configuration file, `config.yaml` is located in `src/conf`
directory.

To select a specific configuration, use the `--config-name` option when running
the scripts.
You can modify configuration options in two ways:

1. **Direct Editing:** Modify the YAML files directly
2. **Command Line Override:** Use Hydra's `++` syntax to override settings at runtime

For example, to change the training epochs (controlled by `train.epochs`):

```bash
python train.py ++train.epochs=200  # Sets number of epochs to 200
```

This modular configuration system allows for flexible experimentation while
maintaining reproducibility.

#### Project logs

Save and track project logs, experiments, tensorboard files etc. by specifying a
project directory with `project.name`. Tag experiments with `expt`.

#### Data processing

The first step for running the DoMINO pipeline requires processing the raw data
(vtp, vtu and stl). The related configs can be set in the `data_processor` tab.
Also, specify the variable names used in the raw dataset and their types
in `variables.surface` and `variables.volume`.
For example, you can set the input directory for raw data using
`data_processor.input_dir` and output directory for processed files using
`data_processor.output_dir`.

#### Training

Specify the training and validation data paths, bounding box sizes etc. in the
`data` tab and the training configs such as epochs, batch size etc.
in the `train` tab.

#### Testing

The testing is directly carried out on raw files.
Specify the testing configs in the `test` tab.

### Dataset details

In this example, the DoMINO model is trained using DrivAerML dataset from the
[CAE ML Dataset collection](https://caemldatasets.org/drivaerml/).
This high-fidelity, open-source (CC-BY-SA) public dataset is specifically
designed for automotive aerodynamics research. It comprises 500 parametrically
morphed variants of the widely utilized DrivAer notchback generic vehicle.
Mesh generation and scale-resolving computational fluid dynamics (CFD) simulations
were executed using consistent and validated automatic workflows that represent
the industrial state-of-the-art. Geometries and comprehensive aerodynamic data
are published in open-source formats. For more technical details about this dataset,
please refer to their [paper](https://arxiv.org/pdf/2408.11969).

Download the DrivAer ML dataset using the provided `download_aws_dataset.sh`
script or using the [Hugging Face repo](https://huggingface.co/datasets/neashton/drivaerml).

### Data processing for DoMINO model

Each of the raw simulations files in the `vtp`, `vtu` and `stl` format need to be
processed and saved into `npy` files. The data processing script extracts minmal
information from these raw files such as STL mesh, surface mesh and fields,
volume point cloud and fields. Run `process_data.py` with the correct configurations
for kicking off data processing. Additionally, run `cache_data.py` to save outputs
of DoMINO datapipe in the `.npy` files. The DoMINO datapipe is set up to calculate
Signed Distance Field and Nearest Neighbor interpolations on-the-fly during
training. Caching will save these as a preprocessing step and should be used in
cases where the STL surface meshes are upwards of 30 million cells.
Data processing is parallelized and takes a couple of hours to write all the
processed files.

The final processed dataset should be divided and saved into 2 directories,
for training and validation.

### Training the DoMINO model

To train and test the DoMINO model on AWS dataset, follow these steps:

1. Specify the configuration settings in `conf/config.yaml`.

2. Run `train.py` to start the training. Modify data, train and model keys in config file.
  If using cached data then use `conf/cached.yaml` instead of `conf/config.yaml`.

3. Run `test.py` to test on `.vtp` / `.vtu`. Predictions are written to the same file.
  Modify eval key in config file to specify checkpoint, input and output directory.
  Important to note that the data used for testing is in the raw simulation format and
  should not be processed to `.npy`.

4. Download the validation results (saved in form of point clouds in `.vtp` / `.vtu` format),
   and visualize in Paraview.

**Training Guidelines:**

- Duration: A couple of days on a single node of H100 GPU
- Checkpointing: Automatically resumes from latest checkpoint if interrupted
- Multi-GPU Support: Compatible with `torchrun` or MPI for distributed training
- If the training crashes because of OOO, modify the points sampled in volume
  `model.volume_points_sample` and surface `model.volume_points_sample`
  to manage memory requirements for your GPU
- The DoMINO model allows for training both volume and surface fields using a
  single model but currently the recommendation is to train the volume and
  surface models separately. This can be controlled through the `conf/config.yaml`.
- MSE loss for both volume and surface model gives the best results.
- Bounding box is configurable and will depend on the usecase.
  The presets are suitable for the DriveAer-ML dataset.

### Training with Domain Parallelism

DoMINO has support for training and inference using domain parallelism in PhysicsNeMo,
via the `ShardTensor` mechanisms and pytorch's FSDP tools.  `ShardTensor`, built on
PyTorch's `DTensor` object, is a domain-parallel-aware tensor that can live on multiple
GPUs and perform operations in a numerically consistent way.  For more information
about the techniques of domain parallelism and `ShardTensor`, refer to PhysicsNeMo
tutorials such as [`ShardTensor`](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.distributed.shardtensor.html).

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

### Retraining recipe for DoMINO model

To enable retraining the DoMINO model from a pre-trained checkpoint, follow the steps:

1. Add the pre-trained checkpoints in the resume_dir defined in `conf/config.yaml`.

2. Add the volume and surface scaling factors to the output dir defined in  `conf/config.yaml`.

3. Run `retraining.py` for specified number of epochs to retrain model at a small
 learning rate starting from checkpoint.

4. Run `test.py` to test on `.vtp` / `.vtu`. Predictions are written to the same file.
 Modify eval key in config file to specify checkpoint, input and output directory.

5. Download the validation results (saved in form of point clouds in `.vtp` / `.vtu` format),
   and visualize in Paraview.

### DoMINO model pipeline for inference on test samples

After training is completed, `test.py` script can be used to run inference on
test samples. Follow the below steps to run the `test.py`

1. Update the config in the `conf/config.yaml` under the `Testing data Configs`
   tab.

2. The test script is designed to run inference on the raw `.stl`, `.vtp` and
   `.vtu` files for each test sample. Use the same scaling parameters that
   were generated during the training. Typically this is `outputs/<project.name>/`,
   where `project.name` is as defined in the `config.yaml`. Update the
   `eval.scaling_param_path` accordingly.

3. Run the `test.py`. The test script can be run in parallel as well. Refer to
   the training guidelines for Multi-GPU. Note, for running `test.py` in parallel,
   the number of GPUs chosen must be <= the number of test samples.

### DoMINO model pipeline for inference on STLs

The DoMINO model can be evaluated directly on unknown STLs using the pre-trained
 checkpoint. Follow the steps outlined below:

1. Run the `inference_on_stl.py` script to perform inference on an STL.

2. Specify the STL paths, velocity inlets, stencil size and model checkpoint
 path in the script.

3. The volume predictions are carried out on points sampled in a bounding box around STL.

4. The surface predictions are carried out on the STL surface. The drag and lift
 accuracy will depend on the resolution of the STL.

### Incorporating multiple global simulation parameters for training/inference

DoMINO supports incorporating multiple global simulation parameters (such as inlet
velocity, air density, etc.) that can vary across different simulations.

1. Define global parameters in the `variables.global_parameters` section of
   `conf/config.yaml`. Each parameter must specify its type (`vector` or `scalar`)
   and reference values for non-dimensionalization.

2. For `vector` type parameters:
   - If values are single-direction vectors (e.g., [30, 0, 0]), define reference as [30]
   - If values are two-direction vectors (e.g., [30, 30, 0]), define reference as [30, 30]

3. Enable parameter encoding in the model configuration by setting
   `model.encode_parameters: true`. This will:
   - Create a dedicated parameter encoding network (`ParameterModel`)
   - Non-dimensionalize parameters using reference values from `config.yaml`
   - Integrate parameter encodings into both surface and volume predictions

4. Ensure your simulation data includes global parameter values. The DoMINO
   datapipe expects these parameters in the pre-processed `.npy`/`.npz` files:
   - Examine `openfoam_datapipe.py` and `process_data.py` for examples of how global
     parameter values are incorporated for external aerodynamics
   - For the automotive example, `air_density` and `inlet_velocity` remain constant
     across simulations
   - Adapt these files for your specific case to correctly calculate
     `global_params_values` and `global_params_reference` during data preprocessing

5. During training, the model automatically handles global parameter encoding when
   `model.encode_parameters: true` is set
   - You may need to adapt `train.py` if you plan to use global parameters in loss
     functions or de-non-dimensionalization

6. During testing with `test.py`, define `global_params_values` for each test sample:
   - Global parameters must match those defined in `config.yaml`
   - For each parameter (e.g., "inlet_velocity", "air_density"), provide appropriate
     values for each simulation
   - See the `main()` function in `test.py` for implementation examples
   - If using global parameters for de-non-dimensionalization, modify `test_step()`

7. When inferencing on unseen geometries with `inference_on_stl.py`:
   - Define `global_params_values` and `global_params_reference` in both
     `compute_solution_in_volume()` and `compute_solution_on_surface()` methods
   - Adjust these parameters based on your specific use case and parameters defined
     in `config.yaml`

## Extending DoMINO to a custom dataset

This repository includes examples of **DoMINO** training on the DrivAerML dataset.
However, many use cases require training **DoMINO** on a **custom dataset**.
The steps below outline the process.

1. Reorganize that dataset to have the same directory structure as DrivAerML. The
   raw data directory should contain a sepearte directory for each simulation.
   Each simulation directory needs to contain mainly 3 files, `stl`, `vtp` and `vtu`,
   correspoinding to the geometry, surface and volume fields information.
   Additional details such as boundary condition information, for example inlet velocity,
   may be added in a separate `.csv` file, in case these vary from one case to the next.
2. Modify the following parameters in `conf/config.yaml`
   - `project.name`: Specify a name for your project.
   - `expt`: This is the experiment tag.
   - `data_processor.input_dir`: Input directory where the raw simulation dataset is stored.
   - `data_processor.output_dir`: Output directory to save the processed dataset (`.npy`).
   - `data_processor.num_processors`: Number of parallel processors for data processing.
   - `variables.surface`: Variable names of surface fields and fields type (vector or scalar).
   - `variables.volume`: Variable names of volume fields and fields type (vector or scalar).
   - `data.input_dir`: Processed files used for training.
   - `data.input_dir_val`: Processed files used for validation.
   - `data.bounding_box`: Dimensions of computational domain where most prominent solution
     field variations. Volume fields are modeled inside this bounding box.
   - `data.bounding_box_surface`: Dimensions of bounding box enclosing the biggest geometry
     in dataset. Surface fields are modeled inside this bounding box.
   - `train.epochs`: Set the number of training epochs.
   - `model.volume_points_sample`: Number of points to sample in the volume mesh per epoch
   per batch.
     Tune based on GPU memory.
   - `model.surface_points_sample`: Number of points to sample on the surface mesh per epoch
   per batch.
     Tune based on GPU memory.
   - `model.geom_points_sample`: Number of points to sample on STL mesh per epoch per batch.
     Ensure point sampled is lesser than number of points on STL (for coarser STLs).
   - `eval.test_path`: Path of directory of raw simulations files for testing and verification.
   - `eval.save_path`: Path of directory where the AI predicted simulations files are saved.
   - `eval.checkpoint_name`: Checkpoint name `outputs/{project.name}/models` to evaluate
   model.
   - `eval.scaling_param_path`: Scaling parameters populated in `outputs/{project.name}`.
3. Before running `process_data.py` to process the data, be sure to modify `openfoam_datapipe.py`.
   This is the entry point for the user to modify the datapipe for dataprocessing.
   A couple of things that might need to be changed are non-dimensionalizing schemes
   based on the order of your variables and the `DrivAerAwsPaths` class with the
   internal directory structure of your dataset.
   For example, here is the custom class written for a different dataset.

    ```python
    class DriveSimPaths:
        # Specify the name of the STL in your dataset
        @staticmethod
        def geometry_path(car_dir: Path) -> Path:
            return car_dir / "body.stl"

        # Specify the name of the VTU and directory structure in your dataset
        @staticmethod
        def volume_path(car_dir: Path) -> Path:
            return car_dir / "VTK/simpleFoam_steady_3000/internal.vtu"

        # Specify the name of the VTP and directory structure in your dataset
        @staticmethod
        def surface_path(car_dir: Path) -> Path:
            return car_dir / "VTK/simpleFoam_steady_3000/boundary/aero_suv.vtp"
    ```

4. Before running `train.py`, modify the loss functions. The surface loss functions
  currently, specifically `integral_loss_fn`, `loss_fn_surface` and `loss_fn_area`,
  assume the variables to be in a specific order, Pressure followed by Wall-Shear-Stress
  vector.
  Please modify these formulations if your variables are in a different order
  or don't require these losses.
5. Run `test.py` to validate the trained model.
6. Use `inference_on_stl.py` script to deploy the model in applications where inference is
   needed only from STL inputs and the volume mesh is not calculated.

The DoMINO model architecture is used to support the
[Real Time Digital Twin Blueprint](https://github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation)
and the
[DoMINO-Automotive-Aero NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero).

Some of the results are shown below.

![Results from DoMINO for RTWT SC demo](../../../../docs/img/domino_result_rtwt.jpg)

## References

1. [DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations](https://arxiv.org/abs/2501.13350)
