# AeroGraphNet for external aerodynamic evaluation

This example demonstrates how to train the AeroGraphNet model for external aerodynamic
analysis of both simplified (Ahmed body-type) and more realistic (DrivAerNet dataset)
car geometries. AeroGraphNet is based on the MeshGraphNet architecture.
It achieves good accuracy on predicting the pressure and
wall shear stresses on the surface mesh of the respective geometries, as well as
the drag coefficient.

1. [Problem overview](#problem-overview)
2. [Datasets](#datasets)
    1. [Ahmed Body](#ahmed-body)
    2. [DrivAerNet](#drivaernet)
3. [Model](#model-overview-and-architecture)
    1. [MeshGraphNet](#meshgraphnet)
    2. [Bistride Multiscale (BSMS) MGN](#bistride-multiscale-bsms-mgn)
4. [Training](#model-training)
    1. [Ahmed Body](#ahmed-body-training)
        1. [BSMS MGN](#bsms-mgn-training)
    2. [DrivAerNet](#drivaer-training)
5. [Inference](#inference)

## Problem overview

To goal is to develop an AI surrogate model that can use simulation data to learn the
external aerodynamic flow over parameterized car body shape. The trained model can be used
to predict the change in drag coefficient,and surface pressure and wall shear stresses due
to changes in the car geometry. This is a stepping stone to applying similar approaches
to other application areas such as aerodynamic analysis of aircraft wings, more complex
real car geometries, and so on.

## Datasets

AeroGraphNet currently supports two datasets: [Ahmed Body](#ahmed-body) and
[DrivAerNet](#drivaernet).

### Ahmed Body

Industry-standard Ahmed-body geometries are characterized by six design parameters:
length, width, height, ground clearance, slant angle, and fillet radius. Refer
to the [[2, 3](#references)] for details on Ahmed
body geometry. In addition to these design parameters, we include the inlet velocity to
address a wide variation in Reynolds number. We identify the design points using the
Latin hypercube sampling scheme for space filling design of experiments and generate
around 500 design points.

The aerodynamic simulations were performed using the GPU-accelerated OpenFOAM solver
for steady-state analysis, applying the SST K-omega turbulence model. These simulations
consist of 7.2 million mesh points on average, but we use the surface mesh as the input
to training which is roughly around 70k mesh nodes.

To request access to the full dataset, please reach out to the
[NVIDIA PhysicsNeMo team](mailto:physicsnemo-team@nvidia.com).

### DrivAerNet

DrivAerNet [[5](#references)] is a larger dataset which contains around 4000 high-quality
car meshes, coefficients and flow information.
The dataset can be downloaded by following the instructions on the [DrivAerNet GitHub](https://github.com/Mohamedelrefaie/DrivAerNet)
Please see the corresponding [paper](#references) for more details.

## Model overview and architecture

### MeshGraphNet

The AeroGraphNet model is based on the MeshGraphNet [[1](#references)] architecture
which is instrumental for learning from mesh-based data using GNNs.

### Bistride Multiscale (BSMS) MGN

PhysicsNeMo BSMS MGN implementation is based on the BSMS GNN paper [[6](#references)].
The model has two major building blocks:

1. Bi-Stride Pooling and Adjacency Enhancement which precomputes different levels of meshes
    consecutively from the input mesh as the top level.
2. Transition between levels which determines how to do message passing across levels,
    computing the edge weight and node updating after pooling and returning.

Depending on the dataset, the model takes different inputs:

### Ahmed Body dataset

- Ahmed body surface mesh
- Reynolds number
- Geometry parameters (optional, including length, width, height, ground clearance,
slant angle, and fillet radius)
- surface normals (optional)

Output of the model are:

- Surface pressure
- Wall shear stresses
- Drag coefficient - optional, computed using pressure and shear stress outputs.

![Comparison between the AeroGraphNet prediction and the
ground truth for surface pressure, wall shear stresses, and the drag coefficient for one
of the samples from the test dataset.](../../../docs/img/ahmed_body_results.png)

The input to the model is in form of a `.vtp` file and is then converted to
bi-directional DGL graphs in the dataloader. The final results are also written in the
form of `.vtp` files in the inference code. A hidden dimensionality of 256 is used in
the encoder, processor, and decoder. The encoder and decoder consist of two hidden
layers, and the processor includes 15 message passing layers. Batch size per GPU is
set to 1. Summation aggregation is used in the
processor for message aggregation. A learning rate of 0.0001 is used, decaying
exponentially with a rate of 0.99985. Training is performed on 8 NVIDIA A100
GPUs, leveraging data parallelism. Total training time is 4 hours, and training is
performed for 500 epochs.

### DrivAerNet dataset

- Surface mesh

Output of the model are:

- Surface pressure
- Wall shear stresses
- Drag coefficient - optional, can be learned by the model along with other outputs.

The input to the model is the original DrivAerNet dataset. It is recommended to enable
dataset caching (on by default) to speed up the subsequent data loading and training.

![Comparison between the AeroGraphNet prediction and the
ground truth for surface pressure, wall shear stresses, and absolute error for one
of the samples from the test dataset.](../../../docs/img/drivaernet_results.png)

## Model training

The example uses [Hydra](https://hydra.cc/docs/intro/) for experiment configuration.
Hydra provides a convenient way to change almost any experiment parameter,
such as dataset configuration, model and optimizer settings and so on.

For the full set of training script options, run the following command:

```bash
python train.py --help
```

In case of issues with Hydra config, you may get a Hydra error message
that is not particularly useful. In such case, use `HYDRA_FULL_ERROR=1`
environment variable:

```bash
HYDRA_FULL_ERROR=1 python train.py ...
```

This example also requires the `pyvista`, `shapely` and `vtk` libraries. Install with

```bash
pip install pyvista shapely vtk
```

BSMS MGN model requires additional dependency:

```bash
pip install sparse_dot_mkl
```

### Ahmed Body training

The Ahmed Body dataset for this example is not publicly available. To get access,
please reach out to the [NVIDIA PhysicsNeMo team](mailto:physicsnemo-team@nvidia.com).

To train the model, run

```bash
python train.py +experiment=ahmed/mgn data.data_dir=/data/ahmed_body/
```

Make sure to set `data.data_dir` to a proper location.

The following example demonstrates how to change some of the parameters:

```bash
python train.py \
    +experiment=ahmed/mgn \
    data.data_dir=/data/ahmed_body/ \
    model.processor_size=10 \
    optimizer.lr=0.0003 \
    loggers.wandb.mode=online
```

This will change the number of model message passing layers to 10, set learning rate to 0.0003
and enable Weights & Biases logger.

Data parallelism is also supported with multi-GPU runs. To launch a multi-GPU training, run

```bash
mpirun -np <num_GPUs> python train.py +experiment=ahmed/mgn data.data_dir=/data/ahmed_body/
```

If running in a docker container, you may need to include the `--allow-run-as-root` in
the multi-GPU run command.

Progress and loss logs can be monitored using Weights & Biases. To activate that,
add `loggers.wandb.mode=online` to the train script command line. This requires to
have an active Weights & Biases account. You also need to provide your API key.
There are multiple ways for providing the API key but you can simply export it as
an environment variable

```bash
export WANDB_API_KEY=<your_api_key>
```

The URL to the dashboard will be displayed in the terminal after the run is launched.

#### BSMS MGN training

To train BSMS MGN, provide additional parameters, such as number of multi-scale layers,
and, optionally, location of the BSMS cache which would greatly speed up
the training process.
For example, for 6-layer BSMS model, use the following command line:

```bash
python train.py +experiment=ahmed/bsms_mgn \
    data.data_dir=. \
    data.train.num_layers=6 \
    data.val.num_layers=6 \
    data.train.cache_dir=./cache_dir \
    data.val.cache_dir=./cache_dir \
    model.num_mesh_levels=6 \
```

When trained using provided experiment, `ahmed/bsms_mgn`, results should look something like:

| Model | RRMSE |
| :--- | ---: |
| Baseline MGN | 0.21 |
| Level 4 BSMS MGN | 0.16 |
| Level 6 BSMS MGN | 0.11 |

### DrivAer training

To train the MeshGraphNet model, run

```bash
python train.py +experiment=drivaernet/mgn data.data_dir=/data/DrivAerNet/
```

Make sure to set `data.data_dir` to a proper location.

Another option is to train an extended version of MGN, called AeroGraphNet. This model
predicts a drag coefficient directly, along with pressure and WSS.
To use AGN instead of MGN, use `drivaernet/agn` experiment

```bash
python train.py +experiment=drivaernet/agn data.data_dir=/data/DrivAerNet/
```

## Inference

Once the model is trained, run

```bash
python inference.py +experiment=drivaernet/mgn \
    data.data_dir=/data/DrivAerNet/ \
    data.test.num_samples=2 \
    resume_dir=./outputs/
```

Update experiment and data directory as needed. `resume_dir` directory should point
to the output directory of the training which contains model checkpoints.
This example will run inference for only 2 samples, this is just to demonstrate
how various options can be set from the command line.

The inference script will save the predictions for the test dataset in `.vtp` format
in the output directory. Use ParaView or VTK.js to open and explore the results.

## References

1. [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
2. [Some Salient Features Of The Time-Averaged Ground Vehicle Wake](https://doi.org/10.4271/840300)
3. [Ahmed body wiki](https://www.cfd-online.com/Wiki/Ahmed_body)
4. [Deep Learning for Real-Time Aerodynamic Evaluations of Arbitrary Vehicle Shapes](https://arxiv.org/abs/2108.05798)
5. [DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction](https://arxiv.org/abs/2403.08055)
6. [Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network](https://arxiv.org/pdf/2210.02573)
