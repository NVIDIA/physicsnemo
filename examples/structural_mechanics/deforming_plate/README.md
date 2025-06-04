# MeshGraphNet for Modeling Deforming Plate

**Note:** This example is a work in progress and will be updated soon.

This example is a re-implementation of the DeepMind's deforming plate example
<https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets> in PyTorch.
It demonstrates how to train a Graph Neural Network (GNN) for structural
mechanics applications.

## Problem overview

Mesh-based simulations play a central role in modeling complex physical systems across
various scientific and engineering disciplines. They offer robust numerical integration
methods and allow for adaptable resolution to strike a balance between accuracy and
efficiency. Machine learning surrogate models have emerged as powerful tools to reduce
the cost of tasks like design optimization, design space exploration, and what-if
analysis, which involve repetitive high-dimensional scientific simulations.

However, some existing machine learning surrogate models, such as CNN-type models,
are constrained by structured grids,
making them less suitable for complex geometries or shells. The homogeneous fidelity of
CNNs is a significant limitation for many complex physical systems that require an
adaptive mesh representation to resolve multi-scale physics.

Graph Neural Networks (GNNs) present a viable approach for surrogate modeling in science
and engineering. They are data-driven and capable of handling complex physics. Being
mesh-based, GNNs can handle geometry irregularities and multi-scale physics,
making them well-suited for a wide range of applications.

## Dataset

We rely on DeepMind's deforming plate dataset for this example. The dataset includes
1000 training, 100 validation, and 100 test samples that are simulated using COMSOL
with irregular tetrahedral meshes, each for 400 steps.
These samples vary in the geometry and boundary condition. Each sample
has a unique mesh due to geometry variations across samples, and the meshes have 1271
nodes on average. Note that the model can handle different meshes with different number
of nodes and edges as the input.

The datapipe from the vortex shedding example has been adapted to load this dataset.
Currently, we assume that the deformations are small. This limitation will
be addressed in future updates.

## Model overview and architecture

The model is free-running and auto-regressive. It takes the prediction at
the previous time step to predict the solution at the next step.

The model uses the input mesh to construct a bi-directional DGL graph for each sample.

The output of the model is the mesh deformation between two consecutive steps.

![Comparison between the MeshGraphNet prediction and the
ground truth for the deforming plate for different test samples.
](../../../docs/img/deforming_plate.gif)

A hidden dimensionality of 128 is used in the encoder,
processor, and decoder. The encoder and decoder consist of two hidden layers, and
the processor includes 15 message passing layers. Batch size per GPU is set to 1.
Summation aggregation is used in the
processor for message aggregation. A learning rate of 0.0001 is used, decaying
exponentially with a rate of 0.9999991. Training is performed on 8 NVIDIA H100
GPUs, leveraging data parallelism for 25 epochs. The total training time was
20 hours.

## Prerequisites

Install the requirements using:

```bash
pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps
```

## Getting Started

This example requires the `tensorflow` library to load the data in the `.tfrecord`
format. Install with

```bash
pip install tensorflow
```

Note: If installing tensorflow inside the PhysicsNeMo docker container, it's recommended
to use `pip install "tensorflow<=2.17.1"`

To download the data from DeepMind's repo, run

```bash
cd raw_dataset
sh download_dataset.sh deforming_plate
```

To train the model, run

```bash
python train.py
```

Data parallelism is also supported with multi-GPU runs. To launch a multi-GPU training,
run

```bash
mpirun -np <num_GPUs> python train.py
```

If running in a docker container, you may need to include the `--allow-run-as-root` in
the multi-GPU run command.

Progress and loss logs can be monitored using Weights & Biases. To activate that,
set `wandb_mode` to `online` in the `constants.py`. This requires to have an active
Weights & Biases account. You also need to provide your API key. There are multiple ways
for providing the API key but you can simply export it as an environment variable

```bash
export WANDB_API_KEY=<your_api_key>
```

The URL to the dashboard will be displayed in the terminal after the run is launched.
Alternatively, the logging utility in `train.py` can be switched to MLFlow.

Once the model is trained, run

```bash
python inference.py
```

This will save the predictions for the test dataset in `.gif` format in the `animations`
directory.

## References

- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
