# Learning the flow field of Stokes flow

This example demonstrates how to train the MeshGraphNet model to learn the flow field
of Stokes flow and further
improve the accuary of the model predictions by physics-informed inference. This example
also demonstrates how to use physics utilites from
[PhysicsNeMo-Sym](https://github.com/NVIDIA/physicsnemo-sym) to introduce physics-based
constraints.

## Problem overview

The partial differential equation is defined as

$$\begin{aligned}
    -\nu \Delta \mathbf{u} +\nabla p=0, \\
    \nabla \cdot \mathbf{u} = 0,
\end{aligned}$$

where $\mathbf{u} = (u, v)$ defines the velocity and $p$ the pressure, and $\nu$ is the
kinematic viscosity.
The underlying geometry is a pipe without a polygon. On the inlet
$\Gamma_3=0 \times[0,0.4]$, a parabolic inflow profile is prescribed,

$$\begin{aligned}
    \mathbf{u}(0, y)= \mathbf{u}_{\mathrm{in}} =
    \left(\frac{4 U y(0.4-y)}{0.4^2}, 0\right)
\end{aligned}$$

with a maximum velocity $U=0.3$. On the outlet $\Gamma_4=2.2 \times[0,0.4]$, we
define the outflow condition

$$\begin{aligned}
    \nu \partial_\mathbf{n} \mathbf{u}-p \mathbf{n}=0,
\end{aligned}$$

where $\mathbf{n}$ denotes the outer normal vector.

Our goal is to train a MeshGraphNet to learn the map from the polygon geometry to the
velocity and pressure field.
However, sometimes data-driven models may not be able to yield reasonable predictive
accuracy due to network capacity or limited dataset. We can fine-tune our results
using PINNs when the PDE is available. The fine-tuning during inference is much faster
than training the PINN model from the scratch as the model has a better initialization
from the data-driven training.

For the fine-tuning step, we formulate two losses. First loss is to match the
predictions of the original MeshGraphNet model. Second loss includes the physics
losses, i.e. the PDE residuals and the boundary conditions. Having the data loss
helps the PINN model converge faster than training from scratch.

## Dataset

Our dataset provides  numerical simulations of Stokes flow in a pipe domain obstructed
by a random polygon. It contains 1000 random samples and all the simulations were
performed using Fenics. For each sample, the numerical solution cotains the mesh and
the flow information about velocity, pressure, and markers identifying different
boundaries within the domain.

To download the full dataset, please run the bash script in `raw_dataset`

```bash
bash download_dataset.sh
```

## Model overview and architecture

 The inputs of our MeshGraphNet model is:

- mesh

Output of the MeshGraphNet model are:

- velocity field pressure
- pressure field

The input to the model is in form of a `.vtp` file and is then converted to
bi-directional DGL graphs in the dataloader. The final results are also written in the
form of `.vtp` files in the inference code. A hidden dimensionality of 256 is used in
the encoder, processor, and decoder. The encoder and decoder consist of two hidden
layers, and the processor includes 15 message passing layers. Batch size per GPU is
set to 1. Summation aggregation is used in the
processor for message aggregation. A learning rate of 0.0001 is used, decaying
exponentially with a rate of 0.99985.

![Comparison of the MeshGraphNet prediction and the filetered prediction against the
ground truth for velocity and pressure for one
of the samples from the test dataset.](../../../docs/img/stokes.png)

## Prerequisites

Install the requirements using:

```bash
pip install -r requirements.txt
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps
pip install nvidia-physicsnemo.sym --no-build-isolation
```

## Getting Started

The dataset for this example is not publicly available. To get access, please reach out
to the [NVIDIA PhysicsNeMo team](simnet-team@nvidia.com).

Once you've obtained the dataset, follow these steps to preprocess it:

1. **Unzip the Dataset**: If the dataset is compressed, make sure to extract its
contents.

2. **Run the Preprocessing Script**: Execute the provided script to process the dataset.
This will distribute the data
randomly across three directories: `training`, `validation`, and `test`.

```bash
python preprocess.py
````

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

To further fine-tune the model using physics-informed learning, run

```bash
python pi_fine_tuning.py
```

### Note

The fine-tuning step involves training of a PINN model to first refine the
predictions of the MeshGraphNet model followed by an inference of the PINN model.

If you are running this fine-tuning outside of the PhysicsNeMo container, install
PhysicsNeMo Sym using the instructions from [here](https://github.com/NVIDIA/physicsnemo-sym?tab=readme-ov-file#pypi)

This will save the predictions for the test dataset in `.vtp` format in the `results`
directory. Use Paraview to open and explore the results.

## References

- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
