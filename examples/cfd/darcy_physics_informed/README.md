# Physics Guided models for Darcy flow

This example demonstrates physics informing of a data-driven model using to approaces -
the DeepONet approach which computes the gradients using Autograd and an approach using
Numerical derivatives (PINO).

## Problem overview

This is an extension of the 2D darcy flow data-driven problem. In addition to the
data loss, we will demonstrate the use of physics constranints, specifically
the equation residual loss. [PhysicsNeMo Sym](https://github.com/NVIDIA/physicsnemo-sym)
has utilities tailored for physics-informed machine learning. It also presents an
abstracted APIs that allows users to think and model the problem from the lens of
equations, constraints, etc. In this example, we will only levarage the physics-informed
utilites to see how we can add physics to an existing data-driven model with ease while
still maintaining the flexibility to define our own training loop and other details.
For a more abstracted definition of these type of problems, where the training loop
definition and other things is taken care of implictily, you may refer
[PhysicsNeMo Sym](https://github.com/NVIDIA/physicsnemo-sym)

## Dataset

The training and validation datasets for this example can be found on the [Fourier Neural
Operator Github page](https://github.com/neuraloperator/neuraloperator). The downloading
and pre-processing of the data can also be done by running the below set of commands:

```bash
pip install -r requirements.txt
python download_data.py
```

Do demonstrate the usefulness of the Physics loss, we will deliberately choose a smaller
dataset size of 100 samples. In such regiemes, the effect of physics loss is more
evident, as it regularizes the model in the absense of large data.

## Model overview and architecture

In this example we will use a Fourier Neural Operator (FNO). We will demonstrate two
cases, in the first case, the FNO is used as the branch net and use a fully-connected
network for the trunk net. The input to the branch network is the input permeability
field and the input to the trunk network is the x, y coordinates.
The output of the model is the pressure field. Having the mapping between the pressure field
and the input x and y through a fully-differentiable network will allow us to compute
the gradients of the pressure field w.r.t input x and y through automatic differentiation
through PhysicsNeMo sym utils.

In the second case, we will use just FNO and then compute the derivatives in a PINO style,
using Numerical differentiation. Both approaches are viable ways to introduce physics in
the loss function and the use of one over the other can change from case-to-case basis.
With this example, we intend to demonstrate both such cases so that the users can compare
and contrast the two approaches.

In this example we will use the `PDE` class from PhysicsNeMo-Sym to symbolically define
the PDEs and use the `PhysicsInformer` utility to introduce the PDE
constraints. Defining the PDEs sympolically is very convinient and most natural way to
define these PDEs and allows us to print the equations to check for correctness.
This also abstracts out the
complexity of converting the equation into a pytorch representation. PhysicsNeMo Sym also
provides several complex, well tested PDEs like 3D Navier-Stokes, Linear elasticity,
Electromagnetics, etc. pre-defined which can be used directly in physics-informing
applications.

## Getting Started

To get started with the example, simply run,

```bash
python 
darcy_physics_informed_deeponet.py
```

or

```bash
python 
darcy_physics_informed_fno.py
```

### Note

If you are running this example outside of the PhysicsNeMo container, install
PhysicsNeMo Sym using the instructions from [here](https://github.com/NVIDIA/physicsnemo-sym?tab=readme-ov-file#pypi)

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://www.nature.com/articles/s42256-021-00302-5)
- [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794)
