# Frequently Asked Questions about PhysicsNeMo

## Table of contents

- [What is the recommended hardware for training using PhysicsNeMo framework?](#what-is-the-recommended-hardware-for-training-using-physicsnemo-framework)
- [What model architectures are in PhysicsNeMo?](#what-model-architectures-are-in-physicsnemo)
- [How do I use physics-informed training with PhysicsNeMo?](#how-do-i-use-physics-informed-training-with-physicsnemo)
- [What can I do if I dont see a PDE in PhysicsNeMo?](#what-can-i-do-if-i-dont-see-a-pde-in-physicsnemo)
- [What is the difference between the pip install and the container?](#what-is-the-difference-between-the-pip-install-and-the-container)

## What is the recommended hardware for training using PhysicsNeMo framework?

Please refer to the recommended hardware section:
[System Requirements](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html#system-requirements)

## What model architectures are in PhysicsNeMo?

Nvidia PhysicsNeMo is built on top of PyTorch and you can build and train any model
architecture you want in PhysicsNeMo. PhysicsNeMo however has a catalog of models that
have been packaged in a configurable form to make it easy to retrain with new data or certain
config parameters. Examples include GNNs like MeshGraphNet or Neural Operators like FNO.
PhysicsNeMo samples have more models that illustrate how a specific approach with a specific
model architecture can be applied to a specific problem.
These are reference starting points for users to get started.

You can find the list of built in model architectures
[here](https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models).

## How do I use physics-informed training with PhysicsNeMo?

PhysicsNeMo includes a `physicsnemo.sym` module (install with
`pip install "nvidia-physicsnemo[sym]"`) that provides symbolic PDE definition,
automatic spatial derivative computation, and physics-informed residual evaluation.
Define your equations using SymPy, then use `PhysicsInformer` to compute PDE
residuals automatically.

See the [LDC PINNs example](examples/cfd/ldc_pinns/) and the
[Darcy physics-informed example](examples/cfd/darcy_physics_informed/) for
complete training scripts.

> **Note:** The separate [PhysicsNeMo-Sym](https://github.com/NVIDIA/physicsnemo-sym)
> repository is being archived. Its core functionality has been upstreamed into
> PhysicsNeMo. See the [migration guide](v2.0-MIGRATION-GUIDE.md#physicsnemo-sym--physicsnemosym)
> for details.

## What can I do if I dont see a PDE in PhysicsNeMo?

Define your PDE using SymPy and the `physicsnemo.sym.eq.pde.PDE` base class.
See the [LDC PINNs example](examples/cfd/ldc_pinns/train.py) for an inline
Navier-Stokes definition, or the
[MHD PINO example](examples/cfd/mhd_pino/losses/mhd_pde.py) for a custom MHD PDE.

## What is the difference between the pip install and the container?

There is no functional difference between the two. This is to simplify the ease of
installing and setting up the PhysicsNeMo environment. Please refer to the
[getting started guide](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html#physicsnemo-with-docker-image-recommended)
on how to install using Pip or using the container.
