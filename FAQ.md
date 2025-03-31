# Frequently Asked Questions about PhysicsNeMo

## Table of contents

- [What is the recommended hardware for training using PhysicsNeMo framework?](#what-is-the-recommended-hardware-for-training-using-physicsnemo-framework)
- [What model architectures are in PhysicsNeMo?](#what-model-architectures-are-in-physicsnemo)
- [What is the difference between PhysicsNeMo Core and Symbolic?](#what-is-the-difference-between-physicsnemo-core-and-symbolic)
- [What can I do if I dont see a PDE in PhysicsNeMo?](#what-can-i-do-if-i-dont-see-a-pde-in-physicsnemo)
- [What is the difference between the pip install and the container?](#what-is-the-difference-between-the-pip-install-and-the-container)

## What is the recommended hardware for training using PhysicsNeMo framework?

Please refer to the recommended hardware section:
[System Requirments](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html#system-requirements)

## What model architectures are in PhysicsNeMo?

Nvidia PhysicsNeMo is built on top of PyTorch and you can build and train any model
architecture you want in PhysicsNeMo. PhysicsNeMo however has a catalog of models that
have been packaged in a configurable form to make it easy to retrain with new data or certain
config parameters. Examples include GNNs like MeshGraphNet or Neural Operators like FNO.
PhysicsNeMo samples have more models that illustrate how a specific approach with a specifc
model architecture can be applied to a specific problem.
These are reference starting points for users to get started.

You can find the list of built in model architectures
[here](https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models) and
[here](https://github.com/NVIDIA/physicsnemo-sym/tree/main/physicsnemo/sym/models)

## What is the difference between PhysicsNeMo Core and Symbolic?

PhysicsNeMo core is the foundational module that provides the core algorithms, network
architectures and utilities that cover a broad spectrum of Physics-ML approaches.
PhysicsNeMo Symbolic provides pythonic APIs, algorithms and utilities to be used with
PhysicsNeMo core, to explicitly physics inform the model training. This includes symbolic
APIs for PDEs, domain sampling and PDE-based residuals. It also provides higher level
abstraction to compose a training loop from specification of the geometry, PDEs and
constraints like boundary conditions using simple symbolic APIs.
So if you are familiar with PyTorch and want to train model from a dataset, you start
with PhysicsNeMo core and you import PhysicsNeMo symbolic to bring in explicit domain knowledge.
Please refer to the [DeepONet example](https://github.com/physicsnemo/tree/main/examples/cfd/darcy_deeponet_physics)
that illustrates the concept.
If you are an engineer or domain expert accustomed to using numerical solvers, you can
use PhysicsNeMo Symbolic to define your problem at a higher level of abstraction. Please
refer to the [Lid Driven cavity](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/basics/lid_driven_cavity_flow.html)
that illustrates the concept.

## What can I do if I dont see a PDE in PhysicsNeMo?

PhysicsNeMo Symbolic provides a well documeted
[example](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html#writing-custom-pdes-and-boundary-initial-conditions)
that walks you through how to define a custom PDE. Please see the source [here](https://github.com/NVIDIA/physicsnemo-sym/tree/main/physicsnemo/sym/eq/pdes)
to see the built-in PDE implementation as an additional reference for your own implementation.

## What is the difference between the pip install and the container?

There is no functional difference between the two. This is to simplify the ease of
installing and setting up the PhysicsNeMo environment. Please refer to the
[getting started guide](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html#physicsnemo-with-docker-image-recommended)
on how to install using Pip or using the container.
