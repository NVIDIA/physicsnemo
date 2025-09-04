# Molecular Dynamics using GNNs

This example demonstrates how to leverage the optimized model implementations in PhysicsNeMo
for different domains. This example showcases how you can leverage the MeshGraphNet
model in PhysicsNeMo for developing a DL model for predicting forces/potential for a Lennard
Jones System as described in the [paper here](https://arxiv.org/abs/2112.03383).

## Problem overview

The goal is to train an AI model that can predict the forces on atoms of a
Lennard Jones system (liquid Argon) given the positions of its atoms.

## Dataset

The model is trained on data generated using OpenMM MD simulator. The dataset consists
of 10000 samples of the 258 atom system. For original dataset please refer
the [original publication](https://arxiv.org/abs/2112.03383) and
[Git repo](https://github.com/BaratiLab/GAMD) of the origial work.

## Model overview and architecture

The model uses a MeshGraphNet model for the prediction of forces. Since all the atoms
in this system are of same type (i.e. Argon), the node encoder is dropped.
The graph edges are generated based on nearest-neighbor search.

![Results from PhysicsNeMo training for the LJ system.](../../../docs/img/lj_system_physicsnemo_results.png)

## Prerequisites

Install the requirements using:

```bash
pip install -r requirements.txt
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps
```

## Getting Started

To download the data, run

```bash
python download_data.py
```

To train the model, run

```bash
python lennard_jones_system.py
```

Distributed Data Parallel training is enabled for this example. To run the example on
multiple GPUs, run

```bash
mpirun -np <num_GPUs> python lennard_jones_system.py
```

If running in a docker container, you may need to include the `--allow-run-as-root` in
the multi-GPU run command.

While the current example trains a light-weight model that can be run on any GPU, on
8 A100s, the training time per epoch is around 90 seconds. The validation error
computation that's run every epoch on the test dataset takes around 65 seconds.
Thus total time per epoch is ~155 seconds (Full training takes roughly 1.3 hrs
(30 epochs)).

## References

[Graph Neural Networks Accelerated Molecular Dynamics](https://arxiv.org/pdf/2112.03383.pdf)
