# Denoising Pre-trained Operator Transformer for Navier-Stokes Equations

This example demonstrates how to set up a Denoising Pre-trained Operator
Transformer for solving 2D Naiver-Stokes equation in PhysicsNeMo.

**Note:** This example runs on a single GPU. Multi-GPU training and
pre-trained weights loading are in development.

**Note:** This example is a work in progress and will be updated in the next release.

## Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

## Getting Started

To train the model, run

```bash
python train_dpot.py
```

training data will be generated on the fly.

Set train to False in config file to inference the model.

## TODO

- Add utility for plotting the inference results.
- Add support for multi-GPU training.
- Better handling of the dataset download.
- More comprehensive unit tests.

## References

- [DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training](https://arxiv.org/abs/2403.03542)
