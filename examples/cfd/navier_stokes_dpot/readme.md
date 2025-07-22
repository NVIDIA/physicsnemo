# Denoising Pre-trained Operator Transformer for Navier-Stokes Equations

This example demonstrates how to set up a Denoising Pre-trained Operator Transformer for solving 2D Naiver-Stokes Equation using inside of PhysicsNeMo.
This example runs on a single GPU. Multi-GPU training and pre-trained weights loading are in development.

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

## Additional Information

## References

- [DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training](https://arxiv.org/abs/2403.03542)
