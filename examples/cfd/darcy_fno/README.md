# Fourier Neural Operater for Darcy Flow

This example demonstrates how to set up a data-driven model for a 2D Darcy flow using
the Fourier Neural Operator (FNO) architecture inside of PhysicsNeMo.
This example runs on a single GPU, go to the
`darcy_nested_fno` example for exploring a multi-GPU training.

## Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

## Getting Started

To train the model, run

```bash
python train_fno_darcy.py
```

training data will be generated on the fly.

## Additional Information

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
