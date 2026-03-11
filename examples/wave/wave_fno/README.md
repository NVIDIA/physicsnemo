# Fourier Neural Operator for 2D Wave Equation

This example demonstrates how to train a Fourier Neural Operator (FNO) to learn
the solution operator for the 2D wave equation inside of PhysicsNeMo.

The wave equation is a fundamental hyperbolic PDE:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

The FNO learns to map the initial wavefield $u(x, y, 0)$ to the solution at a
later time $u(x, y, T)$.

Training data is generated on the fly using a leapfrog finite-difference solver
with periodic boundary conditions.

## Problem Setup

- **Domain**: $[0, 1]^2$ with periodic boundaries
- **Wave speed**: $c = 1.0$
- **Initial condition**: Superposition of random Fourier modes
- **Target**: Solution at $T = 0.5$
- **Resolution**: $128 \times 128$

## Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

## Getting Started

To train the model, run

```bash
python train_fno_wave.py
```

Training data is generated on the fly.

## Additional Information

This fills the hyperbolic PDE gap in PhysicsNeMo examples. The existing examples
focus on elliptic (Darcy) and parabolic (Navier-Stokes) problems. Wave equations
are critical for acoustics, seismology, and electromagnetic applications.

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [PDEBench: An Extensive Benchmark for Scientific Machine Learning](https://arxiv.org/abs/2210.07182)
