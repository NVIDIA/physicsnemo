# Physics Informed FNO for Nonlinear Shallow Water Equations

This example demonstrates physics informing of a
data-driven model using numerical derivatives (PINO).

## Problem overview

To examine the properties of PINOs with 3 coupled nonlinear equations, we
examined the ability of the networks to reproduce the nonlinear shallow water
equations. These equations are applicable in several physical scenarios
including tsunami modeling.  We assumed that the total fluid column height
$\eta(x,y,t)$ was composed of a mean height plus some perturbation,
but the initial velocity fields $u(x,y,t)$ and $v(x,y,t)$ were initially
zero. These equations are given by

$$\begin{align}
\frac{\partial(\eta)}{\partial t}+\frac{\partial(\eta u)}{\partial x}+
\frac{\partial(\eta v)}{\partial y}&=0,  \\
\frac{\partial(\eta u)}{\partial t}+
\frac{\partial}{\partial x}\left(\eta u^{2}+\frac{1}{2} g
\eta^{2}\right)+
\frac{\partial(\eta u v)}{\partial y}&=\nu\left(u_{xx} + u_{yy}\right), \\
\frac{\partial(\eta v)}{\partial t}+\frac{\partial(\eta u v)}{\partial x}+
\frac{\partial}{\partial y}\left(\eta v^{2}+\frac{1}{2} g
\eta^{2}\right)&=\nu\left(v_{xx} + v_{yy}\right),
\end{align}$$

$$\begin{align}
\textrm{with} \quad \eta(x,y,0) = \eta_{0}(x,y),\ u(x,y,0)=0,\
v(x,y,0)=0,\ \quad
x,y \in[0,1), \ t \in[0,1],
\end{align}$$

where the gravitational coefficient $g=1$ and the viscosity coefficient
$\nu=0.002$ to prevent the formation of shocks. Below we plot how each of these
fields evolves in space and time according to the PINO predictions and the
simulated data.  We observe that the error in each of these cases is relatively small.

<!-- {: .center} -->
![Nonlinear Shallow Water Equations 2D predictions](../../../docs/img/SWE_0.png)

We will demonstrate the use of data loss and physics constraints,
specifically the equation residual loss, to create accurate predictions.
The `physicsnemo.sym` module (install with `pip install "nvidia-physicsnemo[sym]"`)
has utilities tailored for physics-informed machine learning, and we leverage them
here to add physics to an existing data-driven model with ease while keeping a
fully explicit training loop.

If you previously used the (now archived)
[`physicsnemo-sym`](https://github.com/NVIDIA/physicsnemo-sym) repository,
where the `Solver` / `Domain` / `Constraint` abstractions handled the
training loop and physics losses implicitly, see the
[PhysicsNeMo v2.0 Migration Guide](../../../v2.0-MIGRATION-GUIDE.md#physicsnemo-sym--physicsnemosym)
for how the equivalent pieces look in this newer, explicit style.

## Dataset

The training and validation datasets for this example can be found on the
[PINO Applications Github page](https://github.com/shawnrosofsky/PINO_Applications).

To demonstrate the usefulness of the Physics loss, we will deliberately choose a smaller
dataset size of 45 samples. In such regiemes, the effect of physics loss is more
evident, as it regularizes the model in the absence of large data.

## Model overview and architecture

In this example, we will use a Fourier Neural Operator (FNO). and then compute the
derivatives in a PINO style, using Numerical differentiation with Fourier derivatives.
With this example, we intend to demonstrate how to implement multiple
equations into the loss function.

In this example, we will also use the `PDE` class from
`physicsnemo.sym` to symbolically define the PDEs.
This is a convenient and natural way to define PDEs and allows
us to print the equations to check for correctness. This also abstracts out the
complexity of converting the equation into a pytorch representation. `physicsnemo.sym` also
provides several complex, well-tested PDEs like 3D Navier-Stokes, Linear elasticity,
Electromagnetics, etc. pre-defined which can be used directly in physics-informing
applications. We will also give you the option to choose between the
derivative functions from `physicsnemo.sym` or from the original paper.

## Prerequisites

Install the requirements using:

```bash
pip install -r requirements.txt
pip install "nvidia-physicsnemo[sym]"
```

## Getting Started

The downloading and pre-processing of the data can also be done by running
the below set of commands:

```bash
python download_data.py
```

To get started with the example, simply run,

```bash
python train_swe_nl_pino.py
```

## References

- [Applications of physics informed neural operators](https://arxiv.org/abs/2203.12634)
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794)
