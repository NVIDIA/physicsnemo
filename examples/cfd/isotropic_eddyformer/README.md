# EddyFormer for 3D Isotropic Turbulence

This example demonstrates how to use the EddyFormer model for simulating
a three-dimensional isotropic turbulence. This example runs on a single GPU.

## Problem Overview

This example focuses on **three-dimensional homogeneous isotropic turbulence (HIT)** sustained by large-scale forcing. The flow is governed by the incompressible Navier–Stokes equations with an external forcing term:

\[
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}
= \nu \nabla^2 \mathbf{u} + \mathbf{f}(\mathbf{x})
\]

where:

- **\(\mathbf{u}(\mathbf{x}, t)\)** — velocity field in a 3D periodic domain  
- **\(\nu = 0.01\)** — kinematic viscosity  
- **\(\mathbf{f}(\mathbf{x})\)** — isotropic forcing applied at the largest scales

### Forcing Mechanism

To maintain statistically steady turbulence, a **constant-power forcing** is applied to the lowest Fourier modes (\(|\mathbf{k}| \le 1\)). The forcing injects a prescribed amount of energy \(P_{\text{in}} = 1.0\) into the system:

\[
\mathbf{f}(\mathbf{x}) =
\frac{P_{\text{in}}}{E_1}
\sum_{\substack{|\mathbf{k}| \le 1 \\ \mathbf{k} \neq 0}}
\hat{\mathbf{u}}_{\mathbf{k}} e^{i \mathbf{k} \cdot \mathbf{x}}
\]

where:

\[
E_1 = \frac{1}{2}
\sum_{|\mathbf{k}| \le 1} 
\hat{\mathbf{u}}_{\mathbf{k}} \cdot \hat{\mathbf{u}}_{\mathbf{k}}^{*}
\]

is the kinetic energy contained in the forced low-wavenumber modes.

Under this forcing, the flow reaches a **statistically steady state** with a Taylor-scale Reynolds number of:

**\(\mathrm{Re}_\lambda \approx 94\)**

### Task Description

The objective of this example is to **predict the future velocity field** of the turbulent flow. Given \(\mathbf{u}(\mathbf{x}, t)\), the task is:

> **Predict the velocity field \(\mathbf{u}(\mathbf{x}, t + \Delta t)\) with \(\Delta t = 0.5\).**

This requires modeling nonlinear, chaotic, multi-scale turbulent dynamics, including:

- energy injection at large scales  
- nonlinear transfer across the inertial range  
- dissipation at the smallest scales  

### Dataset Summary

- **DNS resolution:** \(384^3\) (used to generate the dataset)  
- **Stored dataset resolution:** \(96^3\)  
- **Kolmogorov scale resolution:** ~0.5 η  
- **Forcing:** applied to modes with \(|\mathbf{k}| \le 1\)  
- **Viscosity:** \(\nu = 0.01\)  
- **Input power:** \(P_{\text{in}} = 1.0\)  
- **Flow regime:** statistically steady HIT at \(\mathrm{Re}_\lambda \approx 94\)

## Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

## Download the Dataset

The dataset is publicly available at [Huggingface](https://huggingface.co/datasets/ydu11/re94).
To download the dataset, run (you might need to install the Huggingface CLI):

```bash
bash download_dataset.sh
```

## Getting Started

To train the model, run

```bash
python train_ef_isotropic.py
```

## References

- [EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](https://arxiv.org/abs/2510.24173)
