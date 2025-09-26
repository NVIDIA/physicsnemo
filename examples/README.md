<!-- markdownlint-disable -->
# NVIDIA PhysicsNeMo Examples

## Introduction

This repository provides sample applications demonstrating use of specific Physics-ML
model architectures that are easy to train and deploy. These examples aim to show how
such models can help solve real world problems.

## Introductory examples for learning key ideas

|Use case|Concepts covered|
| --- | --- |
|[Darcy Flow](./cfd/darcy_fno/)|Introductory example for learning basics of data-driven models on Physics-ML datasets|
|[Darcy Flow (Data + Physics)](./cfd/darcy_physics_informed/)|Data-driven training with physics-based constraints|
|[Lid Driven Cavity Flow](./cfd/ldc_pinns/)|Purely physics-driven (no external simulation/experimental data) training|
|[Vortex Shedding](./cfd/vortex_shedding_mgn/)|Introductory example for learning the basics of MeshGraphNets in PhysicsNeMo|
|[Medium-range global weather forecast using FCN-AFNO](./weather/fcn_afno/)|Introductory example on training data-driven models for global weather forecasting (auto-regressive model)|
|[Lagrangian Fluid Flow](./cfd/lagrangian_mgn/)|Introductory example for data-driven training on Lagrangian meshes|
|[Stokes Flow (Physics Informed Fine-Tuning)](./cfd/stokes_mgn/)|Data-driven training followed by physics-based fine-tuning|

## Domain-specific examples

The several examples inside PhysicsNeMo can be classified based on their domains as below:

> **NOTE:**  The below classification is not exhaustive by any means!
    One can classify single example into multiple domains and we encourage
    the users to review the entire list.

> **NOTE:**  * Indicates externally contributed examples.

### CFD

|Use case|Model|Transient|
| --- | --- |  --- |
|[Vortex Shedding](./cfd/vortex_shedding_mgn/)|MeshGraphNet|YES|
|[Drag prediction - External Aero](./cfd/external_aerodynamics/)|MeshGraphNet, UNet, DoMINO, FigConvNet, Transolver|NO|
|[Drag prediction - External Aero - Mixture of Experts](./cfd/external_aerodynamics/)|MoE Model|NO|
|[Navier-Stokes Flow](./cfd/navier_stokes_rnn/)|RNN|YES|
|[Gray-Scott System](./cfd/gray_scott_rnn/)|RNN|YES|
|[Lagrangian Fluid Flow](./cfd/lagrangian_mgn/)|MeshGraphNet|YES|
|[Darcy Flow using Nested-FNOs](./cfd/darcy_nested_fnos/)|Nested-FNO|NO|
|[Darcy Flow using Transolver*](./cfd/darcy_transolver/)|Transolver (Transformer-based)|NO|
|[Darcy Flow (Data + Physics Driven) using DeepONet approach](./cfd/darcy_physics_informed/)|FNO (branch) and MLP (trunk)|NO|
|[Darcy Flow (Data + Physics Driven) using PINO approach (Numerical gradients)](./cfd/darcy_physics_informed/)|FNO|NO|
|[Stokes Flow (Physics Informed Fine-Tuning)](./cfd/stokes_mgn/)|MeshGraphNet and MLP|NO|
|[Lid Driven Cavity Flow](./cfd/ldc_pinns/)|MLP|NO
|[Magnetohydrodynamics using PINO (Data + Physics Driven)*](./cfd/mhd_pino/)|FNO|YES|
|[Shallow Water Equations using PINO (Data + Physics Driven)*](./cfd/swe_nonlinear_pino/)|FNO|YES|
|[Shallow Water Equations using Distributed GNNs](./cfd/swe_distributed_gnn/)|GraphCast|YES|
|[Vortex Shedding with Temporal Attention](./cfd/vortex_shedding_mesh_reduced/)|MeshGraphNet|YES|
|[Data Center Airflow](./cfd/datacenter/)|3D UNet|NO|
|[Fluid Super-resolution*](./cfd/flow_reconstruction_diffusion/)|Denoising Diffusion Probablistic Model|YES|
|[Pre-trained DPOT for Navier-Stokes*](./cfd/navier_stokes_dpot/)|Denoising Operator Transformer|YES|

### Weather

|Use case|Model|
| --- | --- |
|[Medium-range global weather forecast using FCN-SFNO](https://github.com/NVIDIA/modulus-makani)|FCN-SFNO|
|[Medium-range global weather forecast using GraphCast](./weather/graphcast/)|GraphCast|
|[Medium-range global weather forecast using FCN-AFNO](./weather/fcn_afno/)|FCN-AFNO|
|[Medium-range and S2S global weather forecast using DLWP](./weather/dlwp/)|DLWP|
|[Coupled Ocean-Atmosphere Medium-range and S2S global weather forecast using DLWP-HEALPix](./weather/dlwp_healpix/)|DLWP-HEALPix|
|[Medium-range and S2S global weather forecast using Pangu](./weather/pangu_weather/)|Pangu|
|[Diagonistic (Precipitation) model using AFNO](./weather/diagnostic/)|AFNO|
|[Unified Recipe for training several Global Weather Forecasting models](./weather/unified_recipe/)|AFNO, FCN-SFNO, GraphCast|
|[Generative Correction Diffusion Model for Km-scale Atmospheric Downscaling](./weather/corrdiff/)|CorrDiff|
|[StormCast: Generative Diffusion Model for Km-scale, Convection allowing Model Emulation](./weather/stormcast/)|StormCast|
|[Medium-range global weather forecast using Mixture of Experts](./weather/mixture_of_experts/)|MoE Model|
|[Generative Data Assimilation of Sparse Weather Observations](./weather/regen/)|Denoising Diffusion Model|
|[Flood Forecasting](./weather/flood_modeling/)|GNN + KAN|

### Structural Mechanics

|Use case|Model|
| --- | --- |
|[Deforming Plate](./structural_mechanics/deforming_plate/)|MeshGraphNet|

### Healthcare

|Use case|Model|
| --- | --- |
|[Cardiovascular Simulations*](./healthcare/bloodflow_1d_mgn/)|MeshGraphNet|
|[Brain Anomaly Detection](./healthcare/brain_anomaly_detection/)|FNO|

### Additive Manufacturing

|Use case|Model|
| --- | --- |
|[Metal Sintering Simulation*](./additive_manufacturing/sintering_physics/)|MeshGraphNet|

### Molecular Dymanics

|Use case|Model|
| --- | --- |
|[Force Prediciton for Lennard Jones system](./molecular_dynamics/lennard_jones/)|MeshGraphNet|

### Geophysics

|Use case|Model|
| --- | --- |
|[Diffusion model for full-waveform inversion](./geophysics/diffusion_fwi/)|UNet, Global Filter Net|

### Generative

|Use case|Model|
| --- | --- |
|[TopoDiff*](./generative/topodiff)|Conditional diffusion-model|

## Additional examples

In addition to the examples in this repo, more Physics-ML usecases and examples
can be referenced from the [PhysicsNeMo-Sym examples](https://github.com/NVIDIA/physicsnemo-sym/blob/main/examples/README.md).

## NVIDIA support

In each of the example READMEs, we indicate the level of support that will be provided.
Some examples are under active development/improvement and might involve rapid changes.
For stable examples, please refer the tagged versions.

## Feedback / Contributions

We're posting these examples on GitHub to better support the community, facilitate
feedback, as well as collect and implement contributions using
[GitHub issues](https://github.com/NVIDIA/physicsnemo/issues) and pull requests.
We welcome all contributions!
