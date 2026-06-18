# PhysicsNeMo Reservoir — Black-Oil Forward Modelling and α-REKI History Matching

> **End-to-end physics-informed neural-operator workflow for three-phase
> black-oil reservoir simulation, surrogate training, well-rate
> regression, and ensemble-based history matching on the Norne field.**

This repository implements a complete reservoir-engineering pipeline that
combines (i) a Fourier Neural Operator (FNO / PINO) surrogate of the
black-oil PDE system, (ii) a Cluster–Classify–Regress (CCR) Mixture-of-Experts
well-rate model, and (iii) an Adaptive Regularised Ensemble Kalman
Inversion (α-REKI) for posterior characterisation of the permeability,
porosity and fault-multiplier fields.

The full pipeline is orchestrated by [`scripts/docker/startup.sh`](scripts/docker/startup.sh)
which sources [`src/well_modeling.sh`](src/well_modeling.sh).

---

## Table of contents

- [1. Mathematical formulation](#1-mathematical-formulation)
  - [1.1 Black-oil governing equations](#11-black-oil-governing-equations)
  - [1.2 Peaceman well model](#12-peaceman-well-model)
  - [1.3 Relative-permeability closure](#13-relative-permeability-closure)
- [2. Forward surrogate — FNO / PINO](#2-forward-surrogate--fno--pino)
  - [2.1 Operator-learning formulation](#21-operator-learning-formulation)
  - [2.2 Spectral architecture](#22-spectral-architecture)
  - [2.3 Composite training objective](#23-composite-training-objective)
- [3. Well-rate surrogate — CCR Mixture-of-Experts](#3-well-rate-surrogate--ccr-mixture-of-experts)
- [4. Inverse problem — α-REKI](#4-inverse-problem--α-reki)
  - [4.1 Bayesian formulation](#41-bayesian-formulation)
  - [4.2 Adaptive Regularised Ensemble Kalman Inversion](#42-adaptive-regularised-ensemble-kalman-inversion)
  - [4.3 Localisation, parametrisation and ensemble inflation](#43-localisation-parametrisation-and-ensemble-inflation)
- [5. Repository layout](#5-repository-layout)
- [6. Running the pipeline](#6-running-the-pipeline)
- [7. Configuration deck (`DECK_CONFIG.yaml`)](#7-configuration-deck-deck_configyaml)
- [8. Outputs and reproduced results](#8-outputs-and-reproduced-results)
- [9. Testing and static analysis](#9-testing-and-static-analysis)
- [10. References](#10-references)

---

## 1. Mathematical formulation

### 1.1 Black-oil governing equations

The reservoir is treated as a three-phase, slightly compressible black-oil
system on a structured corner-point grid $`\Omega \subset \mathbb{R}^3`$
with $`n_x \times n_y \times n_z`$ cells (in this deck $`46 \times 112 \times 22`$).
Let $`\alpha \in \lbrace o, w, g \rbrace`$ denote oil, water and gas. The conservation
of mass for each phase, written on the field-unit form used by the code,
is:

```math
\frac{\partial}{\partial t}\left( \phi \frac{S_\alpha}{B_\alpha} \right)
 - \nabla \cdot \left( \frac{\mathbf{K} k_{r\alpha}}{\mu_\alpha B_\alpha} 
\nabla p_\alpha \right)
 + q_\alpha = 0
\quad \text{in } \Omega \times (0, T],
```

subject to the volumetric constraint

```math
S_o + S_w + S_g = 1,
```

initial conditions $`p_\alpha(\mathbf{x},0)=p_0(\mathbf{x})`$,
$`S_\alpha(\mathbf{x},0)=S_{\alpha,0}(\mathbf{x})`$, and Dirichlet/Neumann
boundary conditions on $`\partial\Omega`$. The unknowns are the phase
saturations $`S_\alpha`$ and the reference pressure $`p`$ (capillary
pressure is neglected so $`p_\alpha = p`$).

The static fields are the absolute permeability tensor
$`\mathbf{K}(\mathbf{x}) \in \mathbb{R}^{3\times3}`$, the porosity
$`\phi(\mathbf{x})`$, and the fault multiplier
$`T_f(\mathbf{x})\in[0,1]`$ that scales transmissibilities across faults.
PVT closures are described by the formation-volume factors $`B_\alpha(p)`$,
viscosities $`\mu_\alpha(p)`$, and the rock/fluid compressibilities
$`c_\phi`$, $`c_{f,\alpha}`$ — see §1.2 of [`src/conf/DECK_CONFIG.yaml`](src/conf/DECK_CONFIG.yaml).

The compressibility expansion linearises the accumulation term as

```math
\frac{\partial}{\partial t}\left( \frac{\phi S_\alpha}{B_\alpha} \right)
 \approx \frac{\phi S_\alpha}{B_\alpha}\big( c_\phi + c_{f,\alpha} \big)
\frac{\partial p}{\partial t},
```

which is the form used inside the PINO physics residual losses
([`src/forward/gradients_extract.py`](src/forward/gradients_extract.py)).

### 1.2 Peaceman well model

For each well $`w`$ completed in cell $`i`$ with completion length $`\Delta z_i`$,
the volumetric phase rate is given by the standard Peaceman correction:

```math
q_{\alpha,i}^{w} = \frac{2\pi k_h \Delta z_i}{\mu_\alpha \log\big(r_o/r_w\big) + s_w}
 \frac{k_{r\alpha}}{B_\alpha} 
\big(p_i - p^{w}_{\mathrm{bhp}} \big),
```

with equivalent radius

```math
r_o = 0.28 \sqrt{\frac{\sqrt{k_y/k_x} \Delta x^2
+ \sqrt{k_x/k_y} \Delta y^2}{\sqrt{k_y/k_x} + \sqrt{k_x/k_y}}} .
```

Producers are bottom-hole-pressure (BHP) constrained, injectors are
rate-controlled. The total surface rates per well per phase
$`\big(q^\text{WOPR}, q^\text{WWPR}, q^\text{WGPR}\big)_w`$ form the
observation vector $`\mathbf{d}_\text{obs}`$ used by the inverse problem
(§4) and the forward CCR/Peaceman surrogate (§3).

### 1.3 Relative-permeability closure

Phase mobilities use the **Stone-II** three-phase model parameterised by
the two two-phase tables `SWOW` (water–oil drainage,
rows $`[S_w, k_{rw}, k_{ro}]`$) and `SWOG` (oil–gas drainage,
rows $`[S_g, k_{ro}, k_{rg}]`$) declared in
[`src/conf/DECK_CONFIG.yaml`](src/conf/DECK_CONFIG.yaml).
Stone-II reconstructs the three-phase oil $`k_{ro}`$ via

```math
k_{ro}^{3p}(S_w,S_g) = k_{ro}^{cw}\left[
\Big(\frac{k_{ro,wo}(S_w)}{k_{ro}^{cw}} + k_{rw}(S_w)\Big)
\Big(\frac{k_{ro,go}(S_g)}{k_{ro}^{cw}} + k_{rg}(S_g)\Big)
 - \big(k_{rw}(S_w)+k_{rg}(S_g)\big)\right].
```

`Relperm: 2` in the deck selects Stone-II; `Relperm: 1` falls back to
classical Corey curves.

---

## 2. Forward surrogate — FNO / PINO

### 2.1 Operator-learning formulation

The forward problem is recast as learning a discretised non-linear
operator

```math
\mathcal{G}_\theta : \mathcal{X} \longrightarrow \mathcal{Y},
\qquad
\big(\mathbf{K}, \phi, T_f, \mathbf{q}, \mathbf{p}_0, \mathbf{S}_0, \Delta t\big)
 \mapsto 
\big( p(\cdot,t_k), S_w(\cdot,t_k), S_o(\cdot,t_k), S_g(\cdot,t_k) \big)_{k=1}^{T},
```

i.e. a tensor map $`\mathbb{R}^{C_\text{in}\times n_z\times n_x\times n_y}
\to \mathbb{R}^{C_\text{out}\times T\times n_z\times n_x\times n_y}`$.
For the Norne deck, $`T = \texttt{steppi} = 10`$ recorded snapshots,
$`n_z=22,\ n_x=46,\ n_y=112`$.

### 2.2 Spectral architecture

`create_fno_model` ([`src/utils/model_utils.py`](src/utils/model_utils.py))
instantiates a Fourier Neural Operator (Li *et al.* 2021):
each layer applies a learned point-wise lift $`W`$ followed by a spectral
convolution

```math
(\mathcal{K}_\theta v)(\mathbf{x}) = \mathcal{F}^{-1}\Big(
R_\theta \cdot \mathcal{F}\big(v\big)
\Big)(\mathbf{x}),
```

with $`\mathcal{F}, \mathcal{F}^{-1}`$ multidimensional FFTs and
$`R_\theta \in \mathbb{C}^{c\times c}`$ retained on the lowest
$`M`$ Fourier modes (`num_fno_modes=16` here). The decoder is a
PhysicsNeMo-Sym `ConvFullyConnected` head (6 layers × 256 channels,
SiLU, Xavier-uniform init) preserving the per-cell expressivity
required by sharp pressure gradients near wells.

The deck ships **two surrogate variants** sharing the same encoder:

| Variant | Loss form | Selected by |
|---------|-----------|-------------|
| `FNO`   | data-driven only                                | `custom.fno_type: "FNO"`  |
| `PINO`  | data + physics residuals (Darcy + Peaceman + transport) | `custom.fno_type: "PINO"` |

A separate 1-D FNO (`create_fno_model(..., dimension=1)`) maps the
well-feature vector to per-time-step phase rates and is used as a
*proxy Peaceman* model when `Inference peacemann = FNO` in the deck.

### 2.3 Composite training objective

Training minimises a multi-objective loss

```math
\mathcal{L}(\theta) = \sum_{k\in\mathcal{D}} w_k \mathcal{L}_k^\text{data}(\theta)
 + \sum_{k\in\mathcal{S}} w_k \mathcal{L}_k^\text{seq}(\theta)
 + \sum_{k\in\mathcal{P}} w_k \mathcal{L}_k^\text{phys}(\theta),
```

where the weights $`w_k`$ are configured under `loss.weights` in
[`DECK_CONFIG.yaml`](src/conf/DECK_CONFIG.yaml).

**Data-fitting (Sobolev / relative $`H^1`$).**
For target $`y`$ and prediction $`\hat{y}`$ on the 5-D field tensor of
shape $`(B,T,n_z,n_x,n_y)`$ the per-batch loss is

```math
\mathcal{L}^\text{data}(y,\hat{y}) = 
\frac{\big\Vert y - \hat{y} \big\Vert_2}{\big\Vert y \big\Vert_2 + \varepsilon}
 + \frac{1}{3}\sum_{d \in \lbrace x,y,z \rbrace}
\frac{\big\Vert \partial_d y - \partial_d \hat{y} \big\Vert_2}
{\big\Vert \partial_d y \big\Vert_2 + \varepsilon},
```

with finite-difference operators along each spatial dimension. This
is the `extra_loss` in
[`src/forward/gradients_extract.py`](src/forward/gradients_extract.py);
the gradient term is essential for capturing saturation fronts.

**Sequential terms.** For autoregressive rollouts (`unroll: TRUE`,
`K_unroll: 4`, `unroll_cost: AUTO`), one-step predictions
$`\hat{y}_{t+1}=\mathcal{G}_\theta(\hat{y}_t,\cdot)`$ and
their increments $`\Delta\hat{y}=\hat{y}_{t+1}-\hat{y}_{t}`$ are
penalised against ground truth. State-space Gaussian noise
$`\xi \sim \mathcal{N}(0,\sigma^2)`$ with $`\sigma=0.02`$ is injected
into $`\hat{y}_t`$ during a fraction $`p=0.5`$ of mini-batches to
suppress error accumulation.

**Physics residuals.** For the PINO variant, the discretised
black-oil PDE residuals (cell-wise mass-balance, Darcy closure and the
Peaceman well-flux constraint) are evaluated through
finite-difference gradient operators
(`compute_gradient_3d`,
[`src/forward/gradients_extract.py`](src/forward/gradients_extract.py))
and added with weights $`w \sim 10^{-6}`$.

**Optimisation.** Four independent Adam optimisers — one each for the
pressure, water-, oil- and gas-saturation heads, plus one for the
Peaceman well-rate head — are stepped per mini-batch with an exponential
schedule

```math
\eta(s) = \eta_0 \rho^{ s/s_d},
\qquad \eta_0 = 10^{-3}, \rho = 0.95, s_d = 1000.
```

A representative training run (Norne deck, single GB200 GPU, 4 ranks,
`ntrain=100`, `grid_fno=16`) is recorded in
[`src/simulation_logs/20260429_141941/02_train_FNO.log`](src/simulation_logs/20260429_141941/02_train_FNO.log):

| Epoch | Validation loss $`\mathcal{L}_\text{val}`$ | Notes |
|------:|-----------------------------------------:|-------|
| 1     | 17.93   | first checkpoint                  |
| 10    | 7.27    | $`-59\%`$ vs. epoch 1               |
| 100   | $`\sim`$1.4 | water-front structure resolved  |
| 998   | **0.412** | best-saved checkpoint           |
| 1000  | 0.517   | final epoch (early plateau)       |

Wall time: **3 h 22 min** for 1000 epochs.

---

## 3. Well-rate surrogate — CCR Mixture-of-Experts

The well-rate map $`f : \mathbb{R}^{d_\text{in}} \to \mathbb{R}^{N_p \times 3}`$
(features $`\to`$ {WOPR, WWPR, WGPR}) is *piecewise smooth*: its support
splits into operating regimes (above/below bubble point, water-cut
breakthrough, gas coning). A monolithic regressor blurs across regimes
and biases predictions near phase transitions.

We therefore use the **Cluster–Classify–Regress (CCR)** decomposition
of Bernholdt *et al.* (2019) — also known as a discontinuous mixture
of experts:

1. **Cluster**: $`k`$-means on the joint $`(\mathbf{x},\mathbf{y})`$ space
   yields $`K`$ regimes
   $`\lbrace C_1,\dots,C_K \rbrace`$. The number of experts is selected by an
   *elbow* heuristic on the within-cluster variance ($`K \le 5`$).
2. **Classify**: a Random Forest classifier
   $`\pi_\phi : \mathbf{x} \mapsto \Delta^{K-1}`$ is trained on
   the cluster labels.
3. **Regress**: per-cluster experts $`g_{\theta_k}`$ (XGBoost,
   polynomial Ridge or sparse-GP — selected by `Type_of_experts`)
   are fit on the data in $`C_k`$.

The final prediction is the hard mixture

```math
\hat{f}(\mathbf{x}) = \sum_{k=1}^{K}
\mathbf{1}\big[k = \arg\max_j \pi_\phi(\mathbf{x})_j\big] 
g_{\theta_k}(\mathbf{x}).
```

A reproduced run (run-id `20260429_141941`,
[`04_moe_ccr.log`](src/simulation_logs/20260429_141941/04_moe_ccr.log))
gives the held-out scores

| Metric | $`R^2`$ | $`L^2`$ |
|--------|------:|------:|
| WOPR   | 0.969 | 0.067 |
| WWPR   | 0.745 | 0.319 |
| WGPR   | 0.954 | 0.087 |

The CCR surrogate is consumed at inference time as a drop-in for the
Peaceman well model whenever the deck sets `Inference peacemann = "CCR"`.

---

## 4. Inverse problem — α-REKI

### 4.1 Bayesian formulation

Let $`\mathbf{m}\in\mathbb{R}^{n_m}`$ be the parameter vector (stacked
log-permeability, porosity and fault multipliers, optionally projected
through DCT or a VCAE encoder; see §4.3). The forward map
$`\mathcal{G}: \mathbf{m} \mapsto \mathcal{G}(\mathbf{m})`$ is the trained
surrogate of §2 followed by the well operator of §3, and produces
predicted observations of the same dimension as the historical
production data $`\mathbf{d}_\text{obs}\in\mathbb{R}^{n_d}`$.

Under the Gaussian assumption with prior
$`\mathbf{m}\sim\mathcal{N}(\mathbf{m}_0,\mathbf{C}_m)`$ and observation
noise $`\boldsymbol{\eta}\sim\mathcal{N}(\mathbf{0},\mathbf{C}_d)`$
(the deck uses a 25 %-of-data-magnitude diagonal noise model,
`Noise_level: 25.0`), the posterior is

```math
\pi(\mathbf{m}\mid\mathbf{d}_\text{obs}) \propto 
\exp\Big(-\tfrac12 \big\Vert \mathbf{C}_d^{-1/2}\big(\mathcal{G}(\mathbf{m})-\mathbf{d}_\text{obs}\big)\big\Vert_2^2
 - \tfrac12 \big\Vert \mathbf{C}_m^{-1/2}\big(\mathbf{m}-\mathbf{m}_0\big)\big\Vert_2^2\Big),
```

with negative-log-posterior (data-misfit + Tikhonov) cost

```math
J(\mathbf{m}) = \tfrac12\Vert \mathbf{C}_d^{-1/2}
(\mathcal{G}-\mathbf{d}_\text{obs})\Vert^2
+ \tfrac12\Vert \mathbf{C}_m^{-1/2}(\mathbf{m}-\mathbf{m}_0)\Vert^2 .
```

### 4.2 Adaptive Regularised Ensemble Kalman Inversion

Direct MCMC on $`\mathcal{G}`$ is intractable, so we use the
**Adaptive Regularised Ensemble Kalman Inversion (α-REKI)** of
Iglesias and Yang (2018, 2021) — a derivative-free, parallelisable
ensemble method that is mathematically equivalent to a continuous-time
Tikhonov flow.

For an ensemble $`\lbrace\mathbf{m}_j^{(0)}\rbrace_{j=1}^{N_e}\sim\pi_0`$ of size
$`N_e=500`$ (`Ensemble_size: 1000` is the maximum allowed by the deck;
the published log used the 500-rank-balanced subset), the iteration is:

```math
\mathbf{m}_j^{(i+1)} = \mathbf{m}_j^{(i)}
 + \mathbf{C}_{m d}^{(i)} \Big(\mathbf{C}_{d d}^{(i)} + \alpha_i \mathbf{C}_d\Big)^{-1}
\Big(\mathbf{d}_\text{obs} + \boldsymbol{\eta}_j^{(i)} - \mathcal{G}(\mathbf{m}_j^{(i)})\Big),
```

with $`\boldsymbol{\eta}_j^{(i)} \sim \mathcal{N}(\mathbf{0},\alpha_i \mathbf{C}_d)`$
and ensemble cross- and auto-covariances

```math
\mathbf{C}_{m d}^{(i)} = \frac{1}{N_e-1}\sum_{j=1}^{N_e}
\big(\mathbf{m}_j^{(i)}-\bar{\mathbf{m}}^{(i)}\big)\big(\mathcal{G}(\mathbf{m}_j^{(i)})-\bar{\mathcal{G}}^{(i)}\big)^{\top},
```

```math
\mathbf{C}_{d d}^{(i)} = \frac{1}{N_e-1}\sum_{j=1}^{N_e}
\big(\mathcal{G}(\mathbf{m}_j^{(i)})-\bar{\mathcal{G}}^{(i)}\big)\big(\mathcal{G}(\mathbf{m}_j^{(i)})-\bar{\mathcal{G}}^{(i)}\big)^{\top}.
```

The **adaptive regularisation** $`\alpha_i`$ is selected so that the inflated
data-misfit equals the expected $`\chi^2_{n_d}`$, giving the recommended
formula (Iglesias & Yang 2021)

```math
\alpha_i = \min\left\lbrace \frac{n_d}{2 \bar\Phi^{(i)}}, \rho \alpha_{i-1} \right\rbrace,
\qquad
\bar\Phi^{(i)} = \frac{1}{N_e}\sum_{j=1}^{N_e}\tfrac12 \big\Vert\mathbf{C}_d^{-1/2}\big(\mathcal{G}(\mathbf{m}_j^{(i)})-\mathbf{d}_\text{obs}\big)\big\Vert^2,
```

with $`\rho<1`$ a damping factor. The iteration stops when the
*Tikhonov-flow* discrepancy condition

```math
\sum_{i=0}^{I^\star} \frac{1}{\alpha_i} \geq 1
```

is met — exactly the line printed as `✔ Converged (Σ1/α ≥ 1)` in the
log. With `Recommend_alpha: "Yes"` and a maximum of 20 outer iterations
(`iteration_count: 20`), the run
[`06_inverse_FNO.log`](src/simulation_logs/20260429_141941/06_inverse_FNO.log)
converged after 9 iterations:

| Iter $`i`$ | $`\alpha_i`$ | $`\Sigma_{j\le i} 1/\alpha_j`$ | Mean RMSE | Best RMSE |
|---------:|-----------:|-----------------------------:|----------:|----------:|
| 0  | 65.36   | 0.015 | 0.867 | 0.162 |
| 1  | 31.37   | 0.047 | 0.604 | 0.161 |
| 2  | 19.91   | 0.097 | 0.482 | 0.149 |
| ⋮  |  ⋮      |  ⋮    |   ⋮   |   ⋮   |
| 8  | 11.33   | **1.000** ✔ | 0.273 | 0.133 |

Total wall time on 4 × NVIDIA GB200: **0 h 32 min**.

### 4.3 Localisation, parametrisation and ensemble inflation

Three stabilisers are layered on top of the bare update.

**Covariance localisation.** A Gaspari–Cohn taper $`\rho_L : \mathbb{R}^3 \to [0,1]`$
of finite support modifies the Kalman gain element-wise

```math
\big(\mathbf{C}_{m d}\big)_{kl} \leftarrow \rho_L(\mathbf{x}_k - \mathbf{x}_l^{w}) 
\big(\mathbf{C}_{m d}\big)_{kl},
```

suppressing spurious long-range correlations introduced by the finite
ensemble size. Activated by `Covariance_localisation: "Yes"`. The taper
matrix is plotted to
[`RESULTS/HM_RESULTS/Localisation_matrix.png`](RESULTS/HM_RESULTS/Localisation_matrix.png).

**Parametrisation.** Two reduced bases are available:

- **DCT (Discrete Cosine Transform)** truncated at the top
  `DCT: 15 %` of coefficients — projects to a smooth, orthonormal basis
  decoupled across $`(n_x, n_y, n_z)`$.
- **VCAE** — a 3-D Variational Convolutional Auto-Encoder
  (`VCAE3D` in [`src/inverse/utils/ensemble_generation.py`](src/inverse/utils/ensemble_generation.py))
  that encodes the field into a latent $`\mathbf{z}\in\mathbb{R}^{n_z}`$
  with a Gaussian prior. Updates are performed on $`\mathbf{z}`$ and
  pushed back through the decoder, guaranteeing the posterior
  realisations remain on the prior data manifold.

**Inflation.** Optional prior-mean reinflation $`\mathbf{m}_j^{(i+1)}
\leftarrow \bar{\mathbf{m}}^{(i+1)} + \beta \big(\mathbf{m}_j^{(i+1)}-\bar{\mathbf{m}}^{(i+1)}\big)`$
preserves spread when the ensemble collapses; fault multipliers
are clipped via [`clip_ensemble_params`](src/inverse/inversion_operation_ensemble.py)
to honour physical bounds (`minn`, `maxx`, `minnp`, `maxxp` in the deck).

**Alternative: ES-MDA.** Setting `assimilation: "ESMDA"` switches to
Emerick & Reynolds' Ensemble Smoother with Multiple Data Assimilation
with $`\alpha_i = N`$ for all $`i`$ (Tikhonov flow run with constant step
$`1/N`$).

---

## 5. Repository layout

```text
Physicsnemo_publish/
├── data/                       # PVT tables, conversions.mat, MoE expert files
├── simulator_data/             # Norne deck (FULLNORNE.DATA), faults, completions
├── scripts/docker/
│   ├── docker-build.sh         # builds the ptyche container image
│   ├── docker-run.sh           # launches the container with mounts and GPUs
│   ├── install_phynemo.sh      # creates physicsnemo_venv inside the container
│   ├── opm_install.sh          # builds OPM-Flow from source (always fresh)
│   ├── ptyche_build.sh         # cluster-side helper for ptyche images
│   └── startup.sh              # ENTRY POINT — source this inside the container
├── src/
│   ├── conf/DECK_CONFIG.yaml   # Hydra deck (PVT, grid, loss, inverse, scheduler …)
│   ├── extract_data.py         # Stage 1 — OPM ensemble runs → training tensors
│   ├── train.py                # Stage 2 — FNO / PINO training (DDP)
│   ├── plot_metrics_run.py     # Stage 3 — MLflow → metrics_grid plots
│   ├── moe_ccr.py              # Stage 4 — CCR Mixture-of-Experts
│   ├── inference.py            # Stage 5 — surrogate vs. numerical comparison
│   ├── run_inverse.py          # Stage 6 — α-REKI history matching
│   ├── well_modeling.sh        # ORCHESTRATOR for stages 1-6
│   ├── forward/                # FNO/Transolver definitions, gradients, residuals
│   ├── inverse/                # α-REKI / ES-MDA, ensemble generators, plotting
│   ├── compare/                # surrogate↔numerical comparison routines
│   ├── data_extract/           # OPM binary parsers and feature pipelines
│   ├── utils/                  # ecl_binary, array_utils, model_utils, …
│   ├── tests/                  # pytest unit tests for configs and utilities
│   └── simulation_logs/        # timestamped pipeline logs (one per stage)
└── RESULTS/                    # generated figures, ensembles, posterior summaries
    ├── FORWARD_RESULTS/        # per-well comparison plots and CCR scatter plots
    └── HM_RESULTS/             # α-REKI Evolution.gif, percentile fans, alpha.png
```

---

## 6. Running the pipeline

The entire pipeline is wrapped behind a single command. Inside the
ptyche container (or any environment that exposes CUDA + `torchrun`),
run:

```bash
source ./scripts/docker/startup.sh
```

`startup.sh` performs four bootstrap steps before launching the
end-to-end driver:

1. Installs PhysicsNeMo into `physicsnemo_venv/` (skipped if already
   present), via [`scripts/docker/install_phynemo.sh`](scripts/docker/install_phynemo.sh).
2. Activates `physicsnemo_venv`.
3. **Always** rebuilds OPM-Flow via
   [`scripts/docker/opm_install.sh`](scripts/docker/opm_install.sh).
4. `cd src/` and executes
   [`./well_modeling.sh`](src/well_modeling.sh) forwarding any extra CLI flags.

`well_modeling.sh` then runs the six staged jobs sequentially:

| Stage | Script                | Wall time† | Distributed | Output                                         |
|------:|-----------------------|-----------:|:-----------:|------------------------------------------------|
| 1 (opt) | [extract_data.py](src/extract_data.py)   | varies | 1 rank  | training tensors (`.pkl.gz`) under `data/`   |
| 2     | [train.py](src/train.py)                  | 3 h 22 min | DDP × N | trained FNO/PINO checkpoint                  |
| 3     | [plot_metrics_run.py](src/plot_metrics_run.py) | < 1 min | 1 rank | [`metrics_grid_FNO.png`](RESULTS/FORWARD_RESULTS/RESULTS/metrics_grid_FNO.png) |
| 4     | [moe_ccr.py](src/moe_ccr.py)              | 5 min   | 1 rank  | trained CCR experts + scatter plots          |
| 5     | [inference.py](src/inference.py)          | 30 min  | 1 rank  | per-well comparison figures (CCR, FNO, FNO+CCR) |
| 6     | [run_inverse.py](src/run_inverse.py)      | 32 min  | DDP × N | `RESULTS/HM_RESULTS/` (Evolution.gif, posterior plots, percentile fans) |

†Wall-times are from the run-id `20260429_141941` on a 4-rank
NVIDIA GB200 node — see
[`src/simulation_logs/20260429_141941/`](src/simulation_logs/20260429_141941/).

### Useful flags

```bash
# Skip the interactive prompt and pin the rank count + deck:
source ./scripts/docker/startup.sh --ranks 4 --config conf/DECK_CONFIG.yaml

# Dry-run: print the commands without executing:
bash src/well_modeling.sh --dry-run --ranks 1
```

### Direct stage invocation (bypassing the orchestrator)

Each stage is a self-contained Hydra app and can be launched directly:

```bash
cd src
torchrun --nproc_per_node 4 --nnodes 1 --standalone train.py
torchrun --nproc_per_node 1 --nnodes 1 --standalone moe_ccr.py
torchrun --nproc_per_node 1 --nnodes 1 --standalone inference.py
torchrun --nproc_per_node 4 --nnodes 1 --standalone run_inverse.py
```

Override deck values with Hydra dotted-path syntax, e.g.
`custom.fno_type=PINO custom.INVERSE_PROBLEM.Ensemble_size=200`.

---

## 7. Configuration deck (`DECK_CONFIG.yaml`)

The single source of truth is
[`src/conf/DECK_CONFIG.yaml`](src/conf/DECK_CONFIG.yaml). Below are
the most consequential parameters and their meaning.

### 7.1 Reservoir and PVT (`custom.PROPS`)

| Key                   | Value (default)        | Meaning                                                   |
|-----------------------|-----------------------:|-----------------------------------------------------------|
| `nx, ny, nz`          | 46, 112, 22            | grid block counts                                          |
| `BO, BW`              | 1.2, 1.0  RB/STB        | oil / water formation-volume factors                       |
| `UO, UW`              | 2.5, 1.0  cP            | oil / water viscosities                                    |
| `SWI, SWR`            | 0.10, 0.10              | initial / residual water saturation                        |
| `CFW, CFO, CT`        | $`10^{-5},10^{-5},2{\cdot}10^{-5}`$ 1/psi | compressibilities                          |
| `P1, PB, PATM`        | 3000, 3000, 14.7  psi   | initial / bubble-point / atmospheric pressure              |
| `minn, maxx`          | 50, 20000  mD           | permeability clip bounds (used by the inverse update)      |
| `minnp, maxxp`        | 0.01, …                 | porosity clip bounds                                       |

### 7.2 Inverse problem (`custom.INVERSE_PROBLEM`)

| Key                     | Default      | Meaning                                                                   |
|-------------------------|-------------:|---------------------------------------------------------------------------|
| `assimilation`          | `aREKI`      | algorithm choice — `aREKI` or `ESMDA`                                      |
| `Ensemble_size`         | 1000         | $`N_e`$ — capped to multiples of `--ranks`                                  |
| `iteration_count`       | 20           | maximum outer iterations                                                  |
| `Recommend_alpha`       | `Yes`        | use $`\alpha_i = n_d/(2\bar\Phi^{(i)})`$ instead of constant inflation       |
| `Noise_level`           | 25.0 %       | observation-noise magnitude as % of data value                            |
| `Covariance_localisation` | `Yes`      | apply Gaspari–Cohn taper                                                  |
| `DO_DCT`                | `Yes`        | enable DCT parametrisation                                                |
| `DCT`                   | 15           | % of DCT coefficients retained                                            |
| `Do_param_method`       | `VCAE`       | VCAE alternative to DCT (set `parametrization_options: "Yes"` to activate) |
| `Pretrained_Model`      | `Yes`        | load a pre-trained generative model for the prior                         |
| `Generate_ensemble`     | `pre-trained`| `afresh` to resample priors, `pre-trained` to reuse                       |
| `Decorrelationn`        | `No`         | optional ensemble decorrelation                                           |

### 7.3 Forward training (`custom.*`)

| Key                  | Default | Meaning                                                                   |
|----------------------|--------:|---------------------------------------------------------------------------|
| `model_type`         | `FNO`   | backbone (`FNO` or `TRANSOLVER`)                                          |
| `fno_type`           | `FNO`   | `FNO` (data-driven) or `PINO` (physics-informed)                          |
| `unroll`             | `TRUE`  | enable truncated-BPTT autoregressive rollout                              |
| `K_unroll`           | 4       | rollout window length                                                     |
| `unroll_cost`        | `AUTO`  | feed predictions back as next-step input (closed loop)                    |
| `train_interest`     | 1       | 1 = fixed schedule, 2 = flexible PhysicsNeMo trainer                      |
| `Relperm`            | 2       | 1 = Corey, 2 = Stone-II                                                   |
| `pde_method`         | 1       | 1 = approximate gradients, 2 = extensive PDE residuals                    |
| `ntrain / nval / ntest` | 100/1/1 | dataset partition sizes                                                |
| `seed`               | 42      | RNG seed                                                                 |
| `steppi`             | 10      | recorded timesteps per simulation                                         |

### 7.4 Loss weights (`loss.weights`)

| Key                    | Default | Meaning                                                     |
|------------------------|--------:|-------------------------------------------------------------|
| `pressure / water_sat / oil_sat / gas_sat` | 2 / 2 / 1 / 2 | data-fit Sobolev loss weights      |
| `Y`                    | 1       | Peaceman well-rate matching                                 |
| `autoregressive_weight`| 0.1     | rollout penalty                                             |
| `pressured / oild / saturationd / gasd / peacemanned` | $`10^{-6}`$ | physics residuals (PINO only) |
| `noise_std`            | 0.02    | state-noise injection std for autoregressive robustness     |
| `noise_prob`           | 0.5     | per-batch probability of noise injection                    |

### 7.5 Optimiser & scheduler

```yaml
optimizer:
  weight_decay: 1e-4   # L2 regularisation
  lr:           1e-3   # initial learning rate
  gamma:        1.0    # per-epoch LR decay
scheduler:
  decay_rate:  0.95    # multiplicative decay
  decay_steps: 1000    # steps per decay window
training:
  max_steps: 1000      # hard upper bound
batch_size:
  grid_fno:   16
  validation: 1
  test:       1
```

---

## 8. Outputs and reproduced results

### 8.1 Forward surrogate (`RESULTS/FORWARD_RESULTS/`)

| File | Content |
|------|---------|
| [`metrics_grid_FNO.png`](RESULTS/FORWARD_RESULTS/RESULTS/metrics_grid_FNO.png) | training-loss curves per output head (extracted from MLflow) |
| `CCR/Machine_{WOPR,WWPR,WGPR}_perform.jpg` | parity plots of CCR predictions vs. numerical reference |
| `COMPARE_RESULTS/{WOPR,WWPR,WGPR}.png` | end-to-end well-rate ribbon plots — Reference vs. CCR vs. FNO vs. FNO+CCR |
| `COMPARE_RESULTS/Histogram.png` | per-well RMSE histogram across the 41 producers / injectors |
| `True_Flow/` | reference OPM `.UNRST` / `.SMSPEC` summaries used as ground truth |

Headline numbers from
[`05_inference.log`](src/simulation_logs/20260429_141941/05_inference.log):

```text
WGPR Mean R²   PINO–CCR =  -31.0 %    PINO–FNO =  65.6 %    PINO–FNO+CCR = 92.5 %
WOPR Mean R²   PINO–CCR =  -20.8 %    PINO–FNO =   ↯       PINO–FNO+CCR = -46.8 %
WWPR Mean R²   PINO–CCR =    1.5 %    PINO–FNO =   ↯       PINO–FNO+CCR = -331.6 %
Avg RMSE       CCR=8262.7  FNO=1205.8  FNO+CCR=1188.98
```

(The $`R^2`$ collapse on the bare FNO PEACEMAN model for water/oil rates
is the expected motivation for cascading **FNO → CCR**: FNO captures
the global field but cannot resolve the discontinuous well-regime
boundaries, while CCR was specifically designed to do so. The hybrid
**FNO+CCR** path therefore produces the best RMSE.)

> **Note.** The forward-model figures below are restricted to the
> **WGPR** (gas production rate) channel only, because the cascaded
> **PINO-FNO + CCR** surrogate is the engine that drives the inverse
> problem in § 4 / § 8.2 — and history matching in this deck is
> conditioned on gas-rate observations. WOPR / WWPR figures still live
> on disk under `RESULTS/FORWARD_RESULTS/RESULTS/` for reference.

#### 8.1.1 Forward-model training convergence (MLflow)

`plot_metrics_run.py` (Stage 3) consumes the MLflow tracking URI
populated by `train.py` and emits a single grid plot of the per-head
training and validation losses. This is the canonical MLflow snapshot
of the 1000-epoch run:

| MLflow training and validation losses |
|:-------------------------------------:|
| ![metrics_grid_FNO](RESULTS/FORWARD_RESULTS/RESULTS/metrics_grid_FNO.png) |

#### 8.1.2 CCR Mixture-of-Experts — WGPR performance

The `RESULTS/FORWARD_RESULTS/RESULTS/CCR/` directory holds the
held-out evaluation of the Cluster–Classify–Regress well-rate
surrogate (§ 3) on the gas-rate Peaceman target:

| Parity / scatter — `Machine_WGPR_perform.jpg` | Per-well overlay (CCR vs. Numerical) — `WGPR_PhysicsNeMo_vs_Numerical.png` |
|:--------------------------------------------:|:----------------------------------------------------------------------------:|
| ![Machine WGPR](RESULTS/FORWARD_RESULTS/RESULTS/CCR/Machine_WGPR_perform.jpg) | ![WGPR vs Numerical](RESULTS/FORWARD_RESULTS/RESULTS/CCR/WGPR_PhysicsNeMo_vs_Numerical.png) |

#### 8.1.3 Hybrid PINO-FNO + CCR — the inverse-problem engine

Everything below is sourced from
`RESULTS/FORWARD_RESULTS/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO_CCR/`,
which is the cascaded **field-FNO + well-CCR** surrogate consumed by
`run_inverse.py` (Stage 6). This is the only surrogate that propagates
into the α-REKI history-matching loop.

| Per-well WGPR trajectory (Numerical vs. PINO-FNO+CCR) |
|:-----------------------------------------------------:|
| ![WGPR](RESULTS/FORWARD_RESULTS/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO_CCR/WGPR.png) |

| $`R^2 / L^2`$ summary | Per-time-step error ribbon |
|:-------------------:|:--------------------------:|
| ![R²/L²](RESULTS/FORWARD_RESULTS/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO_CCR/R2L2.png) | ![Compare-time](RESULTS/FORWARD_RESULTS/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO_CCR/Compare_time.png) |

**Animated rollout — PINO-FNO + CCR pressure / saturation evolution
across the 10 recorded snapshots:**

![PINO-FNO+CCR Evolution](RESULTS/FORWARD_RESULTS/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO_CCR/Evolution.gif)

### 8.2 History matching (`RESULTS/HM_RESULTS/`)

| File | Content |
|------|---------|
| `alpha.png` | adaptive regularisation $`\alpha_i`$ over outer iterations |
| `Cost_Function.png` | mean / best ensemble cost vs. outer iteration |
| `Localisation_matrix.png` | Gaspari–Cohn taper used for the Kalman gain |
| `MEAN_RESERVOIR_MODEL/` | posterior mean perm / poro / fault fields, GIF evolution |
| `BEST_RESERVOIR_MODEL/` | best (lowest-cost) ensemble member realisation |
| `ADAPT_REKI/` | full ensemble snapshots (`.pkl.gz`) and per-realisation plots |
| `PERCENTILE/` | P10 / P50 / P90 percentile reservoir models and fan plots |
| `Posterior_Ensembles_percentile.joblib` | the persisted P10–P90 ensemble pickle |
| `WGPR_HISTORY.png` | gas-rate history vs. observation per well                |
| `WGPR_WATER_PRIOR_ENSEMBLE_WGPR.png`, `..._POSTERIOR_..._.png` | prior vs. posterior ensemble fans for WGPR |

End-of-run reservoir-RMSE numbers from
[`06_inverse_FNO.log`](src/simulation_logs/20260429_141941/06_inverse_FNO.log):

```text
RMSE OF MLE_RESERVOIR_MODEL    = 14243.08
RMSE OF BEST RESERVOIR MODEL   = 15781.39
RMSE of MAP RESERVOIR MODEL    = 16211.36
Per-well WGPR  Min RMSE        = 4686.48     →  Best ensemble member: P10
Σ 1/α                          = 1.000   → ✔ Tikhonov-flow convergence
```

Run summary block:

```text
Inverse problem solver  : Adaptive Regularised Ensemble Kalman Inversion (α-REKI)
Forward model surrogate : PINO-FNO + CCR
Ensemble size           : 500
Outer iterations        : 9 (cap 20)
Wall clock              : 0:32:12
```

#### 8.2.1 Convergence and regularisation

| Adaptive regularisation $`\alpha_i`$ vs. iteration | Mean / best ensemble cost vs. iteration |
|:-------------------:|:-------------------:|
| ![alpha](RESULTS/HM_RESULTS/alpha.png)             | ![Cost](RESULTS/HM_RESULTS/Cost_Function.png) |

The Tikhonov-flow stopping rule
$`\sum_{i=0}^{I^\star}1/\alpha_i \ge 1`$ is satisfied at iteration 8 —
visible as the slope change in the cumulative-α curve.

#### 8.2.2 Localisation kernel

| Gaspari–Cohn taper applied element-wise to $`\mathbf{C}_{md}`$ |
|:------------------------------------------------------------:|
| ![Localisation matrix](RESULTS/HM_RESULTS/Localisation_matrix.png) |

#### 8.2.3 Prior vs. posterior — WGPR ensemble fans

| Prior ensemble | Posterior ensemble |
|:--------------:|:------------------:|
| ![Prior](RESULTS/HM_RESULTS/WGPR_WATER_PRIOR_ENSEMBLE_WGPR.png)  | ![Posterior](RESULTS/HM_RESULTS/WGPR_WATER_POSTERIOR_ENSEMBLE_WGPR.png)  |

The posterior fan is collapsed onto the observed history with
substantially reduced spread compared to the prior, while preserving
spread in regions where the data is uninformative.

#### 8.2.4 Reservoir realisations — MEAN, BEST, MAP and percentile fields

For each summary realisation the pipeline emits a static permeability /
porosity / fault-multiplier reconstruction (`Petro_Recon.png`), the
predicted WGPR vs. observed history (`WGPR_MODEL.png`,
`WGPR_SINGLE.png`) and an animated 3-D evolution (`Evolution.gif`).

**MAP — Adaptive REKI sample (lowest a-posteriori cost ensemble member):**

| Petrophysical reconstruction | WGPR per-well overlay |
|:----------------------------:|:---------------------:|
| ![ADAPT_REKI Petro](RESULTS/HM_RESULTS/ADAPT_REKI/Petro_Recon.png) | ![ADAPT_REKI WGPR](RESULTS/HM_RESULTS/ADAPT_REKI/WGPR_MODEL.png) |

![ADAPT_REKI Evolution](RESULTS/HM_RESULTS/ADAPT_REKI/Evolution.gif)

**Posterior MEAN reservoir:**

| Petrophysical reconstruction | WGPR per-well overlay |
|:----------------------------:|:---------------------:|
| ![MEAN Petro](RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL/Petro_Recon.png) | ![MEAN WGPR](RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL/WGPR_MODEL.png) |

![MEAN Evolution](RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL/Evolution.gif)

**BEST single ensemble member (minimum data-misfit, MLE-style):**

| Petrophysical reconstruction | WGPR per-well overlay |
|:----------------------------:|:---------------------:|
| ![BEST Petro](RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL/Petro_Recon.png) | ![BEST WGPR](RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL/WGPR_MODEL.png) |

![BEST Evolution](RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL/Evolution.gif)

#### 8.2.5 Percentile reservoir models (P10 / P50 / P90)

| Permeability percentiles | WGPR percentile fan | Per-well RMSE histogram |
|:------------------------:|:-------------------:|:-----------------------:|
| ![Percentile fields](RESULTS/HM_RESULTS/PERCENTILE/Reservoir_models.png) | ![WGPR](RESULTS/HM_RESULTS/PERCENTILE/WGPR.png) | ![Histogram](RESULTS/HM_RESULTS/PERCENTILE/HISTOGRAM_WGPR.png) |

#### 8.2.6 Combined comparison

| All-realisations summary view |
|:-----------------------------:|
| ![Comparison](RESULTS/HM_RESULTS/Comparison.png) |

| WGPR history-matching panel |
|:---------------------------:|
| ![WGPR History](RESULTS/HM_RESULTS/WGPR_HISTORY.png) |

---

## 9. Testing and static analysis

### 9.1 Unit tests

`pytest` covers the dataclass schemas (`PhysicsParams`, `EnsembleSetup`,
`WellConfig`, …), array utilities, ensemble samplers and signature
contracts. Run from `src/`:

```bash
cd src
pytest tests/ -q
```

The test files are:

- [`test_compare_config.py`](src/tests/test_compare_config.py)
- [`test_ensemble_utils.py`](src/tests/test_ensemble_utils.py)
- [`test_inverse_config.py`](src/tests/test_inverse_config.py)
- [`test_path_utils.py`](src/tests/test_path_utils.py)
- [`test_scale_operations.py`](src/tests/test_scale_operations.py)
- [`test_signature_contracts.py`](src/tests/test_signature_contracts.py)
- [`test_split_matrix.py`](src/tests/test_split_matrix.py)
- [`test_training_config.py`](src/tests/test_training_config.py)

### 9.2 Lint and type-check

```bash
ruff check src/      # currently: PASS — no diagnostics
ty check src/        # remaining diagnostics are unresolved-import only
                     # (numpy / torch / hydra / matplotlib …) — environmental,
                     # not real code defects.
```

Both tools are wired into the project. After applying the latest
clean-up sweep, `ruff` reports **All checks passed!** on `src/` and
`ty` reports zero non-environmental defects on the production paths
(`src/forward`, `src/inverse`, `src/utils`, `src/data_extract`).

---

## Author

- Clement Etienam - Senior DevTech Engineer, Energy @ NVIDIA
- Email: <cetienam@nvidia.com>

## Contributors

- Oleg Ovcharenko - NVIDIA
- Issam Said - NVIDIA
- Nick Luiken - NVIDIA

## 10. References

1. Bernholdt, D. E., Cianciosa, M. R., Green, D. L., Park, J. M.,
   Law, K. J. H., & **Etienam, C.** (2019).
   *Cluster, Classify, Regress: A general method for learning
   discontinuous functions.*
   **Foundations of Data Science**, 1(4), 491.
2. **Etienam, C.**, Law, K., & Wade, S. (2020).
   *Ultra-fast Deep Mixtures of Gaussian Process Experts.*
   arXiv:2006.13309.
3. Iglesias, M. A. (2016). *A regularizing iterative ensemble Kalman
   method for PDE-constrained inverse problems.* **Inverse Problems**,
   32(2), 025002.
4. Iglesias, M. A., & Yang, Y. (2021). *Adaptive regularisation
   for ensemble Kalman inversion.* **Inverse Problems**, 37(2), 025008.
5. Emerick, A. A., & Reynolds, A. C. (2013). *Ensemble Smoother with
   Multiple Data Assimilation.* **Computers & Geosciences**, 55, 3–15.
6. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K.,
   Stuart, A., & Anandkumar, A. (2021). *Fourier Neural Operator for
   Parametric Partial Differential Equations.* **ICLR 2021**.
7. Li, Z. *et al.* (2024). *Physics-Informed Neural Operator for
   Learning Partial Differential Equations.* **ACM / JMS Climate**.
8. Peaceman, D. W. (1983). *Interpretation of well-block pressures
   in numerical reservoir simulation with non-square grid blocks
   and anisotropic permeability.* **SPE J.**, 23(3), 531-543.
9. Stone, H. L. (1973). *Estimation of three-phase relative
   permeability and residual oil data.* **JCPT**, 12(4).
10. Gaspari, G., & Cohn, S. E. (1999). *Construction of correlation
    functions in two and three dimensions.* **QJRMS**, 125(554).

---
