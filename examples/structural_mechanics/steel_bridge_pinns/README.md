<!-- markdownlint-disable MD013 MD033 MD041 -->

# Steel bridge: CAE reference model vs physics-informed neural network

This example builds a complete structural digital-twin workflow for a steel plate-girder
bridge and compares the two ways of obtaining its response:

1. a **CAE model** - a 2D plane-stress finite element (Q4) reference solution, verified
   against Euler-Bernoulli beam theory;
2. a **PINN model** - a physics-informed neural network that minimises the total
   potential energy of the same structure, optionally warm started from a handful of
   CAE measurements (32 displacement "sensors", i.e. under 0.5% of the domain).

Both models solve the same boundary-value problem, so the comparison is apples to
apples: field error against the reference, stress error, data efficiency, and cost.

## The structure

| Item | Value |
| --- | --- |
| Idealisation | 2D plane stress, web of a simply supported plate girder, thickness `t = 20 mm` |
| Span / depth | `L = 12 m` / `H = 1.2 m` (`L/H = 10`, slender) |
| Material | steel, `E = 210 GPa`, `nu = 0.3`, `rho = 7850 kg/m3` |
| Bearings | left pad pinned, right pad on rollers, `0.10 m` up from the bottom flange |
| Deck load | two wheel patches of `60 kN` each, `0.6 m` long, centred at `0.35 L` and `0.55 L` |
| Self weight | included as a body force `rho g` |
| Analytic checks | `M_max = 322.5 kNm`, `sigma_b = 67.2 MPa`, `delta_mid = 7.48 mm` |

The problem is deliberately bending-dominated: that is where the difference between a
mesh-based solver and a smooth neural surrogate is most visible.

## CAE reference model (`steel_bridge_cae.py`)

* Four-node bilinear isoparametric elements, 2x2 Gauss integration, sparse direct solve.
* Element stresses are recovered at the Gauss points, extrapolated to the nodes and
  averaged, giving a smooth nodal field for plotting and for the error metrics.
* Sensor measurements are sampled from the solved field with a fixed seed, so every
  learning variant sees exactly the same sparse data.
* The module also provides the beam-theory reference used for validation and for the
  non-dimensionalisation of the loss terms.

Verification (12 m mesh convergence study, `python train.py experiment=fea`):

| Mesh | Elements | Midspan deflection | Error vs beam theory |
| --- | --- | --- | --- |
| 60 x 12 | 720 | 7.203 mm | +3.65% |
| 120 x 24 | 2 880 | 7.359 mm | +1.55% |
| 240 x 48 | 11 520 | 7.420 mm | +0.73% |

The converged solution takes `0.7 s` on a laptop CPU, and the vertical reactions equal
the applied load (`142.18 kN`) to machine precision.  The bearing pad corners are stress
singularities in a continuum model (peak `337 MPa` versus a nominal bending stress of
`67 MPa`), so stress metrics in this example exclude a strip next to the bearings.

## PINN model (`steel_bridge_pinn.py`)

The network maps `(x, y)` to the displacement field and is trained by minimising the
**total potential energy**

```text
Pi[u] = 1/2 * int_Omega  eps : C : eps * t dA
      +     int_Omega  rho * g * u_y * t dA
      -     int_Gamma  traction . u * t ds
```

with Monte-Carlo estimates of the integrals (2 048 interior points per step).  This
variational form needs only first derivatives of the network.  The strong form
(`div(sigma) = rho g`, implemented with `PhysicsInformer`) was tried as well: on this
slender domain the residuals are dominated by transverse derivatives and the
optimisation collapses to the trivial zero field, which is a known failure mode of
strong-form PINNs for beam-like structures.

Other ingredients:

* the two bearing pads are enforced **exactly** by a smooth multiplicative mask on the
  network output, so there is no Dirichlet penalty term to balance;
* the last layer is zero-initialised, so training starts from the stress-free state;
* scales (`u_ref`, `W_ref`) come from beam theory, not from the CAE solution, which
  keeps the comparison fair and lets the physics-only variant run without any data;
* the data-informed variant first fits the 32 measurements for 300 steps (warm start),
  then minimises `energy + 10 * data` with the energy term ramped in over 500 steps;
* gradient clipping keeps the energy term from taking a destructive first step.

## Running

```bash
pip install -r requirements.txt   # matplotlib, scipy (torch/numpy/hydra come with PhysicsNeMo)

python train.py experiment=fea            # CAE reference only
python train.py experiment=pinn_physics   # purely physics-driven PINN
python train.py experiment=pinn_warm      # few-shot warm start + energy training
python train.py experiment=supervised     # data-only ablation
python train.py experiment=compare        # everything + tables and figures
```

`experiment=compare` takes about 10 minutes on 8 CPU threads (no GPU required) and
writes `metrics.json`, `comparison.md`, the field/error figures and the trained
`torch` checkpoints to the hydra output directory.

Unit tests (12 tests, ~6 s, CPU only):

```bash
python -m pytest tests
```

## Results

Reference: CAE, 240 x 48 elements.  Errors are relative L2 norms on a `241 x 61`
evaluation grid that differs from both the finite element mesh and the collocation
points, so they measure interpolation quality rather than in-sample fit.

| Variant | Sensors | Displ. error | von Mises error | Midspan | Bottom-flange `sigma_xx` | Sensor RMS | Energy balance | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAE reference | - | - | - | 7.420 mm | 65.8 MPa | - | 1.00 | 0.7 s |
| `pinn_physics` | 0 | 11.8% | 9.7% | 6.599 mm | 58.1 MPa | - | 1.13 | 125 s |
| `pinn_warm` | 32 | **2.6%** | **3.2%** | **7.311 mm** | 65.5 MPa | 0.13 mm | 1.23 | 60 s |
| `supervised` | 32 | 4.2% | **114%** | 7.181 mm | 17.9 MPa | 0.014 mm | - | 12 s |

`energy balance` is `2 U / |W_ext|`, which equals one when the field sits at the minimum
of the potential energy (i.e. satisfies equilibrium); values above one mean the predicted
field is too stiff for the load it carries.  `sensor RMS` is the misfit on the
measurements the model was trained on.

Figures produced by `experiment=compare`:

| File | Content |
| --- | --- |
| `cae_reference.png` | CAE displacement, von Mises and the deflection profile vs beam theory |
| `fields_<variant>.png` | CAE and PINN fields side by side on a common colour scale |
| `error_<variant>.png` | Displacement and stress difference maps |
| `convergence.png` | Error and midspan deflection during training |
| `data_efficiency.png` | Displacement and stress error versus the number of measurements |

### Data efficiency

Both learning variants, retrained with `8 / 16 / 32 / 64` measurements (2 000 steps each):

| Sensors | `pinn_warm` displ. | `pinn_warm` von Mises | `supervised` displ. | `supervised` von Mises |
| --- | --- | --- | --- | --- |
| 8 | 3.7% | 6.4% | 31.1% | 1936% |
| 16 | 3.0% | 3.6% | 9.0% | 128% |
| 32 | 2.7% | 4.3% | 4.9% | 157% |
| 64 | 2.8% | 4.6% | 4.4% | 48% |

## Insights

1. **The physics prior replaces most of the data.** Eight measurement points already
   bring the energy-trained PINN to 3.7% field error, and going to 64 points buys less
   than one percentage point.  The data-only surrogate needs 32-64 points to reach a
   comparable displacement error and still has 48-1936% stress error.
2. **Sparse data cannot constrain stresses.** With 32 sensors the supervised model fits
   the measurements to `0.014 mm` - better than the PINN - yet its von Mises field is
   off by 114% and the bottom-flange stress is `17.9 MPa` instead of `65.8 MPa`.
   Stresses are derivatives of the field, so they are far more sensitive to what happens
   between the sensors; a physics-consistent surrogate is the only one usable for
   stress-based checks (yielding, fatigue, bolt design).
3. **Physics alone is not enough either.** Without measurements the network plateaus at
   ~12% after 8 000 steps (twice the training time of the data-informed run) and is
   systematically too stiff (6.60 mm versus 7.42 mm).  The energy landscape is flat along
   low-frequency bending modes, so the optimiser needs many steps to grow the deflection;
   a few measurements anchor the amplitude immediately.
4. **CAE still wins on cost and accuracy for a single design.** 0.7 s and machine-precision
   equilibrium versus 60 s of training for a 2.6% error.  The PINN only pays off when the
   same model is queried many times (optimisation loops, monitoring, inverse problems) or
   when the mesh is impractical (moving boundaries, parameterised geometry).
5. **Use the variational form, not the strong form, for slender structures.** The
   equilibrium residual formulation was implemented first and reproducibly collapsed to
   the zero field; the energy formulation on the same network, BCs and optimiser
   converges.  Residual weighting alone could not fix it because transverse derivatives
   are `(L/H)^2 = 100` times larger than axial ones.
6. **Bearings need care in both methods.** Point supports create a stress singularity
   that dominates peak-stress comparisons (`337 MPa` locally versus `67 MPa` nominal).
   Distributing the reaction over a 0.1 m pad, and excluding that strip from the stress
   metric, makes the comparison meaningful - the same is true when reading CAE results
   for design.
7. **A hybrid workflow is the practical sweet spot.** Use CAE to build the model and
   generate a few tens of measurements, use those to anchor a physics-consistent
   surrogate, and keep the CAE solver for verification - which is exactly the loop this
   example automates.

## Files

| File | Purpose |
| --- | --- |
| `steel_bridge_cae.py` | Q4 plane-stress finite element reference, sensor sampling, beam theory |
| `steel_bridge_pinn.py` | Displacement network, energy functional, hard bearing constraints, diagnostics |
| `train.py` | Hydra entry point: CAE verification, the three learning variants, sweep, reporting |
| `compare.py` | Metrics (relative L2, gauge stress, energy balance) and all figures |
| `config.yaml` | Geometry, material, loads, model and optimiser settings |
| `tests/test_steel_bridge.py` | Patch test, rigid-body check, beam-theory agreement, constitutive and energy checks |

## Notes and limitations

* Linear elasticity, small strains, 2D plane stress with a constant thickness: suitable
  for a global bending assessment, not for local details (connections, stiffeners,
  welds) or for buckling.
* Only the web is modelled; the flanges, stiffeners and the deck plate are folded into
  the equivalent section through the thickness and depth of the idealisation.
* The evaluation grid is fixed for all variants, but the reported PINN numbers depend on
  the random seed; the spread between seeds is a few tenths of a percentage point for the
  data-informed variant and larger (1-2 points) for the physics-only variant.
* Timing is CPU only (8 threads, `torch.set_num_threads`); on a GPU the training cost
  drops by roughly an order of magnitude, which shifts the cost comparison in insight 4.
