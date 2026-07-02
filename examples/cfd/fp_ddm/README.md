# FP-DDM Thermal Domain Decomposition

This example applies an overlapping Schwarz domain-decomposition method to the
steady two-dimensional thermal equation

$$
-\nabla \cdot \left(k(\mathbf{x}) \nabla T(\mathbf{x})\right)
= q(\mathbf{x}),
$$

with spatially varying conductivity, volumetric heat source, and Dirichlet
temperature boundary conditions. Each subdomain is solved with either the
included matrix-free finite-volume reference solver or a physics-informed
[PhysicsNeMo FNO](../../../docs/api/models/fnos.rst).

The FNO is trained without solution labels. Its objective combines the thermal
equation residual with the boundary residual. During a decomposed solve,
neighboring patches exchange their overlap values and repeat local solves until
the interface residual converges.

## Method

One FP-DDM iteration performs the following operations:

1. Optionally adapt the FNO on the current subdomain batch using the same
   physics and boundary losses used during training.
2. Solve every overlapping subdomain with the configured local solver.
3. Update each artificial boundary from the neighboring overlap.
4. Measure interface consistency and assemble optional global diagnostics.
5. Stop at the requested tolerance, iteration limit, or patience limit.

The `parallel` interface handler optimizes all artificial boundaries in one
batched step. The `gradient` handler updates patches sequentially, while
`exchange` performs a direct relaxed Dirichlet exchange.

## Requirements

Install PhysicsNeMo, then install the example dependencies from the repository
root:

```bash
pip install -r examples/cfd/fp_ddm/requirements.txt
```

SciPy and OpenSimplex provide the smooth synthetic material layouts. Matplotlib
and ImageIO are only needed when `run.visualize=true`.

## Train The FNO

The default configuration generates local thermal problems on the fly and
trains one model directly with the physics-informed objective:

```bash
python examples/cfd/fp_ddm/train.py
```

All settings use Hydra overrides. This short command validates the full
training and checkpoint path; it is not a quality benchmark:

```bash
python examples/cfd/fp_ddm/train.py \
    training.epochs=1 \
    training.samples=64 \
    training.batch_size=8 \
    training.num_workers=0 \
    training.max_minutes=0 \
    model.modes=4 \
    model.width=8 \
    model.layers=1 \
    training.output_dir=outputs/fp_ddm/train_smoke
```

Distributed training uses the same entry point:

```bash
torchrun --standalone --nproc_per_node=4 \
    examples/cfd/fp_ddm/train.py \
    training.output_dir=outputs/fp_ddm/train_distributed
```

Training writes the latest resumable state to `checkpoints/latest` and the
lowest-validation-loss model to `checkpoints/best`. Resume with
`training.resume_dir=<path-to-latest>`.

## Run FP-DDM

Run the finite-volume reference path first:

The `fem` solver key is retained from the original scripts; its local reference
implementation is the matrix-free finite-volume stencil in `thermal.py`.

```bash
python examples/cfd/fp_ddm/run_fpddm.py \
    run.solver=fem \
    domain.rows=3 \
    domain.columns=3 \
    run.max_iterations=10 \
    run.visualize=false \
    run.output_dir=outputs/fp_ddm/fem
```

Use a trained FNO checkpoint for neural local solves:

```bash
python examples/cfd/fp_ddm/run_fpddm.py \
    run.solver=fno \
    run.checkpoint_dir=outputs/train/checkpoints/best \
    domain.rows=3 \
    domain.columns=3 \
    run.max_iterations=3 \
    run.visualize=false \
    run.output_dir=outputs/fp_ddm/fno
```

The checkpoint is used directly; there is no separate fine-tuning phase.
Physics-guided test-time adaptation is optional and disabled by default. Enable
it with `run.ttt_steps=<steps>`.

Each run writes the resolved Hydra configuration, per-iteration metrics,
NumPy reference fields when requested, and a JSON summary. Visualization also
writes field images, iteration series, and MP4 animations.

## Configuration

The main configuration groups are:

| Group | Purpose |
| --- | --- |
| `model` | PhysicsNeMo FNO architecture and nondimensionalization |
| `dataset` | Synthetic local conductivity, boundary, and source fields |
| `training` | Epochs, sample count, optimizer, logging, and checkpoints |
| `domain` | Patch grid, patch resolution, overlap, and outer boundary |
| `fem` | Matrix-free reference-solver convergence settings |
| `run` | Local solver, interface update, TTT, and Schwarz stopping |
| `visualization` | Plot ranges and animation frame rate |

The supplied workload uses zero heat source, matching the original FP-DDM
setup. Nonzero `dataset.q_min` and `dataset.q_max` values are supported by both
the thermal residual and the finite-volume solver. The synthetic global layout
currently requires a square patch grid with square patches.

## Reproduced Baseline

The following baseline was measured on July 2, 2026, with one NVIDIA A100 80 GB
GPU, seed 10, the 4.20-million-parameter default FNO, a configured sample count
of 500,000, batch size 256, and 10 epochs. The split contained
450,000 on-the-fly training problems and 25,000 fixed problems for each of
validation and test. Training completed in 5 minutes 55 seconds. The best
validation physics metric was 0.1769 at epoch 9.

The rollout used a 3-by-3 patch decomposition, producing a 90-by-90 global
field, with the default parallel interface update and no test-time adaptation.
NRMAE is mean absolute error divided by the reference temperature range.

| Run | Iterations | NRMAE | R-squared | Interface RMSE |
| --- | ---: | ---: | ---: | ---: |
| FNO measured setting | 3 | 0.0643 | 0.8777 | 6.48 |
| FNO forced long rollout | 50 | 0.3107 | -1.2657 | 0.837 |
| Finite-volume reference | 50 | 0.00591 | 0.9988 | 0.762 |

The three-iteration FNO run completed in 6.87 seconds, including synthetic
layout generation, the full-domain reference solve, checkpoint loading, and
CUDA warmup. Its normalized PDE RMSE decreased from 9.59 to 7.95. The
50-iteration stress run is intentionally reported because it exposes an
important limitation: lower interface disagreement did not imply lower error
for this checkpoint. Choose the Schwarz stopping configuration using held-out
problems rather than assuming monotonic neural-rollout accuracy.

## Validation And Scope

The automated tests cover data orientation, the source term, FNO output and
boundary behavior, mixed-convergence finite-volume batches, interface exchange,
PhysicsNeMo checkpoint round trips, test-time adaptation state restoration, and
an end-to-end Schwarz run.

This is a research example of the FP-DDM algorithm, not a validated
production-scale thermal simulator. A smoke checkpoint only verifies execution.
Neural-solver accuracy must be established with a sustained training run and
reported together with its exact configuration. Scaling to very large global
problems requires distributed ownership of subdomains and interfaces; this
example currently batches local solves on one process and retains the assembled
global fields for diagnostics.

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794)
