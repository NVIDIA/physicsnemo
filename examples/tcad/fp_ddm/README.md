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
neighboring patches exchange their overlap values and repeat local solves until a
configured stopping condition or iteration budget is reached.

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
pip install -r examples/tcad/fp_ddm/requirements.txt
```

SciPy and OpenSimplex provide the smooth synthetic material layouts. Matplotlib
and ImageIO are only needed when `run.visualize=true`.

## Train The FNO

The default configuration generates local thermal problems on the fly and
trains one model directly with the physics-informed objective:

```bash
python examples/tcad/fp_ddm/train.py
```

All settings may be changed with Hydra overrides. Distributed training uses the
same entry point:

```bash
torchrun --standalone --nproc_per_node=4 \
    examples/tcad/fp_ddm/train.py \
    training.output_dir=outputs/fp_ddm/train_distributed
```

Training writes the latest resumable state to `checkpoints/latest` and the model
with the lowest isolated-patch validation loss to `checkpoints/best`. The latter
is not necessarily the best checkpoint for a recursive Schwarz rollout, so
validate candidate checkpoints on representative decompositions. Resume with
`training.resume_dir=<path-to-latest>`.

## Run FP-DDM

Run the finite-volume reference path first:

The `fem` solver key is retained from the original scripts; its local reference
implementation is the matrix-free finite-volume stencil in `thermal.py`.

```bash
python examples/tcad/fp_ddm/run_fpddm.py \
    run.solver=fem \
    domain.rows=3 \
    domain.columns=3 \
    run.max_iterations=10 \
    run.visualize=false \
    run.output_dir=outputs/fp_ddm/fem
```

Use a trained FNO checkpoint for neural local solves:

```bash
python examples/tcad/fp_ddm/run_fpddm.py \
    run.solver=fno \
    run.checkpoint_dir=outputs/train/checkpoints/best \
    domain.rows=3 \
    domain.columns=3 \
    run.max_iterations=50 \
    run.visualize=false \
    run.output_dir=outputs/fp_ddm/fno
```

The checkpoint is used directly by default. Physics-guided test-time adaptation
is optional and disabled; enable it with `run.ttt_steps=<steps>`.

Each run writes the resolved Hydra configuration, per-iteration metrics,
and NumPy reference fields when requested. Visualization also writes field
images, iteration series, and MP4 animations.

The overlap RMSE is the stopping metric. Reaching `max_iterations` is an
iteration-budget limit, not convergence.

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

The supplied workload uses zero heat source, matching the original FP-DDM setup.
The synthetic global layout currently requires a square patch grid with square
patches. Full-domain reference solves are enabled by default and limited to at
most 26 subdomains. Larger requests emit a warning and continue without
reference metrics; disable them explicitly with `run.ground_truth=false`.

## Validation And Scope

Run the focused tests from the repository root:

```bash
pytest -q test/examples/test_fp_ddm.py
```

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
