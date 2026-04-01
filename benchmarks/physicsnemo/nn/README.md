# PhysicsNeMo NN Benchmarks

This directory contains ASV benchmarks for `physicsnemo.nn`.

## Functional Benchmark Workflow

1. Implement or update the functional `FunctionSpec`.
2. Add representative `make_inputs_forward(device=...)` cases.
3. Add `make_inputs_backward(device=...)` when backward timing is meaningful.
4. Register the `FunctionSpec` in `benchmarks/physicsnemo/nn/functional/registry.py`.
5. Run ASV and regenerate plots.

## Key Files

- Benchmark registry:
  - `benchmarks/physicsnemo/nn/functional/registry.py`
- ASV benchmark runner:
  - `benchmarks/physicsnemo/nn/functional/benchmark_functionals.py`
- Plot generation:
  - `benchmarks/physicsnemo/nn/functional/plot_functional_benchmarks.py`

## Relevant Standards / APIs

- Functional benchmark conventions:
  - `CODING_STANDARDS/FUNCTIONAL_APIS.md`
- `FunctionSpec` behavior:
  - `physicsnemo/core/function_spec.py`

## Common Commands

Run all configured benchmarks:

```bash
./benchmarks/run_benchmarks.sh
```

Run selected functionals:

```bash
PHYSICSNEMO_ASV_FUNCTIONALS=Interpolation,RadiusSearch ./benchmarks/run_benchmarks.sh
```

Run selected benchmark phases:

```bash
PHYSICSNEMO_ASV_PHASES=forward ./benchmarks/run_benchmarks.sh
PHYSICSNEMO_ASV_PHASES=forward,backward ./benchmarks/run_benchmarks.sh
```

Override benchmark device:

```bash
PHYSICSNEMO_ASV_DEVICE=cuda ./benchmarks/run_benchmarks.sh
PHYSICSNEMO_ASV_DEVICE=cpu ./benchmarks/run_benchmarks.sh
```

## Benchmark Image Outputs

Plots are written to:

- `docs/img/nn/functional/<category>/<functional_name>/benchmark_forward.png`
- `docs/img/nn/functional/<category>/<functional_name>/benchmark_backward.png`
