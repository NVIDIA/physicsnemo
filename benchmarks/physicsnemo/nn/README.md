# PhysicsNeMo NN Benchmarks

This directory contains ASV benchmarks for `physicsnemo.nn`.

For functionals, the benchmark flow is intentionally simple:

1. Implement or update the functional `FunctionSpec`.
2. Add representative `make_inputs(device=...)` cases to that `FunctionSpec`.
3. Register the `FunctionSpec` in `benchmarks/physicsnemo/nn/functional/registry.py`.
4. Run ASV and regenerate plots.

## Where to read more

- Functional benchmark rules and expectations:
  - `CODING_STANDARDS/FUNCTIONAL_APIS.md`
- `FunctionSpec` behavior and required hooks:
  - `physicsnemo/core/function_spec.py`

## Where to edit

- Benchmark registry (which functionals are benchmarked):
  - `benchmarks/physicsnemo/nn/functional/registry.py`
- ASV benchmark runner for functionals:
  - `benchmarks/physicsnemo/nn/functional/benchmark_functionals.py`
- Plot generation:
  - `benchmarks/physicsnemo/nn/functional/plot_functional_benchmarks.py`

## Example functionals to copy

- `physicsnemo/nn/functional/interpolation/interpolation.py`
- `physicsnemo/nn/functional/radius_search/radius_search.py`
- `physicsnemo/nn/functional/knn/knn.py`

## Common commands

Run benchmarks (repo root):

```bash
./benchmarks/run_benchmarks.sh
```

Run only selected functionals while iterating:

```bash
PHYSICSNEMO_ASV_FUNCTIONALS=Interpolation,RadiusSearch ./benchmarks/run_benchmarks.sh
```

Plots are written under:

- `docs/nn/functional/<functional_name>/benchmark.png`
