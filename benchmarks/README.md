# PhysicsNeMo Benchmarks with ASV

This directory contains ASV-based benchmarks for PhysicsNeMo. Benchmarks are
discovered from the `benchmarks/` tree and configured via `asv.conf.json` in the
repository root.

Resources:

* [ASV documentation](https://asv.readthedocs.io/en/latest/index.html)

## Running a benchmark

Run all benchmarks from the repo root:

```sh
./benchmarks/run_benchmarks.sh
```

Note: the first run may take a while because ASV builds its virtual environment.

Run a subset by name or regex:

```sh
./benchmarks/run_benchmarks.sh -b knn
```

## Publishing and viewing results

Publish results to the local HTML report:

```sh
asv publish
```

Preview the report in a local web server:

```sh
asv preview
```

The generated site is written to `.asv/html/` (open `index.html` if you prefer).

## Adding a new benchmark

1. Add a new file under `benchmarks/` following the package structure (for
   example, `benchmarks/physicsnemo/nn/neighbors/my_benchmark.py`).
2. Define a benchmark class and at least one `time_*` method.
   See [documentation](https://asv.readthedocs.io/en/latest/writing_benchmarks.html#benchmark-types)
   for available benchmark types.
3. Use `setup()` to create inputs and keep benchmarks deterministic.
   See [documentation](https://asv.readthedocs.io/en/latest/writing_benchmarks.html#benchmark-attributes)
   for available benchmark attributes.

Example:

```py
import torch


class MyOpBenchmark:
    params = [1024, 4096]
    param_names = ["n"]

    def setup(self, n: int) -> None:
        self.x = torch.randn(n, n, device="cuda")

    def time_my_op(self, n: int) -> None:
        _ = self.x @ self.x
        torch.cuda.synchronize()
```
