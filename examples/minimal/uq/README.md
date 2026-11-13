# Composable Uncertainty Quantification Examples

These examples use `Mean(mode="tensor")` and `Variance(mode="tensor")`.
The default constructors preserve the legacy metric behavior. Models, sampling,
feature extraction, and application decisions remain separate from statistics.

## Run

```bash
python examples/minimal/uq/streaming_predictions.py
python examples/minimal/uq/force_statistics.py
python examples/minimal/uq/feature_scoring.py
python examples/minimal/uq/geometry_scoring.py
python examples/minimal/uq/weighted_mean.py
torchrun --standalone --nproc-per-node=4 examples/minimal/uq/distributed_moments.py
```

The first example requires `gpytorch`. Add `--conformal` to demonstrate pointwise
conformal calibration using `PointwiseCalibrator.update` and `aux={"sigma": ...}`
without modifying those APIs. The small untrained models illustrate interfaces,
not calibrated coverage or out-of-distribution (OOD) detection quality.

## Statistical Axes Determine Communication

The distributed examples use a 2 × 2 mesh over the member and space dimensions
on four GPUs. They also run on two GPUs:

- By default, they distribute members (2 × 1)
- `--member-groups 1` distributes space (1 × 2)

For example:

```bash
torchrun --standalone --nproc-per-node=2 examples/minimal/uq/distributed_moments.py
torchrun --standalone --nproc-per-node=2 \
  examples/minimal/uq/distributed_moments.py --member-groups 1
```

The four-GPU example uses independently sharded simulations and space.
It computes root mean square (RMS) over simulations and space at each time,
and pointwise variance over simulations and time. The direct path reduces
simulations on each update and keeps one spatial field of statistics. The
compound path retains separate time
moments for every simulation, then reduces the simulation axis at the end.
Both give the same population variance for equal time counts. The compound path
uses more state but defers simulation communication. You do not need an
accumulator-local or accumulator-global variant. Unequal time counts would
require a weighted composition, which is outside this initial API.

All participating ranks follow the same update schedule. Empty member ranks
must supply zero-length shards of each global observation batch. Skipping their
updates would leave peers waiting in collectives. The example verifies matching
statistical results but does not claim a performance advantage.

## Gaussian Process (GP) and Observable Semantics

For a GP mixture, total predictive variance is the mean of component variances
plus the population variance of component means. A single GP supplies both
moments directly. For deterministic member predictions, transform each member
into the desired observable before accumulating its moments. For GP field
observables, use cross-point covariance or joint predictive samples. Marginal
pointwise variances alone cannot propagate correlations.

`force_statistics.py` computes forces through autograd before accumulation. It
checks that statistics retain no computation graph while the original forces can
still participate in a training loss. The application controls its energy models
and gradient context. This example needs no GP or application-specific toolkit.

## Scoring and Persistence

`feature_scoring.py` obtains optional GeoTransolver geometry tokens, pools them
in the recipe, fits reusable distance and density scorers, and applies a validation
threshold. Generic scorers do not depend on geometry utilities. For mesh
descriptors, explicitly import `physicsnemo.experimental.uq.ood.geometry`.
Persist checkpoint identity, pooling, preprocessing, feature schema, and threshold
alongside scorer state. The API reference includes save and restore examples.

`geometry_scoring.py` uses synthetic primitive meshes and explicitly imports
optional mesh descriptors. It saves density tensor state, application metadata,
and policy in one weights-only PyTorch artifact. It also validates feature
identity and checks restored scores and decisions.

## Distributed Calibration

Run from the repo root:

```bash
torchrun --standalone --nproc-per-node=4 examples/minimal/uq/distributed_calibration.py
```

A small synthetic-case iterator keeps data construction separate from calibration.
Its full arrays supply independent test references. Production data loading should
supply aligned local predictions and targets. This example gathers only the final
mean and total variance with `full_tensor()` on all ranks for each complete case,
then fits the calibrators on rank 0. It compares `PointwiseCalibrator`,
`FunctionalBandCalibrator`, and `RiskControlCalibrator` against direct inputs
and checks their existing save and load APIs.
Targets must use the same output units and point ordering as the predictions.
An ensemble member or spatial shard is not a separate calibration case.
The deployment process controls distribution and loading of the resulting
predictor artifact.

The four-GPU `distributed_moments.py` example also streams one member for each update
with unequal member counts (3 and 2) and spatial partitions (3 and 7). All ranks
follow the same update schedule. Exhausted groups contribute zero-length shards
with shard shapes for each step. Skipping a collective update can deadlock.

## Adding a Statistic

`Mean(mode="tensor")` and `Variance(mode="tensor")` compose a `TensorAccumulator`
with `MeanRule` and `VarianceRule`. The accumulator owns:

- Detached input preparation
- Precision
- Layout checks for each field
- Observation counts
- Reset behavior
- State ownership

Rules provide only `summarize`, `merge`, and `result`. Both summarization and result
calculation receive `reduce_sum`, which completes pending distributed sums before
nonlinear operations and preserves unreduced spatial shards.

`weighted_mean.py` implements an external `WeightedMeanRule` with two sufficient
statistics: weighted sum and weight sum. It uses the same accumulator without
subclassing it or changing the library. Auxiliary tensors must match the shape
and layout of `values`. Broadcast weights explicitly if needed. Rule state can
contain multiple tensor shapes and Python integers. Rules must be pure. Two rules
are equal only when their statistical configuration matches. A frozen dataclass
works for these examples.

```python
from physicsnemo.metrics.general.tensor_accumulator import TensorAccumulator
from weighted_mean import WeightedMeanRule  # Defined in this example directory

stats = TensorAccumulator(WeightedMeanRule())
stats.update(predictions, weights=weights, dim=0)
weighted_mean = stats.finalize()
```

The weighted example checks for invalid weights across the whole logical tensor,
so its `update` and `finalize` require every mesh rank to participate. These checks
add collectives. This example demonstrates reuse and correctness, not improved
performance. The generic accumulator does not make arbitrary statistical rules
distributable automatically. Each rule must have a valid merge operation for
independent observations and use completed reductions before nonlinear arithmetic.

Run the weighted-mean example tests with:

```bash
python -m pytest -o addopts="" examples/minimal/uq/test_weighted_mean.py
```

## Distributed Regression Tests

The library tests reuse the session's `distributed_mesh_2d` fixture. On two GPUs,
they swap the member and space roles to exercise both communication paths in
continuous integration (CI). On four GPUs, they also cover simultaneous member
and spatial distribution.
The tests check uneven and empty partitions, GP mixtures, weighted statistics,
and state ownership against direct tensor references.

```bash
torchrun --standalone --nproc-per-node=2 -m pytest -o addopts="" \
  test/domain_parallel/test_streaming_moments.py --multigpu-static -k cuda
torchrun --standalone --nproc-per-node=4 -m pytest -o addopts="" \
  test/domain_parallel/test_streaming_moments.py --multigpu-static -k cuda
```
