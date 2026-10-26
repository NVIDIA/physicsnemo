# Composable UQ examples

These examples use `Mean(mode="tensor")` and `Variance(mode="tensor")`.
The default constructors preserve the old metric behavior. Models, sampling,
feature extraction, and application decisions remain separate from statistics.

## Run

```bash
python examples/minimal/uq/streaming_predictions.py
python examples/minimal/uq/force_statistics.py
torchrun --standalone --nproc-per-node=4 examples/minimal/uq/distributed_moments.py
```

The first example requires `gpytorch`. Add `--conformal` after the conformal PR
stack is available; it uses `PointwiseCalibrator.update` and `aux={"sigma": ...}`
without modifying those APIs. The small untrained models illustrate interfaces,
not calibrated coverage or OOD detection quality.

## Statistical axes determine communication

The distributed example uses independently sharded simulations and space.
It computes RMS over simulations/space at each time and pointwise variance over
simulations/time. The direct path reduces simulations on each update and keeps
one spatial field of statistics. The compound path retains separate time
moments for every simulation, then reduces the simulation axis at the end.
Both give the same population variance for equal time counts. The compound path
uses more state but defers simulation communication. No accumulator-local/global
variant is needed. Unequal time counts would require a weighted composition,
which is outside this initial API.

All participating ranks follow the same update schedule. Empty member ranks
must supply zero-length shards of each global observation batch; skipping their
updates would leave peers waiting in collectives. The example validates equal
statistical results, not a performance advantage.

## GP and observable semantics

For a GP mixture, total predictive variance is the mean of component variances
plus the population variance of component means. A single GP supplies both
moments directly. For deterministic member predictions, transform each member
into the desired observable before accumulating its moments. For GP field
observables, use cross-point covariance or joint predictive samples: marginal
pointwise variances alone cannot propagate correlations.

`force_statistics.py` computes forces through autograd before accumulation. It
checks that statistics retain no computation graph while the original forces can
still participate in a training loss. Energy models and gradient context remain
application-owned; this example needs no GP or application-specific toolkit.

## Distributed calibration

After the conformal stack is present, run:

```bash
torchrun --standalone --nproc-per-node=4 examples/minimal/uq/distributed_calibration.py
```

This example gathers the final mean and total variance for each complete case,
then fits on rank zero. It compares Pointwise, FunctionalBand, and RiskControl
calibrators against direct inputs and checks their existing save/load APIs.
Targets must use the same output units and point ordering as the predictions.
An ensemble member or spatial shard is not a separate calibration case.
Deployment owns distribution/loading of the resulting predictor artifact.

The four-GPU `distributed_moments.py` example also streams one member per update
with unequal member counts (3 and 2) and spatial partitions (3 and 7). All ranks
follow the same update schedule; exhausted groups contribute zero-length shards
with per-step shard shapes. Skipping a collective update can deadlock.
