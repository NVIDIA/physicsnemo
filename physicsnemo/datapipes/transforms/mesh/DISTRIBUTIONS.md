# Distribution-Parametrized Mesh Augmentations

## Why

Mesh augmentations (`RandomScaleMesh`, `RandomTranslateMesh`,
`RandomRotateMesh`) need to sample random parameters (scale factors,
translation offsets, rotation angles) on every call.  A uniform
distribution is the simplest choice but is not always the best, nor
should we lock into that design explicitly.  Other alternatives exist:

- **Gaussian** (`Normal`) concentrates samples near a center value,
  making it ideal for small perturbations around an identity
  transformation (e.g. scale factors near 1.0, small angles near 0).
- **Laplace** has a sharper peak than Gaussian but heavier tails,
  producing most samples near the center with occasional large ones.
- **Cauchy** has very heavy tails, useful when rare extreme
  augmentations are desirable.
- **LogNormal** is positive-valued by construction, making it a
  natural fit for scale factors that must stay positive.
- **Exponential**, **Gumbel**, **Weibull** cover various skewed or
  extreme-value scenarios.

Rather than adding bespoke parameters for each distribution family,
which would be cumbersome and create too much code to maintain,
the augmentations accept any `torch.distributions.Distribution`
object directly.  This delegates the full parametric flexibility of
PyTorch's distributions library to the user with zero custom
abstractions.

## How It Works:  Inverse cumulative distributions

`torch.distributions.Distribution.sample()` does **not** accept a
`torch.Generator`.  This is a problem because the augmentation
pipeline relies on seeded generators for reproducibility, which is
essential in ML pipelines.

We solve this with the **inverse transform method** (a.k.a. inverse
CDF / quantile-function sampling):

1. Draw `U ~ Uniform(0, 1)` using `torch.rand(generator=generator)`.
   This step is reproducible because `torch.rand` accepts a generator,
   which is seeded.
2. Compute `X = distribution.icdf(U)`.  The inverse CDF transforms
   the uniform variate into a sample from the target distribution.

By the probability integral transform, `X` is exactly distributed
according to `distribution`.  Reproducibility comes from step 1:
the same generator seed always produces the same `U`, and `icdf` is
a deterministic function.

For distributions that do **not** implement `icdf`, the code falls
back to `distribution.sample()` and emits a warning that generator
reproducibility is lost.  In practice this is only a small subset of
distributions.

## Reproducibility

Reproducibility flows from the `DataLoader`.  The loader seeds a
master `torch.Generator` and passes it to
`MeshDataset.set_generator(parent_gen)`, which forks the parent
into independent children — one for the reader and one per
transform.  `MeshDataset.set_epoch(epoch)` reseeds every child
with `initial_seed() + epoch` so each epoch is different but
deterministic.  Deterministic transforms silently ignore both calls.

For standalone usage outside a `DataLoader`, call `set_generator`
on the transform directly:

```python
aug = RandomScaleMesh(distribution=D.Normal(1.0, 0.05))
aug.set_generator(torch.Generator().manual_seed(42))
result = aug(mesh)  # reproducible
```


## Python Usage

```python
import torch
import torch.distributions as D
from physicsnemo.datapipes.transforms.mesh import (
    RandomScaleMesh,
    RandomTranslateMesh,
    RandomRotateMesh,
)
```

### RandomScaleMesh

```python
# Default: Uniform(0.9, 1.1)
aug = RandomScaleMesh()

# Gaussian perturbation around identity scale
aug = RandomScaleMesh(distribution=D.Normal(loc=1.0, scale=0.05))
aug.set_generator(torch.Generator().manual_seed(42))

# LogNormal (always positive, centered near 1)
aug = RandomScaleMesh(distribution=D.LogNormal(loc=0.0, scale=0.1))
```

### RandomTranslateMesh

```python
# Default: Uniform(-0.1, 0.1) per axis (IID)
aug = RandomTranslateMesh()

# Laplace offsets (sharp peak, heavy tails)
aug = RandomTranslateMesh(distribution=D.Laplace(loc=0.0, scale=0.02))
aug.set_generator(torch.Generator().manual_seed(42))

# Per-axis control via batched distribution
aug = RandomTranslateMesh(
    distribution=D.Uniform(
        torch.tensor([-0.1, -0.2, -0.3]),
        torch.tensor([ 0.1,  0.2,  0.3]),
    ),
)

# Per-axis Gaussian with different scales
aug = RandomTranslateMesh(
    distribution=D.Normal(
        loc=torch.zeros(3),
        scale=torch.tensor([0.01, 0.02, 0.05]),
    ),
)
```

### RandomRotateMesh

```python
# Default: Uniform(-pi, pi) axis-aligned rotation
aug = RandomRotateMesh()

# Concentrated small-angle perturbations
aug = RandomRotateMesh(distribution=D.Normal(loc=0.0, scale=0.1))
aug.set_generator(torch.Generator().manual_seed(42))

# Only rotate about z-axis, Laplace angle distribution
aug = RandomRotateMesh(
    axes=["z"],
    distribution=D.Laplace(loc=0.0, scale=0.5),
)

# Uniform SO(3) (ignores distribution, uses quaternion method)
aug = RandomRotateMesh(mode="uniform")
```

## YAML / Hydra Usage

Distributions can be constructed inline using Hydra's `_target_`
syntax:

```yaml
# Gaussian scale perturbation
- _target_: ${dp:RandomScaleMesh}
  distribution:
    _target_: torch.distributions.Normal
    loc: 1.0
    scale: 0.05

# Laplace translation
- _target_: ${dp:RandomTranslateMesh}
  distribution:
    _target_: torch.distributions.Laplace
    loc: 0.0
    scale: 0.02

# Small-angle Gaussian rotation about z only
- _target_: ${dp:RandomRotateMesh}
  axes: ["z"]
  distribution:
    _target_: torch.distributions.Normal
    loc: 0.0
    scale: 0.1
```

For per-axis batched distributions in YAML, pass list parameters:

```yaml
- _target_: ${dp:RandomTranslateMesh}
  distribution:
    _target_: torch.distributions.Uniform
    low: [-0.1, -0.2, -0.3]
    high: [0.1, 0.2, 0.3]
```

## Supported Distributions

The ICDF method works with any `torch.distributions.Distribution`
that implements `icdf()`.  The table below summarises support for the
most commonly used distributions.

| Distribution    | `icdf` | Generator-reproducible | Typical use case                         |
|-----------------|--------|------------------------|------------------------------------------|
| `Uniform`       | Yes    | Yes                    | Bounded ranges (default behaviour)       |
| `Normal`        | Yes    | Yes                    | Small perturbations around a center      |
| `Laplace`       | Yes    | Yes                    | Sharper peak, heavier tails than Gaussian |
| `Cauchy`        | Yes    | Yes                    | Very heavy tails, rare extremes          |
| `LogNormal`     | Yes    | Yes                    | Positive-only (e.g. scale factors)       |
| `Exponential`   | Yes    | Yes                    | One-sided positive values                |
| `Gumbel`        | Yes    | Yes                    | Extreme-value modelling                  |
| `Weibull`       | Yes    | Yes                    | Flexible shape for positive values       |
| `Poisson`       | No     | **No** (fallback)      | Discrete; generator ignored with warning |
| `Gamma`         | No     | **No** (fallback)      | Positive continuous; no closed-form ICDF |
| `Dirichlet`     | No     | **No** (fallback)      | Simplex-valued; no scalar ICDF           |

Distributions without `icdf` will still work via
`distribution.sample()`, but the `torch.Generator` is **not** used
for those draws.  A `UserWarning` is emitted in this case, and
datapipe reproducibility is not possible at the generator level.

## Choosing a Distribution

| Goal                                         | Recommended distribution              |
|----------------------------------------------|---------------------------------------|
| Bounded range `[a, b]`                       | `Uniform(a, b)`                       |
| Small perturbations around a center `c`      | `Normal(loc=c, scale=sigma)`          |
| Sharper peak + occasional large values       | `Laplace(loc=c, scale=b)`             |
| Very heavy tails (rare extreme augmentation) | `Cauchy(loc=c, scale=gamma)`          |
| Strictly positive parameter (e.g. scale)     | `LogNormal(loc=mu, scale=sigma)`      |
| One-sided perturbation from zero             | `Exponential(rate=lambda)`            |
| Per-axis different parameters                | Batched distribution, e.g. `Normal(loc=tensor([...]), scale=tensor([...]))` |
