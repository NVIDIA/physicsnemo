<!-- markdownlint-disable -->
# Functional Generative Networks for Weather

This example scaffolds a PhysicsNeMo-native implementation path for Functional
Generative Networks (FGN) for probabilistic weather forecasting, following the
style of the other `examples/weather/*` recipes.

The intended target is WeatherNext 2, which uses an FGN architecture for global
ensemble weather forecasting.

The current version is intentionally an MVP:

- a latent-conditioned generative model implemented as a PhysicsNeMo `Module`
- a synthetic autoregressive weather dataset for smoke testing
- fair-CRPS training over latent ensemble samples
- autoregressive inference with stochastic trajectory rollout
- Hydra configs and a small pytest smoke test

The example follows PhysicsNeMo and Earth2Studio variable conventions such as
`u10m`, `v10m`, `t2m`, `msl`, `z500`, `q850`, and `w300`.

## Layout

```text
examples/weather/fgn/
  README.md
  requirements.txt
  train.py
  inference.py
  test_training.py
  config/
  datasets/
  utils/
  scripts/
```

## Quick Start

Install any example-specific requirements:

```bash
pip install -r requirements.txt
```

Run a short synthetic training job:

```bash
python train.py --config-name test_fgn
```

Run stochastic inference from the latest checkpoint:

```bash
python inference.py --config-name inference_fgn
```

Run the smoke test:

```bash
pytest test_training.py
```

## Configs

The main entrypoints are:

- `config/fgn.yaml`: default synthetic training config
- `config/test_fgn.yaml`: short smoke-test config
- `config/inference_fgn.yaml`: stochastic rollout inference config

The model config exposes the main FGN-specific choices:

- `model.latent_dim`: dimensionality of the latent perturbation
- `training.loss.num_samples`: number of latent samples per training example
- `training.loss.mse_weight`: optional stabilizing loss on the ensemble mean

## Current Scope

This first implementation targets a synthetic regional grid dataset so the
recipe is runnable without a large external data pipeline. It does not yet
implement:

- WeatherNext 2 global data ingestion
- mixed 1-hour / 6-hour variable cadence
- multi-checkpoint deep-ensemble orchestration
- large-scale distributed training

Those are the planned next steps.
