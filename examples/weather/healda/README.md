# HealDA — AI-based Data Assimilation on the HEALPix Grid

> **🏗️ This recipe is under active construction. 🏗️**
> Structure and functionality are subject to changes.

HealDA is a stateless assimilation model that produces a single
global weather analysis from conventional and satellite
observations. It operates on a HEALPix level-6 padded XY grid
and outputs ERA5-compatible atmospheric variables.

## Setup

Start by installing PhysicsNeMo (if not already installed) with
the `healda` optional dependency group, along with the packages
in `requirements.txt`. Then, copy this folder
(`examples/weather/healda`) to a system with a GPU available.
Also, prepare a dataset that can serve training data according
to the protocols outlined in the
[Generalized Data Loading](#generalized-data-loading) section
below.

### Normalization statistics

Per-sensor observation stats (`configs/normalizations/*.csv`)
and the ERA5 stats table (`configs/era5_13_levels_stats.csv`)
ship with this recipe rather than the installed package, since
they are training-set specific. Point the datapipe at them by
setting `HEALDA_STATS_DIR` before importing
`physicsnemo.experimental.datapipes.healda`:

```bash
export HEALDA_STATS_DIR=$(pwd)/configs
```

If unset, sensor configs fall back to zero-mean / unit-std
(useful for tests and structural checks); the ERA5 stats
loader will raise instead, since there is no sensible default.

## Generalized Data Loading

The `physicsnemo.experimental.datapipes.healda` package provides
a composable data loading pipeline with clear extension points.
The architecture separates components into loaders, transforms,
datasets, and sampling infrastructure.

### Architecture

```text
ObsERA5Dataset(era5_data, obs_loader, transform)
  |  Temporal windowing via FrameIndexGenerator
  |  __getitems__ -> get() per index -> transform.transform()
  v
RestartableDistributedSampler (stateful distributed sampling with checkpointing)
  |
DataLoader (pin_memory, persistent_workers)
  |
prefetch_map(loader, transform.device_transform)
  |
Training loop (GPU-ready batch)
```

### Key Protocols

Custom data sources and transforms plug in via these protocols
(see `physicsnemo.experimental.datapipes.healda.protocols`):

**`ObsLoader`** — the observation loading interface:

```python
class MyObsLoader:
    async def sel_time(self, times):
        """Return {"obs": [pa.Table, ...]}"""
        ...
```

**`Transform`** / **`DeviceTransform`** — two-stage batch
processing:

```python
class MyTransform:
    def transform(self, times, frames):
        """CPU-side: normalize, encode obs, time features."""
        ...

    def device_transform(self, batch, device):
        """GPU-side: move to device, compute obs features."""
        ...
```

### Provided Implementations

| Component | Module | Description |
|---|---|---|
| `ObsERA5Dataset` | `dataset` | ERA5 state + observations |
| `UFSUnifiedLoader` | `loaders.ufs_obs` | Parquet obs loader |
| `ERA5Loader` | `loaders.era5` | Async ERA5 zarr loader |
| `ERA5ObsTransform` | `transforms.era5_obs` | Two-stage transform |
| `RestartableDistributedSampler` | `samplers` | Stateful distributed sampler |
| `prefetch_map` | `prefetch` | CUDA stream prefetching |

All modules above are under
`physicsnemo.experimental.datapipes.healda`.

### Writing a Custom Observation Loader

Implement `async def sel_time(times)` returning a dict with
observation data per timestamp:

```python
class GOESRadianceLoader:
    def __init__(self, data_path, channels):
        self.data_path = data_path
        self.channels = channels

    async def sel_time(self, times):
        tables = []
        for t in times:
            table = self._load_goes_radiances(t)
            tables.append(table)
        return {"obs": tables}
```

Then pass it to the dataset:

```python
from physicsnemo.experimental.datapipes.healda import (
    ObsERA5Dataset,
)
from physicsnemo.experimental.datapipes.healda.transforms.era5_obs import (
    ERA5ObsTransform,
)
from physicsnemo.experimental.datapipes.healda.configs.variable_configs import (
    VARIABLE_CONFIGS,
)

dataset = ObsERA5Dataset(
    era5_data=era5_xr["data"],
    obs_loader=GOESRadianceLoader(...),
    transform=ERA5ObsTransform(sensors=["goes"], ...),
    variable_config=VARIABLE_CONFIGS["era5"],
)
```

### Putting It Together

A complete training pipeline wires together all the
components — dataset, sampler, DataLoader, and GPU prefetch:

```python
import torch
from torch.utils.data import DataLoader

from physicsnemo.experimental.datapipes.healda import (
    ObsERA5Dataset,
    RestartableDistributedSampler,
    identity_collate,
    prefetch_map,
)
from physicsnemo.experimental.datapipes.healda.loaders.ufs_obs import (
    UFSUnifiedLoader,
)
from physicsnemo.experimental.datapipes.healda.transforms.era5_obs import (
    ERA5ObsTransform,
)
from physicsnemo.experimental.datapipes.healda.configs.variable_configs import (
    VARIABLE_CONFIGS,
)

sensors = ["atms", "mhs", "conv"]

# 1. Build loaders
obs_loader = UFSUnifiedLoader(
    data_path="/path/to/processed_obs",
    sensors=sensors,
    obs_context_hours=(-21, 3),
)
transform = ERA5ObsTransform(
    variable_config=VARIABLE_CONFIGS["era5"],
    sensors=sensors,
)

# 2. Build dataset
dataset = ObsERA5Dataset(
    era5_data=era5_xr["data"],
    obs_loader=obs_loader,
    transform=transform,
    variable_config=VARIABLE_CONFIGS["era5"],
    split="train",
)

# 3. Sampler + DataLoader
sampler = RestartableDistributedSampler(
    dataset, rank=rank, num_replicas=world_size,
)
sampler.set_epoch(0)
dataloader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=2,
    num_workers=8,
    collate_fn=identity_collate,
    pin_memory=True,
    persistent_workers=True,
)

# 4. GPU prefetch (hides CPU→GPU transfer behind training)
device = torch.device("cuda")
loader = prefetch_map(
    dataloader,
    lambda batch: transform.device_transform(batch, device),
    queue_size=1,
)

# 5. Training loop — batches arrive GPU-ready
for batch in loader:
    loss = model(batch)
    ...
```
