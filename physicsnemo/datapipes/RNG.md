# Datapipe RNG & Reproducibility

Deterministic data loading is opt-in: pass `seed=` to `DataLoader` and the
entire pipeline — sampler, reader, and every stochastic transform — becomes
reproducible across runs.

If no seed is passed, all random operations will fall back to their default
behavior - which may still be deterministic if you have set seeds carefully
in pytorch, and executed operations carefully.  In short, using a `seed` in the
`DataLoader` will deploy `torch.Generator` objects at all object-level random
calls, making each object sequentially deterministic.  Your whole pipeline
becomes reproducible. Not using a seed means you rely on globally set behavior.

## Quick start

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True, seed=42)

for epoch in range(n_epochs):
    loader.set_epoch(epoch)   # vary randomness per epoch, still deterministic
    for batch in loader:
        ...
```

## How it works

### Generator forking (`_rng.py`)

The system derives independent `torch.Generator` streams from a single
master seed using `fork_generator(parent, n)`.  Each child is seeded with
`parent.initial_seed() + i + 1`, so children are independent of each other
and stable across runs.  Children are created on the **same device** as the
parent.

### DataLoader

When `seed` is set the DataLoader:

1. Creates a CPU master generator: `torch.Generator().manual_seed(seed)`.
2. Forks it into **2 children**:
   - **Child 0 → sampler** — passed to `RandomSampler(generator=...)`.
   - **Child 1 → dataset** — passed via `dataset.set_generator(...)`.

### Dataset (TensorDict path)

`Dataset.set_generator(generator)` flattens its transform pipeline
(unwrapping `Compose` if present) and forks into
`1 + len(flat_transforms)` children:

- **Child 0 → reader** — passed via `reader.set_generator(...)`.
- **Children 1..N → transforms** (1-to-1 mapping; deterministic transforms
  silently ignore theirs).

If the dataset's `target_device` differs from the child generator's
device, a new generator is created on `target_device` and seeded from
the child's `initial_seed()`.

### MeshDataset

`MeshDataset.set_generator(generator)` follows the same pattern as
`Dataset`: forks into `1 + len(transforms)` children, distributing to
the reader and each transform with device alignment.

### MultiDataset

`MultiDataset.set_generator(generator)` forks into
`len(sub_datasets)` children and calls `set_generator` on each
sub-dataset.

### Epoch reseeding

`DataLoader.set_epoch(epoch)` propagates to the sampler and dataset.
Each component with a generator reseeds it with
`initial_seed() + epoch`, producing a different but deterministic
random sequence every epoch.

## Generator tree

```text
DataLoader(seed=S)
│
├── master = Generator().manual_seed(S)
│
├── fork_generator(master, 2)
│   ├── child[0]  (seed S+1) ──► Sampler
│   └── child[1]  (seed S+2) ──► Dataset / MeshDataset / MultiDataset
│                                  │
│                                  ├── fork_generator(child[1], 1+N_transforms)
│                                  │   ├── child[0] (seed S+3) ──► Reader
│                                  │   ├── child[1] (seed S+4) ──► Transform 0
│                                  │   ├── child[2] (seed S+5) ──► Transform 1
│                                  │   └── ...
```

For `MultiDataset`, the fork distributes one child per sub-dataset,
and each sub-dataset then re-forks internally for its reader and
transforms.

## Device management

`torch.Generator` objects are device-bound and cannot be moved in-place.
Every boundary where a generator might cross devices contains explicit
re-creation logic:

| Location | What happens |
|---|---|
| `fork_generator` | Creates children on `parent.device` |
| `Dataset.set_generator` | If `target_device != child.device`, creates a new generator on `target_device` seeded from the child |
| `MeshDataset.set_generator` | Same device-alignment logic as `Dataset` |
| `MeshTransform.to(device)` | Creates a new generator on `device`, seeded from the original's `initial_seed()` |
| `_sample_distribution` | Draws uniforms on `generator.device` |

All random draws (`torch.rand`, `torch.randn`, `torch.randint`) pass
`device=generator.device` to stay on the correct device.

## Stochastic transforms

### Opting in

Both `Transform` (TensorDict) and `MeshTransform` (Mesh) base classes
define the same generator protocol:

- **`stochastic`** — property; `True` when `self._generator` exists.
- **`set_generator(g)`** — assigns `g` if stochastic; no-op otherwise.
- **`set_epoch(epoch)`** — reseeds with `initial_seed() + epoch`.

To make a transform stochastic, declare
`self._generator: torch.Generator | None = None` in `__init__`.
Deterministic transforms never declare it, so all three methods are
silent no-ops.

### TensorDict stochastic transforms

- **`SubsamplePoints`** — declares `_generator` and passes it to
  `torch.randperm`, `torch.multinomial`, and
  `poisson_sample_indices_fixed`.

### Mesh stochastic transforms

- **`RandomScaleMesh`**, **`RandomTranslateMesh`**,
  **`RandomRotateMesh`** — sample augmentation parameters from
  `torch.distributions.Distribution` objects via ICDF + generator.
- **`SubsampleMesh`** — uses `torch.randperm` / `poisson_sample_indices_fixed`.

### `Compose`

`Compose.set_generator(generator)` forks and distributes one child per
child transform.  `Compose.set_epoch(epoch)` propagates to all children.
When used inside `Dataset`, the dataset flattens `Compose` and assigns
forks per leaf transform directly; `Compose`'s own methods are for
standalone use.

## Readers

The `Reader` base class defines no-op `set_generator` / `set_epoch`.
Readers that use randomness override them:

| Reader | Randomness | Generator support |
|---|---|---|
| `MeshReader` | `torch.randint` (contiguous block selection) | Yes |
| `DomainMeshReader` | `torch.randint` | Yes |
| `NumpyReader` | `torch.randint` (coordinated subsampling) | Yes |
| `ZarrReader` | `torch.randint` | Yes |
| `TensorStoreZarrReader` | `torch.randint` | Yes |
| `HDF5Reader` | None | No-op (inherited) |
| `VTKReader` | None | No-op (inherited) |

## Current limitations

- `DistributedSampler` manages its own seed internally; when using it,
  pass `seed=` at `DistributedSampler` construction time rather than
  relying on DataLoader's seed propagation.
- Legacy datapipes (`cae/`, `gnn/`, `climate/`, `healpix/`,
  `benchmarks/`) are not wired into the generator protocol.
