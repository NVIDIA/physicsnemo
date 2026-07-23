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
master seed using `fork_generator(parent, n)`.  Child *i* is seeded with
`derive_seed(parent.initial_seed(), i)` (SeedSequence mixing, see below),
so children are well-mixed, independent of each other, and stable across
runs — nearby master seeds or forks at different pipeline depths do not
produce overlapping child streams.  Children are created on the **same
device** as the parent.

For RNG that must be reproducible regardless of *execution order* (e.g.
reader subsampling, which runs on a pool of worker threads), `_rng.py`
also provides coordinate-based seeding:

- **`derive_seed(base_seed, *coords)`** — mixes a base seed with integer
  coordinates (typically `epoch` and sample `index`) into a single
  well-mixed 64-bit seed via `numpy.random.SeedSequence`.  The result
  depends only on the inputs, not on call order or thread.
- **`spawn_generator(base_seed, *coords, device=...)`** — returns a fresh
  `torch.Generator` seeded with `derive_seed(base_seed, *coords)`.

Because each call returns an independent generator seeded purely from its
coordinates, draws are reproducible irrespective of order and safe to
compute concurrently from multiple threads (no shared mutable state).

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
Stochastic transforms reseed their generators with
`derive_seed(base_seed, epoch)`, where `base_seed` was captured once at
`set_generator` time -- a different but deterministic random sequence
every epoch that depends only on `(base_seed, epoch)`, never on how many
times `set_epoch` was called before, so resuming a run at epoch *N*
reproduces the same stream as reaching epoch *N* sequentially.  Readers
instead store the epoch and fold it into each sample's derived seed (see
[Readers](#readers)), so their per-sample RNG also varies
deterministically per epoch without relying on a shared,
sequentially-drawn generator.  The sampler (e.g. torch's
`DistributedSampler`) applies its own epoch-pure reseeding.

## Generator tree

```text
DataLoader(seed=S)
│
├── master = Generator().manual_seed(S)
│
├── fork_generator(master, 2)          # child[i] seeded derive_seed(S, i)
│   ├── child[0] ──► Sampler
│   └── child[1] ──► Dataset / MeshDataset / MultiDataset
│                      │
│                      ├── fork_generator(child[1], 1+N_transforms)
│                      │   ├── child[0] ──► Reader
│                      │   ├── child[1] ──► Transform 0
│                      │   ├── child[2] ──► Transform 1
│                      │   └── ...
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
  `weighted_multinomial` for exact and Poisson-gap sampling.

### Mesh stochastic transforms

- **`RandomScaleMesh`**, **`RandomTranslateMesh`**,
  **`RandomRotateMesh`** — sample augmentation parameters from
  `torch.distributions.Distribution` objects via ICDF + generator.
- **`SubsampleMesh`** — uses `weighted_multinomial` with the exact or
  Poisson-gap strategy.

### `Compose`

`Compose.set_generator(generator)` forks and distributes one child per
child transform.  `Compose.set_epoch(epoch)` propagates to all children.
When used inside `Dataset`, the dataset flattens `Compose` and assigns
forks per leaf transform directly; `Compose`'s own methods are for
standalone use.

## Readers

Reader subsampling runs on the dataset's worker-thread pool (the threaded
`prefetch` producer path; `Dataset` defaults to `num_workers=2`), so the
*order* in which samples are drawn is non-deterministic.  A single shared,
sequentially-drawn generator would therefore not be reproducible with
`num_workers > 1`.  To avoid this, readers derive RNG **per
`(base_seed, epoch, index)`** instead of from one shared stream:

- **`set_generator(g)`** stores `g.initial_seed()` as the reader's base
  seed (it does *not* keep the generator itself).
- **`set_epoch(e)`** stores the epoch.
- Each `reader[index]` then calls
  `spawn_generator(base_seed, epoch, index)` to obtain a fresh generator
  for that sample's draws (the `Reader` base class exposes this as
  `_index_generator(index)`).

The draw for a given sample depends only on `(base_seed, epoch, index)`,
so it is **identical regardless of read order or worker thread** —
reproducible for any `num_workers` — while still differing across indices
and across epochs.  When no seed has been set, the per-sample generator is
`None` and draws fall back to the global default RNG.

Transforms remain reproducible because they run on the main thread in
sampler order (via the consume stage), so their sequentially-drawn
generators are unaffected by the threaded producer.

| Reader | Randomness | Per-`(seed, epoch, index)` RNG |
|---|---|---|
| `MeshReader` | `torch.randint` (cyclic block selection) | Yes |
| `DomainMeshReader` | `torch.randint` (cyclic block selection) | Yes |
| `NumpyReader` | `torch.randint` (cyclic coordinated subsampling) | Yes |
| `ZarrReader` | `torch.randint` (cyclic coordinated subsampling) | Yes |
| `TensorStoreZarrReader` | `torch.randint` (cyclic coordinated subsampling) | Yes |
| `HDF5Reader` | None | n/a (inherited base) |
| `VTKReader` | None | n/a (inherited base) |

## Iterable & descriptor paths: per-`(epoch, position)` seeding

Map-style datasets have a stable sample `index`, so readers key their
per-sample RNG on `(base_seed, epoch, index)` (see [Readers](#readers)).
Generator-style (`IterableDatasetBase`) and future descriptor-keyed
sources have **no stable index**: samples are produced in sequence with no
addressable position in a corpus. They therefore key on the **monotonic
emission position** within the epoch instead:

- **map-style:** `derive_seed(base_seed, epoch, index)` — reproducible for
  any read order / `num_workers`, since the index is intrinsic to the
  sample.
- **iterable / descriptor:** `derive_seed(base_seed, epoch, position)` —
  where `position` is a 0-based counter of emissions in the current epoch.
  Reproducible across runs and distinct across epochs and positions.

Both schemes use the same `derive_seed`/`spawn_generator` primitives; only
the coordinate that stands in for "which sample" differs. The iterable
path runs entirely on the main thread in emission order, so the position
counter is unambiguous (there is no worker-thread reordering to defend
against). See `tutorial_5_iterable_online_simulation.py` for a worked
example seeding an online electrostatics simulation per `(epoch, position)`.

## Current limitations

- `DistributedSampler` manages its own seed internally; when using it,
  pass `seed=` at `DistributedSampler` construction time rather than
  relying on DataLoader's seed propagation.
- Legacy datapipes (`cae/`, `gnn/`, `climate/`, `healpix/`,
  `benchmarks/`) are not wired into the generator protocol.
