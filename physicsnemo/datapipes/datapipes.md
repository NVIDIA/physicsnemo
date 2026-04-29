# Datapipes -- Design Overview

A GPU-centric, modular data pipeline for scientific machine learning.
The system uses **threads and CUDA streams** to overlap disk I/O,
host-to-device transfer, and GPU-side transforms within a single
process.  The result is low latency, zero inter-process serialization,
and natural support for GPU-accelerated preprocessing -- properties
that matter when datasets are large, batches are small, and transforms
benefit from GPU execution.

## Architecture

The pipeline has four composable layers:

```text
Reader  -->  Dataset  -->  DataLoader  -->  Training loop
 (I/O)      (transforms)   (batching)
```

```text
                        ┌─────────────────────────────────────────────────┐
                        │                   DataLoader                    │
  ┌──────────┐          │  ┌──────────────────────────────────────────┐   │
  │  Sampler │─indices─▶   │               Dataset                    │   │
  └──────────┘          │  │                                          │   │
                        │  │  Reader ──► Device transfer ──► Transforms│  │
                        │  │  (CPU I/O)   (non_blocking)    (Compose) │   │
                        │  └──────────────┬───────────────────────────┘   │
                        │                 │                               │
                        │                 ▼                               │
                        │            Collator                             │
                        └────────────────┬────────────────────────────────┘
                                         │
                                         ▼
                                 Batched TensorDict
                                  (training loop)
```

Three dataset types share this pattern:

| Type | Data model | Transform base |
|------|------------|----------------|
| `Dataset` | `TensorDict` fields | `Transform` |
| `MeshDataset` | `Mesh` / `DomainMesh` tensorclasses | `MeshTransform` |
| `MultiDataset` | Union of child `DatasetBase` instances | Delegates to children |

All three inherit from `DatasetBase`, which provides thread-pool
prefetching and a `Future`-based cache (see
[Performance](#performance-threading-and-stream-based-concurrency) below).

## Composability

### Readers

A `Reader` is an ABC with a single contract:

```python
class Reader(ABC):
    @abstractmethod
    def _load_sample(self, index: int) -> dict[str, Tensor]: ...
```

`__getitem__` wraps the result in a `TensorDict` on CPU (optionally
pinned).

### Transforms

Transforms are pure functions on `TensorDict` (or `Mesh`):

```python
class Transform(ABC):
    @abstractmethod
    def __call__(self, data: TensorDict) -> TensorDict: ...
```

For meshes, the `MeshTransform` ABC provides the same interface with
`__call__(Mesh) -> Mesh` plus `apply_to_domain(DomainMesh)` for
multi-region consistency.

### Collators

Collators combine per-sample `(TensorDict, metadata)` tuples into batches:

| Collator | Strategy |
|----------|----------|
| `DefaultCollator` | `TensorDict.stack()` -- all samples must share shape |
| `ConcatCollator` | `torch.cat()` along an axis with optional `batch_idx` -- for variable-length point clouds |
| `FunctionCollator` | Wraps any callable |

### Registry and Hydra integration

All readers, transforms, datasets, and the DataLoader are decorated with
`@register()`, placing them in a global `COMPONENT_REGISTRY`.  The helper
`register_resolvers()` (called at import time) registers an OmegaConf
resolver so Hydra configs can reference components by short name:

```yaml
dataset:
  _target_: ${dp:Dataset}
  reader:
    _target_: ${dp:ZarrReader}
    path: /data/field.zarr
    fields: [pressure, velocity]
  transforms:
    - _target_: ${dp:Normalize}
      fields: [pressure]
      method: mean_std
      means: {pressure: 0.0}
      stds:  {pressure: 1.0}
    - _target_: ${dp:SubsamplePoints}
      input_keys: [pressure, velocity]
      n_points: 10000
  device: cuda
```

The equivalent Python:

```python
from physicsnemo.datapipes import Dataset, ZarrReader, Normalize, SubsamplePoints

dataset = Dataset(
    ZarrReader("/data/field.zarr", fields=["pressure", "velocity"]),
    transforms=[
        Normalize(["pressure"], method="mean_std",
                  means={"pressure": 0.0}, stds={"pressure": 1.0}),
        SubsamplePoints(["pressure", "velocity"], n_points=10000),
    ],
    device="cuda",
)
```

## Performance: threading and stream-based concurrency

### Why threads + streams

Scientific ML data loading is dominated by disk I/O and GPU-side
preprocessing.  Threads are a natural fit:

- **Shared state** -- threads share memory, file handles, and the CUDA
  context within a single process, so there is no serialization or
  duplication overhead.
- **I/O concurrency** -- the GIL is released during disk reads and CUDA
  kernel launches, so multiple threads usefully overlap I/O with GPU work.
- **Stream parallelism** -- each prefetched sample is assigned its own
  CUDA stream, allowing host-to-device transfers and GPU transforms to
  run concurrently with the main training computation.

### Thread-pool prefetch

`DatasetBase` owns a `ThreadPoolExecutor` (configurable via
`num_workers`, default 2).  Calling `prefetch(index)` submits the
load-and-transform pipeline to the pool and stashes the `Future`:

```python
def prefetch(self, index, stream=None):
    if index in self._prefetch_futures:
        return
    executor = self._ensure_executor()
    self._prefetch_futures[index] = executor.submit(self._load, index)
```

`__getitem__` pops the `Future` if one exists, otherwise loads
synchronously:

```python
def __getitem__(self, index):
    future = self._prefetch_futures.pop(index, None)
    if future is not None:
        return future.result()
    return self._load(index)
```

This means the DataLoader can keep the next batch loading in background
threads while the current batch is being consumed by the model.

### CUDA stream overlap

When GPU execution is available, `Dataset` (and `MeshDataset`) override
`prefetch` to run device transfer and transforms on a caller-supplied
CUDA stream, then record an event for later synchronization:

```python
def _load_and_transform(self, index, stream=None):
    result = _PrefetchResult(index=index)
    data, metadata = self.reader[index]           # CPU I/O in worker thread

    if stream is not None:
        with torch.cuda.stream(stream):
            data = data.to(device, non_blocking=True)  # H2D on stream
            data = self.transforms(data)               # GPU transforms on stream
        result.event = torch.cuda.Event()
        result.event.record(stream)                    # mark completion

    result.data, result.metadata = data, metadata
    return result
```

On retrieval, `__getitem__` synchronizes the event before returning:

```python
if result.event is not None:
    result.event.synchronize()
return result.data, result.metadata
```

The `DataLoader` owns a pool of `num_streams` CUDA streams (default 4)
and round-robins them across samples.  It also maintains a sliding
prefetch window of `prefetch_factor` batches (default 2) ahead of the
current yield position:

```python
# Prefetch the next batch as we yield the current one
for sample_idx in all_batches[next_prefetch_idx]:
    stream = self._streams[stream_idx % self.num_streams]
    self.dataset.prefetch(sample_idx, stream=stream)
    stream_idx += 1
```

### Concurrency timeline

The diagram below shows how threads and streams overlap for a two-sample
batch with `prefetch_factor=1`:

```text
Main thread       Worker 1            Worker 2            Stream 1    Stream 2
    │                 │                   │                   │           │
    ├─prefetch(0,S1)─►│                   │                   │           │
    ├─prefetch(1,S2)─────────────────────►│                   │           │
    │                 ├─ Read (I/O)       │                   │           │
    │                 │                   ├─ Read (I/O)       │           │
    │                 ├─ to(device) ─────────────────────────►│           │
    │                 ├─ transforms ─────────────────────────►│           │
    │                 ├─ event.record() ─────────────────────►│           │
    │                 │                   ├─ to(device) ─────────────────►│
    │                 │                   ├─ transforms ─────────────────►│
    │                 │                   ├─ event.record() ─────────────►│
    ├─ event.synchronize() ×2             │                   │           │
    ├─ collate + yield batch              │                   │           │
    │                 │                   │                   │           │
```

While the main thread consumes batch N, worker threads are already
loading batch N+1 on different streams.

### Pinned memory

Readers can set `pin_memory=True` to allocate CPU tensors in pinned
(page-locked) memory.  Pinned memory enables truly asynchronous
`non_blocking` transfers to GPU, so the CUDA stream overlap described
above is most effective when the reader pins its output.

### Debugging

Prefetching can be toggled at runtime for debugging:

```python
loader.disable_prefetch()   # synchronous, single-stream -- easy to debug
loader.enable_prefetch()    # re-enable after debugging
```

Setting `use_streams=False` or `prefetch_factor=0` at construction time
also forces synchronous execution.

## RNG and reproducibility

Deterministic data loading is opt-in.  Passing `seed=` to `DataLoader`
creates a master `torch.Generator` that is forked into independent
streams for the sampler, the reader, and every stochastic transform.
`set_epoch(epoch)` reseeds all streams deterministically so each epoch
produces a different but reproducible random sequence.  The full
generator tree, device management rules, and per-component details are
documented in **[RNG.md](RNG.md)**.

## Augmentations

Mesh augmentations (`RandomScaleMesh`, `RandomTranslateMesh`,
`RandomRotateMesh`) accept any `torch.distributions.Distribution` to
parametrize their random sampling.  To preserve reproducibility with
seeded `torch.Generator` objects (which `Distribution.sample()` does not
accept), the augmentations use **inverse CDF sampling**: draw
`U ~ Uniform(0,1)` via `torch.rand(generator=g)`, then compute
`X = distribution.icdf(U)`.  This gives exact samples from the target
distribution while keeping all randomness under generator control.
Full usage examples, YAML configuration, and the supported-distribution
table are in **[transforms/mesh/DISTRIBUTIONS.md](transforms/mesh/DISTRIBUTIONS.md)**.
