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
- **Stream parallelism** -- when enabled, each prefetched sample is
  assigned a CUDA stream so its host-to-device transfer can overlap with
  the main training computation.

### Producer / consumer split

Prefetching is split into two stages so that **no device kernels are
launched off the main thread** -- a hard requirement for Warp-based
transforms, which must share the model's single launching thread:

- `_load_host` is the **producer**.  It runs on a worker thread and does
  only thread-safe work: reading, decoding, and staging into pinned host
  memory.  It returns a `HostPayload`.
- `_consume` is the **consumer**.  It runs on whatever thread calls
  `__getitem__` (the main thread, in practice) and performs the
  host-to-device transfer and device transforms (including Warp kernels).

`DatasetBase` owns a `ThreadPoolExecutor` (configurable via
`num_workers`) and exposes a FIFO prefetch primitive.  `submit(work_item,
stream=...)` runs only the producer on the pool and returns a
`PrefetchHandle` bundling the future with the stream the consumer should
use; `consume(handle)` resolves it on the calling thread:

```python
def submit(self, work_item, stream=None):
    future = self._executor.submit(self._load_host, work_item)
    return PrefetchHandle(future=future, stream=stream)

def consume(self, handle):
    payload = handle.future.result()       # re-raises producer errors
    return self._consume(payload, handle.stream)   # H2D + transforms here
```

Correlation is purely by handle identity (FIFO), so work items need not
be hashable, unique, or even integers -- an `int` index is just the
common case.  The index-keyed `prefetch(index)` / `__getitem__(index)`
convenience API is a thin layer over `submit`/`consume` for random
access, and is what map-style tests and `MultiDataset` use.

### Self-priming dispatch (IOPump)

The threaded producer is driven by `IOPump`, a dedicated dispatcher
thread that keeps a *bounded* number of samples in flight regardless of
the consumer's cadence.  It pulls a work-item stream **lazily** (one item
per free backpressure slot, so an arbitrarily long or unbounded source
never materializes up front), calls `submit` for each, and hands the
returned handles back to the main thread in FIFO order.  The source
interleaves `BATCH_BOUNDARY` markers between work items; the pump forwards
them in place without consuming a slot, so the consumer reassembles
dynamically-sized batches from the boundaries -- the DataLoader never
builds the epoch's batch list in advance.  Because dispatch lives off the
main thread, the pipeline stays primed even while the main thread is busy
launching kernels or running the model.  This path is active whenever
`prefetch_factor > 0`; set `prefetch_factor=0` for fully synchronous
iteration.

### CUDA stream handoff

CUDA streams are an *optional* accelerator layered on top of the threaded
producer.  When `use_streams=True` (and CUDA is available), each sample is
round-robined a **preprocessing stream**.  The consumer runs *both* the
host-to-device copy and the transforms on that stream, then hands the
result to the compute stream via a CUDA **event** (never a host
`synchronize()`):

```python
def _consume(self, payload, stream=None):
    data = payload.data
    if device is not None and stream is not None:
        compute_stream = torch.cuda.current_stream()
        # Bind torch AND Warp to the preprocessing stream.
        with preprocessing_stream(stream):              # torch + wp.ScopedStream
            data = data.to(device, non_blocking=True)   # H2D on prep stream
            data = self.transforms(data)                # transforms on SAME stream
        data.record_stream(compute_stream)              # keep memory alive
        event = torch.cuda.Event()
        event.record(stream)
        compute_stream.wait_event(event)                # order, no host block
    else:
        data = self.transforms(data)
    return data, payload.metadata
```

**The single launching thread -- not a single stream -- is Warp's real
invariant.**  Warp kernels may run on any CUDA stream provided they are
launched from the main thread *and* Warp's current stream matches torch's.
`preprocessing_stream` (in `protocols.py`) binds both via
`wp.ScopedStream(wp.stream_from_torch(stream))`, so transforms (including
Warp mesh-query / BVH kernels) run correctly on the side stream.  A
previous `cudaErrorIllegalAddress` here was a torch/Warp stream
*divergence* (data on a side stream, the Warp kernel on Warp's own
stream), not a prohibition on non-default streams; binding both fixes it
and lets GPU preprocessing genuinely overlap training.  `record_stream`
keeps the device tensors from being recycled while the compute stream
reads them; the pinned host source is held by the caching host allocator
until the copy completes.

### Concurrency timeline

With everything launched from the main thread, the worker pool, the
preprocessing stream, and the compute stream form a triple buffer:

```text
Worker pool       │ load N+2 ─ load N+1 ...   (host I/O + thread-safe CPU work)
Preprocess stream │            H2D + Warp transforms for N+1
Compute stream    │                         train N
```

GPU preprocessing of batch N+1 genuinely overlaps training of batch N on
a separate stream; the two are ordered by a CUDA event, never a host-side
`synchronize`.  A transform (or generator) that forces a host readback
simply serializes itself -- a property of that code, not of the pipeline.

### Two data paths: map/descriptor vs iterable

The DataLoader selects one of two mutually-exclusive paths by dataset
type:

- **Preload path (`DatasetBase`)** -- map-style and descriptor-keyed
  datasets.  Uses the worker pool + `IOPump` described above: workers do
  thread-safe host I/O, the main thread consumes handles (H2D + transforms
  on the preprocessing stream).  This is the path for storage-backed data
  addressable by index.
- **Generator path (`IterableDatasetBase`)** -- iterable datasets that
  *produce* data (online simulation, procedural samplers, unbounded
  streams).  Driven **main-thread-only**: no sampler, no pump, no worker
  pool.  `__iter__` may freely launch Warp kernels and use CUDA streams
  (the single-launching-thread invariant holds), and the loader still
  drives generation on a preprocessing stream with the same event handoff,
  so generation of batch N+1 overlaps training of batch N.

An iterable dataset yields either per-sample `(data, metadata)` (the
loader collates `batch_size` of them, `drop_last` trims the tail) or, when
`yields_batches = True`, ready-made batches that the loader passes through
unchanged.  Iterable datasets have no length: `len(loader)` raises
`TypeError`, and `shuffle`/`sampler` are ignored.  See
`examples/minimal/datapipes/tutorial_5_iterable_online_simulation.py` for
a Warp `Darcy2D` online simulation wired through this path.

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

`use_streams=False` keeps the threaded producer but drops the CUDA
stream handoff (the consumer copies and transforms on the default
stream); `prefetch_factor=0` forces fully synchronous execution.

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
