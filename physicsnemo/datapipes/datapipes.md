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

`Dataset` and `MeshDataset` inherit from `DatasetBase`, which provides
thread-pool prefetching via a FIFO (First In / First Out)
`submit`/`consume` primitive driven by the `IOPump` (the index-keyed
`prefetch`/`__getitem__` cache is a thin random-access layer on top).
`MultiDataset` implements the same map-style surface by delegating
`submit`, `consume`, `prefetch`, `_pop_events`, and `close` to the child
`DatasetBase` instance that owns each sample; see
[Performance](#performance-threading-and-stream-based-concurrency) below.

## Composability

### Readers

A `Reader` is an ABC with one main loading hook plus a required length:

```python
class Reader(ABC):
    @abstractmethod
    def _load_sample(self, index: int) -> dict[str, Tensor]: ...

    @abstractmethod
    def __len__(self) -> int: ...
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

Collators combine per-sample `(data, metadata)` tuples into batches,
where `data` is usually a `TensorDict`, `Mesh`, or `DomainMesh` depending
on the dataset and collator:

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

For the provided `Dataset` and `MeshDataset` implementations, prefetching
is split into two stages so that **no device kernels are launched off the
main thread** -- device kernels must share the model's single launching
thread:

- `_load_host` is the **producer**.  It runs on a worker thread and does
  only thread-safe work: reading, decoding, and staging into pinned host
  memory.  It returns a `HostPayload`.
- `_consume` is the **consumer**.  It runs on whatever thread calls
  `__getitem__` (the main thread, in practice) and performs the
  host-to-device transfer and device transforms.

`DatasetBase` owns a `ThreadPoolExecutor` (configurable via
`num_workers`) and exposes a FIFO prefetch primitive.  `submit(work_item,
stream=...)` runs `_load_host` on the pool and returns a `PrefetchHandle`
bundling the future with the stream the consumer should use;
`consume(handle)` resolves it on the calling thread.  Subclasses that do
not override `_load_host` and `_consume` fall back to running their full
`_load` pipeline in the worker, so the main-thread launch guarantee
belongs to split implementations such as `Dataset` and `MeshDataset`:

```python
def submit(self, work_item, stream=None):
    future = self._executor.submit(self._load_host, work_item)
    return PrefetchHandle(future=future, stream=stream)

def consume(self, handle, *, defer_sync=False):
    payload = handle.future.result()       # re-raises producer errors
    # H2D + transforms here; defer_sync controls who gates the compute stream
    return self._consume(payload, handle.stream, defer_sync=defer_sync)
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
launching kernels or running the model.  For map-style datasets, this
path is active whenever `prefetch_factor > 0`; set `prefetch_factor=0`
to use synchronous map-style iteration.

### CUDA stream handoff

CUDA streams are an *optional* accelerator layered on top of the threaded
producer.  When `use_streams=True` (and CUDA is available), each sample is
round-robined a **preprocessing stream**.  The consumer runs *both* the
host-to-device copy and the transforms on that stream, then hands the
result to the compute stream via a CUDA **event** (never a host
`synchronize()`).  Crucially, *who* enqueues the compute-stream wait
depends on the `defer_sync` flag (see
[One-batch lookahead](#one-batch-lookahead-deferred-sync) below): the
DataLoader defers it so the wait lands *after* the previous batch's model
work, while standalone callers wait inline:

```python
def _consume(self, payload, stream=None, *, defer_sync=False):
    data = payload.data
    if device is not None and stream is not None:
        compute_stream = torch.cuda.current_stream()
        # Bind torch to the preprocessing stream.
        with preprocessing_stream(stream):              # torch.cuda.stream
            data = data.to(device, non_blocking=True)   # H2D on prep stream
            data = self.transforms(data)                # transforms on SAME stream
        event = torch.cuda.Event()
        event.record(stream)
        if defer_sync:
            self._events_pending.append(event)          # DataLoader gates later
        else:
            compute_stream.wait_event(event)            # inline order, no host block
    else:
        data = self.transforms(data)
    return data, payload.metadata
```

`preprocessing_stream` (in `protocols.py`) binds torch's current stream
to the preprocessing stream via `torch.cuda.stream(stream)`, so the
host-to-device copy and the transforms run on the side stream and GPU
preprocessing genuinely overlaps training.  The pinned host source is
held by the caching host allocator until the copy completes.

The event only orders *kernels*; device-memory lifetime needs its own
guard.  The sample's tensors are allocated on the preprocessing stream,
so when their Python references drop, the caching allocator would return
the blocks to that stream's pool immediately -- and a later sample's
host-to-device copy on the same round-robin stream could overwrite them
while compute-stream reads (collate, model) are still pending.
`_consume` therefore records every returned CUDA tensor against the
compute stream (`record_consumer_stream` in `protocols.py`, wrapping
`torch.Tensor.record_stream`), which defers allocator reuse of those
blocks until the compute stream's work at free time has completed.  The
cost is that freed blocks stay unavailable until the compute stream
catches up -- bounded by the prefetch depth's worth of samples.

### One-batch lookahead (deferred sync)

The compute-stream wait recorded above is only half the overlap story:
*when* it is enqueued decides whether preprocessing actually overlaps
training.  The DataLoader's prefetch loop (`_iter_prefetch`) keeps a
**one-batch lookahead** so the wait lands at the right point:

- `drain_one_batch()` consumes pump items up to the next `BATCH_BOUNDARY`,
  calling `consume(item, defer_sync=True)` on each.  With `defer_sync=True`
  the consumer enqueues the H2D copy + transforms on the preprocessing
  stream and *records* an event, but appends it to `_events_pending`
  instead of making the compute stream wait.
- `_pop_events()` (on `DatasetBase`) hands those recorded events back to
  the loop.
- Before yielding batch N, the loop eagerly drains batch **N+1**'s items
  (launching their preprocessing on the side streams), then calls
  `gate_compute_stream(events)` to issue `compute_stream.wait_event` for
  batch N's events -- right before the `yield`.

This ordering is the whole point.  Because the wait for batch N is
enqueued *after* the previous iteration's `yield` already enqueued batch
N-1's model kernels, the compute-stream order becomes
`..., model_{N-1}, wait(prep_N), model_N, ...`.  Batch N's preprocessing
(already in flight on its own stream) overlaps batch N-1's compute, and
each model only ever blocks on its own batch's preprocessing -- never on
the next batch's.  If `_consume` instead waited inline during the
lookahead drain, that wait would be ordered *ahead* of batch N's model
kernels and the model would block on batch N+1's preprocessing -- the
opposite of overlap.  Standalone callers (no DataLoader to insert the
gate) leave `defer_sync=False` and get the inline wait so their result is
immediately safe to use.

### Concurrency timeline

With everything launched from the main thread, the worker pool, the
preprocessing stream, and the compute stream form a triple buffer:

```text
Worker pool       │ load N+1 ─ load N ...     (host I/O + thread-safe CPU work)
Preprocess stream │           H2D + transforms for N
Compute stream    │ train N-1 ─ wait(prep_N) ─ train N
```

GPU preprocessing of batch N genuinely overlaps training of batch N-1 on
a separate stream; the two are ordered by a CUDA event, never a host-side
`synchronize`.  The ordering is what the one-batch lookahead buys: the
compute stream's `wait(prep_N)` is enqueued by `gate_compute_stream` only
*after* batch N-1's model kernels and only *after* batch N's preprocessing
has been launched on its side stream (see
[One-batch lookahead](#one-batch-lookahead-deferred-sync)), so each model
blocks on its own batch's preprocessing and nothing else.  A transform (or
generator) that forces a host readback simply serializes itself -- a
property of that code, not of the pipeline.

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
  pool.  `__iter__` may freely launch device kernels and use CUDA streams,
  and the loader still drives generation on a preprocessing stream with a
  CUDA event handoff.  This path does not use the map-style `IOPump`,
  `_events_pending`, or one-batch deferred-sync loop.

An iterable dataset yields either per-sample `(data, metadata)` (the
loader collates `batch_size` of them, `drop_last` trims the tail) or, when
`yields_batches = True`, ready-made batches that the loader passes through
unchanged.  Iterable datasets have no length: `len(loader)` raises
`TypeError`, and `shuffle`/`sampler` are ignored.  See
`examples/minimal/datapipes/tutorial_5_iterable_online_simulation.py` for
an online electrostatics simulation wired through this path.

### Pinned memory

Readers can set `pin_memory=True` to allocate CPU tensors in pinned
(page-locked) memory.  Pinned memory enables truly asynchronous
`non_blocking` transfers to GPU, so the CUDA stream overlap described
above is most effective when the reader pins its output.

### Debugging

Prefetching can be toggled at runtime for debugging:

```python
loader.disable_prefetch()   # fully synchronous: pump + streams off
loader.enable_prefetch()    # restore prefetch (streams too, when CUDA)
```

Toggles take effect at the next iteration.  The two halves can also be
controlled independently at construction time: `use_streams=False` keeps
the threaded producer but drops the CUDA stream handoff (the consumer
copies and transforms on the default stream), while `prefetch_factor=0`
disables the threaded pump for fully synchronous map-style execution.
Iterable datasets use their separate main-thread-only generator path
regardless of `prefetch_factor`.

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
parametrize distribution-backed random sampling.  To preserve
reproducibility with seeded `torch.Generator` objects (which
`Distribution.sample()` does not accept), `RandomScaleMesh`,
`RandomTranslateMesh`, and `RandomRotateMesh(mode="axis_aligned")` use
**inverse CDF sampling**: draw `U ~ Uniform(0,1)` via
`torch.rand(generator=g)`, then compute `X = distribution.icdf(U)`.  This
gives exact samples from the target distribution while keeping randomness
under generator control for distributions that implement `icdf()`.
Distributions without `icdf()` fall back to `Distribution.sample()` with
a warning and are not generator-reproducible.  `RandomRotateMesh` defaults
to `mode="uniform"`, which ignores `axes` and `distribution` and samples
uniform SO(3) rotations via random quaternions.
Full usage examples, YAML configuration, and the supported-distribution
table are in **[transforms/mesh/DISTRIBUTIONS.md](transforms/mesh/DISTRIBUTIONS.md)**.
