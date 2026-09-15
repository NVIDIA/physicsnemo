# Design: a generic two-tier cache for datapipes readers

Status: implemented (rollout steps 1-5 and 7), validated against real zarr
and tensorstore. Step 6 (derived-getter `_cache` seeding /
`share_cache_field`) remains deferred pending profiling of what actually
computes pre-subsample. Measured on a 5-boundary DomainMesh: 109
filesystem ops per sample uncached, 0 ops on the source when disk-warm
(all traffic on the local mirror), 0 ops total when RAM-warm.
Scope: `physicsnemo/datapipes/` readers. First integrations: `MeshReader` /
`DomainMeshReader` (the motivation), then the zarr/tensorstore/HDF5 readers.

## 1. Problem

On Lustre, per-sample read cost for the mesh readers is dominated by
*metadata operations*, not data bandwidth. A `.pmsh` / `.pdmsh` file is a
tensordict memmap directory tree, and `Mesh.load` / `DomainMesh.load` (via
tensordict `_load_memmap`) performs, per sample, every epoch:

- 1 `meta.json` open+read and 1 `readdir` **per node** (root, `point_data`,
  `cell_data`, `global_data`, `_cache`; for `DomainMesh`, that set again per
  boundary). A domain mesh with 5 boundaries is ~25–40 nodes.
- ≥1 `stat` + open + mmap **per tensor leaf** — including one tiny `.memmap`
  file per `global_data` scalar (`U_inf`, `rho_inf`, …).
- For `DomainMeshReader.extra_boundaries`: one `glob` (readdir) per boundary
  per sample at read time, plus a full `Mesh.load` tree walk on the match.

Total: **~50–150 small, latency-bound filesystem operations per sample**, of
which only the handful of large memmaps actually faulted in carry payload.
Everything else is immutable across epochs and ranks. The zarr readers have
the same disease in smaller form: `TensorStoreZarrReader._read_attributes`
does an uncontexted `KvStore.open` + full `store.list()` per sample;
`ZarrReader` re-runs `array_keys()` + `attrs.keys()` per sample. In the aero
recipe, `MeshReaderWithGlobalData` re-reads a ~6-scalar `global_data`
TensorDict from Lustre on every sample, every epoch. The pipeline layer adds
its own repeat offenders: `Reader.field_names` re-loads sample 0 on every
access, and `len(reader)` is called 3× per `__getitem__`.

## 2. Constraints from the environment

Facts that shape the design (`datapipes.md`, `dataloader.py`, `io_pump.py`,
`protocols.py`):

1. **Single process, multiple threads.** The physicsnemo DataLoader is an
   `IOPump` dispatcher thread plus a per-dataset `ThreadPoolExecutor`;
   readers are never pickled or forked. The cache is an ordinary in-process
   object shared by reference. It must be *thread-safe*, not cross-process —
   except that under DDP, multiple ranks per node may share one disk cache
   directory, which requires atomic-write discipline, not IPC.
2. **Reads are stateless; RNG derives per `(seed, epoch, index)`.**
   Subsampled or transformed outputs are epoch-dependent. The cache holds
   **raw, pre-subsampling artifacts only** — never sampled results.
3. **Datasets are effectively write-once.** Validation defaults to "trust
   immutable", with an opt-in staleness check for development.
4. **Reader discovery attributes are de-facto public.** The aero recipe
   reaches into `reader._root` / `reader._paths`; caching must not alter
   discovery semantics.
5. **No pickle on the disk tier.** The repo already carries
   `weights_only=False` footguns (`experimental/utils/cached_dataset.py`
   warns about its own); a shared cache directory must be safe to load from.
6. **No second deserialization path.** A cached read must go through the
   same loader as an uncached read (`Mesh.load`, `zarr.open`, …). Caching a
   bespoke re-serialization of a loaded object creates a shadow decoder that
   must stay bit-equivalent with the real one forever; this design rejects
   that category entirely.
7. **No assumed tiers.** RAM-only, disk-only, both, or neither must all be
   valid configurations. "Disk" is any directory (NVMe, tmpfs, node-local
   scratch); nothing may require it, and nothing may require RAM.

## 3. Core object: `DatasetCache`

One new module, `physicsnemo/datapipes/caching.py`, exporting a single
reader-agnostic object. Readers depend on it; it depends on no reader.

```python
cache = DatasetCache(
    ram_bytes_limit   = 2 * 2**30,               # None disables the RAM tier
    disk_dir          = "/local/nvme/pn-cache",  # None disables the disk tier
    disk_bytes_limit  = 200 * 2**30,
    eviction          = "largest",               # "largest" | "fifo" | "lru"
    max_item_bytes    = 8 * 2**20,               # admission control (blobs)
    small_file_bytes  = 64 * 2**10,              # copy-vs-symlink cutoff (trees)
    validate          = "none",                  # "none" | "mtime"
)
```

### 3.1 API — one hot-path call, two entry flavors

```python
value = cache.get_or_load(key, loader)              # blob entry
obj   = cache.get_or_load(key, loader, src=path)    # tree-backed entry
```

Uniform hit/miss model: **every** read asks the cache; work happens only on
a miss. Callers have no separate warm/cold code path.

**Blob entries** (no `src`): the loader runs on a miss; the result is stored
write-through to both configured tiers and returned. For small values —
attrs dicts, key lists, glob resolutions, global-data TensorDicts.

**Tree-backed entries** (`src=path`): for sources that are *directory trees
of many small files plus a few large ones* (tensordict memmap trees, zarr
directory stores). The loader is the caller's **stock loader** and is always
invoked with a path; each tier has its natural representation and neither
requires the other:

- **RAM representation: the loaded object itself.** Large leaves of a
  memmap-loaded object are `MemoryMappedTensor`s — pointers, not resident
  bytes (pages fault in on touch and are kernel-reclaimable) — so caching
  the object costs approximately its small-metadata bytes, exactly the data
  worth keeping resident. A RAM hit is **zero filesystem operations** and
  returns a shallow structural copy (fresh tensordict structure, shared
  tensor storage) so downstream structural mutation cannot corrupt the
  cached entry.
- **Disk representation: a sparse mirror.** On a disk miss, the source tree
  is walked once and materialized under the disk tier: every file whose
  `st_size` is under `small_file_bytes` is copied; every larger file becomes
  an absolute symlink to the source. No format knowledge — the rule is pure
  file size. The stock loader then runs against the mirror: its many small
  metadata ops hit local disk (and the kernel dentry/page cache) instead of
  Lustre; large-file opens follow symlinks to the source, unchanged.

Resolution order: RAM hit → object. RAM miss with disk tier → mirror
(hit-or-materialize) → `loader(mirror_path)` → insert object into RAM. RAM
miss without disk tier → `loader(src)` → insert into RAM. No RAM tier →
`loader(mirror_path)` per read. Tree-backed values are **never serialized**:
the mirror is their disk form, the object their RAM form — so the codec
(§3.4) never sees them.

**Keys.** `key = (kind: str, identity: str)`. `kind` names the entry type
and embeds a format version (`"mesh/v1"`, `"zarr-attrs/v1"`). `identity` is
the **fully resolved source path** — `str(path.resolve())`, symlinks and
relative roots collapsed — so the same physical file reached through
different dataset roots, patterns, or train/val reader instances shares one
entry. Sub-file entries append a suffix:
`(kind, f"{realpath}::derived/normals")`. Staleness/fingerprint data (source
mtime, library versions) lives in entry headers, not keys.

**Concurrency.** Single-flight striped locks: concurrent `get_or_load` on
one key from multiple worker threads runs the loader/mirror exactly once;
other threads wait on the in-flight result. Distinct keys never contend.

**Admission.** Blob values whose estimated size exceeds `max_item_bytes` are
returned but not admitted to either tier — large arrays cannot pollute the
cache even if a caller misclassifies one. (Tree entries are inherently
bounded: mirrors hold only sub-threshold files plus symlinks; RAM objects
count only resident bytes, see §3.3.)

**Secondary API.** `invalidate(key)` (used by the fallback path), `stats()`
(per-tier hits / misses / bytes / evictions / entries), `clear(kind=None)`,
`close()` (drops the RAM tier; disk entries persist across runs by design —
that persistence is why later runs on a node start warm).

### 3.2 Tiers

**RAM tier** — `dict[key, Entry]` under a lock, storing deserialized blob
values and tree-backed objects (zero per-hit cost). Eviction to stay under
`ram_bytes_limit`; evicting a blob never writes to disk (write-through
already did), evicting an object just drops the reference.

**Disk tier** — under `disk_dir/<kind>/`: blob entries as single files
`<sha1(identity)>.pnc`; tree entries as mirror directories. All writes go to
a `*.tmp` sibling followed by atomic `os.replace`/`os.rename`, so DDP ranks
sharing the directory can never observe a torn entry; a rank losing the
write race overwrites with identical content. A read hitting
`FileNotFoundError` (another rank evicted it) degrades to a miss. Size
accounting: scan once at init, then track own writes/deletes; accounting
races are tolerated because the limit is a soft target, not a correctness
invariant.

### 3.3 Size accounting

Blob sizes come from the codec's `estimate_size` (tensor `nbytes` + fixed
per-object overhead). Tree-backed RAM entries count **resident bytes only**:
small tensor `nbytes`, skipping `MemoryMappedTensor` leaves (whose `nbytes`
reports full on-disk size and would wreck largest-first eviction). Tree disk
entries count the mirror's on-disk bytes (small files + symlinks — tiny). A
`size_hint` argument overrides when the caller knows better. One documented
budget: each cached object holds ~N_large mmaps, and `vm.max_map_count`
(~65k default) bounds cacheable entries in the tens of thousands — the byte
limit binds long before that in practice.

### 3.4 Eviction policies

A policy is a pure function over entry records `(key, size, insert_seq,
last_access_seq)` selecting victims until the tier is under its limit.

- **`largest` (default): evict largest entries first**, FIFO tie-break.
  Rationale: every cached item saves roughly the same thing — one or more
  Lustre metadata round-trips — regardless of byte size. Benefit per item is
  ~constant while holding cost is proportional to size, so largest-first
  maximizes resident item count and therefore metadata ops avoided per byte
  of cache. It also sheds anomalously large entries before touching the sea
  of small ones.
- `fifo`: least bookkeeping; the right loss profile for a strict shuffle
  over more samples than fit (where every policy loses equally).
- `lru`: for genuine locality (curriculum sampling, repeated validation
  subsets).

Policies register by name; adding one is one method. Evicting a mirror while
a loaded object still mmaps into it is safe on POSIX (inodes survive until
unmapped); a load racing an eviction gets a filesystem error and falls back
(§5.2).

### 3.5 Serialization — no pickle, blobs only

A disk blob entry is a small JSON header (schema version, key, kind,
optional source fingerprint) followed by named raw tensor buffers with
dtype/shape (safetensors-style layout). Values are restricted to JSON-able
metadata, `bytes`, and flat `{name: tensor}` maps — sufficient for every
blob use case and safe to load from a shared directory. Tree-backed entries
bypass serialization entirely (§3.1).

### 3.6 Validation

`validate="none"` (default): zero stat calls — the point of the cache.
`validate="mtime"`: one `stat` of the source path per key **per process
lifetime** (memoized), compared to the mtime in the entry header. For
iterating on datasets during development; production write-once training
never pays it. Independently, a moved/deleted source breaks mirror symlinks
at load time, which surfaces as the ordinary fallback path.

## 4. Integration level 1: generic opt-in in `Reader`

Every reader gains `cache: DatasetCache | None = None` (default preserves
today's behavior exactly) and the base class provides one protected helper:

```python
def _cached(self, kind, path, loader, *, src=None):
    if self._cache is None:
        return loader(src) if src is not None else loader()
    return self._cache.get_or_load((kind, str(path.resolve())), loader, src=src)
```

Readers route whichever loads they choose through it:

- A reader whose whole sample is small wraps `_load_sample` outright — e.g.
  the aero recipe's `MeshReaderWithGlobalData` external read becomes
  `self._cached("global-data/v1", ext_path, lambda:
  TensorDict.load_memmap(ext_path))`, eliminating the per-sample memmap walk
  *and* the `exists()` stat.
- Readers with bulk arrays wrap only their small pieces:
  `TensorStoreZarrReader` caches `_read_attributes` results and per-field
  open specs (shape/dtype/chunk grid, enabling `assume_metadata` on
  `ts.open`); `ZarrReader` caches per-group `array_keys()` + attrs;
  `HDF5Reader`/`NumpyReader` directory modes cache per-file key lists and
  shapes; `DomainMeshReader` caches `extra_boundaries` glob resolutions.
- Directory-tree formats pass `src=` and their stock loader — zarr directory
  stores mirror `.zarray`/`.zattrs`/`.zgroup` and symlink chunks with zero
  zarr-specific cache code.

One `DatasetCache` instance may be shared across readers (train/val/test) —
`kind` + resolved-path identity keep entries distinct while sharing the byte
budgets. `DatasetCache` gets an `@register()` entry so Hydra configs build it
via the `${dp:...}` resolver and pass it to readers, matching the registry
pattern in `datapipes.md`. Dataset `close()` → `reader.close()` is the
teardown hook.

## 5. Integration level 2: mesh readers

`Mesh` is a tensorclass: attributes `points`, `cells`, `point_data`,
`cell_data` (all O(N)), `global_data` (tiny), and an internal
`_cache: TensorDict` where derived-quantity getters memoize
(`cell_centroids`, `cell_areas`, `cell_normals`, `point_normals`,
curvatures; `mesh.py:774, 803, 836, 892`). `DomainMesh` combines an
`interior: Mesh` with `boundaries: dict[str, Mesh]` and a domain-level
`global_data`. **Nothing in
`physicsnemo/mesh/` changes**; the integration is a tree-backed entry plus a
few blob entries in `datapipes/readers/mesh.py`.

### 5.1 The sample as a tree-backed entry

```python
def _load_sample(self, index):
    path = self._paths[index]
    if self._cache is None:
        return Mesh.load(path)
    return self._cache.get_or_load(
        ("mesh/v1", str(path.resolve())), Mesh.load, src=path,
    )
```

That is the whole integration for the load path. Consequences, per tier:

- **RAM hit**: the loaded `Mesh`/`DomainMesh` object, shallow-copied — zero
  filesystem ops. `global_data`, structure, and small `_cache` leaves are
  resident; `points`/`cells`/field data remain memmap pointers into Lustre,
  faulting pages only when touched (the subsampled block).
- **Disk hit, RAM miss**: `Mesh.load(mirror)` — the ~50–150 small metadata
  ops run against local disk / dentry cache; the N_large big-leaf opens
  follow symlinks to Lustre. Same Lustre traffic as a RAM hit's first touch.
- **Full miss**: mirror materialization walks the Lustre tree exactly once,
  then loads from the mirror — the miss path *is* the hit path.
- The small-vs-large question ("cache global data but not points") is
  answered by the `st_size` rule and memmap-pointer semantics — no attribute
  names anywhere in the cache.

There is no harvest, no reconstruct, no manifest format: `Mesh.load` is the
only decoder, cold and warm, and the cache never interprets tensordict's
layout. Format compatibility with future tensordict versions is automatic.

### 5.2 Fallback

If a load against a mirror fails (evicted mid-read, source moved, torn
state) or a cached object is structurally unusable: `invalidate(key)` and
fall back to stock `Mesh.load(src)` for that read; the next `get_or_load`
re-materializes. The cache can be slower on a bad entry, never differently
correct.

### 5.3 Derived getters and extra boundaries

- Because RAM entries are shared objects, getter memoization into
  `mesh._cache` can persist across reads for free. Default hits return a
  shallow copy (writes don't propagate back — safe); an opt-in
  `share_cache_field=True` mode shares the `_cache` TensorDict by reference,
  making cross-read memoization automatic (its writes are idempotent
  memoization, so concurrent same-key writes are benign). Genuinely small
  derived values can additionally be blob entries under
  `(kind, f"{realpath}::derived/<key>")`, merged into `mesh._cache` after
  load — valid only for pre-subsample computation on the raw mesh.
- `extra_boundaries`: the per-sample glob resolution is a blob entry; each
  matched boundary mesh is its own tree-backed entry.

### 5.4 Reader metadata

`field_names`, shapes, `n_points`/`n_cells`, `_get_sample_metadata` answers
are small blob entries populated from a loaded sample — pure cache reads
when warm, with zero mesh construction.

## 6. Non-goals

- **No caching of large arrays by value.** Bulk data stays on the memmap
  path via symlinks/pointers; `max_item_bytes` and the `st_size` rule
  enforce it. NVMe bulk staging is a different tool (tensorstore's chunk
  cache already exists for zarr).
- **No caching of subsampled/transformed samples** — epoch-dependent RNG
  makes reuse wrong (§2.2).
- **No second deserialization path** (§2.6) — the rejected alternative was a
  harvested manifest + reconstruct; it shadowed tensordict's decoder.
- **No cross-process coordination beyond atomic rename.** No file locks, no
  shared-memory tier, no server. Ranks duplicate RAM-tier contents — the
  cheap, correct trade at these item sizes.
- **No write-behind.** Entries are small; synchronous write-through keeps
  crash semantics trivial (an entry fully exists or doesn't).

## 7. Testing plan

Follows `test/datapipes/readers/test_mesh_readers.py` conventions — real
tiny meshes from `physicsnemo.mesh.primitives.basic` saved to `tmp_path`, no
mocked filesystems.

1. **`DatasetCache` unit (blobs)**: hit/miss/promotion across tiers;
   byte-limit enforcement and eviction order per policy (largest-first with
   FIFO tie-break); `max_item_bytes` bypass; single-flight under a thread
   pool; atomic-rename crash safety (interrupt between tmp write and replace
   ⇒ clean miss); two instances on one `disk_dir` (simulated multi-rank)
   with concurrent read/write/evict.
2. **`DatasetCache` unit (trees)**: mirror correctness (copy-vs-symlink by
   size, absolute link targets); `Mesh.load(mirror) == Mesh.load(src)`
   tensor-for-tensor for each primitive mesh and a multi-boundary
   `DomainMesh`, including `global_data`, `_cache` leaves, and
   `extra_boundaries`; RAM-object hit returns structurally independent,
   storage-shared copies; every tier combination (RAM-only, disk-only,
   both, neither).
3. **The actual claim — metadata-op counting**: audit `open`/`os.stat`/
   `os.scandir` via `sys.addaudithook`; assert RAM-warm per-sample counts
   are zero and disk-warm counts touch only the mirror + N_large source
   opens.
4. **Fallback**: delete a mirror / move the source / bump the kind version ⇒
   silent stock load, invalidation, re-population.
5. **End-to-end**: `MeshDataset` + `DataLoader` over a cached
   `DomainMeshReader`, 2 epochs, bitwise-equal outputs vs uncached at the
   same seed, for each tier combination.

## 8. Rollout order

1. `caching.py`: `DatasetCache` (blob entries), tiers, policies, codec,
   registry entry; unit tests. **[done]**
2. Tree-backed entries (`src=`): RAM object caching + sparse mirror.
   **[done]**
3. `cache=` plumbing in `Reader` base + the `_cached` helper (shared
   implementation: `cached_or_load`); mesh reader integration (§5.1–5.2),
   including `extra_boundaries` entries. **[done]**
4. Aero-recipe `MeshReaderWithGlobalData` global-data entry (one line,
   biggest per-line payoff). **[done]**
5. Zarr key-discovery + tensorstore attrs entries; tensorstore per-field
   resolved-spec caching with `assume_metadata=True` on warm opens
   (validated: warm reads succeed with the array metadata files deleted);
   VTK per-sample file resolution; uniform `cache=` kwarg on HDF5/numpy
   (accepted, currently routes nothing -- their per-sample cost is data,
   not repeated metadata). **[done]** Deferred from this step: zarr
   directory-store *tree* entries (the store-handle cache already keeps
   groups open; revisit if listing cost shows up).
6. Derived-getter blob seeding and `share_cache_field` (§5.3) once profiling
   shows what runs pre-subsample. **[deferred]**
7. Pipeline freebies: memoize `Reader.field_names`, hoist the triple
   `len(self)` in `Reader.__getitem__`. **[done]**
