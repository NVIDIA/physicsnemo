# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A two-tier cache for the small, repeated reads that datapipe readers make.

Why this exists
---------------
Reading one sample from a mesh or zarr dataset is not one read. It is a
``meta.json``, a directory listing, a ``stat`` and an ``open`` per array,
and only then the bytes. On a network filesystem such as Lustre each of
those small operations is a round trip to a metadata server, and they are
repeated, identically, every epoch. :class:`DatasetCache` remembers the
results so a rank pays for them once.

What is cached
--------------
Two kinds of entry, both behind one call, :meth:`DatasetCache.get_or_load`:

**Blob entries** are small metadata values: attribute dicts, key listings,
glob results, store specs, and tensor-shaped metadata such as a scalar
Reynolds number or a ``(1, 3)`` inflow vector. On a miss the caller's loader
runs and the value is kept in RAM and, if a disk tier is configured, written
as a ``.json`` file. Small tensors travel inside the JSON as tagged nodes;
anything over a few thousand elements is bulk data, not metadata, and stays
RAM-only.

**Tree entries** are whole samples that live on disk as a directory of many
small files plus a few large arrays (a tensordict memmap tree, a zarr
store). The caller passes ``src=<directory>`` and a stock loader such as
``Mesh.load``. The cache never re-implements that loader. Instead the disk
tier builds a *sparse mirror* of the directory on local storage: small
files are copied, large files become symlinks back to the source. The stock
loader then runs against the mirror, so its metadata reads hit local disk
while the bulk bytes still come from the source. In RAM the entry is the
loaded object itself.

How a lookup flows
------------------
::

    get_or_load(key, loader, src=None)
        │
        ▼
    RAM tier ── hit ──▶ return value          (tree entries: shallow copy)
        │
       miss
        │
        ▼
    disk tier ── hit ──▶ blob: json.loads(file)
        │                tree: loader(mirror_dir)
       miss
        │
        ▼
    blob: value = loader()          tree: mirror_dir = sparse_mirror(src)
          write value.json                value = loader(mirror_dir)
        │
        ▼
    RAM tier stores value ──▶ return value

On-disk layout of a cache directory
-----------------------------------
::

    <disk_dir>/
      zarr-attrs/v1/                       ← one directory per "kind"
        3fa9…c2.json                       ← blob entry
      mesh/v1/
        8b1d…e7.tree/                      ← tree entry: sparse mirror
          meta.json                        ←   copied (small)
          points.memmap  → /lustre/…       ←   symlink (large)
          point_data/…
        8b1d…e7.size                       ← sidecar: mirror size in bytes

File names are a hash of the entry's identity, so paths stay short and
free of the source path's characters.

Rules for callers
-----------------
Cache only **raw, immutable** artifacts. Never cache a subsampled or
RNG-dependent result. Treat returned objects as read-only: RAM hits share
tensor storage between callers.

Multi-GPU behaviour
-------------------
The physicsnemo ``DataLoader`` is thread-based, so each rank process owns
one ``DatasetCache``. The RAM tier is per rank. The disk tier may be shared
by every rank on a node: writes are atomic (temporary name, then rename),
an entry one rank writes is adopted into another rank's accounting the
first time it is hit, and losing a write race or reading an entry that
another rank is evicting simply degrades to a miss.

What is deliberately not here yet: tensor data
----------------------------------------------
Today no *bulk* tensor bytes are ever staged locally. Blobs carry only
small tensors, and a tree mirror symlinks every large file back to the
source, so the bulk of each sample is still read from the network
filesystem on every epoch. The cache removes the metadata cost and nothing
else. That is by design for this version, and it is also the obvious next
step.

The extension is a **third entry flavour for tensor blocks**:

- **Format.** ``torch.save`` on write, ``torch.load(weights_only=True)``
  on read. One call each way, tensors only, and the restricted unpickler
  is the sanctioned safe path. Do not raise the JSON blob's tensor cap to
  carry bulk data, and do not add a second deserializer for trees; both
  were considered and rejected.
- **Keys are block-aligned.** An entry is ``(kind, "<array-path>::<block
  index>")`` for a fixed block size, not an arbitrary row range. A read
  of rows ``[1000, 2000)`` becomes the blocks that cover it, each a cache
  lookup, and a later read of ``[1500, 2500)`` reuses the shared block
  rather than searching for an overlapping range. Assembling a range from
  blocks is a concatenation; the sampler's contiguous cyclic-block reads
  map onto it directly.
- **Fill with the reads that already happen.** ``preadv``-based row reads
  in the mesh readers produce exactly one contiguous byte range per block.
  Writing that range through to the disk tier as it is read turns the
  first epoch's I/O into the cache fill, with no separate staging pass
  and no upfront cost.
- **Streaming the whole dataset onto a node.** With a
  ``DistributedSampler``, each epoch hands every rank a different subset,
  and every rank writes what it reads into the shared disk tier. Over a
  few epochs the node accumulates the whole dataset on NVMe without any
  rank coordinating with another; adopt-on-hit accounting already makes
  that safe, and the budget clamp keeps it inside the device. From then
  on the network filesystem is idle.
- **Budgets.** Tensor blocks are large, so ``max_item_bytes`` and the
  ``"largest"`` eviction policy, both tuned for metadata, would evict them
  first. The tensor flavour needs its own admission limit and should
  default to LRU or FIFO within its own budget rather than sharing the
  metadata tiers' accounting.

Everything above the flavour line already works for it: the ledger, the
atomic writes, the shared-directory adoption, the startup scan, and the
disk-directory checks. The work is the block keying, the read-through
fill in the readers, and the separate budget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from tensordict import TensorDictBase
from tensordict.memmap import MemoryMappedTensor

from physicsnemo.datapipes.registry import register

logger = logging.getLogger(__name__)

CacheKey = tuple[str, str]
"""A cache key is ``(kind, identity)``.

``kind`` names the entry type and carries a format version, for example
``"mesh/v1"``. Bump the version when the cached representation changes and
old entries silently stop matching.

``identity`` is the normalized absolute source path. It may end in
``"::suffix"`` when one source file yields several entries.
"""

# get_or_load holds one of these locks for the duration of a load, so two
# threads asking for the same key run the loader once. Sixty-four stripes
# keep unrelated keys from waiting on each other.
_N_LOCK_STRIPES = 64

_BLOB_SUFFIX = ".json"
_TREE_SUFFIX = ".tree"
_SIZE_SUFFIX = ".size"


# ---------------------------------------------------------------------------
# Blob format
# ---------------------------------------------------------------------------
#
# Blobs are metadata, and metadata is JSON: zarr attributes are JSON on
# disk, tensorstore specs are JSON, key listings are lists of strings. So the
# on-disk form is plain JSON. Nothing executable can be loaded from it, and
# any tool can read it.
#
# Some metadata is tensor-shaped: a scalar Reynolds number, a (1, 3) inflow
# velocity, a TensorDict of a few such values. Those are carried inside the
# JSON as tagged nodes, e.g.
#
#     {"__tensor__": {"dtype": "float32", "shape": [1, 3], "data": [[...]]}}
#     {"__tensordict__": {"batch_size": [], "items": {"Re": {"__tensor__": …}}}}
#
# with a hard cap on element count so a bulk array can never be JSON-encoded
# by mistake. Anything over the cap, or not representable at all, is served
# from RAM only (see DatasetCache._load_blob).
#
# Bulk tensor data belongs in a separate, block-aligned entry flavour; see
# "What is deliberately not here yet" in the module docstring. Do not raise
# the cap to carry it.

# Largest tensor (in elements) that may travel inside a JSON blob. A (1, 3)
# vector is 3; a 64x64 lookup table is 4096. Anything bigger is bulk data.
_MAX_JSON_TENSOR_NUMEL = 4096

_TENSOR_TAG = "__tensor__"
_TENSORDICT_TAG = "__tensordict__"


def _tensor_to_node(t: torch.Tensor) -> dict[str, Any]:
    if t.numel() > _MAX_JSON_TENSOR_NUMEL:
        raise TypeError(
            f"tensor with {t.numel()} elements exceeds the JSON blob cap of "
            f"{_MAX_JSON_TENSOR_NUMEL}; bulk data is not a blob"
        )
    t = t.detach().cpu()
    if t.is_complex():
        raise TypeError("complex tensors are not JSON-representable")
    return {
        _TENSOR_TAG: {
            "dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            # float32 -> Python float -> float32 round-trips exactly, as does
            # every narrower float type; ints and bools are exact by nature.
            "data": t.tolist(),
        }
    }


def _node_to_tensor(node: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, node["dtype"], None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Not a torch dtype: {node['dtype']!r}")
    return torch.tensor(node["data"], dtype=dtype).reshape(node["shape"])


def _json_default(value: Any) -> Any:
    """``json.dumps`` hook: tag tensors and TensorDicts, refuse everything else."""
    if isinstance(value, torch.Tensor):
        return _tensor_to_node(value)
    if isinstance(value, TensorDictBase):
        return {
            _TENSORDICT_TAG: {
                "batch_size": list(value.batch_size),
                "items": dict(value.items()),  # values encoded recursively
            }
        }
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON-representable")


def _json_object_hook(obj: dict[str, Any]) -> Any:
    """``json.loads`` hook: rebuild tagged tensors and TensorDicts."""
    if len(obj) == 1:
        if _TENSOR_TAG in obj:
            return _node_to_tensor(obj[_TENSOR_TAG])
        if _TENSORDICT_TAG in obj:
            from tensordict import TensorDict

            node = obj[_TENSORDICT_TAG]
            return TensorDict(node["items"], batch_size=node["batch_size"])
    return obj


def encode_blob(value: Any) -> bytes:
    """Serialize a metadata value to JSON bytes.

    Parameters
    ----------
    value : Any
        Dicts with string keys, lists, tuples, strings, numbers, booleans,
        ``None``, ``Path`` (as a string), and small tensors or TensorDicts
        (see ``_MAX_JSON_TENSOR_NUMEL``), nested freely.

    Returns
    -------
    bytes
        Compact JSON with tagged nodes for tensors.

    Raises
    ------
    TypeError
        If *value* contains a tensor over the element cap, a complex
        tensor, or any other object JSON cannot express.

    Notes
    -----
    JSON has no tuple type, so tuples decode as lists. Tensors decode on
    CPU with their original dtype and shape.
    """
    return json.dumps(value, separators=(",", ":"), default=_json_default).encode()


def decode_blob(data: bytes) -> Any:
    """Inverse of :func:`encode_blob`.

    Parameters
    ----------
    data : bytes
        Bytes produced by :func:`encode_blob`.

    Returns
    -------
    Any
        The decoded value, tensors and TensorDicts rebuilt.

    Raises
    ------
    ValueError
        If *data* is not valid JSON or names an unknown dtype.
    """
    return json.loads(data, object_hook=_json_object_hook)


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

# Rough allowance for Python object headers, dict slots, and the like.
_PER_OBJECT_OVERHEAD = 128


def estimate_resident_size(value: Any, *, small_file_bytes: int = 0) -> int:
    """Estimate how many bytes of RAM *value* occupies.

    Used to charge tree entries against the RAM tier's budget.

    Parameters
    ----------
    value : Any
        Tensors, TensorDicts, tensorclasses such as ``Mesh``, and plain
        containers of those.
    small_file_bytes : int, default=0
        A memory-mapped tensor at or under this size is counted in full;
        a larger one is counted as a pointer. Small mapped files are
        metadata in practice and end up resident once touched. Large ones
        are the bulk data the cache deliberately leaves on the source
        filesystem.

    Returns
    -------
    int
        Estimated bytes, never less than ``_PER_OBJECT_OVERHEAD``.

    Notes
    -----
    Sizing must never break a read, so any error inside the walk yields
    the minimum size rather than propagating.
    """

    def _size(v: Any) -> int:
        if isinstance(v, torch.Tensor):
            nbytes = v.numel() * v.element_size()
            if isinstance(v, MemoryMappedTensor) and nbytes > small_file_bytes:
                return _PER_OBJECT_OVERHEAD
            return nbytes + _PER_OBJECT_OVERHEAD
        if isinstance(v, TensorDictBase):
            return _PER_OBJECT_OVERHEAD + sum(_size(x) for x in v.values())
        if hasattr(v, "_tensordict"):  # tensorclass (Mesh, DomainMesh, ...)
            return _size(v._tensordict)
        if isinstance(v, dict):
            return _PER_OBJECT_OVERHEAD + sum(_size(k) + _size(x) for k, x in v.items())
        if isinstance(v, (list, tuple, set)):
            return _PER_OBJECT_OVERHEAD + sum(_size(x) for x in v)
        if isinstance(v, (bytes, bytearray, str)):
            return len(v) + _PER_OBJECT_OVERHEAD
        return _PER_OBJECT_OVERHEAD

    try:
        return _size(value)
    except Exception:  # noqa: BLE001 - sizing must never break a read
        return _PER_OBJECT_OVERHEAD


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------
#
# A policy turns an entry's bookkeeping into a sort key. When a tier is over
# budget, entries are removed in ascending key order until it is back under.

EVICTION_POLICIES: dict[str, Callable[["_EntryMeta"], tuple]] = {
    # Largest first, oldest as tie-break. Every cached item saves roughly
    # the same number of metadata round trips regardless of its size, so
    # dropping the biggest entries keeps the most items per byte of budget.
    "largest": lambda e: (-e.size, e.insert_seq),
    "fifo": lambda e: (e.insert_seq,),
    "lru": lambda e: (e.last_access,),
}

# When over budget, evict down to this fraction of the limit rather than to
# the limit itself. Eviction sorts the whole table, so the headroom means
# that happens once per ~10% of churn instead of on every insert into a
# full tier.
_EVICT_TO_FRACTION = 0.9


@dataclass
class _EntryMeta:
    """What the ledger knows about one entry.

    Attributes
    ----------
    size : int
        Bytes charged against the tier's budget.
    insert_seq : int
        Insertion order, for FIFO.
    last_access : int
        Most recent hit, for LRU.
    kind : str
        The entry's kind, so ``clear(kind)`` can find it.
    value : Any
        The cached object. Set for RAM entries only; disk entries keep
        their data in the file.
    """

    size: int
    insert_seq: int
    last_access: int
    kind: str
    value: Any = None


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class _Ledger:
    """The bookkeeping both tiers share: a byte-budgeted table of entries.

    Each tier owns one ledger. The RAM tier keys it by cache key and stores
    the cached object in the entry. The disk tier keys it by file path and
    stores nothing but sizes. Both get the same counters and the same
    eviction.

    Eviction never touches the outside world. :meth:`record` removes
    victims from the table and hands them back, and the owning tier
    disposes of them after the lock is released. That matters for the disk
    tier, where disposal is file deletion and must not stall other
    threads' lookups.

    Parameters
    ----------
    limit_bytes : int
        Budget for the tier.
    policy : Callable[[_EntryMeta], tuple]
        One of :data:`EVICTION_POLICIES`.
    """

    def __init__(self, limit_bytes: int, policy: Callable[[_EntryMeta], tuple]):
        self.limit_bytes = limit_bytes
        self._policy = policy
        self._lock = threading.Lock()
        self._entries: dict[Any, _EntryMeta] = {}
        self._seq = 0
        self._total = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any) -> _EntryMeta | None:
        """Look *key* up, counting a hit or a miss.

        Parameters
        ----------
        key : Any
            Ledger key.

        Returns
        -------
        _EntryMeta or None
            The entry, with its recency refreshed, or ``None``.
        """
        meta = self.touch(key)
        if meta is None:
            self.miss()
        return meta

    def touch(self, key: Any) -> _EntryMeta | None:
        """Like :meth:`get`, but an unknown key counts nothing.

        The disk tier uses this when a file exists on disk but is not yet
        in this process's ledger because another rank wrote it. That is a
        hit to adopt, not a miss to count.

        Parameters
        ----------
        key : Any
            Ledger key.

        Returns
        -------
        _EntryMeta or None
            The entry, or ``None`` if unknown.
        """
        with self._lock:
            meta = self._entries.get(key)
            if meta is not None:
                self._seq += 1
                meta.last_access = self._seq
                self.hits += 1
            return meta

    def miss(self) -> None:
        """Count one miss."""
        with self._lock:
            self.misses += 1

    def record(
        self, key: Any, size: int, kind: str, *, value: Any = None, hit: bool = False
    ) -> list[tuple[Any, _EntryMeta]]:
        """Insert or replace *key*, then evict if over budget.

        Parameters
        ----------
        key : Any
            Ledger key.
        size : int
            Bytes to charge.
        kind : str
            Entry kind.
        value : Any, optional
            Cached object, RAM tier only.
        hit : bool, default=False
            Also count a hit. Used when adopting an entry that was found
            on disk.

        Returns
        -------
        list of (key, _EntryMeta)
            Entries evicted to make room. Already removed from the table;
            the caller disposes of whatever they refer to.
        """
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._total -= old.size
            self._seq += 1
            self._entries[key] = _EntryMeta(size, self._seq, self._seq, kind, value)
            self._total += size
            if hit:
                self.hits += 1
            return self._evict_locked()

    def _evict_locked(self) -> list[tuple[Any, _EntryMeta]]:
        """Evict in policy order until under ``limit * _EVICT_TO_FRACTION``.

        Caller holds the lock.
        """
        if self._total <= self.limit_bytes:
            return []
        target = self.limit_bytes * _EVICT_TO_FRACTION
        victims = []
        for key, meta in sorted(
            self._entries.items(), key=lambda kv: self._policy(kv[1])
        ):
            if self._total <= target:
                break
            del self._entries[key]
            self._total -= meta.size
            self.evictions += 1
            victims.append((key, meta))
        return victims

    def pop(self, key: Any) -> _EntryMeta | None:
        """Remove *key* from the table and return its entry, if any."""
        with self._lock:
            meta = self._entries.pop(key, None)
            if meta is not None:
                self._total -= meta.size
            return meta

    def pop_kind(self, kind: str | None) -> list[Any]:
        """Remove every entry of *kind*, or every entry if ``None``.

        Returns
        -------
        list
            The removed keys, for the caller to dispose of.
        """
        with self._lock:
            keys = [
                k for k, m in self._entries.items() if kind is None or m.kind == kind
            ]
            for k in keys:
                self._total -= self._entries.pop(k).size
            return keys

    def stats(self) -> dict[str, int]:
        """Counters: ``hits``, ``misses``, ``evictions``, ``bytes``, ``entries``."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "bytes": self._total,
                "entries": len(self._entries),
            }


# ---------------------------------------------------------------------------
# RAM tier
# ---------------------------------------------------------------------------


class _RamTier:
    """In-process store of loaded objects, keyed by cache key.

    A thin layer over :class:`_Ledger` that keeps the cached object inside
    the ledger entry. Per rank process; see the module docstring.

    Parameters
    ----------
    limit_bytes : int
        Budget for this tier.
    policy : Callable[[_EntryMeta], tuple]
        Eviction policy.
    """

    def __init__(self, limit_bytes: int, policy: Callable[[_EntryMeta], tuple]):
        self._ledger = _Ledger(limit_bytes, policy)

    @property
    def limit_bytes(self) -> int:
        return self._ledger.limit_bytes

    def get(self, key: CacheKey) -> tuple[Any, bool]:
        """Return ``(value, True)`` on a hit, ``(None, False)`` on a miss."""
        meta = self._ledger.get(key)
        return (meta.value, True) if meta is not None else (None, False)

    def put(self, key: CacheKey, value: Any, size: int) -> None:
        """Store *value* charged at *size* bytes.

        A value larger than the whole budget is not stored. Evicted values
        need no cleanup beyond dropping the reference, so the victims
        returned by the ledger are discarded.
        """
        if size > self._ledger.limit_bytes:
            return
        self._ledger.record(key, size, key[0], value=value)

    def invalidate(self, key: CacheKey) -> None:
        self._ledger.pop(key)

    def clear(self, kind: str | None = None) -> None:
        self._ledger.pop_kind(kind)

    def stats(self) -> dict[str, int]:
        return self._ledger.stats()


# ---------------------------------------------------------------------------
# Disk tier
# ---------------------------------------------------------------------------


def _sanitize_kind(kind: str) -> Path:
    """Turn a kind such as ``"mesh/v1"`` into a safe relative directory."""
    parts = [re.sub(r"[^A-Za-z0-9._-]", "_", p) or "_" for p in kind.split("/")]
    return Path(*parts)


def _dir_size(path: Path) -> int:
    """Bytes of the files and symlinks under *path*, ignoring directories.

    This is the same quantity ``_DiskTier._materialize`` adds up while
    writing a mirror, so a freshly computed size and a sidecar agree.
    """
    total = 0
    for p in path.rglob("*"):
        try:
            st = p.lstat()
        except OSError:
            continue
        if not p.is_dir() or p.is_symlink():
            total += st.st_size
    return total


def _remove(path: Path) -> None:
    """Delete a blob, a mirror, or a sidecar. Already gone is fine."""
    try:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError:
        pass  # another process may have removed it first


class _DiskTier:
    """A directory of JSON blobs and sparse tree mirrors on local storage.

    Point it at node-local NVMe, tmpfs, or scratch. Every rank on the node
    may use the same directory; see the module docstring for the layout.

    **Sharing between processes.** Writes go to a temporary name and are
    renamed into place, so a reader sees either nothing or a complete
    entry. Each process keeps its own ledger of what is in the directory.
    When a lookup finds an entry on disk that the ledger does not know,
    another process wrote it; the entry is *adopted* into the ledger on the
    spot, so every process's view converges on the shared contents and the
    byte limit is enforced against the directory rather than against this
    process's own writes alone. It remains a soft limit.

    **Sidecars.** A mirror ``<digest>.tree`` has a neighbour
    ``<digest>.size`` holding its byte total, so neither the startup scan
    nor adoption has to walk the mirror.

    Parameters
    ----------
    root : Path
        The cache directory. Created if missing.
    limit_bytes : int
        Budget for this tier.
    policy : Callable[[_EntryMeta], tuple]
        Eviction policy.
    small_file_bytes : int
        Files at or under this size are copied into a mirror; larger files
        are symlinked back to the source.
    """

    def __init__(
        self,
        root: Path,
        limit_bytes: int,
        policy: Callable[[_EntryMeta], tuple],
        small_file_bytes: int,
    ):
        self.root = root
        self.small_file_bytes = small_file_bytes
        self._ledger = _Ledger(limit_bytes, policy)
        self.root.mkdir(parents=True, exist_ok=True)
        self._scan()

    @property
    def limit_bytes(self) -> int:
        return self._ledger.limit_bytes

    # -- bookkeeping --------------------------------------------------------

    def _kind_of(self, path: Path) -> str:
        """Recover the (sanitized) kind from an entry's location."""
        return str(path.parent.relative_to(self.root))

    def _dispose(self, paths: Iterable[Path]) -> None:
        """Delete entries the ledger has already forgotten, sidecars included."""
        for path in paths:
            _remove(path)
            if path.name.endswith(_TREE_SUFFIX):
                _remove(path.with_suffix(_SIZE_SUFFIX))

    def _record(self, path: Path, size: int, *, hit: bool = False) -> None:
        """Add *path* to the ledger and delete whatever that evicts."""
        victims = self._ledger.record(path, size, self._kind_of(path), hit=hit)
        self._dispose(p for p, _ in victims)

    def _hit_or_adopt(self, path: Path, size_fn: Callable[[], int]) -> None:
        """Count a hit on *path*, adopting it first if this process never saw it.

        *size_fn* is only called on adoption, so the common path stays a
        single locked dictionary lookup.
        """
        if self._ledger.touch(path) is None:
            self._record(path, size_fn(), hit=True)

    def _tree_size(self, path: Path) -> int:
        """Mirror size from its sidecar, or by walking it if the sidecar is gone."""
        try:
            return int(path.with_suffix(_SIZE_SUFFIX).read_text())
        except (OSError, ValueError):
            return _dir_size(path)

    def _scan(self) -> None:
        """Adopt entries left by earlier runs, oldest first.

        One walk of the cache directory. Mirrors are recognised by suffix
        and not descended into: their contents are the dataset's own files,
        not entries. Temporary files and directories from interrupted
        writes are deleted along the way. A mirror without a sidecar is
        skipped here and adopted, with a computed size, on first hit.
        """
        found: list[tuple[float, Path, int]] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            here = Path(dirpath)
            keep = []
            for d in dirnames:
                p = here / d
                if ".tmp-" in d:
                    _remove(p)
                elif d.endswith(_TREE_SUFFIX):
                    try:
                        mtime = p.lstat().st_mtime
                        size = int(p.with_suffix(_SIZE_SUFFIX).read_text())
                    except (OSError, ValueError):
                        continue
                    found.append((mtime, p, size))
                else:
                    keep.append(d)
            dirnames[:] = keep  # os.walk only descends into what we leave here
            for f in filenames:
                p = here / f
                if ".tmp-" in f:
                    _remove(p)
                elif f.endswith(_BLOB_SUFFIX):
                    try:
                        st = p.lstat()
                    except OSError:
                        continue
                    found.append((st.st_mtime, p, st.st_size))
        for _, p, size in sorted(found):
            self._record(p, size)

    def _entry_path(self, key: CacheKey, suffix: str) -> Path:
        """``<root>/<kind>/<sha1(identity)><suffix>``.

        The hash is a filename, not a security boundary: it keeps paths
        short and free of the source path's characters.
        """
        kind, identity = key
        digest = hashlib.sha1(identity.encode(), usedforsecurity=False).hexdigest()
        return self.root / _sanitize_kind(kind) / f"{digest}{suffix}"

    def _tmp_path(self, final: Path) -> Path:
        """A temporary name beside *final*, unique to this process and thread."""
        return final.with_name(
            f"{final.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )

    # -- blobs ------------------------------------------------------------

    _MISS = object()

    def blob_get(self, key: CacheKey) -> tuple[Any, int] | object:
        """Read a blob.

        Returns
        -------
        (value, nbytes) or _MISS
            The decoded value and the file size, or :attr:`_MISS`. A file
            that fails to decode is deleted and reported as a miss.
        """
        path = self._entry_path(key, _BLOB_SUFFIX)
        try:
            data = path.read_bytes()
        except OSError:
            self._ledger.miss()
            return self._MISS
        try:
            value = decode_blob(data)
        except Exception:  # noqa: BLE001 - corrupt entry: drop and miss
            logger.warning("Dropping corrupt cache blob %s", path)
            self.invalidate(key)
            self._ledger.miss()
            return self._MISS
        self._hit_or_adopt(path, lambda: len(data))
        return value, len(data)

    def blob_put(self, key: CacheKey, encoded: bytes) -> None:
        """Write a blob atomically. A failed write is logged, not raised."""
        path = self._entry_path(key, _BLOB_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_path(path)
        try:
            tmp.write_bytes(encoded)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("Failed to write cache blob %s: %s", path, e)
            tmp.unlink(missing_ok=True)
            return
        self._record(path, len(encoded))

    # -- trees ------------------------------------------------------------

    def tree_get(self, key: CacheKey) -> Path | None:
        """Return the mirror directory for *key*, or ``None`` if there is none."""
        path = self._entry_path(key, _TREE_SUFFIX)
        if not path.is_dir():
            self._ledger.miss()
            return None
        self._hit_or_adopt(path, lambda: self._tree_size(path))
        return path

    def tree_put(self, key: CacheKey, src: Path) -> Path:
        """Build a sparse mirror of *src* and return its path.

        The mirror is assembled under a temporary name, its sidecar is
        written, and then the directory is renamed into place. If another
        process renamed its own mirror first, ours is discarded and theirs
        is returned; both were built from the same source, so the sidecar
        we already wrote is correct for either.

        Parameters
        ----------
        key : CacheKey
            Entry key.
        src : Path
            Source directory.

        Returns
        -------
        Path
            The mirror directory to hand to the stock loader.
        """
        final = self._entry_path(key, _TREE_SUFFIX)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_path(final)
        # resolve() here is on the miss path and gives symlinks a real target.
        size = self._materialize(src.resolve(), tmp)
        sidecar = final.with_suffix(_SIZE_SUFFIX)
        tmp_sidecar = self._tmp_path(sidecar)
        tmp_sidecar.write_text(str(size))
        os.replace(tmp_sidecar, sidecar)
        try:
            os.rename(tmp, final)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not final.is_dir():
                raise
            return final
        self._record(final, size)
        return final

    def _materialize(self, src: Path, dst: Path) -> int:
        """Recursively copy small files and symlink large ones; return bytes written."""
        size = 0
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                size += self._materialize(child, dst / child.name)
            return size
        if src.lstat().st_size <= self.small_file_bytes:
            shutil.copyfile(src, dst)
        else:
            os.symlink(src, dst)
        return dst.lstat().st_size

    # -- shared -----------------------------------------------------------

    def invalidate(self, key: CacheKey) -> None:
        """Forget and delete *key*, whichever flavour it is."""
        for suffix in (_BLOB_SUFFIX, _TREE_SUFFIX):
            path = self._entry_path(key, suffix)
            self._ledger.pop(path)
            self._dispose([path])

    def clear(self, kind: str | None = None) -> None:
        """Forget and delete every entry of *kind*, or everything."""
        target = None if kind is None else str(_sanitize_kind(kind))
        self._dispose(self._ledger.pop_kind(target))

    def stats(self) -> dict[str, int]:
        return self._ledger.stats()


# ---------------------------------------------------------------------------
# Disk directory checks
# ---------------------------------------------------------------------------
#
# Run once when a DatasetCache is built, before the disk tier exists. The
# tier itself knows nothing about filesystems; these checks only decide
# whether the directory is a sensible place for one and how big it may get.

# Filesystem types that mean "this is the network storage the cache exists
# to avoid". Matched by prefix against the lowercased type from the mount
# table, so "nfs4", "fuse.sshfs", "smb3" are all caught.
_NETWORK_FS_PREFIXES = (
    "lustre",
    "nfs",
    "gpfs",
    "beegfs",
    "ceph",
    "gluster",
    "panfs",
    "cifs",
    "smb",
    "afs",
    "9p",
    "daos",
    "weka",
    "fuse.sshfs",
    "fuse.gcsfuse",
    "fuse.s3fs",
)

# The disk budget is clamped to this fraction of the free space at startup.
_DISK_BUDGET_FRACTION = 0.8


def _nearest_existing(path: Path) -> Path:
    """*path* or its closest existing ancestor (the directory may not exist yet)."""
    path = Path(os.path.realpath(path))
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _filesystem_type(path: Path) -> str | None:
    """Lowercased filesystem type of the mount holding *path*, or ``None``.

    Uses the mount table via :mod:`psutil`. Any failure returns ``None``;
    a detection problem must never stop training.
    """
    try:
        import psutil

        real = str(_nearest_existing(path))
        mounts = sorted(
            psutil.disk_partitions(all=True), key=lambda m: -len(m.mountpoint)
        )
        for m in mounts:
            root = m.mountpoint.rstrip("/")
            # root == "" is the filesystem root, which holds everything.
            if not root or real == root or real.startswith(root + "/"):
                return m.fstype.lower()
    except Exception:  # noqa: BLE001 - best effort only
        return None
    return None


def check_disk_dir(path: Path, limit_bytes: int, *, allow_network: bool = False) -> int:
    """Vet a disk-tier directory and return the byte budget it can support.

    Three checks, in order:

    1. **Network filesystems are refused.** A cache on Lustre or NFS turns
       every one of its many small files into a metadata-server round
       trip, which is the cost the cache exists to remove. Pass
       ``allow_network=True`` to downgrade the error to a warning.
    2. **tmpfs is called out.** On many nodes ``/tmp`` is memory. The
       tier still works, but it is then RAM shared with training and with
       every rank's RAM tier, and the budget below matters all the more.
    3. **The budget is clamped to the device.** If *limit_bytes* exceeds
       ``_DISK_BUDGET_FRACTION`` of the free space, the smaller value is
       returned with a warning, so a 200 GiB default cannot fill a 100 GiB
       partition. Free space is measured once, at startup, and does not
       count entries already in the directory.

    Parameters
    ----------
    path : Path
        The intended ``disk_dir``. Need not exist yet.
    limit_bytes : int
        The configured ``disk_bytes_limit``.
    allow_network : bool, default=False
        Warn instead of raising on a network filesystem.

    Returns
    -------
    int
        The effective byte budget, ``<= limit_bytes``.

    Raises
    ------
    ValueError
        If *path* is on a network filesystem and *allow_network* is False.
    """
    fstype = _filesystem_type(path)
    if fstype is not None and fstype.startswith(_NETWORK_FS_PREFIXES):
        msg = (
            f"DatasetCache disk_dir {str(path)!r} is on a {fstype!r} filesystem. "
            "The disk tier writes many small files and must live on node-local "
            "storage (NVMe, tmpfs, or $TMPDIR)."
        )
        if not allow_network:
            raise ValueError(msg + " Pass allow_network_disk=True to override.")
        logger.warning("%s Continuing because allow_network_disk=True.", msg)
    if fstype == "tmpfs":
        logger.warning(
            "DatasetCache disk_dir %s is tmpfs: the disk tier is RAM-backed on "
            "this node and shares memory with training and the RAM tiers.",
            path,
        )
    effective = limit_bytes
    try:
        free = shutil.disk_usage(_nearest_existing(path)).free
    except OSError:
        free = None
    if free is not None and limit_bytes > free * _DISK_BUDGET_FRACTION:
        effective = int(free * _DISK_BUDGET_FRACTION)
        logger.warning(
            "DatasetCache disk_bytes_limit %.1f GiB exceeds %d%% of the %.1f GiB "
            "free at %s; using %.1f GiB.",
            limit_bytes / 2**30,
            int(_DISK_BUDGET_FRACTION * 100),
            free / 2**30,
            path,
            effective / 2**30,
        )
    logger.info(
        "DatasetCache disk tier at %s (%s), budget %.1f GiB",
        path,
        fstype or "unknown fs",
        effective / 2**30,
    )
    return effective


# ---------------------------------------------------------------------------
# DatasetCache
# ---------------------------------------------------------------------------


@register()
class DatasetCache:
    """Two-tier cache for the small, repeated reads that readers make.

    The module docstring explains the design; this class is its front
    door. Both tiers are optional and independent: ``DatasetCache()`` is a
    RAM-only cache, ``DatasetCache(ram_bytes_limit=None, disk_dir=...)`` is
    disk-only. One instance is normally shared by every reader in a
    process, since keys are namespaced by kind.

    Parameters
    ----------
    ram_bytes_limit : int or None, default=2 GiB
        Budget for the RAM tier, **per process** (so per rank). ``None``
        disables the tier.
    disk_dir : Path or str or None, default=None
        Directory for the disk tier: node-local NVMe, tmpfs, or scratch.
        ``None`` disables the tier. Safe to share between the ranks on a
        node.
    disk_bytes_limit : int, default=200 GiB
        Budget for the disk tier. A soft limit; see the notes.
    eviction : {"largest", "fifo", "lru"}, default="largest"
        Which entries go first when a tier is over budget. ``"largest"``
        keeps the most items per byte, which is what a metadata cache
        wants.
    max_item_bytes : int, default=8 MiB
        A blob larger than this is returned to the caller but never
        cached, so a bulk array that was routed here by mistake cannot
        crowd out metadata.
    small_file_bytes : int, default=64 KiB
        In a mirror, files at or under this size are copied and larger
        files are symlinked. In RAM sizing, a memory-mapped tensor at or
        under this size counts as resident.
    validate : {"none", "mtime"}, default="none"
        ``"none"`` trusts that sources never change, which costs nothing.
        ``"mtime"`` stats each tree source once per process and drops
        entries older than it. Use it while a dataset is still being
        edited.
    allow_network_disk : bool, default=False
        The disk tier refuses to live on a network filesystem (Lustre,
        NFS, and friends), since that recreates the cost it removes. Set
        True to turn that error into a warning. See :func:`check_disk_dir`.

    Notes
    -----
    **Sizing the RAM tier for multi-GPU.** Each rank has its own RAM tier.
    A ``DistributedSampler`` hands different samples to each rank every
    epoch, so over time every rank's RAM tier fills with every sample's
    metadata. The bytes are small, but each memory-mapped leaf held there
    is a live mapping; for very large datasets, check the count against
    the kernel's per-process mapping limit.

    **The disk limit is soft.** Each process enforces it against the
    entries it has seen, and it learns of other ranks' entries as it hits
    them, so a freshly started rank on a busy node may briefly overshoot.
    At startup the limit is also clamped to the free space on the device;
    see :func:`check_disk_dir`.

    **Choosing ``disk_dir``.** Node-local NVMe is best. ``$TMPDIR`` is a
    good default on schedulers that provide per-job local scratch. Note
    that ``/tmp`` is memory on many nodes, and that anything under it is
    usually gone after the job, so later runs start cold.

    **Returned objects are read-only.** RAM hits share tensor storage with
    every other caller.

    Examples
    --------
    >>> import torch
    >>> cache = DatasetCache(ram_bytes_limit=2**20)
    >>> calls = []
    >>> def loader():
    ...     calls.append(1)
    ...     return {"Re": torch.tensor(1.0e6)}
    >>> a = cache.get_or_load(("global-data/v1", "/data/run_1"), loader)
    >>> b = cache.get_or_load(("global-data/v1", "/data/run_1"), loader)
    >>> len(calls)
    1
    """

    def __init__(
        self,
        *,
        ram_bytes_limit: int | None = 2 * 2**30,
        disk_dir: Path | str | None = None,
        disk_bytes_limit: int = 200 * 2**30,
        eviction: str = "largest",
        max_item_bytes: int = 8 * 2**20,
        small_file_bytes: int = 64 * 2**10,
        validate: str = "none",
        allow_network_disk: bool = False,
    ) -> None:
        if eviction not in EVICTION_POLICIES:
            raise ValueError(
                f"Unknown eviction policy {eviction!r}; "
                f"choose from {sorted(EVICTION_POLICIES)}"
            )
        if validate not in ("none", "mtime"):
            raise ValueError(f"validate must be 'none' or 'mtime', got {validate!r}")
        policy = EVICTION_POLICIES[eviction]
        self.max_item_bytes = max_item_bytes
        self.small_file_bytes = small_file_bytes
        self.validate = validate
        self._ram = (
            _RamTier(ram_bytes_limit, policy) if ram_bytes_limit is not None else None
        )
        self._disk = None
        if disk_dir is not None:
            disk_dir = Path(disk_dir)
            budget = check_disk_dir(
                disk_dir, disk_bytes_limit, allow_network=allow_network_disk
            )
            self._disk = _DiskTier(disk_dir, budget, policy, small_file_bytes)
        self._locks = [threading.RLock() for _ in range(_N_LOCK_STRIPES)]
        # Small bookkeeping sets that share one lock: keys already checked
        # under validate="mtime", and kinds already warned about as RAM-only.
        self._validated: set[CacheKey] = set()
        self._ram_only_kinds: set[str] = set()
        self._validated_lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def get_or_load(
        self,
        key: CacheKey,
        loader: Callable[..., Any],
        *,
        src: Path | str | None = None,
    ) -> Any:
        """Return the value for *key*, running *loader* only on a miss.

        Parameters
        ----------
        key : CacheKey
            ``(kind, identity)``; see :data:`CacheKey`.
        loader : Callable
            For a blob entry, called with no arguments and must return a
            metadata value: JSON types plus small tensors and TensorDicts
            (anything else is cached in RAM only).
            For a tree entry, called with one ``Path`` argument: the
            mirror directory when the disk tier has one, otherwise *src*.
        src : Path or str, optional
            Giving a source directory makes this a tree entry.

        Returns
        -------
        Any
            The loaded or cached value. For tree entries served from RAM
            this is a shallow copy (fresh structure, shared tensor
            storage) when the object supports ``.copy()``.

        Notes
        -----
        Concurrent calls for one key run the loader once: the call holds
        one of a fixed set of re-entrant locks for its duration. A loader
        may call back into the cache from the same thread. It must not
        wait on another thread that might itself be inside the cache.
        """
        key = (str(key[0]), str(key[1]))
        lock = self._locks[hash(key) % _N_LOCK_STRIPES]
        with lock:
            self._maybe_validate(key, src)
            if self._ram is not None:
                value, hit = self._ram.get(key)
                if hit:
                    return self._copy_on_hit(value) if src is not None else value
            if src is not None:
                return self._load_tree(key, loader, Path(src))
            return self._load_blob(key, loader)

    def invalidate(self, key: CacheKey) -> None:
        """Drop *key* from every tier. A key that was never cached is fine."""
        key = (str(key[0]), str(key[1]))
        if self._ram is not None:
            self._ram.invalidate(key)
        if self._disk is not None:
            self._disk.invalidate(key)

    def clear(self, kind: str | None = None) -> None:
        """Drop every entry, or only those of one *kind*."""
        if self._ram is not None:
            self._ram.clear(kind)
        if self._disk is not None:
            self._disk.clear(kind)

    def stats(self) -> dict[str, dict[str, int]]:
        """Counters for this process, keyed ``"ram"`` and/or ``"disk"``.

        Each tier reports ``hits``, ``misses``, ``evictions``, ``bytes``,
        and ``entries``.
        """
        out: dict[str, dict[str, int]] = {}
        if self._ram is not None:
            out["ram"] = self._ram.stats()
        if self._disk is not None:
            out["disk"] = self._disk.stats()
        return out

    def close(self) -> None:
        """Release the RAM tier. Disk entries stay for the next run."""
        if self._ram is not None:
            self._ram.clear()

    def __repr__(self) -> str:
        ram = self._ram.limit_bytes if self._ram is not None else None
        disk = str(self._disk.root) if self._disk is not None else None
        return f"DatasetCache(ram_bytes_limit={ram}, disk_dir={disk!r})"

    # -- internals ----------------------------------------------------------

    def _copy_on_hit(self, value: Any) -> Any:
        """Shallow-copy a tree object so callers cannot mutate the cached one."""
        copy = getattr(value, "copy", None)
        return copy() if callable(copy) else value

    def _load_tree(
        self, key: CacheKey, loader: Callable[[Path], Any], src: Path
    ) -> Any:
        """RAM missed. Load a tree entry via the mirror if possible, else the source.

        A mirror that fails to load (torn, stale, or evicted by another
        rank between our lookup and our read) is invalidated and the
        source is used instead. The loader is trusted to raise on a broken
        mirror; it is the same loader that would raise on a broken source.
        """
        value = None
        if self._disk is not None:
            local = self._disk.tree_get(key)
            if local is None:
                try:
                    local = self._disk.tree_put(key, src)
                except OSError as e:
                    logger.warning("Cache mirror of %s failed: %s", src, e)
                    local = None
            if local is not None:
                try:
                    value = loader(local)
                except Exception as e:  # noqa: BLE001 - stale/torn/evicted mirror
                    logger.warning(
                        "Loading %s from cache mirror failed (%s); "
                        "falling back to source",
                        src,
                        e,
                    )
                    self.invalidate(key)
                    value = None
        if value is None:
            value = loader(src)
        if self._ram is not None:
            size = estimate_resident_size(value, small_file_bytes=self.small_file_bytes)
            self._ram.put(key, value, size)
        return self._copy_on_hit(value)

    def _load_blob(self, key: CacheKey, loader: Callable[[], Any]) -> Any:
        """RAM missed. Try the disk blob, else run the loader and write through.

        The same byte count is charged to both tiers: the JSON length when
        the value is JSON, an estimate otherwise. Values over
        ``max_item_bytes`` are returned but not cached anywhere.
        """
        if self._disk is not None:
            found = self._disk.blob_get(key)
            if found is not self._disk._MISS:
                value, size = found
                if self._ram is not None and size <= self.max_item_bytes:
                    self._ram.put(key, value, size)
                return value
        value = loader()
        encoded: bytes | None = None
        if self._disk is not None:
            try:
                encoded = encode_blob(value)
            except TypeError:
                self._warn_ram_only(key[0], value)
        size = len(encoded) if encoded is not None else estimate_resident_size(value)
        if size > self.max_item_bytes:
            return value
        if encoded is not None:
            self._disk.blob_put(key, encoded)
        if self._ram is not None:
            self._ram.put(key, value, size)
        return value

    def _warn_ram_only(self, kind: str, value: Any) -> None:
        """Warn once per kind that its values cannot be written to disk."""
        with self._validated_lock:
            if kind in self._ram_only_kinds:
                return
            self._ram_only_kinds.add(kind)
        logger.warning(
            "Cache kind %r holds %s values that cannot be written as a JSON "
            "blob (not representable, or a tensor over %d elements); they are "
            "served from the RAM tier only and never written to disk.",
            kind,
            type(value).__name__,
            _MAX_JSON_TENSOR_NUMEL,
        )

    def _maybe_validate(self, key: CacheKey, src: Path | str | None) -> None:
        """Under ``validate="mtime"``, drop disk entries older than their source.

        Runs once per key per process. A source that cannot be stat'ed is
        left alone.
        """
        if self.validate != "mtime" or src is None:
            return
        with self._validated_lock:
            if key in self._validated:
                return
            self._validated.add(key)
        try:
            src_mtime = Path(src).stat().st_mtime
        except OSError:
            return
        if self._disk is not None:
            for suffix in (_BLOB_SUFFIX, _TREE_SUFFIX):
                path = self._disk._entry_path(key, suffix)
                try:
                    if path.lstat().st_mtime < src_mtime:
                        logger.info("Cache entry for %s is stale; invalidating", src)
                        self.invalidate(key)
                        return
                except OSError:
                    continue


def cached_or_load(
    cache: DatasetCache | None,
    kind: str,
    path: Path | str,
    loader: Callable[..., Any],
    *,
    src: Path | str | None = None,
) -> Any:
    """Run *loader* through *cache* if there is one, or directly if not.

    This is what a reader's ``_cached`` helper calls, so readers need no
    ``if self._cache is not None`` branches of their own.

    Parameters
    ----------
    cache : DatasetCache or None
        ``None`` means call the loader and return.
    kind : str
        Entry kind with a format version, e.g. ``"zarr-attrs/v1"``.
    path : Path or str
        The path that identifies the entry.
    loader : Callable
        See :meth:`DatasetCache.get_or_load`.
    src : Path or str, optional
        Source directory for a tree entry.

    Returns
    -------
    Any
        The loaded or cached value.

    Notes
    -----
    Identity is ``os.path.abspath(path)``, which is pure string
    normalization. It is deliberately not ``Path.resolve()``: resolving
    stats every component of the path, on the source filesystem, for
    every sample, which is exactly the cost this cache exists to remove.
    The consequence is that two symlinks to one dataset are two entries.
    That wastes cache space; it never returns the wrong data.
    """
    if cache is None:
        return loader(src) if src is not None else loader()
    return cache.get_or_load((kind, os.path.abspath(path)), loader, src=src)
