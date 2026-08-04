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
Generic two-tier cache for datapipes readers.

On network filesystems (Lustre in particular), per-sample read cost is often
dominated by small, latency-bound metadata operations -- ``meta.json`` reads,
``readdir`` calls, per-leaf ``stat``/open/mmap -- that are identical every
epoch. :class:`DatasetCache` removes those repeated hits with two optional,
independent tiers:

- a **RAM tier** (byte-limited dict of deserialized values / loaded objects),
- a **disk tier** (byte-limited directory of serialized blobs and sparse
  directory mirrors -- point it at node-local NVMe, tmpfs, or scratch).

Entries come in two flavors, both behind one call, ``get_or_load``:

- **Blob entries**: small values (attribute dicts, key lists, glob results,
  global-data TensorDicts). The loader runs on a miss; the result is written
  through to both configured tiers. Disk serialization is a no-pickle,
  header+raw-buffers format that is safe to load from a shared directory.
- **Tree-backed entries** (``src=path``): sources that are directory trees of
  many small files plus a few large ones (tensordict memmap trees, zarr
  directory stores). The caller's *stock loader* is always used -- there is
  deliberately no second deserialization path. The RAM representation is the
  loaded object itself (memmap leaves are pointers, so only small resident
  metadata is counted); the disk representation is a *sparse mirror* of the
  source tree (files under ``small_file_bytes`` copied, larger files
  symlinked back to the source), against which the stock loader runs.

Caching is only valid for **raw, immutable artifacts**: never cache
subsampled or otherwise RNG-dependent results, and treat returned objects as
read-only (RAM-tier hits share tensor storage across reads).

The cache is thread-safe for the single-process, multi-threaded physicsnemo
DataLoader. Multiple processes (e.g. DDP ranks on one node) may share a
``disk_dir``: all writes are atomic (tmp + rename), and losing a write race
or reading a concurrently evicted entry degrades to a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.memmap import MemoryMappedTensor

from physicsnemo.datapipes.registry import register

logger = logging.getLogger(__name__)

CacheKey = tuple[str, str]
"""Cache key: ``(kind, identity)``.

``kind`` names the entry type and embeds a format version (``"mesh/v1"``);
``identity`` is the fully resolved source path (plus an optional
``"::suffix"`` for sub-file entries).
"""

_N_LOCK_STRIPES = 64

_BLOB_MAGIC = b"PNCB1\n"
_BLOB_SUFFIX = ".pnc"
_TREE_SUFFIX = ".tree"


# ---------------------------------------------------------------------------
# Blob codec (no pickle)
# ---------------------------------------------------------------------------
#
# A blob file is: magic, u64 header length, JSON header, concatenated raw
# buffers. The header holds a JSON "structure" tree whose leaves reference
# buffers by index. Supported values: JSON scalars, str, bytes, torch.Tensor,
# dict/list/tuple compositions thereof, and TensorDict.


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _str_to_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Not a torch dtype: {name!r}")
    return dtype


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu().flatten()
    if t.numel() == 0:
        return b""
    return t.view(torch.uint8).numpy().tobytes()


def _encode_node(value: Any, buffers: list[bytes]) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"t": "json", "v": value}
    if isinstance(value, (bytes, bytearray)):
        buffers.append(bytes(value))
        return {"t": "bytes", "i": len(buffers) - 1}
    if isinstance(value, torch.Tensor):
        buffers.append(_tensor_to_bytes(value))
        return {
            "t": "tensor",
            "i": len(buffers) - 1,
            "dtype": _dtype_to_str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, TensorDictBase):
        return {
            "t": "tensordict",
            "v": {str(k): _encode_node(v, buffers) for k, v in value.items()},
        }
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("Only str-keyed dicts are cacheable as blobs")
        return {
            "t": "dict",
            "v": {k: _encode_node(v, buffers) for k, v in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {
            "t": "list" if isinstance(value, list) else "tuple",
            "v": [_encode_node(v, buffers) for v in value],
        }
    if isinstance(value, Path):
        return {"t": "path", "v": str(value)}
    raise TypeError(f"Value of type {type(value).__name__} is not cacheable as a blob")


def _decode_node(node: dict[str, Any], buffers: list[bytes]) -> Any:
    t = node["t"]
    if t == "json":
        return node["v"]
    if t == "bytes":
        return buffers[node["i"]]
    if t == "tensor":
        dtype = _str_to_dtype(node["dtype"])
        shape = node["shape"]
        raw = buffers[node["i"]]
        if len(raw) == 0:
            return torch.empty(shape, dtype=dtype)
        # bytearray copy: frombuffer requires a writable buffer for a
        # writable tensor, and we must not alias the file read buffer.
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    if t == "tensordict":
        return TensorDict(
            {k: _decode_node(v, buffers) for k, v in node["v"].items()},
            device=torch.device("cpu"),
        )
    if t == "dict":
        return {k: _decode_node(v, buffers) for k, v in node["v"].items()}
    if t == "list":
        return [_decode_node(v, buffers) for v in node["v"]]
    if t == "tuple":
        return tuple(_decode_node(v, buffers) for v in node["v"])
    if t == "path":
        return Path(node["v"])
    raise ValueError(f"Unknown blob node type: {t!r}")


def encode_blob(value: Any, *, kind: str, identity: str) -> bytes:
    """Serialize a blob value to the on-disk format (no pickle)."""
    buffers: list[bytes] = []
    structure = _encode_node(value, buffers)
    header = json.dumps(
        {
            "version": 1,
            "kind": kind,
            "identity": identity,
            "structure": structure,
            "buffer_sizes": [len(b) for b in buffers],
        }
    ).encode()
    return b"".join([_BLOB_MAGIC, struct.pack("<Q", len(header)), header, *buffers])


def decode_blob(data: bytes) -> Any:
    """Deserialize a blob previously produced by :func:`encode_blob`."""
    if data[: len(_BLOB_MAGIC)] != _BLOB_MAGIC:
        raise ValueError("Not a datapipes cache blob (bad magic)")
    offset = len(_BLOB_MAGIC)
    (header_len,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    header = json.loads(data[offset : offset + header_len])
    offset += header_len
    buffers: list[bytes] = []
    for size in header["buffer_sizes"]:
        buffers.append(data[offset : offset + size])
        offset += size
    return _decode_node(header["structure"], buffers)


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

_PER_OBJECT_OVERHEAD = 128


def estimate_resident_size(value: Any, *, small_file_bytes: int = 0) -> int:
    """Estimate the resident RAM footprint of a value, in bytes.

    ``MemoryMappedTensor`` leaves are pointers into files, not resident
    bytes: large ones count as zero, while ones at or under
    ``small_file_bytes`` count in full (they are metadata-class and resident
    in practice once touched). Everything else counts its ``nbytes`` / length
    plus a small fixed per-object overhead.
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
# Eviction policies
# ---------------------------------------------------------------------------
#
# A policy maps an entry's metadata to a sort key; entries are evicted in
# ascending key order until the tier is back under its byte limit.

EVICTION_POLICIES: dict[str, Callable[["_EntryMeta"], tuple]] = {
    # Largest first (FIFO tie-break): every cached item saves ~the same
    # number of metadata round-trips regardless of size, so shedding the
    # biggest entries keeps the most items resident per byte of cache.
    "largest": lambda e: (-e.size, e.insert_seq),
    "fifo": lambda e: (e.insert_seq,),
    "lru": lambda e: (e.last_access,),
}


@dataclass
class _EntryMeta:
    size: int
    insert_seq: int
    last_access: int
    kind: str


# ---------------------------------------------------------------------------
# RAM tier
# ---------------------------------------------------------------------------


@dataclass
class _RamEntry:
    value: Any
    meta: _EntryMeta
    is_tree_object: bool


class _RamTier:
    """Byte-limited, thread-safe in-process store of deserialized values."""

    def __init__(self, limit_bytes: int, policy: Callable[[_EntryMeta], tuple]):
        self.limit_bytes = limit_bytes
        self._policy = policy
        self._entries: dict[CacheKey, _RamEntry] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._total = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: CacheKey) -> tuple[Any, bool]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None, False
            self._seq += 1
            entry.meta.last_access = self._seq
            self.hits += 1
            return entry.value, True

    def put(
        self, key: CacheKey, value: Any, size: int, *, is_tree_object: bool
    ) -> None:
        with self._lock:
            if size > self.limit_bytes:
                return
            old = self._entries.pop(key, None)
            if old is not None:
                self._total -= old.meta.size
            self._seq += 1
            self._entries[key] = _RamEntry(
                value, _EntryMeta(size, self._seq, self._seq, key[0]), is_tree_object
            )
            self._total += size
            self._evict_locked()

    def _evict_locked(self) -> None:
        if self._total <= self.limit_bytes:
            return
        victims = sorted(self._entries.items(), key=lambda kv: self._policy(kv[1].meta))
        for key, entry in victims:
            if self._total <= self.limit_bytes:
                break
            del self._entries[key]
            self._total -= entry.meta.size
            self.evictions += 1

    def invalidate(self, key: CacheKey) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total -= entry.meta.size

    def clear(self, kind: str | None = None) -> None:
        with self._lock:
            if kind is None:
                self._entries.clear()
                self._total = 0
                return
            for key in [k for k in self._entries if k[0] == kind]:
                self._total -= self._entries.pop(key).meta.size

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "bytes": self._total,
                "entries": len(self._entries),
            }


# ---------------------------------------------------------------------------
# Disk tier
# ---------------------------------------------------------------------------


def _sanitize_kind(kind: str) -> Path:
    parts = [re.sub(r"[^A-Za-z0-9._-]", "_", p) or "_" for p in kind.split("/")]
    return Path(*parts)


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            total += p.lstat().st_size
        except OSError:
            pass
    return total


class _DiskTier:
    """Byte-limited directory of serialized blobs and sparse tree mirrors.

    Safe to share between processes: writes are tmp + atomic rename, reads
    that race an eviction degrade to a miss. The byte limit is a soft target
    enforced from this process's view of the directory.
    """

    def __init__(
        self,
        root: Path,
        limit_bytes: int,
        policy: Callable[[_EntryMeta], tuple],
        small_file_bytes: int,
    ):
        self.root = root
        self.limit_bytes = limit_bytes
        self._policy = policy
        self.small_file_bytes = small_file_bytes
        self._lock = threading.Lock()
        self._entries: dict[Path, _EntryMeta] = {}  # entry path -> meta
        self._seq = 0
        self._total = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self._scan()

    def _scan(self) -> None:
        """Adopt pre-existing entries (persisted from earlier runs/ranks)."""
        found: list[tuple[float, Path, int]] = []
        for pattern in (f"*{_BLOB_SUFFIX}", f"*{_TREE_SUFFIX}"):
            for p in self.root.rglob(pattern):
                try:
                    if p.name.endswith(_TREE_SUFFIX) and p.is_dir():
                        found.append((p.lstat().st_mtime, p, _dir_size(p)))
                    elif p.name.endswith(_BLOB_SUFFIX) and p.is_file():
                        found.append((p.lstat().st_mtime, p, p.lstat().st_size))
                except OSError:
                    continue
        # Clean up orphaned tmp entries from interrupted writes.
        for p in self.root.rglob("*.tmp-*"):
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError:
                pass
        for mtime, p, size in sorted(found):
            self._seq += 1
            kind = str(p.parent.relative_to(self.root))
            self._entries[p] = _EntryMeta(size, self._seq, self._seq, kind)
            self._total += size

    def _entry_path(self, key: CacheKey, suffix: str) -> Path:
        kind, identity = key
        # Filename hash only (not security-sensitive): stable, short paths.
        digest = hashlib.sha1(identity.encode(), usedforsecurity=False).hexdigest()
        return self.root / _sanitize_kind(kind) / f"{digest}{suffix}"

    def _tmp_path(self, final: Path) -> Path:
        return final.with_name(
            f"{final.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )

    def _record(self, path: Path, size: int, kind: str) -> None:
        with self._lock:
            old = self._entries.pop(path, None)
            if old is not None:
                self._total -= old.size
            self._seq += 1
            self._entries[path] = _EntryMeta(size, self._seq, self._seq, kind)
            self._total += size
            self._evict_locked()

    def _touch(self, path: Path) -> None:
        with self._lock:
            meta = self._entries.get(path)
            if meta is not None:
                self._seq += 1
                meta.last_access = self._seq

    def _evict_locked(self) -> None:
        if self._total <= self.limit_bytes:
            return
        victims = sorted(self._entries.items(), key=lambda kv: self._policy(kv[1]))
        for path, meta in victims:
            if self._total <= self.limit_bytes:
                break
            del self._entries[path]
            self._total -= meta.size
            self.evictions += 1
            try:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            except OSError:
                pass  # another rank may have evicted it already

    # -- blobs ------------------------------------------------------------

    _MISS = object()

    def blob_get(self, key: CacheKey) -> Any:
        path = self._entry_path(key, _BLOB_SUFFIX)
        try:
            data = path.read_bytes()
        except OSError:
            with self._lock:
                self.misses += 1
            return self._MISS
        try:
            value = decode_blob(data)
        except Exception:  # noqa: BLE001 - corrupt entry: drop and miss
            logger.warning("Dropping corrupt cache blob %s", path)
            self.invalidate(key)
            with self._lock:
                self.misses += 1
            return self._MISS
        self._touch(path)
        with self._lock:
            self.hits += 1
        return value

    def blob_put(self, key: CacheKey, encoded: bytes) -> None:
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
        self._record(path, len(encoded), key[0])

    # -- trees ------------------------------------------------------------

    def tree_get(self, key: CacheKey) -> Path | None:
        path = self._entry_path(key, _TREE_SUFFIX)
        if path.is_dir():
            self._touch(path)
            with self._lock:
                self.hits += 1
            return path
        with self._lock:
            self.misses += 1
        return None

    def tree_put(self, key: CacheKey, src: Path) -> Path:
        """Materialize a sparse mirror of *src* and return its path.

        Files at or under ``small_file_bytes`` are copied; larger files
        become absolute symlinks back to the source. The finished mirror is
        moved into place atomically; if another process won the race, its
        mirror is used and ours is discarded.
        """
        final = self._entry_path(key, _TREE_SUFFIX)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_path(final)
        size = self._materialize(src.resolve(), tmp)
        try:
            os.rename(tmp, final)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not final.is_dir():
                raise
            return final
        self._record(final, size, key[0])
        return final

    def _materialize(self, src: Path, dst: Path) -> int:
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
        for suffix in (_BLOB_SUFFIX, _TREE_SUFFIX):
            path = self._entry_path(key, suffix)
            with self._lock:
                meta = self._entries.pop(path, None)
                if meta is not None:
                    self._total -= meta.size
            try:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            except OSError:
                pass

    def clear(self, kind: str | None = None) -> None:
        with self._lock:
            targets = [
                p
                for p, meta in self._entries.items()
                if kind is None or meta.kind == str(_sanitize_kind(kind))
            ]
            for p in targets:
                self._total -= self._entries.pop(p).size
        for p in targets:
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError:
                pass

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "bytes": self._total,
                "entries": len(self._entries),
            }


# ---------------------------------------------------------------------------
# DatasetCache
# ---------------------------------------------------------------------------


@register()
class DatasetCache:
    """Two-tier (RAM + disk) cache for small, immutable reader artifacts.

    See the module docstring for the design. Both tiers are optional and
    independent; ``DatasetCache()`` defaults to a RAM-only cache. One
    instance may be shared by any number of readers -- keys are namespaced
    by ``kind`` and identified by fully resolved source path.

    Parameters
    ----------
    ram_bytes_limit : int or None, default=2 GiB
        Byte budget for the RAM tier. ``None`` disables the tier.
    disk_dir : Path or str or None, default=None
        Directory for the disk tier (node-local NVMe, tmpfs, scratch).
        ``None`` disables the tier. May be shared between processes.
    disk_bytes_limit : int, default=200 GiB
        Byte budget for the disk tier (soft target).
    eviction : str, default="largest"
        Eviction policy: ``"largest"`` (default -- maximizes the number of
        metadata items kept off the source filesystem per byte of cache),
        ``"fifo"``, or ``"lru"``.
    max_item_bytes : int, default=8 MiB
        Blob admission limit: larger values are returned but never cached,
        so misclassified bulk arrays cannot pollute the cache.
    small_file_bytes : int, default=64 KiB
        Tree mirrors copy files at or under this size and symlink larger
        ones; RAM sizing counts memmap leaves at or under it as resident.
    validate : str, default="none"
        ``"none"`` trusts sources to be immutable (zero stat overhead).
        ``"mtime"`` stats each tree source once per process and invalidates
        entries older than the source -- for iterating on datasets during
        development.

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
        self._disk = (
            _DiskTier(Path(disk_dir), disk_bytes_limit, policy, small_file_bytes)
            if disk_dir is not None
            else None
        )
        self._locks = [threading.RLock() for _ in range(_N_LOCK_STRIPES)]
        self._validated: set[CacheKey] = set()
        self._validated_lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def get_or_load(
        self,
        key: CacheKey,
        loader: Callable[..., Any],
        *,
        src: Path | str | None = None,
        size_hint: int | None = None,
    ) -> Any:
        """Return the cached value for *key*, loading (and caching) on miss.

        Blob entries (``src=None``): ``loader()`` is called with no
        arguments on a miss and its result is written through to both tiers.

        Tree-backed entries (``src`` given): ``loader`` is always called
        with a path -- the disk-tier sparse mirror when available, otherwise
        the source -- and the loaded object itself is kept in the RAM tier.
        RAM hits return a shallow structural copy when the object supports
        ``.copy()`` (fresh structure, shared tensor storage); treat all
        returned data as read-only.

        Single-flight: concurrent calls for the same key run the loader
        once. Loaders must not call back into the cache (lock stripes).
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
                return self._load_tree(key, loader, Path(src), size_hint)
            return self._load_blob(key, loader, size_hint)

    def invalidate(self, key: CacheKey) -> None:
        """Drop *key* from all tiers (missing entries are fine)."""
        key = (str(key[0]), str(key[1]))
        if self._ram is not None:
            self._ram.invalidate(key)
        if self._disk is not None:
            self._disk.invalidate(key)

    def clear(self, kind: str | None = None) -> None:
        """Drop all entries (optionally only those of one ``kind``)."""
        if self._ram is not None:
            self._ram.clear(kind)
        if self._disk is not None:
            self._disk.clear(kind)

    def stats(self) -> dict[str, dict[str, int]]:
        """Per-tier hit/miss/eviction/byte/entry counters."""
        out: dict[str, dict[str, int]] = {}
        if self._ram is not None:
            out["ram"] = self._ram.stats()
        if self._disk is not None:
            out["disk"] = self._disk.stats()
        return out

    def close(self) -> None:
        """Release the RAM tier. Disk entries persist for later runs."""
        if self._ram is not None:
            self._ram.clear()

    def __repr__(self) -> str:
        ram = self._ram.limit_bytes if self._ram is not None else None
        disk = str(self._disk.root) if self._disk is not None else None
        return f"DatasetCache(ram_bytes_limit={ram}, disk_dir={disk!r})"

    # -- internals ----------------------------------------------------------

    def _copy_on_hit(self, value: Any) -> Any:
        copy = getattr(value, "copy", None)
        return copy() if callable(copy) else value

    def _load_tree(
        self,
        key: CacheKey,
        loader: Callable[[Path], Any],
        src: Path,
        size_hint: int | None,
    ) -> Any:
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
                except Exception as e:  # noqa: BLE001 - stale/torn mirror
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
            size = (
                size_hint
                if size_hint is not None
                else estimate_resident_size(
                    value, small_file_bytes=self.small_file_bytes
                )
            )
            self._ram.put(key, value, size, is_tree_object=True)
        return self._copy_on_hit(value)

    def _load_blob(
        self, key: CacheKey, loader: Callable[[], Any], size_hint: int | None
    ) -> Any:
        if self._disk is not None:
            value = self._disk.blob_get(key)
            if value is not self._disk._MISS:
                if self._ram is not None:
                    size = (
                        size_hint
                        if size_hint is not None
                        else estimate_resident_size(value)
                    )
                    if size <= self.max_item_bytes:
                        self._ram.put(key, value, size, is_tree_object=False)
                return value
        value = loader()
        encoded: bytes | None = None
        if self._disk is not None:
            try:
                encoded = encode_blob(value, kind=key[0], identity=key[1])
            except TypeError as e:
                logger.warning("Value for %s is not disk-cacheable: %s", key, e)
        size = (
            size_hint
            if size_hint is not None
            else (
                len(encoded) if encoded is not None else estimate_resident_size(value)
            )
        )
        if size > self.max_item_bytes:
            return value
        if encoded is not None:
            self._disk.blob_put(key, encoded)
        if self._ram is not None:
            self._ram.put(key, value, size, is_tree_object=False)
        return value

    def _maybe_validate(self, key: CacheKey, src: Path | str | None) -> None:
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
    """Route a small, immutable read through an optional cache.

    The shared implementation behind readers' ``_cached`` helpers. With
    ``cache=None`` this is a transparent call to *loader* (``loader(src)``
    when *src* is given, else ``loader()``); otherwise the entry is keyed by
    ``(kind, resolved path)`` and handled by
    :meth:`DatasetCache.get_or_load`.
    """
    if cache is None:
        return loader(src) if src is not None else loader()
    return cache.get_or_load((kind, str(Path(path).resolve())), loader, src=src)
