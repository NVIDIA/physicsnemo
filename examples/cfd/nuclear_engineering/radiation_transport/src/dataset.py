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

"""RTE data-source layer: mesh reader, PhysicsNeMo Dataset, and stats loaders.

This module is the bottom of the data dependency tree. It provides the
low-level Mesh access (``MeshDataReader``), a thin file-indexed
``physicsnemo.datapipes.Dataset`` subclass (``RTEBaseDataset``), and
helpers for reading the RTE-specific YAML statistics files into
PhysicsNeMo ``Normalize`` kwargs.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import yaml
from physicsnemo.datapipes.dataset import Dataset as PhysicsNeMoDataset
from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform
from physicsnemo.mesh import Mesh
from tensordict import TensorDict


# =========================================================================
# Mesh reader
# =========================================================================
#
# Filename-indexed reader over a directory of ``<name>.mesh/`` memmap
# directories. Inherits from ``physicsnemo.datapipes.readers.base.Reader``;
# returns ``TensorDict`` from ``load()``. The TensorDict carries both the
# tensor fields and non-tensor metadata (``metadata``, ``filename``) via
# ``NonTensorData`` entries.
#
# RTE-specific kwargs (``load_flux``, optional field loading, etc.) live on the
# filename-indexed ``load(filename, ...)`` entry. The int-indexed
# ``_load_sample(index)`` required by the PhysicsNeMo ``Reader`` contract uses
# defaults.


@register("RTEMeshReader")
class MeshDataReader(Reader):
    """Filename-indexed reader over a directory of RTE Mesh memmap stores.

    The ``TensorDict`` returned by ``load(filename, ...)`` carries the
    tensor fields RTE training and inference rely on. The on-disk format
    is the PhysicsNeMo ``Mesh`` memmap layout (``<name>.mesh/`` +
    ``<name>.attrs.json`` sidecar).

    Example:
        >>> reader = MeshDataReader("/path/to/mesh_stores/lattice")
        >>> filenames = reader.get_filenames()
        >>> td = reader.load(filenames[0])
        >>> print(td["coordinates"].shape)  # (N, 3)
    """

    def __init__(
        self,
        data_path: Path | str,
        case_type: Optional[str] = None,
        cache_static_arrays: bool = True,
        max_cache_size: int = 200,
    ):
        super().__init__(pin_memory=False, include_index_in_metadata=False)

        self.data_path = Path(data_path)
        self.case_type = case_type
        self.cache_static_arrays = cache_static_arrays
        self.max_cache_size = max_cache_size

        self._static_cache: OrderedDict[
            Tuple[str, bool, bool, bool], Dict[str, torch.Tensor]
        ] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._cache_lock = threading.Lock()

        self._metadata_cache: Dict[str, Dict] = {}

        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
        if not self.data_path.is_dir():
            raise ValueError(f"Data path {self.data_path} is not a directory")

        self._filenames: List[str] = self._scan_filenames()

    # ------------------------------------------------------------------
    # PhysicsNeMo ``Reader`` contract
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._filenames)

    def _load_sample(self, index: int) -> Dict[str, torch.Tensor]:
        td = self.load(self._filenames[index])
        return {key: td[key] for key in td.keys() if isinstance(td[key], torch.Tensor)}

    def _get_sample_metadata(self, index: int) -> Dict:
        filename = self._filenames[index]
        meta = self.get_metadata(filename)
        meta["filename"] = filename
        return meta

    # ------------------------------------------------------------------
    # Filename discovery + caching
    # ------------------------------------------------------------------

    def _scan_filenames(self) -> List[str]:
        filenames = []
        for item in self.data_path.iterdir():
            if item.is_dir() and item.name.endswith(".mesh"):
                if self.case_type is None or item.name.startswith(self.case_type):
                    filenames.append(item.name)
        return sorted(filenames)

    def get_filenames(self) -> List[str]:
        """Return a fresh list of discovered mesh store names."""
        return list(self._filenames)

    def get_cache_stats(self) -> Dict[str, float]:
        """Return current static-array cache hit/miss counters and size."""
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0
            return {
                "cache_size": len(self._static_cache),
                "max_cache_size": self.max_cache_size,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_evictions": self._cache_evictions,
                "hit_rate": hit_rate,
            }

    def clear_cache(self):
        """Drop the static-array cache and reset hit/miss/eviction counters."""
        with self._cache_lock:
            self._static_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_evictions = 0

    # ------------------------------------------------------------------
    # Sidecar + Mesh helpers
    # ------------------------------------------------------------------

    def _sidecar_path(self, filename: str) -> Path:
        # ``<name>.mesh`` -> ``<name>.attrs.json``
        stem = filename[: -len(".mesh")] if filename.endswith(".mesh") else filename
        return self.data_path / f"{stem}.attrs.json"

    def _read_sidecar(self, filename: str) -> Dict:
        sidecar = self._sidecar_path(filename)
        if not sidecar.exists():
            return {}
        with open(sidecar, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # The filename-indexed ``load`` stays the primary entry point
    # ------------------------------------------------------------------

    def load(
        self,
        filename: str,
        load_material_properties: bool = True,
        load_geometric_features: bool = True,
        load_sim_times: bool = True,
        load_sigma_fields: bool = True,
        load_flux: bool = True,
    ) -> TensorDict:
        """Load a Mesh memmap store into a ``TensorDict``.

        Tensor fields (``coordinates``, ``cell_areas``, ``scalar_flux``, and
        the optional ``sim_times`` / ``material_properties`` / ``geometric_features``
        / ``sigma_*`` / ``Q``) are stored as ``torch.Tensor`` entries. The
        sidecar attrs dict is stored as ``NonTensorData`` under ``metadata``.

        ``load_flux=False`` returns a placeholder ``scalar_flux`` of shape
        ``(1, N)`` — the caller is expected to overwrite it from a memory
        cache.
        """
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh store {filepath} not found")

        cache_key = (
            filename,
            bool(load_material_properties),
            bool(load_geometric_features),
            bool(load_sigma_fields),
        )

        # ``Mesh.load`` returns memmap-backed tensors. The single-process
        # ``physicsnemo.datapipes.DataLoader`` (CUDA streams, no fork) keeps
        # the tensors live in this process, so we let Mesh hand back the
        # memmap views directly
        mesh = Mesh.load(str(filepath))
        point_data = mesh.point_data
        global_data = mesh.global_data

        # ------- flux + timesteps (steady-state first -> final snapshots) -------
        if "scalar_flux" in point_data.keys():
            flux_nT = point_data["scalar_flux"]  # (N, T)
            num_cells = flux_nT.shape[0]
            num_timesteps = flux_nT.shape[1] if flux_nT.ndim == 2 else 1
        else:
            flux_nT = None
            num_cells = mesh.points.shape[0]
            num_timesteps = 1

        if not load_flux:
            scalar_flux = torch.zeros((1, num_cells), dtype=torch.float32)
            sim_times_t = None
        else:
            if flux_nT is None:
                raise KeyError(f"scalar_flux missing from {filepath}")
            full = flux_nT.transpose(0, 1).contiguous().to(torch.float32)  # (T, N)
            resolved = [0] if num_timesteps == 1 else [0, num_timesteps - 1]
            scalar_flux = full[resolved].contiguous()
            if (
                load_sim_times
                and "sim_times" in global_data.keys()
                and global_data["sim_times"].numel() > 0
            ):
                sim_t = global_data["sim_times"].to(torch.float32)
                sim_times_t = sim_t[resolved].contiguous()
            else:
                sim_times_t = None

        # ------- static-arrays cache lookup -------
        with self._cache_lock:
            cache_hit = self.cache_static_arrays and cache_key in self._static_cache
            cached_entry = None
            if cache_hit:
                self._cache_hits += 1
                self._static_cache.move_to_end(cache_key)
                cached_entry = dict(self._static_cache[cache_key])

        td = TensorDict({}, batch_size=[])
        td["scalar_flux"] = scalar_flux
        if sim_times_t is not None:
            td["sim_times"] = sim_times_t

        if cache_hit:
            for key, tensor in cached_entry.items():
                td[key] = tensor
        else:
            self._cache_misses += 1
            td["coordinates"] = mesh.points.to(torch.float32).contiguous()
            if "cell_areas" in point_data.keys():
                td["cell_areas"] = (
                    point_data["cell_areas"].to(torch.float32).contiguous()
                )

            if load_material_properties and "material_properties" in point_data.keys():
                td["material_properties"] = (
                    point_data["material_properties"].to(torch.int32).contiguous()
                )
            elif (
                load_material_properties
                and "material_properties" not in point_data.keys()
            ):
                warnings.warn(f"Material properties not found in {filename}.")

            if load_geometric_features and "geometric_features" in point_data.keys():
                td["geometric_features"] = (
                    point_data["geometric_features"].to(torch.float32).contiguous()
                )

            if load_sigma_fields:
                for key in ("sigma_t", "sigma_s", "sigma_a", "Q"):
                    if key in point_data.keys():
                        td[key] = point_data[key].to(torch.float32).contiguous()

            if self.cache_static_arrays:
                self._maybe_cache_entry(cache_key, td)

        sidecar = self._read_sidecar(filename)
        attrs = dict(sidecar.get("raw_attrs", {}))
        td.set_non_tensor("metadata", attrs)
        return td

    def _maybe_cache_entry(
        self,
        cache_key: Tuple[str, bool, bool, bool],
        td: TensorDict,
    ) -> None:
        with self._cache_lock:
            if (
                self.max_cache_size > 0
                and len(self._static_cache) >= self.max_cache_size
                and cache_key not in self._static_cache
            ):
                evicted = next(iter(self._static_cache))
                del self._static_cache[evicted]
                self._cache_evictions += 1

            entry: Dict[str, torch.Tensor] = {}
            for key in (
                "coordinates",
                "cell_areas",
                "material_properties",
                "geometric_features",
                "sigma_t",
                "sigma_s",
                "sigma_a",
                "Q",
            ):
                if key in td:
                    entry[key] = td[key]
            self._static_cache[cache_key] = entry

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_metadata(self, filename: str) -> Dict:
        """Return metadata (sidecar attrs + shape facts) without a full load."""
        cached = self._metadata_cache.get(filename)
        if cached is not None:
            return cached

        filepath = self.data_path / filename
        mesh = Mesh.load(str(filepath))
        point_data = mesh.point_data
        global_data = mesh.global_data

        sidecar = self._read_sidecar(filename)
        metadata: Dict = dict(sidecar.get("raw_attrs", {}))

        if "scalar_flux" in point_data.keys():
            flux_shape = point_data["scalar_flux"].shape  # (N, T)
            metadata["num_cells"] = int(flux_shape[0])
            metadata["num_timesteps"] = int(flux_shape[1]) if len(flux_shape) > 1 else 1
        else:
            metadata["num_cells"] = int(mesh.points.shape[0])
            metadata["num_timesteps"] = 1

        metadata["has_geometric_features"] = "geometric_features" in point_data.keys()
        metadata["has_material_properties"] = "material_properties" in point_data.keys()
        has_sim_times = (
            "sim_times" in global_data.keys() and global_data["sim_times"].numel() > 0
        )
        metadata["has_sim_times"] = has_sim_times
        if has_sim_times:
            try:
                metadata["max_sim_time"] = float(global_data["sim_times"][-1].item())
            except Exception as exc:  # pragma: no cover — defensive
                raise ValueError(
                    f"Failed to read sim_times tail from {filename}"
                ) from exc

        self._metadata_cache[filename] = metadata
        return metadata


# =========================================================================
# PhysicsNeMo Dataset
# =========================================================================
#
# ``RTEBaseDataset`` extends :class:`physicsnemo.datapipes.Dataset` so the
# example plugs into ``physicsnemo.datapipes.DataLoader`` (CUDA-stream-based,
# single-process, no fork). The class still owns the file-split logic and
# the in-memory flux cache; everything device-transfer / thread-prefetch
# related is inherited from the base class.


class RTEBaseDataset(PhysicsNeMoDataset):
    """File-indexed steady-state dataset over a directory of mesh stores.

    Wraps :class:`MeshDataReader` and produces ``(TensorDict, metadata)``
    tuples per the :class:`physicsnemo.datapipes.Dataset` contract. The
    metadata dict carries the source sidecar attrs plus ``filename``,
    ``max_timestep``, ``max_sim_time`` and the resolved ``sim_time`` so the
    rest of the pipeline can read them without unpacking ``NonTensorData``.

    The TensorDict still carries the per-sample tensor fields the reader
    returned (``coordinates``, ``cell_areas``, ``scalar_flux``, etc.).
    Transforms run on it in order; the trailing model adapter (e.g.
    :class:`TransolverAdapter`) is wired in by the caller via the
    ``transforms`` arg.
    """

    def __init__(
        self,
        data_path: Path | str,
        case_type: Optional[str] = None,
        phase: str = "train",
        split_file: Optional[Path | str] = None,
        seed: Optional[int] = None,
        load_material_properties: bool = True,
        load_geometric_features: bool = True,
        load_sigma_fields: bool = True,
        cache_static_arrays: bool = True,
        max_cache_size: int = 200,
        transforms: Optional[Transform | Sequence[Transform]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.data_path = Path(data_path)
        self.case_type = case_type
        self.phase = phase
        self.split_file = Path(split_file) if split_file else None
        self.seed = seed
        self.load_material_properties = load_material_properties
        self.load_geometric_features = load_geometric_features
        self.load_sigma_fields = load_sigma_fields

        reader = MeshDataReader(
            data_path,
            case_type,
            cache_static_arrays=cache_static_arrays,
            max_cache_size=max_cache_size,
        )

        if self.split_file is None:
            raise ValueError(
                "split_file is required. RTE datasets must use explicit "
                "train/val/test splits from a JSON split file."
            )
        self.filenames = self._load_split_from_file()

        if not self.filenames:
            raise ValueError(f"No files in {phase} split")

        # In-memory cache for flux data (populated by preload_to_memory when
        # enabled). Values are ``dict`` mirrors of the cached tensor entries.
        self._memory_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        super().__init__(reader=reader, transforms=transforms, device=device)

    # ------------------------------------------------------------------
    # Split machinery
    # ------------------------------------------------------------------

    def _load_split_from_file(self) -> List[str]:
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        with open(self.split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        if "splits" not in split_data:
            raise ValueError("Invalid split file format: missing 'splits' key")
        if self.phase not in split_data["splits"]:
            raise ValueError(
                f"Phase '{self.phase}' not found in split file. "
                f"Available: {list(split_data['splits'].keys())}"
            )
        filenames = split_data["splits"][self.phase]
        # Split files may list basenames with or without a ``.mesh`` suffix.
        # Normalize to always point at a mesh store.
        normalized: List[str] = []
        for f in filenames:
            base = f[: -len(".mesh")] if f.endswith(".mesh") else f
            normalized.append(base + ".mesh")
        return normalized

    def preload_to_memory(self, verbose: bool = True, num_workers: int = 8) -> dict:
        """Preload static arrays and first/final flux snapshots."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_files = len(self.filenames)
        self._memory_cache = {}

        if verbose:
            print(
                f"Preloading {num_files} files (first+final flux, {num_workers} workers)..."
            )

        start = time.perf_counter()

        def load_one(filename: str):
            td = self.reader.load(
                filename,
                load_material_properties=self.load_material_properties,
                load_geometric_features=self.load_geometric_features,
                load_sim_times=True,
                load_sigma_fields=self.load_sigma_fields,
            )
            entry: Dict[str, torch.Tensor] = {
                "scalar_flux": td["scalar_flux"].clone(),
            }
            if "sim_times" in td:
                entry["sim_times"] = td["sim_times"].clone()
            return filename, entry

        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_one, fn): fn for fn in self.filenames}
            for fut in as_completed(futures):
                filename, entry = fut.result()
                completed += 1
                self._memory_cache[filename] = entry
                if verbose and completed % 50 == 0:
                    elapsed = time.perf_counter() - start
                    rate = completed / elapsed
                    eta = (num_files - completed) / rate if rate > 0 else 0
                    print(
                        f"  Preloaded {completed}/{num_files} files "
                        f"({rate:.1f} files/s, ETA: {eta:.0f}s)"
                    )

        elapsed = time.perf_counter() - start
        cache_stats = self.reader.get_cache_stats()
        if verbose:
            print(
                f"Preload complete: {num_files} files in {elapsed:.1f}s "
                f"({num_files / elapsed:.1f} files/s)."
            )

        return {
            "num_files": num_files,
            "elapsed_seconds": elapsed,
            "cache_stats": cache_stats,
            "flux_cached": True,
        }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.filenames)

    def _read_sample(self, filename: str) -> TensorDict:
        """Read one sample, honoring the in-memory flux cache when populated."""
        if self._memory_cache is not None and filename in self._memory_cache:
            cached = self._memory_cache[filename]
            td = self.reader.load(
                filename,
                load_material_properties=self.load_material_properties,
                load_geometric_features=self.load_geometric_features,
                load_sim_times=False,
                load_sigma_fields=self.load_sigma_fields,
                load_flux=False,
            )
            td["scalar_flux"] = cached["scalar_flux"]
            if "sim_times" in cached:
                td["sim_times"] = cached["sim_times"]
        else:
            td = self.reader.load(
                filename,
                load_material_properties=self.load_material_properties,
                load_geometric_features=self.load_geometric_features,
                load_sim_times=True,
                load_sigma_fields=self.load_sigma_fields,
            )
        return td

    def _build_metadata(self, filename: str, td: TensorDict) -> Dict[str, Any]:
        """Per-sample metadata dict consumed by transforms + downstream code."""
        attrs = dict(td["metadata"]) if "metadata" in td else {}
        file_meta = self.reader.get_metadata(filename)
        attrs["max_timestep"] = file_meta["num_timesteps"] - 1
        attrs["max_sim_time"] = file_meta.get("max_sim_time")
        if "sim_times" in td and td["sim_times"].numel() > 0:
            attrs["sim_time"] = float(td["sim_times"][-1].item())
        else:
            attrs["sim_time"] = None
        attrs["filename"] = filename
        attrs["case_type"] = self.case_type
        return attrs

    def _read_one(self, idx: int) -> Tuple[TensorDict, Dict[str, Any]]:
        """Reader-side hook: load CPU TensorDict + metadata for index ``idx``.

        Routes through ``_read_sample`` (honoring the in-memory flux cache)
        and ``_build_metadata`` (filename / max_timestep / sim_time). The
        ``filename`` and ``metadata`` NonTensorData entries are kept on the
        TensorDict so transforms that read them (e.g. ``SteadyStateSampler``)
        keep working unchanged.
        """
        filename = self.filenames[idx]
        td = self._read_sample(filename)
        metadata = self._build_metadata(filename, td)
        td.set_non_tensor("filename", filename)
        td.set_non_tensor("metadata", metadata)
        return td, metadata

    def _load(self, idx: int) -> Tuple[TensorDict, Dict[str, Any]]:
        """Synchronous load: ``_read_one`` -> device -> transforms."""
        td, metadata = self._read_one(idx)
        if self.target_device is not None:
            td = td.to(self.target_device, non_blocking=True)
        if self.transforms is not None:
            td = self.transforms(td)
        return td, metadata

    def _load_and_transform(self, index, stream=None):
        """Stream-aware variant of ``_load`` used by the prefetch path."""
        from physicsnemo.datapipes.protocols import _PrefetchResult

        result = _PrefetchResult(index=index)
        try:
            td, metadata = self._read_one(index)

            if self.target_device is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        td = td.to(self.target_device, non_blocking=True)
                else:
                    td = td.to(self.target_device, non_blocking=True)

            if self.transforms is not None:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        td = self.transforms(td)
                    result.event = torch.cuda.Event()
                    result.event.record(stream)
                else:
                    td = self.transforms(td)

            result.data = td
            result.metadata = metadata
        except Exception as exc:  # pragma: no cover — surfaced via __getitem__
            result.error = exc
        return result

    def get_transformed_sample(self, idx: int) -> TensorDict:
        """Backwards-compat helper: return the transformed TensorDict only."""
        td, _ = self._load(idx)
        return td


# =========================================================================
# Stats loaders
# =========================================================================
#
# Non-breaking stats-file shim for PhysicsNeMo ``Normalize``. RTE's custom
# normalization transforms are replaced with
# ``physicsnemo.datapipes.transforms.Normalize``. The on-disk YAML stats files
# stay in their current RTE-specific schema; these helpers read them and
# produce the ``(means, stds)`` dicts PhysicsNeMo expects.


def load_flux_stats(path: Union[str, Path]) -> dict:
    """Read an RTE flux statistics YAML.

    Returns a plain dict with keys ``log_flux_mean``, ``log_flux_std``,
    ``clip_threshold``. Raises if any required key is missing.
    """
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Flux statistics file not found: {stats_path}")
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
    for key in ("log_flux_mean", "log_flux_std", "clip_threshold"):
        if key not in stats:
            raise ValueError(f"Flux statistics file missing required key: {key}")
    return stats


def flux_normalize_kwargs(
    stats: Mapping,
    field: str = "scalar_flux",
) -> dict:
    """Build ``Normalize`` kwargs for the log-clipped flux field.

    Example:
        stats = load_flux_stats(path)
        Normalize(**flux_normalize_kwargs(stats))
    """
    return {
        "input_keys": [field],
        "method": "mean_std",
        "means": {field: float(stats["log_flux_mean"])},
        "stds": {field: float(stats["log_flux_std"])},
    }


def load_material_stats(path: Union[str, Path]) -> dict:
    """Read an RTE material statistics YAML.

    Returns the full per-property nested dict. Each of ``sigma_a``,
    ``sigma_s``, ``sigma_t``, ``Q`` must be present with ``mean``, ``std``,
    ``min``, ``max`` sub-keys.
    """
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Material statistics file not found: {stats_path}")
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)
    required = ("sigma_a", "sigma_s", "sigma_t", "Q")
    for key in required:
        if key not in stats:
            raise ValueError(
                f"Material statistics file missing required property: {key}"
            )
        for sub in ("mean", "std"):
            if sub not in stats[key]:
                raise ValueError(
                    f"Material statistics[{key!r}] missing required sub-key: {sub!r}"
                )
    return stats


def material_normalize_kwargs(
    stats: Mapping,
    field: str = "physical_properties",
    order: Sequence[str] = ("sigma_a", "sigma_s", "sigma_t", "Q"),
) -> dict:
    """Build ``Normalize`` kwargs for ``physical_properties`` as (N, 4).

    The 4 columns are normalized independently via broadcasting: a per-column
    ``torch.Tensor`` of shape ``(4,)`` is passed as the mean and the std,
    delegating the math to ``physicsnemo.datapipes.transforms.Normalize``.
    """
    means = torch.tensor([float(stats[k]["mean"]) for k in order], dtype=torch.float32)
    stds = torch.tensor([float(stats[k]["std"]) for k in order], dtype=torch.float32)
    return {
        "input_keys": [field],
        "method": "mean_std",
        "means": {field: means},
        "stds": {field: stds},
    }


def coord_bounds_for_case(case_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(bbox_min, bbox_max)`` as float32 tensors for a known case.

    Shared with ``loader._build_transforms``; encapsulates the per-case
    global domain bounds that were hardcoded in ``CoordinateNormalizer``.
    """
    # Sibling import deferred to call time to avoid a circular import at
    # module load (transforms.py is allowed to import from dataset.py if it
    # ever needs stats helpers, though currently it does not).
    from transforms import GLOBAL_DOMAIN_BOUNDS

    if case_type not in GLOBAL_DOMAIN_BOUNDS:
        raise ValueError(
            f"Unknown case_type '{case_type}'. "
            f"Expected one of: {list(GLOBAL_DOMAIN_BOUNDS.keys())}"
        )
    bounds = GLOBAL_DOMAIN_BOUNDS[case_type]
    return (
        torch.as_tensor(bounds["min"], dtype=torch.float32),
        torch.as_tensor(bounds["max"], dtype=torch.float32),
    )


def coord_translate_scale_params(
    case_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(center, half_extent)`` for ``Translate`` + ``Scale``.

    RTE's ``CoordinateNormalizer`` produced ``(x - bbox_min) * 2 / (bbox_max -
    bbox_min) - 1``. Equivalently: subtract the bbox center, then divide by the
    bbox half-extent. This helper returns the two tensors in that form so the
    caller can wire them straight into
    ``Translate(center_key_or_value=center, subtract=True)`` followed by
    ``Scale(scale=half_extent, divide=True)``.
    """
    bbox_min, bbox_max = coord_bounds_for_case(case_type)
    center = 0.5 * (bbox_min + bbox_max)
    half_extent = 0.5 * (bbox_max - bbox_min)
    return center, half_extent
