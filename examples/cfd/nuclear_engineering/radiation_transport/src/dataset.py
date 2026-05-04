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

"""RTE data-source layer: zarr reader, PyTorch Dataset, and stats loaders.

This module is the bottom of the data dependency tree. It provides the
low-level Zarr access (``ZarrDataReader``), a thin file-indexed
``Dataset`` wrapper (``RTEBaseDataset``), and helpers for reading the
RTE-specific YAML statistics files into PhysicsNeMo ``Normalize`` kwargs.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
import zarr
from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register
from tensordict import TensorDict
from torch.utils.data import Dataset


# =========================================================================
# Zarr reader
# =========================================================================
#
# Low-level reader for RTE simulation data stored in zarr. Inherits from
# ``physicsnemo.datapipes.readers.base.Reader``; returns ``TensorDict`` from
# ``load()``. The TensorDict carries both the tensor fields and non-tensor
# metadata (``metadata``, ``filename``) via ``NonTensorData`` entries.
#
# RTE-specific kwargs (``load_flux``, optional field loading, etc.) live on the
# filename-indexed ``load(filename, ...)`` entry. The int-indexed
# ``_load_sample(index)`` required by the PhysicsNeMo ``Reader`` contract uses
# defaults.


_TENSOR_FIELD_NAMES = (
    "coordinates",
    "cell_areas",
    "scalar_flux",
    "timesteps",
    "sim_times",
    "material_properties",
    "geometric_features",
    "sigma_t",
    "sigma_s",
    "sigma_a",
    "Q",
)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    """Zero-copy when possible; always returns a CPU ``torch.Tensor``."""
    return torch.from_numpy(np.ascontiguousarray(array))


@register("RTEZarrReader")
class ZarrDataReader(Reader):
    """Filename-indexed reader over a directory of RTE zarr stores.

    Inherits from ``physicsnemo.datapipes.readers.base.Reader`` so the reader
    plugs into any PhysicsNeMo-native pipeline via ``__getitem__(int)`` →
    ``(TensorDict, metadata_dict)``. RTE pipelines still reach it via
    ``load(filename, **kwargs)`` for the steady-state loader controls the
    training data loaders rely on.

    Example:
        >>> reader = ZarrDataReader("/path/to/zarr_stores/lattice")
        >>> filenames = reader.get_filenames()
        >>> td = reader.load(filenames[0])
        >>> print(td["coordinates"].shape)  # (N, 3)

    The LRU cache is retained verbatim: it stores tensor fields keyed by
    filename and evicts in insertion order when ``max_cache_size`` is hit.
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

        # LRU cache for static arrays keyed by filename; values are dicts of
        # ``torch.Tensor``.
        self._static_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._cache_lock = threading.Lock()

        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
        if not self.data_path.is_dir():
            raise ValueError(f"Data path {self.data_path} is not a directory")

        # Discover files once at construction time so ``_load_sample`` has a
        # stable filename→int mapping for the PhysicsNeMo ``Reader`` protocol.
        self._filenames: List[str] = self._scan_filenames()

    # ------------------------------------------------------------------
    # PhysicsNeMo ``Reader`` contract
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._filenames)

    def _load_sample(self, index: int) -> Dict[str, torch.Tensor]:
        """Int-indexed load using defaults (load everything, no slicing)."""
        td = self.load(self._filenames[index])
        return {key: td[key] for key in td.keys() if isinstance(td[key], torch.Tensor)}

    def _get_sample_metadata(self, index: int) -> Dict:
        """Return per-sample metadata dict for ``(TensorDict, metadata)`` tuple."""
        filename = self._filenames[index]
        meta = self.get_metadata(filename)
        meta["filename"] = filename
        return meta

    # ------------------------------------------------------------------
    # Filename discovery + caching (unchanged behavior)
    # ------------------------------------------------------------------

    def _scan_filenames(self) -> List[str]:
        filenames = []
        for item in self.data_path.iterdir():
            if item.suffix == ".zarr" or (
                item.is_dir() and item.name.endswith(".zarr")
            ):
                if self.case_type is None or item.name.startswith(self.case_type):
                    filenames.append(item.name)
        return sorted(filenames)

    def get_filenames(self) -> List[str]:
        """Return a fresh list of discovered zarr store names."""
        return list(self._filenames)

    def get_cache_stats(self) -> Dict[str, float]:
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
        with self._cache_lock:
            self._static_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_evictions = 0

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
        """Load a zarr store into a ``TensorDict``.

        Tensor fields (``coordinates``, ``cell_areas``, ``scalar_flux``,
        ``timesteps`` and the optional ``sim_times`` / ``material_properties``
        / ``geometric_features`` / ``sigma_*`` / ``Q``) are stored as
        ``torch.Tensor`` entries. The zarr store's ``.attrs`` dict is stored
        as ``NonTensorData`` under the ``metadata`` key.

        ``load_flux=False`` returns a placeholder ``scalar_flux`` of shape
        ``(1, N)`` — the caller is expected to overwrite it from a memory
        cache. The sentinel matches the pre-Phase-I behavior.
        """
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Zarr store {filepath} not found")

        z = zarr.open(str(filepath), mode="r")

        # ------- flux + timesteps (steady-state first -> final snapshots) -------
        if not load_flux:
            flux_shape = z["scalar_flux"].shape
            num_cells = flux_shape[-1]
            scalar_flux = np.zeros((1, num_cells), dtype=np.float32)
            timesteps_array = np.array([0])
            sim_times = None
        else:
            flux_array = z["scalar_flux"]
            if len(flux_array.shape) == 1:
                scalar_flux = np.array(flux_array, dtype=np.float32)[None, :]
                timesteps_array = np.array([0])
                resolved = [0]
            else:
                num_timesteps = flux_array.shape[0]
                resolved = [0] if num_timesteps == 1 else [0, num_timesteps - 1]
                scalar_flux = np.stack(
                    [
                        np.array(flux_array[idx], dtype=np.float32)
                        for idx in resolved
                    ],
                    axis=0,
                )
                timesteps_array = np.array([z["timesteps"][idx] for idx in resolved])
            if load_sim_times and "sim_times" in z:
                sim_times = np.array(
                    [z["sim_times"][idx] for idx in resolved], dtype=np.float32
                )
            else:
                sim_times = None

        # ------- static-arrays cache lookup -------
        with self._cache_lock:
            cache_hit = self.cache_static_arrays and filename in self._static_cache
            cached_entry = None
            if cache_hit:
                self._cache_hits += 1
                self._static_cache.move_to_end(filename)
                cached_entry = dict(self._static_cache[filename])  # shallow copy

        td = TensorDict({}, batch_size=[])
        td["scalar_flux"] = _to_tensor(scalar_flux)
        td["timesteps"] = _to_tensor(np.asarray(timesteps_array))
        if sim_times is not None:
            td["sim_times"] = _to_tensor(np.asarray(sim_times))

        if cache_hit:
            # Reuse cached static tensors; copy references, not data.
            for key, tensor in cached_entry.items():
                td[key] = tensor
        else:
            self._cache_misses += 1
            cell_centers = np.array(z["cell_centers"], dtype=np.float32)
            cell_areas = np.array(z["cell_areas"], dtype=np.float32)
            td["coordinates"] = _to_tensor(cell_centers)
            td["cell_areas"] = _to_tensor(cell_areas)

            if load_material_properties and "material_properties" in z:
                td["material_properties"] = _to_tensor(
                    np.array(z["material_properties"], dtype=np.int32)
                )
            elif load_material_properties and "material_properties" not in z:
                warnings.warn(f"Material properties not found in {filename}.")

            if load_geometric_features and "geometric_features" in z:
                td["geometric_features"] = _to_tensor(
                    np.array(z["geometric_features"], dtype=np.float32)
                )

            if load_sigma_fields:
                for key in ("sigma_t", "sigma_s", "sigma_a", "Q"):
                    if key in z:
                        td[key] = _to_tensor(np.array(z[key], dtype=np.float32))

            if self.cache_static_arrays:
                self._maybe_cache_entry(filename, td)

        # Non-tensor metadata ride as NonTensorData so transforms and adapters
        # that access ``td["metadata"]`` keep working unchanged.
        attrs = dict(z.attrs) if hasattr(z, "attrs") else {}
        td.set_non_tensor("metadata", attrs)
        return td

    def _maybe_cache_entry(self, filename: str, td: TensorDict) -> None:
        """LRU-cache the static tensor fields of ``td`` under ``filename``."""
        with self._cache_lock:
            if (
                self.max_cache_size > 0
                and len(self._static_cache) >= self.max_cache_size
                and filename not in self._static_cache
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
            self._static_cache[filename] = entry

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_metadata(self, filename: str) -> Dict:
        """Return metadata without loading full sample data."""
        filepath = self.data_path / filename
        z = zarr.open(str(filepath), mode="r")

        metadata = dict(z.attrs) if hasattr(z, "attrs") else {}
        flux_shape = z["scalar_flux"].shape
        metadata["num_timesteps"] = flux_shape[0] if len(flux_shape) > 1 else 1
        metadata["num_cells"] = flux_shape[-1]
        metadata["has_geometric_features"] = "geometric_features" in z
        metadata["has_material_properties"] = "material_properties" in z
        metadata["has_sim_times"] = "sim_times" in z
        if "sim_times" in z:
            try:
                metadata["max_sim_time"] = float(z["sim_times"][-1])
            except Exception as exc:  # pragma: no cover — defensive
                raise ValueError(
                    f"Failed to read sim_times tail from {filename}"
                ) from exc
        return metadata

    def validate(self, filename: str) -> bool:
        """Assert the zarr store has the required top-level arrays."""
        filepath = self.data_path / filename
        z = zarr.open(str(filepath), mode="r")

        required = ["cell_centers", "cell_areas", "scalar_flux", "timesteps"]
        for key in required:
            if key not in z:
                raise ValueError(f"Zarr store missing required key: {key}")

        nc_centers = z["cell_centers"].shape[0]
        nc_areas = z["cell_areas"].shape[0]
        nc_flux = z["scalar_flux"].shape[-1]
        if nc_centers != nc_flux:
            raise ValueError(
                f"Shape mismatch: cell_centers has {nc_centers} cells, "
                f"scalar_flux has {nc_flux}"
            )
        if nc_areas != nc_flux:
            raise ValueError(
                f"Shape mismatch: cell_areas has {nc_areas} cells, "
                f"scalar_flux has {nc_flux}"
            )
        return True


# =========================================================================
# PyTorch Dataset
# =========================================================================
#
# Minimal PyTorch ``Dataset`` that wraps ``ZarrDataReader`` and produces
# per-sample ``TensorDict`` outputs. The reader returns TensorDicts directly;
# this layer only glues together file selection, the preload cache, and
# per-sample metadata enrichment.


class RTEBaseDataset(Dataset):
    """File-indexed steady-state dataset over a directory of zarr stores.

    Output of ``__getitem__`` is a ``TensorDict`` with the tensor fields the
    reader returned, plus ``filename`` (``NonTensorData``), an updated
    ``metadata`` ``NonTensorData`` entry (``max_timestep`` / ``max_sim_time``).
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
    ):
        self.data_path = Path(data_path)
        self.case_type = case_type
        self.phase = phase
        self.split_file = Path(split_file) if split_file else None
        self.seed = seed
        self.load_material_properties = load_material_properties
        self.load_geometric_features = load_geometric_features
        self.load_sigma_fields = load_sigma_fields

        self.reader = ZarrDataReader(
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
        return [f if f.endswith(".zarr") else f + ".zarr" for f in filenames]

    def preload_to_memory(self, verbose: bool = True, num_workers: int = 8) -> dict:
        """Preload static arrays and first/final flux snapshots."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_files = len(self.filenames)
        self._memory_cache = {}

        if verbose:
            print(f"\nPreloading {num_files} files with steady-state flux...")
            print("  Loading ONLY first and final snapshots (2 per file)")
            print(f"  Parallel I/O workers: {num_workers}")

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
                "timesteps": td["timesteps"].clone(),
            }
            if "sim_times" in td:
                entry["sim_times"] = td["sim_times"].clone()
            return filename, td, entry

        completed = 0
        first_logged = False
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_one, fn): fn for fn in self.filenames}
            for fut in as_completed(futures):
                filename, td, entry = fut.result()
                completed += 1
                self._memory_cache[filename] = entry
                if verbose and not first_logged:
                    n_cells = td["coordinates"].shape[0]
                    print(f"\n  First file diagnostics ({filename}):")
                    print(f"    scalar_flux shape: {tuple(td['scalar_flux'].shape)}")
                    print(f"    num_cells: {n_cells:,}")
                    print(f"    sigma_t loaded: {'sigma_t' in td}")
                    print(f"    sigma_s loaded: {'sigma_s' in td}")
                    print("")
                    first_logged = True
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
            print("\nPreload complete!")
            print(f"  Files loaded: {num_files}")
            print(f"  Time: {elapsed:.1f}s ({num_files/elapsed:.1f} files/s)")
            print(f"  Static arrays cache: {cache_stats['cache_size']} files")
            print(f"  Cache hits: {cache_stats['cache_hits']}")
            print(f"  Cache misses: {cache_stats['cache_misses']}")
            flux_mem = sum(
                cached["scalar_flux"].element_size() * cached["scalar_flux"].numel()
                + cached["timesteps"].element_size() * cached["timesteps"].numel()
                + (
                    cached["sim_times"].element_size() * cached["sim_times"].numel()
                    if "sim_times" in cached
                    else 0
                )
                for cached in self._memory_cache.values()
            )
            print(
                f"  Flux cache: {len(self._memory_cache)} simulations "
                f"({flux_mem / 1024**2:.1f} MB)"
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

    def __getitem__(self, idx: int) -> TensorDict:
        filename = self.filenames[idx]

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
            td["timesteps"] = cached.get("timesteps", td["timesteps"])
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

        # Enrich metadata with the per-sample info transforms rely on.
        # ``td["metadata"]`` is a NonTensorData dict of zarr attrs; extend it.
        attrs = dict(td["metadata"]) if "metadata" in td else {}
        file_meta = self.reader.get_metadata(filename)
        attrs["max_timestep"] = file_meta["num_timesteps"] - 1
        attrs["max_sim_time"] = file_meta.get("max_sim_time")
        if "sim_times" in td and td["sim_times"].numel() > 0:
            attrs["sim_time"] = float(td["sim_times"][-1].item())
        else:
            attrs["sim_time"] = None
        td.set_non_tensor("metadata", attrs)
        td.set_non_tensor("filename", filename)

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
    method: str = "mean_std",
) -> dict:
    """Build ``Normalize`` kwargs for ``physical_properties`` as (N, 4).

    The 4 columns are normalized independently via broadcasting: a per-column
    ``torch.Tensor`` of shape ``(4,)`` is passed as the mean and the std. This
    mirrors what the custom ``MaterialPropertyNormalizer`` did column-by-column,
    but delegates the math to ``physicsnemo.datapipes.transforms.Normalize``.
    """
    if method == "mean_std":
        means = torch.tensor(
            [float(stats[k]["mean"]) for k in order], dtype=torch.float32
        )
        stds = torch.tensor(
            [float(stats[k]["std"]) for k in order], dtype=torch.float32
        )
        return {
            "input_keys": [field],
            "method": "mean_std",
            "means": {field: means},
            "stds": {field: stds},
        }
    if method == "min_max":
        mins = torch.tensor(
            [float(stats[k]["min"]) for k in order], dtype=torch.float32
        )
        maxs = torch.tensor(
            [float(stats[k]["max"]) for k in order], dtype=torch.float32
        )
        return {
            "input_keys": [field],
            "method": "min_max",
            "mins": {field: mins},
            "maxs": {field: maxs},
        }
    raise ValueError(f"Unknown method: {method}. Expected 'mean_std' or 'min_max'.")


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
