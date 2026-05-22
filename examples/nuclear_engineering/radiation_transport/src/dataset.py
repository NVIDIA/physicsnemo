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

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch
import yaml
from physicsnemo.datapipes.dataset import Dataset as PhysicsNeMoDataset
from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform
from physicsnemo.mesh import Mesh
from tensordict import TensorDict


@register("RTEMeshReader")
class MeshDataReader(Reader):
    """Filename-indexed reader over a directory of RTE Mesh memmap stores.

    The ``TensorDict`` returned by ``load(filename)`` carries the tensor
    fields RTE training and inference rely on. The on-disk format is the
    PhysicsNeMo ``Mesh`` memmap layout (``<name>.pmsh/`` + ``<name>.attrs.json``
    sidecar).

    Example:
        >>> reader = MeshDataReader(
        ...     "/path/to/mesh_stores/lattice",
        ...     filenames=["lattice_abs10.0_scatter0.1_p0.015_q6.pmsh"],
        ... )
        >>> td = reader.load(reader.get_filenames()[0])
        >>> print(td["coordinates"].shape)  # (N, 2)
    """

    def __init__(
        self,
        data_path: Path | str,
        filenames: Sequence[str],
        case_type: Optional[str] = None,
        cache_static_arrays: bool = True,
    ):
        super().__init__(pin_memory=False, include_index_in_metadata=False)

        self.data_path = Path(data_path)
        self.case_type = case_type
        self.cache_static_arrays = cache_static_arrays

        # Plain dict cache of static-only fields keyed by filename. Mesh
        # stores are small enough that the full train+val split fits in RAM
        # without eviction.
        self._static_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        self._metadata_cache: Dict[str, Dict] = {}

        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")
        if not self.data_path.is_dir():
            raise ValueError(f"Data path {self.data_path} is not a directory")

        # ``filenames`` is required for train/val/test list
        # (typically derived from a split JSON) so that
        # ``Reader.__getitem__(idx)`` maps to a stable, intended file.
        self._filenames: List[str] = list(filenames)

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

    def get_filenames(self) -> List[str]:
        """Return a copy of the filenames the reader was constructed with."""
        return list(self._filenames)

    def _sidecar_path(self, filename: str) -> Path:
        # ``<name>.pmsh`` -> ``<name>.attrs.json``
        stem = filename[: -len(".pmsh")] if filename.endswith(".pmsh") else filename
        return self.data_path / f"{stem}.attrs.json"

    def _read_sidecar(self, filename: str) -> Dict:
        sidecar = self._sidecar_path(filename)
        if not sidecar.exists():
            return {}
        with open(sidecar, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self, filename: str) -> TensorDict:
        """Load a Mesh memmap store into a ``TensorDict``.

        Reads cell-primary fields from ``mesh.cell_data`` and derives
        ``coordinates`` and ``cell_areas`` from the mesh topology. Returned
        tensor fields: ``coordinates``, ``cell_areas``, ``scalar_flux``,
        ``sim_times``, ``material_properties``, ``sigma_a/s/t``, ``Q``,
        plus the eight hohlraum geometry parameters (``ulr, llr, urr, lrr,
        hlr, hrr, cx, cy``) when present on the store (hohlraum only).
        """
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh store {filepath} not found")

        mesh = Mesh.load(str(filepath))
        cell_data = mesh.cell_data
        global_data = mesh.global_data

        # Flux + timesteps (first -> final-time snapshots from the curated
        # time series). ``cell_data['scalar_flux']`` is ``(n_cells, T)``.
        if "scalar_flux" not in cell_data.keys():
            raise KeyError(f"cell_data['scalar_flux'] missing from {filepath}")
        flux_nT = cell_data["scalar_flux"]
        num_timesteps = flux_nT.shape[1] if flux_nT.ndim == 2 else 1
        full = flux_nT.transpose(0, 1).contiguous().to(torch.float32)  # (T, n_cells)
        resolved = [0] if num_timesteps == 1 else [0, num_timesteps - 1]

        td = TensorDict({}, batch_size=[])
        td["scalar_flux"] = full[resolved].contiguous()
        if "sim_time" in global_data.keys() and global_data["sim_time"].numel() > 0:
            td["sim_times"] = (
                global_data["sim_time"].to(torch.float32)[resolved].contiguous()
            )

        if self.cache_static_arrays and filename in self._static_cache:
            for key, tensor in self._static_cache[filename].items():
                td[key] = tensor
        else:
            # Coordinates and cell areas come from the topology (Mesh
            # properties) so the cell-primary fields share the same (n_cells,)
            # indexing.
            td["coordinates"] = mesh.cell_centroids.to(torch.float32).contiguous()
            td["cell_areas"] = mesh.cell_areas.to(torch.float32).contiguous()
            if "material_id" not in cell_data.keys():
                raise KeyError(f"cell_data['material_id'] missing from {filepath}")
            td["material_properties"] = (
                cell_data["material_id"].to(torch.int32).contiguous()
            )
            for key in ("sigma_t", "sigma_s", "sigma_a", "Q"):
                if key not in cell_data.keys():
                    raise KeyError(f"cell_data['{key}'] missing from {filepath}")
                td[key] = cell_data[key].to(torch.float32).contiguous()

            # Hohlraum geometry parameters: eight 0-D float32 tensors in
            # ``mesh.global_data``.
            for key in ("ulr", "llr", "urr", "lrr", "hlr", "hrr", "cx", "cy"):
                if key in global_data.keys():
                    td[key] = global_data[key].to(torch.float32).contiguous()

            if self.cache_static_arrays:
                cached_keys = (
                    "coordinates",
                    "cell_areas",
                    "material_properties",
                    "sigma_t",
                    "sigma_s",
                    "sigma_a",
                    "Q",
                    "ulr",
                    "llr",
                    "urr",
                    "lrr",
                    "hlr",
                    "hrr",
                    "cx",
                    "cy",
                )
                self._static_cache[filename] = {
                    k: td[k] for k in cached_keys if k in td
                }

        return td

    def get_metadata(self, filename: str) -> Dict:
        """Return metadata (sidecar attrs + shape facts) without a full load."""
        cached = self._metadata_cache.get(filename)
        if cached is not None:
            return cached

        filepath = self.data_path / filename
        mesh = Mesh.load(str(filepath))
        cell_data = mesh.cell_data
        global_data = mesh.global_data

        sidecar = self._read_sidecar(filename)
        metadata: Dict = {k: v for k, v in sidecar.items() if k != "missing_fields"}

        if "scalar_flux" not in cell_data.keys():
            raise KeyError(f"cell_data['scalar_flux'] missing from {filepath}")
        flux_shape = cell_data["scalar_flux"].shape  # (n_cells, T)
        metadata["num_cells"] = int(flux_shape[0])
        metadata["num_timesteps"] = int(flux_shape[1]) if len(flux_shape) > 1 else 1

        metadata["has_material_properties"] = "material_id" in cell_data.keys()
        has_sim_times = (
            "sim_time" in global_data.keys() and global_data["sim_time"].numel() > 0
        )
        metadata["has_sim_times"] = has_sim_times
        if has_sim_times:
            metadata["max_sim_time"] = float(global_data["sim_time"][-1].item())

        self._metadata_cache[filename] = metadata
        return metadata


class RTEBaseDataset(PhysicsNeMoDataset):
    """File-indexed final-time dataset over a directory of mesh stores.

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
        cache_static_arrays: bool = True,
        transforms: Optional[Transform | Sequence[Transform]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.data_path = Path(data_path)
        self.case_type = case_type
        self.phase = phase
        self.split_file = Path(split_file) if split_file else None
        self.seed = seed

        if self.split_file is None:
            raise ValueError(
                "split_file is required. RTE datasets must use explicit "
                "train/val/test splits from a JSON split file."
            )
        self.filenames = self._load_split_from_file()

        if not self.filenames:
            raise ValueError(f"No files in {phase} split")

        # Hand the split list to the reader so its int-indexed
        # ``__getitem__`` (called by ``Dataset._load``) resolves to the
        # split's files.
        reader = MeshDataReader(
            data_path=data_path,
            filenames=self.filenames,
            case_type=case_type,
            cache_static_arrays=cache_static_arrays,
        )

        super().__init__(reader=reader, transforms=transforms, device=device)

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
        # Split files may list basenames with or without a ``.pmsh`` suffix.
        # Normalize to always point at a mesh store.
        normalized: List[str] = []
        for f in filenames:
            base = f[: -len(".pmsh")] if f.endswith(".pmsh") else f
            normalized.append(base + ".pmsh")
        return normalized


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
