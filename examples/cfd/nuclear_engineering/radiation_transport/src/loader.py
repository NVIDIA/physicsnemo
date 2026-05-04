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

"""Data plumbing: TransolverAdapter, collate, datapipe orchestration, DataLoader builder.

This module is the "wiring" layer of the RTE Transolver example. It composes
the ``RTEBaseDataset`` (data source) with a ``Compose`` of transforms and a
``TransolverAdapter`` to produce model-ready batches, and exposes a single
``build_dataloaders`` entry point used by the training and evaluation
scripts.

Sections:

* Adapter — ``ModelAdapter`` base + ``TransolverAdapter`` + ``_as_dict`` helper.
* Collation — ``collate_no_padding`` (batch_size=1 unsqueeze).
* Stats / kwargs translation — ``build_rte_dataset_kwargs``.
* Pipeline orchestration — ``RTEDataPipe`` + ``_build_transforms`` +
  ``_build_adapter`` + ``from_config`` + ``create_dataset``.
* Distributed preload barrier — file-marker rank-sequencing helpers.
* DataLoader builder — ``build_dataloaders`` + ``_make_loader`` +
  ``_log_material_sanity``.
"""

from __future__ import annotations

# =========================================================================
# Imports
# =========================================================================

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.distributed as torch_dist
from omegaconf import DictConfig
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms import Compose, Normalize, Scale, Translate
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from dataset import (
    RTEBaseDataset,
    coord_translate_scale_params,
    flux_normalize_kwargs,
    load_flux_stats,
    load_material_stats,
    material_normalize_kwargs,
)
from material import MaterialPropertyExtractor
from transforms import (
    TRANSFORM_REGISTRY,
    FourierFeatures,
    LoadGroundTruthQoI,
    NextStepSampler,
    RTEBackupCoords,
    RTEFluxLogClip,
    SpatialSampler,
    SteadyStateSampler,
    td_from_dict,
)


# Register the material transform into the local TRANSFORM_REGISTRY so
# config-driven lookups by name resolve correctly. ``material.py`` does not
# decorate ``MaterialPropertyExtractor`` with ``@_register_local`` (to avoid
# importing the registry plumbing from a sibling), so we wire it up here at
# loader-import time. This keeps the registry a single source of truth even
# for transforms that live outside ``transforms.py``.
TRANSFORM_REGISTRY.setdefault("RTEMaterialPropertyExtractor", MaterialPropertyExtractor)


# =========================================================================
# Adapter
# =========================================================================
#
# ``ModelAdapter`` is the abstract base class for converting a transformed
# RTE sample into a model-specific input dict. Only ``TransolverAdapter`` is
# shipped here (GenericAdapter and GeoTransolverAdapter were dropped as part
# of the upstream Transolver-only consolidation).


class ModelAdapter(ABC):
    """Abstract base class for model-specific data adapters.

    Adapters take a transformed sample (a ``TensorDict`` or plain dict with
    numpy arrays / torch tensors) and convert it to the format expected by a
    particular model (e.g. Transolver).
    """

    @abstractmethod
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert a sample to model-specific format."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _as_dict(sample: Union[TensorDict, Dict[str, Any]]) -> Dict[str, Any]:
    """Unwrap a ``TensorDict`` into a plain dict; pass through regular dicts.

    Tensor entries remain ``torch.Tensor`` references (no copy). NonTensorData
    entries are transparently unwrapped via bracket access.
    """
    if isinstance(sample, TensorDict):
        return {k: sample[k] for k in sample.keys()}
    return sample


@register("RTETransolverAdapter")
class TransolverAdapter(ModelAdapter):
    """Adapter for the Transolver model.

    Maps RTE data to Transolver's expected input format:

    * ``fx`` — spatial coordinates ``[x, y, z]`` (plus Fourier features when enabled).
    * ``embedding`` — material properties ``[sigma_a, sigma_s, sigma_t, Q]``
      (or just the first three when ``include_q_in_embedding=False``).
    * ``flux_target`` — target flux to predict.
    * ``time`` — normalized timestep (always present).

    Extra fields (``coordinates_unnormalized``, ``material_labels``,
    ``cell_areas``, ``sigma_t``, ``sigma_s``, ``sim_time``, ``ground_truth_qoi``,
    ``flux_normalization_stats``, ``geometry_params``) are passed through when
    present in the sample.
    """

    def __init__(
        self,
        add_batch_dim: bool = False,
        include_q_in_embedding: bool = True,
    ):
        self.add_batch_dim = add_batch_dim
        self.include_q_in_embedding = include_q_in_embedding

    def __call__(
        self, data: Union[TensorDict, Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Convert sample to Transolver format."""
        sample = _as_dict(data)
        result: Dict[str, Any] = {}

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.from_numpy(x).float()

        if "coordinates" in sample:
            coords = to_tensor(sample["coordinates"])
            if self.add_batch_dim:
                coords = coords.unsqueeze(0)
            result["fx"] = coords

        if "physical_properties" in sample:
            mat_props = to_tensor(sample["physical_properties"])
            if not self.include_q_in_embedding:
                mat_props = mat_props[..., :3]
            if self.add_batch_dim:
                mat_props = mat_props.unsqueeze(0)
            result["embedding"] = mat_props

        if "coordinates_unnormalized" in sample:
            coords_unnorm = to_tensor(sample["coordinates_unnormalized"])
            if self.add_batch_dim:
                coords_unnorm = coords_unnorm.unsqueeze(0)
            result["coordinates_unnormalized"] = coords_unnorm

        if (
            "material_properties" in sample
            and sample["material_properties"] is not None
        ):
            mat_labels = sample["material_properties"]
            if isinstance(mat_labels, np.ndarray):
                mat_labels = torch.from_numpy(mat_labels.astype(np.int64))
            elif isinstance(mat_labels, torch.Tensor):
                mat_labels = mat_labels.long()
            if self.add_batch_dim:
                mat_labels = mat_labels.unsqueeze(0)
            result["material_labels"] = mat_labels

        if "flux_target" in sample:
            flux_tgt = to_tensor(sample["flux_target"])
            if flux_tgt.ndim == 1:
                flux_tgt = flux_tgt.unsqueeze(-1)
            if self.add_batch_dim:
                flux_tgt = flux_tgt.unsqueeze(0)
            result["flux_target"] = flux_tgt

        metadata = sample.get("metadata", {}) or {}
        max_timestep = metadata.get("max_timestep") if isinstance(metadata, dict) else None
        max_sim_time = metadata.get("max_sim_time") if isinstance(metadata, dict) else None
        timestep_target = sample.get("timestep_target")

        if "sim_times" in sample and max_sim_time is not None:
            sim_times = sample["sim_times"]
            sim_time_target = (
                float(sim_times[-1].item())
                if isinstance(sim_times, torch.Tensor)
                else float(sim_times[-1])
            )
            time_normalized = sim_time_target / float(max_sim_time)
            time_tensor = torch.tensor([[time_normalized]], dtype=torch.float32)
            if not self.add_batch_dim:
                time_tensor = time_tensor.squeeze(0)
            result["time"] = time_tensor
        elif timestep_target is not None and max_timestep is not None:
            if float(max_timestep) == 0:
                time_normalized = 1.0
            else:
                time_normalized = float(timestep_target) / float(max_timestep)
            time_tensor = torch.tensor([[time_normalized]], dtype=torch.float32)
            if not self.add_batch_dim:
                time_tensor = time_tensor.squeeze(0)
            result["time"] = time_tensor
        else:
            time_tensor = torch.tensor([[0.0]], dtype=torch.float32)
            if not self.add_batch_dim:
                time_tensor = time_tensor.squeeze(0)
            result["time"] = time_tensor

        if "cell_areas" in sample:
            cell_areas = to_tensor(sample["cell_areas"])
            if self.add_batch_dim:
                cell_areas = cell_areas.unsqueeze(0)
            result["cell_areas"] = cell_areas

        if "sigma_t" in sample:
            sigma_t = to_tensor(sample["sigma_t"])
            if self.add_batch_dim:
                sigma_t = sigma_t.unsqueeze(0)
            result["sigma_t"] = sigma_t

        if "sigma_s" in sample:
            sigma_s = to_tensor(sample["sigma_s"])
            if self.add_batch_dim:
                sigma_s = sigma_s.unsqueeze(0)
            result["sigma_s"] = sigma_s

        if "sim_times" in sample:
            sim_times_arr = sample["sim_times"]
            if isinstance(sim_times_arr, torch.Tensor) and sim_times_arr.numel() > 0:
                sim_time = torch.tensor(
                    [float(sim_times_arr[-1].item())], dtype=torch.float32
                )
            elif hasattr(sim_times_arr, "__len__") and len(sim_times_arr) > 0:
                sim_time = torch.tensor([float(sim_times_arr[-1])], dtype=torch.float32)
            else:
                sim_time = torch.tensor([0.0], dtype=torch.float32)
            if self.add_batch_dim:
                sim_time = sim_time.unsqueeze(0)
            result["sim_time"] = sim_time

        if "ground_truth_qoi" in sample:
            qoi_value = sample["ground_truth_qoi"]
            if not isinstance(qoi_value, torch.Tensor):
                qoi_value = torch.tensor([qoi_value], dtype=torch.float32)
            else:
                qoi_value = qoi_value.float()
            if self.add_batch_dim:
                qoi_value = qoi_value.unsqueeze(0)
            result["ground_truth_qoi"] = qoi_value

        if "flux_normalization_stats" in sample:
            result["flux_normalization_stats"] = sample["flux_normalization_stats"]

        filename = sample.get("filename", "") or ""
        if filename and "hohlraum" in filename.lower():
            geometry_params = self._extract_geometry_params(filename)
            if geometry_params:
                result["geometry_params"] = geometry_params

        metadata_dict = {
            "timestep_input": sample.get("timestep_input"),
            "timestep_target": sample.get("timestep_target"),
            "max_timestep": max_timestep,
            "filename": sample.get("filename"),
            "case_type": (
                metadata.get("case_type") if isinstance(metadata, dict) else None
            ),
        }
        result["metadata"] = {k: v for k, v in metadata_dict.items() if v is not None}

        return result

    def _extract_geometry_params(self, filename: str) -> dict:
        """Extract hohlraum geometry parameters from the zarr filename."""
        filename = filename.replace(".zarr", "")
        parts = filename.split("_")
        geometry_params: Dict[str, float] = {}
        for part in parts:
            if part.startswith("ulr"):
                geometry_params["ulr"] = float(part[3:])
            elif part.startswith("llr"):
                geometry_params["llr"] = float(part[3:])
            elif part.startswith("urr"):
                geometry_params["urr"] = float(part[3:])
            elif part.startswith("lrr"):
                geometry_params["lrr"] = float(part[3:])
            elif part.startswith("hlr"):
                geometry_params["hlr"] = float(part[3:])
            elif part.startswith("hrr"):
                geometry_params["hrr"] = float(part[3:])
            elif part.startswith("cx"):
                geometry_params["cx"] = float(part[2:])
            elif part.startswith("cy"):
                geometry_params["cy"] = float(part[2:])
        return geometry_params

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"add_batch_dim={self.add_batch_dim}, "
            f"include_q_in_embedding={self.include_q_in_embedding})"
        )


# =========================================================================
# Collation
# =========================================================================
#
# Only ``collate_no_padding`` ships with the upstream example: all configs
# use ``batch_size=1`` with a fixed ``num_spatial_points`` from
# ``SpatialSampler``, so no padding is needed. The PyG-graph collator was
# dropped along with the MeshGraphNet adapter.


@register("RTECollateNoPadding")
def collate_no_padding(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch-size-1 collate: unsqueeze each tensor, pass non-tensors through.

    Asserts ``len(batch) == 1`` to keep us honest if someone flips the config
    without wiring a real multi-sample collator.
    """
    assert len(batch) == 1, "collate_no_padding requires batch_size=1"
    item = batch[0]
    result: Dict[str, Any] = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.unsqueeze(0)
        else:
            result[k] = v
    return result


# =========================================================================
# Stats / kwargs translation
# =========================================================================
#
# ``build_rte_dataset_kwargs`` translates a Hydra config into the kwargs
# ``create_dataset`` expects. Used by both training (phases ``train``/``val``)
# and evaluation (phase ``test``).


def build_rte_dataset_kwargs(
    cfg: DictConfig,
    *,
    adapter: str,
    num_spatial_points_key: str = "num_spatial_points",
    num_spatial_points_override: Optional[int] = None,
    split_file_override: Optional[str] = None,
    extra_kwargs: Optional[dict] = None,
) -> dict:
    """Translate a Hydra config into the kwargs ``create_dataset`` expects.

    Callers that run against a checkpoint's saved config (evaluation) can
    provide overrides for values the CLI wants to win over the config
    (``split_file``) or that diverge between current and checkpoint shapes
    (``num_spatial_points``).
    """
    data_cfg = cfg.data

    flux_stats_file = data_cfg.get("flux_normalization_stats_file")
    if flux_stats_file is None:
        raise ValueError(
            "data.flux_normalization_stats_file must be specified in config."
        )

    task = data_cfg.get("task")
    if task is None:
        raise ValueError("data.task must be specified in config.")

    flux_clip_threshold = data_cfg.get("flux_clip_threshold")
    if flux_clip_threshold is None:
        raise ValueError("data.flux_clip_threshold must be specified in config.")

    # num_spatial_points — override for eval when the checkpoint's model
    # config carries the authoritative value.
    if num_spatial_points_override is not None:
        num_spatial_points = num_spatial_points_override
    elif "." in num_spatial_points_key:
        parts = num_spatial_points_key.split(".")
        num_spatial_points = cfg
        for part in parts:
            num_spatial_points = num_spatial_points[part]
    else:
        num_spatial_points = cfg.model[num_spatial_points_key]

    case_cfg = cfg.get("case", {})
    split_file = (
        split_file_override
        if split_file_override
        else case_cfg.get("split_file") or data_cfg.get("split_file")
    )

    seed = data_cfg.get("seed", None)
    if seed is None and "train" in cfg:
        seed = cfg.train.get("seed", None)

    # Fourier features config
    use_fourier_features = data_cfg.get("use_fourier_features", False)
    fourier_num_frequencies = None
    fourier_coord_dims = None
    fourier_base_frequency = None
    if use_fourier_features:
        fourier_cfg = data_cfg.get("fourier_features")
        if fourier_cfg is None:
            raise ValueError(
                "use_fourier_features=True but data.fourier_features is missing."
            )
        fourier_num_frequencies = fourier_cfg.get("num_frequencies")
        fourier_coord_dims = fourier_cfg.get("coord_dims")
        fourier_base_frequency = fourier_cfg.get("base_frequency")
        if any(
            v is None
            for v in (
                fourier_num_frequencies,
                fourier_coord_dims,
                fourier_base_frequency,
            )
        ):
            raise ValueError(
                "fourier_features config must specify num_frequencies, "
                f"coord_dims, base_frequency. Got: {dict(fourier_cfg)}"
            )

    kwargs = {
        "data_path": cfg.case.data_path,
        "task": task,
        "num_spatial_points": num_spatial_points,
        "adapter": adapter,
        "flux_normalization_stats_file": flux_stats_file,
        "normalize_coordinates": data_cfg.get("normalize_coordinates", True),
        "flux_clip_threshold": flux_clip_threshold,
        "split_file": split_file,
        "train_split": data_cfg.get("train_split", 0.7),
        "val_split": data_cfg.get("val_split", 0.15),
        "seed": seed,
        "expand_timesteps": data_cfg.get("expand_timesteps", True),
        "temporal_stride": data_cfg.get("temporal_stride", 1),
        "load_ground_truth_qoi": data_cfg.get("load_ground_truth_qoi", False),
        "cache_static_arrays": data_cfg.get("cache_static_arrays", True),
        "max_cache_size": data_cfg.get("max_cache_size", 200),
        "include_q_in_embedding": cfg.model.get("include_q_in_embedding", True),
        "use_fourier_features": use_fourier_features,
        "fourier_num_frequencies": fourier_num_frequencies,
        "fourier_coord_dims": fourier_coord_dims,
        "fourier_base_frequency": fourier_base_frequency,
    }

    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return kwargs


# =========================================================================
# Pipeline orchestration
# =========================================================================
#
# ``RTEDataPipe`` composes ``RTEBaseDataset`` (data source) with a ``Compose``
# of transforms and a ``TransolverAdapter`` (model adapter). ``from_config``
# is the simple-configuration entry point; ``_build_transforms`` and
# ``_build_adapter`` are internal builders used by ``from_config``.
# ``create_dataset`` is the convenience wrapper used by ``build_dataloaders``.


@register("RTEDataPipe")
class RTEDataPipe(Dataset):
    """High-level composable datapipe for RTE data.

    Combines:

    * ``RTEBaseDataset`` (data source / file enumeration / timestep expansion)
    * ``Compose`` of transforms (preprocessing pipeline)
    * ``TransolverAdapter`` (model-specific tensor packaging)

    For the canonical training configuration, use ``RTEDataPipe.from_config``;
    for fully custom pipelines, instantiate directly with explicit
    ``transforms`` and ``adapter``.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        transforms: Optional[Compose] = None,
        adapter: Optional[Any] = None,
        case_type: Optional[Literal["lattice", "hohlraum"]] = None,
        phase: Literal["train", "val", "test"] = "train",
        split_file: Optional[Union[str, Path]] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        seed: Optional[int] = None,
        expand_timesteps: bool = True,
        temporal_stride: int = 1,
        cache_static_arrays: bool = True,
        max_cache_size: int = 200,
        task: Literal["next_step", "steady_state"] = "next_step",
    ):
        """Initialize the datapipe (see ``from_config`` for the simple path)."""
        self.base_dataset = RTEBaseDataset(
            data_path=data_path,
            case_type=case_type,
            phase=phase,
            split_file=split_file,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
            load_sigma_fields=True,  # load precomputed material properties for speed
            expand_timesteps=expand_timesteps,
            temporal_stride=temporal_stride,
            cache_static_arrays=cache_static_arrays,
            max_cache_size=max_cache_size,
            task=task,
        )

        self.transforms = transforms
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Any:
        """Get a sample from the dataset.

        ``base_dataset[idx]`` returns a ``TensorDict`` with tensor fields plus
        ``metadata`` / ``filename`` / ``_timestep_idx`` as ``NonTensorData``
        entries. Transforms consume and return ``TensorDict``; the adapter
        converts to the model-specific format.
        """
        td = self.base_dataset[idx]
        if not isinstance(td, TensorDict):
            # Defensive path for callers that still return a dict.
            td = td_from_dict(td)

        if self.transforms is not None:
            td = self.transforms(td)

        if self.adapter is not None:
            return self.adapter(td)
        return td

    @classmethod
    def from_config(
        cls,
        data_path: Union[str, Path],
        case_type: Optional[Literal["lattice", "hohlraum"]] = None,
        task: Literal["next_step", "steady_state"] = "next_step",
        adapter: Optional[Literal["transolver", None]] = "transolver",
        phase: Literal["train", "val", "test"] = "train",
        # Data processing options
        num_spatial_points: int = 2048,
        flux_normalization_stats_file: Optional[Union[str, Path]] = None,
        normalize_coordinates: bool = True,
        flux_clip_threshold: float = 1e-8,
        load_ground_truth_qoi: bool = False,
        # Advanced options
        temporal_stride: int = 1,
        expand_timesteps: bool = True,
        split_file: Optional[Union[str, Path]] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        seed: Optional[int] = None,
        # Cache options
        cache_static_arrays: bool = True,
        max_cache_size: int = 200,
        # Transolver-specific options
        include_q_in_embedding: bool = True,
        # Fourier features options
        use_fourier_features: bool = False,
        fourier_num_frequencies: int = 3,
        fourier_coord_dims: int = 2,
        fourier_base_frequency: float = 1.0,
    ) -> "RTEDataPipe":
        """Create a datapipe from a simple configuration.

        The transform pipeline and adapter are built by ``_build_transforms``
        and ``_build_adapter`` respectively; this method is a thin
        orchestrator that validates required inputs, composes both stages,
        and returns the configured datapipe.
        """
        if flux_normalization_stats_file is None:
            raise ValueError(
                "flux_normalization_stats_file is required. "
                "Run compute_normalizations.py first to generate statistics file."
            )

        transforms = _build_transforms(
            data_path=data_path,
            case_type=case_type,
            task=task,
            flux_normalization_stats_file=flux_normalization_stats_file,
            flux_clip_threshold=flux_clip_threshold,
            temporal_stride=temporal_stride,
            seed=seed,
            num_spatial_points=num_spatial_points,
            normalize_coordinates=normalize_coordinates,
            use_fourier_features=use_fourier_features,
            fourier_num_frequencies=fourier_num_frequencies,
            fourier_coord_dims=fourier_coord_dims,
            fourier_base_frequency=fourier_base_frequency,
            load_ground_truth_qoi=load_ground_truth_qoi,
        )

        adapter_obj = _build_adapter(
            adapter,
            include_q_in_embedding=include_q_in_embedding,
        )

        # steady_state always uses t=0 → t=T, so per-step expansion is moot.
        if task == "steady_state":
            expand_timesteps = False

        return cls(
            data_path=data_path,
            transforms=transforms,
            adapter=adapter_obj,
            case_type=case_type,
            phase=phase,
            split_file=split_file,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
            expand_timesteps=expand_timesteps,
            temporal_stride=temporal_stride,
            cache_static_arrays=cache_static_arrays,
            max_cache_size=max_cache_size,
            task=task,
        )

    def preload_to_memory(self, verbose: bool = True, num_workers: int = 8) -> dict:
        """Preload all static arrays into main process memory.

        Workers inherit the populated cache via fork, eliminating disk I/O.
        Uses parallel I/O for faster loading on multi-core systems.
        """
        return self.base_dataset.preload_to_memory(
            verbose=verbose, num_workers=num_workers
        )

    def get_raw_sample(self, idx: int) -> TensorDict:
        """Get a raw sample as a ``TensorDict`` (pre-transform, pre-adapter)."""
        td = self.base_dataset[idx]
        if not isinstance(td, TensorDict):
            td = td_from_dict(td)
        return td

    def get_transformed_sample(self, idx: int) -> TensorDict:
        """Get sample with transforms applied but no adapter (``TensorDict``)."""
        td = self.get_raw_sample(idx)
        if self.transforms is not None:
            td = self.transforms(td)
        return td

    def __repr__(self) -> str:
        lines = [
            f"{self.__class__.__name__}(",
            f"  base_dataset={self.base_dataset}",
            f"  transforms={self.transforms}",
            f"  adapter={self.adapter}",
            ")",
        ]
        return "\n".join(lines)


def _build_transforms(
    *,
    data_path: Union[str, Path],
    case_type: Optional[str],
    task: str,
    flux_normalization_stats_file: Union[str, Path],
    flux_clip_threshold: float,
    temporal_stride: int,
    seed: Optional[int],
    num_spatial_points: int,
    normalize_coordinates: bool,
    use_fourier_features: bool,
    fourier_num_frequencies: int,
    fourier_coord_dims: int,
    fourier_base_frequency: float,
    load_ground_truth_qoi: bool,
) -> Compose:
    """Assemble the canonical RTE transform pipeline.

    Normalization steps are delegated to PhysicsNeMo primitives:

    1. ``RTEFluxLogClip`` + ``Normalize`` — flux: log+clip, then z-score.
    2. Temporal sampler — ``NextStepSampler`` | ``SteadyStateSampler``.
    3. ``MaterialPropertyExtractor`` — always.
    4. ``Normalize`` (``physical_properties``) — per-column z-score via broadcast.
    5. ``SpatialSampler`` — always.
    6. ``RTEBackupCoords`` + ``Translate`` + ``Scale`` — when normalize_coordinates.
    7. ``FourierFeatures`` — when use_fourier_features.
    8. ``LoadGroundTruthQoI`` — when load_ground_truth_qoi.
    """
    flux_stats = load_flux_stats(flux_normalization_stats_file)
    if abs(flux_stats["clip_threshold"] - flux_clip_threshold) > 1e-10:
        raise ValueError(
            f"Clip threshold mismatch: got {flux_clip_threshold}, "
            f"stats computed with {flux_stats['clip_threshold']}"
        )

    transform_list = [
        RTEFluxLogClip(
            clip_threshold=flux_clip_threshold,
            log_flux_mean=flux_stats["log_flux_mean"],
            log_flux_std=flux_stats["log_flux_std"],
        ),
        Normalize(**flux_normalize_kwargs(flux_stats, field="scalar_flux")),
    ]

    if task == "next_step":
        transform_list.append(NextStepSampler(stride=temporal_stride, seed=seed))
    elif task == "steady_state":
        transform_list.append(SteadyStateSampler())
    else:
        raise ValueError(f"Unknown task: {task}")

    transform_list.append(MaterialPropertyExtractor(case_type=case_type))

    material_stats_path = (
        Path(flux_normalization_stats_file).parent / f"{case_type}_material_stats.yaml"
    )
    if not material_stats_path.exists():
        raise FileNotFoundError(
            f"Material statistics file not found: {material_stats_path}\n"
            f"Run compute_normalizations.py to generate it."
        )
    material_stats = load_material_stats(material_stats_path)
    transform_list.append(
        Normalize(
            **material_normalize_kwargs(
                material_stats, field="physical_properties", method="mean_std"
            )
        )
    )

    transform_list.append(SpatialSampler(num_points=num_spatial_points, seed=seed))

    if normalize_coordinates:
        if case_type is None:
            raise ValueError(
                "case_type is required when normalize_coordinates=True "
                "(used to look up the global domain bounds)."
            )
        center, half_extent = coord_translate_scale_params(case_type)
        # RTEBackupCoords preserves raw coords AND writes bbox_min / bbox_max
        # so downstream readers keep working.
        transform_list.append(
            RTEBackupCoords(
                bbox_min=center - half_extent,
                bbox_max=center + half_extent,
            )
        )
        transform_list.append(
            Translate(
                input_keys=["coordinates"],
                center_key_or_value=center,
                subtract=True,
            )
        )
        transform_list.append(
            Scale(
                input_keys=["coordinates"],
                scale=half_extent,
                divide=True,
            )
        )

    if use_fourier_features:
        transform_list.append(
            FourierFeatures(
                num_frequencies=fourier_num_frequencies,
                coord_dims=fourier_coord_dims,
                base_frequency=fourier_base_frequency,
                append_to_coordinates=True,
            )
        )

    if load_ground_truth_qoi:
        transform_list.append(LoadGroundTruthQoI(data_path=data_path))

    return Compose(transform_list)


def _build_adapter(
    adapter: Optional[str],
    *,
    include_q_in_embedding: bool,
):
    """Build the model-specific output adapter (or ``None``).

    Collapsed to a constant for the upstream Transolver-only example: the
    only valid non-``None`` value is ``"transolver"``. Kept as a function
    for plug-in clarity so ``from_config`` flows naturally.
    """
    if adapter is None:
        return None
    if adapter == "transolver":
        return TransolverAdapter(include_q_in_embedding=include_q_in_embedding)
    raise ValueError(
        f"Unknown adapter: {adapter!r}. The upstream example ships only "
        "'transolver' (or None for raw TensorDicts)."
    )


def create_dataset(
    case_type: Literal["lattice", "hohlraum"],
    data_path: Union[str, Path],
    phase: Literal["train", "val", "test"] = "train",
    task: Literal["next_step", "steady_state"] = "next_step",
    adapter: Optional[Literal["transolver"]] = "transolver",
    **kwargs,
) -> RTEDataPipe:
    """Create a dataset for the given case type."""
    if case_type not in ("lattice", "hohlraum"):
        raise ValueError(
            f"Unknown case_type: {case_type!r}. Expected 'lattice' or 'hohlraum'."
        )
    return RTEDataPipe.from_config(
        data_path=data_path,
        case_type=case_type,
        phase=phase,
        task=task,
        adapter=adapter,
        **kwargs,
    )


# =========================================================================
# Distributed preload barrier
# =========================================================================
#
# File-marker rank-sequencing helpers used to serialize the per-rank zarr
# preload step in multi-GPU training. Sequencing avoids I/O contention and
# leverages OS page cache reuse across ranks.


def _wait_for_rank_preload(
    my_rank: int,
    target_rank: int,
    barrier_dir: str,
    timeout: int = 7200,
) -> None:
    barrier_path = Path(barrier_dir)
    barrier_path.mkdir(parents=True, exist_ok=True)
    marker_file = barrier_path / f".preload_done_rank{target_rank}"

    if my_rank == target_rank:
        marker_file.touch()
        return

    start = time.time()
    while not marker_file.exists():
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Rank {my_rank} timed out waiting for rank {target_rank} "
                f"preload after {timeout}s."
            )
        time.sleep(1.0)


def _cleanup_preload_markers(barrier_dir: str, world_size: int) -> None:
    barrier_path = Path(barrier_dir)
    for r in range(world_size):
        marker_file = barrier_path / f".preload_done_rank{r}"
        try:
            marker_file.unlink()
        except FileNotFoundError:
            pass


def _distributed_preload(
    datasets: Dict[str, Any],
    dist,
    cfg: DictConfig,
    logger: logging.Logger,
) -> None:
    """Preload static arrays (and steady-state flux) into main-process memory.

    Distributed variant sequences ranks via file markers to avoid I/O
    contention and leverage OS page cache reuse. Single-process variant is
    a straight call to ``preload_to_memory`` on each dataset.
    """
    targets = [
        (phase, ds) for phase, ds in datasets.items() if phase in ("train", "val")
    ]
    if not targets:
        return

    if dist.distributed and torch_dist.is_initialized() and dist.world_size > 1:
        if dist.rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("DISTRIBUTED PRELOADING")
            logger.info("=" * 60)
            logger.info(f"Ranks preload sequentially (world_size={dist.world_size})")

        barrier_dir = cfg.output
        if dist.rank == 0:
            _cleanup_preload_markers(barrier_dir, dist.world_size)
            print("[Rank 0] Cleaned up stale preload markers", flush=True)

        time.sleep(1.0)

        if dist.rank > 0:
            print(
                f"[Rank {dist.rank}] Waiting for rank {dist.rank - 1} to finish preloading...",
                flush=True,
            )
        for prev in range(dist.rank):
            _wait_for_rank_preload(dist.rank, prev, barrier_dir, timeout=7200)

        print(f"[Rank {dist.rank}] Starting preload...", flush=True)
        for _, ds in targets:
            ds.preload_to_memory(verbose=True)
        print(f"[Rank {dist.rank}] Preload complete!", flush=True)

        _wait_for_rank_preload(dist.rank, dist.rank, barrier_dir, timeout=7200)
        print(
            f"[Rank {dist.rank}] Signaled completion, waiting for all ranks...",
            flush=True,
        )
        for other in range(dist.world_size):
            _wait_for_rank_preload(dist.rank, other, barrier_dir, timeout=7200)
        print(f"[Rank {dist.rank}] All ranks completed preloading!", flush=True)

        torch_dist.barrier()

        if dist.rank == 0:
            _cleanup_preload_markers(barrier_dir, dist.world_size)
            logger.info("=" * 60 + "\n")
    else:
        if dist is None or dist.rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("SINGLE-GPU PRELOADING")
            logger.info("=" * 60)
            logger.info("Loading static arrays into memory...")
        for _, ds in targets:
            ds.preload_to_memory(verbose=(dist is None or dist.rank == 0))
        if dist is None or dist.rank == 0:
            logger.info("=" * 60 + "\n")


# =========================================================================
# DataLoader builder
# =========================================================================
#
# ``build_dataloaders`` is the main entry point used by ``train.py`` and
# ``inference.py``. It orchestrates dataset creation, distributed-preload
# synchronization, material sanity logging, sampler construction, and
# per-phase ``DataLoader`` assembly.


def _log_material_sanity(dataset, cfg: DictConfig, logger: logging.Logger) -> None:
    """Log material-property ranges from the first sample for diagnostics."""
    if len(dataset) == 0:
        return
    sample = dataset.get_transformed_sample(0)
    if "physical_properties" not in sample or sample["physical_properties"] is None:
        return

    phys = sample["physical_properties"]
    if isinstance(phys, torch.Tensor):
        phys = phys.detach().cpu().numpy()

    sigma_a = phys[:, 0]
    sigma_s = phys[:, 1]
    Q = phys[:, 3]

    logger.info("\nMaterial property ranges (first sample):")
    logger.info(
        f"  sigma_a: [{sigma_a.min():.2f}, {sigma_a.max():.2f}] "
        f"(unique: {len(set(sigma_a.tolist()))})"
    )
    logger.info(
        f"  sigma_s: [{sigma_s.min():.2f}, {sigma_s.max():.2f}] "
        f"(unique: {len(set(sigma_s.tolist()))})"
    )
    logger.info(f"  Q: {sorted(set(Q.tolist()))}")
    if len(set(sigma_a.tolist())) > 1 or len(set(sigma_s.tolist())) > 1:
        logger.info(f"  Heterogeneous materials detected ({cfg.case.type})")
    else:
        logger.info(f"  Homogeneous materials ({cfg.case.type})")


def _make_loader(
    dataset,
    cfg: DictConfig,
    phase: str,
    sampler: Optional[Sampler],
    collate_fn: Optional[Callable],
    test_batch_size: int,
    test_num_workers: int,
) -> DataLoader:
    """Assemble a ``DataLoader`` for one phase, reading per-phase config.

    The ``test`` phase has no matching ``cfg.test.*`` block; callers pass
    ``test_batch_size`` / ``test_num_workers`` explicitly.
    """
    if phase == "test":
        return DataLoader(
            dataset,
            batch_size=test_batch_size,
            num_workers=test_num_workers,
            shuffle=False,
            pin_memory=False,
            collate_fn=collate_fn,
        )

    phase_cfg = cfg.train.dataloader if phase == "train" else cfg.train.val.dataloader
    sampler_cfg = cfg.train.sampler if phase == "train" else cfg.train.val.sampler
    num_workers = phase_cfg.num_workers

    # sampler handles shuffling when present; keep ``shuffle=False`` to avoid
    # the PyTorch "sampler is incompatible with shuffle" error.
    shuffle_train = sampler_cfg.shuffle if phase == "train" else False
    shuffle = shuffle_train if sampler is None else False

    kwargs = {
        "batch_size": phase_cfg.batch_size,
        "pin_memory": phase_cfg.pin_memory,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "drop_last": sampler_cfg.get("drop_last", False),
        "sampler": sampler,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = phase_cfg.get("prefetch_factor", 2)
        kwargs["persistent_workers"] = phase_cfg.get("persistent_workers", False)

    return DataLoader(dataset, **kwargs)


class DistributedEvalSampler(Sampler[int]):
    """Shard eval data across ranks without padding or duplicate samples."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.num_replicas) + 1


def build_dataloaders(
    cfg: DictConfig,
    dist=None,
    *,
    adapter: str = "transolver",
    collate_fn: Optional[Callable] = None,
    extra_dataset_kwargs: Optional[dict] = None,
    phases: Iterable[str] = ("train", "val"),
    num_spatial_points_key: str = "num_spatial_points",
    num_spatial_points_override: Optional[int] = None,
    split_file_override: Optional[str] = None,
    test_batch_size: int = 1,
    test_num_workers: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, DataLoader], Optional[DistributedSampler]]:
    """Build per-phase ``DataLoader`` s for training and/or evaluation.

    Args:
        cfg: Hydra configuration (training cfg or a loaded checkpoint cfg).
        dist: ``DistributedManager`` for training; ``None`` for eval.
        adapter: Model adapter identifier (only ``"transolver"`` is shipped).
        collate_fn: Collate function. Defaults to ``collate_no_padding``.
        extra_dataset_kwargs: Additional kwargs forwarded to ``create_dataset``.
        phases: Which splits to build (subset of ``{"train", "val", "test"}``).
        num_spatial_points_key: Where to read ``num_spatial_points`` from
            the config (dotted path). Overridden by
            ``num_spatial_points_override`` when the caller already knows
            the authoritative value (eval path).
        num_spatial_points_override: Optional explicit ``num_spatial_points``.
        split_file_override: CLI override for ``data.split_file``.
        test_batch_size / test_num_workers: Used only when ``test`` is in
            ``phases``.
        logger: Optional logger; defaults to module logger.

    Returns:
        ``({phase: DataLoader}, train_sampler)``. ``train_sampler`` is
        ``None`` when ``train`` is not in ``phases`` or ``dist`` is not
        distributed.
    """
    logger = logger or logging.getLogger(__name__)
    phases = tuple(phases)

    # Hardcode the collate to ``collate_no_padding`` — the upstream example
    # ships only the point-cloud adapter, which requires batch_size=1.
    if collate_fn is None:
        collate_fn = collate_no_padding

    rank_zero = dist is None or dist.rank == 0

    if rank_zero:
        logger.info(f"Loading {cfg.case.type} data from: {cfg.case.data_path}")

    common_kwargs = build_rte_dataset_kwargs(
        cfg,
        adapter=adapter,
        num_spatial_points_key=num_spatial_points_key,
        num_spatial_points_override=num_spatial_points_override,
        split_file_override=split_file_override,
        extra_kwargs=extra_dataset_kwargs,
    )

    if rank_zero:
        logger.info(f"Task mode: {common_kwargs['task']}")
        if common_kwargs["split_file"]:
            logger.info(f"Using predefined splits from: {common_kwargs['split_file']}")
        if common_kwargs["max_cache_size"] == -1:
            logger.info("Data caching: UNLIMITED")
        else:
            logger.info(
                f"Data caching: LRU max_cache_size={common_kwargs['max_cache_size']}"
            )

    datasets = {
        phase: create_dataset(cfg.case.type, phase=phase, **common_kwargs)
        for phase in phases
    }

    # Distributed/single preloading (training only; eval skips).
    preload_data = False
    if "data" in cfg:
        preload_data = cfg.data.get("preload_data", False)
    if (
        preload_data
        and common_kwargs["max_cache_size"] == -1
        and dist is not None
        and any(p in datasets for p in ("train", "val"))
    ):
        _distributed_preload(datasets, dist, cfg, logger)

    if rank_zero:
        split_summary = ", ".join(f"{p}={len(datasets[p])}" for p in phases)
        logger.info(f"\nData split summary: {split_summary}")
        if "train" in datasets:
            _log_material_sanity(datasets["train"], cfg, logger)

    # Samplers + loaders.
    train_sampler: Optional[DistributedSampler] = None
    loaders: Dict[str, DataLoader] = {}
    for phase in phases:
        sampler = None
        if dist is not None and dist.distributed and phase in ("train", "val"):
            if phase == "train":
                sampler = DistributedSampler(
                    datasets[phase],
                    num_replicas=dist.world_size,
                    rank=dist.rank,
                    shuffle=cfg.train.sampler.shuffle,
                    drop_last=cfg.train.sampler.get("drop_last", False),
                )
                train_sampler = sampler
            else:
                sampler = DistributedEvalSampler(
                    datasets[phase],
                    num_replicas=dist.world_size,
                    rank=dist.rank,
                )

        loaders[phase] = _make_loader(
            datasets[phase],
            cfg,
            phase,
            sampler,
            collate_fn,
            test_batch_size=test_batch_size,
            test_num_workers=test_num_workers,
        )

    return loaders, train_sampler


__all__ = [
    "ModelAdapter",
    "TransolverAdapter",
    "collate_no_padding",
    "build_rte_dataset_kwargs",
    "RTEDataPipe",
    "create_dataset",
    "build_dataloaders",
]
