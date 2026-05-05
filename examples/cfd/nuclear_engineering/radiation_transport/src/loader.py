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

"""Data plumbing: TransolverAdapter (Transform), collate, dataset+loader builder.

This module composes :class:`RTEBaseDataset` (a
:class:`physicsnemo.datapipes.Dataset` subclass) with a ``Compose`` of
transforms — including the trailing :class:`TransolverAdapter` Transform —
and exposes a single ``build_dataloaders`` entry point used by ``train.py``
and ``inference.py``.

Sections:

* Adapter — :class:`TransolverAdapter` (a :class:`Transform`).
* Collation — :func:`collate_no_padding` (batch_size=1 unsqueeze).
* Stats / kwargs translation — :func:`_build_rte_dataset_kwargs`.
* Pipeline orchestration — :func:`_build_transforms` + :func:`_build_rte_dataset`.
* Distributed preload barrier — file-marker rank-sequencing helpers.
* DataLoader builder — :func:`build_dataloaders` + :func:`_make_loader` +
  :func:`_log_material_sanity`.
"""

from __future__ import annotations

# =========================================================================
# Imports
# =========================================================================

import logging
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.distributed as torch_dist
from omegaconf import DictConfig
from physicsnemo.datapipes import DataLoader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms import Compose, Normalize, Scale, Translate
from physicsnemo.datapipes.transforms.base import Transform
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
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
    FourierFeatures,
    RTEBackupCoords,
    RTEFluxLogClip,
    SpatialSampler,
    SteadyStateSampler,
)


# =========================================================================
# Adapter
# =========================================================================
#
# ``TransolverAdapter`` is the trailing :class:`Transform` in the RTE pipeline.
# It rewrites the field-name layout of the TensorDict to match what
# :class:`physicsnemo.models.transolver.Transolver` expects (``fx``,
# ``embedding``, ``flux_target``, ...) and drops fields the model never reads.


@register("RTETransolverAdapter")
class TransolverAdapter(Transform):
    """Pack a transformed RTE ``TensorDict`` into Transolver-ready fields.

    Output TensorDict keys:

    * ``fx`` — spatial coordinates (plus Fourier features when enabled).
    * ``embedding`` — material properties ``[sigma_a, sigma_s, sigma_t, Q]``
      (or just the first three when ``include_q_in_embedding=False``).
    * ``flux_target`` — target flux to predict.

    Pass-through fields when present: ``coordinates_unnormalized``,
    ``material_labels``, ``cell_areas``, ``sigma_t``, ``sigma_s``,
    ``sim_time``, and ``flux_normalization_stats`` (NonTensorData).

    A trimmed ``metadata`` dict (timestep / filename / case_type) is also
    re-attached as NonTensorData so downstream physics-loss + inference
    code paths keep their existing ``batch["metadata"]`` access pattern.

    The output has no batch dimension; :func:`collate_no_padding` adds one.
    """

    def __init__(self, include_q_in_embedding: bool = True):
        super().__init__()
        self.include_q_in_embedding = include_q_in_embedding

    def __call__(self, data: TensorDict) -> TensorDict:
        out = TensorDict({}, batch_size=data.batch_size, device=data.device)

        if "coordinates" in data:
            out["fx"] = data["coordinates"]

        if "physical_properties" in data:
            mat_props = data["physical_properties"]
            if not self.include_q_in_embedding:
                mat_props = mat_props[..., :3]
            out["embedding"] = mat_props

        if "coordinates_unnormalized" in data:
            out["coordinates_unnormalized"] = data["coordinates_unnormalized"]

        if "material_properties" in data and data["material_properties"] is not None:
            out["material_labels"] = data["material_properties"].long()

        if "flux_target" in data:
            flux_tgt = data["flux_target"]
            if flux_tgt.ndim == 1:
                flux_tgt = flux_tgt.unsqueeze(-1)
            out["flux_target"] = flux_tgt

        for key in ("cell_areas", "sigma_t", "sigma_s"):
            if key in data:
                out[key] = data[key]

        if "sim_times" in data and data["sim_times"].numel() > 0:
            out["sim_time"] = data["sim_times"][-1].reshape(1).to(torch.float32)
        elif "sim_times" in data:
            out["sim_time"] = torch.tensor(
                [0.0], dtype=torch.float32, device=data.device
            )

        if "flux_normalization_stats" in data:
            out.set_non_tensor(
                "flux_normalization_stats", data["flux_normalization_stats"]
            )

        # Trim metadata to the keys downstream code reads. The full metadata
        # dict is also delivered as the second tuple element by the dataset.
        src_meta = data["metadata"] if "metadata" in data else {}
        if not isinstance(src_meta, dict):
            src_meta = {}
        trimmed = {
            "timestep_input": data["timestep_input"]
            if "timestep_input" in data
            else None,
            "timestep_target": data["timestep_target"]
            if "timestep_target" in data
            else None,
            "max_timestep": src_meta.get("max_timestep"),
            "filename": data["filename"] if "filename" in data else None,
            "case_type": src_meta.get("case_type"),
        }
        trimmed = {k: v for k, v in trimmed.items() if v is not None}
        out.set_non_tensor("metadata", trimmed)
        if "filename" in data:
            out.set_non_tensor("filename", data["filename"])
        return out

    def extra_repr(self) -> str:
        return f"include_q_in_embedding={self.include_q_in_embedding}"


# =========================================================================
# Collation
# =========================================================================
#
# All configs use ``batch_size=1`` with a fixed ``num_spatial_points`` from
# ``SpatialSampler``, so no padding is needed. ``build_dataloaders_for_training``
# enforces ``batch_size=1`` upstream, so this collate just unsqueezes the
# single sample.
#
# ``physicsnemo.datapipes.DataLoader`` passes ``list[tuple[TensorDict, dict]]``
# into the collate function. We unpack the tuple, unsqueeze each tensor in the
# TensorDict, and merge any TD-side NonTensorData ("metadata",
# "flux_normalization_stats", "filename") plus the second-element metadata
# back into the returned dict.


@register("RTECollateNoPadding")
def collate_no_padding(
    batch: Sequence[Tuple[TensorDict, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Batch-size-1 collate.

    Expects the :class:`physicsnemo.datapipes.DataLoader` calling convention:
    ``list[tuple[TensorDict, dict]]``. Returns a plain dict (not a TensorDict)
    so downstream training code can keep using ``batch["fx"]`` / ``batch["filename"]``
    etc. without unpacking.
    """
    assert len(batch) == 1, (
        f"collate_no_padding requires batch_size=1; got {len(batch)}"
    )
    item = batch[0]
    # ``physicsnemo.datapipes.DataLoader`` always passes (TensorDict, dict)
    # tuples through. Be defensive against accidental dict-only inputs from
    # legacy callers.
    if isinstance(item, tuple) and len(item) == 2:
        td, metadata = item
    else:
        td = item
        metadata = {}

    out: Dict[str, Any] = {}
    if isinstance(td, TensorDict):
        for key in td.keys():
            value = td[key]
            if isinstance(value, torch.Tensor):
                out[key] = value.unsqueeze(0)
            else:
                out[key] = value
    elif isinstance(td, dict):
        for key, value in td.items():
            out[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else value

    # Merge the trailing metadata dict back into the batch under "metadata".
    # ``filename`` is also surfaced at the top level for the legacy access
    # pattern ``batch["filename"]`` used by some downstream code.
    if metadata:
        existing = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
        merged_meta = {**metadata, **(existing or {})}
        out["metadata"] = merged_meta
        if "filename" in metadata and "filename" not in out:
            out["filename"] = metadata["filename"]
    return out


# =========================================================================
# Stats / kwargs translation
# =========================================================================
#
# ``_build_rte_dataset_kwargs`` translates a Hydra config into the kwargs
# ``_build_rte_dataset`` expects. Required keys (``flux_normalization_stats_file``,
# ``flux_clip_threshold``, ``case.split_file``) raise a clear ``KeyError`` on
# direct access if missing; no manual ``None``-checking is layered on top.


def _build_rte_dataset_kwargs(cfg: DictConfig) -> dict:
    """Translate a Hydra config into the kwargs ``_build_rte_dataset`` expects."""
    data_cfg = cfg.data
    use_fourier_features = data_cfg.get("use_fourier_features", False)
    fourier_cfg = data_cfg.get("fourier_features") if use_fourier_features else None

    return {
        "data_path": cfg.case.data_path,
        "num_spatial_points": cfg.model.num_spatial_points,
        "flux_normalization_stats_file": data_cfg.flux_normalization_stats_file,
        "normalize_coordinates": data_cfg.get("normalize_coordinates", True),
        "flux_clip_threshold": data_cfg.flux_clip_threshold,
        "split_file": cfg.case.split_file,
        "seed": (
            data_cfg.get("seed", None)
            if data_cfg.get("seed", None) is not None
            else cfg.get("train", {}).get("seed", None)
        ),
        "cache_static_arrays": data_cfg.get("cache_static_arrays", True),
        "max_cache_size": data_cfg.get("max_cache_size", 200),
        "include_q_in_embedding": cfg.model.get("include_q_in_embedding", True),
        "use_fourier_features": use_fourier_features,
        "fourier_num_frequencies": fourier_cfg.num_frequencies if fourier_cfg else None,
        "fourier_coord_dims": fourier_cfg.coord_dims if fourier_cfg else None,
        "fourier_base_frequency": fourier_cfg.base_frequency if fourier_cfg else None,
    }


# =========================================================================
# Pipeline orchestration
# =========================================================================
#
# ``_build_rte_dataset`` is the high-level builder used by ``build_dataloaders``;
# ``compute_normalizations.py`` instantiates ``RTEBaseDataset`` directly with
# ``transforms=None`` to walk raw samples for stats.


def _build_rte_dataset(
    case_type: Literal["lattice", "hohlraum"],
    data_path: Union[str, Path],
    phase: Literal["train", "val", "test"],
    *,
    num_spatial_points: int,
    flux_normalization_stats_file: Union[str, Path],
    normalize_coordinates: bool,
    flux_clip_threshold: float,
    split_file: Union[str, Path],
    seed: Optional[int],
    cache_static_arrays: bool,
    max_cache_size: int,
    include_q_in_embedding: bool,
    use_fourier_features: bool,
    fourier_num_frequencies: Optional[int],
    fourier_coord_dims: Optional[int],
    fourier_base_frequency: Optional[float],
    device: Optional[Union[str, torch.device]] = None,
) -> RTEBaseDataset:
    """Build the canonical training/inference RTE dataset (transforms baked in)."""
    if case_type not in ("lattice", "hohlraum"):
        raise ValueError(
            f"Unknown case_type: {case_type!r}. Expected 'lattice' or 'hohlraum'."
        )

    transforms = _build_transforms(
        data_path=data_path,
        case_type=case_type,
        flux_normalization_stats_file=flux_normalization_stats_file,
        flux_clip_threshold=flux_clip_threshold,
        seed=seed,
        num_spatial_points=num_spatial_points,
        normalize_coordinates=normalize_coordinates,
        use_fourier_features=use_fourier_features,
        fourier_num_frequencies=fourier_num_frequencies,
        fourier_coord_dims=fourier_coord_dims,
        fourier_base_frequency=fourier_base_frequency,
        include_q_in_embedding=include_q_in_embedding,
    )

    return RTEBaseDataset(
        data_path=data_path,
        case_type=case_type,
        phase=phase,
        split_file=split_file,
        seed=seed,
        load_sigma_fields=True,
        cache_static_arrays=cache_static_arrays,
        max_cache_size=max_cache_size,
        transforms=transforms,
        device=device,
    )


def _build_transforms(
    *,
    data_path: Union[str, Path],
    case_type: Optional[str],
    flux_normalization_stats_file: Union[str, Path],
    flux_clip_threshold: float,
    seed: Optional[int],
    num_spatial_points: int,
    normalize_coordinates: bool,
    use_fourier_features: bool,
    fourier_num_frequencies: int,
    fourier_coord_dims: int,
    fourier_base_frequency: float,
    include_q_in_embedding: bool = True,
) -> Compose:
    """Assemble the canonical RTE transform pipeline.

    Steps:

    1. ``RTEFluxLogClip`` + ``Normalize`` — flux: log+clip, then z-score.
    2. ``SteadyStateSampler`` — first snapshot input, final snapshot target.
    3. ``MaterialPropertyExtractor`` — always.
    4. ``Normalize`` (``physical_properties``) — per-column z-score via broadcast.
    5. ``SpatialSampler`` — always.
    6. ``RTEBackupCoords`` + ``Translate`` + ``Scale`` — when normalize_coordinates.
    7. ``FourierFeatures`` — when use_fourier_features.
    8. ``TransolverAdapter`` — repack into Transolver-ready fields.
    """
    flux_stats = load_flux_stats(flux_normalization_stats_file)
    if abs(flux_stats["clip_threshold"] - flux_clip_threshold) > 1e-10:
        raise ValueError(
            f"Clip threshold mismatch: got {flux_clip_threshold}, "
            f"stats computed with {flux_stats['clip_threshold']}"
        )

    transform_list: List[Transform] = [
        RTEFluxLogClip(
            clip_threshold=flux_clip_threshold,
            log_flux_mean=flux_stats["log_flux_mean"],
            log_flux_std=flux_stats["log_flux_std"],
        ),
        Normalize(**flux_normalize_kwargs(flux_stats, field="scalar_flux")),
    ]

    transform_list.append(SteadyStateSampler())
    transform_list.append(MaterialPropertyExtractor())

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
            **material_normalize_kwargs(material_stats, field="physical_properties")
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
        transform_list.append(RTEBackupCoords())
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

    transform_list.append(
        TransolverAdapter(include_q_in_embedding=include_q_in_embedding)
    )

    return Compose(transform_list)


# =========================================================================
# Distributed preload barrier
# =========================================================================
#
# File-marker rank-sequencing helpers used to serialize the per-rank mesh
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
# per-phase :class:`physicsnemo.datapipes.DataLoader` assembly.


def _log_material_sanity(dataset, cfg: DictConfig, logger: logging.Logger) -> None:
    """Log material-property ranges from the first sample for diagnostics."""
    if len(dataset) == 0:
        return
    sample = dataset.get_transformed_sample(0)
    # The trailing ``TransolverAdapter`` produces a TensorDict whose material
    # info lives under ``embedding`` (sigma_a, sigma_s, sigma_t, [Q]). Fall back
    # to the un-adapted ``physical_properties`` key if the adapter wasn't run.
    if "embedding" in sample:
        phys = sample["embedding"]
    elif "physical_properties" in sample:
        phys = sample["physical_properties"]
    else:
        return

    if isinstance(phys, torch.Tensor):
        phys = phys.detach().cpu().numpy()

    sigma_a = phys[:, 0]
    sigma_s = phys[:, 1]
    Q = phys[:, 3] if phys.shape[1] >= 4 else None

    logger.info("\nMaterial property ranges (first sample):")
    logger.info(
        f"  sigma_a: [{sigma_a.min():.2f}, {sigma_a.max():.2f}] "
        f"(unique: {len(set(sigma_a.tolist()))})"
    )
    logger.info(
        f"  sigma_s: [{sigma_s.min():.2f}, {sigma_s.max():.2f}] "
        f"(unique: {len(set(sigma_s.tolist()))})"
    )
    if Q is not None:
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
) -> DataLoader:
    """Assemble a :class:`physicsnemo.datapipes.DataLoader` for one phase.

    The ``test`` phase has no matching ``cfg.test.*`` block; callers pass
    ``test_batch_size`` explicitly. Stream-based prefetching defaults
    (``num_streams=4``, ``use_streams=true``) come from the per-phase
    Hydra config when present.
    """
    if phase == "test":
        return DataLoader(
            dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    phase_cfg = cfg.train.dataloader if phase == "train" else cfg.train.val.dataloader
    sampler_cfg = cfg.train.sampler if phase == "train" else cfg.train.val.sampler

    # sampler handles shuffling when present; keep ``shuffle=False`` to avoid
    # the "sampler is incompatible with shuffle" path inside the DataLoader.
    shuffle_train = sampler_cfg.shuffle if phase == "train" else False
    shuffle = shuffle_train if sampler is None else False

    seed = cfg.train.get("seed", None)
    seed = int(seed) if seed is not None else None

    return DataLoader(
        dataset,
        batch_size=phase_cfg.batch_size,
        shuffle=shuffle,
        drop_last=sampler_cfg.get("drop_last", False),
        sampler=sampler,
        collate_fn=collate_fn,
        prefetch_factor=phase_cfg.get("prefetch_factor", 2),
        num_streams=phase_cfg.get("num_streams", 4),
        use_streams=phase_cfg.get("use_streams", True),
        seed=seed,
    )


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
    collate_fn: Optional[Callable] = None,
    phases: Iterable[str] = ("train", "val"),
    test_batch_size: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, DataLoader], Optional[DistributedSampler]]:
    """Build per-phase DataLoaders for training and/or evaluation.

    Args:
        cfg: Hydra configuration (training cfg or a loaded checkpoint cfg).
        dist: ``DistributedManager`` for training; ``None`` for eval.
        collate_fn: Collate function. Defaults to :func:`collate_no_padding`.
        phases: Which splits to build (subset of ``{"train", "val", "test"}``).
        test_batch_size: Used only when ``test`` is in ``phases``.
        logger: Optional logger; defaults to module logger.

    Returns:
        ``({phase: DataLoader}, train_sampler)``. ``train_sampler`` is
        ``None`` when ``train`` is not in ``phases`` or ``dist`` is not
        distributed.
    """
    logger = logger or logging.getLogger(__name__)
    phases = tuple(phases)

    if collate_fn is None:
        collate_fn = collate_no_padding

    rank_zero = dist is None or dist.rank == 0

    if rank_zero:
        logger.info(f"Loading {cfg.case.type} data from: {cfg.case.data_path}")

    common_kwargs = _build_rte_dataset_kwargs(cfg)

    if rank_zero:
        logger.info("Mapping mode: steady-state first-to-final flux")
        if common_kwargs["split_file"]:
            logger.info(f"Using predefined splits from: {common_kwargs['split_file']}")
        if common_kwargs["max_cache_size"] == -1:
            logger.info("Data caching: UNLIMITED")
        else:
            logger.info(
                f"Data caching: LRU max_cache_size={common_kwargs['max_cache_size']}"
            )

    # Pick a sensible device default. The PhysicsNeMo Dataset will move
    # tensors there before transforms run; transforms then operate on GPU.
    if dist is not None and getattr(dist, "device", None) is not None:
        device = dist.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = {
        phase: _build_rte_dataset(
            cfg.case.type, phase=phase, device=device, **common_kwargs
        )
        for phase in phases
    }

    # Distributed/single preloading (training only; eval skips).
    if (
        cfg.data.get("preload_data", False)
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
                    seed=int(cfg.train.get("seed", 0) or 0),
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
        )

    return loaders, train_sampler


__all__ = [
    "TransolverAdapter",
    "collate_no_padding",
    "build_dataloaders",
    "DistributedEvalSampler",
]
