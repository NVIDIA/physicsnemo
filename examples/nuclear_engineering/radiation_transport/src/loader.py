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

import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from omegaconf import DictConfig
from physicsnemo.datapipes import DataLoader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms import Compose, Normalize, Scale, Translate
from physicsnemo.datapipes.transforms.base import Transform
from tensordict import TensorDict
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from dataset import (
    RTEBaseDataset,
    flux_normalize_kwargs,
    load_flux_stats,
    load_material_stats,
    material_normalize_kwargs,
)
from transforms import (
    FourierFeatures,
    MaterialPropertyExtractor,
    RTEBackupCoords,
    RTEFluxLogClip,
    SpatialSampler,
    FinalTimeSampler,
    coord_translate_scale_params,
)

__all__ = [
    "TransolverAdapter",
    "collate_no_padding",
    "build_dataloaders",
]


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
    ``sim_time``, ``flux_normalization_stats`` (NonTensorData), and the
    eight hohlraum geometry parameters (``ulr``, ``llr``, ``urr``,
    ``lrr``, ``hlr``, ``hrr``, ``cx``, ``cy``).

    The output has no batch dimension; :func:`collate_no_padding` adds one.
    """

    def __init__(self, include_q_in_embedding: bool = True):
        super().__init__()
        self.include_q_in_embedding = include_q_in_embedding

    # Simple passthroughs: same key on both sides, no transform.
    _PASSTHROUGH_KEYS = (
        "coordinates_unnormalized",
        "cell_areas",
        "sigma_t",
        "sigma_s",
    )

    def __call__(self, data: TensorDict) -> TensorDict:
        out = TensorDict({}, batch_size=data.batch_size, device=data.device)

        # Rename: coordinates -> fx (Transolver's positional input).
        if "coordinates" in data:
            out["fx"] = data["coordinates"]

        # Passthroughs.
        for key in self._PASSTHROUGH_KEYS:
            if key in data:
                out[key] = data[key]

        # physical_properties -> embedding (optionally drop Q for hohlraum).
        if "physical_properties" in data:
            mat_props = data["physical_properties"]
            if not self.include_q_in_embedding:
                mat_props = mat_props[..., :3]
            out["embedding"] = mat_props

        # material_properties -> material_labels (long dtype for embedding lookups).
        if "material_properties" in data:
            out["material_labels"] = data["material_properties"].to(dtype=torch.long)

        # flux_target promoted to shape (N, 1) if delivered as (N,).
        if "flux_target" in data:
            flux_tgt = data["flux_target"]
            out["flux_target"] = (
                flux_tgt.unsqueeze(-1) if flux_tgt.ndim == 1 else flux_tgt
            )

        # sim_times -> single-scalar sim_time at the final snapshot;
        # zero-tensor placeholder when the source series is empty.
        if "sim_times" in data:
            sim_times = data["sim_times"]
            out["sim_time"] = (
                sim_times[-1].reshape(1).to(dtype=torch.float32)
                if sim_times.numel() > 0
                else torch.zeros(1, dtype=torch.float32, device=data.device)
            )

        # NonTensorData passthroughs.
        if "flux_normalization_stats" in data:
            out.set_non_tensor(
                "flux_normalization_stats", data["flux_normalization_stats"]
            )

        # Forward the eight hohlraum geometry parameters (0-D float32
        # tensors). Lattice samples never carry these keys.
        for key in ("ulr", "llr", "urr", "lrr", "hlr", "hrr", "cx", "cy"):
            if key in data:
                out[key] = data[key]

        return out

    def extra_repr(self) -> str:
        return f"include_q_in_embedding={self.include_q_in_embedding}"


@register("RTECollateNoPadding")
def collate_no_padding(
    batch: Sequence[Tuple[TensorDict, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Batch-size-1 collate for the ``physicsnemo.datapipes.DataLoader``.

    Unsqueezes each tensor in the TensorDict to add a ``B=1`` leading
    dim, passes NonTensorData entries through unchanged, and merges the
    trailing metadata dict under ``batch["metadata"]``. Returns a plain
    dict so downstream code can use ``batch["fx"]`` / ``batch["filename"]``
    without unpacking a TensorDict. ``build_dataloaders_for_training``
    enforces ``batch_size=1`` upstream so no padding is needed.
    """
    assert len(batch) == 1, (
        f"collate_no_padding requires batch_size=1; got {len(batch)}"
    )
    td, metadata = batch[0]

    out: Dict[str, Any] = {}
    for key in td.keys():
        value = td[key]
        out[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else value

    # Merge the trailing metadata dict (filename / case_type / num_cells
    # / num_timesteps / max_sim_time) under ``out["metadata"]``. Surface
    # ``filename`` at the top level too for callers that use
    # ``batch["filename"]`` directly (e.g. inference figure naming).
    if metadata:
        existing = out.get("metadata") or {}
        merged = {**metadata, **existing}
        out["metadata"] = merged
        if "filename" in merged and "filename" not in out:
            out["filename"] = merged["filename"]
    return out


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
        "seed": data_cfg.get("seed") or cfg.train.get("seed"),
        "cache_static_arrays": data_cfg.get("cache_static_arrays", True),
        "include_q_in_embedding": cfg.model.get("include_q_in_embedding", True),
        "use_fourier_features": use_fourier_features,
        "fourier_num_frequencies": fourier_cfg.num_frequencies if fourier_cfg else None,
        "fourier_coord_dims": fourier_cfg.coord_dims if fourier_cfg else None,
        "fourier_base_frequency": fourier_cfg.base_frequency if fourier_cfg else None,
    }


def _build_rte_dataset(
    case_type: str,
    data_path: Union[str, Path],
    phase: str,
    num_spatial_points: int,
    flux_normalization_stats_file: Union[str, Path],
    normalize_coordinates: bool,
    flux_clip_threshold: float,
    split_file: Union[str, Path],
    seed: Optional[int],
    cache_static_arrays: bool,
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
        cache_static_arrays=cache_static_arrays,
        transforms=transforms,
        device=device,
    )


def _build_transforms(
    case_type: str,
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
    """Assemble the canonical RTE transform pipeline."""
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

    transform_list.append(FinalTimeSampler())
    transform_list.append(MaterialPropertyExtractor())

    material_stats_path = (
        Path(flux_normalization_stats_file).parent / f"{case_type}_material_stats.yaml"
    )
    if not material_stats_path.exists():
        raise FileNotFoundError(
            f"Material statistics file not found: {material_stats_path}\n"
            f"Run src/compute_normalizations.py to generate it."
        )
    material_stats = load_material_stats(material_stats_path)
    transform_list.append(
        Normalize(
            **material_normalize_kwargs(material_stats, field="physical_properties")
        )
    )

    transform_list.append(SpatialSampler(num_points=num_spatial_points, seed=seed))

    if normalize_coordinates:
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


def build_dataloaders(
    cfg: DictConfig,
    dist=None,
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
        logger.info("Mapping mode: first-snapshot -> final-time flux")
        if common_kwargs["split_file"]:
            logger.info(f"Using predefined splits from: {common_kwargs['split_file']}")

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

    if rank_zero:
        split_summary = ", ".join(f"{p}={len(datasets[p])}" for p in phases)
        logger.info(f"\nData split summary: {split_summary}")

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
                sampler = DistributedSampler(
                    datasets[phase],
                    num_replicas=dist.world_size,
                    rank=dist.rank,
                    shuffle=False,
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
