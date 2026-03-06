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
Wrapper so train/inference use the physicsnemo datapipes (reader + transforms)
instead of the VTK/cache dataset. Same interface as dataset.py; switch via import:

  from physicsnemo_dataset import AirFRANSDataSet, AirFRANSSample, compute_max_mesh_sizes
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh

from dataset import (
    AirFRANSSample,
    AirFRANSDataSet as _OriginalAirFRANSDataSet,
    compute_max_mesh_sizes as _compute_max_mesh_sizes,
)

logger = logging.getLogger(__name__)


def _max_sizes_to_save_dict(result: TensorDict) -> dict:
    """Convert max_sizes TensorDict to a plain dict of tensors for safe torch.save."""
    return {
        bc_type: {
            "n_points": result[bc_type, "n_points"].cpu().clone(),
            "n_cells": result[bc_type, "n_cells"].cpu().clone(),
        }
        for bc_type in result.keys(include_nested=False)
    }


def _max_sizes_from_save_dict(data: dict, device: torch.device) -> TensorDict:
    """Reconstruct max_sizes TensorDict from a plain dict (from torch.load)."""
    return TensorDict(
        {
            bc_type: TensorDict(
                {
                    "n_points": v["n_points"].to(device),
                    "n_cells": v["n_cells"].to(device),
                },
            )
            for bc_type, v in data.items()
        },
    )


def compute_max_mesh_sizes(
    dataloader: Iterable[AirFRANSSample],
    device: torch.device,
    *,
    face_downsampling_ratio: float = 1.0,
    rank: int = 0,
    cache_dir: Path | None = None,
    airfrans_task: str | None = None,
    split: Literal["train", "test"] | None = None,
):
    """Compute max mesh sizes per BC type, with optional disk cache.

    When cache_dir, airfrans_task, and split are provided, loads from or saves to
    cache_dir / "max_mesh_sizes" / "{task}_{split}_ratio{ratio}_n{n_samp}.pt".
    """
    if (
        cache_dir is not None
        and airfrans_task is not None
        and split is not None
    ):
        cache_subdir = cache_dir / "max_mesh_sizes"
        n_samp = len(dataloader.dataset)
        cache_path = cache_subdir / (
            f"{airfrans_task}_{split}_ratio{face_downsampling_ratio}_n{n_samp}.pt"
        )
        if cache_path.exists():
            if rank == 0:
                logger.info("Loaded max mesh sizes from cache: %s", cache_path)
            try:
                data = torch.load(cache_path, map_location=device, weights_only=True)
                return _max_sizes_from_save_dict(data, device)
            except (pickle.UnpicklingError, TypeError):
                # Old cache was saved as TensorDict; load and re-save in plain dict
                legacy = torch.load(cache_path, map_location=device, weights_only=False)
                result = _max_sizes_from_save_dict(
                    _max_sizes_to_save_dict(legacy), device
                )
                if rank == 0:
                    torch.save(_max_sizes_to_save_dict(result), cache_path)
                return result
        result = _compute_max_mesh_sizes(
            dataloader,
            device,
            face_downsampling_ratio=face_downsampling_ratio,
            rank=rank,
        )
        if rank == 0:
            cache_subdir.mkdir(parents=True, exist_ok=True)
            torch.save(_max_sizes_to_save_dict(result), cache_path)
            logger.info("Cached max mesh sizes to %s", cache_path)
        return result
    return _compute_max_mesh_sizes(
        dataloader,
        device,
        face_downsampling_ratio=face_downsampling_ratio,
        rank=rank,
    )


def _structured_tensordict_to_airfrans_sample(data: TensorDict) -> AirFRANSSample:
    """Convert pipeline output (structured TensorDict from ToAirFRANSSampleStructure) to AirFRANSSample."""
    im = data["interior_mesh"]
    interior_mesh = Mesh(
        points=im["points"],
        cells=im["cells"],
        point_data=im["point_data"],
        global_data=im["global_data"],
    )

    bm_td = data["boundary_meshes"]
    boundary_meshes = TensorDict(
        {
            bc_name: Mesh(points=bc["points"], cells=bc["cells"])
            for bc_name, bc in bm_td.items()
        },
    )

    return AirFRANSSample(
        interior_mesh=interior_mesh,
        boundary_meshes=boundary_meshes,
        reference_lengths=data["reference_lengths"],
        dimensional_constants=data["dimensional_constants"],
    )


def _collate_single(samples: Sequence[tuple[TensorDict, dict[str, Any]]]) -> AirFRANSSample:
    """Collate for batch_size=1: convert structured TensorDict to AirFRANSSample."""
    data, _ = samples[0]
    return _structured_tensordict_to_airfrans_sample(data)


class AirFRANSDataSet:
    """
    Drop-in replacement for dataset.AirFRANSDataSet using the physicsnemo datapipe.

    Uses physicsnemo DataLoader (not torch DataLoader). Pipeline includes
    ToAirFRANSSampleStructure so output is converted to AirFRANSSample in the collate.
    """

    _config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")

    @classmethod
    def get_split_paths(
        cls,
        data_dir: Path,
        task: Literal["full", "scarce", "reynolds", "aoa"],
        split: Literal["train", "test"],
    ) -> list[Path]:
        """Same as dataset.AirFRANSDataSet.get_split_paths (reads manifest.json)."""
        return _OriginalAirFRANSDataSet.get_split_paths(data_dir, task, split)

    @classmethod
    def make_dataloader(
        cls,
        sample_paths: Sequence[Path],
        cache_dir: Path,
        *,
        world_size: int = 1,
        rank: int = 0,
        num_workers: int = 0,
        task: str = "full",
        split: Literal["train", "test"] = "train",
    ):
        """Build physicsnemo DataLoader for the given task/split (config from Hydra)."""
        import hydra
        from hydra import compose, initialize_config_dir
        from torch.utils.data.distributed import DistributedSampler

        from physicsnemo.datapipes import DataLoader as PhysicsnemoDataLoader
        from physicsnemo.datapipes import Dataset as PhysicsnemoDataset

        with initialize_config_dir(config_dir=cls._config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"task={task}",
                    f"split={split}",
                    "reader.pin_memory=false",
                ],
            )
        datapipe: PhysicsnemoDataset = hydra.utils.instantiate(cfg.dataset)
        sampler = DistributedSampler(
            datapipe,
            num_replicas=world_size,
            rank=rank,
        )
        return PhysicsnemoDataLoader(
            datapipe,
            batch_size=1,
            sampler=sampler,
            collate_fn=_collate_single,
        )

    @classmethod
    def preprocess(
        cls,
        sample_path: Path | None = None,
        *,
        split: Literal["train", "test"] = "test",
        index: int = 0,
        task: str = "full",
    ) -> AirFRANSSample:
        """Load one sample. If sample_path is provided and exists, use dataset preprocess; else load by split/index from datapipe."""
        if sample_path is not None and Path(sample_path).exists():
            return _OriginalAirFRANSDataSet.preprocess(sample_path)
        import hydra
        from hydra import compose, initialize_config_dir

        from physicsnemo.datapipes import Dataset as PhysicsnemoDataset

        with initialize_config_dir(config_dir=cls._config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[f"task={task}", f"split={split}", "reader.pin_memory=false"],
            )
        datapipe: PhysicsnemoDataset = hydra.utils.instantiate(cfg.dataset)
        data, _ = datapipe[index]
        return _structured_tensordict_to_airfrans_sample(data)

    @staticmethod
    def postprocess(
        pred_mesh: Mesh,
        true_mesh: Mesh,
        *,
        fields: Sequence[str | tuple[str, ...]] | None = None,
        show: bool = True,
        show_error: bool = True,
    ) -> Mesh:
        """Delegate to dataset.AirFRANSDataSet.postprocess."""
        return _OriginalAirFRANSDataSet.postprocess(
            pred_mesh, true_mesh, fields=fields, show=show, show_error=show_error
        )

    @staticmethod
    def visualize_output_distributions(
        output_dict: TensorDict,
        show: bool = True,
    ) -> None:
        """Delegate to dataset.AirFRANSDataSet.visualize_output_distributions."""
        _OriginalAirFRANSDataSet.visualize_output_distributions(output_dict, show=show)
