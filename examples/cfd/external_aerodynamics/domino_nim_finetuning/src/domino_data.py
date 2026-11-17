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

"""Data and normalization setup for the DoMINO NIM fine-tuning example."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from physicsnemo.datapipes.cae.domino_datapipe import (
    compute_scaling_factors,
    create_domino_dataset,
)
from physicsnemo.distributed import DistributedManager


def prepare_datasets(cfg: DictConfig):
    """Create training and validation datasets and preserve NIM scaling files."""
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    keys = ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]
    target_keys = []
    if cfg.model.model_type in ["volume", "combined"]:
        keys.extend(["volume_mesh_centers", "volume_fields"])
        target_keys.append("volume_fields")
    if cfg.model.model_type in ["surface", "combined"]:
        keys.extend(
            [
                "surface_mesh_centers",
                "surface_normals",
                "surface_areas",
                "surface_fields",
            ]
        )
        target_keys.append("surface_fields")

    references = []
    for parameter in cfg.variables.global_parameters.values():
        if parameter.type == "vector":
            references.extend(parameter.reference)
        else:
            references.append(parameter.reference)
    optional_keys = {
        "global_params_values": torch.tensor(references).reshape(-1, 1),
        "global_params_reference": torch.tensor(references).reshape(-1, 1),
    }

    if cfg.model.normalization == "min_max_scaling":
        factor_indices = (3, 2)  # Statistics are returned as mean, std, min, max.
    elif cfg.model.normalization == "mean_std_scaling":
        factor_indices = (0, 1)
    else:
        raise ValueError(f"Invalid normalization mode: {cfg.model.normalization}")

    output_dir = Path(cfg.project_dir)
    paths = {
        key: output_dir / key.replace("_fields", "_scaling_factors.npy")
        for key in target_keys
    }
    missing = [key for key, path in paths.items() if not path.exists()]
    factors = {key: np.load(path) for key, path in paths.items() if path.exists()}
    if missing:
        if cfg.data_processor.use_cache:
            raise FileNotFoundError(
                "Compute scaling factors on uncached training data before using cached data."
            )
        statistics = compute_scaling_factors(
            cfg=cfg,
            input_path=cfg.data.input_dir,
            target_keys=missing,
            max_samples=cfg.data.max_samples_for_statistics,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for key in missing:
            factors[key] = np.stack(
                [statistics[index][key].cpu().numpy() for index in factor_indices]
            )
            if DistributedManager().rank == 0:
                np.save(paths[key], factors[key])

    volume_factors = factors.get("volume_fields")
    surface_factors = factors.get("surface_fields")
    datasets = [
        create_domino_dataset(
            cfg,
            phase=phase,
            keys_to_read=keys,
            keys_to_read_if_available=optional_keys,
            vol_factors=volume_factors,
            surf_factors=surface_factors,
        )
        for phase in ["train", "val"]
    ]
    return datasets[0], datasets[1], volume_factors, surface_factors


def collate_domino_samples(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Combine the per-example batch dimensions without adding another one."""
    return {
        key: torch.cat([sample[key] for sample in samples], dim=0) for key in samples[0]
    }
