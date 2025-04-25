# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
This script processes DoMINODataPipe format files into cached versions
for faster loading during training. It processes files in parallel and can be
configured through config.yaml in the data_processing tab.
"""

from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    compute_scaling_factors,
)
import hydra
import time
import numpy as np
import os
from pathlib import Path
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from physicsnemo.distributed import DistributedManager


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    compute_scaling_factors(cfg, cfg.data_processor.output_dir, use_cache=False)
    assert cfg.data_processor.use_cache, "Cache must be enabled for cache processing!"
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    vol_save_path = os.path.join(cfg.project_dir, "volume_scaling_factors.npy")

    surf_save_path = os.path.join(cfg.project_dir, "surface_scaling_factors.npy")
    if os.path.exists(vol_save_path):
        vol_factors = np.load(vol_save_path)
    else:
        vol_factors = None
    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path)
    else:
        surf_factors = None

    # Set up variables based on model type
    model_type = cfg.model.model_type
    volume_variable_names = []
    surface_variable_names = []

    if model_type in ["volume", "combined"]:
        volume_variable_names = list(cfg.variables.volume.solution.keys())
    if model_type in ["surface", "combined"]:
        surface_variable_names = list(cfg.variables.surface.solution.keys())

    # Create dataset once
    dataset = DoMINODataPipe(
        data_path=cfg.data_processor.output_dir,  # Caching comes after data processing
        phase="train",  # Phase doesn't matter for caching
        grid_resolution=cfg.model.interp_res,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        normalize_coordinates=True,
        sampling=False,
        sample_in_bbox=True,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        positional_encoding=cfg.model.positional_encoding,
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        scaling_type=cfg.model.normalization,
        model_type=cfg.model.model_type,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        num_surface_neighbors=cfg.model.num_surface_neighbors,
        resample_surfaces=False,
        for_caching=True,
        deterministic_seed=True,
    )

    # Create output directory on rank 0
    output_dir = Path(cfg.data_processor.cached_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Wait for directory creation
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        shuffle=False,  # No need to shuffle for preprocessing
    )

    # Create dataloader with distributed sampler
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,  # Process one at a time for caching
        num_workers=0,  # Must be 0 due to GPU operations in dataset
    )

    # Process and cache files
    for _, sample in enumerate(dataloader):
        filename = sample["filename"][
            0
        ]  # batch size 1, we can just pull out the filename
        output_file = output_dir / f"{filename}_cached.npy"

        if output_file.exists():
            print(f"Rank {dist.rank}: Skipping {filename} - cache exists")
            continue

        print(f"Rank {dist.rank}: Processing {filename}")
        start_time = time.time()

        try:
            # Remove batch dimension since we're processing one at a time
            processed_data = {k: v[0] for k, v in sample.items()}
            if cfg.model.model_type == "volume" or cfg.model.model_type == "combined":
                print(
                    f"{filename}: volume min/max: {torch.amin(processed_data['volume_fields'], 0)}, {torch.amax(processed_data['volume_fields'], 0)}"
                )
            if cfg.model.model_type == "surface" or cfg.model.model_type == "combined":
                print(
                    f"{filename}: surface min/max: {torch.amin(processed_data['surface_fields'], 0)}, {torch.amax(processed_data['surface_fields'], 0)}"
                )
            np.save(output_file, processed_data)
            print(
                f"Rank {dist.rank}: Completed {filename} in {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            print(f"Rank {dist.rank}: Error processing {filename}: {str(e)}")

    # Wait for all processes to complete
    if dist.world_size > 1:
        torch.distributed.barrier()

    if dist.rank == 0:
        print("All processing complete!")


if __name__ == "__main__":
    main()
