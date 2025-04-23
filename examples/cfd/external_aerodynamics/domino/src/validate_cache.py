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
    CachedDoMINODataset,
    DoMINODataPipe,
)
import hydra
import numpy as np
import os
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from physicsnemo.distributed import DistributedManager


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    assert cfg.data_processor.use_cache, "Cache must be enabled to be validated!"
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    vol_save_path = os.path.join(
        cfg.project_dir, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        cfg.project_dir, "surface_scaling_factors.npy"
    )
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
    dataset_orig = DoMINODataPipe(
        data_path=cfg.data_processor.output_dir,  # Caching comes after data processing
        phase="train",  # Phase doesn't matter for caching
        grid_resolution=cfg.model.interp_res,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        normalize_coordinates=True,
        sampling=True,
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
        for_caching=False,
        deterministic_seed=True,
    )

    dataset_cached = CachedDoMINODataset(
        data_path=cfg.data_processor.cached_dir,
        phase="train",
        sampling=True,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        model_type=cfg.model.model_type,
        deterministic_seed=True,
    )

    # Wait for directory creation
    if dist.world_size > 1:
        torch.distributed.barrier()

    def get_dataloader(dataset, world_size, rank):
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,  # Process one at a time for caching
            num_workers=0,  # Must be 0 due to GPU operations in dataset
        )

    dataloader_orig = get_dataloader(dataset_orig, dist.world_size, dist.rank)
    dataloader_cached = get_dataloader(dataset_cached, dist.world_size, dist.rank)

    # Process and cache files
    for _, (sample_orig, sample_cached) in enumerate(
        zip(dataloader_orig, dataloader_cached)
    ):
        filename_orig = sample_orig["filename"][0]
        filename_cached = sample_cached["filename"][0]
        mismatched = False
        if filename_orig != filename_cached:
            print(
                f"Rank {dist.rank}: Mismatched filenames: {filename_orig} != {filename_cached}"
            )
            mismatched = True
        for k, v in sample_orig.items():
            if k in ["filename"]:
                continue
            if k not in sample_cached:
                print(f"Rank {dist.rank}: Key {k} missing from cached sample")
                mismatched = True
            elif not torch.allclose(v, sample_cached[k]):
                print(f"Rank {dist.rank}: Mismatched values for key {k}")
                # Get boolean mask of mismatches
                mismatches = v != sample_cached[k]
                # Get indices where values mismatch
                mismatch_indices = torch.nonzero(mismatches, as_tuple=False)
                print(
                    f"  Found {len(mismatch_indices)} mismatches, of {v.numel()} total values"
                )
                print(f" Tensor shape: {v.shape}, vs {sample_cached[k].shape}")
                # Get the actual values at those positions
                for idx in mismatch_indices[:5]:  # Show only first 5 mismatches
                    idx_tuple = tuple(
                        idx.tolist()
                    )  # Convert index tensor to tuple for indexing
                    val1 = v[idx_tuple].item()
                    val2 = sample_cached[k][idx_tuple].item()
                    print(f"  Index {idx_tuple}: {val1} vs {val2}")
                mismatched = True
        if mismatched:
            print(f"FAILED Rank {dist.rank}: {filename_orig}")
        else:
            print(f"Rank {dist.rank}: {filename_orig} validated")

    # Wait for all processes to complete
    if dist.world_size > 1:
        torch.distributed.barrier()

    if dist.rank == 0:
        print("All processing complete!")


if __name__ == "__main__":
    main()
