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

"""Validate cached DoMINO outputs against full, unsampled preprocessing."""

from pathlib import Path

import hydra
import torch
from cache_utils import cache_filename, create_cache_dataset
from omegaconf import DictConfig

from physicsnemo.datapipes.cae.domino_datapipe import CachedDoMINODataset
from physicsnemo.distributed import DistributedManager


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compare every tensor by filename, failing on missing or invalid data."""
    if not cfg.data_processor.use_cache:
        raise ValueError("Cache must be enabled to be validated!")
    DistributedManager.initialize()
    dist = DistributedManager()
    original = create_cache_dataset(cfg)
    cached = CachedDoMINODataset(cfg.data_processor.cached_dir, sampling=False)
    # CachedDoMINODataset shuffles its indices, so dataset positions cannot
    # establish correspondence with the original CAEDataset file ordering.
    cached_indices = {
        Path(cached.filenames[file_index]).name: index
        for index, file_index in enumerate(cached.indices)
    }
    for index in range(dist.rank, len(original), dist.world_size):
        filename = cache_filename(original, index)
        if filename not in cached_indices:
            raise FileNotFoundError(f"Cached sample missing: {filename}")
        sample_original = original[index]
        sample_cached = cached[cached_indices[filename]]
        for key, value in sample_original.items():
            if key not in sample_cached:
                raise ValueError(f"{filename}: key {key} missing from cached sample")
            expected = value[0].cpu()
            actual = sample_cached[key]
            if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
                raise ValueError(f"{filename}: mismatched shape for key {key}")
            if not torch.allclose(expected, actual):
                raise ValueError(f"{filename}: mismatched values for key {key}")
        print(f"Rank {dist.rank}: {filename} validated")
    if dist.world_size > 1:
        torch.distributed.barrier()
    if dist.rank == 0:
        print("All processing complete!")


if __name__ == "__main__":
    main()
