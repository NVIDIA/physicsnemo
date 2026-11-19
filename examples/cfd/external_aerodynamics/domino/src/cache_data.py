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

"""Cache full DoMINO preprocessing outputs for later training."""

import time
from pathlib import Path

import hydra
import numpy as np
import torch
from cache_utils import cache_filename, create_cache_dataset
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Write unbatched CPU arrays using the current datapipe and statistics."""
    if not cfg.data_processor.use_cache:
        raise ValueError("Cache must be enabled for cache processing!")
    DistributedManager.initialize()
    dist = DistributedManager()
    dataset = create_cache_dataset(cfg)
    output_dir = Path(cfg.data_processor.cached_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Strided indices avoid DistributedSampler's padding/duplicate writes.
    for index in range(dist.rank, len(dataset), dist.world_size):
        filename = cache_filename(dataset, index)
        output_file = output_dir / filename
        if output_file.exists():
            print(f"Rank {dist.rank}: Skipping {filename} - cache exists")
            continue
        start_time = time.perf_counter()
        sample = dataset[index]
        # The datapipe already supplies a batch dimension; the cache stores
        # individual examples as NumPy arrays, including expanded neighbors.
        processed_data = {key: value[0].cpu().numpy() for key, value in sample.items()}
        np.save(output_file, processed_data)
        print(
            f"Rank {dist.rank}: Completed {filename} in {time.perf_counter() - start_time:.2f}s"
        )
    if dist.world_size > 1:
        torch.distributed.barrier()
    if dist.rank == 0:
        print("All processing complete!")


if __name__ == "__main__":
    main()
