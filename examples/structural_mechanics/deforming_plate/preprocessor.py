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

import os
import torch
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from physicsnemo.distributed.manager import DistributedManager

from deforming_plate_dataset import DeformingPlateDataset
from helpers import add_world_edges


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Set up output directory
    output_dir = to_absolute_path(cfg.preprocess_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = DeformingPlateDataset(
        name="deforming_plate_train",
        data_dir=to_absolute_path(cfg.data_dir),
        split="train",
        num_samples=cfg.num_training_samples,
        num_steps=cfg.num_training_time_steps,
    )

    num_samples = cfg.num_training_samples
    num_steps = cfg.num_training_time_steps

    # Split the samples among ranks
    per_rank = num_samples // dist.world_size
    start = dist.rank * per_rank
    end = (
        (dist.rank + 1) * per_rank if dist.rank != dist.world_size - 1 else num_samples
    )

    for sample_idx in tqdm(range(start, end), desc=f"Rank {dist.rank} preprocessing"):
        sample_file = os.path.join(output_dir, f"sample_{sample_idx:05d}.pt")
        if os.path.exists(sample_file):
            continue  # Skip if already processed

        sample_data = []
        for t in range(num_steps - 1):
            idx = sample_idx * (num_steps - 1) + t
            graph = dataset[idx].to(dist.device)
            graph, mesh_edge_features, world_edge_features = add_world_edges(graph)
            sample_data.append(
                {
                    "graph": graph,
                    "mesh_edge_features": mesh_edge_features,
                    "world_edge_features": world_edge_features,
                }
            )
        torch.save(sample_data, sample_file)
    print(f"Rank {dist.rank} finished processing samples {start} to {end - 1}")


if __name__ == "__main__":
    main()
