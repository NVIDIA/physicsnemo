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
This is a standalone script for benchmarking and testing the Transolver
datapipe in surface or volume mode.
"""

from pathlib import Path

import time
import os
import re
import torch

import numpy as np

from typing import Literal, Any


import hydra
from omegaconf import DictConfig, OmegaConf


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.transolver_datapipe import (
    create_transolver_dataset,
)


from physicsnemo.utils.profiling import profile, Profiler


@profile
def main(cfg: DictConfig):
    """Main training function

    Args:
        cfg: Hydra configuration object
    """

    DistributedManager.initialize()

    # Set up distributed training
    dist_manager = DistributedManager()

    # Set up logging
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Load the normalization file:
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    if cfg.data.mode == "surface":
        norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    elif cfg.data.mode == "volume":
        norm_file = str(Path(norm_dir) / "volume_fields_normalization.npz")

    norm_data = np.load(norm_file)
    norm_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
        "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
    }
    # Training dataset

    train_dataloader = create_transolver_dataset(
        cfg.data,
        phase="train",
        scaling_factors=norm_factors,
    )

    # Validation dataset

    val_dataloader = create_transolver_dataset(
        cfg.data,
        phase="val",
        scaling_factors=norm_factors,
    )

    num_replicas = dist_manager.world_size
    data_rank = dist_manager.rank

    # Set up distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataloader,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataloader,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=False,  # No shuffling for validation
        drop_last=True,
    )

    # Training loop
    logger.info("Starting IO benchmark...")
    for epoch in range(1):
        # Set the epoch in the samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_dataloader.dataset.set_indices(list(train_sampler))
        val_dataloader.dataset.set_indices(list(val_sampler))

        start_time = time.time()
        # Training phase
        start = time.time()
        with Profiler():
            for i_batch, data in enumerate(train_dataloader):
                print(f"Train {i_batch} elapsed time: {time.time() - start}")
                start = time.time()

        end_time = time.time()
        train_duration = end_time - start_time

        # Log epoch results
        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] [duration: {train_duration:.2f}s]"
        )

    logger.info("Benchmark completed!")


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
def launch(cfg: DictConfig):
    """Launch training with hydra configuration

    Args:
        cfg: Hydra configuration object
    """

    # If you want to use `line_profiler` or PyTorch's profiler, enable them here.

    profiler = Profiler()
    if cfg.profile:
        profiler.enable("torch")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
