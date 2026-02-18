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
This code defines a distributed pipeline for training the DoMINO model on
CFD datasets. It includes the computation of scaling factors, instantiating
the DoMINO model and datapipe, automatically loading the most recent checkpoint,
training the model in parallel using DistributedDataParallel across multiple
GPUs, calculating the loss and updating model parameters using mixed precision.
This is a common recipe that enables training of combined models for surface and
volume as well either of them separately. Validation is also conducted every epoch,
where predictions are compared against ground truth values. The code logs training
and validation metrics to TensorBoard. The train tab in config.yaml can be used to
specify batch size, number of epochs and other training parameters.
"""

import time
import os
import re
import torch
import torchinfo

from typing import Literal, Any


import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# This will set up the cupy-ecosystem and pytorch to share memory pools
from physicsnemo.utils.memory import unified_gpu_memory


import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from nvtx import annotate as nvtx_annotate
import torch.cuda.nvtx as nvtx


from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    compute_scaling_factors,
    create_domino_dataset,
)
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino.utils import *

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time

from utils import (
    ScalingFactors,
    get_keys_to_read,
    coordinate_distributed_environment,
    load_scaling_factors,
)


from physicsnemo.utils.profiling import profile, Profiler


def benchmark_io_epoch(
    dataloader,
    logger,
    gpu_handle,
    epoch_index,
    device,
):
    dist = DistributedManager()

    # If you tell the dataloader the indices in advance, it will preload
    # and pre-preprocess data
    # dataloader.set_indices(indices)

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    for i_batch, sample_batched in enumerate(dataloader):
        # Gather data and report
        elapsed_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_memory_used = gpu_end_info.used / (1024**3)
        gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

        logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
        logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb\n"
        logging_string += f"  GPU memory delta: {gpu_memory_delta:.3f} Gb\n"
        logging_string += f"  Time taken: {elapsed_time:.2f} seconds\n"
        logger.info(logging_string)
        gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)

    return


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize NVML
    nvmlInit()

    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    model_type = cfg.model.model_type

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    ################################
    # Get scaling factors
    ################################
    vol_factors, surf_factors = load_scaling_factors(cfg)

    keys_to_read, keys_to_read_if_available = get_keys_to_read(
        cfg, model_type, get_ground_truth=True
    )

    domain_mesh, data_mesh, placements = coordinate_distributed_environment(cfg)

    train_dataset = create_domino_dataset(
        cfg,
        phase="train",
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
        device_mesh=domain_mesh,
        placements=placements,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=data_mesh.size(), rank=data_mesh.get_local_rank()
    )

    for epoch in range(0, cfg.train.epochs):
        start_time = time.perf_counter()
        logger.info(f"Device {dist.device}, epoch {epoch}:")

        train_sampler.set_epoch(epoch)

        train_dataset.dataset.set_indices(list(train_sampler))

        epoch_start_time = time.perf_counter()
        with Profiler():
            benchmark_io_epoch(
                dataloader=train_dataset,
                logger=logger,
                gpu_handle=gpu_handle,
                epoch_index=epoch,
                device=dist.device,
            )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist.device}, Epoch {epoch} took {epoch_end_time - epoch_start_time:.3f} seconds"
        )


if __name__ == "__main__":
    main()
