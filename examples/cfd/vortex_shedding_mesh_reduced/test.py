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

import time

import numpy as np
import torch
import wandb as wb
from tqdm import tqdm

from constants import Constants
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from train import Mesh_ReducedTrainer

C = Constants()

if __name__ == "__main__":
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    # initialize loggers
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Testing started...")
    position_mesh = torch.from_numpy(np.loadtxt(C.mesh_dir)).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(C.pivotal_dir)).to(dist.device)
    loss_total = 0
    relative_error_total = 0

    for graph in tqdm(trainer.dataloader_test):
        loss, relative_error, relative_error_s = trainer.test(
            graph, position_mesh, position_pivotal
        )
        loss_total = loss_total + loss
        relative_error_total = relative_error_total + relative_error
    n = len(trainer.dataloader_test)
    avg_relative_error = relative_error_total / n
    avg_loss = loss_total / n
    rank_zero_logger.info(
        f"avg_loss: {avg_loss:10.3e}, avg_relative_error: {avg_relative_error:10.3e},time per epoch: {(time.time()-start):10.3e}"
    )
    print(relative_error_s)

    rank_zero_logger.info("Testing completed!")
