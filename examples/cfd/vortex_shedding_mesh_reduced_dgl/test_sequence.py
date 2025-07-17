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
import time

import numpy as np
import torch
import wandb as wb
from torch.cuda.amp import GradScaler

from constants import Constants
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced
from train_sequence import Sequence_Trainer

C = Constants()

if __name__ == "__main__":
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    position_mesh = torch.from_numpy(np.loadtxt(C.mesh_dir)).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(C.pivotal_dir)).to(dist.device)
    # Load Graph Encoder
    Encoder = Mesh_Reduced(
        C.num_input_features, C.num_edge_features, C.num_output_features
    )
    Encoder = Encoder.to(dist.device)
    _ = load_checkpoint(
        os.path.join(C.ckpt_path, C.ckpt_name),
        models=Encoder,
        scaler=GradScaler(),
        device=dist.device,
    )

    trainer = Sequence_Trainer(
        wb,
        dist,
        produce_latents=False,
        Encoder=Encoder,
        position_mesh=position_mesh,
        position_pivotal=position_pivotal,
        rank_zero_logger=rank_zero_logger,
    )
    trainer.model.eval()
    start = time.time()
    rank_zero_logger.info("Testing started...")
    for graph in trainer.dataloader_graph_test:
        g = graph.to(dist.device)

        break
    ground_trueth = trainer.dataset_graph_test.solution_states

    i = 0
    relative_error_sum_u = 0
    relative_error_sum_v = 0
    relative_error_sum_p = 0

    for lc in trainer.dataloader_test:
        ground = ground_trueth[i].to(dist.device)

        graph.ndata["x"]
        samples, relative_error_u, relative_error_v, relative_error_p = trainer.sample(
            lc[0][:, 0:2],
            lc[1],
            ground,
            lc[0],
            Encoder,
            g,
            position_mesh,
            position_pivotal,
        )
        relative_error_sum_u = relative_error_sum_u + relative_error_u
        relative_error_sum_v = relative_error_sum_v + relative_error_v
        relative_error_sum_p = relative_error_sum_p + relative_error_p
        i = i + 1
    relative_error_mean_u = relative_error_sum_u / i
    relative_error_mean_v = relative_error_sum_v / i
    relative_error_mean_p = relative_error_sum_p / i

    # avg_loss = loss_total/n_batch
    rank_zero_logger.info(
        f"relative_error_mean_u: {relative_error_mean_u:10.3e},relative_error_mean_v: {relative_error_mean_v:10.3e},relative_error_mean_p: {relative_error_mean_p:10.3e},\\\
            time cost: {(time.time()-start):10.3e}"
    )
    # wb.log({"loss": loss.detach().cpu()})

    rank_zero_logger.info("Sampling completed!")
