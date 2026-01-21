# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import logging
import os
import time

import numpy as np
import torch
import wandb as wb

from torch_geometric.loader import DataLoader as PyGDataLoader

from torch.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from constants import Constants
from physicsnemo.datapipes.gnn.vortex_shedding_re300_1000_dataset import (
    VortexSheddingRe300To1000Dataset,
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced

C = Constants()

logging.basicConfig(level=logging.INFO)


class Mesh_ReducedTrainer:
    def __init__(self, wb, dist, rank_zero_logger):
        self.dist = dist
        dataset_train = VortexSheddingRe300To1000Dataset(
            name="vortex_shedding_train", split="train"
        )

        dataset_test = VortexSheddingRe300To1000Dataset(
            name="vortex_shedding_train", split="test"
        )

        sampler = DistributedSampler(
            dataset_train,
            shuffle=True,
            drop_last=True,
            num_replicas=dist.world_size,
            rank=dist.rank,
        )

        self.dataloader = PyGDataLoader(
            dataset_train,
            batch_size=C.batch_size,
            sampler=sampler,
            pin_memory=True,
        )

        self.dataloader_test = PyGDataLoader(
            dataset_test,
            batch_size=C.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        self.model = Mesh_Reduced(
            C.num_input_features, C.num_edge_features, C.num_output_features
        )

        if C.jit:
            self.model = torch.compile(self.model.to(dist.device))
        else:
            self.model = self.model.to(dist.device)
        if C.watch_model and not C.jit and dist.rank == 0:
            wb.watch(self.model)
        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        # instantiate loss, optimizer, and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: C.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def forward(self, graph, position_mesh, position_pivotal):
        with autocast("cuda", enabled=C.amp):
            z = self.model.encode(
                graph.x,
                graph.edge_attr,
                graph,
                position_mesh,
                position_pivotal,
            )
            x = self.model.decode(
                z, graph.edge_attr, graph, position_mesh, position_pivotal
            )
            loss = self.criterion(x, graph.x)
            return loss

    def train(self, graph, position_mesh, position_pivotal):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph, position_mesh, position_pivotal)
        self.backward(loss)
        self.scheduler.step()
        return loss

    @torch.no_grad()
    def test(self, graph, position_mesh, position_pivotal):
        graph = graph.to(self.dist.device)
        with autocast("cuda", enabled=C.amp):
            z = self.model.encode(
                graph.x,
                graph.edge_attr,
                graph,
                position_mesh,
                position_pivotal,
            )
            x = self.model.decode(
                z, graph.edge_attr, graph, position_mesh, position_pivotal
            )
            loss = self.criterion(x, graph.x)

            relative_error = loss / self.criterion(graph.x, graph.x * 0.0).detach()
            relative_error_s_record = []
            for i in range(C.num_input_features):
                loss_s = self.criterion(x[:, i], graph.x[:, i])
                relative_error_s = (
                    loss_s / self.criterion(graph.x[:, i], graph.x[:, i] * 0.0).detach()
                )
                relative_error_s_record.append(relative_error_s)

        return loss, relative_error, relative_error_s_record

    def backward(self, loss):
        # backward pass
        if C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # save constants to JSON file
    if dist.rank == 0:
        os.makedirs(C.ckpt_path, exist_ok=True)
        with open(
            os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".json")), "w"
        ) as json_file:
            json_file.write(C.model_dump_json(indent=4))

    # initialize loggers
    initialize_wandb(
        project="PhysicsNeMo-Launch",
        entity="PhysicsNeMo",
        name="Vortex_Shedding-Training",
        group="Vortex_Shedding-DDP-Group",
        mode=C.wandb_mode,
    )  # Wandb logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    position_mesh = torch.from_numpy(np.loadtxt(C.mesh_dir)).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(C.pivotal_dir)).to(dist.device)
    for epoch in range(trainer.epoch_init, C.epochs):
        for graph in tqdm(trainer.dataloader):
            loss = trainer.train(graph, position_mesh, position_pivotal)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time() - start):10.3e}"
        )
        wb.log({"loss": loss.detach().cpu()})

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and epoch % 100 == 0:
            save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")
