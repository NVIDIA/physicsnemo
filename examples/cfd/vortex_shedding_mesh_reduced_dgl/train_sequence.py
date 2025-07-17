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
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from constants import Constants
from physicsnemo.datapipes.gnn.vortex_shedding_re300_1000_dataset_dgl import (
    LatentDataset,
    VortexSheddingRe300To1000Dataset,
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.mesh_reduced.mesh_reduced import Mesh_Reduced
from physicsnemo.models.mesh_reduced.temporal_model import Sequence_Model

C = Constants()


class Sequence_Trainer:
    """Sequence trainer"""

    def __init__(
        self,
        wb,
        dist,
        produce_latents=True,
        Encoder=None,
        position_mesh=None,
        position_pivotal=None,
        rank_zero_logger=None,
    ):
        self.dist = dist
        dataset_train = LatentDataset(
            split="train",
            produce_latents=produce_latents,
            Encoder=Encoder,
            position_mesh=position_mesh,
            position_pivotal=position_pivotal,
            dist=dist,
        )

        dataset_test = LatentDataset(
            split="test",
            produce_latents=produce_latents,
            Encoder=Encoder,
            position_mesh=position_mesh,
            position_pivotal=position_pivotal,
            dist=dist,
        )

        self.dataloader = GraphDataLoader(
            dataset_train,
            batch_size=C.batch_size_sequence,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        self.dataloader_test = GraphDataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        self.dataset_graph_train = VortexSheddingRe300To1000Dataset(
            name="vortex_shedding_train", split="train"
        )

        self.dataset_graph_test = VortexSheddingRe300To1000Dataset(
            name="vortex_shedding_train", split="test"
        )

        self.dataloader_graph = GraphDataLoader(
            self.dataset_graph_train,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        self.dataloader_graph_test = GraphDataLoader(
            self.dataset_graph_test,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )
        self.model = Sequence_Model(C.sequence_dim, C.sequence_context_dim, dist)

        if C.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
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
            os.path.join(C.ckpt_sequence_path, C.ckpt_sequence_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def denormalize(self, sample):
        for j in range(sample.size()[0]):
            sample[j] = self.dataset_graph_train.denormalize(
                sample[j],
                self.dataset_graph_train.node_stats["node_mean"].to(self.dist.device),
                self.dataset_graph_train.node_stats["node_std"].to(self.dist.device),
            )
        return sample

    @torch.no_grad()
    def sample(
        self,
        z0,
        context,
        ground_trueth,
        true_latent,
        encoder,
        graph,
        position_mesh,
        position_pivotal,
    ):
        self.model.eval()
        x_samples = []
        z0 = z0.to(self.dist.device)
        context = context.to(self.dist.device)
        z_samples = self.model.sample(z0, 399, context)
        for i in range(401):
            z_sample = z_samples[0, i]
            z_sample = z_sample.reshape(256, 3)

            x_sample = encoder.decode(
                z_sample, graph.edata["x"], graph, position_mesh, position_pivotal
            )
            x_samples.append(x_sample.unsqueeze(0))
        x_samples = torch.cat(x_samples)
        x_samples = self.denormalize(x_samples)

        ground_trueth = self.denormalize(ground_trueth)

        loss_record_u = []
        loss_record_v = []
        loss_record_p = []

        for i in range(400):
            loss = self.criterion(
                ground_trueth[i + 1 : i + 2, :, 0], x_samples[i + 1 : i + 2, :, 0]
            )
            relative_error = (
                loss
                / self.criterion(
                    ground_trueth[i + 1 : i + 2, :, 0],
                    ground_trueth[i + 1 : i + 2, :, 0] * 0.0,
                ).detach()
            )
            loss_record_u.append(relative_error)
        relative_error_u = torch.mean(torch.tensor(loss_record_u))
        for i in range(400):
            loss = self.criterion(
                ground_trueth[i + 1 : i + 2, :, 1], x_samples[i + 1 : i + 2, :, 1]
            )
            relative_error = (
                loss
                / self.criterion(
                    ground_trueth[i + 1 : i + 2, :, 1],
                    ground_trueth[i + 1 : i + 2, :, 1] * 0.0,
                ).detach()
            )
            loss_record_v.append(relative_error)
        relative_error_v = torch.mean(torch.tensor(loss_record_v))
        for i in range(400):
            loss = self.criterion(
                ground_trueth[i + 1 : i + 2, :, 2], x_samples[i + 1 : i + 2, :, 2]
            )
            relative_error = (
                loss
                / self.criterion(
                    ground_trueth[i + 1 : i + 2, :, 2],
                    ground_trueth[i + 1 : i + 2, :, 2] * 0.0,
                ).detach()
            )
            loss_record_p.append(relative_error)
        relative_error_p = torch.mean(torch.tensor(loss_record_p))

        return x_samples, relative_error_u, relative_error_v, relative_error_p

    def forward(self, z, context=None):
        with autocast(enabled=C.amp):
            prediction = self.model(z, context)
            loss = self.criterion(z[:, 1:], prediction[:, :-1])
            relative_error = torch.sqrt(
                loss / self.criterion(z[:, 1:], z[:, 1:] * 0.0)
            ).detach()
            return loss, relative_error

    def train(self, z, context):
        z = z.to(self.dist.device)
        context = context.to(self.dist.device)
        self.optimizer.zero_grad()
        loss, relative_error = self.forward(z, context)
        self.backward(loss)
        self.scheduler.step()
        return loss, relative_error

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
        os.makedirs(C.ckpt_sequence_path, exist_ok=True)
        with open(
            os.path.join(
                C.ckpt_sequence_path, C.ckpt_sequence_name.replace(".pt", ".json")
            ),
            "w",
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
        produce_latents=C.produce_latents,
        Encoder=Encoder,
        position_mesh=position_mesh,
        position_pivotal=position_pivotal,
        rank_zero_logger=rank_zero_logger,
    )
    start = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(trainer.epoch_init, C.epochs_sequence):
        n_batch = 0.0
        loss_total = 0.0
        for lc in tqdm(trainer.dataloader):
            loss, relative_error = trainer.train(lc[0], lc[1])
            loss_total = loss_total + loss
            n_batch = n_batch + 1
        avg_loss = loss_total / n_batch
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {avg_loss:10.3e}, relative_error: {relative_error:10.3e},time per epoch: {(time.time()-start):10.3e}"
        )
        wb.log({"loss": loss.detach().cpu()})

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and epoch % 5000 == 0:
            save_checkpoint(
                os.path.join(C.ckpt_sequence_path, C.ckpt_sequence_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")
