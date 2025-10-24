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

import logging
import time

from dgl.dataloading import GraphDataLoader

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from loggers import CompositeLogger, ExperimentLogger, get_gpu_info, init_python_logging


logger = logging.getLogger("lmgn")

# Experiment logger will be set later during initialization.
elogger: ExperimentLogger = None


class MGNTrainer:
    def __init__(self, cfg: DictConfig):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()

        self.dt = cfg.data.train.dt
        self.dim = cfg.dim

        self.amp = cfg.amp.enabled

        # MGN with recompute_activation currently supports only SiLU activation function.
        mlp_act = cfg.model.mlp_activation_fn
        if cfg.model.recompute_activation and mlp_act.lower() != "silu":
            raise ValueError(
                f"recompute_activation only supports SiLU activation function, "
                f"but got {mlp_act}. Please either set activation='silu' "
                f"or disable recompute_activation."
            )

        # instantiate dataset
        logger.info("Loading the training dataset...")
        self.dataset = instantiate(cfg.data.train)
        logger.info(f"Using {len(self.dataset)} training samples.")

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            **cfg.train.dataloader,
            use_ddp=self.dist.world_size > 1,
        )

        # instantiate the model
        logger.info("Creating the model...")
        # instantiate the model
        self.model = instantiate(cfg.model)

        if cfg.compile.enabled:
            self.model = torch.compile(self.model, **cfg.compile.args).to(
                self.dist.device
            )
        else:
            self.model = self.model.to(self.dist.device)
            elogger.watch_model(self.model)

        # distributed data parallel for multi-node training
        if self.dist.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate loss
        self.criterion = instantiate(cfg.loss)

        # instantiate optimizer, and scheduler
        self.optimizer = instantiate(cfg.optimizer, self.model.parameters())

        num_iterations = cfg.train.epochs * len(self.dataloader)
        lrs_cfg = cfg.lr_scheduler
        lrs_with_num_iter = {
            "torch.optim.lr_scheduler.CosineAnnealingLR": "T_max",
            "torch.optim.lr_scheduler.OneCycleLR": "total_steps",
        }
        if (num_iter_key := lrs_with_num_iter.get(lrs_cfg._target_)) is not None:
            if lrs_cfg[num_iter_key] is None:
                lrs_cfg[num_iter_key] = num_iterations
        self.scheduler = instantiate(cfg.lr_scheduler, self.optimizer)

        self.scaler = GradScaler()

        # load checkpoint
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        self.epoch_init += 1

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss_pos, loss_vel, loss_acc, loss_acc_norm = self.forward(graph)
        self.backward(loss_acc_norm)
        self.scheduler.step()
        loss = loss_acc + loss_vel + loss_pos
        return {
            "loss": loss.item(),
            "loss_pos": loss_pos.item(),
            "loss_vel": loss_vel.item(),
            "loss_acc": loss_acc.item(),
            "loss_acc_norm": loss_acc_norm.item(),
        }

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.amp):
            gt_pos, gt_vel, gt_acc = self.dataset.unpack_targets(graph)
            # Predict the acceleration using normalized inputs and targets.
            pred_acc = self.model(graph.ndata["x"], graph.edata["x"], graph)
            mask = graph.ndata["mask"].unsqueeze(-1)
            num_nz = mask.sum() * self.dim
            loss_acc_norm = mask * self.criterion(pred_acc, gt_acc)
            loss_acc_norm = loss_acc_norm.sum() / num_nz

            with torch.no_grad():
                pos, vel, _ = self.dataset.unpack_inputs(graph)
                # Use the integrator to get the next position and velocity.
                pred_pos, pred_vel = self.dataset.time_integrator(
                    position=pos,
                    velocity=vel[-1],
                    acceleration=pred_acc,
                    dt=self.dt,
                    denormalize=True,
                )

                # Position loss.
                loss_pos = mask * self.criterion(pred_pos, gt_pos)
                loss_pos = loss_pos.sum() / num_nz
                # loss_vel and loss_acc are denormalized.
                loss_vel = mask * self.criterion(
                    pred_vel, self.dataset.denormalize_velocity(gt_vel)
                )
                loss_vel = loss_vel.sum() / num_nz

                loss_acc = mask * self.criterion(
                    self.dataset.denormalize_acceleration(pred_acc),
                    self.dataset.denormalize_acceleration(gt_acc),
                )
                loss_acc = loss_acc.sum() / num_nz

            return loss_pos, loss_vel, loss_acc, loss_acc_norm

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    init_python_logging(cfg, dist.rank)
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    logger.info(get_gpu_info())

    # Initialize loggers.
    global elogger
    elogger = CompositeLogger(cfg)

    trainer = MGNTrainer(cfg)
    start = time.time()
    logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.train.epochs + 1):
        epoch_losses = {}
        for graph in trainer.dataloader:
            losses = trainer.train(graph)
            for k, l in losses.items():
                epoch_losses.setdefault(k, []).append(l)

        mean_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

        last_lr = trainer.scheduler.get_last_lr()[0]
        logger.info(
            f"epoch: {epoch:5,}, loss: {mean_losses['loss']:10.3e}, "
            f"position loss: {mean_losses['loss_pos']:10.3e}, "
            f"velocity loss: {mean_losses['loss_vel']:10.3e}, "
            f"accel loss: {mean_losses['loss_acc']:10.3e}, "
            f"accel loss (norm): {mean_losses['loss_acc_norm']:10.3e}, "
            f"lr: {last_lr:10.3e}, "
            f"time per epoch: {(time.time() - start):10.3e}"
        )
        elogger.log(mean_losses, epoch)
        elogger.log_scalar("lr", last_lr, epoch)

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and epoch % cfg.train.checkpoint_save_freq == 0:
            save_checkpoint(
                cfg.output,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    logger.info("Training completed!")
    elogger.close()


if __name__ == "__main__":
    main()
