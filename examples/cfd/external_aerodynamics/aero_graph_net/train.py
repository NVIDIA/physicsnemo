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

from collections import defaultdict
from functools import partial
import logging
import time
from typing import Mapping

import hydra
from hydra.utils import instantiate, to_absolute_path

from dgl.dataloading import GraphDataLoader

from omegaconf import DictConfig, OmegaConf

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from loggers import CompositeLogger, ExperimentLogger, init_python_logging
from utils import batch_as_dict


logger = logging.getLogger("agnet")

# Experiment logger will be set later during initialization.
elogger: ExperimentLogger = None


class MGNTrainer:
    def __init__(self, cfg: DictConfig):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()

        # instantiate training dataset
        logger.info("Loading the training dataset...")
        self.dataset = instantiate(cfg.data.train)
        logger.info(f"Using {len(self.dataset)} training samples.")

        # instantiate validation dataset
        logger.info("Loading the validation dataset...")
        self.validation_dataset = instantiate(cfg.data.val)
        logger.info(f"Using {len(self.validation_dataset)} validation samples.")

        logger.info("Creating the dataloaders...")
        # instantiate training dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            **cfg.train.dataloader,
            use_ddp=self.dist.world_size > 1,
        )

        # instantiate validation dataloader
        self.validation_dataloader = GraphDataLoader(
            self.validation_dataset,
            **cfg.val.dataloader,
        )

        logger.info("Creating the model...")
        # instantiate the model
        self.model = instantiate(cfg.model)

        if cfg.compile.enabled:
            self.model = torch.compile(self.model, **cfg.compile.args).to(
                self.dist.device
            )
        else:
            self.model = self.model.to(self.dist.device)

        # distributed data parallel for multi-GPU/multi-node training
        if self.dist.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )
        # Set the original model getter to simplify access.
        assert not hasattr(self.model, "model")
        type(self.model).model = (
            (lambda m: m.module) if self.dist.distributed else (lambda m: m)
        )

        # enable train mode
        self.model.train()

        # instantiate losses.
        self.loss = instantiate(cfg.loss)

        # instantiate optimizer, and scheduler
        self.optimizer = instantiate(cfg.optimizer, self.model.parameters())
        self.scheduler = instantiate(cfg.lr_scheduler, self.optimizer)

        self.scaler = instantiate(cfg.amp.scaler)
        self.autocast = partial(
            torch.cuda.amp.autocast,
            enabled=cfg.amp.enabled,
            dtype=hydra.utils.get_object(cfg.amp.autocast.dtype),
        )

        # load checkpoint
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=self.model.model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.visualizers = instantiate(cfg.visualizers)

    def train(self, batch: Mapping[str, Tensor]):
        self.optimizer.zero_grad()
        losses = self.forward(batch)
        self.backward(losses["total"])
        self.scheduler.step()
        return losses

    def forward(self, batch):
        # forward pass
        batch = dict(batch)
        graph = batch.pop("graph")
        with self.autocast():
            pred = batch_as_dict(
                self.model(graph.ndata["x"], graph.edata["x"], graph, **batch)
            )
            # Graph data (e.g. p and WSS) loss.
            graph_loss = self.loss.graph(pred["graph"], graph.ndata["y"])
            losses = {"graph": graph_loss}
            # Compute C_d loss, if requested.
            if (pred_c_d := pred.get("c_d")) is not None:
                c_d_loss = self.loss.c_d(pred_c_d, batch["c_d"])
                losses["c_d"] = c_d_loss
            # Get total loss and detach intermediate losses.
            total_loss = sum(losses.values())
            losses = {k: v.detach() for k, v in losses.items()}
            losses["total"] = total_loss

            return losses

    def backward(self, loss):
        # backward pass.
        # If AMP is disabled, the scaler will fall back to the default behavior.
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @torch.no_grad()
    def validation(self, epoch: int):
        losses_agg = defaultdict(float)
        for batch in self.validation_dataloader:
            batch = batch_as_dict(batch, self.dist.device)
            graph = batch.pop("graph")
            pred = batch_as_dict(
                self.model(graph.ndata["x"], graph.edata["x"], graph, **batch)
            )
            pred_g, gt_g = self.dataset.denormalize(
                pred["graph"], graph.ndata["y"], self.dist.device
            )
            losses_agg["graph"] += self.loss.graph(pred_g, gt_g)
            if (pred_c_d := pred.get("c_d")) is not None:
                losses_agg["c_d"] += self.loss.c_d(pred_c_d, batch["c_d"])

        losses_agg["total"] = sum(losses_agg.values())

        # Visualize last batch.
        for vis in self.visualizers.values():
            vis(graph, pred_g, gt_g, epoch, elogger)

        # Log losses.
        num_batches = len(self.validation_dataloader)
        loss_str = []
        for k, v in losses_agg.items():
            loss = v / num_batches
            elogger.log_scalar(f"val/loss/{k}", loss, epoch)
            loss_str.append(f"{k}: {loss:6.4f}")

        logger.info(f"Validation loss: {', '.join(loss_str)}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    init_python_logging(cfg, dist.rank)
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    # initialize loggers
    global elogger
    elogger = CompositeLogger(cfg)

    trainer = MGNTrainer(cfg)
    start = time.time()
    logger.info("Training started...")

    for epoch in range(trainer.epoch_init + 1, cfg.train.epochs + 1):
        losses_agg = defaultdict(float)
        for batch in trainer.dataloader:
            batch = batch_as_dict(batch, dist.device)
            losses = trainer.train(batch)
            for k, v in losses.items():
                losses_agg[k] += v.detach().cpu().numpy()
        num_batches = len(trainer.dataloader)
        for k, v in losses_agg.items():
            losses_agg[k] /= num_batches

        cur_lr = trainer.scheduler.get_last_lr()[0]
        logger.info(
            f"epoch: {epoch:5,}, loss: {losses_agg['total']:.5f}, "
            f"lr: {cur_lr:.7f}, "
            f"time per epoch: {(time.time() - start):5.2f}"
        )
        for k, v in losses_agg.items():
            elogger.log_scalar(f"train/loss/{k}", v, epoch)
        elogger.log_scalar("lr", cur_lr, epoch)

        # validation
        # TODO(akamenev): redundant restriction, val should run on all ranks.
        if dist.rank == 0:
            trainer.validation(epoch)

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and epoch % cfg.train.checkpoint_save_freq == 0:
            save_checkpoint(
                cfg.output,
                models=trainer.model.model(),
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
