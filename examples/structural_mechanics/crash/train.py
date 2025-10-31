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

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

# Import unified datapipe
from datapipe import SimSample, simsample_collate


class Trainer:
    """Trainer for crash simulation models with unified SimSample input."""

    def __init__(self, cfg: DictConfig, logger0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.cfg = cfg
        self.rollout_steps = cfg.training.num_time_steps - 1
        self.amp = cfg.training.amp

        # --- Consistency check between model and datapipe ---
        model_name = cfg.model._target_
        datapipe_name = cfg.datapipe._target_

        if "MeshGraphNet" in model_name and "GraphDataset" not in datapipe_name:
            raise ValueError(
                f"Model {model_name} requires a graph datapipe, "
                f"but you selected {datapipe_name}."
            )
        if "Transolver" in model_name and "PointCloudDataset" not in datapipe_name:
            raise ValueError(
                f"Model {model_name} requires a point-cloud datapipe, "
                f"but you selected {datapipe_name}."
            )

        # Dataset
        reader = instantiate(cfg.reader)
        logging.getLogger().setLevel(logging.INFO)
        dataset = instantiate(
            cfg.datapipe,
            name="crash_train",
            reader=reader,
            split="train",
            logger=logger0,
        )
        logging.getLogger().setLevel(logging.INFO)
        # Move stats to device
        self.data_stats = dict(
            node={k: v.to(self.dist.device) for k, v in dataset.node_stats.items()},
            edge={
                k: v.to(self.dist.device)
                for k, v in getattr(dataset, "edge_stats", {}).items()
            },
            feature={
                k: v.to(self.dist.device)
                for k, v in getattr(dataset, "feature_stats", {}).items()
            },
        )

        # Sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
            shuffle=True,
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # variable N per sample
            shuffle=(sampler is None),
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.training.num_dataloader_workers,
            sampler=sampler,
            collate_fn=simsample_collate,
        )
        self.sampler = sampler

        # Model
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.dist.device)
        self.model.train()

        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # Loss
        self.criterion = torch.nn.MSELoss()

        # Optimizer
        self.optimizer = None
        try:
            if cfg.training.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(
                    self.model.parameters(), lr=cfg.training.start_lr
                )
        except ImportError:
            logger0.warning("Apex not installed, falling back to Adam optimizer.")
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.training.start_lr
            )
        logger0.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.end_lr
        )
        self.scaler = GradScaler("cuda", enabled=self.amp)

        # Checkpoint
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            cfg.training.ckpt_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        if self.dist.rank == 0:
            self.writer = SummaryWriter(log_dir=cfg.training.tensorboard_log_dir)

    def train(self, sample: SimSample):
        self.optimizer.zero_grad()
        loss = self.forward(sample)
        self.backward(loss)
        return loss

    def forward(self, sample: SimSample):
        with autocast(device_type="cuda", enabled=self.amp):
            T = self.rollout_steps

            # Model forward
            pred = self.model(sample=sample, data_stats=self.data_stats)

            # Reshape target
            target_flat = sample.node_target  # [N, T*Fo]
            N = target_flat.size(0)
            Fo = 3  # output features per node
            assert target_flat.size(1) == T * Fo, (
                f"target dim {target_flat.size(1)} != {T * Fo}"
            )
            target = target_flat.view(N, T, Fo).transpose(0, 1).contiguous()  # [T,N,Fo]

            return self.criterion(pred, target)

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()

    trainer = Trainer(cfg, logger0)
    logger0.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0
        start = time.time()

        for sample in trainer.dataloader:
            sample = sample[0].to(dist.device)  # SimSample .to()
            loss = trainer.train(sample)
            total_loss += loss.item()
            num_batches += 1

        trainer.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        logger0.info(
            f"epoch: {epoch + 1}, avg_loss: {avg_loss:10.3e}, "
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e}, "
            f"time per epoch: {(time.time() - start):10.3e}"
        )

        if dist.rank == 0:
            trainer.writer.add_scalar("loss", avg_loss, epoch)
            trainer.writer.add_scalar(
                "learning_rate", trainer.optimizer.param_groups[0]["lr"], epoch
            )

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                cfg.training.ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved model on rank {dist.rank}")

    logger0.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
