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

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

import hydra
import omegaconf
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint

# Optional: tabulate for metrics tables, torchinfo for model summary
_tabulate = OptionalImport("tabulate")
_torchinfo = OptionalImport("torchinfo")

# Import unified datapipe
from datapipe import SimSample, simsample_collate
from omegaconf import open_dict


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
        if "FIGConvUNet" in model_name and "PointCloudDataset" not in datapipe_name:
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

        if cfg.training.num_validation_samples > 0:
            self.num_validation_replicas = min(
                self.dist.world_size, cfg.training.num_validation_samples
            )
            self.num_validation_samples = (
                cfg.training.num_validation_samples
                // self.num_validation_replicas
                * self.num_validation_replicas
            )
            logger0.info(f"Number of validation samples: {self.num_validation_samples}")

            # Create a validation dataset
            val_cfg = self.cfg.datapipe
            with open_dict(val_cfg):  # or open_dict(cfg) to open the whole tree
                val_cfg.data_dir = self.cfg.training.raw_data_dir_validation
                val_cfg.num_samples = self.num_validation_samples
            val_dataset = instantiate(
                val_cfg,
                name="crash_validation",
                reader=reader,
                split="validation",
                logger=logger0,
            )

            if self.dist.rank < self.num_validation_replicas:
                # Sampler
                if self.dist.world_size > 1:
                    sampler = DistributedSampler(
                        val_dataset,
                        num_replicas=self.num_validation_replicas,
                        rank=self.dist.rank,
                        shuffle=False,
                        drop_last=True,
                    )
                else:
                    sampler = None

                self.val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1,  # variable N per sample
                    shuffle=(sampler is None),
                    drop_last=True,
                    pin_memory=True,
                    num_workers=cfg.training.num_dataloader_workers,
                    sampler=sampler,
                    collate_fn=simsample_collate,
                )
            else:
                self.val_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(val_dataset, []), batch_size=1
                )

        # Model
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.dist.device)
        self.model.train()

        # Log model summary and parameter count (optional: torchinfo)
        if self.dist.rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters())
            logger0.info(f"Model parameters: {num_params:,}")
            if _torchinfo.available:
                try:
                    logger0.info(f"\n{_torchinfo.summary(self.model, verbose=0)}")
                except Exception:
                    logger0.info(
                        "(torchinfo summary skipped: model requires sample input)"
                    )

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
            # Model forward - returns [N, T, Fo]
            pred = self.model(sample=sample, data_stats=self.data_stats)

            # Target is [N, T, Fo]
            target = sample.node_target
            return self.criterion(pred, target)

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation error computation"""
        self.model.eval()

        MSE = torch.zeros(1, device=self.dist.device)
        MSE_w_time = torch.zeros(self.rollout_steps, device=self.dist.device)
        for idx, sample in enumerate(self.val_dataloader):
            sample = sample[0].to(self.dist.device)  # SimSample .to()

            # Model forward - returns [N, T, Fo]
            pred = self.model(sample=sample, data_stats=self.data_stats)

            # Target is [N, T, Fo]
            target = sample.node_target

            # Compute and add error
            SqError = torch.square(pred - target)
            MSE_w_time += torch.mean(
                SqError, dim=(0, 2)
            )  # mean over N, Fo per timestep
            MSE += torch.mean(SqError)

        # Sum errors across all ranks
        if self.dist.world_size > 1:
            torch.distributed.all_reduce(MSE, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(MSE_w_time, op=torch.distributed.ReduceOp.SUM)

        val_stats = {
            "MSE_w_time": MSE_w_time / self.num_validation_samples,
            "MSE": MSE / self.num_validation_samples,
        }

        self.model.train()  # Switch back to training mode
        return val_stats


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()

    # Log full config and paths
    logger0.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger0.info(f"Output directory: {cfg.training.tensorboard_log_dir}")
    logger0.info(f"Checkpoint directory: {cfg.training.ckpt_path}")

    trainer = Trainer(cfg, logger0)
    logger0.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0
        start = time.time()
        batch_start = start
        epoch_len = len(trainer.dataloader)
        log_every = max(1, epoch_len // 10)  # Log ~10 times per epoch

        for batch_idx, sample in enumerate(trainer.dataloader):
            sample = sample[0].to(dist.device)  # SimSample .to()
            loss = trainer.train(sample)
            total_loss += loss.detach().item()
            num_batches += 1

            # Per-batch progress
            if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                batch_duration = time.time() - batch_start
                mem_gb = (
                    torch.cuda.memory_reserved() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                logger0.info(
                    f"Epoch {epoch + 1} [{batch_idx + 1}/{epoch_len}] "
                    f"Loss: {loss.detach().item():.6f} "
                    f"Duration: {batch_duration:.2f}s Mem: {mem_gb:.2f}GB"
                )
            batch_start = time.time()

        trainer.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        epoch_duration = time.time() - start
        logger0.info(
            f"Epoch {epoch + 1}/{cfg.training.epochs} "
            f"avg_loss: {avg_loss:.6f} "
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e} "
            f"duration: {epoch_duration:.2f}s"
        )

        if dist.rank == 0:
            trainer.writer.add_scalar("loss", avg_loss, epoch)
            trainer.writer.add_scalar(
                "learning_rate", trainer.optimizer.param_groups[0]["lr"], epoch
            )

        if dist.world_size > 1:
            torch.distributed.barrier()

        if dist.rank == 0 and (epoch + 1) % cfg.training.save_checkpoint_freq == 0:
            save_checkpoint(
                cfg.training.ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved model on rank {dist.rank}")

        # Validation
        if (
            cfg.training.num_validation_samples > 0
            and (epoch + 1) % cfg.training.validation_freq == 0
        ):
            val_stats = trainer.validate(epoch)

            # Log validation metrics
            mse_val = val_stats["MSE"].item()
            mse_w_time = val_stats["MSE_w_time"]
            logger0.info(f"Validation epoch {epoch + 1}: MSE: {mse_val:.6f}")
            if _tabulate.available and dist.rank == 0:
                rows = [["MSE (overall)", f"{mse_val:.6f}"]]
                for i, m in enumerate(mse_w_time):
                    rows.append([f"timestep_{i}_MSE", f"{m.item():.6f}"])
                logger0.info(
                    f"\nValidation metrics:\n{_tabulate.tabulate(rows, headers=['Metric', 'Value'], tablefmt='pretty')}\n"
                )

            if dist.rank == 0:
                # Log to tensorboard
                trainer.writer.add_scalar("val/MSE", val_stats["MSE"].item(), epoch)

                # Log individual timestep relative errors
                for i in range(len(val_stats["MSE_w_time"])):
                    trainer.writer.add_scalar(
                        f"val/timestep_{i}_MSE",
                        val_stats["MSE_w_time"][i].item(),
                        epoch,
                    )

    logger0.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
