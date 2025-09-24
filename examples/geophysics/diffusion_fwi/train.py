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

import hydra
import torch
import wandb
import importlib.util
import time
import datetime
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)

from datasets.dataset import EFWIDatapipe
from utils.preconditioning import edm_precond
from utils.nn import DiffusionFWINet
from utils.metrics import EDMLoss
from utils.diffusion import DiffusionAdapter
from datasets.transforms import ZscoreNormalize, Interpolate


@hydra.main(version_base="1.3", config_path="conf", config_name="config_train")
def main(cfg: DictConfig) -> None:
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # General python logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)

    # Initialize Weights & Biases
    checkpoint_dir = get_checkpoint_dir(str(cfg.io.checkpoint_dir), "diffusion_fwi")
    if cfg.io.load_checkpoint:
        metadata: Dict[str, Any] = {"wandb_id": None}
        load_checkpoint(checkpoint_dir, metadata_dict=metadata)
        wandb_id: str = metadata["wandb_id"]
        resume: str = "must"
        rank_zero_logger.info(f"Resuming wandb run with ID: {wandb_id}")
    else:
        wandb_id, resume = None, None
    initialize_wandb(
        project="DiffusionFWI-Training",
        entity=cfg.wandb.entity if hasattr(cfg.wandb, "entity") else "PhysicsNeMo",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        results_dir=cfg.wandb.results_dir,
        wandb_id=wandb_id,
        resume=resume,
        save_code=True,
        name=f"train-{timestamp}",
    )

    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")

    # Initialize diffusion model
    model_args = getattr(cfg.model, "model_args", None)
    if model_args is not None:
        model_args = OmegaConf.to_container(model_args)
        rank_zero_logger.info(f"Using model configuration: {model_args}")
    else:
        model_args = {}
    model_arch = DiffusionFWINet(
        x_resolution=list(cfg.model.x_resolution),
        x_channels=cfg.model.x_channels,
        y_resolution=list(cfg.model.y_resolution),
        y_channels=cfg.model.y_channels,
        encoder_hidden_channels=cfg.model.encoder_hidden_channels,
        num_encoder_blocks=cfg.model.num_encoder_blocks,
        N_grid_channels=cfg.model.N_grid_channels,
        model_channels=cfg.model.model_channels,
        channel_mult=list(cfg.model.channel_mult),
        num_blocks=cfg.model.num_blocks,
        **model_args,
    ).to(dist.device)
    # Thin wrapper around the model_backbone to convert it into a conditional
    # diffusion model compatible with EDM preconditioning and ResidualLoss
    model = DiffusionAdapter(
        model=model_arch,
        args_map=("x", "t", {"y": "y"}),
    )

    rank_zero_logger.info(
        f"Using model DiffusionFWINet with {model.num_parameters()} parameters."
    )

    # Distributed learning (Data parallel)
    if dist.world_size > 1:
        # Wrap the conditional model in DistributedDataParallel
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # EDM preconditioning wrapper
    model_fn = partial(edm_precond, model, sigma_data=0.5)

    # Initialize the training dataset
    train_dataset = EFWIDatapipe(
        data_dir=to_absolute_path(cfg.dataset.directory),
        phase="train",
        batch_size_per_device=cfg.training.batch_size_per_device,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    # Define dataset transform
    # Zscore normalization
    stats_mean = train_dataset.get_stats("mean")
    stats_std = train_dataset.get_stats("std")
    train_dataset = ZscoreNormalize(train_dataset, stats_mean, stats_std)
    img_H, img_W = list(cfg.model.x_resolution)

    # Interpolation to the UNet model accepted resolution
    interp_size = {var: (img_H, img_W) for var in cfg.dataset.x_vars}
    interp_size.update({var: (img_W,) for var in cfg.dataset.y_vars})
    interp_dim = {var: (-2, -1) for var in cfg.dataset.x_vars}
    interp_dim.update({var: (-1,) for var in cfg.dataset.y_vars})
    interp_mode = {var: "bilinear" for var in cfg.dataset.x_vars}
    interp_mode.update({var: "bilinear" for var in cfg.dataset.y_vars})
    train_dataset = Interpolate(
        train_dataset,
        size=interp_size,
        dim=interp_dim,
        mode=interp_mode,
    )

    # Initialize the validation dataset
    val_dataset = EFWIDatapipe(
        data_dir=to_absolute_path(cfg.dataset.directory),
        phase="test",
        batch_size_per_device=cfg.val.batch_size_per_device,
        shuffle=True,
        num_workers=cfg.val.num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    val_dataset = ZscoreNormalize(val_dataset, stats_mean, stats_std)
    val_dataset = Interpolate(
        val_dataset,
        size=interp_size,
        dim=interp_dim,
        mode=interp_mode,
    )

    # Loss
    loss_fn = EDMLoss(P_mean=0.0, P_std=1.0, sigma_data=0.5)

    # Create optimizer
    optimizer_class = None
    if torch.cuda.is_available():
        try:
            optimizer_class = getattr(
                importlib.import_module("apex.optimizers"), "FusedAdam"
            )
            rank_zero_logger.info("Using FusedAdam optimizer")
            use_FusedAdam = True
        except ImportError:
            pass
    if optimizer_class is None:
        optimizer_class = torch.optim.AdamW
        rank_zero_logger.info("Using AdamW optimizer")
        use_FusedAdam = False
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.training.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.training.max_epochs, eta_min=cfg.training.scheduler.eta_min
    )

    # Load checkpoint if explicitly requested
    loaded_epoch, total_samples_trained = 0, 0
    if dist.world_size > 1:
        torch.distributed.barrier()
    if cfg.io.load_checkpoint:
        metadata = {"total_samples_trained": total_samples_trained}
        loaded_epoch = load_checkpoint(
            checkpoint_dir,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
            metadata_dict=metadata,
        )
        total_samples_trained = metadata["total_samples_trained"]

    # Log initial learning rate
    current_lr = optimizer.param_groups[0]["lr"]
    rank_zero_logger.info(f"Starting learning rate: {current_lr}")
    if dist.rank == 0:
        wandb.log({"lr": current_lr, "epoch": loaded_epoch})

    # Training loop
    rank_zero_logger.info("Training started...")
    for epoch in range(max(1, loaded_epoch + 1), cfg.training.max_epochs + 1):
        model.train()
        epoch_loss, epoch_samples = 0.0, 0
        time_start = time.time()
        train_dataset.set_epoch(epoch)

        for i, data in enumerate(train_dataset):
            x = torch.cat(
                [
                    data.get(var, None)
                    for var in list(cfg.dataset.x_vars)
                    if data.get(var) is not None
                ],
                dim=1,
            )
            y = torch.cat(
                [
                    data.get(var, None)
                    for var in list(cfg.dataset.y_vars)
                    if data.get(var) is not None
                ],
                dim=1,
            )
            batch_size = x.shape[0]
            epoch_samples += batch_size

            optimizer.zero_grad(**({} if use_FusedAdam else {"set_to_none": True}))

            loss = loss_fn(
                model=model_fn,  # Use model_fn instead of model
                x=x,
                cond={"y": y},
            )
            loss = torch.mean(loss)

            epoch_loss += loss.item() * batch_size

            # Optimize
            loss.backward()
            optimizer.step()

            # Log mini-batch metrics
            current_lr = optimizer.param_groups[0]["lr"]
            batch_metrics = {"batch_loss": loss.item(), "lr": current_lr}
            if dist.rank == 0:
                wandb.log(batch_metrics)
            if i % cfg.io.log_freq == 0:
                rank_zero_logger.info(
                    f"lr: {current_lr}, batch: {i}, batch loss: {loss.item()}"
                )

        # Compute mean loss for the epoch
        mean_loss, epoch_samples_all_ranks = average_loss(
            dist, epoch_loss, epoch_samples
        )
        time_end = time.time()
        total_samples_trained += epoch_samples_all_ranks

        # Log epoch metrics
        metrics = {
            "epoch": epoch,
            "mean_loss": mean_loss,
            "time_per_epoch": time_end - time_start,
            "lr": current_lr,
            "total_samples_trained": total_samples_trained,
            "epoch_samples": epoch_samples_all_ranks,
        }
        if dist.rank == 0:
            wandb.log(metrics)
        msg = f"epoch: {epoch}, mean loss: {mean_loss:10.3e}"
        msg += f", time per epoch: {(time_end - time_start):10.3e}"
        msg += f", total samples: {total_samples_trained}"
        rank_zero_logger.info(msg)

        # Synchronize processes before validation
        if dist.world_size > 1:
            torch.distributed.barrier()

        # Run validation
        model.eval()
        mean_val_loss = validation_step(
            model_fn,
            val_dataset,
            loss_fn,
            dist,
            cfg,
        )
        # Log validation metrics
        val_metrics = {
            "val_loss": mean_val_loss,
            "epoch": epoch,
            "total_samples_trained": total_samples_trained,
        }
        if dist.rank == 0:
            wandb.log(val_metrics)
        rank_zero_logger.info(f"epoch: {epoch}, val loss: {mean_val_loss}")

        # Adjust learning rate based on validation loss
        scheduler.step()

        # Save checkpoint periodically
        if dist.world_size > 1:
            torch.distributed.barrier()
        if epoch % cfg.io.checkpoint_freq == 0 and dist.rank == 0:
            save_checkpoint(
                checkpoint_dir,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metadata={
                    "total_samples_trained": total_samples_trained,
                    "wandb_id": wandb.run.id,
                },
            )
            rank_zero_logger.info(f"Saved checkpoint at epoch {epoch}")

    # Finish logging
    wandb.finish()
    rank_zero_logger.info("Training completed!")


@torch.no_grad()
def validation_step(model, dataset, loss_fn, dist, cfg):
    """
    Perform validation on a dataset and return the average loss.
    """
    loss_epoch = 0.0
    num_samples = 0.0

    for i, data in enumerate(dataset):
        x = torch.cat(
            [
                data.get(var, None)
                for var in list(cfg.dataset.x_vars)
                if data.get(var) is not None
            ],
            dim=1,
        )
        y = torch.cat(
            [
                data.get(var, None)
                for var in list(cfg.dataset.y_vars)
                if data.get(var) is not None
            ],
            dim=1,
        )

        # Forward pass with validation data
        loss = loss_fn(
            model=model,
            x=x,
            cond={"y": y},
        )
        loss = torch.mean(loss)
        loss_epoch += loss.item() * x.shape[0]
        num_samples += x.shape[0]

    # Average validation loss across all ranks
    mean_val_loss, num_samples_all_ranks = average_loss(dist, loss_epoch, num_samples)

    return mean_val_loss


def average_loss(
    dist: DistributedManager,
    loss_value: float,
    sample_count: int,
) -> tuple[float, int]:
    """
    Average the loss value over all ranks.
    """
    if dist.world_size > 1:
        tensor = torch.tensor([loss_value, float(sample_count)], device=dist.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor[0].item() / tensor[1].item(), tensor[1].item()
    else:
        return (loss_value / sample_count), sample_count


if __name__ == "__main__":
    main()
