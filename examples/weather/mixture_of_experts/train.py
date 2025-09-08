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
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)
from mowe_model import MoWE
from mowe_dataloader import MoWEDatapipe


def weighted_mse_loss(x, y, t):
    """Computes the weighted MSE loss between prediction and target. Using 1/t weighting"""
    # Ensure t is flat (N,) for broadcasting
    if t.dim() > 1:
        t = t.squeeze()
    # Calculate MSE for each item in the batch
    mse_per_item = torch.mean((x - y) ** 2, dim=(1, 2, 3))
    # Weight by lead time and take the mean over the batch
    l2_loss = (mse_per_item / t.detach()).mean()
    return l2_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config_base")
def main(cfg: DictConfig) -> None:
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # General python logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging(f"launch-{timestamp}.log")

    # Initialize Weights & Biases
    checkpoint_dir = get_checkpoint_dir(
        str(cfg.io.checkpoint_dir), str(cfg.io.model_name)
    )
    metadata = {"wandb_id": None}
    if cfg.io.load_checkpoint:
        try:
            load_checkpoint(checkpoint_dir, metadata_dict=metadata)
            wandb_id, resume = metadata["wandb_id"], "must"
            rank_zero_logger.info(f"Resuming wandb run with ID: {wandb_id}")
        except:
            rank_zero_logger.warning("Checkpoint not found. Starting a new run.")
            wandb_id, resume = None, "allow"
    else:
        wandb_id, resume = None, "allow"

    if dist.rank == 0:
        initialize_wandb(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if hasattr(cfg.wandb, "entity") else "PhysicsNeMo",
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            results_dir=cfg.wandb.results_dir,
            wandb_id=wandb_id,
            resume=resume,
            save_code=True,
        )
    LaunchLogger.initialize(use_mlflow=False)

    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")

    model = MoWE(
        input_size=tuple(cfg.model_params.input_size),
        n_models=cfg.model_params.n_models,
        patch_size=cfg.model_params.patch_size,
        in_channels=cfg.model_params.in_channels,
        out_channels=cfg.model_params.out_channels,
        hidden_size=cfg.model_params.hidden_size,
        depth=cfg.model_params.depth,
        num_heads=cfg.model_params.num_heads,
        mlp_ratio=cfg.model_params.mlp_ratio,
        attention_backbone=cfg.model_params.attention_backbone,
        layernorm_backbone=cfg.model_params.layernorm_backbone,
        noise_dim=cfg.model_params.noise_dim,
        return_probabilities=cfg.model_params.return_probabilities,
        bias=cfg.model_params.bias,
    ).to(dist.device)

    if dist.rank == 0:
        logger.info(
            f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    # # Distributed learning (Data parallel)
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # --- Training Datapipe (for all ranks) ---
    train_datapipe = MoWEDatapipe(
        data_dirs=[to_absolute_path(path) for path in cfg.data.data_dirs],
        stats_dir=to_absolute_path(cfg.data.stats_dir) if cfg.data.stats_dir else None,
        in_channels=cfg.data.in_channels,
        out_channels=cfg.data.out_channels,
        orig_index=cfg.data.orig_index,
        max_lead_time=cfg.data.max_lead_time,
        shuffle_model_idx=cfg.data.shuffle_model_idx,
        shuffle_channel_order=cfg.data.shuffle_channel_order,
        model_time_offsets=cfg.data.model_time_offsets,
        batch_size=cfg.data.batch_size_train,
        num_samples_per_year=cfg.data.num_samples_per_year,
        shuffle=True,
        num_workers=cfg.data.num_workers_train,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        mode="train",
        ratio=cfg.training.train_split_ratio,
    )
    logger.success(
        f"Loaded training datapipe of size {len(train_datapipe)} for rank {dist.rank}"
    )

    # --- Validation Datapipe (for all ranks) ---
    validation_datapipe = MoWEDatapipe(
        data_dirs=[to_absolute_path(path) for path in cfg.data.data_dirs],
        stats_dir=to_absolute_path(cfg.data.stats_dir) if cfg.data.stats_dir else None,
        in_channels=cfg.data.in_channels,
        out_channels=cfg.data.out_channels,
        orig_index=cfg.data.orig_index,
        max_lead_time=cfg.data.max_lead_time,
        shuffle_model_idx=cfg.data.shuffle_model_idx,
        shuffle_channel_order=cfg.data.shuffle_channel_order,
        model_time_offsets=cfg.data.model_time_offsets,
        num_samples_per_year=cfg.data.num_samples_per_year,
        batch_size=cfg.data.batch_size_validation,
        num_workers=cfg.data.num_workers_validation,
        shuffle=False,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        mode="valid",
        ratio=cfg.training.train_split_ratio,
    )
    logger.success(
        f"Loaded validation datapipe of size {len(validation_datapipe)} for rank {dist.rank}"
    )

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
        optimizer,
        T_max=cfg.training.max_epochs,  # number of epochs over which to anneal
        eta_min=5e-6,  # minimum LR at the end of the cycle
    )

    scaler = GradScaler("cuda")
    # Load checkpoint if explicitly requested
    loaded_epoch = 0
    if dist.world_size > 1:
        torch.distributed.barrier()
    if cfg.io.load_checkpoint:
        loaded_epoch = load_checkpoint(
            checkpoint_dir,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
            metadata_dict=metadata,
        )

    # Log initial learning rate
    current_lr = optimizer.param_groups[0]["lr"]
    rank_zero_logger.info(f"Starting learning rate: {current_lr}")
    if dist.rank == 0:
        wandb.log({"lr": current_lr, "epoch": loaded_epoch})

    # Training loop
    rank_zero_logger.info("Training started...")
    for epoch in range(max(1, loaded_epoch + 1), cfg.training.max_epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        time_start = time.time()

        # Use LaunchLogger for training
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_datapipe), epoch_alert_freq=1
        ) as launchlog:
            for i, data in enumerate(train_datapipe):
                optimizer.zero_grad()
                invar = data[0]["invar"]
                outvar = data[0]["outvar"]
                t = data[0]["lead_time"].squeeze()  # Squeeze here

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    noise = None
                    if cfg.model_params.bias:
                        probabilities, bias = model(invar, t, noise)
                        outpred = (
                            torch.einsum("nmchw,nmchw->nchw", probabilities, invar)
                            + bias
                        )
                    else:
                        probabilities = model(invar, t, noise)
                        outpred = torch.einsum(
                            "nmchw,nmchw->nchw", probabilities, invar
                        )
                    loss = weighted_mse_loss(outpred, outvar, t)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_total_loss += loss.item()

                # Log mini-batch metrics
                current_lr = optimizer.param_groups[0]["lr"]
                batch_metrics = {"batch_loss": loss.item(), "lr": current_lr}
                launchlog.log_minibatch(batch_metrics)
                if dist.rank == 0:
                    wandb.log(batch_metrics)
                if i % cfg.io.log_freq == 0:
                    rank_zero_logger.info(
                        f"lr: {current_lr}, batch: {i}, batch loss: {loss.item()}"
                    )

            # Compute mean loss for the epoch
            mean_loss = average_loss(dist, epoch_total_loss, len(train_datapipe))
            time_end = time.time()

            # Log epoch metrics
            metrics = {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "time_per_epoch": time_end - time_start,
                "lr": current_lr,
            }
            launchlog.log_epoch(metrics)
            if dist.rank == 0:
                wandb.log(metrics)
            msg = f"epoch: {epoch}, mean loss: {mean_loss:10.3e}"
            msg += f", time per epoch: {(time_end - time_start):10.3e}"
            rank_zero_logger.info(msg)

        # Synchronize processes before validation
        if dist.world_size > 1:
            torch.distributed.barrier()

        # Run validation with LaunchLogger
        with LaunchLogger("valid", epoch=epoch) as launchlog:
            model.eval()
            mean_val_loss, mean_aurora_loss, mean_pangu_loss = validation_step(
                model, validation_datapipe, weighted_mse_loss, dist, cfg
            )
            # Log validation metrics
            val_metrics = {
                "Val MSE loss": mean_val_loss,
                "Aurora loss": mean_aurora_loss,
                "Pangu loss": mean_pangu_loss,
                "epoch": epoch,
            }
            launchlog.log_epoch(val_metrics)
            if dist.rank == 0:
                wandb.log(val_metrics)
            rank_zero_logger.info(f"epoch: {epoch}, val loss: {mean_val_loss}")
            rank_zero_logger.info(
                f"Aurora loss: {mean_aurora_loss}, Pangu loss: {mean_pangu_loss}"
            )

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
                    "wandb_id": wandb.run.id,
                },
            )
            rank_zero_logger.info(f"Saved checkpoint at epoch {epoch}")

    # Finish logging
    wandb.finish()
    # if dist.rank == 0:
    #     mlflow.end_run()
    rank_zero_logger.info("Training completed!")


@torch.no_grad()
def validation_step(model, dataset, loss_fn, dist, cfg):
    """
    Perform validation on a dataset and return the average loss.
    """
    epoch_val_loss = 0.0
    epoch_aurora_loss = 0.0
    epoch_pangu_loss = 0.0

    for i, data in enumerate(dataset):
        invar, outvar, t = (
            data[0]["invar"],
            data[0]["outvar"],
            data[0]["lead_time"].squeeze(),
        )

        aurora_pred = invar[:, 1]
        pangu_pred = invar[:, 2]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            noise = None
            if cfg.model_params.bias:
                probabilities, bias = model(invar, t, noise)
                outpred = torch.einsum("nmchw,nmchw->nchw", probabilities, invar) + bias
            else:
                probabilities = model(invar, t, noise)
                outpred = torch.einsum("nmchw,nmchw->nchw", probabilities, invar)
            val_loss = loss_fn(outpred, outvar, t)
            aurora_loss = loss_fn(aurora_pred, outvar, t)
            pangu_loss = loss_fn(pangu_pred, outvar, t)

        epoch_val_loss += val_loss.item()
        epoch_aurora_loss += aurora_loss.item()
        epoch_pangu_loss += pangu_loss.item()

    num_samples = len(dataset)
    # Average validation loss across all ranks
    mean_val_loss = average_loss(dist, epoch_val_loss, num_samples)
    mean_aurora_loss = average_loss(dist, epoch_aurora_loss, num_samples)
    mean_pangu_loss = average_loss(dist, epoch_pangu_loss, num_samples)

    return mean_val_loss, mean_aurora_loss, mean_pangu_loss


def average_loss(dist, loss_value: float, sample_count: int) -> float:
    """
    Average the loss value over all ranks.
    """
    if dist.world_size > 1:
        tensor = torch.tensor(loss_value / sample_count, device=dist.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor.item() / dist.world_size
    else:
        return loss_value / sample_count


if __name__ == "__main__":
    main()
