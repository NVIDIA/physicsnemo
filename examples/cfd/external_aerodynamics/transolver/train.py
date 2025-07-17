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
import torch
import hydra
import omegaconf
from tabulate import tabulate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from typing import Literal

from physicsnemo.utils.profiling import profile, Profiler

from datapipe import DomainParallelZarrDataset
from loss import loss_fn
from metrics import metrics_fn

from contextlib import nullcontext
from torch.amp import autocast, GradScaler

# import transformer_engine.pytorch as te
# from transformer_engine.common.recipe import Format, DelayedScaling


def get_autocast_context(precision: str) -> nullcontext:
    """
    Returns the appropriate autocast context for mixed precision training.

    Args:
        precision (str): The desired precision. Supported values are "float16", "bfloat16", or any other string for no autocast.

    Returns:
        Context manager: An autocast context for the specified precision, or a nullcontext if precision is not recognized.
    """
    if precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    # elif precision == "float8":
    #     print("Using float8 autocast")
    #     fp8_format = Format.HYBRID
    #     fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    #     return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        return nullcontext()


def cast_precisions(
    features: torch.Tensor, embeddings: torch.Tensor, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Casts the features and embeddings tensors to the specified precision.

    Args:
        features (torch.Tensor): The input features tensor.
        embeddings (torch.Tensor): The input embeddings tensor.
        precision (str): The desired precision ("float16", "bfloat16", or other for no cast).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The features and embeddings tensors cast to the specified precision.
    """
    if precision == "float16":
        return features.to(torch.float16), embeddings.to(torch.float16)
    elif precision == "bfloat16":
        return features.to(torch.bfloat16), embeddings.to(torch.bfloat16)
    else:
        return features, embeddings


@profile
def preprocess_surface_data(batch):

    mesh_centers = batch["surface_mesh_centers"]
    normals = batch["surface_normals"]
    targets = batch["surface_fields"]
    node_features = torch.stack(
        [batch["air_density"], batch["stream_velocity"]], dim=-1
    ).to(torch.float32)

    # Normalize the surface fields:
    norm_mean = targets.mean(dim=0)
    norm_std = targets.std(dim=0)
    targets = (targets - norm_mean) / norm_std
    # fourier_sin_features = [
    #     torch.sin(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(4)
    # ]
    # fourier_cos_features = [
    #     torch.cos(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(4)
    # ]

    # Calculate center of mass
    sizes = batch["stl_areas"]
    centers = batch["stl_centers"]

    total_weighted_position = torch.einsum("i,ij->j", sizes, centers)
    total_size = torch.sum(sizes)
    center_of_mass = total_weighted_position[None, ...] / total_size

    # Subtract the COM from the centers:
    mesh_centers = mesh_centers - center_of_mass

    embeddings = torch.cat(
        [
            mesh_centers,
            normals,
            # *fourier_sin_features,
            # *fourier_cos_features
        ],
        dim=-1,
    )

    node_features = node_features.unsqueeze(0).broadcast_to(embeddings.shape[0], -1)

    others = {
        "surface_areas": sizes,
        "surface_normals": normals,
        "stream_velocity": batch["stream_velocity"],
        "air_density": batch["air_density"],
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }

    return node_features, embeddings, targets, others


def preprocess_volume_data(batch):

    mesh_centers = batch["volume_mesh_centers"]
    targets = batch["volume_fields"]
    node_features = torch.stack(
        [batch["air_density"], batch["stream_velocity"]], dim=-1
    ).to(torch.float32)

    embeddings = mesh_centers

    # Normalize the surface fields:
    norm_mean = targets.mean(dim=0)
    norm_std = targets.std(dim=0)
    targets = (targets - norm_mean) / norm_std

    node_features = node_features.unsqueeze(0).broadcast_to(embeddings.shape[0], -1)

    others = {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }
    return node_features, embeddings, targets, others


@profile
def downsample_surface(features, embeddings, targets, num_keep=1024):
    # Determine the number of samples to keep (e.g., 50% of original size)
    num_samples = features.shape[0]
    # Generate random indices to keep (faster for large num_samples)
    indices = torch.multinomial(
        torch.ones(num_samples, device=features.device), num_keep, replacement=False
    )

    # Use the same indices to downsample all tensors
    downsampled_features = features[indices]
    downsampled_embeddings = embeddings[indices]
    downsampled_targets = targets[indices]

    downsampled_features = downsampled_features.unsqueeze(0)
    downsampled_embeddings = downsampled_embeddings.unsqueeze(0)
    downsampled_targets = downsampled_targets.unsqueeze(0)

    return downsampled_features, downsampled_embeddings, downsampled_targets


@profile
def downsample_volume(features, embeddings, targets, num_keep=1024):
    # Determine the number of samples to keep (e.g., 50% of original size)
    num_samples = features.shape[0]
    # The volume data is so large, that we'll sample randints
    # which will very rarely duplicate
    indices = torch.randint(0, num_samples, (num_keep,), device=features.device)

    # Use the same indices to downsample all tensors
    downsampled_features = features[indices]
    downsampled_embeddings = embeddings[indices]
    downsampled_targets = targets[indices]

    downsampled_features = downsampled_features.unsqueeze(0)
    downsampled_embeddings = downsampled_embeddings.unsqueeze(0)
    downsampled_targets = downsampled_targets.unsqueeze(0)

    return downsampled_features, downsampled_embeddings, downsampled_targets


@profile
def train_epoch(
    dataloader,
    sampler,
    model,
    optimizer,
    scheduler,
    logger,
    writer,
    epoch,
    cfg,
    dist_manager,
    scaler=None,  # <-- Added scaler argument
):
    """Train for one epoch

    Args:
        dataloader: Training data loader
        model: The model to train
        logger: Python logger instance
        writer: Tensorboard writer
        cfg: Configuration object
    """
    model.train()
    total_loss = 0
    total_metrics = {}

    epoch_indices = list(sampler)
    epoch_len = len(epoch_indices)
    precision = getattr(cfg.training, "precision", "float32")
    context = get_autocast_context(precision)
    start_time = time.time()
    with Profiler():
        for i, batch_idx in enumerate(epoch_indices):
            batch = dataloader[batch_idx]
            # Get data from batch
            if cfg.data.mode == "surface":
                features, embeddings, targets, others = preprocess_surface_data(batch)
                features, embeddings, targets = downsample_surface(
                    features, embeddings, targets, cfg.data.resolution
                )
            elif cfg.data.mode == "volume":
                features, embeddings, targets, others = preprocess_volume_data(batch)
                features, embeddings, targets = downsample_volume(
                    features, embeddings, targets, cfg.data.resolution
                )
            else:
                raise ValueError(f"Unknown data mode: {cfg.data.mode}")

            # preload the next batch, if we're not on the last batch
            if i < epoch_len - 1:
                dataloader.preload(epoch_indices[i + 1])

            # Cast precisions:
            features, embeddings = cast_precisions(features, embeddings, precision)

            with context:
                outputs = model(features, embeddings)
                loss = loss_fn(outputs, targets, others, cfg.data.mode)

            metrics = metrics_fn(outputs, targets, others, dist_manager, cfg.data.mode)

            optimizer.zero_grad()
            if precision == "float16" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            end_time = time.time()

            # Logging
            total_loss += loss.item()

            if i == 0:
                total_metrics = metrics
            else:
                total_metrics = {
                    k: total_metrics[k] + metrics[k].item() for k in metrics.keys()
                }

            duration = end_time - start_time
            start_time = end_time
            images_per_second = 1 / duration

            logger.info(
                f"Epoch {epoch} [{i}/{epoch_len}] Loss: {loss.item():.6f} Duration: {duration:.2f}s"
            )
            if dist_manager.rank == 0:
                writer.add_scalar(
                    "batch/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    i + epoch_len * epoch,
                )
                writer.add_scalar("batch/loss", loss.item(), i + epoch_len * epoch)
                writer.add_scalar(
                    "batch/throughpu_per_gpu", images_per_second, i + epoch_len * epoch
                )
                for metric_name, metric_value in metrics.items():
                    writer.add_scalar(
                        f"batch/{metric_name}", metric_value, i + epoch_len * epoch
                    )

    avg_loss = total_loss / epoch_len
    avg_metrics = {k: v / epoch_len for k, v in total_metrics.items()}
    if dist_manager.rank == 0:
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        for metric_name, metric_value in avg_metrics.items():
            writer.add_scalar(f"epoch/{metric_name}", metric_value, epoch)
        # Print average metrics using tabulate
        metrics_table = tabulate(
            [[k, v] for k, v in avg_metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        print(f"\nEpoch {epoch} Average Metrics:\n{metrics_table}\n")
    return avg_loss


@profile
def val_epoch(
    dataloader,
    sampler,
    model,
    logger,
    val_writer,
    epoch,
    cfg,
    dist_manager,
):
    """Validation for one epoch

    Args:
        dataloader: Validation data loader
        sampler: Validation data sampler
        model: The model to evaluate
        logger: Python logger instance
        writer: Tensorboard writer
        epoch: Current epoch number
        cfg: Configuration object
        dist_manager: Distributed manager instance
    """

    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_metrics = {}

    epoch_indices = list(sampler)
    epoch_len = len(epoch_indices)
    precision = getattr(cfg.training, "precision", "float32")
    context = get_autocast_context(precision)

    start_time = time.time()
    with torch.no_grad():  # Disable gradient computation
        for i, batch_idx in enumerate(epoch_indices):
            batch = dataloader[batch_idx]
            # Get data from batch
            if cfg.data.mode == "surface":
                features, embeddings, targets, others = preprocess_surface_data(batch)
                features, embeddings, targets = downsample_surface(
                    features, embeddings, targets, cfg.data.resolution
                )
            elif cfg.data.mode == "volume":
                features, embeddings, targets, others = preprocess_volume_data(batch)
                features, embeddings, targets = downsample_volume(
                    features, embeddings, targets, cfg.data.resolution
                )
            else:
                raise ValueError(f"Unknown data mode: {cfg.data.mode}")

            with context:
                outputs = model(features, embeddings)
                loss = loss_fn(outputs, targets, others, cfg.data.mode)

            metrics = metrics_fn(outputs, targets, others, dist_manager, cfg.data.mode)

            if i == 0:
                total_metrics = metrics
            else:
                total_metrics = {
                    k: total_metrics[k] + metrics[k].item() for k in metrics.keys()
                }

            # Logging
            total_loss += loss.item()

            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            logger.info(
                f"Val [{i}/{epoch_len}] Loss: {loss.item():.6f} Duration: {duration:.2f}s"
            )
            # We don't add individual loss measurements in the validation loop.

    avg_loss = total_loss / epoch_len
    avg_metrics = {k: v / epoch_len for k, v in total_metrics.items()}
    if dist_manager.rank == 0:
        val_writer.add_scalar("epoch/loss", avg_loss, epoch)
        for metric_name, metric_value in avg_metrics.items():
            val_writer.add_scalar(f"epoch/{metric_name}", metric_value, epoch)
        # Print average metrics using tabulate
        metrics_table = tabulate(
            [[k, v] for k, v in avg_metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        print(f"\nEpoch {epoch} Validation Average Metrics:\n{metrics_table}\n")
    return avg_loss


@profile
def main(cfg: DictConfig):
    """Main training function

    Args:
        cfg: Hydra configuration object
    """

    DistributedManager.initialize()

    # Set up distributed training
    dist_manager = DistributedManager()

    # Set up logging
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)
    if dist_manager.rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/train",
            )
        )
        val_writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/val",
            )
        )
    else:
        writer = None
        val_writer = None

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    # Set up model
    model = hydra.utils.instantiate(cfg.model)

    model.to(dist_manager.device)

    if dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=dist_manager.device,
        )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # Set up data
    device_mesh = None
    placements = None

    # Training dataset
    train_dataset = DomainParallelZarrDataset(
        data_path=cfg.data.train.data_path,
        device_mesh=device_mesh,
        placements=placements,
        max_workers=cfg.data.max_workers,
        pin_memory=cfg.data.pin_memory,
        keys_to_read=cfg.data.data_keys,
        large_keys=cfg.data.large_keys,
    )

    # Validation dataset
    val_dataset = DomainParallelZarrDataset(
        data_path=cfg.data.val.data_path,  # Assuming validation data path is configured
        device_mesh=device_mesh,
        placements=placements,
        max_workers=cfg.data.max_workers,
        pin_memory=cfg.data.pin_memory,
        keys_to_read=cfg.data.data_keys,
        large_keys=cfg.data.large_keys,
    )

    # Set up distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist_manager.world_size,
        rank=dist_manager.rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=dist_manager.world_size,
        rank=dist_manager.rank,
        shuffle=False,  # No shuffling for validation
        drop_last=True,
    )

    # Set up optimizer and scheduler
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # Set up learning rate scheduler based on config
    scheduler_cfg = cfg.scheduler
    scheduler_name = scheduler_cfg.name
    scheduler_params = dict(scheduler_cfg.params)

    if scheduler_name == "OneCycleLR":
        scheduler_params.setdefault("max_lr", cfg.optimizer.lr)
        # Dynamically compute total_steps
        total_steps = len(list(train_sampler)) * cfg.training.num_epochs
        scheduler_params["total_steps"] = total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    precision = getattr(cfg.training, "precision", "float32")
    scaler = GradScaler() if precision == "float16" else None

    ckpt_args = {
        "path": f"{cfg.output_dir}/runs/{cfg.run_id}/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)

    if cfg.training.compile:
        model = torch.compile(model)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        # Set the epoch in the samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        start_time = time.time()
        # Training phase
        train_loss = train_epoch(
            train_dataset,
            train_sampler,
            model,
            optimizer,
            scheduler,
            logger,
            writer,
            epoch,
            cfg,
            dist_manager,
            scaler,
        )
        end_time = time.time()
        train_duration = end_time - start_time

        start_time = time.time()
        # Validation phase
        val_loss = val_epoch(
            val_dataset,
            val_sampler,
            model,
            logger,
            val_writer,
            epoch,
            cfg,
            dist_manager,
        )
        end_time = time.time()
        val_duration = end_time - start_time

        # Log epoch results
        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] Train Loss: {train_loss:.6f} [duration: {train_duration:.2f}s] Val Loss: {val_loss:.6f} [duration: {val_duration:.2f}s]"
        )

        # save checkpoint
        if epoch % cfg.training.save_interval == 0 and dist_manager.rank == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="conf", config_name="train")
def launch(cfg: DictConfig):
    """Launch training with hydra configuration

    Args:
        cfg: Hydra configuration object
    """
    profiler = Profiler()
    # profiler.enable("torch")
    # profiler.enable("line_profiler")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
