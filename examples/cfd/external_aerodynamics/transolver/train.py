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


from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor import distribute_module

from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from typing import Literal

from physicsnemo.utils.profiling import profile, Profiler

from datapipe import DomainParallelZarrDataset
from loss import loss_fn
from metrics import metrics_fn
from preprocess import (
    preprocess_volume_data,
    preprocess_surface_data,
    downsample_volume,
    downsample_surface,
)

from contextlib import nullcontext
from torch.amp import autocast, GradScaler

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


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
    elif precision == "float8":
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
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


def forward_pass(
    batch: dict,
    model: torch.nn.Module,
    precision: str,
    output_pad_size: int | None,
    dist_manager: DistributedManager,
    cfg: DictConfig,
):
    """
    Run the forward pass of the model for one batch, including metrics and loss calculation.
    """

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

    # Cast precisions:
    features, embeddings = cast_precisions(features, embeddings, precision)
    with get_autocast_context(precision):
        print(
            f"features shape: {features.shape} nd placements: {features._spec.placements}"
        )
        print(
            f"embeddings shape: {embeddings.shape} nd placements: {embeddings._spec.placements}"
        )
        print(
            f"targets shape: {targets.shape} nd placements: {targets._spec.placements}"
        )
        print(f"cat shape: {torch.cat((features, embeddings), dim=-1).shape}")
        outputs = model(features, embeddings)
        if output_pad_size is not None:
            # Remove the padded outputs:
            outputs = outputs[:, :, :-output_pad_size]
        loss = loss_fn(outputs, targets, others, cfg.data.mode)

    metrics = metrics_fn(outputs, targets, others, dist_manager, cfg.data.mode)

    return loss, metrics


@profile
def train_epoch(
    dataloader,
    sampler: torch.utils.data.Sampler | None,
    model: torch.nn.Module,
    output_pad_size: int | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: PythonLogger,
    writer: SummaryWriter,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        dataloader (list[dict]): Training data loader
        sampler (torch.utils.data.Sampler | None): Sampler for distributed or sequential sampling.
        model (torch.nn.Module): The neural network model to train.
        output_pad_size (int | None): Optional output padding size for lowest precisions (FP8).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        logger (PythonLogger): Logger for training progress.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        epoch (int): Current epoch number.
        cfg (DictConfig): Hydra configuration object.
        dist_manager (DistributedManager): Distributed manager from physicsnemo.
        scaler (GradScaler | None, optional): Gradient scaler for mixed precision training.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    total_metrics = {}

    epoch_indices = list(sampler) if sampler is not None else range(len(dataloader))
    epoch_len = len(epoch_indices)
    precision = getattr(cfg.training, "precision", "float32")
    start_time = time.time()
    with Profiler():
        for i, batch_idx in enumerate(epoch_indices):

            if i > 10:
                break
            batch = dataloader[batch_idx]
            # preload the next batch, if we're not on the last batch
            if i < epoch_len - 1 and sampler is not None:
                dataloader.preload(epoch_indices[i + 1])

            loss, metrics = forward_pass(
                batch, model, precision, output_pad_size, dist_manager, cfg
            )

            optimizer.zero_grad()
            if precision == "float16" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if not isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()

            end_time = time.time()

            # Logging
            this_loss = loss.detach().item()
            total_loss += this_loss

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
                f"Epoch {epoch} [{i}/{epoch_len}] Loss: {this_loss:.6f} Duration: {duration:.2f}s"
            )
            if dist_manager.rank == 0:
                writer.add_scalar(
                    "batch/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    i + epoch_len * epoch,
                )
                writer.add_scalar("batch/loss", this_loss, i + epoch_len * epoch)
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
    sampler: torch.utils.data.Sampler | None,
    model: torch.nn.Module,
    output_pad_size: int | None,
    logger: PythonLogger,
    val_writer: SummaryWriter,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
) -> float:
    """
    Run validation for one epoch.

    Args:
        dataloader (list[dict]): Validation data loader.
        sampler (torch.utils.data.Sampler | None): Sampler for distributed or sequential sampling.
        model (torch.nn.Module): The model to evaluate.
        output_pad_size (int | None): Optional output padding size for lowest precisions (FP8).
        logger (PythonLogger): Logger for validation progress.
        val_writer (SummaryWriter): TensorBoard writer for logging validation metrics.
        epoch (int): Current epoch number.
        cfg (DictConfig): Hydra configuration object.
        dist_manager (DistributedManager): Distributed manager instance.

    Returns:
        float: The average validation loss for the epoch.
    """

    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_metrics = {}

    epoch_indices = list(sampler) if sampler is not None else range(len(dataloader))
    epoch_len = len(epoch_indices)
    precision = getattr(cfg.training, "precision", "float32")

    start_time = time.time()
    with torch.no_grad():  # Disable gradient computation
        for i, batch_idx in enumerate(epoch_indices):
            if i > 10:
                break
            # Get data from batch
            batch = dataloader[batch_idx]

            # preload the next batch, if we're not on the last batch
            if i < epoch_len - 1 and sampler is not None:
                dataloader.preload(epoch_indices[i + 1])

            loss, metrics = forward_pass(
                batch, model, precision, output_pad_size, dist_manager, cfg
            )

            if i == 0:
                total_metrics = metrics
            else:
                total_metrics = {
                    k: total_metrics[k] + metrics[k].item() for k in metrics.keys()
                }

            # Logging
            this_loss = loss.detach().item()
            total_loss += this_loss

            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            logger.info(
                f"Val [{i}/{epoch_len}] Loss: {this_loss:.6f} Duration: {duration:.2f}s"
            )
            # We don't add individual loss measurements to tensorboard in the validation loop.

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

    if cfg.training.precision == "float8":
        # we have to manipulate the output shape
        # to enable fp8 computations with transformer_engine.
        # need the output to be divisible by 16.
        # if (cfg.model.embedding_dim + cfg.model.functional_dim) % 16 != 0:

        if cfg.model.out_dim % 16 != 0:
            # pad the output:
            output_pad_size = 16 - (cfg.model.out_dim % 16)
            cfg.model.out_dim += output_pad_size
            logger.info(
                f"Padding output dimension to {cfg.model.out_dim} for fp8 autocast"
            )
        else:
            output_pad_size = None
    else:
        input_pad_size = None
        output_pad_size = None

    # Set up model
    model = hydra.utils.instantiate(cfg.model)

    model.to(dist_manager.device)

    # Configure domain parallelism, if enabled:
    domain_size = int(cfg.training.domain_parallelism)

    if domain_size > 1:
        # You can use -1 to one axis to indicate that you want to use all the GPUs in that dimension.
        mesh = dist_manager.initialize_mesh(
            mesh_shape=(-1, domain_size), mesh_dim_names=("ddp", "domain")
        )
        # This is a subset of all the GPUs, and will vary depending on the process.
        # Think of this as slicing the global mesh along the domain axis.
        # It will contain only the GPUs that this process is sharing data with.
        domain_mesh = mesh["domain"]
        ddp_mesh = mesh["ddp"]
        placements = (Shard(1),)
    else:
        domain_mesh = None
        ddp_mesh = None
        placements = None

    if domain_size > 1:
        # Instead of DDP, for sharding we use FSDP.  It's possible to use FSDP in the DDP
        # mode, but since it's not pure data parallel we have to me more careful.

        # First, distribute the model so that each GPU has the copy with DTensor weights:
        model = distribute_module(model, domain_mesh)

        model = FSDP(
            model,
            device_mesh=mesh["ddp"],
            sharding_strategy=ShardingStrategy.NO_SHARD,
        )

    elif dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=dist_manager.device,
        )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # Training dataset

    train_dataset = DomainParallelZarrDataset(
        data_path=cfg.data.train.data_path,
        device_mesh=domain_mesh,
        placements=placements,
        max_workers=cfg.data.max_workers,
        pin_memory=cfg.data.pin_memory,
        keys_to_read=cfg.data.data_keys,
        large_keys=cfg.data.large_keys,
    )

    # Validation dataset

    val_dataset = DomainParallelZarrDataset(
        data_path=cfg.data.val.data_path,  # Assuming validation data path is configured
        device_mesh=domain_mesh,
        placements=placements,
        max_workers=cfg.data.max_workers,
        pin_memory=cfg.data.pin_memory,
        keys_to_read=cfg.data.data_keys,
        large_keys=cfg.data.large_keys,
    )

    if ddp_mesh is not None:
        num_replicas = ddp_mesh.size()
        data_rank = ddp_mesh.get_local_rank()
    else:
        num_replicas = dist_manager.world_size
        data_rank = dist_manager.rank

    # Set up distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=False,  # No shuffling for validation
        drop_last=True,
    )

    print(
        f"num_replicas: {num_replicas}, data_rank: {data_rank}, val targets: {list(val_sampler)}"
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
        "path": f"{cfg.output_dir}/{cfg.run_id}/checkpoints",
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
            output_pad_size,
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
            output_pad_size,
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

        if scheduler_name == "StepLR":
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
    profiler.enable("line_profiler")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
