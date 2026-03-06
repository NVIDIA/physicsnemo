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
"""Training script for the Domino transient conjugate heat-transfer task.

Assumes data has already been processed into ``<processed_dir>/train|val`` with
stats in ``<processed_dir>/stats`` (produced by ``process_data.py``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.datapipes.cae.domino_datapipe import create_domino_dataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino.utils.utils import dict_to_device
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from utils import (
    count_global_features,
    infer_shapes,
    load_scaling,
    masked_mse,
)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    surf_weight: float,
    vol_weight: float,
    logger: PythonLogger,
    log_interval: int = 5,
) -> Dict[str, float]:
    """Train the model for one epoch."""

    # Set model to training mode
    model.train(True)

    # Initialize totals
    totals = {"loss": 0.0, "loss_surface": 0.0, "loss_volume": 0.0}
    num_batches = 0

    # Train on each batch
    for batch_idx, batch in enumerate(loader, start=1):
        # Convert batch to device
        batch = dict_to_device(batch, device)

        # Zero out gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass (with autocast if enabled)
        with autocast(enabled=use_amp):
            # Forward pass
            pred_vol, pred_surf = model(batch)

            # Initialize loss tensors
            loss_surface = torch.tensor(0.0, device=device)
            loss_volume = torch.tensor(0.0, device=device)

            # Compute surface and volume losses
            if pred_surf is not None and "surface_fields" in batch:
                loss_surface = masked_mse(
                    pred_surf, batch["surface_fields"], batch.get("surface_valid_mask")
                )
            if pred_vol is not None and "volume_fields" in batch:
                loss_volume = masked_mse(
                    pred_vol, batch["volume_fields"], batch.get("volume_valid_mask")
                )

            # Compute total loss
            loss = surf_weight * loss_surface + vol_weight * loss_volume

        # Scale and step the optimizer if AMP is enabled
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update totals
        totals["loss"] += loss.item()
        totals["loss_surface"] += loss_surface.item()
        totals["loss_volume"] += loss_volume.item()
        num_batches += 1

        # Log progress
        if log_interval is not None and batch_idx % max(1, log_interval) == 0:
            logger.info(
                f"  batch {batch_idx:05d} | loss={loss.item():.4e} "
                f"(surf={loss_surface.item():.4e}, vol={loss_volume.item():.4e})"
            )

    # Return the average loss for each metric
    for key in totals:
        totals[key] = totals[key] / max(1, num_batches)
    return totals


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> Dict[str, float]:
    """Evaluate the model on the validation set."""

    # Set model to evaluation mode
    model.eval()

    # Initialize totals
    totals = {"loss": 0.0, "loss_surface": 0.0, "loss_volume": 0.0}
    num_batches = 0

    # Evaluate on each batch
    for batch in loader:
        # Convert batch to device
        batch = dict_to_device(batch, device)

        # Forward pass (with autocast if enabled)
        with autocast(enabled=use_amp):
            # Forward pass
            pred_vol, pred_surf = model(batch)

            # Initialize loss tensors
            loss_surface = torch.tensor(0.0, device=device)
            loss_volume = torch.tensor(0.0, device=device)

            # Compute surface and volume losses
            if pred_surf is not None and "surface_fields" in batch:
                loss_surface = masked_mse(
                    pred_surf, batch["surface_fields"], batch.get("surface_valid_mask")
                )
            if pred_vol is not None and "volume_fields" in batch:
                loss_volume = masked_mse(
                    pred_vol, batch["volume_fields"], batch.get("volume_valid_mask")
                )

            # Compute total loss
            loss = loss_surface + loss_volume

        # Update totals
        totals["loss"] += loss.item()
        totals["loss_surface"] += loss_surface.item()
        totals["loss_volume"] += loss_volume.item()
        num_batches += 1

    for key in totals:
        totals[key] = totals[key] / max(1, num_batches)
    return totals


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    # Set struct to false
    OmegaConf.set_struct(cfg, False)

    # Get script directory
    script_dir = Path(__file__).resolve().parent

    # Resolve path relative to script directory
    def resolve(path_like: str | os.PathLike[str]) -> Path:
        path = Path(path_like).expanduser()
        if not path.is_absolute():
            path = (script_dir / path).resolve()
        return path

    # Get model type
    model_type = str(cfg.model.model_type).lower()
    include_surface = model_type in {"surface", "combined"}
    include_volume = model_type in {"volume", "combined"}
    if not include_surface and not include_volume:
        raise ValueError(
            "At least one of surface or volume predictions must be enabled."
        )

    # Get processed root directory
    processed_root = resolve(cfg.data.processed_dir)

    # Get train, val, and stats directories
    train_dir = processed_root / "train"
    val_dir = processed_root / "val"
    stats_dir = processed_root / "stats"
    if not train_dir.exists():
        raise FileNotFoundError(f"Processed train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Processed val directory not found: {val_dir}")
    if not stats_dir.exists():
        raise FileNotFoundError(f"Processed stats directory not found: {stats_dir}")

    # Resolve checkpoint directory
    checkpoint_dir = resolve(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load scaling factors (compute if missing).
    volume_scaling, surface_scaling = load_scaling(
        stats_dir, include_surface, include_volume
    )
    if not include_volume:
        volume_scaling = None
    if not include_surface:
        surface_scaling = None

    # Infer output channels and future steps from the first sample.
    first_sample = next(iter(sorted(train_dir.glob("*.npz"))))
    surface_channels, volume_channels, num_future_steps = infer_shapes(
        first_sample, cfg, include_surface, include_volume
    )

    # Update cfg for dataset creation.
    cfg.data.input_dir = str(train_dir)
    cfg.data.input_dir_val = str(val_dir)
    cfg.project_dir = str(stats_dir)
    cfg.train.checkpoint_dir = str(checkpoint_dir)
    cfg.data_processor = OmegaConf.create({"use_cache": False})

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    logger = RankZeroLoggingWrapper(PythonLogger("train"), dist)

    # Resolve log directory
    log_dir = resolve(cfg.logging.log_dir)
    writer = None
    if dist.rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))

    # Get surface and volume variable names
    surface_names = (
        list(cfg.variables.surface.solution.keys()) if include_surface else []
    )
    volume_names = list(cfg.variables.volume.solution.keys()) if include_volume else []

    # Create train and val datasets
    train_dataset = create_domino_dataset(
        cfg,
        phase="train",
        volume_variable_names=volume_names,
        surface_variable_names=surface_names,
        vol_factors=volume_scaling,
        surf_factors=surface_scaling,
    )
    val_dataset = create_domino_dataset(
        cfg,
        phase="val",
        volume_variable_names=volume_names,
        surface_variable_names=surface_names,
        vol_factors=volume_scaling,
        surf_factors=surface_scaling,
    )

    # Create distributed samplers
    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=True
        )
        if dist.world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=False
        )
        if dist.world_size > 1
        else None
    )

    # Create model
    num_global_features = count_global_features(
        cfg, future_steps=max(num_future_steps, 1)
    )
    model = DoMINO(
        input_features=3,
        output_features_vol=volume_channels if include_volume else None,
        output_features_surf=surface_channels if include_surface else None,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(device)

    # Distribute model
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if device.type == "cuda" else None,
            output_device=dist.device if device.type == "cuda" else None,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # Initialize optimizer, scheduler, and gradient scaler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    target_lr = float(cfg.train.get("lr_min", cfg.train.lr))
    end_factor = target_lr / cfg.train.lr if cfg.train.lr > 0 else 1.0
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=max(cfg.train.epochs, 1),
    )
    scaler = GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    # Load checkpoint if resume is enabled
    start_epoch = 0
    if cfg.train.resume:
        start_epoch = load_checkpoint(
            cfg.train.checkpoint_dir,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        logger.info(f"Resumed from epoch {start_epoch}")

    # Initialize checkpoint interval and last validation metrics
    checkpoint_interval = max(1, int(cfg.train.get("checkpoint_interval", 1)))
    last_val_metrics: Optional[Dict[str, float]] = None

    # Training loop
    for epoch in range(start_epoch, cfg.train.epochs):
        # Set the epoch in the samplers
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            train_dataset.dataset.set_indices(list(train_sampler))
        else:
            shuffle_indices = torch.randperm(len(train_dataset)).tolist()
            train_dataset.dataset.set_indices(shuffle_indices)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
            val_dataset.dataset.set_indices(list(val_sampler))
        else:
            val_dataset.dataset.set_indices(list(range(len(val_dataset))))

        # Train one epoch
        train_metrics = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            device,
            scaler,
            use_amp=cfg.train.amp and device.type == "cuda",
            surf_weight=cfg.model.surf_loss_scaling,
            vol_weight=cfg.model.vol_loss_scaling,
            logger=logger,
            log_interval=cfg.train.log_interval if cfg.train.log_interval else 5,
        )

        # Evaluate on the validation set
        if (epoch + 1) % cfg.train.val_interval == 0:
            val_metrics = evaluate(
                model,
                val_dataset,
                device,
                use_amp=cfg.train.amp and device.type == "cuda",
            )
            logger.info(
                f"Epoch {epoch + 1:03d} val | loss={val_metrics['loss']:.4e} "
                f"(surf={val_metrics['loss_surface']:.4e}, vol={val_metrics['loss_volume']:.4e})"
            )
            last_val_metrics = val_metrics

        logger.info(
            f"Epoch {epoch + 1:03d} train | loss={train_metrics['loss']:.4e} "
            f"(surf={train_metrics['loss_surface']:.4e}, vol={train_metrics['loss_volume']:.4e})"
        )

        # Log metrics to TensorBoard
        if writer is not None:
            step = epoch + 1
            writer.add_scalars(
                "loss",
                {
                    "train": train_metrics["loss"],
                    **({"val": last_val_metrics["loss"]} if last_val_metrics else {}),
                },
                step,
            )
            writer.add_scalar("loss_surface/train", train_metrics["loss_surface"], step)
            writer.add_scalar("loss_volume/train", train_metrics["loss_volume"], step)
            if last_val_metrics is not None:
                writer.add_scalar(
                    "loss_surface/val", last_val_metrics["loss_surface"], step
                )
                writer.add_scalar(
                    "loss_volume/val", last_val_metrics["loss_volume"], step
                )
            writer.flush()

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if dist.rank == 0 and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                cfg.train.checkpoint_dir,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
            )

    # Log training complete
    logger.info("Training complete.")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
