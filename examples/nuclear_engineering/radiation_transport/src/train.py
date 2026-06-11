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

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.datapipes import DataLoader
from physicsnemo.utils.logging.launch import LaunchLogger

from checkpointing import create_optimizer, resume_if_available
from loader import build_dataloaders, collate_no_padding
from losses import create_scheduler, parse_loss_config
from trainer import (
    parse_amp,
    run_training_loop,
    set_seed,
    setup_training_environment,
    wrap_ddp,
)


def build_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    """Instantiate the Transolver model from the Hydra ``model`` group.

    Two RTE-specific keys (``num_spatial_points``, ``include_q_in_embedding``)
    are stripped from the config before ``hydra.utils.instantiate`` because
    they are consumed by the data pipeline, not the model constructor.
    """
    cfg_model = OmegaConf.to_container(cfg.model, resolve=True)
    for k in ("num_spatial_points", "include_q_in_embedding"):
        cfg_model.pop(k, None)
    return hydra.utils.instantiate(cfg_model).to(device)


def build_dataloaders_for_training(
    cfg: DictConfig, dist: Any, logger: Any
) -> Tuple[DataLoader, DataLoader, Optional[Any]]:
    """Build train / val DataLoaders for the Transolver point-cloud adapter."""
    if cfg.train.dataloader.batch_size != 1:
        raise ValueError(
            "Only batch_size=1 is supported for the Transolver point-cloud adapter."
        )
    loaders, train_sampler = build_dataloaders(
        cfg,
        dist,
        collate_fn=collate_no_padding,
        phases=("train", "val"),
        logger=logger,
    )
    return loaders["train"], loaders["val"], train_sampler


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the Transolver RTE surrogate."""
    dist, logger = setup_training_environment(cfg, "Transolver")

    seed = cfg.train.get("seed", None)
    if seed is not None:
        set_seed(seed + dist.rank if dist.distributed else seed)
        logger.info(f"Random seed: {seed}")
    else:
        logger.info("Random seed: not set (non-reproducible)")

    grad_accum_steps = cfg.train.get("gradient_accumulation_steps", 1)
    use_amp, amp_dtype = parse_amp(cfg)

    amp_info = (
        f"ENABLED (dtype={cfg.train.get('amp_dtype', 'bf16')})"
        if use_amp
        else "DISABLED"
    )
    batch_size = cfg.train.dataloader.batch_size
    world_size = dist.world_size if dist.distributed else 1
    logger.info(f"Device: {dist.device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"AMP (mixed precision): {amp_info}")
    logger.info(f"Effective batch size: {batch_size * grad_accum_steps * world_size}")

    train_loader, val_loader, _ = build_dataloaders_for_training(cfg, dist, logger)

    logger.info("\nInitializing Transolver model...")
    model = build_model(cfg, dist.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Transolver initialized — {num_params:,} trainable parameters")
    model = wrap_ddp(model, dist, logger)

    optimizer_cfg = cfg.train.get("optimizer", {})
    optimizer = create_optimizer(
        model=model,
        optimizer_type=optimizer_cfg.get("type", "adam"),
        learning_rate=cfg.train.learning_rate,
        weight_decay=optimizer_cfg.get(
            "weight_decay", cfg.train.get("weight_decay", 0.0)
        ),
        muon_momentum_beta=optimizer_cfg.get("muon_momentum_beta", 0.95),
        logger=logger,
    )
    scheduler = create_scheduler(cfg, optimizer, logger)
    # GradScaler is only meaningful for fp16 AMP; bf16 doesn't underflow and
    # disabling avoids the overhead + masks fp16-specific failure modes.
    scaler = GradScaler(enabled=use_amp and amp_dtype is torch.float16)
    LaunchLogger.initialize(use_wandb=False, use_mlflow=False)
    use_tensorboard = cfg.train.get("tensorboard", True)
    writer = (
        SummaryWriter(os.path.join(cfg.output, "tensorboard"))
        if (use_tensorboard and dist.rank == 0)
        else None
    )
    checkpoint_dir = os.path.join(cfg.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_cfg = parse_loss_config(cfg, dist, logger)
    loss_metric = cfg.train.get("loss_metric", "mse")
    loss_cfg["loss_metric"] = loss_metric
    logger.info(f"Loss metric: {loss_metric}")

    start_epoch, best_val_loss = resume_if_available(
        cfg, model, optimizer, scheduler, scaler, dist, logger
    )

    logger.info("\n" + "=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    run_training_loop(
        cfg=cfg,
        dist=dist,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_cfg=loss_cfg,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        writer=writer,
        best_val_loss=best_val_loss,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
