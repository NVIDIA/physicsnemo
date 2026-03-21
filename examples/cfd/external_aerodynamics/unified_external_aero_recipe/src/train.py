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

"""
Unified External Aerodynamics - Surface Training Script

Trains a GeoTransolver (or other point-cloud model) on surface pressure
and wall shear stress using the mesh datapipe infrastructure.

Usage::

    # Single-GPU
    python src/train.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=N src/train.py
"""

import os
import sys
import time
import collections
from contextlib import nullcontext
from pathlib import Path

import hydra
import omegaconf
from omegaconf import DictConfig

import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from tabulate import tabulate

from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.profiling import profile, Profiler

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver

from datasets import build_surface_dataset, load_dataset_config
from collate import surface_collate
from metrics import MetricCalculator
from loss import LossCalculator

from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)

if TE_AVAILABLE:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
else:
    te, Format, DelayedScaling = None, None, None

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])


def get_autocast_context(precision: str):
    """Return an autocast context manager for the given precision.

    Parameters
    ----------
    precision : str
        One of ``"float16"``, ``"bfloat16"``, ``"float8"``, or ``"float32"``.
        For ``"float8"``, Transformer Engine must be available.

    Returns
    -------
    contextlib.AbstractContextManager
        An autocast context manager for the requested precision, or a
        no-op ``nullcontext`` when no casting is needed.
    """
    if precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    elif precision == "float8" and TE_AVAILABLE:
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        return nullcontext()


def forward_pass(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    precision: str,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
) -> tuple[torch.Tensor, dict[str, float], tuple]:
    """Run forward pass, compute loss and metrics.

    Parameters
    ----------
    batch : dict
        Keys: ``geometry`` (B,N,3), ``local_embedding`` (B,N,3),
        ``global_embedding`` (B,1,3), ``fields`` (B,N,4).
    model : torch.nn.Module
        Point-cloud model with forward(local_embedding, geometry, global_embedding, ...).
    precision : str
        One of "float32", "float16", "bfloat16", "float8".
    loss_calculator : LossCalculator
    metric_calculator : MetricCalculator

    Returns
    -------
    loss, metrics_dict, (outputs, targets)
    """
    targets = batch.pop("fields")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(precision)
    if dtype is not None:
        batch = {k: v.to(dtype) for k, v in batch.items()}

    with get_autocast_context(precision):
        outputs = model(**batch)
        loss, loss_dict = loss_calculator(outputs, targets)

    metrics = {k: v.item() for k, v in loss_dict.items()}
    with torch.no_grad():
        metrics.update(metric_calculator(outputs, targets))

    return loss, metrics, (outputs, targets)


@profile
def train_epoch(
    dataloader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger,
    writer,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    precision = getattr(cfg, "precision", "float32")
    n_batches = 0

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(dist_manager.device) for k, v in batch.items()}

        loss, metrics, _ = forward_pass(
            batch, model, precision, loss_calculator, metric_calculator
        )

        optimizer.zero_grad()
        if precision == "float16" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if cfg.training.get("scheduler_update_mode", "epoch") == "step":
            scheduler.step()

        this_loss = loss.detach().item()
        total_loss += this_loss
        n_batches += 1

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + (
                v if isinstance(v, float) else v.item()
            )

        mem_gb = (
            torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        )
        logger.info(f"Epoch {epoch} [{i}] Loss: {this_loss:.6f} Mem: {mem_gb:.2f}GB")

        if dist_manager.rank == 0 and writer is not None:
            step = i + n_batches * epoch
            writer.add_scalar("batch/loss", this_loss, step)
            writer.add_scalar("batch/lr", optimizer.param_groups[0]["lr"], step)

        if cfg.profile and i >= 10:
            break

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    if dist_manager.rank == 0 and writer is not None:
        writer.add_scalar("epoch/train_loss", avg_loss, epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f"epoch/{k}", v, epoch)
        table = tabulate(
            [[k, f"{v:.6f}"] for k, v in avg_metrics.items()],
            headers=["Metric", "Value"],
            tablefmt="pretty",
        )
        logger.info(f"\nEpoch {epoch} Train Metrics:\n{table}\n")

    return avg_loss


@profile
def val_epoch(
    dataloader,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger,
    writer,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
) -> float:
    model.eval()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    precision = getattr(cfg, "precision", "float32")
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(dist_manager.device) for k, v in batch.items()}

            loss, metrics, _ = forward_pass(
                batch, model, precision, loss_calculator, metric_calculator
            )

            total_loss += loss.item()
            n_batches += 1
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + (
                    v if isinstance(v, float) else v.item()
                )

            if cfg.profile and i >= 10:
                break

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    if dist_manager.rank == 0 and writer is not None:
        writer.add_scalar("epoch/val_loss", avg_loss, epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f"epoch/val_{k}", v, epoch)
        table = tabulate(
            [[k, f"{v:.6f}"] for k, v in avg_metrics.items()],
            headers=["Metric", "Value"],
            tablefmt="pretty",
        )
        logger.info(f"\nEpoch {epoch} Val Metrics:\n{table}\n")

    return avg_loss


def build_dataloaders(cfg: DictConfig):
    """Build train and val dataloaders from dataset configs."""
    recipe_root = Path(__file__).resolve().parent.parent
    batch_size = cfg.training.get("batch_size", 1)
    num_workers = cfg.training.get("num_workers", 0)

    datasets = []
    for ds_key in cfg.data:
        ds_cfg_block = cfg.data[ds_key]
        config_path = recipe_root / ds_cfg_block.config
        if not config_path.exists():
            continue
        train_dir = ds_cfg_block.get("train_dir", "")
        if train_dir and not Path(train_dir).exists():
            continue
        ds_yaml = load_dataset_config(config_path)
        datasets.append(build_surface_dataset(ds_yaml))

    if not datasets:
        raise RuntimeError("No valid datasets found. Check data paths in config.")

    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        from physicsnemo.datapipes import MultiDataset

        train_dataset = MultiDataset(*datasets, output_strict=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=surface_collate,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=surface_collate,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


@profile
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir

    writer = None
    val_writer = None
    if dist_manager.rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(
            log_dir=os.path.join(cfg.output_dir, cfg.run_id, "train")
        )
        val_writer = SummaryWriter(
            log_dir=os.path.join(cfg.output_dir, cfg.run_id, "val")
        )

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"Model: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")

    model.to(dist_manager.device)

    if dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=dist_manager.device,
        )

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train samples: {len(train_loader.dataset)}")

    if dist_manager.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_loader.dataset, shuffle=True, drop_last=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=cfg.training.get("batch_size", 1),
            sampler=train_sampler,
            num_workers=cfg.training.get("num_workers", 0),
            collate_fn=surface_collate,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_sampler = None

    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer, params=model.parameters()
    )
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    precision = cfg.precision
    scaler = GradScaler() if precision == "float16" else None

    ds_cfg = cfg.dataset
    targets = omegaconf.OmegaConf.to_container(ds_cfg.targets, resolve=True)
    metrics_list = omegaconf.OmegaConf.to_container(
        ds_cfg.get("metrics", ["l1", "l2", "mae"]), resolve=True
    )

    metric_calculator = MetricCalculator(
        target_config=targets, metrics=metrics_list, prefix=ds_cfg.name
    )
    loss_calculator = LossCalculator(
        target_config=targets,
        loss_type=cfg.training.get("loss_type", "huber"),
        prefix=ds_cfg.name,
    )
    logger.info(f"Loss: {loss_calculator}")
    logger.info(f"Metrics: {metric_calculator}")

    ckpt_args = {
        "path": os.path.join(checkpoint_dir, cfg.run_id, "checkpoints"),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)

    if cfg.compile:
        model = torch.compile(model)

    logger.info("Starting training...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            loss_calculator,
            metric_calculator,
            logger,
            writer,
            epoch,
            cfg,
            dist_manager,
            scaler,
        )

        val_loss = val_epoch(
            val_loader,
            model,
            loss_calculator,
            metric_calculator,
            logger,
            val_writer,
            epoch,
            cfg,
            dist_manager,
        )

        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] "
            f"Train: {train_loss:.6f}  Val: {val_loss:.6f}"
        )

        if epoch % cfg.training.save_interval == 0 and dist_manager.rank == 0:
            save_checkpoint(**ckpt_args, epoch=epoch + 1)

        if cfg.training.get("scheduler_update_mode", "epoch") == "epoch":
            scheduler.step()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="../conf", config_name="train_surface")
def launch(cfg: DictConfig):
    profiler = Profiler()
    if cfg.profile:
        profiler.enable("torch")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
