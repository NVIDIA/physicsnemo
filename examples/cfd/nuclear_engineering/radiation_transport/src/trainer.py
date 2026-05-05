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

"""Training loop, epoch step, DDP primitives, and environment setup.

Consolidates the per-batch loss composition + gradient-accumulation step,
the epoch-driven training loop with checkpointing, and the DDP / environment
boilerplate that sits beside them:

* DDP primitives — ``set_seed``, ``setup_training_environment``, ``wrap_ddp``,
  ``log_effective_batch_size``, ``synchronize_output_directory``,
  ``aggregate_validation_loss``.
* Per-step / per-epoch helpers — ``compute_losses``, ``grad_step``,
  ``flush_partial_accumulation``.
* Training loop — ``run_training_loop``.

Optimizer / scheduler construction and checkpoint save/load live in
``checkpointing.py`` and ``losses.py`` respectively.
"""

import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import save_checkpoint
from physicsnemo.utils.logging.launch import LaunchLogger

from checkpointing import (
    save_best_checkpoint,
    save_best_qoi_checkpoint,
    save_latest_checkpoint,
)
from loader import set_epoch_on_transforms
from losses import compute_physics_loss, masked_mse_loss, region_weighted_loss_fn


# =========================================================================
# DDP primitives & environment setup
# =========================================================================


def _setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a console (and optional file) logger with a consistent format."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate logs from Hydra
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility across all RNGs.

    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def synchronize_output_directory(
    cfg: DictConfig,
    dist: DistributedManager,
) -> str:
    """Synchronize the output directory across DDP ranks via torch broadcast.

    Hydra's timestamp-based output paths can otherwise produce one folder per
    rank. Rank 0 creates the directory; the resolved path is broadcast from
    rank 0 to every other rank, which updates ``cfg.output`` in place if it
    differs and ensures the directory exists locally. Ends with a barrier so
    no rank proceeds before the directory is in place.

    Args:
        cfg: Hydra configuration with an ``output`` field.
        dist: DistributedManager instance.

    Returns:
        The synchronized output directory path.
    """
    if "output" not in cfg:
        output_dir = os.path.join("outputs", "default")
        OmegaConf.set_struct(cfg, False)
        cfg.output = output_dir
        OmegaConf.set_struct(cfg, True)

    output_dir = cfg.output

    if not dist.distributed:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    # By the time this is called, DistributedManager.initialize() has run, so
    # torch_dist.is_initialized() is True for distributed runs.
    if dist.rank == 0:
        print(f"[Rank 0] Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    payload = [output_dir]
    torch_dist.broadcast_object_list(payload, src=0)
    synced_output_dir = payload[0]

    if dist.rank != 0:
        if synced_output_dir != output_dir:
            print(f"[Rank {dist.rank}] Syncing to output: {synced_output_dir}")
            OmegaConf.set_struct(cfg, False)
            cfg.output = synced_output_dir
            OmegaConf.set_struct(cfg, True)
            output_dir = synced_output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Rank {dist.rank}] Synchronized to output directory: {output_dir}")

    torch_dist.barrier()
    return output_dir


def aggregate_validation_loss(
    loss_sum: float,
    num_batches: int,
    dist: DistributedManager,
) -> Tuple[float, int]:
    """Aggregate validation loss across all DDP ranks.

    Sums the per-rank loss totals and batch counts then returns the global
    mean. In single-GPU mode this reduces to ``loss_sum / num_batches``.

    Args:
        loss_sum: Sum of losses on this rank.
        num_batches: Number of batches processed on this rank.
        dist: DistributedManager instance.

    Returns:
        ``(global_mean_loss, global_num_batches)``.
    """
    if not dist.distributed:
        return loss_sum / max(num_batches, 1), num_batches

    loss_tensor = torch.tensor([loss_sum], dtype=torch.float64, device=dist.device)
    torch_dist.all_reduce(loss_tensor, op=torch_dist.ReduceOp.SUM)

    count_tensor = torch.tensor([num_batches], dtype=torch.int64, device=dist.device)
    torch_dist.all_reduce(count_tensor, op=torch_dist.ReduceOp.SUM)

    global_loss_sum = loss_tensor.item()
    global_num_batches = count_tensor.item()

    return global_loss_sum / max(global_num_batches, 1), global_num_batches


def aggregate_validation_metrics(
    metric_sums: Mapping[str, float],
    metric_counts: Mapping[str, int],
    dist: DistributedManager,
) -> Dict[str, float]:
    """Aggregate named validation metrics across ranks."""
    if not dist.distributed:
        return {
            key: metric_sums[key] / metric_counts[key]
            for key in metric_sums
            if metric_counts.get(key, 0) > 0
        }

    gathered = [None for _ in range(dist.world_size)]
    torch_dist.all_gather_object(
        gathered,
        (dict(metric_sums), dict(metric_counts)),
    )

    total_sums: Dict[str, float] = {}
    total_counts: Dict[str, int] = {}
    for rank_sums, rank_counts in gathered:
        for key, value in rank_sums.items():
            total_sums[key] = total_sums.get(key, 0.0) + float(value)
        for key, value in rank_counts.items():
            total_counts[key] = total_counts.get(key, 0) + int(value)

    return {
        key: total_sums[key] / total_counts[key]
        for key in total_sums
        if total_counts.get(key, 0) > 0
    }


def setup_training_environment(
    cfg: DictConfig,
    model_name: str,
) -> Tuple[DistributedManager, Any]:
    """Initialize DDP, sync the output dir, build a logger, and log a banner.

    Args:
        cfg: Hydra configuration.
        model_name: Human-readable model name for logging (e.g. "Transolver").

    Returns:
        ``(dist, logger)``.
    """
    initialize_distributed_manager()
    dist = DistributedManager()

    synchronize_output_directory(cfg, dist)

    log_file = os.path.join(cfg.output, "train.log") if dist.rank == 0 else None
    logger = _setup_logger(f"RTE_{model_name}", log_file)

    if dist.rank == 0:
        logger.info("=" * 70)
        logger.info(f"RTE {model_name} Training - {cfg.case.type.upper()}")
        logger.info("=" * 70)
        if dist.distributed:
            logger.info(f"Distributed training: {dist.world_size} GPUs")
        logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}\n")

    return dist, logger


def initialize_distributed_manager() -> None:
    """Initialize distributed state without misreading an interactive SLURM shell.

    PhysicsNeMo's default initializer auto-detects SLURM variables. In an
    allocated shell, plain ``python src/train.py`` can inherit those variables
    even though only one Python process was launched, causing process-group
    setup to wait for ranks that do not exist. For this example, DDP should be
    entered via ``torchrun`` (or an explicit PhysicsNeMo init method); otherwise
    run as a normal single process.
    """
    if DistributedManager.is_initialized():
        return

    explicit_method = os.getenv("PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD")
    torchrun_env = os.getenv("RANK") is not None and os.getenv("WORLD_SIZE") is not None
    openmpi_env = os.getenv("OMPI_COMM_WORLD_RANK") is not None

    if explicit_method or torchrun_env or openmpi_env:
        DistributedManager.initialize()
        return

    DistributedManager._shared_state["_is_initialized"] = True
    dist = DistributedManager()
    dist._initialization_method = "single"
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.device)


def wrap_ddp(
    model: nn.Module,
    dist: DistributedManager,
    logger: Any,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap ``model`` with DistributedDataParallel if running distributed.

    Returns the unwrapped model in single-GPU mode.
    """
    if not dist.distributed:
        return model

    ddps = torch.cuda.Stream()
    with torch.cuda.stream(ddps):
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
        )
    torch.cuda.current_stream().wait_stream(ddps)

    if dist.rank == 0:
        fup = " (find_unused_parameters=True)" if find_unused_parameters else ""
        logger.info(f"Using DistributedDataParallel with {dist.world_size} GPUs{fup}")

    return model


def log_effective_batch_size(
    cfg: DictConfig,
    dist: DistributedManager,
    logger: Any,
    grad_accum_steps: int,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Log device, batch size, gradient accumulation, and effective batch size."""
    if dist.rank != 0:
        return

    logger.info(f"Device: {dist.device}")
    logger.info(f"Batch size: {cfg.train.dataloader.batch_size}")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")

    if extra_info:
        for key, value in extra_info.items():
            logger.info(f"{key}: {value}")

    world_mult = dist.world_size if dist.distributed else 1
    effective_batch = cfg.train.dataloader.batch_size * grad_accum_steps * world_mult
    logger.info(f"Effective batch size: {effective_batch}")


# =========================================================================
# Per-step / per-epoch helpers
# =========================================================================


def compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_inputs: Mapping[str, Any],
    loss_cfg: Mapping[str, Any],
    case_type: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
    """Compose the per-batch training loss.

    Args:
        pred, target: ``(B, N, 1)`` tensors.
        loss_inputs: presence-driven dispatch dict. Recognized keys:
            - ``padding_mask`` ``(B, N)``: enables masked MSE.
            - ``material_labels`` ``(B, N)`` or ``(B, N, 1)``: enables
              region-weighted loss when ``loss_cfg['use_region_weighted_loss']``.
            - ``coordinates_unnormalized`` ``(B, N, D)``, ``cell_areas``
              ``(B, N)``, ``sigma_t`` ``(B, N)``, ``sigma_s`` ``(B, N)``,
              ``sim_time`` ``(B,)`` or ``(B, 1)``: required for physics loss.
            - ``metadata``, ``flux_normalization_stats``: optional physics
              context.
        loss_cfg: ``use_region_weighted_loss``, ``region_weight_cfg``,
            ``loss_metric`` ("mse"|"rmse"), ``use_physics_loss``,
            ``physics_loss_weight``, ``physics_loss_mse_weight``, ``qoi_region``.

    Returns:
        ``(loss, loss_mse, loss_qoi_or_None, qoi_details_dict)``.
    """
    use_region_weighted = loss_cfg.get("use_region_weighted_loss", False)
    loss_metric = loss_cfg.get("loss_metric", "mse")

    if use_region_weighted and "material_labels" in loss_inputs:
        rw = loss_cfg.get("region_weight_cfg") or {}
        loss_mse = region_weighted_loss_fn(
            pred,
            target,
            material_labels=loss_inputs["material_labels"],
            case_type=case_type,
            loss_type=loss_metric,
            padded_value=-10,
            void_weight=rw.get("void_weight", 3.0),
            material_weight=rw.get("material_weight", 1.0),
        )
    else:
        loss_mse = masked_mse_loss(pred, target, loss_inputs.get("padding_mask"))
        if loss_metric == "rmse":
            loss_mse = torch.sqrt(loss_mse)

    if not loss_cfg.get("use_physics_loss", False):
        return loss_mse, loss_mse, None, {}

    physics_w = loss_cfg.get("physics_loss_weight", 0.1)
    if not physics_w:
        # Zero (or missing/None) weight -> physics loss is disabled; skip the
        # QoI computation entirely.
        return loss_mse, loss_mse, None, {}

    with autocast(enabled=False, device_type=device.type):
        loss_qoi, qoi_details = compute_physics_loss(
            case_type=case_type,
            predicted_flux=pred,
            target_flux=target,
            cell_centers=loss_inputs["coordinates_unnormalized"],
            cell_areas=loss_inputs["cell_areas"],
            sigma_t=loss_inputs["sigma_t"],
            sigma_s=loss_inputs["sigma_s"],
            sim_time=loss_inputs["sim_time"],
            metadata=loss_inputs.get("metadata"),
            flux_normalization_stats=loss_inputs.get("flux_normalization_stats"),
            qoi_region=loss_cfg.get("qoi_region", "center"),
        )

    mse_w = loss_cfg.get("physics_loss_mse_weight", 1.0)
    loss = mse_w * loss_mse + physics_w * loss_qoi
    return loss, loss_mse, loss_qoi, qoi_details


def _finalize_step(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    max_grad_norm: float,
) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _scale_pending_gradients(model: torch.nn.Module, factor: float) -> None:
    """Scale accumulated gradients in-place before optimizer finalization."""
    if factor == 1.0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def grad_step(
    loss: torch.Tensor,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    *,
    step_idx: int,
    accum_steps: int,
    max_grad_norm: float = 10.0,
) -> None:
    """Scale, backward, and (on accumulation boundary) step + clip + zero_grad.

    Callers invoke this every batch with ``step_idx=i``. Trailing partial
    accumulation windows are flushed by a separate call to
    ``flush_partial_accumulation`` after the loop finishes.
    """
    scaler.scale(loss / accum_steps).backward()
    if (step_idx + 1) % accum_steps != 0:
        return
    _finalize_step(scaler, optimizer, model, max_grad_norm)


def flush_partial_accumulation(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    *,
    total_steps: int,
    accum_steps: int,
    max_grad_norm: float = 10.0,
) -> None:
    """Flush leftover gradients when ``total_steps % accum_steps != 0``."""
    remainder = total_steps % accum_steps
    if remainder == 0:
        return
    _scale_pending_gradients(model, accum_steps / remainder)
    _finalize_step(scaler, optimizer, model, max_grad_norm)


# =========================================================================
# Training loop
# =========================================================================


def _coerce_optional_checkpoint_interval(value: Any) -> Optional[int]:
    """Parse an optional checkpoint cadence; None/0 disables the feature."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None

    interval = int(value)
    if interval < 0:
        raise ValueError("latest_checkpoint_interval must be >= 0 or null")
    return interval


def _finite_metric_values(metrics: Mapping[str, Optional[float]]) -> bool:
    """Return True when all present metric values are finite."""
    for value in metrics.values():
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return False
    return True


def run_training_loop(
    cfg: DictConfig,
    dist: Any,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: Optional[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_epoch_fn: Callable[..., None],
    validate_fn: Callable[..., tuple],
    train_epoch_kwargs: Dict[str, Any],
    validate_kwargs: Dict[str, Any],
    logger: Any,
    checkpoint_dir: str,
    writer: Optional[Any],
    best_val_losses: List,
    start_epoch: int,
    case_type: str,
    before_epoch_fn: Optional[Callable[[int], tuple]] = None,
    after_epoch_fn: Optional[Callable[[int, Any, Any, float, float], None]] = None,
    finally_fn: Optional[Callable[[], None]] = None,
    best_qoi_loss: float = float("inf"),
) -> None:
    """Run the main training loop: epochs, validation, checkpointing, logging.

    The caller owns model / dataloader / optimizer / scheduler / scaler
    construction and supplies ``train_epoch_fn`` and ``validate_fn``. This
    function drives the epoch loop, aggregates validation loss across DDP
    ranks, steps the scheduler, saves checkpoints, and handles graceful
    completion / interrupt.

    Args:
        cfg: Hydra config (uses ``train.epochs``, ``train.checkpoint_interval``,
            ``train.scheduler_type``).
        dist: DistributedManager instance.
        model: Model (possibly DDP-wrapped).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        train_sampler: DistributedSampler for training (or None).
        optimizer: Optimizer.
        scheduler: LR scheduler.
        scaler: GradScaler for AMP.
        train_epoch_fn: ``(train_loader, model, optimizer, scaler, device,
            launch_logger, **train_epoch_kwargs) -> None``.
        validate_fn: ``(val_loader, model, device, launch_logger,
            **validate_kwargs) -> (val_loss_sum, val_num_batches)``.
        train_epoch_kwargs: kwargs passed to ``train_epoch_fn``.
        validate_kwargs: kwargs passed to ``validate_fn``.
        logger: Logger (rank 0).
        checkpoint_dir: Directory for checkpoints.
        writer: TensorBoard SummaryWriter (rank 0) or None.
        best_val_losses: List of best validation losses (updated in place).
        start_epoch: First epoch index to run.
        case_type: Case type string for metadata (e.g. "lattice", "hohlraum").
        before_epoch_fn: Optional ``(epoch) -> (extra_train_kwargs,
            extra_validate_kwargs)``. Used by callers that want to inject
            per-epoch state (e.g. physics-loss warmup weights).
        after_epoch_fn: Optional ``(epoch, train_log, val_log, val_loss,
            current_lr) -> None`` for custom rank-0 logging.
        finally_fn: Optional no-arg callback called at the start of the
            ``finally`` block (e.g. memory diagnostics).
        best_qoi_loss: Best QoI loss seen so far (lower is better).
    """
    training_completed = False
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # Propagate epoch to spatial samplers / other stochastic transforms
            # so each rank reshuffles deterministically per epoch.
            set_epoch_on_transforms(train_loader, epoch)
            set_epoch_on_transforms(val_loader, epoch)

            train_kw = dict(train_epoch_kwargs)
            val_kw = dict(validate_kwargs)
            if before_epoch_fn is not None:
                extra_train, extra_val = before_epoch_fn(epoch)
                train_kw.update(extra_train)
                val_kw.update(extra_val)

            with LaunchLogger(
                "train",
                epoch=epoch,
                num_mini_batch=len(train_loader),
                mini_batch_log_freq=10,
            ) as train_log:
                train_epoch_fn(
                    train_loader,
                    model,
                    optimizer,
                    scaler,
                    dist.device,
                    train_log,
                    **train_kw,
                )

            with LaunchLogger(
                "val", epoch=epoch, num_mini_batch=len(val_loader)
            ) as val_log:
                validation_result = validate_fn(
                    val_loader,
                    model,
                    dist.device,
                    val_log,
                    **val_kw,
                )
                if len(validation_result) == 2:
                    val_loss_sum, val_num_batches = validation_result
                    val_metric_sums: Dict[str, float] = {}
                    val_metric_counts: Dict[str, int] = {}
                else:
                    (
                        val_loss_sum,
                        val_num_batches,
                        val_metric_sums,
                        val_metric_counts,
                    ) = validation_result

            train_loss = train_log.epoch_losses.get("loss", 0.0)
            val_loss, _ = aggregate_validation_loss(val_loss_sum, val_num_batches, dist)
            val_metrics = aggregate_validation_metrics(
                val_metric_sums, val_metric_counts, dist
            )
            val_log.epoch_losses.update(val_metrics)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            if dist.rank == 0:
                if after_epoch_fn is not None:
                    after_epoch_fn(epoch, train_log, val_log, val_loss, current_lr)
                else:
                    logger.info(
                        f"Epoch {epoch}: train_loss={train_loss:.4e}, "
                        f"val_loss={val_loss:.4e}, lr={current_lr:.2e}"
                    )
                if writer:
                    writer.add_scalar("Loss/train", train_loss, epoch)
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    writer.add_scalar("Learning_Rate", current_lr, epoch)
                    writer.flush()

                val_loss_qoi = val_metrics.get("loss_qoi")
                metadata_best_qoi_loss = best_qoi_loss
                if val_loss_qoi is not None:
                    metadata_best_qoi_loss = min(best_qoi_loss, val_loss_qoi)

                checkpoint_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_loss_qoi": val_loss_qoi,
                }
                can_write_checkpoint = _finite_metric_values(checkpoint_metrics)
                if not can_write_checkpoint:
                    logger.warning(
                        "Skipping checkpoint saves for epoch %s because at least "
                        "one checkpoint metric is NaN or inf: %s",
                        epoch,
                        checkpoint_metrics,
                    )
                else:
                    best_val_losses[:] = save_best_checkpoint(
                        checkpoint_dir=Path(checkpoint_dir),
                        epoch=epoch,
                        val_loss=val_loss,
                        best_val_losses=best_val_losses,
                        save_checkpoint_fn=save_checkpoint,
                        logger=logger,
                        models=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        metadata={
                            "best_val_losses": best_val_losses,
                            "best_qoi_loss": metadata_best_qoi_loss,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_loss_qoi": val_loss_qoi,
                            "case_type": case_type,
                        },
                    )

                    if val_loss_qoi is not None:
                        best_qoi_loss = save_best_qoi_checkpoint(
                            checkpoint_dir=Path(checkpoint_dir),
                            epoch=epoch,
                            qoi_error=val_loss_qoi,
                            best_qoi_error=best_qoi_loss,
                            save_checkpoint_fn=save_checkpoint,
                            logger=logger,
                            models=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            metadata={
                                "best_val_losses": best_val_losses,
                                "best_qoi_loss": metadata_best_qoi_loss,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "val_loss_qoi": val_loss_qoi,
                                "case_type": case_type,
                            },
                        )

                    if epoch % cfg.train.checkpoint_interval == 0:
                        save_checkpoint(
                            path=checkpoint_dir,
                            models=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            epoch=epoch,
                            metadata={
                                "best_val_losses": best_val_losses,
                                "best_qoi_loss": best_qoi_loss,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "val_loss_qoi": val_loss_qoi,
                                "case_type": case_type,
                            },
                        )
                        logger.info(f"  Saved checkpoint at epoch {epoch + 1}")

                    latest_checkpoint_interval = _coerce_optional_checkpoint_interval(
                        cfg.train.get("latest_checkpoint_interval", 1)
                    )
                    if latest_checkpoint_interval and (
                        epoch % latest_checkpoint_interval == 0
                    ):
                        save_latest_checkpoint(
                            checkpoint_dir=Path(checkpoint_dir),
                            epoch=epoch,
                            save_checkpoint_fn=save_checkpoint,
                            logger=logger,
                            models=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            metadata={
                                "best_val_losses": best_val_losses,
                                "best_qoi_loss": best_qoi_loss,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "val_loss_qoi": val_loss_qoi,
                                "case_type": case_type,
                            },
                        )

                if val_loss_qoi is not None and writer:
                    writer.add_scalar("Loss/val_qoi", val_loss_qoi, epoch)

            if dist.distributed:
                torch_dist.barrier()

        training_completed = True

    except KeyboardInterrupt:
        training_completed = False
        if dist.rank == 0:
            logger.info("\n" + "=" * 70)
            logger.info("Training interrupted by user")
            logger.info("=" * 70)
        # Re-raise so the process exits non-zero and callers can distinguish
        # an interrupted run from a clean finish.
        raise

    finally:
        if finally_fn is not None:
            finally_fn()
        if writer:
            writer.close()

        if dist.rank == 0:
            if training_completed:
                logger.info("\n" + "=" * 70)
                logger.info("Training completed!")
                loss_strs = [f"{loss:.6f}" for loss, _ in best_val_losses]
                logger.info(f"Top validation losses: {loss_strs}")
                if best_qoi_loss < float("inf"):
                    logger.info(f"Best QoI loss: {best_qoi_loss:.6e}")
                logger.info(f"Checkpoints saved to: {checkpoint_dir}")
                logger.info("=" * 70)

                completion_marker = os.path.join(checkpoint_dir, ".training_complete")
                with open(completion_marker, "w") as f:
                    f.write(f"completed_epochs={cfg.train.epochs}\n")
                    f.write(f"target_epochs={cfg.train.epochs}\n")
                logger.info(f"Training complete marker written to: {completion_marker}")
            else:
                logger.info("\n" + "=" * 70)
                logger.info("Training interrupted (no completion marker written)")
                logger.info("=" * 70)
