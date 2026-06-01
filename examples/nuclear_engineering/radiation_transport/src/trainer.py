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

import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.utils import reduce_loss
from physicsnemo.utils.checkpoint import save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.launch import LaunchLogger

from checkpointing import save_best_checkpoint
from losses import (
    compute_physics_loss,
    physics_loss_weight_for_epoch,
    region_weighted_loss_fn,
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_AMP_DTYPES: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def parse_amp(cfg: DictConfig) -> Tuple[bool, torch.dtype]:
    """Read ``cfg.train.amp`` and ``cfg.train.amp_dtype`` into ``(use_amp, dtype)``."""
    name = cfg.train.get("amp_dtype", "bf16")
    if name not in _AMP_DTYPES:
        raise ValueError(
            f"Unsupported amp_dtype {name!r}; allowed: {sorted(_AMP_DTYPES)}."
        )
    return cfg.train.get("amp", True), _AMP_DTYPES[name]


def synchronize_output_directory(
    cfg: DictConfig,
    dist: DistributedManager,
) -> str:
    """Ensure ``cfg.output`` exists; barrier so DDP ranks don't race past it.

    Rank 0 creates the directory tree;
    a final barrier keeps the other ranks from proceeding before it lands.
    """
    if "output" not in cfg:
        OmegaConf.set_struct(cfg, False)
        cfg.output = os.path.join("outputs", "default")
        OmegaConf.set_struct(cfg, True)

    output_dir = cfg.output
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    if dist.distributed:
        torch_dist.barrier()
    return output_dir


def aggregate_validation_loss(
    loss_sum: float,
    num_batches: int,
    dist: DistributedManager,
) -> float:
    """Aggregate validation loss across DDP ranks via ``reduce_loss``.

    Returns the rank-0 mean-of-means; non-rank-0 ranks get their local mean
    (unused downstream). Eval sampler pads the split to equal length across
    ranks, so the mean-of-means equals the global mean up to at most
    ``world_size - 1`` duplicate samples.
    """
    per_rank_mean = loss_sum / max(num_batches, 1)
    if not dist.distributed:
        return per_rank_mean
    reduced = reduce_loss(per_rank_mean, dst_rank=0, mean=True)
    return reduced if reduced is not None else per_rank_mean


def aggregate_validation_metrics(
    metric_sums: Mapping[str, float],
    metric_counts: Mapping[str, int],
    dist: DistributedManager,
) -> Dict[str, float]:
    """Aggregate named validation metrics via tensor ``all_reduce`` over a
    known schema.

    Every rank emits the same metric keys (the schema is fixed at config
    time, not per-batch), so we sort keys, stack values into a single
    tensor, and issue one collective per (sums, counts) pair.
    """
    if not dist.distributed:
        return {
            key: metric_sums[key] / metric_counts[key]
            for key in metric_sums
            if metric_counts.get(key, 0) > 0
        }

    keys = sorted(metric_sums.keys())
    if not keys:
        return {}

    sums = torch.tensor(
        [float(metric_sums[k]) for k in keys],
        dtype=torch.float64,
        device=dist.device,
    )
    counts = torch.tensor(
        [int(metric_counts.get(k, 0)) for k in keys],
        dtype=torch.int64,
        device=dist.device,
    )
    torch_dist.all_reduce(sums, op=torch_dist.ReduceOp.SUM)
    torch_dist.all_reduce(counts, op=torch_dist.ReduceOp.SUM)

    return {
        key: float(sums[i].item() / counts[i].item())
        for i, key in enumerate(keys)
        if counts[i].item() > 0
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
    DistributedManager.initialize()
    dist = DistributedManager()

    synchronize_output_directory(cfg, dist)

    logger = RankZeroLoggingWrapper(PythonLogger(f"RTE_{model_name}"), dist)
    if dist.rank == 0:
        logger.file_logging(os.path.join(cfg.output, "train.log"))

    logger.info("=" * 70)
    logger.info(f"RTE {model_name} Training - {cfg.case.type.upper()}")
    logger.info("=" * 70)
    if dist.distributed:
        logger.info(f"Distributed training: {dist.world_size} GPUs")
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}\n")

    return dist, logger


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

    fup = " (find_unused_parameters=True)" if find_unused_parameters else ""
    logger.info(f"Using DistributedDataParallel with {dist.world_size} GPUs{fup}")
    return model


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
            - ``material_labels`` ``(B, N)`` or ``(B, N, 1)``: enables
              region-weighted loss when ``loss_cfg['use_region_weighted_loss']``.
            - ``coordinates_unnormalized`` ``(B, N, D)``, ``cell_areas``
              ``(B, N)``, ``sigma_t`` ``(B, N)``, ``sigma_s`` ``(B, N)``,
              ``sim_time`` ``(B,)`` or ``(B, 1)``: required for physics loss.
            - ``metadata``, ``flux_normalization_stats``: optional physics
              context.
        loss_cfg: ``use_region_weighted_loss``, ``region_weight_cfg``,
            ``loss_metric`` ("mse"|"rmse"), ``use_physics_loss``,
            ``physics_loss_weight``, ``physics_loss_mse_weight``.

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
            void_weight=rw.get("void_weight", 3.0),
            material_weight=rw.get("material_weight", 1.0),
        )
    else:
        loss_mse = ((pred - target) ** 2).mean()
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
            sample=loss_inputs,
            flux_normalization_stats=loss_inputs.get("flux_normalization_stats"),
        )

    mse_w = loss_cfg.get("physics_loss_mse_weight", 1.0)
    loss = mse_w * loss_mse + physics_w * loss_qoi
    return loss, loss_mse, loss_qoi, qoi_details


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor entries of a batch dict to ``device``; pass through the rest."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def forward(
    model: nn.Module,
    batch: Dict[str, Any],
) -> torch.Tensor:
    """Run a forward pass with the Transolver-expected input keys."""
    return model(fx=batch["fx"], embedding=batch["embedding"])


_PHYSICS_KEYS = (
    "coordinates_unnormalized",
    "cell_areas",
    "sigma_t",
    "sigma_s",
    "sim_time",
)


def loss_inputs(batch: Dict[str, Any], require_physics: bool = False) -> Dict[str, Any]:
    """Assemble the optional/physics inputs consumed by ``compute_losses``.

    Always copies ``material_labels`` if present. Physics-loss tensors are
    copied only when all of ``_PHYSICS_KEYS`` are in the batch;
    ``require_physics=True`` raises if any is missing. ``metadata`` and
    ``flux_normalization_stats`` are forwarded when present.
    """
    inputs: Dict[str, Any] = {}
    if "material_labels" in batch:
        inputs["material_labels"] = batch["material_labels"]

    missing = [k for k in _PHYSICS_KEYS if k not in batch]
    if missing:
        if require_physics:
            msg = f"Missing physics-loss input(s): {missing}."
            if "coordinates_unnormalized" in missing:
                msg += " (Enable the RTEBackupCoords transform in the data pipeline.)"
            raise KeyError(msg)
        return inputs

    for k in _PHYSICS_KEYS:
        inputs[k] = batch[k]
    for k in ("ulr", "llr", "urr", "lrr", "hlr", "hrr", "cx", "cy"):
        if k in batch:
            inputs[k] = batch[k]
    if "flux_normalization_stats" in batch:
        inputs["flux_normalization_stats"] = batch["flux_normalization_stats"]
    return inputs


def _log_minibatch(
    launch_logger: LaunchLogger,
    loss: torch.Tensor,
    loss_mse: torch.Tensor,
    loss_qoi: Optional[torch.Tensor],
    qoi_details: Dict[str, float],
    scale: float,
) -> None:
    metrics = {"loss": loss.item() * scale, "loss_mse": loss_mse.item()}
    if loss_qoi is not None:
        metrics["loss_qoi"] = loss_qoi.item()
        metrics.update(qoi_details)
    launch_logger.log_minibatch(metrics)


def train_epoch(
    cfg: DictConfig,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    launch_logger: LaunchLogger,
    loss_cfg: Dict[str, Any],
) -> None:
    """Run one Transolver training epoch.

    Reads ``cfg.case.type``, ``cfg.train.amp*``, and
    ``cfg.train.gradient_accumulation_steps`` directly so callers only
    thread the per-epoch ``loss_cfg`` (which is pre-processed by
    :func:`losses.parse_loss_config` and varies per epoch via warmup).
    """
    case_type = cfg.case.type
    use_amp, amp_dtype = parse_amp(cfg)
    accum_steps = cfg.train.get("gradient_accumulation_steps", 1)
    max_grad_norm = float(cfg.train.get("max_grad_norm", 10.0))

    model.train()
    epoch_len = len(dataloader)

    for i, batch in enumerate(dataloader):
        # Gradient accumulation with DDP-aware grad-sync skip: zero at window
        # start, run backward inside ``model.no_sync()`` until the boundary
        # step (or the final batch of the epoch), then step + clip + update.
        if i % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        is_step_boundary = (i + 1) % accum_steps == 0 or (i + 1) == epoch_len

        batch = to_device(batch, device)

        with autocast(enabled=use_amp, device_type=device.type, dtype=amp_dtype):
            prediction = forward(model, batch)

        pred, target = prediction, batch["flux_target"]

        loss, loss_mse, loss_qoi, qoi_details = compute_losses(
            pred=pred.float(),
            target=target.float(),
            loss_inputs=loss_inputs(
                batch, require_physics=loss_cfg.get("use_physics_loss", False)
            ),
            loss_cfg=loss_cfg,
            case_type=case_type,
            device=device,
        )

        _log_minibatch(
            launch_logger,
            loss,
            loss_mse,
            loss_qoi,
            qoi_details,
            scale=1,
        )

        sync_ctx = (
            model.no_sync()
            if (not is_step_boundary and hasattr(model, "no_sync"))
            else nullcontext()
        )
        with sync_ctx:
            scaler.scale(loss / accum_steps).backward()

        if is_step_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()


@torch.no_grad()
def validate(
    cfg: DictConfig,
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    launch_logger: LaunchLogger,
    loss_cfg: Dict[str, Any],
) -> Tuple[float, int, Dict[str, float], Dict[str, int]]:
    """Run validation and return loss plus metric sums/counts for DDP reduce."""
    case_type = cfg.case.type
    use_amp, amp_dtype = parse_amp(cfg)

    model.eval()
    eval_model = model.module if hasattr(model, "module") else model

    loss_sum = 0.0
    num_batches = 0
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    def accumulate_metric(name: str, value: Any) -> None:
        scalar = float(value)
        metric_sums[name] = metric_sums.get(name, 0.0) + scalar
        metric_counts[name] = metric_counts.get(name, 0) + 1

    for batch in dataloader:
        batch = to_device(batch, device)

        with autocast(enabled=use_amp, device_type=device.type, dtype=amp_dtype):
            prediction = forward(eval_model, batch)

        pred, target = prediction, batch["flux_target"]

        loss, loss_mse, loss_qoi, qoi_details = compute_losses(
            pred=pred.float(),
            target=target.float(),
            loss_inputs=loss_inputs(
                batch, require_physics=loss_cfg.get("use_physics_loss", False)
            ),
            loss_cfg=loss_cfg,
            case_type=case_type,
            device=device,
        )

        _log_minibatch(launch_logger, loss, loss_mse, loss_qoi, qoi_details, scale=1)

        loss_sum += loss.item()
        num_batches += 1
        accumulate_metric("loss_mse", loss_mse.item())
        if loss_qoi is not None:
            accumulate_metric("loss_qoi", loss_qoi.item())
        for key, value in qoi_details.items():
            accumulate_metric(key, value)

    return loss_sum, num_batches, metric_sums, metric_counts


def _format_epoch_log(
    epoch: int,
    train_log: Any,
    val_log: Any,
    val_loss: float,
    current_lr: float,
) -> str:
    """Build the per-epoch rank-0 log line.

    Emits ``train_loss`` / ``val_loss`` first, then ``train_X`` / ``val_X``
    pairs for every other metric key present in either log (sorted), then
    ``lr``. Joined with ", ".
    """
    parts = [
        f"train_loss={train_log.epoch_losses.get('loss', 0.0):.4e}",
        f"val_loss={val_loss:.4e}",
    ]
    extra_keys = sorted(
        {k for k in (*train_log.epoch_losses, *val_log.epoch_losses) if k != "loss"}
    )
    for key in extra_keys:
        short = key.removeprefix("loss_")
        if key in train_log.epoch_losses:
            parts.append(f"train_{short}={train_log.epoch_losses[key]:.4e}")
        if key in val_log.epoch_losses:
            parts.append(f"val_{short}={val_log.epoch_losses[key]:.4e}")
    parts.append(f"lr={current_lr:.2e}")
    return f"Epoch {epoch}: " + ", ".join(parts)


def run_training_loop(
    cfg: DictConfig,
    dist: DistributedManager,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    loss_cfg: Dict[str, Any],
    logger: Any,
    checkpoint_dir: str,
    writer: Optional[SummaryWriter],
    best_val_loss: float,
    start_epoch: int,
) -> None:
    """Run the main training loop: epochs, validation, checkpointing, logging.

    Drives the epoch loop, applies the physics-loss warmup ramp inline,
    aggregates validation loss across DDP ranks, steps the scheduler, and
    saves the single best-by-val_loss checkpoint. The per-epoch train and
    validate steps are :func:`train_epoch` and :func:`validate` in this
    module; they read case type, AMP, and gradient-accumulation settings
    from ``cfg`` directly.

    Args:
        cfg: Hydra config (uses ``train.epochs``, ``case.type``, ``train.amp*``,
            and ``train.gradient_accumulation_steps``).
        dist: DistributedManager instance.
        model: Model (possibly DDP-wrapped). The DistributedSampler is
            already attached to ``train_loader``; the loader forwards
            ``set_epoch`` to it.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        scaler: GradScaler for AMP.
        loss_cfg: Loss configuration from
            :func:`losses.parse_loss_config`. The trainer applies the
            physics-loss warmup ramp to the ``physics_loss_weight`` per epoch
            for the training pass; validation always uses the unwarmed dict.
        logger: Logger (rank 0).
        checkpoint_dir: Directory for checkpoints.
        writer: TensorBoard SummaryWriter (rank 0) or None.
        best_val_loss: Best validation loss seen so far (lower is better).
        start_epoch: First epoch index to run.
    """
    case_type = cfg.case.type

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loader.set_epoch(epoch)
        val_loader.set_epoch(epoch)

        current_physics_weight = physics_loss_weight_for_epoch(loss_cfg, epoch)
        epoch_loss_cfg = {
            **loss_cfg,
            "physics_loss_weight": current_physics_weight,
        }
        if current_physics_weight != loss_cfg["physics_loss_weight"]:
            logger.info(
                f"Physics loss warmup: epoch {epoch}, "
                f"weight={current_physics_weight:.6f}"
            )

        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(train_loader),
            mini_batch_log_freq=10,
        ) as train_log:
            train_epoch(
                cfg,
                train_loader,
                model,
                optimizer,
                scaler,
                dist.device,
                train_log,
                loss_cfg=epoch_loss_cfg,
            )

        with LaunchLogger(
            "val", epoch=epoch, num_mini_batch=len(val_loader)
        ) as val_log:
            (
                val_loss_sum,
                val_num_batches,
                val_metric_sums,
                val_metric_counts,
            ) = validate(
                cfg,
                val_loader,
                model,
                dist.device,
                val_log,
                loss_cfg=loss_cfg,
            )

        train_loss = train_log.epoch_losses.get("loss", 0.0)
        val_loss = aggregate_validation_loss(val_loss_sum, val_num_batches, dist)
        val_metrics = aggregate_validation_metrics(
            val_metric_sums, val_metric_counts, dist
        )
        val_log.epoch_losses.update(val_metrics)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_loss_qoi = val_metrics.get("loss_qoi")

        if dist.rank == 0:
            logger.info(
                _format_epoch_log(epoch, train_log, val_log, val_loss, current_lr)
            )

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Learning_Rate", current_lr, epoch)

            if not (
                math.isfinite(train_loss)
                and math.isfinite(val_loss)
                and (val_loss_qoi is None or math.isfinite(val_loss_qoi))
            ):
                logger.warning(
                    "Skipping checkpoint save for epoch %s because at least "
                    "one checkpoint metric is NaN or inf: "
                    "train_loss=%s, val_loss=%s, val_loss_qoi=%s",
                    epoch,
                    train_loss,
                    val_loss,
                    val_loss_qoi,
                )
            else:
                best_val_loss = save_best_checkpoint(
                    checkpoint_dir=Path(checkpoint_dir),
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    save_checkpoint_fn=save_checkpoint,
                    logger=logger,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    metadata={
                        "best_val_loss": val_loss,
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

    if writer:
        writer.close()

    logger.info("=" * 70)
    logger.info("Training completed!")
    if best_val_loss < float("inf"):
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("=" * 70)
