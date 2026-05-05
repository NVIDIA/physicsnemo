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

"""Hydra entry point for the RTE Transolver training sample.

Composes the flat ``src/`` modules (``loader``, ``losses``, ``checkpointing``,
``trainer``) into a single training driver. The Transolver model spec
(``build_model``, ``to_device``, ``forward``, ``loss_inputs``,
``build_dataloaders_for_training``) is inlined at the top of this file —
there is no model-spec dispatcher because only one model is shipped.

Usage::

    python src/train.py case=lattice data=lattice case.data_root=...

Multi-GPU::

    torchrun --nproc_per_node=N src/train.py case=lattice data=lattice ...
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast

from physicsnemo.datapipes import DataLoader
from physicsnemo.utils.logging.launch import LaunchLogger

from checkpointing import create_training_components, resume_or_pretrain
from loader import build_dataloaders, collate_no_padding
from losses import parse_loss_config
from trainer import (
    compute_losses,
    flush_partial_accumulation,
    grad_step,
    log_effective_batch_size,
    run_training_loop,
    set_seed,
    setup_training_environment,
    wrap_ddp,
)


# =========================================================================
# Inlined Transolver helpers (was training/model_specs/transolver.py)
# =========================================================================
#
# A single-model sample doesn't need a spec dispatcher — these helpers are
# called directly from ``train_epoch`` / ``validate`` / ``main`` below.


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
            "Only batch_size=1 is supported for the Transolver point-cloud "
            "adapter (variable-length padding collate was removed)."
        )
    loaders, train_sampler = build_dataloaders(
        cfg,
        dist,
        collate_fn=collate_no_padding,
        phases=("train", "val"),
        logger=logger,
    )
    return loaders["train"], loaders["val"], train_sampler


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


def loss_inputs(
    batch: Dict[str, Any], *, require_physics: bool = False
) -> Dict[str, Any]:
    """Assemble the dict of optional/physics inputs consumed by ``compute_losses``.

    Physics loss requires raw, unnormalized coordinates. The model embedding
    can contain material features, so it is never a safe coordinate fallback.
    """
    inputs: Dict[str, Any] = {}
    if batch.get("padding_mask") is not None:
        inputs["padding_mask"] = batch["padding_mask"]
    if "material_labels" in batch:
        inputs["material_labels"] = batch["material_labels"]
    physics_keys = ("cell_areas", "sigma_t", "sigma_s", "sim_time")
    if require_physics and not all(k in batch for k in physics_keys):
        missing = [k for k in physics_keys if k not in batch]
        raise KeyError(f"Missing physics-loss input(s): {missing}")
    if all(k in batch for k in physics_keys):
        if "coordinates_unnormalized" not in batch:
            if require_physics:
                raise KeyError(
                    "coordinates_unnormalized is required when physics loss is "
                    "enabled. Enable coordinate backup before normalization."
                )
            return inputs
        inputs["coordinates_unnormalized"] = batch["coordinates_unnormalized"]
        for k in physics_keys:
            inputs[k] = batch[k]
        for k in ("metadata", "flux_normalization_stats"):
            if k in batch:
                inputs[k] = batch[k]
    return inputs


# =========================================================================
# Per-epoch train / validate (Transolver-specialized)
# =========================================================================


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
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    launch_logger: LaunchLogger,
    *,
    loss_cfg: Dict[str, Any],
    case_type: str,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
) -> None:
    """Run one Transolver training epoch."""
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        batch = to_device(batch, device)

        with autocast(enabled=use_amp, device_type=device.type, dtype=amp_dtype):
            prediction = forward(model, batch)

        # Transolver predicts absolute flux directly — no reconstruction step.
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

        grad_step(
            loss,
            scaler,
            optimizer,
            model,
            step_idx=i,
            accum_steps=gradient_accumulation_steps,
        )

    flush_partial_accumulation(
        scaler,
        optimizer,
        model,
        total_steps=len(dataloader),
        accum_steps=gradient_accumulation_steps,
    )


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    launch_logger: LaunchLogger,
    *,
    loss_cfg: Dict[str, Any],
    case_type: str,
    use_amp: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[float, int, Dict[str, float], Dict[str, int]]:
    """Run validation and return loss plus metric sums/counts for DDP reduce."""
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


# =========================================================================
# AMP helper
# =========================================================================


def _parse_amp(cfg: DictConfig) -> Tuple[bool, Optional[torch.dtype], str]:
    """Read ``cfg.train.amp`` and ``cfg.train.amp_dtype`` into (use_amp, dtype, label)."""
    use_amp = cfg.train.get("amp", True)
    dtype_str = str(cfg.train.get("amp_dtype", "bf16")).lower()
    if dtype_str in ("bf16", "bfloat16"):
        return use_amp, torch.bfloat16, dtype_str
    if dtype_str in ("fp16", "float16"):
        return use_amp, torch.float16, dtype_str
    raise ValueError(
        f"Unsupported amp_dtype {dtype_str!r}; "
        "allowed values are 'bf16', 'bfloat16', 'fp16', 'float16'."
    )


# =========================================================================
# Hydra main
# =========================================================================


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the Transolver RTE surrogate."""
    # --- environment, seed, AMP ---
    dist, logger = setup_training_environment(cfg, "Transolver")

    seed = cfg.train.get("seed", None)
    if seed is not None:
        effective_seed = seed + dist.rank if dist.distributed else seed
        deterministic = cfg.train.get("deterministic", False)
        set_seed(effective_seed, deterministic=deterministic)
        if dist.rank == 0:
            logger.info(
                f"Random seed: {seed}"
                + (" (deterministic mode)" if deterministic else "")
            )
    elif dist.rank == 0:
        logger.info("Random seed: not set (non-reproducible)")

    grad_accum_steps = cfg.train.get("gradient_accumulation_steps", 1)
    use_amp, amp_dtype, amp_dtype_label = _parse_amp(cfg)

    amp_info = "ENABLED" if use_amp else "DISABLED"
    if use_amp:
        amp_info += f" (dtype={amp_dtype_label})"
    log_effective_batch_size(
        cfg,
        dist,
        logger,
        grad_accum_steps,
        extra_info={"AMP (mixed precision)": amp_info},
    )

    # --- dataloaders ---
    train_loader, val_loader, train_sampler = build_dataloaders_for_training(
        cfg, dist, logger
    )

    # --- model ---
    if dist.rank == 0:
        logger.info("\nInitializing Transolver model...")
    model = build_model(cfg, dist.device)
    if dist.rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Transolver initialized — {num_params:,} trainable parameters")
    model = wrap_ddp(model, dist, logger)

    # --- training components ---
    use_tensorboard = cfg.train.get("tensorboard", True)
    optimizer, scheduler, scaler, writer, checkpoint_dir, best_val_losses = (
        create_training_components(
            cfg, model, dist, logger, tensorboard=use_tensorboard
        )
    )

    # --- loss config (case-specific physics weight comes via Hydra interpolation:
    #     ``training/base.yaml::physics_loss.weight: ${case.physics_loss_weight}``,
    #     replacing the deleted ``_apply_case_weight_override`` function) ---
    loss_cfg = parse_loss_config(cfg, dist, logger)

    loss_metric = cfg.train.get("loss_metric", "mse")
    loss_cfg["loss_metric"] = loss_metric
    if dist.rank == 0:
        logger.info(f"Loss metric: {loss_metric}")

    # --- physics-loss warmup state ---
    use_physics_loss = loss_cfg["use_physics_loss"]
    physics_loss_weight_base = loss_cfg["physics_loss_weight"]
    physics_loss_warmup_epochs = 0
    physics_loss_warmup_start = 0.0
    if use_physics_loss:
        physics_loss_warmup_epochs = cfg.train.physics_loss.get("warmup_epochs", 0)
        physics_loss_warmup_start = cfg.train.physics_loss.get(
            "warmup_start_fraction", 0.0
        )
        if dist.rank == 0 and physics_loss_warmup_epochs > 0:
            logger.info(f"  Physics-loss warmup epochs: {physics_loss_warmup_epochs}")
            logger.info(
                f"  Physics-loss warmup start fraction: {physics_loss_warmup_start}"
            )

    # --- resume / pretrain ---
    start_epoch, resumed_val_losses, best_qoi_loss = resume_or_pretrain(
        cfg, model, optimizer, scheduler, scaler, dist, logger
    )
    if resumed_val_losses:
        best_val_losses = resumed_val_losses

    # --- per-epoch hooks ---
    shared_kwargs = {
        "loss_cfg": loss_cfg,
        "case_type": cfg.case.type,
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
    }
    train_epoch_kwargs = {
        **shared_kwargs,
        "gradient_accumulation_steps": grad_accum_steps,
    }
    validate_kwargs = dict(shared_kwargs)

    def before_epoch_fn(epoch: int):
        """Per-epoch physics-loss-weight ramp-up.

        Linearly ramps ``physics_loss_weight`` from
        ``warmup_start_fraction * base`` to ``base`` over the first
        ``warmup_epochs``. After warmup, the weight stays at ``base``.

        Validation always uses the unwarmed-up final ``loss_cfg`` so val_loss
        is comparable across epochs and best-checkpoint selection is meaningful.
        """
        if not use_physics_loss or physics_loss_warmup_epochs <= 0:
            return {}, {}
        if epoch >= physics_loss_warmup_epochs:
            current_weight = physics_loss_weight_base
        else:
            progress = epoch / max(1, physics_loss_warmup_epochs)
            current_weight = (
                physics_loss_warmup_start + (1.0 - physics_loss_warmup_start) * progress
            ) * physics_loss_weight_base
        if dist.rank == 0 and epoch < physics_loss_warmup_epochs:
            logger.info(
                f"Physics loss warmup: epoch {epoch}, "
                f"weight={current_weight:.6f} (target={physics_loss_weight_base})"
            )
        epoch_loss_cfg = {**loss_cfg, "physics_loss_weight": current_weight}
        return {"loss_cfg": epoch_loss_cfg}, {"loss_cfg": loss_cfg}

    def after_epoch_fn(epoch, train_log, val_log, val_loss, current_lr):
        train_loss = train_log.epoch_losses.get("loss", 0.0)
        log_msg = f"Epoch {epoch}: train_loss={train_loss:.4e}, val_loss={val_loss:.4e}"
        log_keys = ["loss_mse", "loss_qoi"]
        for key in sorted(train_log.epoch_losses.keys()):
            if key.startswith("loss_qoi_") and key not in log_keys:
                log_keys.append(key)
        for key in log_keys:
            t = train_log.epoch_losses.get(key)
            if t is not None:
                log_msg += f", train_{key.replace('loss_', '')}={t:.4e}"
            v = val_log.epoch_losses.get(key)
            if v is not None:
                log_msg += f", val_{key.replace('loss_', '')}={v:.4e}"
        log_msg += f", lr={current_lr:.2e}"
        logger.info(log_msg)

    # --- dispatch to shared training loop ---
    if dist.rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("Starting training...")
        logger.info("=" * 70)

    run_training_loop(
        cfg=cfg,
        dist=dist,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_epoch_fn=train_epoch,
        validate_fn=validate,
        train_epoch_kwargs=train_epoch_kwargs,
        validate_kwargs=validate_kwargs,
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        writer=writer,
        best_val_losses=best_val_losses,
        start_epoch=start_epoch,
        case_type=cfg.case.type,
        before_epoch_fn=before_epoch_fn,
        after_epoch_fn=after_epoch_fn,
        best_qoi_loss=best_qoi_loss,
    )


if __name__ == "__main__":
    main()
