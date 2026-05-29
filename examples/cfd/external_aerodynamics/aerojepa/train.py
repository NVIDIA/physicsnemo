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

r"""Hydra-driven training entry point for the AeroJEPA SuperWing recipe.

Composes the recipe's data, model, and training configs and runs the
JEPA training loop.
Usage::

    python train.py data.path=/path/to/SuperWing_Dataset

See ``conf/config.yaml`` for the full configuration surface.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.experimental.models.aerojepa import TokenSet
from src.datapipes import (
    SuperWingDataset,
    build_superwing_split_manifest,
    compute_superwing_normalization_stats,
    superwing_collate,
)
from src.losses import (
    build_recon_loss_from_config,
    build_sigreg_from_config,
    compute_latent_loss,
)
from src.training import (
    ExponentialMovingAverage,
    build_lr_scheduler,
    build_optimizer,
    get_autocast_context,
    linear_warmup_weight,
    move_batch_to_device,
    set_seed,
)


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def _ensure_superwing_artifacts(data_cfg: DictConfig) -> tuple[str, str]:
    r"""Ensure the split manifest and normalization-stats JSONs exist.

    Builds them under ``data.path`` on the first run; returns the resolved
    paths.
    """
    root = Path(str(data_cfg.path))
    split_path = (
        Path(str(data_cfg.split_manifest))
        if data_cfg.split_manifest
        else root / "split_by_geometry.json"
    )
    stats_path = (
        Path(str(data_cfg.normalization_stats_path))
        if data_cfg.normalization_stats_path
        else root / "normalization_stats_train.json"
    )

    if not split_path.exists():
        log.info("Building split manifest at %s", split_path)
        manifest = build_superwing_split_manifest(
            root_dir=str(root),
            train_ratio=float(data_cfg.train_ratio),
            val_ratio=float(data_cfg.val_ratio),
            seed=int(data_cfg.split_seed),
        )
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if not stats_path.exists():
        log.info("Building normalization stats at %s", stats_path)
        compute_superwing_normalization_stats(
            root_dir=str(root),
            split_manifest_path=str(split_path),
            gen_param_columns=list(data_cfg.gen_params_columns),
            gen_param_names=list(data_cfg.gen_params_names),
            max_target_samples=int(data_cfg.normalization_max_target_samples),
            save_path=str(stats_path),
        )

    return str(split_path), str(stats_path)


def _build_loader(
    data_cfg: DictConfig,
    *,
    split: str,
    split_manifest_path: str,
    normalization_stats_path: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    deterministic = (
        bool(data_cfg.train_deterministic_sampling)
        if split == "train"
        else bool(data_cfg.eval_deterministic_sampling)
    )
    return_origingeom = (
        bool(data_cfg.val_return_origingeom) if split != "train" else False
    )
    dataset = SuperWingDataset(
        root_dir=str(data_cfg.path),
        split=split,
        split_manifest_path=split_manifest_path,
        normalization_stats_path=normalization_stats_path,
        surface_points=int(data_cfg.surface_points),
        target_encoder_points=int(data_cfg.target_encoder_points),
        query_points=int(data_cfg.query_points),
        eval_full_grid_query=bool(data_cfg.eval_full_grid_query),
        return_origingeom=return_origingeom,
        return_full_fields=False,
        deterministic_sampling=deterministic,
        normalize_xyz=bool(data_cfg.normalize_xyz),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(data_cfg.num_workers),
        pin_memory=bool(data_cfg.pin_memory),
        collate_fn=superwing_collate,
        drop_last=False,
        persistent_workers=int(data_cfg.num_workers) > 0,
    )


# --------------------------------------------------------------------------- #
# Per-sample training step
# --------------------------------------------------------------------------- #


def _slice_batch_sample(batch: dict, idx: int) -> dict[str, Any]:
    """Extract a single-sample dict from a padded batch."""
    sample: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and not k.endswith("_n"):
            n_key = f"{k}_n"
            if n_key in batch:
                length = int(batch[n_key][idx].item())
                sample[k] = v[idx, :length]
            else:
                sample[k] = v[idx]
        elif isinstance(v, list):
            sample[k] = v[idx]
        else:
            sample[k] = v
    return sample


def _forward_sample(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, TokenSet, TokenSet, torch.Tensor]:
    """Run encoders + predictor + decoder on one sample.

    Returns
    -------
    pred_field : torch.Tensor
        Decoded field at the query points, shape ``(N_q, C)``.
    pred_features : torch.Tensor
        Predictor output features matching ``target_tokens.coords``.
    target_tokens : TokenSet
        Target encoder's output (kept for the latent / SIGReg losses).
    predictor_tokens : TokenSet
        TokenSet wrapping the predictor's output features.
    cond_global : torch.Tensor
        Decoder-side conditioning vector.
    """
    ctx = model.encode_geometry_and_flow(
        context_pos=sample["context_pos"],
        context_feat=sample["context_feat"],
        target_surface_pos=sample["target_surface_pos"],
        target_surface_main_feat=sample["target_surface_main_feat"],
        target_volume_pos=sample["target_volume_pos"],
        target_volume_feat=sample["target_volume_feat"],
        gen_params=sample["gen_params"],
    )
    target_tokens: TokenSet = ctx["target_tokens"]
    context_tokens: TokenSet = ctx["context_tokens"]
    cond_global: torch.Tensor = ctx["cond_global"]

    conditions = sample["gen_params"].unsqueeze(0)
    pred_features = model.predict_field_tokens(
        context_tokens=context_tokens,
        target_positions=target_tokens.coords,
        conditions=conditions,
    )
    if pred_features.ndim == 3 and int(pred_features.shape[0]) == 1:
        pred_features = pred_features[0]

    predictor_tokens = TokenSet(
        features=pred_features,
        coords=target_tokens.coords,
        mask=target_tokens.mask,
        global_token=target_tokens.global_token,
    )

    pred_field = model.decode_field(
        target_tokens=predictor_tokens,
        cond_global=cond_global,
        query_pos=sample["query_pos"],
        query_sdf=sample["query_sdf"],
    )
    return pred_field, pred_features, target_tokens, predictor_tokens, cond_global


def _compute_total_loss(
    *,
    pred_field: torch.Tensor,
    query_target: torch.Tensor,
    pred_features: torch.Tensor,
    target_tokens: TokenSet,
    recon_loss_fn: torch.nn.Module,
    sigreg_loss_fn: torch.nn.Module,
    loss_cfg: DictConfig,
    epoch: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine recon + latent + sigreg losses with linear warmup."""
    recon_term = recon_loss_fn(pred_field, query_target)
    latent_term = compute_latent_loss(
        pred_features.unsqueeze(0),
        target_tokens.features.unsqueeze(0),
        mse_weight=float(loss_cfg.latent.mse_weight),
        cosine_weight=float(loss_cfg.latent.cosine_weight),
        mask=(
            target_tokens.mask.unsqueeze(0) if target_tokens.mask is not None else None
        ),
    )
    sigreg_term = sigreg_loss_fn(
        target_tokens.features,
        target_tokens.mask,
    )

    recon_w = linear_warmup_weight(
        float(loss_cfg.recon.weight),
        float(loss_cfg.recon.warmup_epochs),
        epoch,
    )
    latent_w = linear_warmup_weight(
        float(loss_cfg.latent.weight),
        float(loss_cfg.latent.warmup_epochs),
        epoch,
    )
    sigreg_w = linear_warmup_weight(
        float(loss_cfg.sigreg.weight),
        float(loss_cfg.sigreg.warmup_epochs),
        epoch,
    )

    total = recon_w * recon_term + latent_w * latent_term + sigreg_w * sigreg_term
    return total, {
        "recon": float(recon_term.detach().item()),
        "latent": float(latent_term.detach().item()),
        "sigreg": float(sigreg_term.detach().item()),
        "recon_w": recon_w,
        "latent_w": latent_w,
        "sigreg_w": sigreg_w,
    }


# --------------------------------------------------------------------------- #
# Epoch loop
# --------------------------------------------------------------------------- #


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    recon_loss_fn: torch.nn.Module,
    sigreg_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any,
    ema: ExponentialMovingAverage | None,
    device: torch.device,
    precision: str,
    grad_clip_norm: float,
    loss_cfg: DictConfig,
    epoch: int,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals: dict[str, float] = {
        "loss": 0.0,
        "recon": 0.0,
        "latent": 0.0,
        "sigreg": 0.0,
    }
    n_samples = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        sample_losses: list[torch.Tensor] = []
        for sample_idx in range(int(batch["context_pos"].shape[0])):
            sample = _slice_batch_sample(batch, sample_idx)
            with get_autocast_context(device, precision):
                pred_field, pred_features, target_tokens, _, _ = _forward_sample(
                    model, sample
                )
                loss, parts = _compute_total_loss(
                    pred_field=pred_field,
                    query_target=sample["query_target"],
                    pred_features=pred_features,
                    target_tokens=target_tokens,
                    recon_loss_fn=recon_loss_fn,
                    sigreg_loss_fn=sigreg_loss_fn,
                    loss_cfg=loss_cfg,
                    epoch=float(epoch),
                )
            sample_losses.append(loss)
            for k in ("recon", "latent", "sigreg"):
                totals[k] += float(parts[k])
            totals["loss"] += float(loss.detach().item())
            n_samples += 1

        if is_train:
            batch_loss = torch.stack(sample_losses).mean()
            batch_loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(grad_clip_norm)
                )
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.update(model)

    if n_samples == 0:
        return {k: float("nan") for k in totals}
    return {k: v / float(n_samples) for k, v in totals.items()}


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema: ExponentialMovingAverage | None,
    epoch: int,
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        payload["ema_shadow"] = ema.shadow
        payload["ema_decay"] = ema.decay
    torch.save(payload, path)
    log.info("Saved checkpoint to %s", path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — train an AeroJEPA model on SuperWing."""
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = output_dir / cfg.output_dir / "checkpoints"
    tb_dir = output_dir / cfg.output_dir / "tensorboard"
    log.info("Output dir: %s", output_dir)

    # Data prep + loaders.
    split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    train_loader = _build_loader(
        cfg.data,
        split="train",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
    )
    val_loader = _build_loader(
        cfg.data,
        split="val",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        batch_size=int(cfg.training.eval_batch_size),
        shuffle=False,
    )
    log.info(
        "Train / val samples: %d / %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # Model.
    model = hydra.utils.instantiate(cfg.model).to(device)
    log.info(
        "Model parameters: %.2f M",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    # Losses, optimiser, scheduler, EMA.
    recon_loss_fn = build_recon_loss_from_config(cfg.training.loss.recon).to(device)
    sigreg_loss_fn = build_sigreg_from_config(cfg.training.loss.sigreg).to(device)
    optimizer = build_optimizer(model, cfg.training.optimizer)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        name=str(cfg.training.scheduler.name),
        epochs=int(cfg.training.epochs),
        steps_per_epoch=max(1, len(train_loader)),
        warmup_epochs=float(cfg.training.scheduler.warmup_epochs),
    )
    ema: ExponentialMovingAverage | None = None
    if bool(cfg.training.ema.enabled):
        ema = ExponentialMovingAverage(model, decay=float(cfg.training.ema.decay))

    writer = SummaryWriter(log_dir=str(tb_dir))

    grad_clip_norm = float(cfg.training.grad_clip_norm)
    save_every = int(cfg.training.save_every_epochs)
    max_eval_batches = int(cfg.training.max_eval_batches)

    best_val_loss = float("inf")
    for epoch in range(int(cfg.training.epochs)):
        t0 = time.time()
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            recon_loss_fn=recon_loss_fn,
            sigreg_loss_fn=sigreg_loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            device=device,
            precision=str(cfg.training.precision),
            grad_clip_norm=grad_clip_norm,
            loss_cfg=cfg.training.loss,
            epoch=epoch,
            max_batches=None,
        )
        train_time = time.time() - t0

        if ema is not None:
            ema.apply_to(model)
        try:
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                recon_loss_fn=recon_loss_fn,
                sigreg_loss_fn=sigreg_loss_fn,
                optimizer=None,
                lr_scheduler=None,
                ema=None,
                device=device,
                precision=str(cfg.training.precision),
                grad_clip_norm=0.0,
                loss_cfg=cfg.training.loss,
                epoch=epoch,
                max_batches=max_eval_batches,
            )
        finally:
            if ema is not None:
                ema.restore(model)

        log.info(
            "epoch=%03d  train_loss=%.4f  val_loss=%.4f  "
            "train_recon=%.4f val_recon=%.4f  time=%.1fs",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["recon"],
            val_metrics["recon"],
            train_time,
        )

        for split_name, m in (("train", train_metrics), ("val", val_metrics)):
            for k, v in m.items():
                writer.add_scalar(f"{split_name}/{k}", v, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % save_every == 0 or epoch + 1 == int(cfg.training.epochs):
            _save_checkpoint(
                path=ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                epoch=epoch + 1,
                cfg=cfg,
            )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            _save_checkpoint(
                path=ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                epoch=epoch + 1,
                cfg=cfg,
            )

    writer.close()
    log.info("Training done. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
