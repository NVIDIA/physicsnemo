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
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.distributed import DistributedManager
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
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, Any]:
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
    sampler = None
    if world_size > 1:
        # Shard the dataset across ranks. ``drop_last`` on the train sampler
        # keeps the per-rank batch count equal so the per-step gradient
        # all-reduce stays collectively balanced.
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(shuffle),
            drop_last=(split == "train"),
        )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(data_cfg.num_workers),
        pin_memory=bool(data_cfg.pin_memory),
        collate_fn=superwing_collate,
        drop_last=False,
        persistent_workers=int(data_cfg.num_workers) > 0,
        prefetch_factor=(
            int(data_cfg.get("prefetch_factor", 4))
            if int(data_cfg.num_workers) > 0
            else None
        ),
    )
    return loader, sampler


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
    *,
    run_reconstruction: bool = True,
) -> tuple[
    torch.Tensor | None, torch.Tensor, TokenSet, TokenSet, TokenSet, torch.Tensor
]:
    """Run the encoders + predictor (and optionally the decoder) on one sample.

    Parameters
    ----------
    run_reconstruction : bool, optional
        When ``False`` the decoder is skipped and ``pred_field`` is ``None``.
        Used when the reconstruction term has zero weight (e.g. the latent
        phase of two-phase training) to avoid the decoder forward.

    Returns
    -------
    pred_field : torch.Tensor or None
        Decoded field at the query points, shape ``(N_q, C)``; ``None`` when
        ``run_reconstruction`` is ``False``.
    pred_features : torch.Tensor
        Predictor output features matching ``target_tokens.coords``.
    context_tokens : TokenSet
        Context encoder's output (kept for the optional context SIGReg).
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

    pred_field = None
    if run_reconstruction:
        pred_field = model.decode_field(
            target_tokens=predictor_tokens,
            cond_global=cond_global,
            query_pos=sample["query_pos"],
            query_sdf=sample["query_sdf"],
        )
    return (
        pred_field,
        pred_features,
        context_tokens,
        target_tokens,
        predictor_tokens,
        cond_global,
    )


def _compute_total_loss(
    *,
    pred_field: torch.Tensor | None,
    query_target: torch.Tensor,
    pred_features: torch.Tensor,
    context_tokens: TokenSet,
    target_tokens: TokenSet,
    recon_loss_fn: torch.nn.Module,
    sigreg_loss_fn: torch.nn.Module,
    sigreg_context_loss_fn: torch.nn.Module,
    loss_cfg: DictConfig,
    term_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine recon + latent + target/context SIGReg with the given weights.

    ``term_weights`` holds the (already phase-resolved and warmed-up) scalar
    weights for ``latent``, ``sigreg``, ``sigreg_context`` and ``recon``. The
    latent and SIGReg terms are always computed (cheap, and used for logging);
    the reconstruction term is added only when ``pred_field`` is provided.
    """
    latent_term = compute_latent_loss(
        pred_features.unsqueeze(0),
        target_tokens.features.unsqueeze(0),
        mse_weight=float(loss_cfg.latent.mse_weight),
        cosine_weight=float(loss_cfg.latent.cosine_weight),
        mask=(
            target_tokens.mask.unsqueeze(0) if target_tokens.mask is not None else None
        ),
    )
    sigreg_term = sigreg_loss_fn(target_tokens.features, target_tokens.mask)
    sigreg_context_term = sigreg_context_loss_fn(
        context_tokens.features, context_tokens.mask
    )

    total = (
        term_weights["latent"] * latent_term
        + term_weights["sigreg"] * sigreg_term
        + term_weights["sigreg_context"] * sigreg_context_term
    )
    recon_value = torch.zeros((), device=latent_term.device)
    if pred_field is not None:
        recon_term = recon_loss_fn(pred_field, query_target)
        total = total + term_weights["recon"] * recon_term
        recon_value = recon_term.detach()

    # Parts are returned as detached device tensors (not floats): the training
    # loop accumulates them on-device and syncs to the host only at logging
    # cadence, avoiding a per-sample .item() stall in the hot loop.
    return total, {
        "recon": recon_value,
        "latent": latent_term.detach(),
        "sigreg": sigreg_term.detach(),
        "sigreg_context": sigreg_context_term.detach(),
    }


# --------------------------------------------------------------------------- #
# Two-phase training (optional)
#
# Phase 1 trains the context + target encoders and the predictor in latent
# space (decoder frozen, reconstruction off). Phase 2 freezes those and trains
# only the decoder to reconstruct the field from the frozen latents. Freezing
# is done purely via ``requires_grad`` on a single optimizer, so parameters
# with no gradient are simply skipped by the optimizer step.
# --------------------------------------------------------------------------- #


def _resolve_phase(
    epoch: int, two_phase_cfg: DictConfig | None
) -> dict[str, Any] | None:
    """Resolve the phase for ``epoch`` (1-indexed), or ``None`` if disabled."""
    if two_phase_cfg is None or not bool(two_phase_cfg.get("enabled", False)):
        return None
    phase1_epochs = int(two_phase_cfg.get("phase1_epochs", 0))
    if phase1_epochs <= 0:
        raise ValueError(
            "training.two_phase_training.phase1_epochs must be > 0 when enabled."
        )
    if int(epoch) <= phase1_epochs:
        return {
            "name": "phase1_latent",
            "epoch_in_phase": int(epoch),
            "config": two_phase_cfg,
        }
    return {
        "name": "phase2_reconstruction",
        "epoch_in_phase": int(epoch) - phase1_epochs,
        "config": two_phase_cfg,
    }


def _set_requires_grad(module: torch.nn.Module | None, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(bool(enabled))


def _apply_phase(
    model: torch.nn.Module, phase: dict[str, Any] | None, *, is_train: bool
) -> None:
    """Freeze / unfreeze submodules and set train/eval modes for ``phase``.

    With ``phase is None`` the whole model just follows ``is_train``. Otherwise
    every parameter is frozen first, then the phase's trainable submodules are
    re-enabled; frozen submodules also drop to eval mode so their dropout /
    norm statistics stay fixed.
    """
    if phase is None:
        return
    pcfg = phase["config"]
    _set_requires_grad(model, False)
    model.eval()
    if phase["name"] == "phase1_latent":
        train_ctx = bool(pcfg.get("phase1_train_context_encoder", True))
        train_tgt = bool(pcfg.get("phase1_train_target_encoder", True))
        train_pred = bool(pcfg.get("phase1_train_predictor", True))
        _set_requires_grad(model.context_encoder, train_ctx)
        _set_requires_grad(model.target_encoder, train_tgt)
        _set_requires_grad(model.predictor, train_pred)
        model.context_encoder.train(is_train and train_ctx)
        model.target_encoder.train(is_train and train_tgt)
        model.predictor.train(is_train and train_pred)
    elif phase["name"] == "phase2_reconstruction":
        train_dec = bool(pcfg.get("phase2_train_decoder", True))
        _set_requires_grad(model.decoder, train_dec)
        model.decoder.train(is_train and train_dec)


def _compute_term_weights(
    epoch: int, loss_cfg: DictConfig, phase: dict[str, Any] | None
) -> dict[str, float]:
    """Phase-aware, linearly-warmed-up weights for the four loss terms."""

    def warm(weight, warmup_epochs, ep) -> float:
        return linear_warmup_weight(float(weight), float(warmup_epochs), float(ep))

    if phase is None:
        return {
            "latent": warm(
                loss_cfg.latent.weight, loss_cfg.latent.warmup_epochs, epoch
            ),
            "sigreg": warm(
                loss_cfg.sigreg.weight, loss_cfg.sigreg.warmup_epochs, epoch
            ),
            "sigreg_context": warm(
                loss_cfg.sigreg_context.weight,
                loss_cfg.sigreg_context.get("warmup_epochs", 0),
                epoch,
            ),
            "recon": warm(loss_cfg.recon.weight, loss_cfg.recon.warmup_epochs, epoch),
        }

    pcfg = phase["config"]
    ep_in_phase = int(phase["epoch_in_phase"])
    ctx_warmup = loss_cfg.sigreg_context.get("warmup_epochs", 0)
    if phase["name"] == "phase1_latent":
        return {
            "latent": warm(
                pcfg.get("phase1_latent_weight", loss_cfg.latent.weight),
                pcfg.get("phase1_latent_warmup_epochs", loss_cfg.latent.warmup_epochs),
                epoch,
            ),
            "sigreg": warm(
                pcfg.get("phase1_sigreg_weight", loss_cfg.sigreg.weight),
                pcfg.get("phase1_sigreg_warmup_epochs", loss_cfg.sigreg.warmup_epochs),
                epoch,
            ),
            "sigreg_context": warm(
                pcfg.get(
                    "phase1_sigreg_context_weight", loss_cfg.sigreg_context.weight
                ),
                pcfg.get("phase1_sigreg_context_warmup_epochs", ctx_warmup),
                epoch,
            ),
            "recon": warm(
                pcfg.get("phase1_recon_weight", 0.0),
                pcfg.get("phase1_recon_warmup_epochs", 0),
                ep_in_phase,
            ),
        }
    # phase2_reconstruction: reconstruction only by default.
    return {
        "latent": float(pcfg.get("phase2_latent_weight", 0.0)),
        "sigreg": float(pcfg.get("phase2_sigreg_weight", 0.0)),
        "sigreg_context": float(pcfg.get("phase2_sigreg_context_weight", 0.0)),
        "recon": warm(
            pcfg.get("phase2_recon_weight", loss_cfg.recon.weight),
            pcfg.get("phase2_recon_warmup_epochs", 0),
            ep_in_phase,
        ),
    }


# --------------------------------------------------------------------------- #
# Epoch loop
# --------------------------------------------------------------------------- #


def _all_reduce_grads(model: torch.nn.Module, world_size: int) -> None:
    """Average gradients across ranks (manual data parallel).

    AeroJEPA's training step runs through several model methods rather than a
    single ``forward``, so ``DistributedDataParallel``'s forward-hook gradient
    sync cannot be used; gradients are reduced explicitly instead. Call after
    the per-sample accumulation, on unscaled gradients, before
    ``optimizer.step()``. Every rank starts from identical weights and applies
    identical averaged gradients, so the models stay in lockstep.
    """
    # SAFETY: reduce only parameters that actually received a gradient, and do
    # NOT materialize zero grads for the ones that didn't. Every rank runs the
    # identical forward/backward graph on the same trainable parameter set, so
    # the reached (grad-is-not-None) set is identical across ranks and the
    # coalesced buffers line up element-for-element. Materializing zeros for
    # unreached params (the previous approach) is not only unnecessary for the
    # collective but actively harmful: AdamW's decoupled weight decay applies
    # ``p *= 1 - lr*wd`` to every parameter whose ``.grad`` is not None, so a
    # forced zero grad silently decays otherwise-untouched parameters on
    # multi-GPU runs only — a single-vs-multi-GPU divergence. Skipping them
    # keeps the optimizer's per-parameter behavior identical to single-GPU.
    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    if not grads:
        return
    # Coalesce every gradient into one contiguous buffer and all-reduce ONCE,
    # rather than a separate (latency-bound) NCCL call per parameter — the
    # per-parameter version dominated the profile (~600 tiny all-reduces/step).
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    recon_loss_fn: torch.nn.Module,
    sigreg_loss_fn: torch.nn.Module,
    sigreg_context_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any,
    ema: ExponentialMovingAverage | None,
    device: torch.device,
    precision: str,
    grad_clip_norm: float,
    loss_cfg: DictConfig,
    epoch: int,
    max_batches: int | None,
    phase: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    writer: SummaryWriter | None = None,
    log_every: int = 50,
    world_size: int = 1,
    is_main: bool = True,
    profile: bool = False,
    profile_steps: int = 12,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    _apply_phase(model, phase, is_train=is_train)

    term_weights = _compute_term_weights(epoch, loss_cfg, phase)
    run_reconstruction = float(term_weights["recon"]) != 0.0

    # Accumulate metrics on-device (float64) and sync to the host only at
    # logging cadence / epoch end — avoids a per-sample .item() pipeline stall.
    totals = {
        k: torch.zeros((), device=device, dtype=torch.float64)
        for k in ("loss", "recon", "latent", "sigreg", "sigreg_context")
    }
    n_samples = 0
    epoch_len = len(loader)
    phase_tag = "train" if is_train else "val"
    step_time = time.time()

    prof = None
    if profile:
        _acts = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            _acts.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(activities=_acts)
        prof.start()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        if profile and batch_idx >= int(profile_steps):
            break
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        n_in_batch = int(batch["context_pos"].shape[0])
        batch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        batch_recon_sum = torch.zeros((), device=device, dtype=torch.float64)
        for sample_idx in range(n_in_batch):
            sample = _slice_batch_sample(batch, sample_idx)
            with (
                torch.set_grad_enabled(is_train),
                get_autocast_context(device, precision),
            ):
                pred_field, pred_features, context_tokens, target_tokens, _, _ = (
                    _forward_sample(
                        model, sample, run_reconstruction=run_reconstruction
                    )
                )
                loss, parts = _compute_total_loss(
                    pred_field=pred_field,
                    query_target=sample["query_target"],
                    pred_features=pred_features,
                    context_tokens=context_tokens,
                    target_tokens=target_tokens,
                    recon_loss_fn=recon_loss_fn,
                    sigreg_loss_fn=sigreg_loss_fn,
                    sigreg_context_loss_fn=sigreg_context_loss_fn,
                    loss_cfg=loss_cfg,
                    term_weights=term_weights,
                )
            if is_train:
                # Accumulate gradients one sample at a time so only a single
                # sample's autograd graph is alive at once; dividing by the
                # batch size makes this identical to one backward on the
                # batch-mean loss, at far lower peak memory. ``scaler`` is a
                # no-op unless fp16 is active (bf16/fp32 need no loss scaling).
                scaler.scale(loss / n_in_batch).backward()
            for k in ("recon", "latent", "sigreg", "sigreg_context"):
                totals[k] += parts[k].double()
            sample_loss = loss.detach().double()
            totals["loss"] += sample_loss
            batch_loss_sum += sample_loss
            batch_recon_sum += parts["recon"].double()
            n_samples += 1

        if is_train:
            # Unscale to true gradients before touching them (clipping and/or
            # cross-rank averaging). ``scaler.unscale_`` is a no-op when the
            # scaler is disabled (bf16 / fp32 / CPU).
            if world_size > 1 or grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
            if world_size > 1:
                _all_reduce_grads(model, world_size)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(grad_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.update(model)

        # Per-step progress (GeoTransolver-style), throttled by ``log_every``.
        now = time.time()
        step_dur = now - step_time
        step_time = now
        if is_main and batch_idx % max(1, int(log_every)) == 0:
            batch_loss = (batch_loss_sum / max(1, n_in_batch)).item()
            batch_recon = (batch_recon_sum / max(1, n_in_batch)).item()
            mem_gb = (
                torch.cuda.memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            log.info(
                "Epoch %03d %s [%d/%d] Loss: %.6f recon: %.4f Duration: %.2fs Mem: %.2fGB",
                epoch,
                phase_tag,
                batch_idx,
                epoch_len,
                batch_loss,
                batch_recon,
                step_dur,
                mem_gb,
            )
            if writer is not None:
                gstep = batch_idx + epoch_len * epoch
                writer.add_scalar(f"batch/{phase_tag}_loss", batch_loss, gstep)
                writer.add_scalar(f"batch/{phase_tag}_recon", batch_recon, gstep)
                writer.add_scalar(
                    f"batch/{phase_tag}_throughput",
                    (n_in_batch / step_dur) if step_dur > 0 else 0.0,
                    gstep,
                )
                if is_train and lr_scheduler is not None:
                    writer.add_scalar(
                        "batch/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        gstep,
                    )

    if prof is not None:
        prof.stop()
        if is_main:
            ka = prof.key_averages()
            try:
                log.info(
                    "Profiler (%d steps) — top ops by CUDA self-time:\n%s",
                    int(profile_steps),
                    ka.table(sort_by="self_cuda_time_total", row_limit=20),
                )
            except Exception:
                pass
            log.info(
                "Profiler — top ops by CPU self-time:\n%s",
                ka.table(sort_by="self_cpu_time_total", row_limit=20),
            )

    # Stack the on-device term sums + the sample count, reduce across ranks
    # when distributed (so epoch means are global, not per-shard), then sync to
    # the host once via a single ``.tolist()`` rather than per-term ``.item()``.
    keys = list(totals.keys())
    count = torch.tensor(float(n_samples), device=device, dtype=torch.float64)
    packed = torch.stack([totals[k] for k in keys] + [count])
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = packed.tolist()
    n_total = reduced[-1]
    if n_total == 0:
        return {k: float("nan") for k in keys}
    return {k: reduced[i] / n_total for i, k in enumerate(keys)}


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
    best_val: float,
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # When EMA is active, validation (and thus model selection) runs on the
    # shadow weights, so persist those as the model state so the checkpoint
    # reproduces the reported metric. ``ema.shadow`` shadows the full
    # ``state_dict`` (parameters and buffers), so it is a complete model state.
    model_state = ema.shadow if ema is not None else model.state_dict()
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val),
        "model": model_state,
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


def _epoch_of(ckpt: Path) -> int:
    """Parse the integer epoch from an ``epoch_<N>.pt`` filename (``-1`` if
    unparseable)."""
    try:
        return int(ckpt.stem.split("_")[-1])
    except ValueError:
        return -1


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return the highest-epoch ``epoch_*.pt`` in ``ckpt_dir``, or ``None``.

    Selected by the integer epoch parsed from the filename (not a
    lexicographic sort), so it stays correct past ``epoch_9999.pt`` and for
    mixed zero-pad widths. Files whose stem does not parse to an epoch (e.g.
    a stray ``epoch_abc.pt``) are ignored, so an all-unparseable directory
    resolves to ``None`` (fresh start) rather than a silently-wrong file.
    Used to resume automatically from a stable checkpoint directory.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    ckpts = [p for p in ckpt_dir.glob("epoch_*.pt") if _epoch_of(p) >= 0]
    return max(ckpts, key=_epoch_of) if ckpts else None


def _load_initial_state(
    cfg: DictConfig,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema: ExponentialMovingAverage | None,
    device: torch.device,
    ckpt_dir: Path,
) -> tuple[int, float]:
    """Restore training state before the loop, resuming automatically.

    Resolution order:

    1. **Explicit resume** (``training.resume.enabled`` + ``checkpoint_path``):
       restore from that specific checkpoint (e.g. one produced elsewhere).
    2. **Automatic resume:** if the run's stable ``ckpt_dir`` already holds an
       ``epoch_*.pt``, restore the latest and continue. This is what lets a
       resubmitted / singleton-chained job pick up where it stopped with no
       config change -- the fix for Hydra's timestamped per-run dirs.
    3. **Init from pretrained** (``training.init_from_checkpoint.path``): load
       model weights only (fresh optimizer, epoch 0). With ``strict=false`` a
       checkpoint holding a subset of modules loads those and leaves the rest
       at initialization (e.g. a decoder on frozen pretrained encoders).
    4. Otherwise start fresh from epoch 0.

    Returns the ``(start_epoch, best_val_loss)`` the training loop should use.
    """
    train_cfg = cfg.training
    resume_cfg = train_cfg.get("resume", None)

    # (1) explicit external path wins; else (2) latest in the stable ckpt_dir.
    explicit = None
    if resume_cfg is not None and bool(resume_cfg.get("enabled", False)):
        explicit = resume_cfg.get("checkpoint_path")
        if not explicit:
            raise ValueError(
                "training.resume.enabled=true requires checkpoint_path; omit "
                "resume.enabled to auto-resume from the run's checkpoint dir."
            )
    resume_path = explicit or _latest_checkpoint(ckpt_dir)

    if resume_path:
        # Trusted (self-produced) checkpoint: weights_only=False so the bundled
        # optimizer / scheduler state loads too.
        payload = torch.load(str(resume_path), map_location=device, weights_only=False)
        strict = bool(resume_cfg.get("strict", True)) if resume_cfg else True
        load_opt = bool(resume_cfg.get("load_optimizer", True)) if resume_cfg else True
        load_sched = (
            bool(resume_cfg.get("load_scheduler", True)) if resume_cfg else True
        )
        model.load_state_dict(payload["model"], strict=strict)
        if load_opt and payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        if load_sched and lr_scheduler is not None and payload.get("lr_scheduler"):
            lr_scheduler.load_state_dict(payload["lr_scheduler"])
        if ema is not None and payload.get("ema_shadow"):
            ema.shadow = {k: v.to(device) for k, v in payload["ema_shadow"].items()}
        start_epoch = int(payload.get("epoch", 0))
        best_val = float(payload.get("best_val_loss", float("inf")))
        mode = "explicit" if explicit else "auto"
        log.info(
            "[resume:%s] Resumed from %s at epoch %d (best_val=%.4e)",
            mode,
            resume_path,
            start_epoch,
            best_val,
        )
        return start_epoch, best_val

    init_cfg = train_cfg.get("init_from_checkpoint", None)
    if init_cfg is not None and init_cfg.get("path"):
        ckpt_path = init_cfg.get("path")
        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        result = model.load_state_dict(
            payload["model"], strict=bool(init_cfg.get("strict", True))
        )
        log.info(
            "Initialised model weights from %s (missing=%d, unexpected=%d)",
            ckpt_path,
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
        return 0, float("inf")

    return 0, float("inf")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — train an AeroJEPA model on SuperWing."""
    DistributedManager.initialize()
    dm = DistributedManager()
    world_size = dm.world_size
    is_main = dm.rank == 0

    # fp16 + multi-GPU is unsupported: the manual gradient all-reduce averages
    # unscaled grads directly, with no cross-rank GradScaler coordination, so a
    # per-rank scaler could diverge (one rank skipping a step on inf/nan while
    # others step). bf16 needs no scaler and is the supported multi-GPU path.
    if world_size > 1 and str(cfg.training.precision).lower() == "fp16":
        raise ValueError(
            "fp16 + multi-GPU (world_size>1) is unsupported by the manual "
            "gradient all-reduce; use precision=bf16."
        )

    # Same seed on every rank -> identical model initialization, which manual
    # gradient averaging then keeps in lockstep. (Dataset subsampling uses its
    # own per-call RNG and is sharded by the DistributedSampler.)
    set_seed(int(cfg.seed))
    device = dm.device
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    # Checkpoints + tensorboard go to a STABLE per-run directory resolved
    # relative to the launch dir (not Hydra's timestamped run dir), so a
    # resubmitted / singleton-chained job resumes from the same place instead
    # of starting fresh in a new `outputs/<date>/<time>/` folder each time.
    run_dir = Path(to_absolute_path(str(cfg.checkpoint_root))) / str(cfg.run_name)
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    if is_main:
        log.info(
            "Hydra dir: %s | run dir: %s  (world_size=%d)",
            output_dir,
            run_dir,
            world_size,
        )

    # Data prep + loaders. Only rank 0 builds the split/stats artifacts; the
    # other ranks wait at the barrier, then read the finished files.
    if is_main:
        split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    if world_size > 1:
        dist.barrier()
    if not is_main:
        split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    train_loader, train_sampler = _build_loader(
        cfg.data,
        split="train",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        world_size=world_size,
        rank=dm.rank,
    )
    val_loader, _ = _build_loader(
        cfg.data,
        split="val",
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        batch_size=int(cfg.training.eval_batch_size),
        shuffle=False,
        world_size=world_size,
        rank=dm.rank,
    )
    if is_main:
        log.info(
            "Train / val samples: %d / %d",
            len(train_loader.dataset),
            len(val_loader.dataset),
        )

    # Model.
    model = hydra.utils.instantiate(cfg.model).to(device)
    if is_main:
        log.info(
            "Model parameters: %.2f M",
            sum(p.numel() for p in model.parameters()) / 1e6,
        )

    # Losses, optimiser, scheduler, EMA.
    recon_loss_fn = build_recon_loss_from_config(cfg.training.loss.recon).to(device)
    sigreg_loss_fn = build_sigreg_from_config(cfg.training.loss.sigreg).to(device)
    # Optional anti-collapse regularizer on the context latents (weight 0 by
    # default); a separate instance from the target-latent SIGReg.
    sigreg_context_loss_fn = build_sigreg_from_config(
        cfg.training.loss.sigreg_context
    ).to(device)
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

    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main else None

    grad_clip_norm = float(cfg.training.grad_clip_norm)
    save_every = int(cfg.training.save_every_epochs)
    max_eval_batches = int(cfg.training.max_eval_batches)
    log_every = int(cfg.training.get("log_every", 50))
    _mtb = cfg.training.get("max_train_batches", None)
    max_train_batches = int(_mtb) if _mtb is not None else None
    profile = bool(cfg.training.get("profile", False))
    profile_steps = int(cfg.training.get("profile_steps", 12))

    # fp16 autocast can overflow gradients, so pair it with a GradScaler.
    # bf16 / fp32 (and CPU) need no scaling, so the scaler is disabled there
    # and every scaler call becomes a transparent no-op.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            str(cfg.training.precision).lower() == "fp16" and device.type == "cuda"
        ),
    )

    two_phase_cfg = cfg.training.get("two_phase_training", None)

    start_epoch, best_val_loss = _load_initial_state(
        cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        device=device,
        ckpt_dir=ckpt_dir,
    )

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # Phase boundary is 1-indexed (phase1 = the first phase1_epochs epochs).
        phase = _resolve_phase(epoch + 1, two_phase_cfg)
        if phase is not None and is_main:
            log.info("epoch=%03d  phase=%s", epoch, phase["name"])
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            recon_loss_fn=recon_loss_fn,
            sigreg_loss_fn=sigreg_loss_fn,
            sigreg_context_loss_fn=sigreg_context_loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            device=device,
            precision=str(cfg.training.precision),
            grad_clip_norm=grad_clip_norm,
            loss_cfg=cfg.training.loss,
            epoch=epoch,
            max_batches=max_train_batches,
            phase=phase,
            scaler=scaler,
            writer=writer,
            log_every=log_every,
            world_size=world_size,
            is_main=is_main,
            profile=profile,
            profile_steps=profile_steps,
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
                sigreg_context_loss_fn=sigreg_context_loss_fn,
                optimizer=None,
                lr_scheduler=None,
                ema=None,
                device=device,
                precision=str(cfg.training.precision),
                grad_clip_norm=0.0,
                loss_cfg=cfg.training.loss,
                epoch=epoch,
                max_batches=max_eval_batches,
                phase=phase,
                writer=writer,
                log_every=log_every,
                world_size=world_size,
                is_main=is_main,
            )
        finally:
            if ema is not None:
                ema.restore(model)

        if is_main:
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
                    best_val=best_val_loss,
                    cfg=cfg,
                )

        # Metrics are global (reduced across ranks), so best_val_loss stays
        # identical on every rank; only rank 0 writes the checkpoint file.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            if is_main:
                _save_checkpoint(
                    path=ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    epoch=epoch + 1,
                    best_val=best_val_loss,
                    cfg=cfg,
                )

    if writer is not None:
        writer.close()
    if is_main:
        log.info("Training done. Best val_loss=%.4f", best_val_loss)
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
