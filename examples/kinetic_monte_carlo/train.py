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

import logging
import random
import time
from typing import cast

import hydra
import numpy as np
import torch
from dataset import ParticlesDataPipe
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.nn import ParticleGeoTransolver, particle_gmm_loss

from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.utils import reduce_loss
from physicsnemo.utils import (
    get_checkpoint_dir,
    load_checkpoint,
    save_checkpoint,
)
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# Numerical floor added to every stat std before division. Lets the script
# z-score variables whose empirical std is exactly zero (e.g. a field that is
# constant across the dataset) without special-casing.
_STD_EPS: float = 1e-8


def _zscore(x: Tensor, mean: float, std: float) -> Tensor:
    """Z-score a tensor with a numerical floor on the std."""
    return (x - mean) / (std + _STD_EPS)


def _stack_named_features(
    sample: TensorDict,
    names: list[str],
    stats: list[tuple[float, float]],
    reference: Tensor,
) -> Tensor:
    """Z-score each named feature in ``sample`` and stack them on a new last axis.

    Each feature is accessed by name. Returns a zero-width trailing axis
    (with ``reference``'s leading dims) when ``names`` is empty.
    """
    if not names:
        return reference[..., :0]
    cols = [
        _zscore(cast(Tensor, sample[name]), mean, std)
        for name, (mean, std) in zip(names, stats)
    ]
    return torch.stack(cols, dim=-1)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_train")
def main(cfg: DictConfig) -> None:
    """Train ParticleGeoTransolver with pure teacher forcing.

    Sample-based loop driven by ``InfiniteSampler``. Each iteration feeds the
    ground-truth state at step ``n`` to the model and regresses against the
    ground-truth next event at step ``n+1``: the inter-event delay and the
    new particle's features (everything except the delay scalar).
    """
    DistributedManager.initialize()
    dist = DistributedManager()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger = PythonLogger("main")
    logger.logger.setLevel(logging.INFO)
    rank_zero = RankZeroLoggingWrapper(logger, dist)
    rank_zero.info(f"Rank {dist.rank}/{dist.world_size} | device {dist.device}")

    checkpoint_dir = get_checkpoint_dir(
        str(cfg.io.checkpoint_dir), "kinetic_monte_carlo"
    )

    # Particle / mesh scalar feature names (x, y, z, time and the inter-event
    # delay are universal and handled separately). The model's feature counts
    # follow directly from these lists.
    particle_feature_names = list(cfg.dataset.particle_feature_names)
    mesh_feature_names = list(cfg.dataset.mesh_feature_names)
    num_particle_features = 3 + len(particle_feature_names) + 1
    num_mesh_features = 3 + len(mesh_feature_names)
    train_test_split = bool(cfg.dataset.train_test_split)

    # Pre-fetch stats up front -- the model needs the four time/delay
    # normalization scalars at construction time so the time embedding buffers
    # are populated before the first forward pass.
    stats_dataset = ParticlesDataPipe(
        data_dir=to_absolute_path(cfg.dataset.data_dir),
        batch_size_per_device=1,
        particle_feature_names=particle_feature_names,
        mesh_feature_names=mesh_feature_names,
        n_steps=cfg.dataset.n_steps,
        num_particles_max=cfg.model.num_particles_max,
        stats_file=to_absolute_path(cfg.dataset.stats_file),
        phase="train" if train_test_split else "all",
        shuffle=False,
        num_workers=0,
        process_rank=0,
        world_size=1,
        start_idx=0,
        seed=cfg.seed,
    )
    coord_mean, coord_std = stats_dataset.get_stats("coords")
    delay_mean, delay_std = stats_dataset.get_stats("delay")
    log_delay_mean, log_delay_std = stats_dataset.get_stats("log_delay")
    particle_feature_stats = [
        stats_dataset.get_stats(n) for n in particle_feature_names
    ]
    mesh_feature_stats = [stats_dataset.get_stats(n) for n in mesh_feature_names]

    model = ParticleGeoTransolver(
        num_particles_max=cfg.model.num_particles_max,
        num_particle_features=num_particle_features,
        num_mesh_features=num_mesh_features,
        num_hidden=cfg.model.num_hidden,
        num_heads=cfg.model.num_heads,
        num_blocks=cfg.model.num_blocks,
        num_flare_global_queries=cfg.model.num_flare_global_queries,
        num_context_slices=cfg.model.num_context_slices,
        time_embed_channels=cfg.model.embeddings.time_embed_channels,
        token_type_embed_dim=cfg.model.embeddings.token_type_embed_dim,
        context_token_dim=cfg.model.context_token_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        num_gmm_components=cfg.model.heads.num_gmm_components,
        dt_conditioning_embed_dim=cfg.model.heads.dt_conditioning_embed_dim,
        delay_mean=delay_mean,
        delay_std=delay_std,
        log_delay_mean=log_delay_mean,
        log_delay_std=log_delay_std,
        log_sigma_clamp=tuple(cfg.model.heads.log_sigma_clamp),
        log_sigma_smooth_beta=cfg.model.heads.log_sigma_smooth_beta,
        log_sigma_clamp_type=cfg.model.heads.log_sigma_clamp_type,
        delay_head_type=cfg.model.heads.delay_head_type,
        mesh_xyz_embedding_type=cfg.model.embeddings.mesh_xyz_type,
        time_delay_embedding_type=cfg.model.embeddings.time_delay_type,
        time_max_positions_lin=cfg.model.embeddings.time_max_positions_lin,
        time_max_positions_log=cfg.model.embeddings.time_max_positions_log,
        time_prescale_lin=cfg.model.embeddings.time_prescale_lin,
        time_prescale_log=cfg.model.embeddings.time_prescale_log,
        mesh_fourier_num_freqs=cfg.model.embeddings.mesh_fourier_num_freqs,
        mesh_fourier_base=cfg.model.embeddings.mesh_fourier_base,
    ).to(dist.device)
    rank_zero.info(f"Model parameters: {model.num_parameters():,}")

    if cfg.io.load_checkpoint:
        load_checkpoint(checkpoint_dir, models=model)

    if cfg.model.compile:
        rank_zero.info("Compiling model with torch.compile ...")
        model = torch.compile(model)

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

    current_samples_trained = 0
    if cfg.io.load_checkpoint:
        metadata = {"current_samples_trained": 0}
        load_checkpoint(checkpoint_dir, metadata_dict=metadata)
        current_samples_trained = int(metadata["current_samples_trained"])
        rank_zero.info(f"Resuming at samples trained: {current_samples_trained}")
    total_batch_size = cfg.training.batch_size_per_gpu * dist.world_size
    rank_zero.info(
        f"Batch size: per-GPU={cfg.training.batch_size_per_gpu} | "
        f"total (across {dist.world_size} ranks)={total_batch_size}"
    )
    sampler_start_idx = current_samples_trained

    train_loader = ParticlesDataPipe(
        data_dir=to_absolute_path(cfg.dataset.data_dir),
        batch_size_per_device=cfg.training.batch_size_per_gpu,
        particle_feature_names=particle_feature_names,
        mesh_feature_names=mesh_feature_names,
        n_steps=cfg.dataset.n_steps,
        num_particles_max=cfg.model.num_particles_max,
        stats_file=to_absolute_path(cfg.dataset.stats_file),
        phase="train" if train_test_split else "all",
        shuffle=True,
        num_workers=cfg.training.num_workers,
        process_rank=dist.rank,
        world_size=dist.world_size,
        start_idx=sampler_start_idx,
        seed=cfg.seed,
    )
    num_training_samples = len(train_loader.dataset)
    rank_zero.info(f"Training dataset: {num_training_samples} samples")
    train_iter = iter(train_loader)

    # Optional held-out test split (config: dataset.train_test_split). When
    # enabled, a test loss is evaluated on it every logging step and reported
    # alongside the train loss.
    test_iter = None
    if train_test_split:
        test_loader = ParticlesDataPipe(
            data_dir=to_absolute_path(cfg.dataset.data_dir),
            batch_size_per_device=cfg.training.batch_size_per_gpu,
            particle_feature_names=particle_feature_names,
            mesh_feature_names=mesh_feature_names,
            n_steps=cfg.dataset.n_steps,
            num_particles_max=cfg.model.num_particles_max,
            stats_file=to_absolute_path(cfg.dataset.stats_file),
            phase="test",
            shuffle=True,
            num_workers=cfg.training.num_workers,
            process_rank=dist.rank,
            world_size=dist.world_size,
            start_idx=0,
            seed=cfg.seed,
        )
        rank_zero.info(f"Test dataset: {len(test_loader.dataset)} samples")
        test_iter = iter(test_loader)

    feature_stats_str = " | ".join(
        f"{name}: ({m:.3g}, {s:.3g})"
        for name, (m, s) in zip(
            particle_feature_names + mesh_feature_names,
            particle_feature_stats + mesh_feature_stats,
        )
    )
    rank_zero.info(
        f"Stats | coords: ({coord_mean:.3e}, {coord_std:.3e}) | "
        f"delay: ({delay_mean:.3e}, {delay_std:.3e}) | "
        f"log_delay: ({log_delay_mean:.4f}, {log_delay_std:.4f}) | "
        f"{feature_stats_str}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.loss.weight_decay,
    )
    cfg_scheduler = cfg.training.scheduler
    if cfg_scheduler is not None and cfg_scheduler.name == "cosine_annealing":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.training.max_training_samples // num_training_samples),
            eta_min=float(cfg_scheduler.eta_min),
        )
    elif cfg_scheduler is None or cfg_scheduler.name is None:
        scheduler = None
    else:
        raise ValueError(
            f"Unknown scheduler name {cfg_scheduler.name!r}; "
            "expected null or 'cosine_annealing'."
        )
    rank_zero.info(
        f"Scheduler: {scheduler.__class__.__name__ if scheduler is not None else 'None (constant lr)'}"
    )

    if dist.world_size > 1:
        torch.distributed.barrier()
    if cfg.io.load_checkpoint:
        load_checkpoint(
            checkpoint_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
        )

    rank_zero.info("Training started...")
    samples_since_logging = 0
    samples_since_checkpoint = 0
    samples_since_scheduler_update = 0
    tick_start = time.time()

    ema_alpha = 2.0 * total_batch_size / max(1, num_training_samples)
    loss_ema: float | None = None
    loss_delay_ema: float | None = None
    loss_pf_ema: float | None = None
    loss_reg_ema: float | None = None
    grad_norm_ema: float | None = None
    loss_test_ema: float | None = None
    rank_zero.info(
        f"Loss EMA: alpha={ema_alpha:.3e} "
        f"(tau ≈ {1.0 / ema_alpha:.0f} optimizer steps ≈ 0.5 epoch)"
    )

    underlying = model.module if dist.world_size > 1 else model

    def compute_loss(sample: TensorDict) -> tuple[Tensor, dict[str, float]]:
        """Normalize a batch, run the model, and return ``(loss, per-term dict)``.

        Shared by the train step and the optional per-step test-loss
        evaluation so both follow identical normalization, forward, and loss
        patterns.
        """
        sample = sample.to(dist.device, non_blocking=True)

        # Normalize every quantity by name (no positional column indexing). The
        # whole n_steps window is normalized once, then sliced per step below.
        # Particle coordinates share the `coords` stats with the mesh; the delay
        # shares `delay` stats so it lives in the same space as t_n and Delta_t.
        particle_coords = _zscore(
            cast(Tensor, sample["particle_coords"]), coord_mean, coord_std
        )  # (B, S, K, 3)
        particle_features = _stack_named_features(
            sample, particle_feature_names, particle_feature_stats, particle_coords
        )  # (B, S, K, P)
        delay = _zscore(
            cast(Tensor, sample["delay"]), delay_mean, delay_std
        )  # (B, S, K)
        particle_state = cast(Tensor, sample["particle_state"])  # (B, S, K)
        mesh_coords = _zscore(
            cast(Tensor, sample["mesh_coords"]), coord_mean, coord_std
        )  # (B, N, 3)
        mesh_features = _stack_named_features(
            sample, mesh_feature_names, mesh_feature_stats, mesh_coords
        )  # (B, N, M)
        t_normalized = _zscore(
            cast(Tensor, sample["time"]), delay_mean, delay_std
        )  # (B, S)

        # Targets at step 1. The new particle is the slot that is real at step 1
        # but absent at step 0.
        delay_target = t_normalized[:, 1] - t_normalized[:, 0]  # (B,)
        new_particle_mask = (
            particle_state[:, 1] - particle_state[:, 0]
        ) > 0.5  # (B, K)
        # The particle-features head predicts coordinates + scalar features;
        # assemble the per-slot target in that order and select the new slot.
        target_per_slot = torch.cat(
            [particle_coords[:, 1], particle_features[:, 1]], dim=-1
        )  # (B, K, 3 + P)
        particle_features_target = (target_per_slot * new_particle_mask[..., None]).sum(
            dim=1
        )  # (B, 3 + P)

        # Forward (step-0 state) + heads + loss.
        h_g = model(
            particle_coords[:, 0],
            particle_features[:, 0],
            delay[:, 0],
            particle_state[:, 0],
            mesh_coords,
            mesh_features,
            t_normalized[:, 0],
        )
        delay_params = underlying.predict_delay(h_g)
        particle_features_params = underlying.predict_particle_features(
            h_g, delay_target
        )
        return particle_gmm_loss(
            delay_target,
            particle_features_target,
            delay_params,
            particle_features_params,
            lambda_particle_features=float(cfg.loss.lambda_particle_features),
            lambda_log_sigma=float(cfg.loss.lambda_log_sigma),
        )

    while current_samples_trained < cfg.training.max_training_samples:
        model.train()
        train_sample, _meta = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = compute_loss(train_sample)
        loss.backward()
        # Global-norm gradient clipping (cfg.loss.grad_clip_norm = null
        # disables). clip_grad_norm_ returns the *pre-clip* total norm so we
        # can log how often the cap is engaging.
        max_grad_norm = (
            float(cfg.loss.grad_clip_norm)
            if cfg.loss.grad_clip_norm is not None
            else float("inf")
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        parts["grad_norm"] = float(grad_norm)
        optimizer.step()

        # Track EMA: total loss, delay-head loss, particle-features-head loss
        loss_val = loss.item()
        loss_ema = (
            loss_val
            if loss_ema is None
            else ema_alpha * loss_val + (1.0 - ema_alpha) * loss_ema
        )
        loss_delay_ema = (
            parts["delay"]
            if loss_delay_ema is None
            else ema_alpha * parts["delay"] + (1.0 - ema_alpha) * loss_delay_ema
        )
        loss_pf_ema = (
            parts["particle_features"]
            if loss_pf_ema is None
            else ema_alpha * parts["particle_features"]
            + (1.0 - ema_alpha) * loss_pf_ema
        )
        loss_reg_ema = (
            parts["log_sigma_reg"]
            if loss_reg_ema is None
            else ema_alpha * parts["log_sigma_reg"] + (1.0 - ema_alpha) * loss_reg_ema
        )
        grad_norm_ema = (
            parts["grad_norm"]
            if grad_norm_ema is None
            else ema_alpha * parts["grad_norm"] + (1.0 - ema_alpha) * grad_norm_ema
        )

        # Held-out test loss: same forward/loss as training, no gradients.
        if test_iter is not None:
            model.eval()
            with torch.no_grad():
                test_sample, _ = next(test_iter)
                test_loss, _ = compute_loss(test_sample)
            test_val = test_loss.item()
            loss_test_ema = (
                test_val
                if loss_test_ema is None
                else ema_alpha * test_val + (1.0 - ema_alpha) * loss_test_ema
            )

        current_samples_trained += total_batch_size
        samples_since_scheduler_update += total_batch_size
        samples_since_logging += total_batch_size
        samples_since_checkpoint += total_batch_size

        if (
            scheduler is not None
            and samples_since_scheduler_update >= num_training_samples
        ):
            scheduler.step()
            samples_since_scheduler_update = 0

        if samples_since_logging >= cfg.io.logging_frequency:
            loss_sum = reduce_loss(loss_ema, dst_rank=0)
            loss_delay_sum = reduce_loss(loss_delay_ema, dst_rank=0)
            loss_pf_sum = reduce_loss(loss_pf_ema, dst_rank=0)
            loss_reg_sum = reduce_loss(loss_reg_ema, dst_rank=0)
            grad_norm_sum = reduce_loss(grad_norm_ema, dst_rank=0)
            test_sum = (
                reduce_loss(loss_test_ema, dst_rank=0)
                if loss_test_ema is not None
                else None
            )
            if dist.rank == 0:
                reduced = loss_sum / dist.world_size
                reduced_delay = loss_delay_sum / dist.world_size
                reduced_pf = loss_pf_sum / dist.world_size
                reduced_reg = loss_reg_sum / dist.world_size
                reduced_grad_norm = grad_norm_sum / dist.world_size
                elapsed = time.time() - tick_start
                steps = samples_since_logging / total_batch_size
                test_str = (
                    f" | test_loss: {test_sum / dist.world_size:.3e}"
                    if test_sum is not None
                    else ""
                )
                rank_zero.info(
                    f"samples: {current_samples_trained:>10d} | "
                    f"loss: {reduced:.3e} "
                    f"(delay: {reduced_delay:.3e}, particle_features: {reduced_pf:.3e}, "
                    f"log_sigma_reg: {reduced_reg:.3e}) | "
                    f"grad_norm (pre-clip, EMA): {reduced_grad_norm:.3e} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"throughput: {samples_since_logging / elapsed / 1000:.3f} ksamp/s | "
                    f"step: {elapsed / steps:.3f}s{test_str}"
                )
            tick_start = time.time()
            samples_since_logging = 0

        if samples_since_checkpoint >= cfg.io.checkpoint_frequency:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    checkpoint_dir,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata={"current_samples_trained": current_samples_trained},
                )
                rank_zero.info(
                    f"Saved checkpoint at samples: {current_samples_trained}"
                )
            samples_since_checkpoint = 0

    rank_zero.info("Training completed.")


if __name__ == "__main__":
    main()
