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
import time as wall_time
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from dataset import ParticlesDataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from utils.nn import (
    ParticleGeoTransolver,
    diagonal_gmm_mean_std,
    sample_diagonal_gmm,
)

from physicsnemo.utils.logging import PythonLogger

# Numerical floor added to every stat std before division. Matches train.py.
_STD_EPS: float = 1e-8

# Fallback offset (in normalized std units) above the delay lower bound used
# when rejection sampling fails to find a positive draw. Kept tiny so the
# fallback delay sits just above zero.
_REJECTION_FALLBACK_OFFSET: float = 1e-12


def _sample_gmm_with_rejection(
    logits: Tensor,
    mu: Tensor,
    log_sigma: Tensor,
    lower_bound: float,
    max_attempts: int = 100,
    generator: torch.Generator | None = None,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, float]]:
    """Sample a positive inter-event delay from the delay head's mixture.

    The delay head is trained without a hard lower bound, so its raw
    samples can fall below ``lower_bound``. This draws from the mixture
    and keeps the first sample above ``lower_bound`` for each batch
    element. If every attempt for an element falls short, it falls back
    to ``lower_bound + _REJECTION_FALLBACK_OFFSET`` so the rollout cannot
    stall; the returned info dict reports how often that fallback fired.
    """
    B, G = logits.shape
    device = logits.device
    log_pi = F.log_softmax(logits, dim=-1)
    sigma = log_sigma.exp()

    cat_u = torch.rand(max_attempts, B, G, generator=generator, device=device)
    gumbel = -torch.log(-torch.log(cat_u.clamp(min=eps, max=1.0 - eps)))
    cat_g = (log_pi.unsqueeze(0) + gumbel).argmax(dim=-1)  # (max_attempts, B)

    mu_attempts = mu.unsqueeze(0).expand(max_attempts, -1, -1)
    sigma_attempts = sigma.unsqueeze(0).expand(max_attempts, -1, -1)
    mu_g = mu_attempts.gather(2, cat_g.unsqueeze(-1)).squeeze(-1)
    sigma_g = sigma_attempts.gather(2, cat_g.unsqueeze(-1)).squeeze(-1)
    noise = torch.randn(
        max_attempts, B, generator=generator, device=device, dtype=mu.dtype
    )
    candidates = mu_g + sigma_g * noise  # (max_attempts, B)

    valid = candidates > lower_bound  # (max_attempts, B)
    any_valid = valid.any(dim=0)  # (B,)
    first_idx = valid.long().argmax(dim=0)  # (B,); 0 when no valid (handled below)
    samples = candidates.gather(0, first_idx.unsqueeze(0)).squeeze(0)
    samples = torch.where(
        any_valid,
        samples,
        torch.full_like(samples, lower_bound + _REJECTION_FALLBACK_OFFSET),
    )
    used_attempts = torch.where(
        any_valid,
        first_idx + 1,
        torch.full_like(first_idx, max_attempts),
    )
    info = {
        "rejection_rate": float((used_attempts - 1).float().mean().item()),
        "fallback_fraction": float((~any_valid).float().mean().item()),
        "max_attempts_used": int(used_attempts.max().item()),
    }
    return samples, info


def _z_score(x: Tensor, mean: float | Tensor, std: float | Tensor) -> Tensor:
    return (x - mean) / (std + _STD_EPS)


def _denormalize_mean(
    value_norm: Tensor, mean: float | Tensor, std: float | Tensor
) -> Tensor:
    return value_norm * (std + _STD_EPS) + mean


def _denormalize_std(std_norm: Tensor, std: float | Tensor) -> Tensor:
    return std_norm * (std + _STD_EPS)


def _stack_named(sample: TensorDict, names: list[str], reference: Tensor) -> Tensor:
    """Stack the named entries of ``sample`` along a new last axis.

    Each entry is accessed by name. Returns a zero-width trailing axis
    (matching ``reference``'s leading dims) when ``names`` is empty.
    """
    if not names:
        return reference[..., :0]
    return torch.stack([cast(Tensor, sample[name]) for name in names], dim=-1)


def _gt_trajectory(
    dataset: ParticlesDataset,
    geometry: str,
    sim_id: int,
    particle_feature_names: list[str],
    mesh_feature_names: list[str],
) -> dict:
    """Pull the ground-truth event trajectory and initial state for one simulation.

    Reads every quantity from the dataset by name. Returns the per-event
    ground-truth arrays (times, positions, scalar features, delays), the
    granular initial particle state (coordinates, scalar-feature block,
    delay, state), and the static mesh tensors. All physical quantities
    are kept in raw units.
    """
    P = len(particle_feature_names)
    indices = dataset.get_sim_indices(geometry, sim_id)
    times_gt: list[float] = []
    positions_gt: list[np.ndarray] = []
    scalar_features_gt: list[np.ndarray] = []
    delay_gt: list[float] = []
    mesh_coords: Tensor | None = None
    mesh_fields: Tensor | None = None
    initial: dict[str, Tensor] | None = None
    initial_time: float = 0.0

    for idx in indices:
        sample, meta = dataset[idx]
        ts_id = int(meta["ts_id"])
        coords = cast(Tensor, sample["particle_coords"])  # (S, K, 3)
        scalars = _stack_named(sample, particle_feature_names, coords)  # (S, K, P)
        if mesh_coords is None:
            mesh_coords = cast(Tensor, sample["mesh_coords"])  # (N, 3)
            mesh_fields = _stack_named(
                sample, mesh_feature_names, mesh_coords
            )  # (N, M)
        if ts_id == 0:
            # Start the rollout from whatever particles the source data carries.
            initial = {
                "coords": coords[0].clone(),  # (K, 3)
                "features": scalars[0].clone(),  # (K, P)
                "delay": cast(Tensor, sample["delay"])[0].clone(),  # (K,)
                "state": cast(Tensor, sample["particle_state"])[0].clone(),  # (K,)
            }
            initial_time = float(cast(Tensor, sample["time"])[0].item())
            continue
        # The most recently created particle is the newest occupied slot.
        # Derived from the live particle count so a non-empty initial state
        # (ts=0 already populated) is handled correctly.
        j = int(cast(Tensor, sample["particle_state"])[0].sum().item()) - 1
        times_gt.append(float(cast(Tensor, sample["time"])[0].item()))
        positions_gt.append(coords[0, j].numpy())
        scalar_features_gt.append(scalars[0, j].numpy())
        delay_gt.append(float(cast(Tensor, sample["delay"])[0, j].item()))

    if mesh_coords is None or mesh_fields is None or initial is None:
        raise RuntimeError(f"Missing data for ({geometry}, sim={sim_id}).")
    return {
        "times_gt": np.array(times_gt, dtype=np.float64),
        "positions_gt": np.stack(positions_gt) if positions_gt else np.zeros((0, 3)),
        "scalar_features_gt": (
            np.stack(scalar_features_gt)
            if scalar_features_gt
            else np.zeros((0, P), dtype=np.float32)
        ),
        "delay_gt": np.array(delay_gt, dtype=np.float32),
        "mesh_coords": mesh_coords,
        "mesh_fields": mesh_fields,
        "initial": initial,
        "initial_time": initial_time,
    }


@torch.no_grad()
def _rollout_ensemble(
    model: ParticleGeoTransolver,
    gt: dict,
    stats: dict,
    num_particles_max: int,
    num_ensemble: int,
    device: torch.device,
    logger,
    logging_frequency: int,
    label: str,
) -> dict:
    """Sample an ensemble of autoregressive trajectories from the trained model.

    All ensemble members share the same initial particle state and the
    same static mesh; they diverge purely through independent draws from
    the two predicted GMMs (next-event delay + new particle's features).
    For every emitted event the function also stores the closed-form mean
    and standard deviation of the predicted distribution, denormalized to
    physical units, so callers can plot uncertainty bands.

    Each ensemble member iterates until its simulation time reaches the
    ground-truth final event time or until the slot budget is exhausted;
    per-member states evolve independently. Returned arrays are padded with
    NaN past each member's actual event count and come with an explicit
    ``n_pred`` count vector so callers can mask.
    """
    P = len(stats["particle_feature_stats"])
    t_end = float(gt["times_gt"][-1]) if gt["times_gt"].size > 0 else 0.0
    B = num_ensemble
    K = num_particles_max

    # Per-feature statistics as tensors, for broadcasting over the scalar blocks.
    pf_mean = torch.tensor(
        [m for m, _ in stats["particle_feature_stats"]], device=device
    )  # (P,)
    pf_std = torch.tensor(
        [s for _, s in stats["particle_feature_stats"]], device=device
    )
    mf_mean = torch.tensor(
        [m for m, _ in stats["mesh_feature_stats"]], device=device
    )  # (M,)
    mf_std = torch.tensor([s for _, s in stats["mesh_feature_stats"]], device=device)

    # Static mesh inputs (replicated across the ensemble) are normalized once.
    mesh_coords = (
        gt["mesh_coords"].to(device).unsqueeze(0).expand(B, -1, -1)
    )  # (B, N, 3)
    mesh_fields = (
        gt["mesh_fields"].to(device).unsqueeze(0).expand(B, -1, -1)
    )  # (B, N, M)
    mesh_coords_n = _z_score(mesh_coords, stats["coord_mean"], stats["coord_std"])
    mesh_fields_n = _z_score(mesh_fields, mf_mean, mf_std)

    # Replicate the granular initial particle state to every ensemble member.
    init = gt["initial"]
    if init["state"].shape[0] != K:
        raise ValueError(
            f"Initial-state slot count ({init['state'].shape[0]}) does not match "
            f"the rollout's num_particles_max={K}. Align "
            "config_generate.dataset.num_particles_max with training."
        )
    coords_raw = (
        init["coords"].to(device).unsqueeze(0).expand(B, -1, -1).clone()
    )  # (B,K,3)
    scalars_raw = (
        init["features"].to(device).unsqueeze(0).expand(B, -1, -1).clone()
    )  # (B,K,P)
    delay_raw = init["delay"].to(device).unsqueeze(0).expand(B, -1).clone()  # (B, K)
    particle_state = (
        init["state"].to(device).unsqueeze(0).expand(B, -1).clone()
    )  # (B, K)

    # Per-member rollout clocks. t_b_last_raw = time of the most recent event
    # (or initial_time for empty initial state); n_e[b] = current event count.
    initial_time = gt["initial_time"]
    t_raw = torch.full((B,), initial_time, dtype=torch.float32, device=device)
    t_b_last_raw = torch.full((B,), initial_time, dtype=torch.float32, device=device)
    n_e = init["state"].to(device).sum().long().repeat(B)  # (B,) initial event count
    done = (t_raw >= t_end) | (n_e >= K)

    delay_lower_bound_normalized = float(
        -stats["delay_mean"] / (stats["delay_std"] + _STD_EPS)
    )

    # Padded per-member trajectory storage; one row per ensemble member, one
    # column per emitted event (NaN-filled past n_pred[b]).
    times_pred = np.full((B, K), np.nan, dtype=np.float64)
    positions_pred = np.full((B, K, 3), np.nan, dtype=np.float32)
    scalar_features_pred = np.full((B, K, P), np.nan, dtype=np.float32)
    delay_pred = np.full((B, K), np.nan, dtype=np.float32)
    delay_mean_pred = np.full((B, K), np.nan, dtype=np.float32)
    delay_std_pred = np.full((B, K), np.nan, dtype=np.float32)
    positions_mean_pred = np.full((B, K, 3), np.nan, dtype=np.float32)
    positions_std_pred = np.full((B, K, 3), np.nan, dtype=np.float32)
    scalar_features_mean_pred = np.full((B, K, P), np.nan, dtype=np.float32)
    scalar_features_std_pred = np.full((B, K, P), np.nan, dtype=np.float32)

    rollout_start = wall_time.time()
    iteration = 0
    while not bool(done.all()):
        iteration += 1
        # Normalize the evolving granular state (mesh was normalized once above).
        coords_n = _z_score(coords_raw, stats["coord_mean"], stats["coord_std"])
        scalars_n = _z_score(scalars_raw, pf_mean, pf_std)
        delay_n = _z_score(delay_raw, stats["delay_mean"], stats["delay_std"])
        t_n = _z_score(t_raw, stats["delay_mean"], stats["delay_std"])

        h_g = model(
            coords_n,
            scalars_n,
            delay_n,
            particle_state,
            mesh_coords_n,
            mesh_fields_n,
            t_n,
        )
        delay_logits, delay_mu, delay_log_sigma, _ = model.predict_delay(
            h_g
        )  # mu, log_sigma: (B, G, 1)
        # The delay head is a plain GMM on R; reject below-bound draws so
        # every sampled inter-event delay is positive. The delay is the
        # D == 1 case, so squeeze the trailing axis for the 1-D sampler.
        delay_norm_sample, rejection_info = _sample_gmm_with_rejection(
            delay_logits,
            delay_mu.squeeze(-1),
            delay_log_sigma.squeeze(-1),
            lower_bound=delay_lower_bound_normalized,
        )  # (B,)
        delay_mix_mean_d, delay_mix_std_d = diagonal_gmm_mean_std(
            delay_logits, delay_mu, delay_log_sigma
        )  # (B, 1)
        delay_norm_mix_mean = delay_mix_mean_d.squeeze(-1)  # (B,)
        delay_norm_mix_std = delay_mix_std_d.squeeze(-1)  # (B,)
        pf_logits, pf_mu, pf_log_sigma, _ = model.predict_particle_features(
            h_g, delay_norm_sample
        )
        pf_norm_sample = sample_diagonal_gmm(
            pf_logits, pf_mu, pf_log_sigma
        )  # (B, 3 + P)
        pf_norm_mix_mean, pf_norm_mix_std = diagonal_gmm_mean_std(
            pf_logits, pf_mu, pf_log_sigma
        )  # (B, 3 + P)

        # Split the predicted vector into coordinates (first 3) and scalar
        # features (the rest), then denormalize each. Means use mu + sigma * z,
        # stds use sigma * z (no shift); the scalar block is denormalized in one
        # broadcast against the per-feature stat tensors.
        delay_sample_raw = _denormalize_mean(
            delay_norm_sample, stats["delay_mean"], stats["delay_std"]
        )
        xyz_sample_raw = _denormalize_mean(
            pf_norm_sample[:, 0:3], stats["coord_mean"], stats["coord_std"]
        )
        scalar_sample_raw = _denormalize_mean(pf_norm_sample[:, 3:], pf_mean, pf_std)
        delay_mix_mean_raw = _denormalize_mean(
            delay_norm_mix_mean, stats["delay_mean"], stats["delay_std"]
        )
        delay_mix_std_raw = _denormalize_std(delay_norm_mix_std, stats["delay_std"])
        xyz_mix_mean_raw = _denormalize_mean(
            pf_norm_mix_mean[:, 0:3], stats["coord_mean"], stats["coord_std"]
        )
        xyz_mix_std_raw = _denormalize_std(pf_norm_mix_std[:, 0:3], stats["coord_std"])
        scalar_mix_mean_raw = _denormalize_mean(
            pf_norm_mix_mean[:, 3:], pf_mean, pf_std
        )
        scalar_mix_std_raw = _denormalize_std(pf_norm_mix_std[:, 3:], pf_std)

        # Per-member ghost slot and updated clocks.
        ghost_idx = n_e.clamp(max=K - 1)  # (B,) safe even if a member just finished
        delay_clamped = torch.where(
            done, torch.zeros_like(delay_sample_raw), delay_sample_raw
        )
        t_b_new = t_raw + delay_clamped
        tau_p_raw = (t_raw - t_b_last_raw) + delay_clamped

        active = ~done
        active_idx = active.nonzero(as_tuple=True)[0]

        # Write the newly created particle into each active member's slot.
        if active_idx.numel() > 0:
            b_idx = active_idx
            g_idx = ghost_idx[b_idx]
            coords_raw[b_idx, g_idx] = xyz_sample_raw[b_idx]
            scalars_raw[b_idx, g_idx] = scalar_sample_raw[b_idx]
            delay_raw[b_idx, g_idx] = tau_p_raw[b_idx]
            particle_state[b_idx, g_idx] = 1.0

            # Persist trajectory + predicted-distribution stats for this event.
            b_np = b_idx.cpu().numpy()
            g_np = g_idx.cpu().numpy()
            times_pred[b_np, g_np] = t_b_new[b_idx].cpu().numpy()
            positions_pred[b_np, g_np] = xyz_sample_raw[b_idx].cpu().numpy()
            scalar_features_pred[b_np, g_np] = scalar_sample_raw[b_idx].cpu().numpy()
            delay_pred[b_np, g_np] = delay_sample_raw[b_idx].cpu().numpy()
            delay_mean_pred[b_np, g_np] = delay_mix_mean_raw[b_idx].cpu().numpy()
            delay_std_pred[b_np, g_np] = delay_mix_std_raw[b_idx].cpu().numpy()
            positions_mean_pred[b_np, g_np] = xyz_mix_mean_raw[b_idx].cpu().numpy()
            positions_std_pred[b_np, g_np] = xyz_mix_std_raw[b_idx].cpu().numpy()
            scalar_features_mean_pred[b_np, g_np] = (
                scalar_mix_mean_raw[b_idx].cpu().numpy()
            )
            scalar_features_std_pred[b_np, g_np] = (
                scalar_mix_std_raw[b_idx].cpu().numpy()
            )

            t_b_last_raw[b_idx] = t_b_new[b_idx]
            t_raw[b_idx] = t_b_new[b_idx]
            n_e[b_idx] = n_e[b_idx] + 1

        # Refresh per-member "done" flags after the update.
        done = (t_raw >= t_end) | (n_e >= K)

        if iteration % logging_frequency == 0:
            min_n = int(n_e.min().item())
            max_n = int(n_e.max().item())
            n_done = int(done.sum().item())
            logger.info(
                f"[{label}] iter {iteration:>5d} | events/member in [{min_n}, {max_n}] "
                f"| done {n_done}/{B} | t in [{float(t_raw.min()):.2e}, "
                f"{float(t_raw.max()):.2e}] | rejection_rate="
                f"{rejection_info['rejection_rate']:.3f} "
                f"fallback_frac={rejection_info['fallback_fraction']:.3f}"
            )

    elapsed = wall_time.time() - rollout_start
    n_pred = n_e.cpu().numpy().astype(np.int64)
    logger.info(
        f"[{label}] rollout done in {elapsed:.1f}s | "
        f"ensemble events/member in [{int(n_pred.min())}, {int(n_pred.max())}] "
        f"| t_end_gt={t_end:.3e}"
    )

    return {
        "times_pred": times_pred,
        "positions_pred": positions_pred,
        "scalar_features_pred": scalar_features_pred,
        "delay_pred": delay_pred,
        "delay_mean_pred": delay_mean_pred,
        "delay_std_pred": delay_std_pred,
        "positions_mean_pred": positions_mean_pred,
        "positions_std_pred": positions_std_pred,
        "scalar_features_mean_pred": scalar_features_mean_pred,
        "scalar_features_std_pred": scalar_features_std_pred,
        "n_pred": n_pred,
    }


@hydra.main(version_base="1.3", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Sample autoregressive trajectories from a trained checkpoint and save them to disk.

    Run this after training to produce one ``.pth`` per ``(geometry,
    sim_id)`` pair listed in the config. Each file holds the ground-truth
    trajectory and an ensemble of ``rollout.num_ensemble`` model-generated
    trajectories that share the same initial state and mesh but diverge
    through independent samples from the two GMM heads. Each generated
    event also carries the closed-form mean and standard deviation of its
    predicted distribution, denormalized to physical units, ready to be
    consumed by the plotting utilities under ``utils/``.

    Use ``sim_id: "all"`` inside ``rollout.simulations`` to expand into
    every simulation available for that geometry group.
    """
    logger = PythonLogger("generate")
    logger.logger.setLevel(logging.INFO)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ckpt_path = to_absolute_path(str(cfg.checkpoint))
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = cast(
        ParticleGeoTransolver,
        ParticleGeoTransolver.from_checkpoint(ckpt_path),
    )
    model = model.to(device).eval()
    logger.info(f"Model parameters: {model.num_parameters():,}")

    particle_feature_names = list(cfg.dataset.particle_feature_names)
    mesh_feature_names = list(cfg.dataset.mesh_feature_names)

    dataset = ParticlesDataset(
        data_dir=to_absolute_path(cfg.dataset.data_dir),
        particle_feature_names=particle_feature_names,
        mesh_feature_names=mesh_feature_names,
        n_steps=1,
        num_particles_max=cfg.dataset.num_particles_max,
        stats_file=to_absolute_path(cfg.dataset.stats_file),
    )
    logger.info(f"Dataset: {len(dataset)} samples | geometries: {dataset.geometries()}")

    coord_mean, coord_std = dataset.get_stats("coords")
    delay_mean, delay_std = dataset.get_stats("delay")
    log_delay_mean, log_delay_std = dataset.get_stats("log_delay")
    stats = {
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "delay_mean": delay_mean,
        "delay_std": delay_std,
        "log_delay_mean": log_delay_mean,
        "log_delay_std": log_delay_std,
        "particle_feature_stats": [
            dataset.get_stats(n) for n in particle_feature_names
        ],
        "mesh_feature_stats": [dataset.get_stats(n) for n in mesh_feature_names],
    }
    feature_stats_str = " | ".join(
        f"{name}: ({m:.3g}, {s:.3g})"
        for name, (m, s) in zip(
            particle_feature_names + mesh_feature_names,
            stats["particle_feature_stats"] + stats["mesh_feature_stats"],
        )
    )
    logger.info(
        f"Stats | coords: ({coord_mean:.3e}, {coord_std:.3e}) | "
        f"delay: ({delay_mean:.3e}, {delay_std:.3e}) | "
        f"log_delay: ({log_delay_mean:.4f}, {log_delay_std:.4f}) | {feature_stats_str}"
    )
    # Sanity-check the log-branch stat baked into the checkpoint against the
    # loaded stats.json (only the log_concat time embedding carries it).
    if hasattr(model.time_embed, "log_delay_mean"):
        embed_log_mean = float(model.time_embed.log_delay_mean.item())
        if abs(embed_log_mean - log_delay_mean) > 1e-3 * (abs(log_delay_mean) + 1.0):
            logger.warning(
                f"Checkpoint's time_embed.log_delay_mean = {embed_log_mean:.4f} "
                f"differs from stats.json log_delay.mean = {log_delay_mean:.4f}; "
                "the rollout will use the values baked into the checkpoint."
            )

    output_dir = Path(to_absolute_path(str(cfg.io.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    pairs: list[tuple[str, int]] = []
    for sim_cfg in cfg.rollout.simulations:
        geometry = str(sim_cfg.geometry)
        sim_id_raw = sim_cfg.sim_id
        if isinstance(sim_id_raw, str) and sim_id_raw.lower() == "all":
            pairs.extend((geometry, sid) for sid in dataset.get_sim_ids(geometry))
        else:
            pairs.append((geometry, int(sim_id_raw)))
    num_ensemble = int(cfg.rollout.num_ensemble)
    logging_frequency = int(cfg.io.logging_frequency)
    logger.info(
        f"Rolling out {len(pairs)} simulation(s) "
        f"| ensemble size per simulation: {num_ensemble}"
    )

    overall_start = wall_time.time()
    for geometry, sim_id in pairs:
        label = f"{geometry}/sim{sim_id}"
        logger.info(f"[{label}] gathering GT trajectory ...")
        gt = _gt_trajectory(
            dataset, geometry, sim_id, particle_feature_names, mesh_feature_names
        )
        if gt["times_gt"].size == 0:
            logger.warning(f"[{label}] no GT events; skipping.")
            continue
        logger.info(
            f"[{label}] GT has {gt['times_gt'].size} events "
            f"| t_end_gt={gt['times_gt'][-1]:.3e} "
            f"| initial particles={int(gt['initial']['state'].sum().item())}"
        )

        pred = _rollout_ensemble(
            model=model,
            gt=gt,
            stats=stats,
            num_particles_max=cfg.dataset.num_particles_max,
            num_ensemble=num_ensemble,
            device=device,
            logger=logger,
            logging_frequency=logging_frequency,
            label=label,
        )

        out = {
            "geometry": geometry,
            "sim_id": sim_id,
            "particle_feature_names": particle_feature_names,
            "mesh_positions": gt["mesh_coords"],
            "initial_particle_coords": gt["initial"]["coords"],
            "initial_particle_state": gt["initial"]["state"],
            "initial_time": float(gt["initial_time"]),
            # Ground truth trajectory (raw physical units).
            "times_gt": torch.from_numpy(gt["times_gt"]).to(torch.float64),
            "positions_gt": torch.from_numpy(gt["positions_gt"]).to(torch.float32),
            "scalar_features_gt": torch.from_numpy(gt["scalar_features_gt"]),
            "delay_gt": torch.from_numpy(gt["delay_gt"]),
            # Predicted trajectory ensemble (raw physical units, NaN past n_pred).
            "times_pred": torch.from_numpy(pred["times_pred"]).to(torch.float64),
            "positions_pred": torch.from_numpy(pred["positions_pred"]),
            "scalar_features_pred": torch.from_numpy(pred["scalar_features_pred"]),
            "delay_pred": torch.from_numpy(pred["delay_pred"]),
            # Predicted-distribution stats per event (denormalized).
            "delay_mean_pred": torch.from_numpy(pred["delay_mean_pred"]),
            "delay_std_pred": torch.from_numpy(pred["delay_std_pred"]),
            "positions_mean_pred": torch.from_numpy(pred["positions_mean_pred"]),
            "positions_std_pred": torch.from_numpy(pred["positions_std_pred"]),
            "scalar_features_mean_pred": torch.from_numpy(
                pred["scalar_features_mean_pred"]
            ),
            "scalar_features_std_pred": torch.from_numpy(
                pred["scalar_features_std_pred"]
            ),
            "n_pred": torch.from_numpy(pred["n_pred"]),
        }
        sim_dir = output_dir / geometry
        sim_dir.mkdir(parents=True, exist_ok=True)
        out_path = sim_dir / f"rollout_sim{sim_id}.pth"
        torch.save(out, out_path)
        logger.info(f"[{label}] saved → {out_path}")

    logger.info(
        f"All rollouts complete in {wall_time.time() - overall_start:.1f}s. "
        f"Outputs in {output_dir}"
    )


if __name__ == "__main__":
    main()
