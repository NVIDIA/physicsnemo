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

"""Standalone evaluation for FGN — paper §4 (arXiv:2506.10772v1).

Iterates the full validation split, runs an M-member AR ensemble rollout for K
lead times, and accumulates the diagnostics shown in Figures 2–3:

  - Fair CRPS per variable per lead (area-weighted via cos-lat)
  - Ensemble-mean RMSE per variable per lead (area-weighted)
  - Spread-skill ratio per variable per lead (area-weighted)
  - Rank histograms per variable (aggregated over all leads)
  - Average-pooled CRPS and max-pooled CRPS (Figure 3 a-b)
  - Derived-variable CRPS: wind speed and z300-z500 (Figure 3 c)
  - Azimuthal power spectra at the final lead (Figure 3 e-j)
  - Fair energy score per lead (multivariate CRPS)

Per-variable CRPS, RMSE, and rank histograms use
``earth2studio.statistics.{crps, rmse, rank_histogram}`` with
``earth2studio.statistics.lat_weight`` for area weighting.
Pooled CRPS and power spectra use the lightweight torch kernels in
``utils/metrics.py``.

All results are written as ``eval_metrics.npz`` + PNG plots to ``eval.outdir``.

Usage::

    python eval.py --config-name eval_fgn \\
        dataset.stats_path=rundir/fgn_2024_val/stats_2024.npz \\
        eval.checkpoint=rundir/fgn_2024_long/0/checkpoints/FGNUNet.0.5000.mdlus

Deep-ensemble::

    python eval.py --config-name eval_fgn \\
        dataset.stats_path=... \\
        "eval.checkpoints=[seed0/FGNUNet.mdlus,seed1/FGNUNet.mdlus]"
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import hydra
import numpy as np
import torch
from datasets import dataset_classes
from datasets.dataset import worker_init
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from utils.config import EvalMainConfig
from utils.metrics import (
    REV_CL_RATIOS,
    REV_THRESHOLD_NAMES,
    REV_THRESHOLDS,
    derived_variable_crps,
    energy_score_per_lead,
    plot_crps_scorecard,
    plot_metric_vs_lead,
    plot_pooled_crps,
    plot_power_spectra,
    plot_rank_histograms,
    plot_rev_curves,
    plot_spread_skill_lines,
    pooled_crps_per_lead,
    power_spectra_per_variable,
    rev_score,
    save_summary,
)
from utils.trainer import find_latest_model_checkpoint

from earth2studio.statistics import crps as e2s_crps
from earth2studio.statistics import rank_histogram as e2s_rh
from earth2studio.statistics import rmse as e2s_rmse
from earth2studio.statistics.weights import lat_weight
from physicsnemo.core import Module

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_checkpoints(cfg: DictConfig) -> list[str]:
    checkpoints = getattr(cfg.eval, "checkpoints", None)
    if checkpoints:
        return [str(c) for c in checkpoints]
    checkpoint = cfg.eval.checkpoint
    if checkpoint == "latest":
        ckpt_dir = Path(cfg.training.rundir) / cfg.training.checkpoint_dir
        return [str(find_latest_model_checkpoint(ckpt_dir))]
    return [str(checkpoint)]


def _make_coords(
    B: int,
    M: int,
    variables: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[OrderedDict, OrderedDict]:
    """CoordSystem pair for (B, M, C, H, W) ensemble and (B, C, H, W) target."""
    x_coords = OrderedDict(
        [
            ("batch", np.arange(B)),
            ("ensemble", np.arange(M)),
            ("variable", np.array(variables)),
            ("lat", lats),
            ("lon", lons),
        ]
    )
    y_coords = OrderedDict(
        [
            ("batch", np.arange(B)),
            ("variable", np.array(variables)),
            ("lat", lats),
            ("lon", lons),
        ]
    )
    return x_coords, y_coords


def _ar_rollout_steps(
    model: torch.nn.Module,
    history: torch.Tensor,
    background: torch.Tensor,
    invariants: torch.Tensor | None,
    num_steps: int,
    latent_dim: int,
    num_members: int,
    device: torch.device,
    output_only: list[int],
):
    """Generator yielding ``(B, M, C, H, W)`` predictions one AR step at a time.

    Mirrors earth2studio's ``yield`` / ``del`` pattern (e.g. GenCast mini):
    each step's tensor is freed after the caller processes it, so only one
    step's worth of data is in memory at a time (~2–3 GB at 0.25° vs. 51 GB
    for the full stacked rollout).
    """
    B, T, C, H, W = history.shape
    per_member_hist = (
        history.unsqueeze(1).expand(B, num_members, T, C, H, W).contiguous()
    )
    for k in range(num_steps):
        members: list[torch.Tensor] = []
        for n in range(num_members):
            latent = torch.randn(B, latent_dim, device=device, dtype=torch.float32)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
            ):
                pred = model(
                    history=per_member_hist[:, n],
                    latent=latent,
                    background=background,
                    invariants=invariants,
                ).float()
            members.append(pred)
        preds_k = torch.stack(members, dim=1)  # (B, M, C, H, W) on GPU
        yield preds_k  # caller processes; history update waits
        # Update history for the next step after the caller is done with preds_k
        if k < num_steps - 1:
            next_frame = preds_k
            if output_only:
                next_frame = next_frame.clone()
                for ci in output_only:
                    next_frame[:, :, ci].zero_()
            per_member_hist = torch.cat(
                [per_member_hist[:, :, 1:], next_frame.unsqueeze(2)], dim=2
            )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def run_eval(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ecfg = EvalMainConfig(**cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)

    # --- Dataset ---
    dataset_cls = dataset_classes[cfg.dataset.name]
    val_dataset = dataset_cls(cfg.dataset, train=False)
    num_workers = int(ecfg.eval.num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(ecfg.eval.batch_size),
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init if num_workers else None,
    )
    log.info(f"Val dataset: {len(val_dataset)} samples, {len(val_loader)} batches")

    invariants = val_dataset.get_invariants()
    if invariants is not None:
        invariants = torch.from_numpy(invariants).to(device, dtype=torch.float32)

    variables = val_dataset.state_channels()
    output_only = val_dataset.output_only_channels()
    C = len(variables)
    K = int(ecfg.eval.future_steps)
    M = int(ecfg.eval.ensemble_size)
    latent_dim = int(ecfg.model.latent_dim)
    pool_sizes = list(ecfg.eval.pool_sizes)
    step_hours = int(getattr(cfg.dataset, "step_hours", 6))
    spatial_stride = int(getattr(cfg.dataset, "spatial_stride", 1))

    # --- lat/lon grids and area weights (earth2studio.statistics.lat_weight) ---
    from datasets.arco import ARCO_LAT, ARCO_LON

    lats_np = ARCO_LAT[::spatial_stride]
    lons_np = ARCO_LON[::spatial_stride]
    H, W = len(lats_np), len(lons_np)

    area_w_1d = lat_weight(torch.from_numpy(lats_np))  # (H,) cos-lat
    area_w_2d = area_w_1d.unsqueeze(-1).expand(H, W).contiguous()  # (H, W)

    # --- earth2studio stat objects (area-weighted, per-batch per-lead) ---
    # Paper §4.1: evaluation uses biased CRPS (fair=False) because the deep
    # ensemble violates the independence assumption of the unbiased estimator.
    crps_fn = e2s_crps(
        ensemble_dimension="ensemble",
        reduction_dimensions=["lat", "lon"],
        weights=area_w_2d,
        fair=False,
    )
    rmse_fn = e2s_rmse(
        reduction_dimensions=["lat", "lon"],
        weights=area_w_2d,
        ensemble_dimension="ensemble",
    )
    rh_fn = e2s_rh(
        ensemble_dimension="ensemble",
        reduction_dimensions=["lat", "lon"],
    )

    # --- Load model(s) ---
    checkpoint_paths = _resolve_checkpoints(cfg)
    log.info(f"Checkpoints: {checkpoint_paths}")
    models: list[torch.nn.Module] = []
    for ckpt_path in checkpoint_paths:
        models.append(Module.from_checkpoint(ckpt_path).to(device).eval())

    n_models = len(models)
    base_m = M // n_models
    rem_m = M - base_m * n_models
    members_per_model = [base_m + (1 if i < rem_m else 0) for i in range(n_models)]

    # --- Accumulators ---
    crps_acc = np.zeros((K, C), dtype=np.float64)
    rmse_acc = np.zeros((K, C), dtype=np.float64)
    spread_acc = np.zeros((K, C), dtype=np.float64)
    rank_acc = np.zeros((M + 1, C), dtype=np.float64)
    energy_acc = np.zeros(K, dtype=np.float64)
    # Spectra / pooled initialised on first batch (size depends on H, W, nbins)
    power_ens_acc: np.ndarray | None = None
    power_tgt_acc: np.ndarray | None = None
    n_thresh = len(REV_THRESHOLDS)
    n_cl = len(REV_CL_RATIOS)
    rev_acc = np.zeros((K, n_thresh, n_cl, C), dtype=np.float64)
    pooled_avg_acc = np.zeros((len(pool_sizes), K, C), dtype=np.float64)
    pooled_max_acc = np.zeros((len(pool_sizes), K, C), dtype=np.float64)
    derived_acc: dict[str, np.ndarray] = {}
    n_batches = 0

    # --- Eval loop ---
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            history = batch["history"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.float32)
            background = batch["background"].to(device, dtype=torch.float32)
            if target.ndim == 4:
                target = target.unsqueeze(1)

            B = history.shape[0]
            inv_b = (
                invariants.unsqueeze(0).expand(B, -1, -1, -1)
                if invariants is not None
                else None
            )
            xc, yc = _make_coords(B, M, variables, lats_np, lons_np)

            # One generator per model; advance all in lockstep, one step at a time.
            # This keeps only (B, M, C, H, W) per step in memory instead of the
            # full (B, K, M, C, H, W) rollout (~2-3 GB vs. ~51 GB at 0.25°).
            model_iters = [
                _ar_rollout_steps(
                    mdl,
                    history,
                    background,
                    inv_b,
                    K,
                    latent_dim,
                    n_mem,
                    device,
                    output_only,
                )
                for mdl, n_mem in zip(models, members_per_model, strict=True)
                if n_mem > 0
            ]

            for k in range(K):
                # Gather one step from each model, concatenate along member dim
                parts = [next(it) for it in model_iters]  # list of (B, n_mem, C, H, W)
                preds_k = torch.cat(parts, dim=1)  # (B, M, C, H, W) on GPU
                tgt_k = target[:, k]  # (B, C, H, W) on GPU

                # --- earth2studio area-weighted per-lead metrics ---
                crps_res, _ = crps_fn(preds_k, xc, tgt_k, yc)
                crps_acc[k] += crps_res.mean(dim=0).cpu().numpy()

                rmse_res, _ = rmse_fn(preds_k, xc, tgt_k, yc)
                rmse_acc[k] += rmse_res.mean(dim=0).cpu().numpy()

                ens_var = preds_k.var(dim=1, unbiased=True)
                w = area_w_2d.to(ens_var.device)
                spread_kc = (ens_var * w).sum(dim=(-2, -1)) / w.sum()
                spread_acc[k] += spread_kc.sqrt().mean(dim=0).cpu().numpy()

                rh_res, _ = rh_fn(preds_k, xc, tgt_k, yc)
                rank_acc += rh_res[1].sum(dim=-2).cpu().numpy()

                # --- full-rollout metrics, called with K=1 via unsqueeze ---
                pk1 = preds_k.unsqueeze(1)  # (B, 1, M, C, H, W)
                tk1 = tgt_k.unsqueeze(1)  # (B, 1, C, H, W)

                energy_acc[k] += float(energy_score_per_lead(pk1, tk1)[0])

                p_avg = pooled_crps_per_lead(pk1, tk1, pool_sizes, "avg")  # (P, 1, C)
                p_max = pooled_crps_per_lead(pk1, tk1, pool_sizes, "max")
                pooled_avg_acc[:, k] += (
                    p_avg[:, 0].cpu().numpy()
                    if isinstance(p_avg, torch.Tensor)
                    else p_avg[:, 0]
                )
                pooled_max_acc[:, k] += (
                    p_max[:, 0].cpu().numpy()
                    if isinstance(p_max, torch.Tensor)
                    else p_max[:, 0]
                )

                ens_mean_k = preds_k.mean(dim=1)  # (B, C, H, W)
                k_vec, ens_spec_k, tgt_spec_k = power_spectra_per_variable(
                    ens_mean_k.unsqueeze(1),
                    tk1,  # (B, 1, C, H, W)
                )  # returns (1, C, nbins) for K=1
                if power_ens_acc is None:
                    nbins = ens_spec_k.shape[-1]
                    power_ens_acc = np.zeros((K, C, nbins), dtype=np.float64)
                    power_tgt_acc = np.zeros((K, C, nbins), dtype=np.float64)
                power_ens_acc[k] += ens_spec_k[0]
                power_tgt_acc[k] += tgt_spec_k[0]

                for dname, vals in derived_variable_crps(pk1, tk1, variables).items():
                    if dname not in derived_acc:
                        derived_acc[dname] = np.zeros(K, dtype=np.float64)
                    derived_acc[dname][k] += float(vals[0])

                # REV (Richardson 2000) — paper §4.1 / Figure 2 g-h
                rev_acc[k] += rev_score(preds_k, tgt_k, REV_THRESHOLDS, REV_CL_RATIOS)

                del preds_k, parts, pk1, tk1, ens_mean_k
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            n_batches += 1
            if (batch_idx + 1) % 50 == 0:
                log.info(f"  {batch_idx + 1}/{len(val_loader)} batches done")

    if n_batches == 0:
        raise RuntimeError("Validation dataset is empty.")

    # --- Normalise ---
    crps_mean = crps_acc / n_batches
    rmse_mean = rmse_acc / n_batches
    spread_mean = spread_acc / n_batches
    ratio_mean = spread_mean / np.maximum(rmse_mean, 1e-12)
    energy_mean = energy_acc / n_batches
    power_ens_mean = power_ens_acc / n_batches
    power_tgt_mean = power_tgt_acc / n_batches
    pooled_avg_mean = pooled_avg_acc / n_batches
    pooled_max_mean = pooled_max_acc / n_batches

    leads = np.arange(1, K + 1, dtype=np.int64)
    lead_hours = leads * step_hours

    # --- Save ---
    out_dir = Path(ecfg.eval.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "crps_per_lead_per_channel": crps_mean,
        "rmse_per_lead_per_channel": rmse_mean,
        "spread_per_lead_per_channel": spread_mean,
        "spread_skill_ratio": ratio_mean,
        "rank_histograms": rank_acc,
        "energy_score_per_lead": energy_mean,
        "avg_pooled_crps": pooled_avg_mean,
        "max_pooled_crps": pooled_max_mean,
        "pool_sizes": np.array(pool_sizes, dtype=np.int64),
        "power_spectrum_k": k_vec,
        "power_spectrum_forecast": power_ens_mean,
        "power_spectrum_truth": power_tgt_mean,
        "variables": np.array(variables, dtype=object),
        "lead_steps": leads,
        "lead_hours": lead_hours,
        "num_batches": np.array(n_batches),
        "checkpoint_paths": np.array(checkpoint_paths, dtype=object),
    }
    for dname, vals in derived_acc.items():
        summary[f"derived_crps_{dname}"] = vals / n_batches
    rev_mean = rev_acc / n_batches
    summary["rev_per_lead"] = rev_mean  # (K, n_thresh, n_cl, C)
    summary["rev_thresholds"] = np.array(REV_THRESHOLDS)
    summary["rev_threshold_names"] = np.array(REV_THRESHOLD_NAMES, dtype=object)
    summary["rev_cl_ratios"] = REV_CL_RATIOS
    save_summary(summary, str(out_dir / "eval_metrics.npz"))
    log.info(f"Saved eval_metrics.npz ({n_batches} batches) → {out_dir}")

    # --- Plots ---
    # Figure 2a: CRPS scorecard heatmap (rows=variables, cols=lead times)
    plot_crps_scorecard(
        crps_mean,
        variables,
        lead_hours,
        str(out_dir / "crps_scorecard.png"),
        title="CRPS scorecard (normalised per variable)",
    )
    # Figure 2a equivalent for RMSE
    plot_crps_scorecard(
        rmse_mean,
        variables,
        lead_hours,
        str(out_dir / "rmse_scorecard.png"),
        title="Ensemble-mean RMSE scorecard (normalised per variable)",
    )
    # Figure 2a equivalent for spread-skill ratio
    plot_crps_scorecard(
        ratio_mean,
        variables,
        lead_hours,
        str(out_dir / "spread_skill_scorecard.png"),
        title="Spread-skill ratio scorecard (normalised per variable)",
    )
    # Figure 2b-f: spread vs RMSE line plots for 5 key variables
    plot_spread_skill_lines(
        spread_mean,
        rmse_mean,
        variables,
        lead_hours,
        str(out_dir / "spread_skill_lines.png"),
    )
    # rank_acc shape: (M+1, C) → plot expects (C, M+1)
    plot_rank_histograms(
        rank_acc.T.astype(np.int64), variables, str(out_dir / "rank_histograms.png")
    )
    # Energy score: single line, fine to keep
    plot_metric_vs_lead(
        energy_mean[:, None],
        ["multivariate"],
        lead_hours,
        "energy score",
        "Energy score per lead (lower is better)",
        str(out_dir / "energy_score_vs_lead.png"),
    )
    plot_power_spectra(
        k_vec,
        power_ens_mean,
        power_tgt_mean,
        variables,
        lead_hours_all=lead_hours,
        out_path=str(out_dir / "power_spectra.png"),
    )
    plot_pooled_crps(
        pooled_avg_mean,
        pool_sizes,
        variables,
        lead_hours,
        str(out_dir / "avg_pooled_crps.png"),
        title="Average-pooled CRPS",
    )
    plot_pooled_crps(
        pooled_max_mean,
        pool_sizes,
        variables,
        lead_hours,
        str(out_dir / "max_pooled_crps.png"),
        title="Max-pooled CRPS",
    )
    # Figure 3c: derived variable CRPS — single line per derived var, readable
    for dname, vals in derived_acc.items():
        plot_metric_vs_lead(
            (vals / n_batches)[:, None],
            [dname],
            lead_hours,
            "CRPS",
            f"{dname} CRPS per lead (Figure 3c)",
            str(out_dir / f"derived_crps_{dname}.png"),
        )
    # Figure 2 g-h: REV curves per variable per threshold
    plot_rev_curves(
        rev_mean,
        variables,
        REV_CL_RATIOS,
        REV_THRESHOLD_NAMES,
        lead_hours,
        str(out_dir / "rev_curves.png"),
    )

    log.info(f"Eval complete. All outputs in {out_dir}")


@hydra.main(version_base=None, config_path="config", config_name="eval_fgn")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
