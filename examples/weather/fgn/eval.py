# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    derived_variable_crps,
    energy_score_per_lead,
    plot_crps_scorecard,
    plot_metric_vs_lead,
    plot_pooled_crps,
    plot_power_spectra,
    plot_rank_histograms,
    plot_spread_skill_lines,
    pooled_crps_per_lead,
    power_spectra_per_variable,
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


def _ar_rollout(
    model: torch.nn.Module,
    history: torch.Tensor,
    background: torch.Tensor,
    invariants: torch.Tensor | None,
    num_steps: int,
    latent_dim: int,
    num_members: int,
    device: torch.device,
    output_only: list[int],
) -> torch.Tensor:
    """Return ``(B, K, M, C, H, W)`` ensemble from an M-member AR rollout.

    Mirrors ``utils/trainer.py:_run_validation_metrics``: each member advances
    independently; predicted-only channels (e.g. tp06) are zeroed before being
    fed back as history on the next step.
    """
    B, T, C, H, W = history.shape
    per_member_hist = (
        history.unsqueeze(1).expand(B, num_members, T, C, H, W).contiguous()
    )
    preds_all: list[torch.Tensor] = []
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
        preds = torch.stack(members, dim=1)  # (B, M, C, H, W)
        preds_all.append(preds.cpu())          # offload to CPU — keep GPU free for next step
        if k < num_steps - 1:
            next_frame = preds
            if output_only:
                next_frame = next_frame.clone()
                for ci in output_only:
                    next_frame[:, :, ci].zero_()
            per_member_hist = torch.cat(
                [per_member_hist[:, :, 1:], next_frame.unsqueeze(2)], dim=2
            )
    return torch.stack(preds_all, dim=1)  # (B, K, M, C, H, W)


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
    crps_fn = e2s_crps(
        ensemble_dimension="ensemble",
        reduction_dimensions=["lat", "lon"],
        weights=area_w_2d,
        fair=True,
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
    # Shape (K, C) for per-lead per-variable metrics.
    crps_acc = np.zeros((K, C), dtype=np.float64)
    rmse_acc = np.zeros((K, C), dtype=np.float64)
    # Spread-skill: accumulate ensemble std and RMSE separately.
    spread_acc = np.zeros((K, C), dtype=np.float64)
    rank_acc = np.zeros((M + 1, C), dtype=np.float64)  # aggregated over leads+batches
    energy_acc = np.zeros(K, dtype=np.float64)
    power_ens_acc: np.ndarray | None = None
    power_tgt_acc: np.ndarray | None = None
    pooled_avg_acc: np.ndarray | None = None
    pooled_max_acc: np.ndarray | None = None
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

            # AR rollout across all models → (B, K, M, C, H, W)
            preds: list[torch.Tensor] = []
            for model, n_mem in zip(models, members_per_model, strict=True):
                if n_mem > 0:
                    preds.append(
                        _ar_rollout(
                            model, history, background, inv_b,
                            K, latent_dim, n_mem, device, output_only,
                        )
                    )
            ensemble = torch.cat(preds, dim=2)  # (B, K, M, C, H, W)

            # Per-lead metrics (earth2studio, area-weighted)
            xc, yc = _make_coords(B, M, variables, lats_np, lons_np)
            for k in range(K):
                ens_k = ensemble[:, k]   # (B, M, C, H, W)
                tgt_k = target[:, k]     # (B, C, H, W)

                crps_res, _ = crps_fn(ens_k, xc, tgt_k, yc)   # (B, C)
                crps_acc[k] += crps_res.mean(dim=0).cpu().numpy()

                rmse_res, _ = rmse_fn(ens_k, xc, tgt_k, yc)   # (B, C) RMSE
                rmse_acc[k] += rmse_res.mean(dim=0).cpu().numpy()

                # Spread: sqrt of mean ensemble variance over lat/lon (area-weighted)
                ens_var = ens_k.var(dim=1, unbiased=True)  # (B, C, H, W) var over members
                w = area_w_2d.to(ens_var.device)
                spread_kc = (ens_var * w).sum(dim=(-2, -1)) / w.sum()  # (B, C) mean var
                spread_acc[k] += spread_kc.sqrt().mean(dim=0).cpu().numpy()

                rh_res, _ = rh_fn(ens_k, xc, tgt_k, yc)
                # rh_res: (2, M+1, B, C) → [bin_centers, bin_counts]; sum over batch
                rank_acc += rh_res[1].sum(dim=-2).cpu().numpy()  # (M+1, C)

            # Full-rollout metrics (utils/metrics.py)
            ens_mean = ensemble.mean(dim=2)  # (B, K, C, H, W)
            k_vec, ens_spec, tgt_spec = power_spectra_per_variable(ens_mean, target)
            if power_ens_acc is None:
                power_ens_acc, power_tgt_acc = ens_spec, tgt_spec
            else:
                power_ens_acc += ens_spec
                power_tgt_acc += tgt_spec

            energy_acc += energy_score_per_lead(ensemble, target)

            p_avg = pooled_crps_per_lead(ensemble, target, pool_sizes, "avg")
            p_max = pooled_crps_per_lead(ensemble, target, pool_sizes, "max")
            if pooled_avg_acc is None:
                pooled_avg_acc, pooled_max_acc = p_avg, p_max
            else:
                pooled_avg_acc += p_avg
                pooled_max_acc += p_max

            for dname, vals in derived_variable_crps(ensemble, target, variables).items():
                derived_acc[dname] = derived_acc.get(dname, 0.0) + vals

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
    save_summary(summary, str(out_dir / "eval_metrics.npz"))
    log.info(f"Saved eval_metrics.npz ({n_batches} batches) → {out_dir}")

    # --- Plots ---
    # Figure 2a: CRPS scorecard heatmap (rows=variables, cols=lead times)
    plot_crps_scorecard(
        crps_mean, variables, lead_hours,
        str(out_dir / "crps_scorecard.png"),
        title="Fair CRPS scorecard (normalised per variable)",
    )
    # Figure 2a equivalent for RMSE
    plot_crps_scorecard(
        rmse_mean, variables, lead_hours,
        str(out_dir / "rmse_scorecard.png"),
        title="Ensemble-mean RMSE scorecard (normalised per variable)",
    )
    # Figure 2a equivalent for spread-skill ratio
    plot_crps_scorecard(
        ratio_mean, variables, lead_hours,
        str(out_dir / "spread_skill_scorecard.png"),
        title="Spread-skill ratio scorecard (normalised per variable)",
    )
    # Figure 2b-f: spread vs RMSE line plots for 5 key variables
    plot_spread_skill_lines(
        spread_mean, rmse_mean, variables, lead_hours,
        str(out_dir / "spread_skill_lines.png"),
    )
    # rank_acc shape: (M+1, C) → plot expects (C, M+1)
    plot_rank_histograms(rank_acc.T.astype(np.int64), variables,
                         str(out_dir / "rank_histograms.png"))
    # Energy score: single line, fine to keep
    plot_metric_vs_lead(
        energy_mean[:, None], ["multivariate"], lead_hours, "energy score",
        "Energy score per lead (lower is better)",
        str(out_dir / "energy_score_vs_lead.png"),
    )
    plot_power_spectra(
        k_vec, power_ens_mean, power_tgt_mean, variables,
        lead_hours_all=lead_hours,
        out_path=str(out_dir / "power_spectra.png"),
    )
    plot_pooled_crps(
        pooled_avg_mean, pool_sizes, variables, lead_hours,
        str(out_dir / "avg_pooled_crps.png"), title="Average-pooled CRPS",
    )
    plot_pooled_crps(
        pooled_max_mean, pool_sizes, variables, lead_hours,
        str(out_dir / "max_pooled_crps.png"), title="Max-pooled CRPS",
    )
    # Figure 3c: derived variable CRPS — single line per derived var, readable
    for dname, vals in derived_acc.items():
        plot_metric_vs_lead(
            (vals / n_batches)[:, None], [dname], lead_hours, "CRPS",
            f"{dname} CRPS per lead (Figure 3c)",
            str(out_dir / f"derived_crps_{dname}.png"),
        )

    log.info(f"Eval complete. All outputs in {out_dir}")


@hydra.main(version_base=None, config_path="config", config_name="eval_fgn")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
