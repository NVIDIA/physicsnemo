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

"""Tropical cyclone track evaluation — Figure 4 of arXiv:2506.10772.

Reproduces:
  - Figure 4a: ensemble-mean TC track position error vs lead time
  - Figure 4b: REV of track-probability predictions vs lead time

Uses earth2studio's ``TCTrackerWuDuan`` (pure-Python, no checkpoint download)
for detecting and tracking TCs in FGN ensemble forecasts, and earth2studio's
``IBTrACS`` data source for ground-truth named-storm positions.

Requirements:
    pip install cucim-cu12   # for TCTrackerWuDuan
    # earth2studio already provides IBTrACS

Usage::

    cd examples/weather/fgn
    python tc_eval.py --config-name fgn_arco \\
        eval.checkpoint_dir=/mnt/data/kashif/fgn_dev/run/0 \\
        eval.outdir=/mnt/data/kashif/fgn_dev/tc_eval \\
        eval.ensemble_size=8 \\
        eval.future_steps=20 \\
        dataset.val_start=2023-06-01 \\
        dataset.val_end=2023-12-01
"""
from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime, timedelta

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from datasets.arco import ARCO_LAT, ARCO_LON, ArcoFGNDataset
from utils.metrics import REV_CL_RATIOS, plot_tc_position_error, plot_tc_track_rev

log = logging.getLogger(__name__)

# TCTrackerWuDuan input variables (subset of FGN ERA5 channels)
TRACKER_VARS = ["u10m", "v10m", "msl", "u850", "v850"]

# Pairing: forecast path ↔ IBTrACS storm within this distance at lead-0
PAIR_DIST_KM = 100.0

# IBTrACS FILL value inside tracker path_buffer
_TRACK_FILL = -9999.0


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorised haversine distance (km) from each (lat1, lon1) to (lat2, lon2)."""
    R = 6371.0
    rlat1 = np.deg2rad(np.asarray(lat1, float))
    rlon1 = np.deg2rad(np.asarray(lon1, float))
    rlat2 = np.deg2rad(float(lat2))
    rlon2 = np.deg2rad(float(lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


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
    """Generator yielding ``(B, M, C, H, W)`` predictions one AR step at a time."""
    B, T, C, H, W = history.shape
    per_hist = history.unsqueeze(1).expand(B, num_members, T, C, H, W).contiguous()
    for k in range(num_steps):
        members = []
        for n in range(num_members):
            z = torch.randn(B, latent_dim, device=device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                pred = model(
                    history=per_hist[:, n],
                    latent=z,
                    background=background,
                    invariants=invariants,
                ).float()
            members.append(pred)
        preds_k = torch.stack(members, dim=1)  # (B, M, C, H, W)
        yield preds_k
        if k < num_steps - 1:
            nxt = preds_k
            if output_only:
                nxt = nxt.clone()
                for ci in output_only:
                    nxt[:, :, ci].zero_()
            per_hist = torch.cat([per_hist[:, :, 1:], nxt.unsqueeze(2)], dim=2)


def _resolve_checkpoints(cfg: DictConfig) -> list[str]:
    from eval import _resolve_checkpoints as _rc
    return _rc(cfg)


@hydra.main(config_path="config", config_name="fgn_arco", version_base=None)
def main(cfg: DictConfig) -> None:
    from physicsnemo.core.module import Module

    try:
        from earth2studio.models.dx import TCTrackerWuDuan
        from earth2studio.data import IBTrACS
    except ImportError as exc:
        log.error("Install earth2studio[cyclone] for TC tracking: %s", exc)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecfg = cfg

    # --- Dataset ---
    val_dataset = ArcoFGNDataset(cfg.dataset, train=False)
    variables = val_dataset.state_channels()
    step_hours = int(getattr(cfg.dataset, "step_hours", 6))
    K = int(ecfg.eval.future_steps)
    M = int(ecfg.eval.ensemble_size)
    latent_dim = int(ecfg.model.latent_dim)
    outdir = str(ecfg.eval.outdir)
    os.makedirs(outdir, exist_ok=True)

    if getattr(cfg.dataset, "spatial_stride", 1) != 1:
        log.error("TC eval requires spatial_stride=1 (tracker expects 721×1440 ERA5 grid)")
        return

    missing = [v for v in TRACKER_VARS if v not in variables]
    if missing:
        log.error("FGN checkpoint is missing TC tracker variables: %s", missing)
        return
    tc_idx = [variables.index(v) for v in TRACKER_VARS]

    # --- Model ---
    ckpt_paths = _resolve_checkpoints(cfg)
    log.info("Loading checkpoint: %s", ckpt_paths[0])
    model = Module.from_checkpoint(ckpt_paths[0]).to(device).eval()

    # --- Tracker (stateful; reset per IC) ---
    tracker = TCTrackerWuDuan().to(device)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    invariants = val_dataset.get_invariants()
    if invariants is not None:
        invariants = torch.from_numpy(invariants).to(device, dtype=torch.float32)

    all_init_times: list[datetime] = []
    # tracks per IC: (M, n_paths, K, 4) — 4=[lat, lon, msl, w10m]; FILL=-9999 for missing
    all_tracks: list[np.ndarray | None] = []

    log.info("Running TC tracker on %d ICs (K=%d steps, M=%d members)…", len(val_dataset), K, M)

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            history = batch["history"].to(device, dtype=torch.float32)
            background = batch["background"].to(device, dtype=torch.float32)
            init_time = datetime.fromisoformat(batch["init_time"][0])

            inv_b = invariants.unsqueeze(0) if invariants is not None else None
            tracker.reset_path_buffer()

            for preds_k in _ar_rollout_steps(
                model, history, background, inv_b,
                K, latent_dim, M, device,
                val_dataset.output_only_channels(),
            ):
                # Denormalize full state so tracker sees physical units (Pa, m/s)
                phys = val_dataset.denormalize_state(preds_k[0])  # (M, C, H, W)
                tc_in = phys[:, tc_idx, :, :]                     # (M, 5, H, W)
                step_coords = OrderedDict([
                    ("batch", np.arange(M)),
                    ("variable", np.array(TRACKER_VARS)),
                    ("lat", ARCO_LAT.astype(np.float64)),
                    ("lon", ARCO_LON.astype(np.float64)),
                ])
                tracker(tc_in, step_coords)

            buf = tracker.path_buffer  # (M, n_paths, K, 4) after K calls
            all_init_times.append(init_time)
            all_tracks.append(buf.cpu().numpy() if buf.numel() > 0 else None)

            if (bi + 1) % 50 == 0:
                log.info("  %d/%d ICs", bi + 1, len(val_dataset))

    # --- IBTrACS ground truth for all forecast step times ---
    all_step_times = sorted({
        t + timedelta(hours=k * step_hours)
        for t in all_init_times
        for k in range(1, K + 1)
    })
    log.info("Querying IBTrACS for %d timestamps…", len(all_step_times))
    ibtracs = IBTrACS(region="ALL", time_tolerance=timedelta(hours=step_hours // 2))
    try:
        df = ibtracs(all_step_times, ["tclat", "tclon"])
    except Exception as exc:
        log.error("IBTrACS fetch failed: %s", exc)
        return

    if df.empty:
        log.warning("IBTrACS returned no data for the requested period.")
        return

    # Normalise time column to timezone-naive datetime for comparison
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    # Named storms only (paper §4.3)
    if "storm_name" in df.columns:
        df = df[~df["storm_name"].str.upper().str.strip().isin(["NOT_NAMED", "UNNAMED", ""])]

    # Storm identifier: prefer storm_id, fall back to storm_name
    id_col = "storm_id" if "storm_id" in df.columns else "storm_name"

    # --- Pair tracks and accumulate metrics per lead ---
    pos_err_acc = np.zeros(K, dtype=np.float64)
    n_pairs = np.zeros(K, dtype=np.int64)

    n_cl = len(REV_CL_RATIOS)
    rev_num = np.zeros((K, n_cl), dtype=np.float64)  # numerator V - V_clim
    rev_n = np.zeros(K, dtype=np.int64)
    clim_active = 0   # total observations (obs=1 events)
    clim_total = 0    # total (obs=0 + obs=1) events

    for ic_idx, (init_time, tracks) in enumerate(zip(all_init_times, all_tracks)):
        if tracks is None:
            continue
        M_act, n_paths, K_act, _ = tracks.shape
        t_lead1 = pd.Timestamp(init_time + timedelta(hours=step_hours))

        # IBTrACS named storms active at lead-0 (first forecast step)
        storms_t0 = df[df["time"] == t_lead1]
        if storms_t0.empty:
            continue

        for _, srow in storms_t0.iterrows():
            s_id = srow[id_col]
            s_lat0 = float(srow["tclat"])
            s_lon0 = float(srow["tclon"]) % 360.0

            # Check if any forecast path start (step 0) is within PAIR_DIST_KM
            lats0 = tracks[:, :, 0, 0].ravel()   # (M*n_paths,)
            lons0 = (tracks[:, :, 0, 1] % 360.0).ravel()
            valid0 = lats0 != _TRACK_FILL
            if not valid0.any():
                continue
            d0 = np.full(len(lats0), np.inf)
            d0[valid0] = _haversine_km(lats0[valid0], lons0[valid0], s_lat0, s_lon0)
            if d0.min() > PAIR_DIST_KM:
                continue  # no member caught this storm

            # Best-path index per member (nearest to storm at step 0)
            d0_by_member = d0.reshape(M_act, n_paths)  # (M, n_paths)
            best_path_per_member = np.argmin(d0_by_member, axis=1)  # (M,)

            # Iterate over leads
            for k in range(K_act):
                t_lead_k = pd.Timestamp(init_time + timedelta(hours=(k + 1) * step_hours))
                truth_rows = df[
                    (df[id_col].astype(str) == str(s_id)) & (df["time"] == t_lead_k)
                ]
                obs_active = not truth_rows.empty

                clim_total += 1
                if obs_active:
                    clim_active += 1

                # Member positions at step k along their best-matched path
                member_lats, member_lons = [], []
                for m in range(M_act):
                    p = best_path_per_member[m]
                    lat_mk = tracks[m, p, k, 0]
                    if lat_mk == _TRACK_FILL:
                        continue
                    member_lats.append(lat_mk)
                    member_lons.append(tracks[m, p, k, 1] % 360.0)

                frac_active = len(member_lats) / M_act

                if obs_active and member_lats and frac_active >= 0.5:
                    # Position error of ensemble mean
                    truth_lat = float(truth_rows["tclat"].iloc[0])
                    truth_lon = float(truth_rows["tclon"].iloc[0]) % 360.0
                    mean_lat = float(np.mean(member_lats))
                    mean_lon = float(np.mean(member_lons))
                    pos_err_acc[k] += _haversine_km(
                        np.array([mean_lat]), np.array([mean_lon]), truth_lat, truth_lon
                    )[0]
                    n_pairs[k] += 1

                # Track-probability REV (Richardson 2000)
                # action = 1 when forecast prob > C/L ratio
                obs_val = 1.0 if obs_active else 0.0
                for ci, cl in enumerate(REV_CL_RATIOS):
                    action = float(frac_active > cl)
                    rev_num[k, ci] += action * obs_val - action * cl
                rev_n[k] += 1

    # --- Compute REV ---
    clim_rate = clim_active / max(clim_total, 1)
    pos_err_mean = np.where(n_pairs > 0, pos_err_acc / np.maximum(n_pairs, 1), np.nan)
    lead_hours = np.arange(1, K + 1) * step_hours

    rev_vals = np.full((K, n_cl), np.nan)
    for k in range(K):
        if rev_n[k] == 0:
            continue
        v_mean = rev_num[k] / rev_n[k]
        v_clim = np.array([max(0.0, clim_rate - cl) for cl in REV_CL_RATIOS])
        v_perf = np.array([clim_rate * max(0.0, 1.0 - cl) for cl in REV_CL_RATIOS])
        denom = v_perf - v_clim
        ok = np.abs(denom) > 1e-8
        rev_vals[k, ok] = (v_mean[ok] - v_clim[ok]) / denom[ok]

    log.info("Paired storms (lead=1): %d", n_pairs[0])
    log.info("Mean position error at lead 1 (%.0fh): %.1f km", lead_hours[0], pos_err_mean[0] if not np.isnan(pos_err_mean[0]) else -1)

    # --- Save and plot ---
    np.savez(
        os.path.join(outdir, "tc_metrics.npz"),
        pos_err=pos_err_mean,
        n_pairs=n_pairs,
        rev=rev_vals,
        lead_hours=lead_hours,
        cl_ratios=np.array(REV_CL_RATIOS),
    )
    plot_tc_position_error(pos_err_mean, lead_hours, os.path.join(outdir, "tc_position_error.png"))
    plot_tc_track_rev(rev_vals, lead_hours, list(REV_CL_RATIOS), os.path.join(outdir, "tc_track_rev.png"))
    log.info("TC eval done → %s", outdir)


if __name__ == "__main__":
    main()
