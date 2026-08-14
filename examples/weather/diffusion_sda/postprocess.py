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

import argparse
import os

import cartopy
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import numpy as np
import torch


CHANNEL_NAMES = [
    "u10m",
    "v10m",
    "u80m",
    "v80m",
    "t2m",
    "d2m",
    "q2m",
    "sp",
    "fg10m",
    "tcc",
    "sde",
    "snowc",
    "refc",
    "rsds",
    "tp",
    "aerot",
]

# HRRR Lambert Conformal projection
PROJECTION = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="HRRR diffusion SDA postprocessing")
    p.add_argument(
        "--no-dps-path", type=str, required=True, help="Path to no-DPS samples .pt"
    )
    p.add_argument("--dps-path", type=str, default=None, help="Path to DPS samples .pt")
    p.add_argument("--output-dir", type=str, default="./outputs/plots")
    p.add_argument(
        "--channel",
        type=int,
        default=4,
        help="Channel index to plot (0=u10m, 1=v10m, 4=t2m, ...)",
    )
    p.add_argument("--sample-idx", type=int, default=0)
    return p.parse_args()


def main():
    """Report per-channel MAE and render ground-truth vs prediction plots."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    no_dps = torch.load(args.no_dps_path, map_location="cpu")
    x0_no_dps = no_dps["x0"]  # (B, 16, H, W)
    gt = no_dps["ground_truth"]  # (B, 16, H, W)
    mask = no_dps["mask"]  # (B, 16, H, W) bool
    cond_spatial = no_dps["cond_spatial"]  # (B, 3, H, W)

    s = args.sample_idx
    ch = args.channel
    ch_name = CHANNEL_NAMES[ch]

    # Recover lat/lon from the normalised spatial condition channels
    lat = cond_spatial[s, 1].numpy() * 90.0  # (H, W), degrees
    lon = cond_spatial[s, 2].numpy() * 360.0  # (H, W), degrees

    # MAE per channel, averaged over spatial dims and samples
    mae_no_dps = (x0_no_dps - gt).abs().mean(dim=(-2, -1))  # (B, C)
    print(f"MAE (no DPS) — {x0_no_dps.shape[0]} samples:")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name:8s}: {mae_no_dps[:, i].mean().item():.4f}")

    has_dps = args.dps_path is not None
    if has_dps:
        dps = torch.load(args.dps_path, map_location="cpu")
        x0_dps = dps["x0"]
        mae_dps = (x0_dps - gt).abs().mean(dim=(-2, -1))
        print(f"\nMAE (DPS) — {x0_dps.shape[0]} samples:")
        for i, name in enumerate(CHANNEL_NAMES):
            print(f"  {name:8s}: {mae_dps[:, i].mean().item():.4f}")

    # Comparison plot: ground truth, no-DPS prediction, (optional) DPS prediction
    nrows = 3 if has_dps else 2
    fig, axes = plt.subplots(
        nrows,
        1,
        subplot_kw={"projection": PROJECTION},
        figsize=(14, 5 * nrows),
    )

    gt_field = gt[s, ch].numpy()
    vmin = float(np.percentile(gt_field, 2))
    vmax = float(np.percentile(gt_field, 98))

    rows = [
        (gt_field, "Ground truth"),
        (x0_no_dps[s, ch].numpy(), "Prediction (no DPS)"),
    ]
    if has_dps:
        rows.append((x0_dps[s, ch].numpy(), "Prediction (DPS)"))

    for ax, (field, title) in zip(axes, rows):
        im = ax.pcolormesh(
            lon,
            lat,
            field,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(
            cartopy.feature.STATES.with_scale("50m"),
            linewidth=0.5,
            edgecolor="black",
            zorder=2,
        )
        ax.set_title(f"{title}: {ch_name}", fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)

    # Overlay observation locations on the last row
    obs_mask_2d = mask[s, ch].numpy()
    if obs_mask_2d.any():
        obs_rows, obs_cols = np.where(obs_mask_2d)
        axes[-1].scatter(
            lon[obs_rows, obs_cols],
            lat[obs_rows, obs_cols],
            s=4,
            marker="x",
            c="black",
            transform=ccrs.PlateCarree(),
            zorder=3,
            label="Observations",
        )

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f"comparison_{ch_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {out_path}")
    plt.close()

    # Difference plot (DPS minus no-DPS) when both are available
    if has_dps:
        diff = x0_dps[s, ch].numpy() - x0_no_dps[s, ch].numpy()
        dmax = float(max(abs(diff.min()), abs(diff.max())))

        fig2, ax2 = plt.subplots(
            1,
            1,
            subplot_kw={"projection": PROJECTION},
            figsize=(14, 5),
        )
        im2 = ax2.pcolormesh(
            lon,
            lat,
            diff,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-dmax,
            vmax=dmax,
        )
        ax2.add_feature(
            cartopy.feature.STATES.with_scale("50m"),
            linewidth=0.5,
            edgecolor="black",
            zorder=2,
        )
        if obs_mask_2d.any():
            ax2.scatter(
                lon[obs_rows, obs_cols],
                lat[obs_rows, obs_cols],
                s=4,
                marker="x",
                c="black",
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
        ax2.set_title(f"Difference (DPS minus no DPS): {ch_name}", fontsize=12)
        plt.colorbar(im2, ax=ax2, shrink=0.5, pad=0.02)
        plt.tight_layout()
        diff_path = os.path.join(args.output_dir, f"difference_{ch_name}.png")
        plt.savefig(diff_path, dpi=150, bbox_inches="tight")
        print(f"Saved difference plot to {diff_path}")
        plt.close()


if __name__ == "__main__":
    main()
