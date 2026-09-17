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

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; chosen before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

# The universal per-particle quantity present in every rollout, in addition to
# the named scalar features. Plotted on a log color scale because inter-event
# delays typically span many orders of magnitude.
_DELAY_KEY = "delay"
_DELAY_LABEL = "inter-event delay"


def _value_label(name: str) -> str:
    """Plot label for a colorable quantity (``delay`` or a named feature)."""
    return _DELAY_LABEL if name == _DELAY_KEY else name


def _is_log_color(name: str) -> bool:
    """Whether quantity ``name`` is colored on a log scale by default."""
    return name == _DELAY_KEY


def _gt_values(blob: dict, name: str, feature_names: list[str]) -> np.ndarray:
    """Ground-truth per-event values for ``name``: ``delay`` or a named feature.

    Returns a ``(n,)`` array (the feature column for a named scalar, or the
    delay vector).
    """
    if name == _DELAY_KEY:
        return blob["delay_gt"].numpy()  # (n,)
    j = feature_names.index(name)
    return blob["scalar_features_gt"].numpy()[:, j]  # (n,)


def _pred_values(blob: dict, name: str, feature_names: list[str]) -> np.ndarray:
    """Predicted per-event values for ``name`` across the ensemble.

    Returns a ``(B, K)`` NaN-padded array (the feature column for a named
    scalar, or the delay matrix).
    """
    if name == _DELAY_KEY:
        return blob["delay_pred"].numpy()  # (B, K)
    j = feature_names.index(name)
    return blob["scalar_features_pred"].numpy()[:, :, j]  # (B, K)


def _draw_box_frame(
    ax,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """Draw the 12 bounding-box edges as thin translucent lines."""
    corners = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i][0], corners[j][0]],
            [corners[i][1], corners[j][1]],
            [corners[i][2], corners[j][2]],
            color="gray",
            linewidth=0.6,
            alpha=0.5,
        )


def _configure_axes(
    ax,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    elev: float,
    azim: float,
) -> None:
    """One-off cosmetic configuration; re-applied per frame after ``ax.cla()``."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.grid(False)
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis.pane.fill = False
        pane_axis.pane.set_edgecolor("none")
    ax.set_xticks(np.round(np.linspace(x_min, x_max, 5), 12))
    ax.set_yticks(np.round(np.linspace(y_min, y_max, 5), 12))
    ax.set_zticks(np.round(np.linspace(z_min, z_max, 5), 12))
    _draw_box_frame(ax, x_min, x_max, y_min, y_max, z_min, z_max)


def _gt_density_curve(
    times: np.ndarray, values: np.ndarray, time_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of all particle ``values[k]`` with ``times[k] <= time_grid[i]``."""
    mean = np.full_like(time_grid, np.nan, dtype=np.float64)
    std = np.full_like(time_grid, np.nan, dtype=np.float64)
    for i, t in enumerate(time_grid):
        present = values[times <= t]
        if present.size > 0:
            mean[i] = float(present.mean())
            std[i] = float(present.std())
    return mean, std


def _pred_pool_curve(
    times_pred: np.ndarray,  # (B, K) NaN-padded
    values_pred: np.ndarray,  # (B, K) NaN-padded
    time_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool every ensemble member's particles by birth time and summarize.

    For each ``time_grid[i]``, gather every value across the ensemble
    whose birth time is at or before ``time_grid[i]`` (NaN-safe), then
    return their pooled mean and standard deviation.
    """
    flat_times = times_pred.ravel()
    flat_values = values_pred.ravel()
    finite = np.isfinite(flat_times) & np.isfinite(flat_values)
    flat_times = flat_times[finite]
    flat_values = flat_values[finite]
    mean = np.full_like(time_grid, np.nan, dtype=np.float64)
    std = np.full_like(time_grid, np.nan, dtype=np.float64)
    for i, t in enumerate(time_grid):
        present = flat_values[flat_times <= t]
        if present.size > 0:
            mean[i] = float(present.mean())
            std[i] = float(present.std())
    return mean, std


def _plot_band(
    ax,
    time_grid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
) -> None:
    """Plot a mean line with a ``±std`` shaded fill, robust to NaN gaps."""
    valid = np.isfinite(mean)
    if not valid.any():
        return
    ax.plot(time_grid[valid], mean[valid], color=color, lw=1.5, label=label)
    ax.fill_between(
        time_grid[valid],
        mean[valid] - std[valid],
        mean[valid] + std[valid],
        color=color,
        alpha=0.25,
    )


def animate_prediction(
    file: str | Path,
    variable: str,
    output: str = "prediction.gif",
    fps: int = 5,
    colormap: str = "plasma",
    num_frames: int = 80,
    dpi: int = 100,
    elev: float = 30.0,
    azim: float = -60.0,
    clip_pct: float = 2.0,
    ensemble_index: int = 0,
) -> None:
    """Animate a generated rollout against its ground truth, with summary 1D panels.

    The top row puts two 3D scatter plots side by side: ground truth on
    the left, one selected ensemble member of the model's prediction on
    the right. Both are colored by the chosen ``variable``, which is one
    of the rollout's named scalar particle features or ``"delay"``
    (colored on a log scale by default since delays span many decades).
    The next row shows the cumulative particle-count curve N(t): a single
    line for ground truth and one line per ensemble member for the
    prediction (with the mean drawn in bold). Below that, one row per
    named scalar feature plus one for the inter-event delay shows that
    quantity's distribution over time as mean ± std shaded bands: ground
    truth aggregates particles from the single GT trajectory, prediction
    aggregates particles pooled across the entire ensemble. A vertical
    cursor on every 1D panel tracks the animation frame.

    Parameters
    ----------
    file : path
        ``.pth`` rollout file produced by ``generate.py``.
    variable : str
        Which quantity colors the top-row scatter plots. One of the
        rollout's named scalar features or ``"delay"``.
    output : path
        Output file; ``.gif`` uses Pillow, ``.mp4`` uses ffmpeg.
    fps : int
        Animation frame rate.
    colormap : str
        Matplotlib colormap name.
    num_frames : int
        Number of animation frames; uniformly spaced over the rollout's
        simulated-time window.
    dpi : int
        Figure DPI.
    elev, azim : float
        3D view angles in degrees.
    clip_pct : float
        Percentile clip on the color range (``2.0`` -> use ``[P2, P98]``).
    ensemble_index : int
        Index of the ensemble member used in the prediction scatter plot.
    """
    file = Path(file)
    print(f"Loading rollout: {file}")
    blob = torch.load(file, map_location="cpu", weights_only=False)

    feature_names = list(blob["particle_feature_names"])
    colorable = feature_names + [_DELAY_KEY]
    if variable not in colorable:
        raise ValueError(f"variable must be one of {colorable}, got {variable!r}")

    geometry = str(blob["geometry"])
    sim_id = int(blob["sim_id"])
    mesh_positions = blob["mesh_positions"].numpy()  # (N, 3)

    # Ground truth, single trajectory (raw units).
    times_gt = blob["times_gt"].numpy()  # (n,)
    positions_gt = blob["positions_gt"].numpy()  # (n, 3)
    # Predicted ensemble, (B, K) NaN-padded with explicit per-member counts.
    times_pred = blob["times_pred"].numpy()  # (B, K)
    positions_pred = blob["positions_pred"].numpy()  # (B, K, 3)
    n_pred = blob["n_pred"].numpy()  # (B,)
    B = times_pred.shape[0]
    if not (0 <= ensemble_index < B):
        raise ValueError(f"ensemble_index={ensemble_index} out of range [0, {B}).")

    # Scatter color values for the chosen variable.
    values_gt = _gt_values(blob, variable, feature_names)  # (n,)
    values_pred = _pred_values(blob, variable, feature_names)  # (B, K)
    color_log = _is_log_color(variable)
    color_label = _value_label(variable)

    # Spatial box from the mesh bounding box so the cube stays put across frames.
    x_min, x_max = float(mesh_positions[:, 0].min()), float(mesh_positions[:, 0].max())
    y_min, y_max = float(mesh_positions[:, 1].min()), float(mesh_positions[:, 1].max())
    z_min, z_max = float(mesh_positions[:, 2].min()), float(mesh_positions[:, 2].max())

    # Color range over the union of GT and pred values (finite only).
    pred_finite = values_pred[np.isfinite(values_pred)]
    pooled = (
        np.concatenate([v for v in (values_gt, pred_finite) if v.size > 0])
        if max(values_gt.size, pred_finite.size) > 0
        else np.array([0.0, 1.0])
    )
    if color_log:
        positive = pooled[pooled > 0]
        pooled = positive if positive.size > 0 else np.array([1e-12, 1.0])
    value_range = (
        float(np.percentile(pooled, clip_pct)),
        float(np.percentile(pooled, 100.0 - clip_pct)),
    )

    # Uniform time axis from 0 to the latest end time across GT / pred.
    t_end = max(
        float(times_gt[-1]) if times_gt.size else 0.0,
        float(np.nanmax(times_pred)) if np.isfinite(times_pred).any() else 0.0,
    )
    if t_end <= 0:
        raise RuntimeError("Both GT and prediction trajectories are empty.")
    frame_times = np.linspace(0.0, t_end, num_frames)

    # Static 1D-panel curves.
    time_grid = np.linspace(0.0, t_end, max(num_frames * 2, 200))
    n_gt_curve = np.searchsorted(times_gt, time_grid, side="right")
    n_pred_per_member = np.stack(
        [
            np.searchsorted(times_pred[b, : int(n_pred[b])], time_grid, side="right")
            for b in range(B)
        ]
    )  # (B, len(time_grid))
    n_pred_mean = n_pred_per_member.mean(axis=0)

    # One band panel per named scalar feature plus one for the delay.
    band_quantities = feature_names + [_DELAY_KEY]
    band_curves = {}
    for name in band_quantities:
        gt_vals = _gt_values(blob, name, feature_names)
        pred_vals = _pred_values(blob, name, feature_names)
        band_curves[name] = (
            _gt_density_curve(times_gt, gt_vals, time_grid),
            _pred_pool_curve(times_pred, pred_vals, time_grid),
        )

    n_band = len(band_quantities)
    fig = plt.figure(figsize=(14, 9 + 2 * n_band), dpi=dpi)
    height_ratios = [4, 1] + [1] * n_band
    gs = fig.add_gridspec(
        2 + n_band, 2, height_ratios=height_ratios, hspace=0.32, wspace=0.05
    )
    ax_gt = fig.add_subplot(gs[0, 0], projection="3d")
    ax_pred = fig.add_subplot(gs[0, 1], projection="3d")
    ax_n = fig.add_subplot(gs[1, :])
    ax_bands = [fig.add_subplot(gs[2 + i, :]) for i in range(n_band)]

    fig.suptitle(
        f"Particle rollout, geometry {geometry}, sim {sim_id} "
        f"| ensemble size {B}, scatter shows member {ensemble_index} "
        f"| color: {color_label}",
        fontsize=12,
    )

    vmin, vmax = value_range
    norm = (
        LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
        if color_log and vmax > 0
        else Normalize(vmin=vmin, vmax=vmax)
    )

    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array(np.array([vmin, vmax], dtype=float))
    cbar = fig.colorbar(sm, ax=[ax_gt, ax_pred], shrink=0.6, pad=0.02, aspect=25)
    cbar.set_label(color_label, fontsize=9)

    # Cumulative particle count over time, static traces.
    color_gt = "tab:blue"
    color_pred = "tab:orange"
    ax_n.plot(
        time_grid,
        n_gt_curve,
        color=color_gt,
        lw=2.0,
        label=f"GT (final n={times_gt.size})",
    )
    for b in range(B):
        ax_n.plot(time_grid, n_pred_per_member[b], color=color_pred, lw=0.6, alpha=0.4)
    ax_n.plot(
        time_grid,
        n_pred_mean,
        color=color_pred,
        lw=2.0,
        label=f"Pred mean (ensemble of {B})",
    )
    ax_n.set_xlabel("time")
    ax_n.set_ylabel("cumulative particle count")
    ax_n.set_xlim(0.0, t_end)
    ax_n.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_n.legend(loc="upper left", fontsize=9)
    cursor_n = ax_n.axvline(0.0, color="k", linestyle="--", lw=1.0, alpha=0.6)

    # One band panel per quantity: distribution vs time (mean ± std shaded).
    cursors = [cursor_n]
    for ax_band, name in zip(ax_bands, band_quantities):
        (gt_mean, gt_std), (pred_mean, pred_std) = band_curves[name]
        _plot_band(ax_band, time_grid, gt_mean, gt_std, color_gt, "GT mean ± std")
        _plot_band(
            ax_band,
            time_grid,
            pred_mean,
            pred_std,
            color_pred,
            f"Pred mean ± std (pooled over ensemble of {B})",
        )
        ax_band.set_xlabel("time")
        ax_band.set_ylabel(_value_label(name))
        ax_band.set_xlim(0.0, t_end)
        ax_band.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_band.legend(loc="upper left", fontsize=9)
        cursors.append(
            ax_band.axvline(0.0, color="k", linestyle="--", lw=1.0, alpha=0.6)
        )

    print(
        f"{color_label} color range: [{value_range[0]:.3e}, {value_range[1]:.3e}]\n"
        f"Frames: {num_frames} | t_end = {t_end:.3e} | geometry: {geometry} "
        f"| sim_id: {sim_id} | ensemble size: {B}"
    )

    # Scatter trajectory for the chosen ensemble member.
    n_member = int(n_pred[ensemble_index])
    member_times = times_pred[ensemble_index, :n_member]  # (n_member,)
    member_positions = positions_pred[ensemble_index, :n_member]  # (n_member, 3)
    member_values = values_pred[ensemble_index, :n_member]  # (n_member,)

    def _update(frame_idx: int) -> None:
        t_now = float(frame_times[frame_idx])
        mask_gt = times_gt <= t_now
        mask_pred = member_times <= t_now

        for ax in (ax_gt, ax_pred):
            ax.cla()
        _configure_axes(ax_gt, x_min, x_max, y_min, y_max, z_min, z_max, elev, azim)
        _configure_axes(ax_pred, x_min, x_max, y_min, y_max, z_min, z_max, elev, azim)

        n_gt_now = int(mask_gt.sum())
        n_pred_now = int(mask_pred.sum())
        ax_gt.set_title(
            f"Ground truth, t={t_now:.2e} (n_particles={n_gt_now})", fontsize=10
        )
        ax_pred.set_title(
            f"Prediction (member {ensemble_index}), t={t_now:.2e} "
            f"(n_particles={n_pred_now})",
            fontsize=10,
        )

        if n_gt_now > 0:
            ax_gt.scatter(
                positions_gt[mask_gt, 0],
                positions_gt[mask_gt, 1],
                positions_gt[mask_gt, 2],
                c=values_gt[mask_gt],
                cmap=colormap,
                norm=norm,
                s=30,
                depthshade=True,
            )
        if n_pred_now > 0:
            ax_pred.scatter(
                member_positions[mask_pred, 0],
                member_positions[mask_pred, 1],
                member_positions[mask_pred, 2],
                c=member_values[mask_pred],
                cmap=colormap,
                norm=norm,
                s=30,
                depthshade=True,
            )

        for cursor in cursors:
            cursor.set_xdata([t_now, t_now])

    anim = FuncAnimation(
        fig, _update, frames=num_frames, interval=1000 // fps, repeat=False
    )
    output_path = Path(output)
    writer = "pillow" if output_path.suffix.lower() == ".gif" else "ffmpeg"
    print(f"Saving animation to {output_path} ...")
    anim.save(str(output_path), writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    print("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate a particles rollout produced by generate.py: GT vs "
        "predicted ensemble (3D scatter on top, cumulative count and per-feature "
        "bands in the panels below)."
    )
    p.add_argument(
        "--file",
        required=True,
        help="Path to the .pth file produced by generate.py",
    )
    p.add_argument(
        "--variable",
        required=True,
        help="Which quantity colors the top-row scatter plots: a named scalar "
        "particle feature or 'delay'.",
    )
    p.add_argument(
        "--output",
        default="prediction.gif",
        help="Output file (.gif or .mp4) [default: prediction.gif]",
    )
    p.add_argument("--fps", type=int, default=5, help="Frames per second [default: 5]")
    p.add_argument(
        "--colormap", default="plasma", help="Matplotlib colormap [default: plasma]"
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=80,
        help="Number of animation frames [default: 80]",
    )
    p.add_argument("--dpi", type=int, default=100, help="Figure DPI [default: 100]")
    p.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="3D view elevation angle [default: 30]",
    )
    p.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="3D view azimuth angle [default: -60]",
    )
    p.add_argument(
        "--clip-pct",
        type=float,
        default=2.0,
        help="Percentile clip for the color range [default: 2.0]",
    )
    p.add_argument(
        "--ensemble-index",
        type=int,
        default=0,
        help="Index of the ensemble member shown in the scatter [default: 0]",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    animate_prediction(
        file=args.file,
        variable=args.variable,
        output=args.output,
        fps=args.fps,
        colormap=args.colormap,
        num_frames=args.num_frames,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        clip_pct=args.clip_pct,
        ensemble_index=args.ensemble_index,
    )
