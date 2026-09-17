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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; chosen before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FixedFormatter, FixedLocator

# Make the sibling `dataset/` package importable when this script is run
# directly from the recipe root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import ParticlesDataset  # noqa: E402

# The universal per-particle quantity present in every dataset, in addition to
# the named scalar features. Colored on a log scale by default because
# inter-event delays typically span many orders of magnitude.
_DELAY_KEY = "delay"
_DELAY_LABEL = "inter-event delay"


def _draw_box_frame(
    ax,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """Draw the 12 bounding-box edges as thin translucent lines.

    Re-issued per frame after ``ax.cla()`` wipes the axes.
    """
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
        (3, 0),  # bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical edges
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


def render_frame(
    axes,
    positions: np.ndarray,
    panel_values: list[np.ndarray],
    panel_ranges: list[tuple[float, float]],
    panel_labels: list[str],
    panel_log: list[bool],
    z_range: tuple[float, float],
    xy_range: tuple[float, float, float, float],
    colormap: str = "plasma",
    title: str = "",
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    """Clear and redraw one 3D scatter axis per colored quantity for a frame.

    Parameters
    ----------
    axes : list of Axes3D
        One 3D axis per colored quantity (named scalar features then the
        inter-event delay), in the same order as ``panel_values``.
    positions : (n, 3) array of real particle positions in mesh units.
    panel_values : list of (n,) arrays, the per-panel color values.
    panel_ranges : list of fixed (vmin, vmax) for the per-panel colorbars.
    panel_labels : list of colorbar/title labels, one per panel.
    panel_log : list of flags; use a log-norm for that panel's colorbar.
    z_range : (z_min, z_max) spatial extent of the mesh along z.
    xy_range : (x_min, x_max, y_min, y_max) from the mesh extent. Bounds the
        axes so they stay fixed across frames as particles appear and move.
    """
    x_min, x_max, y_min, y_max = xy_range
    z_min, z_max = z_range
    for ax, values, (vmin, vmax), cbar_label, use_log in zip(
        axes, panel_values, panel_ranges, panel_labels, panel_log
    ):
        ax.cla()
        ax.view_init(elev=elev, azim=azim)
        norm = (
            LogNorm(vmin=max(vmin, 1e-30), vmax=vmax)
            if use_log and vmax > 0
            else Normalize(vmin=vmin, vmax=vmax)
        )
        if positions.shape[0] > 0:
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=values,
                cmap=colormap,
                norm=norm,
                s=30,
                depthshade=True,
            )

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

        # Colorbar created once on the first frame; frozen across the rollout.
        if not hasattr(ax, "_panel_cbar"):
            sm = ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array(np.array([vmin, vmax], dtype=float))
            cbar = ax.get_figure().colorbar(sm, ax=ax, shrink=0.5, pad=0.12, aspect=20)
            cbar.set_label(cbar_label, fontsize=9)
            if use_log:
                # Pick a handful of decade-aligned ticks for legibility.
                lo = np.floor(np.log10(max(vmin, 1e-30)))
                hi = np.ceil(np.log10(max(vmax, 1e-30)))
                ticks = 10.0 ** np.arange(lo, hi + 1)
                ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
                if ticks.size < 2:
                    ticks = np.array([vmin, vmax])
                labels = [f"{t:.1e}" for t in ticks]
            else:
                span = vmax - vmin
                decimals = (
                    max(2, int(np.ceil(-np.log10(span + 1e-30))) + 1) if span > 0 else 2
                )
                ticks = np.linspace(vmin, vmax, 6)
                labels = [f"{v:.{decimals}f}" for v in ticks]
            cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
            cbar.ax.yaxis.set_major_formatter(FixedFormatter(labels))
            cbar.ax.yaxis.offsetText.set_visible(False)
            ax._panel_cbar = True

        if title:
            ax.set_title(f"{cbar_label}  {title}", fontsize=10)


def animate_simulation(
    data_dir: str | Path,
    geometry: str,
    sim_id: int,
    particle_feature_names: list[str],
    mesh_feature_names: list[str],
    output: str = "animation.gif",
    fps: int = 5,
    colormap: str = "plasma",
    ts_start: int = 0,
    ts_end: int | None = None,
    dpi: int = 100,
    clip_pct: float = 2.0,
    elev: float = 30.0,
    azim: float = -60.0,
    log_delay: bool = True,
) -> None:
    """Create and save a 3D scatter animation for one simulation.

    The top row shows one 3D scatter panel per colored quantity: one per
    named scalar particle feature plus one for the inter-event delay
    (colored on a log scale by default since delays span many orders of
    magnitude). The two bottom panels show the cumulative particle count
    over time and the first scalar feature's mean ± std over the particle
    population present at each timestep.

    Parameters
    ----------
    data_dir : path to the data directory (containing ``samples/`` and
        ``maps/``).
    geometry : opaque geometry-group name (a subdirectory under
        ``samples/``).
    sim_id : simulation index.
    particle_feature_names : names of the scalar particle features, in
        column order. Each one gets a colored scatter panel.
    mesh_feature_names : names of the scalar mesh fields (forwarded to the
        dataset; not plotted here).
    output : output path (.gif uses Pillow, .mp4 uses ffmpeg).
    fps : animation frame rate.
    colormap : matplotlib colormap name.
    ts_start, ts_end : timestep window (ts_end exclusive; None = all).
    dpi : figure resolution.
    clip_pct : percentile clip for the color ranges (e.g. 2.0 -> use
        [P2, P98] instead of [min, max]). Higher values saturate the
        extremes more.
    elev, azim : 3D view angles in degrees.
    log_delay : use a log-norm for the delay panel's colorbar.
    """
    # n_steps=1 -> one snapshot per sample, one sample per ts_id.
    dataset = ParticlesDataset(
        data_dir,
        particle_feature_names=particle_feature_names,
        mesh_feature_names=mesh_feature_names,
        n_steps=1,
    )
    indices = dataset.get_sim_indices(geometry, sim_id)

    # ts_start / ts_end filter on each sample's ts_id (third tuple element).
    indices = [i for i in indices if ts_start <= dataset._samples[i][2]]
    if ts_end is not None:
        indices = [i for i in indices if dataset._samples[i][2] < ts_end]
    if not indices:
        raise ValueError(
            f"No samples remain after ts_start={ts_start}, ts_end={ts_end} filter."
        )

    P = len(particle_feature_names)
    # Colored quantities: each named scalar feature, then the delay.
    panel_names = list(particle_feature_names) + [_DELAY_KEY]
    panel_labels = list(particle_feature_names) + [_DELAY_LABEL]
    panel_log = [False] * P + [log_delay]

    print(f"Loading {len(indices)} frames ...")
    # Per-frame: (positions, [feature columns...], delay, time).
    frames: list[tuple[np.ndarray, list[np.ndarray], np.ndarray, float]] = []
    mesh_positions: np.ndarray | None = None

    for idx in indices:
        sample, _ = dataset[idx]
        mask = sample["particle_state"][0].numpy() > 0  # (K,)
        # Every quantity is read by name and masked to the real particles.
        positions = sample["particle_coords"][0].numpy()[mask]  # (n, 3)
        feature_cols = [
            sample[name][0].numpy()[mask] for name in particle_feature_names
        ]  # P x (n,)
        delay = sample["delay"][0].numpy()[mask]  # (n,)
        time_s = float(sample["time"][0].item())
        frames.append((positions, feature_cols, delay, time_s))
        if mesh_positions is None:
            mesh_positions = sample["mesh_coords"].numpy()  # (N, 3)

    # Per-panel color ranges with percentile clipping; if a panel has 0
    # particles in every frame, fall back to a sane unit range.
    def _pooled(values_per_frame: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
        nonempty = [v for v in values_per_frame if v.size > 0]
        return np.concatenate(nonempty) if nonempty else fallback

    panel_ranges: list[tuple[float, float]] = []
    for p in range(P):
        pooled = _pooled([f[1][p] for f in frames], np.array([0.0, 1.0]))
        panel_ranges.append(
            (
                float(np.percentile(pooled, clip_pct)),
                float(np.percentile(pooled, 100.0 - clip_pct)),
            )
        )
    pooled_delay = _pooled([f[2] for f in frames], np.array([1e-12, 1.0]))
    panel_ranges.append(
        (
            float(np.percentile(pooled_delay, clip_pct)),
            float(np.percentile(pooled_delay, 100.0 - clip_pct)),
        )
    )

    # Spatial box from the mesh extent so the cube stays fixed even when only a
    # couple of particles are present early in the rollout.
    xy_range = (
        float(mesh_positions[:, 0].min()),
        float(mesh_positions[:, 0].max()),
        float(mesh_positions[:, 1].min()),
        float(mesh_positions[:, 1].max()),
    )
    z_range = (float(mesh_positions[:, 2].min()), float(mesh_positions[:, 2].max()))
    range_str = " | ".join(
        f"{lbl}: [{lo:.3g}, {hi:.3g}]"
        for lbl, (lo, hi) in zip(panel_labels, panel_ranges)
    )
    print(f"{range_str}\nz extent: {z_range} | n_frames: {len(frames)}")

    # Static 1D summary curves for the two bottom panels.
    times_arr = np.array([f[3] for f in frames], dtype=np.float64)
    n_particles_arr = np.array([f[0].shape[0] for f in frames], dtype=np.int64)
    # Summarize the first scalar feature when present, else the delay.
    summary_label = panel_labels[0]
    summary_vals = [f[1][0] if P > 0 else f[2] for f in frames]
    summary_mean_arr = np.array(
        [float(v.mean()) if v.size > 0 else np.nan for v in summary_vals]
    )
    summary_std_arr = np.array(
        [float(v.std()) if v.size > 0 else 0.0 for v in summary_vals]
    )

    # Top row: one 3D panel per colored quantity. Two static 1D panels below.
    n_panel = len(panel_names)
    fig = plt.figure(figsize=(7 * n_panel, 12))
    gs = fig.add_gridspec(3, n_panel, height_ratios=[3, 1, 1], hspace=0.32, wspace=0.05)
    scatter_axes = [fig.add_subplot(gs[0, c], projection="3d") for c in range(n_panel)]
    ax_n = fig.add_subplot(gs[1, :])
    ax_summary = fig.add_subplot(gs[2, :])

    fig.suptitle(
        f"Particle rollout, geometry {geometry}, sim {sim_id}",
        fontsize=11,
    )

    # Bottom row 1: cumulative particle count over time.
    ax_n.plot(times_arr, n_particles_arr, color="tab:blue", lw=1.5)
    final_t = float(times_arr[-1])
    final_n = int(n_particles_arr[-1])
    ax_n.plot(final_t, final_n, "o", color="tab:red", markersize=10, zorder=5)
    ax_n.annotate(
        f"final: {final_n} particles",
        xy=(final_t, final_n),
        xytext=(-95, -8),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="tab:red"),
    )
    ax_n.set_xlabel("time")
    ax_n.set_ylabel("number of particles")
    ax_n.grid(True, linestyle=":", alpha=0.5)

    # Bottom row 2: summary-quantity mean +/- std over the particle population
    # present at each timestep.
    valid = ~np.isnan(summary_mean_arr)
    ax_summary.plot(
        times_arr[valid],
        summary_mean_arr[valid],
        color="tab:blue",
        lw=1.5,
        label="mean over particles",
    )
    ax_summary.fill_between(
        times_arr[valid],
        (summary_mean_arr - summary_std_arr)[valid],
        (summary_mean_arr + summary_std_arr)[valid],
        color="tab:blue",
        alpha=0.25,
        label="mean +/- std",
    )
    ax_summary.set_xlabel("time")
    ax_summary.set_ylabel(summary_label)
    ax_summary.legend(loc="best", fontsize=9)
    ax_summary.grid(True, linestyle=":", alpha=0.5)

    def _update(frame_idx: int) -> None:
        positions, feature_cols, delay, time_s = frames[frame_idx]
        panel_values = feature_cols + [delay]
        render_frame(
            scatter_axes,
            positions=positions,
            panel_values=panel_values,
            panel_ranges=panel_ranges,
            panel_labels=panel_labels,
            panel_log=panel_log,
            z_range=z_range,
            xy_range=xy_range,
            colormap=colormap,
            title=f"t = {time_s:.2e}  (n_particles = {positions.shape[0]})",
            elev=elev,
            azim=azim,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=1000 // fps,
        repeat=False,
    )

    output_path = Path(output)
    writer = "pillow" if output_path.suffix.lower() == ".gif" else "ffmpeg"
    print(f"Saving animation to {output_path} ...")
    anim.save(str(output_path), writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    print("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate particles as 3D scatter plots, one colored panel per "
        "named scalar feature plus one for the inter-event delay."
    )
    p.add_argument("--data-dir", required=True, help="Path to the data root")
    p.add_argument(
        "--geometry",
        required=True,
        help="Geometry-group name (a subdirectory under samples/)",
    )
    p.add_argument("--sim-id", required=True, type=int, help="Simulation index")
    p.add_argument(
        "--particle-feature-names",
        nargs="+",
        required=True,
        help="Scalar particle feature names, in column order",
    )
    p.add_argument(
        "--mesh-feature-names",
        nargs="+",
        required=True,
        help="Scalar mesh field names, in column order",
    )
    p.add_argument(
        "--output",
        default="animation.gif",
        help="Output file (.gif or .mp4) [default: animation.gif]",
    )
    p.add_argument("--fps", type=int, default=5, help="Frames per second [default: 5]")
    p.add_argument(
        "--colormap", default="plasma", help="Matplotlib colormap [default: plasma]"
    )
    p.add_argument(
        "--ts-start", type=int, default=0, help="First timestep to include [default: 0]"
    )
    p.add_argument(
        "--ts-end",
        type=int,
        default=None,
        help="Last timestep (exclusive) [default: all]",
    )
    p.add_argument("--dpi", type=int, default=100, help="Figure DPI [default: 100]")
    p.add_argument(
        "--clip-pct",
        type=float,
        default=2.0,
        help="Percentile clip for the color ranges "
        "(0 = use raw min/max) [default: 2.0]",
    )
    p.add_argument(
        "--no-log-delay",
        action="store_true",
        help="Use a linear colorbar for the delay panel (default is log).",
    )
    p.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="3D view elevation angle in degrees [default: 30]",
    )
    p.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="3D view azimuth angle in degrees [default: -60]",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    animate_simulation(
        data_dir=args.data_dir,
        geometry=args.geometry,
        sim_id=args.sim_id,
        particle_feature_names=args.particle_feature_names,
        mesh_feature_names=args.mesh_feature_names,
        output=args.output,
        fps=args.fps,
        colormap=args.colormap,
        ts_start=args.ts_start,
        ts_end=args.ts_end,
        dpi=args.dpi,
        clip_pct=args.clip_pct,
        elev=args.elev,
        azim=args.azim,
        log_delay=not args.no_log_delay,
    )
