# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

r"""
Plotting and visualization utilities for flood prediction.
"""

import os
import warnings

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib defaults
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def create_rollout_animation(
        geometry,
        wd_gt, wd_pred,
        vx_gt, vy_gt,
        vx_pred, vy_pred,
        run_id=None,
        out_dir=".",
        filename_prefix="rollout",
        dt_seconds: float = 1200.0
):
    r"""
    Creates an animation comparing Ground Truth and Predictions in a 3x2 grid.

    Parameters
    ----------
    geometry : np.ndarray or torch.Tensor
        Geometry coordinates of shape :math:`(n_{cells}, 2)`.
    wd_gt : np.ndarray or torch.Tensor
        Ground truth water depth of shape :math:`(T, n_{cells})`.
    wd_pred : np.ndarray or torch.Tensor
        Predicted water depth of shape :math:`(T, n_{cells})`.
    vx_gt : np.ndarray or torch.Tensor
        Ground truth x-velocity of shape :math:`(T, n_{cells})`.
    vy_gt : np.ndarray or torch.Tensor
        Ground truth y-velocity of shape :math:`(T, n_{cells})`.
    vx_pred : np.ndarray or torch.Tensor
        Predicted x-velocity of shape :math:`(T, n_{cells})`.
    vy_pred : np.ndarray or torch.Tensor
        Predicted y-velocity of shape :math:`(T, n_{cells})`.
    run_id : str, optional
        Run identifier for title.
    out_dir : str, optional, default="."
        Output directory for animation file.
    filename_prefix : str, optional, default="rollout"
        Prefix for output filename.
    dt_seconds : float, optional, default=1200.0
        Time step size in seconds.
    """
    # Convert inputs to numpy arrays
    if not isinstance(geometry, np.ndarray) and hasattr(geometry, "cpu"):
        geometry = geometry.cpu().numpy()
    x_coords, y_coords = geometry[:, 0], geometry[:, 1]

    wd_gt, wd_pred = np.asarray(wd_gt), np.asarray(wd_pred)
    vx_gt, vy_gt = np.asarray(vx_gt), np.asarray(vy_gt)
    vx_pred, vy_pred = np.asarray(vx_pred), np.asarray(vy_pred)
    rollout_length = wd_gt.shape[0]

    # Prepare figure with a 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)
    fig.suptitle(f"Rollout Comparison (Run: {run_id or 'unknown'})", fontsize=20)
    (ax_gt_wd, ax_pred_wd), (ax_gt_vx, ax_pred_vx), (ax_gt_vy, ax_pred_vy) = axes

    # Set Color Limits
    depth_max = max(np.nanmax(wd_gt), np.nanmax(wd_pred))
    vx_abs_max = np.max([np.abs(vx_gt), np.abs(vx_pred)])
    vy_abs_max = np.max([np.abs(vy_gt), np.abs(vy_pred)])

    # Row 1: Water Depth
    sc_gt_wd = ax_gt_wd.scatter(x_coords, y_coords, c=wd_gt[0], vmin=0, vmax=depth_max, s=15, cmap='viridis')
    ax_gt_wd.set_title("Ground Truth Depth", pad=10)
    ax_gt_wd.axis('off')
    fig.colorbar(sc_gt_wd, ax=ax_gt_wd, fraction=0.046, pad=0.04).set_label("Depth (m)")

    sc_pred_wd = ax_pred_wd.scatter(x_coords, y_coords, c=wd_pred[0], vmin=0, vmax=depth_max, s=15, cmap='viridis')
    ax_pred_wd.set_title("Predicted Depth", pad=10)
    ax_pred_wd.axis('off')
    fig.colorbar(sc_pred_wd, ax=ax_pred_wd, fraction=0.046, pad=0.04).set_label("Depth (m)")

    # Row 2: X-Velocity (Vx)
    sc_gt_vx = ax_gt_vx.scatter(x_coords, y_coords, c=vx_gt[0], vmin=-vx_abs_max, vmax=vx_abs_max, s=15,
                                cmap='coolwarm')
    ax_gt_vx.set_title(r"Ground Truth $V_{x}$", pad=10)
    ax_gt_vx.axis('off')
    fig.colorbar(sc_gt_vx, ax=ax_gt_vx, fraction=0.046, pad=0.04).set_label(r"$V_{x}$ (m/s)")

    sc_pred_vx = ax_pred_vx.scatter(x_coords, y_coords, c=vx_pred[0], vmin=-vx_abs_max, vmax=vx_abs_max, s=15,
                                    cmap='coolwarm')
    ax_pred_vx.set_title(r"Predicted $V_{x}$", pad=10)
    ax_pred_vx.axis('off')
    fig.colorbar(sc_pred_vx, ax=ax_pred_vx, fraction=0.046, pad=0.04).set_label(r"$V_{x}$ (m/s)")

    # Row 3: Y-Velocity (Vy)
    sc_gt_vy = ax_gt_vy.scatter(x_coords, y_coords, c=vy_gt[0], vmin=-vy_abs_max, vmax=vy_abs_max, s=15,
                                cmap='coolwarm')
    ax_gt_vy.set_title(r"Ground Truth $V_{y}$", pad=10)
    ax_gt_vy.axis('off')
    fig.colorbar(sc_gt_vy, ax=ax_gt_vy, fraction=0.046, pad=0.04).set_label(r"$V_{y}$ (m/s)")

    sc_pred_vy = ax_pred_vy.scatter(x_coords, y_coords, c=vy_pred[0], vmin=-vy_abs_max, vmax=vy_abs_max, s=15,
                                    cmap='coolwarm')
    ax_pred_vy.set_title(r"Predicted $V_{y}$", pad=10)
    ax_pred_vy.axis('off')
    fig.colorbar(sc_pred_vy, ax=ax_pred_vy, fraction=0.046, pad=0.04).set_label(r"$V_{y}$ (m/s)")

    # Animation update function
    def animate(frame_idx):
        time_hours = (frame_idx + 1) * dt_seconds / 3600.0
        fig.suptitle(f"Rollout Comparison (Run: {run_id or 'unknown'}) - Time: {time_hours:.2f} hrs", fontsize=20)
        sc_gt_wd.set_array(wd_gt[frame_idx])
        sc_pred_wd.set_array(wd_pred[frame_idx])
        sc_gt_vx.set_array(vx_gt[frame_idx])
        sc_pred_vx.set_array(vx_pred[frame_idx])
        sc_gt_vy.set_array(vy_gt[frame_idx])
        sc_pred_vy.set_array(vy_pred[frame_idx])
        return sc_gt_wd, sc_pred_wd, sc_gt_vx, sc_pred_vx, sc_gt_vy, sc_pred_vy

    ani = animation.FuncAnimation(fig, animate, frames=rollout_length, interval=200, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{run_id or 'unknown'}.gif")
    ani.save(out_path, writer="pillow", fps=5)
    plt.close(fig)
    # Logging handled by caller


def _r_squared(y_true, y_pred):
    r"""Calculate R^2 score, handling NaNs."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0


def generate_publication_maps(
        geometry,
        wd_gt_array: np.ndarray, wd_pred_array: np.ndarray,
        vx_gt_array: np.ndarray, vy_gt_array: np.ndarray,
        vx_pred_array: np.ndarray, vy_pred_array: np.ndarray,
        steps,
        out_dir: str = ".",
        run_id: str = None,
        filename_prefix: str = "step"
):
    r"""
    Generates high-quality 3x4 comparison maps for specific timesteps.
    
    Columns: Ground Truth | Prediction | Absolute Error | Scatter Plot

    Parameters
    ----------
    geometry : np.ndarray or torch.Tensor
        Geometry coordinates of shape :math:`(n_{cells}, 2)`.
    wd_gt_array : np.ndarray
        Ground truth water depth of shape :math:`(T, n_{cells})`.
    wd_pred_array : np.ndarray
        Predicted water depth of shape :math:`(T, n_{cells})`.
    vx_gt_array : np.ndarray
        Ground truth x-velocity of shape :math:`(T, n_{cells})`.
    vy_gt_array : np.ndarray
        Ground truth y-velocity of shape :math:`(T, n_{cells})`.
    vx_pred_array : np.ndarray
        Predicted x-velocity of shape :math:`(T, n_{cells})`.
    vy_pred_array : np.ndarray
        Predicted y-velocity of shape :math:`(T, n_{cells})`.
    steps : int or List[int]
        Timestep(s) to generate maps for.
    out_dir : str, optional, default="."
        Output directory for maps.
    run_id : str, optional
        Run identifier for filename.
    filename_prefix : str, optional, default="step"
        Prefix for output filename.
    """
    if isinstance(steps, int):
        steps = [steps]
    geo_np = geometry.cpu().numpy() if hasattr(geometry, "cpu") else np.asarray(geometry)
    x, y = geo_np[:, 0], geo_np[:, 1]
    rid = run_id or "unknown"
    os.makedirs(out_dir, exist_ok=True)
    plt.rc("font", family="serif", size=12)

    for t in steps:
        if t < 0 or t >= wd_gt_array.shape[0]:
            warnings.warn(f"Skipping invalid step {t}")
            continue

        wd_gt, wd_pred = wd_gt_array[t], wd_pred_array[t]
        vx_gt, vy_gt = vx_gt_array[t], vy_gt_array[t]
        vx_pred, vy_pred = vx_pred_array[t], vy_pred_array[t]
        err_wd, err_vx, err_vy = np.abs(wd_pred - wd_gt), np.abs(vx_pred - vx_gt), np.abs(vy_pred - vy_gt)

        dmax = max(np.nanmax(wd_gt), np.nanmax(wd_pred))
        emax_wd = np.nanmax(err_wd)
        vx_abs_max = np.max([np.abs(vx_gt), np.abs(vx_pred)]) if np.any(vx_gt) or np.any(vx_pred) else 1.0
        vy_abs_max = np.max([np.abs(vy_gt), np.abs(vy_pred)]) if np.any(vy_gt) or np.any(vy_pred) else 1.0
        emax_vx, emax_vy = np.nanmax(err_vx), np.nanmax(err_vy)

        fig, axs = plt.subplots(3, 4, figsize=(24, 17), dpi=300, constrained_layout=True)

        # Populate Spatial Maps (First 3 columns)
        map_panels = [
            ("(a) Ground Truth Depth", wd_gt, "viridis", 0.0, dmax, "Depth (m)"),
            ("(b) Predicted Depth", wd_pred, "viridis", 0.0, dmax, "Depth (m)"),
            ("(c) Depth Absolute Error", err_wd, "magma", 0.0, emax_wd, "Error (m)"),
            (r"(d) Ground Truth $V_{x}$", vx_gt, "coolwarm", -vx_abs_max, vx_abs_max, r"$V_{x}$ (m/s)"),
            (r"(e) Predicted $V_{x}$", vx_pred, "coolwarm", -vx_abs_max, vx_abs_max, r"$V_{x}$ (m/s)"),
            (r"(f) $V_{x}$ Absolute Error", err_vx, "magma", 0.0, emax_vx, "Error (m/s)"),
            (r"(g) Ground Truth $V_{y}$", vy_gt, "coolwarm", -vy_abs_max, vy_abs_max, r"$V_{y}$ (m/s)"),
            (r"(h) Predicted $V_{y}$", vy_pred, "coolwarm", -vy_abs_max, vy_abs_max, r"$V_{y}$ (m/s)"),
            (r"(i) $V_{y}$ Absolute Error", err_vy, "magma", 0.0, emax_vy, "Error (m/s)"),
        ]

        for i, (title, data, cmap, vmin, vmax, cblabel) in enumerate(map_panels):
            row, col = i // 3, i % 3
            ax = axs[row, col]
            sc = ax.scatter(x, y, c=data, cmap=cmap, vmin=vmin, vmax=vmax, s=6, marker="s", linewidths=0,
                            rasterized=True)
            ax.set_title(title, pad=8, fontsize=14)
            ax.set_aspect("equal")
            ax.axis("off")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cblabel, labelpad=10, fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        # Populate Scatter Plots (4th column)
        scatter_data = [
            ("Depth", wd_gt, wd_pred),
            (r"$V_{x}$", vx_gt, vx_pred),
            (r"$V_{y}$", vy_gt, vy_pred)
        ]

        for i, (var_name, gt, pred) in enumerate(scatter_data):
            ax = axs[i, 3]
            r2 = _r_squared(gt, pred)
            ax.scatter(gt, pred, alpha=0.4, s=8, rasterized=True, c='royalblue', edgecolors='none')
            lims = [min(np.nanmin(gt), np.nanmin(pred)), max(np.nanmax(gt), np.nanmax(pred))]
            if lims[0] < lims[1]:
                ax.plot(lims, lims, 'k--', alpha=0.8, zorder=10, label="1:1 Line")
                ax.set_xlim(lims)
                ax.set_ylim(lims)

            ax.set_aspect('equal', 'box')
            ax.set_xlabel(f"Ground Truth {var_name}")
            ax.set_ylabel(f"Predicted {var_name}")
            ax.set_title(f"{var_name} Correlation\n$R^2 = {r2:.3f}$")
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc="upper left")

        fname = f"{filename_prefix}_{rid}_t{t}.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        # Logging handled by caller


def generate_max_value_maps(
        geometry,
        wd_gt_array: np.ndarray, wd_pred_array: np.ndarray,
        vx_gt_array: np.ndarray, vy_gt_array: np.ndarray,
        vx_pred_array: np.ndarray, vy_pred_array: np.ndarray,
        out_dir: str = ".",
        run_id: str = None,
        filename_prefix: str = "max_values"
):
    r"""
    Generates 3x4 comparison maps of the maximum value over time for each point.
    
    Columns: Ground Truth | Prediction | Absolute Error | Scatter Plot

    Parameters
    ----------
    geometry : np.ndarray or torch.Tensor
        Geometry coordinates of shape :math:`(n_{cells}, 2)`.
    wd_gt_array : np.ndarray
        Ground truth water depth of shape :math:`(T, n_{cells})`.
    wd_pred_array : np.ndarray
        Predicted water depth of shape :math:`(T, n_{cells})`.
    vx_gt_array : np.ndarray
        Ground truth x-velocity of shape :math:`(T, n_{cells})`.
    vy_gt_array : np.ndarray
        Ground truth y-velocity of shape :math:`(T, n_{cells})`.
    vx_pred_array : np.ndarray
        Predicted x-velocity of shape :math:`(T, n_{cells})`.
    vy_pred_array : np.ndarray
        Predicted y-velocity of shape :math:`(T, n_{cells})`.
    out_dir : str, optional, default="."
        Output directory for maps.
    run_id : str, optional
        Run identifier for filename.
    filename_prefix : str, optional, default="max_values"
        Prefix for output filename.
    """
    max_wd_gt, max_wd_pred = np.max(wd_gt_array, axis=0), np.max(wd_pred_array, axis=0)
    err_max_wd = np.abs(max_wd_pred - max_wd_gt)

    max_vx_gt, max_vx_pred = np.max(vx_gt_array, axis=0), np.max(vx_pred_array, axis=0)
    err_max_vx = np.abs(max_vx_pred - max_vx_gt)

    max_vy_gt, max_vy_pred = np.max(vy_gt_array, axis=0), np.max(vy_pred_array, axis=0)
    err_max_vy = np.abs(max_vy_pred - max_vy_gt)

    geo_np = geometry.cpu().numpy() if hasattr(geometry, "cpu") else np.asarray(geometry)
    x, y = geo_np[:, 0], geo_np[:, 1]
    rid = run_id or "unknown"
    os.makedirs(out_dir, exist_ok=True)
    plt.rc("font", family="serif", size=12)

    dmax = max(np.nanmax(max_wd_gt), np.nanmax(max_wd_pred))
    emax_wd = np.nanmax(err_max_wd)
    vx_abs_max = np.max([np.abs(max_vx_gt), np.abs(max_vx_pred)]) if np.any(max_vx_gt) or np.any(max_vx_pred) else 1.0
    vy_abs_max = np.max([np.abs(max_vy_gt), np.abs(max_vy_pred)]) if np.any(max_vy_gt) or np.any(max_vy_pred) else 1.0
    emax_vx, emax_vy = np.nanmax(err_max_vx), np.nanmax(err_max_vy)

    fig, axs = plt.subplots(3, 4, figsize=(24, 17), dpi=300, constrained_layout=True)

    # Populate Spatial Maps (First 3 columns)
    map_panels = [
        ("(a) Max Ground Truth Depth", max_wd_gt, "viridis", 0.0, dmax, "Depth (m)"),
        ("(b) Max Predicted Depth", max_wd_pred, "viridis", 0.0, dmax, "Depth (m)"),
        ("(c) Max Depth Absolute Error", err_max_wd, "magma", 0.0, emax_wd, "Error (m)"),
        (r"(d) Max Ground Truth $V_{x}$", max_vx_gt, "coolwarm", -vx_abs_max, vx_abs_max, r"$V_{x}$ (m/s)"),
        (r"(e) Max Predicted $V_{x}$", max_vx_pred, "coolwarm", -vx_abs_max, vx_abs_max, r"$V_{x}$ (m/s)"),
        (r"(f) Max $V_{x}$ Absolute Error", err_max_vx, "magma", 0.0, emax_vx, "Error (m/s)"),
        (r"(g) Max Ground Truth $V_{y}$", max_vy_gt, "coolwarm", -vy_abs_max, vy_abs_max, r"$V_{y}$ (m/s)"),
        (r"(h) Max Predicted $V_{y}$", max_vy_pred, "coolwarm", -vy_abs_max, vy_abs_max, r"$V_{y}$ (m/s)"),
        (r"(i) Max $V_{y}$ Absolute Error", err_max_vy, "magma", 0.0, emax_vy, "Error (m/s)"),
    ]

    for i, (title, data, cmap, vmin, vmax, cblabel) in enumerate(map_panels):
        row, col = i // 3, i % 3
        ax = axs[row, col]
        sc = ax.scatter(x, y, c=data, cmap=cmap, vmin=vmin, vmax=vmax, s=6, marker="s", linewidths=0, rasterized=True)
        ax.set_title(title, pad=8, fontsize=14)
        ax.set_aspect("equal")
        ax.axis("off")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(cblabel, labelpad=10, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Populate Scatter Plots (4th column)
    scatter_data = [
        ("Max Depth", max_wd_gt, max_wd_pred),
        (r"Max $V_{x}$", max_vx_gt, max_vx_pred),
        (r"Max $V_{y}$", max_vy_gt, max_vy_pred)
    ]

    for i, (var_name, gt, pred) in enumerate(scatter_data):
        ax = axs[i, 3]
        r2 = _r_squared(gt, pred)
        ax.scatter(gt, pred, alpha=0.4, s=8, rasterized=True, c='royalblue', edgecolors='none')
        lims = [min(np.nanmin(gt), np.nanmin(pred)), max(np.nanmax(gt), np.nanmax(pred))]
        if lims[0] < lims[1]:
            ax.plot(lims, lims, 'k--', alpha=0.8, zorder=10, label="1:1 Line")
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        ax.set_aspect('equal', 'box')
        ax.set_xlabel(f"Ground Truth {var_name}")
        ax.set_ylabel(f"Predicted {var_name}")
        ax.set_title(f"{var_name} Correlation\n$R^2 = {r2:.3f}$")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc="upper left")

    fname = f"{filename_prefix}_{rid}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    # Logging handled by caller


def generate_combined_analysis_maps(
        geometry,
        wd_gt_array: np.ndarray, wd_pred_array: np.ndarray,
        vx_gt_array: np.ndarray, vy_gt_array: np.ndarray,
        vx_pred_array: np.ndarray, vy_pred_array: np.ndarray,
        dt: float,
        out_dir: str = ".",
        run_id: str = None,
        inundation_threshold: float = 0.1,
):
    r"""
    Calculates and plots key temporal and hazard metrics in a single 3x4 figure.
    
    Rows: 1. Arrival Time, 2. Inundation Duration, 3. Max Momentum Flux
    Columns: Ground Truth | Predicted | Absolute Error | Scatter Plot

    Parameters
    ----------
    geometry : np.ndarray or torch.Tensor
        Geometry coordinates of shape :math:`(n_{cells}, 2)`.
    wd_gt_array : np.ndarray
        Ground truth water depth of shape :math:`(T, n_{cells})`.
    wd_pred_array : np.ndarray
        Predicted water depth of shape :math:`(T, n_{cells})`.
    vx_gt_array : np.ndarray
        Ground truth x-velocity of shape :math:`(T, n_{cells})`.
    vy_gt_array : np.ndarray
        Ground truth y-velocity of shape :math:`(T, n_{cells})`.
    vx_pred_array : np.ndarray
        Predicted x-velocity of shape :math:`(T, n_{cells})`.
    vy_pred_array : np.ndarray
        Predicted y-velocity of shape :math:`(T, n_{cells})`.
    dt : float
        Time step size in seconds.
    out_dir : str, optional, default="."
        Output directory for maps.
    run_id : str, optional
        Run identifier for filename.
    inundation_threshold : float, optional, default=0.1
        Threshold for inundation classification (meters).

    Returns
    -------
    Tuple[float, float, float, float]
        Tuple of (mae_arrival, mae_duration, rmse_hv2, fhca).
    """
    rid = run_id or "unknown"
    os.makedirs(out_dir, exist_ok=True)
    plt.rc("font", family="serif", size=12)

    # Calculate all required metrics
    # Arrival Time
    def calculate_arrival(arr, threshold, dt_val):  # noqa: D401
        inundated_mask = arr >= threshold
        arrival_times = (np.argmax(inundated_mask, axis=0)).astype(np.float64) * dt_val
        never_inundated_mask = ~inundated_mask.any(axis=0)
        arrival_times[never_inundated_mask] = np.nan
        return arrival_times

    arrival_gt = calculate_arrival(wd_gt_array, inundation_threshold, dt)
    arrival_pred = calculate_arrival(wd_pred_array, inundation_threshold, dt)

    # Inundation Duration
    duration_gt = np.sum(wd_gt_array >= inundation_threshold, axis=0) * dt
    duration_pred = np.sum(wd_pred_array >= inundation_threshold, axis=0) * dt

    # Maximum Momentum Flux (h*V^2)
    v_gt = np.sqrt(vx_gt_array ** 2 + vy_gt_array ** 2)
    v_pred = np.sqrt(vx_pred_array ** 2 + vy_pred_array ** 2)
    hv2_gt, hv2_pred = wd_gt_array * (v_gt ** 2), wd_pred_array * (v_pred ** 2)
    max_hv2_gt, max_hv2_pred = np.max(hv2_gt, axis=0), np.max(hv2_pred, axis=0)

    # Error Metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        err_arrival = np.abs(arrival_pred - arrival_gt)
        err_duration = np.abs(duration_pred - duration_gt)
        err_max_hv2 = np.abs(max_hv2_pred - max_hv2_gt)

        # Scalar metrics for return
        mae_arrival = np.nanmean(err_arrival)
        mae_duration = np.nanmean(err_duration)
        rmse_hv2 = np.sqrt(np.mean((max_hv2_pred - max_hv2_gt) ** 2))

    # Hazard Classification (FHCA) - based on hV
    hv_gt, hv_pred = wd_gt_array * v_gt, wd_pred_array * v_pred
    max_hv_gt = np.max(hv_gt, axis=0)
    zones = {'Low': (0, 0.5), 'Medium': (0.5, 1.5), 'High': (1.5, np.inf)}

    def classify(hv_values, zones_dict):  # noqa: D401
        classes = np.zeros_like(hv_values, dtype=int)
        classes[hv_values >= zones_dict['Medium'][0]] = 1
        classes[hv_values >= zones_dict['High'][0]] = 2
        return classes

    gt_class, pred_class = classify(max_hv_gt, zones), classify(np.max(hv_pred, axis=0), zones)
    fhca = np.mean(gt_class == pred_class)

    # Setup Figure
    geo_np = geometry.cpu().numpy() if hasattr(geometry, "cpu") else np.asarray(geometry)
    x, y = geo_np[:, 0], geo_np[:, 1]
    fig, axs = plt.subplots(3, 4, figsize=(24, 17), dpi=300, constrained_layout=True)

    # Populate Panels
    to_hours = lambda sec: sec / 3600.0 if sec is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vmax_arrival = to_hours(np.nanmax([arrival_gt, arrival_pred]))
        emax_arrival = to_hours(np.nanmean(err_arrival) * 2)
        vmax_duration = to_hours(np.nanmax([duration_gt, duration_pred]))
        emax_duration = to_hours(np.nanmean(err_duration) * 2)
    vmax_hv2 = max(np.nanmax(max_hv2_gt), np.nanmax(max_hv2_pred))
    emax_hv2 = np.nanmax(err_max_hv2)

    # Panel Data Definitions
    map_panels = [
        # Row 1: Arrival Time
        ("Ground Truth Arrival Time", to_hours(arrival_gt), "plasma", 0.0, vmax_arrival, "Time (hours)"),
        ("Predicted Arrival Time", to_hours(arrival_pred), "plasma", 0.0, vmax_arrival, "Time (hours)"),
        ("Arrival Time Absolute Error", to_hours(err_arrival), "magma", 0.0, emax_arrival, "Error (hours)"),
        # Row 2: Duration
        ("Ground Truth Inundation Duration", to_hours(duration_gt), "cividis", 0.0, vmax_duration, "Time (hours)"),
        ("Predicted Inundation Duration", to_hours(duration_pred), "cividis", 0.0, vmax_duration, "Time (hours)"),
        ("Duration Absolute Error", to_hours(err_duration), "magma", 0.0, emax_duration, "Error (hours)"),
        # Row 3: Momentum Flux
        (r"Ground Truth max($h \cdot V^2$)", max_hv2_gt, "YlOrRd", 0.0, vmax_hv2, r"Momentum Flux ($m^3/s^2$)"),
        (r"Predicted max($h \cdot V^2$)", max_hv2_pred, "YlOrRd", 0.0, vmax_hv2, r"Momentum Flux ($m^3/s^2$)"),
        (r"max($h \cdot V^2$) Absolute Error", err_max_hv2, "Blues", 0.0, emax_hv2, r"Error ($m^3/s^2$)"),
    ]

    scatter_panels = [
        ("Arrival Time", arrival_gt, arrival_pred, "hours"),
        ("Inundation Duration", duration_gt, duration_pred, "hours"),
        (r"max($h \cdot V^2$)", max_hv2_gt, max_hv2_pred, r"$m^3/s^2$")
    ]

    # Plotting Loops
    # Plot Maps
    for i, (title, data, cmap, vmin, vmax, cblabel) in enumerate(map_panels):
        row, col = i // 3, i % 3
        ax = axs[row, col]
        if data is not None and np.any(data[~np.isnan(data)]) and np.nanmax(data) > 0:
            sc = ax.scatter(x, y, c=data, cmap=cmap, vmin=vmin, vmax=vmax, s=6, marker="s", linewidths=0,
                            rasterized=True)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cblabel, labelpad=10, fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, pad=8, fontsize=14)
        ax.set_aspect("equal")
        ax.axis("off")

    # Plot Scatters
    for i, (var_name, gt_vals, pred_vals, unit) in enumerate(scatter_panels):
        ax_scatter = axs[i, 3]
        valid_indices = ~np.isnan(gt_vals) & ~np.isnan(pred_vals)
        gt_plot, pred_plot = gt_vals[valid_indices], pred_vals[valid_indices]

        r2 = _r_squared(gt_plot, pred_plot)
        title = f"{var_name} Correlation\n$R^2 = {r2:.3f}$"

        if len(gt_plot) > 0:
            plot_gt = gt_plot / 3600.0 if unit == "hours" else gt_plot
            plot_pred = pred_plot / 3600.0 if unit == "hours" else pred_plot

            ax_scatter.scatter(plot_gt, plot_pred, alpha=0.5, s=10, rasterized=True, c='blue')
            lims = [min(np.min(plot_gt), np.min(plot_pred)), max(np.max(plot_gt), np.max(plot_pred))]
            if lims[0] < lims[1]:
                ax_scatter.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label="1:1 Line")
                ax_scatter.set_xlim(lims)
                ax_scatter.set_ylim(lims)
            ax_scatter.set_aspect('equal', 'box')
            ax_scatter.set_xlabel(f"Ground Truth ({unit})")
            ax_scatter.set_ylabel(f"Prediction ({unit})")
            ax_scatter.set_title(title, pad=8, fontsize=14)
            ax_scatter.legend(loc="upper left")
            ax_scatter.grid(True, linestyle=':')
        else:
            ax_scatter.text(0.5, 0.5, 'No valid data points', ha='center', va='center', transform=ax_scatter.transAxes)
            ax_scatter.set_title(title, pad=8, fontsize=14)

    # Save and return
    out_path = os.path.join(out_dir, f"combined_analysis_{rid}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
        # Logging handled by caller

    return mae_arrival, mae_duration, rmse_hv2, fhca


def plot_volume_conservation(wd_gt_array, wd_pred_array, cell_area, dt, out_dir, run_id):
    r"""
    Calculates and plots the total inundated volume over time.

    Parameters
    ----------
    wd_gt_array : np.ndarray
        Ground truth water depth of shape :math:`(T, n_{cells})`.
    wd_pred_array : np.ndarray
        Predicted water depth of shape :math:`(T, n_{cells})`.
    cell_area : np.ndarray or torch.Tensor
        Cell area of shape :math:`(n_{cells},)`.
    dt : float
        Time step size in seconds.
    out_dir : str
        Output directory for plot.
    run_id : str
        Run identifier for filename.
    """
    if cell_area is None:
        warnings.warn("Skipping volume conservation plot: Cell area not available.")
        return

    # Ensure the cell_area array matches the number of cells in the simulation data.
    num_cells_in_sim = wd_gt_array.shape[1]
    if cell_area.shape[0] != num_cells_in_sim:
        warnings.warn(
            f"Cell area array shape ({cell_area.shape[0]}) does not match simulation cell count ({num_cells_in_sim}). "
            f"Trimming cell area array to match. Please check input data consistency."
        )
        cell_area = cell_area[:num_cells_in_sim]

    # Calculate total volume at each time step
    volume_gt = np.sum(wd_gt_array * cell_area, axis=1)
    volume_pred = np.sum(wd_pred_array * cell_area, axis=1)

    # Create time axis
    time_hours = np.arange(len(volume_gt)) * dt / 3600.0

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.plot(time_hours, volume_gt, label='Ground Truth', color='black', linestyle='-')
    ax.plot(time_hours, volume_pred, label='Prediction', color='red', linestyle='--')

    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel(r'Total Volume ($m^3$)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"Total_Volume_vs_Time_{run_id}.png")
    plt.savefig(save_path)
    plt.close(fig)
    # Logging handled by caller


def plot_aggregated_scalar_metrics(scalar_metrics, out_dir):
    r"""
    Creates and saves a box plot summary of scalar metrics aggregated over the entire test dataset.

    Parameters
    ----------
    scalar_metrics : Dict[str, List[float]]
        Dictionary of scalar metrics with keys: 'h_V2_rmse', 'fhca', 'arrival_mae_hrs', 'duration_mae_hrs'.
    out_dir : str
        Output directory for plot.
    """
    labels = {
        'h_V2_rmse': r'Max $h \cdot V^2$ RMSE' + '\n' + r'($m^3/s^2$)', 'fhca': 'FHCA',
        'arrival_mae_hrs': 'Arrival MAE\n(hours)', 'duration_mae_hrs': 'Duration MAE\n(hours)',
    }

    # Prepare data for boxplots
    hazard_data = [
        np.array(scalar_metrics.get('h_V2_rmse', [])),
        np.array(scalar_metrics.get('fhca', []))
    ]
    hazard_labels = [labels['h_V2_rmse'], labels['fhca']]

    temporal_data = [
        np.array(scalar_metrics.get('arrival_mae_hrs', [])),
        np.array(scalar_metrics.get('duration_mae_hrs', []))
    ]
    temporal_labels = [labels['arrival_mae_hrs'], labels['duration_mae_hrs']]

    fig, axs = plt.subplots(1, 2, figsize=(12, 7), dpi=150)
    fig.suptitle("Aggregated Model Performance Across All Test Simulations", fontsize=20, y=1.0)

    # Boxplot for Hazard Metrics
    bp1 = axs[0].boxplot(hazard_data, vert=True, patch_artist=True, whis=1.5, labels=hazard_labels)
    axs[0].set_title('Hazard Metrics', fontsize=16)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Boxplot for Temporal MAE
    bp2 = axs[1].boxplot(temporal_data, vert=True, patch_artist=True, labels=temporal_labels)
    axs[1].set_title('Temporal Characteristics MAE', fontsize=16)
    axs[1].set_ylabel('Mean Absolute Error (hours)')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Coloring
    colors = ['lightblue', 'lightgreen']
    for patch in bp1['boxes']:
        patch.set_facecolor(colors[0])
    for patch in bp2['boxes']:
        patch.set_facecolor(colors[1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(out_dir, "rollout_scalar_metrics_boxplot.png")
    plt.savefig(save_path)
    plt.close(fig)
    # Logging handled by caller


def plot_conditional_error_analysis(
        wd_gt_array: np.ndarray, wd_pred_array: np.ndarray,
        vx_gt_array: np.ndarray, vy_gt_array: np.ndarray,
        vx_pred_array: np.ndarray, vy_pred_array: np.ndarray,
        out_dir: str,
        run_id: str
):
    r"""
    Creates and saves plots for conditional error analysis.
    
    1. Absolute Depth Error vs. True Water Depth
    2. Absolute Velocity Magnitude Error vs. True Velocity Magnitude

    Parameters
    ----------
    wd_gt_array : np.ndarray
        Ground truth water depth of shape :math:`(T, n_{cells})`.
    wd_pred_array : np.ndarray
        Predicted water depth of shape :math:`(T, n_{cells})`.
    vx_gt_array : np.ndarray
        Ground truth x-velocity of shape :math:`(T, n_{cells})`.
    vy_gt_array : np.ndarray
        Ground truth y-velocity of shape :math:`(T, n_{cells})`.
    vx_pred_array : np.ndarray
        Predicted x-velocity of shape :math:`(T, n_{cells})`.
    vy_pred_array : np.ndarray
        Predicted y-velocity of shape :math:`(T, n_{cells})`.
    out_dir : str
        Output directory for plot.
    run_id : str
        Run identifier for filename.
    """
    # Logging handled by caller

    # Calculate required values
    wd_gt_flat = wd_gt_array.flatten()
    wd_pred_flat = wd_pred_array.flatten()

    # Absolute error for water depth
    err_wd_abs = np.abs(wd_pred_flat - wd_gt_flat)

    # Calculate velocity magnitudes
    v_mag_gt = np.sqrt(vx_gt_array ** 2 + vy_gt_array ** 2).flatten()
    v_mag_pred = np.sqrt(vx_pred_array ** 2 + vy_pred_array ** 2).flatten()

    # Absolute error for velocity magnitude
    err_v_mag_abs = np.abs(v_mag_pred - v_mag_gt)

    # Create the plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    # Plot 1: Depth Error vs. True Depth
    ax1 = axs[0]
    mask_depth = wd_gt_flat > 0.01
    ax1.scatter(wd_gt_flat[mask_depth], err_wd_abs[mask_depth],
                alpha=0.1, s=5, c='blue', rasterized=True, edgecolors='none')
    ax1.set_xlabel("Ground Truth Water Depth (m)", fontsize=14)
    ax1.set_ylabel("Absolute Error (m)", fontsize=14)
    ax1.set_title("(a) Depth Error vs. True Depth", fontsize=16)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    # Plot 2: Velocity Error vs. True Velocity
    ax2 = axs[1]
    mask_vel = v_mag_gt > 0.01
    ax2.scatter(v_mag_gt[mask_vel], err_v_mag_abs[mask_vel],
                alpha=0.1, s=5, c='green', rasterized=True, edgecolors='none')
    ax2.set_xlabel("Ground Truth Velocity Magnitude (m/s)", fontsize=14)
    ax2.set_ylabel("Absolute Error (m/s)", fontsize=14)
    ax2.set_title("(b) Velocity Magnitude Error vs. True Velocity", fontsize=16)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(out_dir, f"conditional_error_analysis_{run_id}.png")
    plt.savefig(save_path)
    plt.close(fig)
    # Logging handled by caller


def plot_event_magnitude_analysis(
        q_peaks: list,
        total_volumes: list,
        avg_rmses_wd: list,
        out_dir: str
):
    r"""
    Creates and saves two separate scatter plots correlating model error with
    hydrograph characteristics.

    Parameters
    ----------
    q_peaks : List[float]
        List of peak discharge values.
    total_volumes : List[float]
        List of total volume values.
    avg_rmses_wd : List[float]
        List of average RMSE values for water depth.
    out_dir : str
        Output directory for plots.

    The function creates two plots:
    1. Overall RMSE vs. Peak Inflow (Q_peak)
    2. Overall RMSE vs. Total Inflow Volume
    """
    # Logging handled by caller
    os.makedirs(out_dir, exist_ok=True)

    # Convert lists to numpy arrays for easier plotting
    q_peaks_arr = np.array(q_peaks)
    total_volumes_arr = np.array(total_volumes)
    avg_rmses_arr = np.array(avg_rmses_wd)

    # Figure 1: RMSE vs. Peak Inflow
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax1.scatter(q_peaks_arr, avg_rmses_arr, alpha=0.7, c='coral', edgecolors='k', s=50)
    ax1.set_xlabel("Peak Inflow ($Q_{peak}$, $m^3/s$)", fontsize=14)
    ax1.set_ylabel("Time-Averaged Water Depth RMSE (m)", fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Add trend line
    z1 = np.polyfit(q_peaks_arr, avg_rmses_arr, 1)
    p1 = np.poly1d(z1)
    ax1.plot(q_peaks_arr, p1(q_peaks_arr), "k--", alpha=0.8, label=f"Trend (slope={z1[0]:.4f})")
    ax1.legend()

    plt.tight_layout()
    save_path1 = os.path.join(out_dir, "rmse_vs_peak_inflow.png")
    plt.savefig(save_path1)
    plt.close(fig1)
    # Logging handled by caller

    # Figure 2: RMSE vs. Total Volume
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2.scatter(total_volumes_arr, avg_rmses_arr, alpha=0.7, c='deepskyblue', edgecolors='k', s=50)
    ax2.set_xlabel("Total Inflow Volume ($m^3$)", fontsize=14)
    ax2.set_ylabel("Time-Averaged Water Depth RMSE (m)", fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Add trend line
    z2 = np.polyfit(total_volumes_arr, avg_rmses_arr, 1)
    p2 = np.poly1d(z2)
    ax2.plot(total_volumes_arr, p2(total_volumes_arr), "k--", alpha=0.8, label=f"Trend (slope={z2[0]:.4g})")
    ax2.legend()

    plt.tight_layout()
    save_path2 = os.path.join(out_dir, "rmse_vs_total_volume.png")
    plt.savefig(save_path2)
    plt.close(fig2)
    # Logging handled by caller

