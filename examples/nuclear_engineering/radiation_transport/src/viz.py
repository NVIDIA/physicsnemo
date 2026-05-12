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

from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

__all__ = ["plot_flux_panels", "plot_qoi_true_vs_pred"]


def plot_flux_panels(
    coordinates: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Union[str, Path],
    log_flux: bool = False,
    figsize: Tuple[int, int] = (16, 5),
    dpi: int = 150,
) -> Path:
    """Render a 3-panel figure: target | prediction | absolute error."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = target.flatten()
    prediction = prediction.flatten()
    error = np.abs(prediction - target)

    x, y = coordinates[:, 0], coordinates[:, 1]
    x_pad = (x.max() - x.min()) * 0.01
    y_pad = (y.max() - y.min()) * 0.01
    xlim = (x.min() - x_pad, x.max() + x_pad)
    ylim = (y.min() - y_pad, y.max() + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    flux_vmin = min(target.min(), prediction.min())
    flux_vmax = max(target.max(), prediction.max())
    flux_norm = None
    if log_flux:
        positive_flux = np.concatenate(
            [target[target > 0.0], prediction[prediction > 0.0]]
        )
        if positive_flux.size:
            flux_vmin = float(positive_flux.min())
            flux_vmax = float(positive_flux.max())
            if flux_vmin == flux_vmax:
                flux_vmax = flux_vmin * 1.01
            flux_norm = LogNorm(vmin=flux_vmin, vmax=flux_vmax)
        else:
            log_flux = False
    cmap_flux = plt.get_cmap("viridis")
    cmap_err = plt.get_cmap("hot")

    for ax, label, vals, cmap, vmin, vmax, norm in (
        (axes[0], "Target", target, cmap_flux, flux_vmin, flux_vmax, flux_norm),
        (
            axes[1],
            "Prediction",
            prediction,
            cmap_flux,
            flux_vmin,
            flux_vmax,
            flux_norm,
        ),
        (axes[2], "Absolute Error", error, cmap_err, 0.0, float(error.max()), None),
    ):
        plot_vals = np.clip(vals, flux_vmin, None) if norm is not None else vals
        sc = ax.scatter(
            x,
            y,
            c=plot_vals,
            cmap=cmap,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
            s=1,
        )
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{label} (log)" if norm is not None else label)
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_qoi_true_vs_pred(
    per_sample_qoi: list[Dict[str, Dict[str, float]]],
    output_path: Union[str, Path],
    dpi: int = 150,
) -> Path:
    """Scatter predicted vs ground-truth QoI values for each component.

    Takes the same per-sample QoI list that ``aggregate_qoi`` consumes; the
    per-component arrays are flattened inline rather than via a separate
    collector.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve first-seen order of component names across samples.
    component_names: list[str] = []
    for sample in per_sample_qoi:
        for name in sample:
            if name not in component_names:
                component_names.append(name)

    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in component_names:
        target_vals: list[float] = []
        pred_vals: list[float] = []
        for sample in per_sample_qoi:
            entry = sample.get(name)
            if entry is None:
                continue
            target_vals.append(entry["ground_truth"])
            pred_vals.append(entry["predicted"])
        if target_vals:
            series[name] = (np.array(target_vals), np.array(pred_vals))

    items = list(series.items())
    if not items:
        plt.close(plt.figure())
        return output_path

    ncols = min(len(items), 3)
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), dpi=dpi, squeeze=False
    )

    for ax, (name, (target, prediction)) in zip(axes.flat, items):
        lo = float(min(target.min(), prediction.min()))
        hi = float(max(target.max(), prediction.max()))
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1e-12)
            lo -= pad
            hi += pad

        ax.scatter(target, prediction, s=18, alpha=0.75)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y = x")
        ax.set_title(name)
        ax.set_xlabel("Ground truth QoI")
        ax.set_ylabel("Predicted QoI")
        ax.set_aspect("equal")
        ax.legend(loc="best")

    for ax in axes.flat[len(items) :]:
        ax.axis("off")

    fig.suptitle("QoI predicted vs. ground truth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
