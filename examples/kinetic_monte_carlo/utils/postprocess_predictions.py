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


def _time_to_count(times: np.ndarray, threshold: int) -> float:
    """First time at which the particle count crosses ``threshold``.

    Parameters
    ----------
    times : (N,) array of per-event birth times in monotonically increasing
        order. ``times[k]`` is the time at which the ``(k + 1)``-th particle
        was created.
    threshold : particle count to reach.

    Returns
    -------
    Time at which the count first reaches ``threshold``, or ``NaN`` if the
    threshold is never reached over the simulation horizon.
    """
    if times.size < threshold or threshold <= 0:
        return float("nan")
    # The kth event makes the count equal to k+1. The first time
    # count >= threshold is therefore times[threshold - 1].
    return float(times[threshold - 1])


def _adaptive_bins(n_samples: int) -> int:
    """Square-root rule clipped to a sensible visual range."""
    return max(8, min(40, int(round(np.sqrt(max(n_samples, 1))))))


def time_to_count_histogram(
    files_dir: str | Path,
    output: str = "time_to_count_histogram.png",
    particle_threshold: int = 50,
    dpi: int = 120,
) -> None:
    """Compute the time to reach N particles for every rollout and plot a histogram.

    Recursively scans ``files_dir`` for ``.pth`` files produced by
    ``generate.py``. For each file, the time to reach ``particle_threshold``
    particles is the time at which the rollout's particle count first reaches
    that number (``NaN`` when the threshold is never reached). Predicted and
    ground-truth distributions are plotted as overlaid transparent histograms
    with vertical lines marking their means.
    """
    files_dir = Path(files_dir)
    if not files_dir.exists():
        raise FileNotFoundError(f"Directory not found: {files_dir}")
    files = sorted(files_dir.rglob("*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files found under {files_dir}")
    print(f"Found {len(files)} rollout file(s) under {files_dir}")

    ttc_pred_list: list[float] = []
    ttc_gt_list: list[float] = []
    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        # times_pred is (B, K) NaN-padded with companion (B,) n_pred count.
        times_pred = blob["times_pred"].numpy()
        n_pred = blob["n_pred"].numpy()
        times_gt = blob["times_gt"].numpy()

        ttc_g = _time_to_count(times_gt, particle_threshold)
        ttc_gt_list.append(ttc_g)
        # One time-to-count value per ensemble member.
        ttcs_p = [
            _time_to_count(times_pred[b, : int(n_pred[b])], particle_threshold)
            for b in range(times_pred.shape[0])
        ]
        ttc_pred_list.extend(ttcs_p)
        ttcs_p_finite = [v for v in ttcs_p if not np.isnan(v)]
        mean_str = f"{np.mean(ttcs_p_finite):.3e}" if ttcs_p_finite else "  -- "
        print(
            f"  {path.relative_to(files_dir)}: ensemble={times_pred.shape[0]} "
            f"| n_gt={times_gt.size} | mean(n_pred)={float(n_pred.mean()):.1f} "
            f"| ttc_gt={ttc_g:.3e} | ttc_pred(mean over ens)={mean_str}"
        )

    ttc_pred = np.array(ttc_pred_list, dtype=np.float64)
    ttc_gt = np.array(ttc_gt_list, dtype=np.float64)
    pred_finite = ttc_pred[~np.isnan(ttc_pred)]
    gt_finite = ttc_gt[~np.isnan(ttc_gt)]

    if pred_finite.size == 0 and gt_finite.size == 0:
        raise RuntimeError(
            f"No simulation reached {particle_threshold} particles in either "
            "pred or GT; consider lowering --particle-threshold."
        )

    pred_mean = float(pred_finite.mean()) if pred_finite.size else float("nan")
    gt_mean = float(gt_finite.mean()) if gt_finite.size else float("nan")
    n_bins = _adaptive_bins(max(pred_finite.size, gt_finite.size))

    print(
        f"\nSummary:\n"
        f"  pred: {pred_finite.size}/{ttc_pred.size} reached threshold | "
        f"mean time = {pred_mean:.3e}\n"
        f"  gt:   {gt_finite.size}/{ttc_gt.size} reached threshold | "
        f"mean time = {gt_mean:.3e}\n"
        f"  bins: {n_bins}"
    )

    combined = np.concatenate([gt_finite, pred_finite])
    edges = np.histogram_bin_edges(combined, bins=n_bins)

    color_gt = "tab:blue"
    color_pred = "tab:orange"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    if gt_finite.size:
        ax.hist(
            gt_finite,
            bins=edges,
            color=color_gt,
            alpha=0.5,
            label=f"Ground truth (n={gt_finite.size})",
        )
    if pred_finite.size:
        ax.hist(
            pred_finite,
            bins=edges,
            color=color_pred,
            alpha=0.5,
            label=f"Prediction (n={pred_finite.size})",
        )
    if not np.isnan(gt_mean):
        ax.axvline(gt_mean, color=color_gt, lw=2.0, label=f"GT mean = {gt_mean:.2e}")
    if not np.isnan(pred_mean):
        ax.axvline(
            pred_mean, color=color_pred, lw=2.0, label=f"Pred mean = {pred_mean:.2e}"
        )

    ax.set_xlabel(f"Time to reach {particle_threshold} particles")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of time to reach {particle_threshold} particles")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    summary = (
        f"Threshold reached\n"
        f"  GT:   {gt_finite.size} / {ttc_gt.size}\n"
        f"  Pred: {pred_finite.size} / {ttc_pred.size}"
    )
    ax.text(
        0.98,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    fig.tight_layout()

    output_path = Path(output)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved histogram to {output_path}")


_MODES = {
    "time_to_count": time_to_count_histogram,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Postprocess one or more rollouts produced by generate.py."
    )
    p.add_argument(
        "--files",
        required=True,
        help="Directory containing .pth rollout files (searched recursively).",
    )
    p.add_argument(
        "--mode",
        default="time_to_count",
        choices=sorted(_MODES),
        help="Postprocessing mode [default: time_to_count]",
    )
    p.add_argument(
        "--output",
        default="time_to_count_histogram.png",
        help="Output figure path [default: time_to_count_histogram.png]",
    )
    p.add_argument(
        "--particle-threshold",
        type=int,
        default=50,
        help="Number of particles to reach [default: 50]",
    )
    p.add_argument("--dpi", type=int, default=120, help="Figure DPI [default: 120]")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "time_to_count":
        time_to_count_histogram(
            files_dir=args.files,
            output=args.output,
            particle_threshold=args.particle_threshold,
            dpi=args.dpi,
        )
    else:
        raise ValueError(f"Unknown --mode {args.mode!r}")
