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

"""ShiftSUV active-learning summary plot.

Compares UQ-guided and class-balanced random sampling on the
out-of-distribution ShiftSUV target, with two seeds per method, against the
full-data asymptote (training on all 1728 SE+SF samples).

Three panels:
  1.  Drag R² vs number of training samples (per-seed lines, mean, min-max
      envelope, dashed asymptote).
  2.  Field MSE vs number of training samples (log y, same overlays).
  3.  Label efficiency: number of labels each method requires to first reach
      X % of the full-data R² asymptote (X ∈ {80, 85, 90, 95, 100}).

The metrics JSON files have appended duplicate entries from the chain-resume
bug (lex-sort of ``checkpoint_round_*`` -> always resumed from round 9).  We
de-duplicate by ``step`` keeping the *latest* occurrence per step, which both
removes duplicates and uses the most recent model state for each AL round.

Usage::

    python plot_al_shiftsuv_summary.py -o plots/al/shiftsuv_summary.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Defaults — locations of the four AL chains and the full-data reference.    #
# --------------------------------------------------------------------------- #
BASE = Path(
    "runs/geotransolver/surface"
)
DEFAULT_RUNS = {
    "uq_s42":  BASE / "al_shift_suv_uq_noreplay"          / "joint_uq"               / "validation_metrics.json",
    "uq_s123": BASE / "al_shift_suv_uq_noreplay_seed123"  / "joint_uq"               / "validation_metrics.json",
    "bal_s42":  BASE / "al_shift_suv_balanced_noreplay"        / "class_balanced_random" / "validation_metrics.json",
    "bal_s123": BASE / "al_shift_suv_balanced_noreplay_seed123" / "class_balanced_random" / "validation_metrics.json",
    "full":    BASE / "al_shift_suv_fulldata_noreplay"    / "class_balanced_random"  / "validation_metrics.json",
}

# Visual styling.
METHOD_COLORS = {"UQ": "#1f77b4", "BAL": "#d62728"}
METHOD_MARKERS = {"UQ": "o", "BAL": "s"}
SEED_STYLES = {42: "-", 123: "--"}
FULL_COLOR = "#2ca02c"


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def load_dedup(path: Path) -> list[dict]:
    """Load metrics JSON and dedup by ``step`` keeping the latest entry."""
    raw = json.load(open(path))
    by_step: dict[int, dict] = {}
    for e in raw:
        by_step[e["step"]] = e   # last write wins -> latest occurrence per step
    return [by_step[k] for k in sorted(by_step.keys())]


def extract(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(n_train, values)`` arrays for a single metric key."""
    pairs = [(r["n_train"], r.get(key)) for r in records if r.get(key) is not None]
    if not pairs:
        return np.array([]), np.array([])
    n, v = zip(*pairs)
    return np.array(n), np.array(v, dtype=float)


def common_grid(
    chains: list[list[dict]],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align two chains on their shared ``n_train`` grid and return (n, vals[seeds, ...])."""
    # Use intersection of n_train values across seeds so per-seed comparisons
    # are apples-to-apples even if one chain advanced further.
    n_sets = []
    for c in chains:
        n, _ = extract(c, key)
        n_sets.append(set(int(x) for x in n))
    common = sorted(set.intersection(*n_sets))
    if not common:
        return np.array([]), np.empty((len(chains), 0))
    vals = np.full((len(chains), len(common)), np.nan)
    for i, c in enumerate(chains):
        n, v = extract(c, key)
        idx = {int(x): j for j, x in enumerate(n)}
        for j, x in enumerate(common):
            if x in idx:
                vals[i, j] = v[idx[x]]
    return np.array(common), vals


# --------------------------------------------------------------------------- #
# Plot helpers                                                                #
# --------------------------------------------------------------------------- #
def _plot_method_with_band(
    ax,
    n: np.ndarray,
    vals: np.ndarray,
    color: str,
    marker: str,
    label_prefix: str,
    seeds: list[int],
) -> None:
    """Plot per-seed lines + min-max envelope + mean curve."""
    if n.size == 0:
        return
    # Per-seed thin lines (linestyle distinguishes seeds).
    for i, s in enumerate(seeds):
        ls = SEED_STYLES.get(s, ":")
        ax.plot(
            n, vals[i],
            color=color, ls=ls, lw=1.2, alpha=0.55,
            marker=marker, ms=4, mfc="none",
            label=f"{label_prefix} seed {s}",
            zorder=2,
        )
    # Min-max envelope (n=2 seeds -> ±σ is degenerate; range is honest).
    lo = np.nanmin(vals, axis=0)
    hi = np.nanmax(vals, axis=0)
    ax.fill_between(n, lo, hi, color=color, alpha=0.18, zorder=1)
    # Mean line, thick.
    mean = np.nanmean(vals, axis=0)
    ax.plot(
        n, mean,
        color=color, ls="-", lw=2.5,
        marker=marker, ms=7,
        label=f"{label_prefix} (mean)",
        zorder=3,
    )


def _draw_asymptote(ax, value: float, label: str) -> None:
    ax.axhline(
        value, color=FULL_COLOR, ls="--", lw=2, alpha=0.85,
        label=label, zorder=0,
    )


def _labels_to_reach(n: np.ndarray, vals: np.ndarray, target: float) -> float | None:
    """First n where vals >= target (mean across seeds).  None if never."""
    mean = np.nanmean(vals, axis=0)
    for i, v in enumerate(mean):
        if v >= target:
            return float(n[i])
    return None


def _labels_to_drop_below(n: np.ndarray, vals: np.ndarray, target: float) -> float | None:
    """First n where vals <= target (mean across seeds).  None if never."""
    mean = np.nanmean(vals, axis=0)
    for i, v in enumerate(mean):
        if v <= target:
            return float(n[i])
    return None


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uq-s42",  default=DEFAULT_RUNS["uq_s42"])
    parser.add_argument("--uq-s123", default=DEFAULT_RUNS["uq_s123"])
    parser.add_argument("--bal-s42", default=DEFAULT_RUNS["bal_s42"])
    parser.add_argument("--bal-s123", default=DEFAULT_RUNS["bal_s123"])
    parser.add_argument("--full",     default=DEFAULT_RUNS["full"])
    parser.add_argument(
        "-o", "--output", default="plots/al/shiftsuv_summary.png",
    )
    parser.add_argument(
        "--title", default="ShiftSUV (OOD): Active Learning vs Full-Data Asymptote",
    )
    args = parser.parse_args()

    uq = [load_dedup(Path(args.uq_s42)), load_dedup(Path(args.uq_s123))]
    bal = [load_dedup(Path(args.bal_s42)), load_dedup(Path(args.bal_s123))]
    full = load_dedup(Path(args.full))

    # Asymptote = best (max R²) and best (min field_mse) observed during the
    # full-data run (effectively epochs 20, 40, 60, 80, 100, ... up to where
    # the chain stopped).  Use min-fmse / max-R² as the "achievable ceiling".
    _, full_r2_vals  = extract(full, "drag_r2")
    _, full_fm_vals  = extract(full, "field_mse")
    full_r2 = float(np.nanmax(full_r2_vals)) if full_r2_vals.size else float("nan")
    full_fm = float(np.nanmin(full_fm_vals)) if full_fm_vals.size else float("nan")
    full_n  = int(full[-1]["n_train"]) if full else 0

    # Common grids for each method.
    n_r2_uq,  vals_r2_uq  = common_grid(uq,  "drag_r2")
    n_r2_bal, vals_r2_bal = common_grid(bal, "drag_r2")
    n_fm_uq,  vals_fm_uq  = common_grid(uq,  "field_mse")
    n_fm_bal, vals_fm_bal = common_grid(bal, "field_mse")

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    # ---- Panel 1: R² ----
    ax = axes[0]
    _plot_method_with_band(
        ax, n_r2_uq, vals_r2_uq,
        METHOD_COLORS["UQ"], METHOD_MARKERS["UQ"], "UQ", [42, 123],
    )
    _plot_method_with_band(
        ax, n_r2_bal, vals_r2_bal,
        METHOD_COLORS["BAL"], METHOD_MARKERS["BAL"], "Class-bal. random", [42, 123],
    )
    _draw_asymptote(ax, full_r2, f"Full data (n={full_n}): R²={full_r2:.3f}")
    # Clip y to a useful range — early-round R² can dip below -40 on OOD data
    # which crushes the interesting 0-1 detail.  Set lower bound at 0 (or just
    # below) so the asymptote and approach curves are readable.
    ax.set_ylim(bottom=-0.05, top=max(1.0, full_r2 * 1.05))
    # Also start x at the first round with R² > -0.5 to skip the warmup dip.
    def _first_useful_n(n_arr, v_arr, floor=-0.5):
        if n_arr.size == 0:
            return 0
        mean = np.nanmean(v_arr, axis=0)
        good = np.where(mean > floor)[0]
        return int(n_arr[good[0]]) if good.size else int(n_arr[0])
    x_lo = min(
        _first_useful_n(n_r2_uq, vals_r2_uq),
        _first_useful_n(n_r2_bal, vals_r2_bal),
    )
    ax.set_xlim(left=max(0, x_lo - 20))
    ax.set_xlabel("Labeled SUV samples (n_train)", fontsize=12)
    ax.set_ylabel("Drag R² on held-out SUVs", fontsize=12)
    ax.set_title("Drag R² vs labeling budget", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=1)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    # ---- Panel 2: Field MSE ----
    ax = axes[1]
    _plot_method_with_band(
        ax, n_fm_uq, vals_fm_uq,
        METHOD_COLORS["UQ"], METHOD_MARKERS["UQ"], "UQ", [42, 123],
    )
    _plot_method_with_band(
        ax, n_fm_bal, vals_fm_bal,
        METHOD_COLORS["BAL"], METHOD_MARKERS["BAL"], "Class-bal. random", [42, 123],
    )
    _draw_asymptote(ax, full_fm, f"Full data (n={full_n}): MSE={full_fm:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("Labeled SUV samples (n_train)", fontsize=12)
    ax.set_ylabel("Surface-field MSE on held-out SUVs", fontsize=12)
    ax.set_title("Surface-field MSE vs labeling budget", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=1)
    ax.grid(True, alpha=0.2, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    # ---- Panel 3: label efficiency ----
    # Use only thresholds *both* methods have actually crossed.  Start from
    # 80/85/90/95 % and add a final tick at the integer floor of the lower
    # of the two peak percentages reached (so neither bar is extrapolated).
    ax = axes[2]
    fixed = [0.80, 0.85, 0.90, 0.95]
    if (
        n_r2_uq.size and n_r2_bal.size and full_r2 > 0
    ):
        uq_peak_pct  = float(np.nanmax(np.nanmean(vals_r2_uq,  axis=0))) / full_r2
        bal_peak_pct = float(np.nanmax(np.nanmean(vals_r2_bal, axis=0))) / full_r2
        min_peak_pct = min(uq_peak_pct, bal_peak_pct)
        # Floor to nearest integer percent so both methods are guaranteed to
        # have reached it.  Only add if it's strictly above 95 % to avoid a
        # duplicate column.
        peak_floor = int(np.floor(min_peak_pct * 100)) / 100.0
        thresholds = fixed + ([peak_floor] if peak_floor > 0.95 else [])
    else:
        thresholds = fixed
    target_levels = [full_r2 * t for t in thresholds]
    width = 0.35
    x = np.arange(len(thresholds))

    def _bar(method_n, method_vals, color, label, offset):
        ns = [
            _labels_to_reach(method_n, method_vals, lvl)
            for lvl in target_levels
        ]
        plotted = [v for v in ns]
        bars = ax.bar(
            x + offset, plotted, width=width, color=color,
            edgecolor="white", linewidth=0.8, label=label,
            alpha=0.85,
        )
        for b, v in zip(bars, ns):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + (max(plotted) if plotted else 1) * 0.02,
                f"{int(v)}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    _bar(n_r2_uq,  vals_r2_uq,  METHOD_COLORS["UQ"],  "UQ", -width / 2)
    _bar(n_r2_bal, vals_r2_bal, METHOD_COLORS["BAL"], "Class-bal. random", +width / 2)
    ax.axhline(full_n, color=FULL_COLOR, ls="--", lw=2, alpha=0.85,
               label=f"Full data ({full_n} labels)", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(t * 100)}% of full-data R²" for t in thresholds],
        rotation=45, ha="right",
    )
    ax.set_ylabel("Labels needed (n_train)", fontsize=12)
    ax.set_title("Label efficiency (R²): budget to reach asymptote", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    # ---- Panel 4: MSE label efficiency ----
    # MSE is "lower = better", so we frame thresholds as "X× full-data MSE".
    # Symmetric construction to the R² panel: ceil to nearest 0.01 above the
    # higher (worse) of the two peak ratios so both methods are guaranteed
    # to have actually crossed the threshold (no extrapolation).
    ax = axes[3]
    fixed_mse = [2.0, 1.5, 1.25, 1.10]
    uq_peak_ratio = bal_peak_ratio = None
    if n_fm_uq.size and n_fm_bal.size and full_fm > 0:
        uq_peak_ratio  = float(np.nanmin(np.nanmean(vals_fm_uq,  axis=0))) / full_fm
        bal_peak_ratio = float(np.nanmin(np.nanmean(vals_fm_bal, axis=0))) / full_fm
        max_peak_ratio = max(uq_peak_ratio, bal_peak_ratio)
        ratio_ceil = math.ceil(max_peak_ratio * 100) / 100.0
        thresholds_mse = fixed_mse + (
            [ratio_ceil] if ratio_ceil < min(fixed_mse) else []
        )
    else:
        thresholds_mse = fixed_mse
    target_mse_levels = [full_fm * r for r in thresholds_mse]
    x_mse = np.arange(len(thresholds_mse))

    def _bar_mse(method_n, method_vals, color, label, offset):
        ns = [
            _labels_to_drop_below(method_n, method_vals, lvl)
            for lvl in target_mse_levels
        ]
        plotted = [v for v in ns]
        bars = ax.bar(
            x_mse + offset, plotted, width=width, color=color,
            edgecolor="white", linewidth=0.8, label=label,
            alpha=0.85,
        )
        for b, v in zip(bars, ns):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + (max(plotted) if plotted else 1) * 0.02,
                f"{int(v)}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    _bar_mse(n_fm_uq,  vals_fm_uq,  METHOD_COLORS["UQ"],  "UQ", -width / 2)
    _bar_mse(n_fm_bal, vals_fm_bal, METHOD_COLORS["BAL"], "Class-bal. random", +width / 2)
    ax.axhline(full_n, color=FULL_COLOR, ls="--", lw=2, alpha=0.85,
               label=f"Full data ({full_n} labels)", zorder=0)
    ax.set_xticks(x_mse)
    # Frame as "X% above full-data MSE" so smaller % = closer to asymptote =
    # better, matching the visual intuition of the R² panel (where higher % =
    # closer to asymptote).
    ax.set_xticklabels(
        [f"{int(round((r - 1) * 100))}% above full-data MSE" for r in thresholds_mse],
        rotation=45, ha="right",
    )
    ax.set_ylabel("Labels needed (n_train)", fontsize=12)
    ax.set_title("Label efficiency (MSE): budget to reach asymptote", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    fig.suptitle(args.title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=3)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")

    # ---- Print numeric summary ----
    print("\n=== Numerical summary ===")
    print(
        f"Full-data ceiling: R²={full_r2:.4f}, fmse={full_fm:.5f}  "
        f"(n={full_n})"
    )
    for name, n, v, key in [
        ("UQ",  n_r2_uq,  vals_r2_uq,  "drag_r2"),
        ("BAL", n_r2_bal, vals_r2_bal, "drag_r2"),
    ]:
        if n.size == 0:
            continue
        last_mean = float(np.nanmean(v[:, -1]))
        last_n = int(n[-1])
        gap = (last_mean / full_r2) * 100 if full_r2 > 0 else 0
        print(
            f"{name:>4}  @ n={last_n:4d}  mean R²={last_mean:.4f}  "
            f"= {gap:5.1f}% of full-data ceiling  "
            f"(per-seed: {', '.join(f'{x:.3f}' for x in v[:, -1])})"
        )
    print("\nLabels needed (mean curve) to reach X% of full-data R²:")
    print(f"{'%target':>8}  {'UQ':>10}  {'BAL':>10}")
    for t, lvl in zip(thresholds, target_levels):
        nu = _labels_to_reach(n_r2_uq, vals_r2_uq, lvl)
        nb = _labels_to_reach(n_r2_bal, vals_r2_bal, lvl)
        print(
            f"{int(t*100):>7}%  "
            f"{(f'{int(nu)}' if nu is not None else '—'):>10}  "
            f"{(f'{int(nb)}' if nb is not None else '—'):>10}"
        )
    print(
        f"\n(Peak reached so far: UQ {uq_peak_pct*100:.1f}%, "
        f"BAL {bal_peak_pct*100:.1f}% of full-data R²)"
    )

    print("\nLabels needed (mean curve) to come within X% above full-data MSE:")
    print(f"{'%above':>8}  {'UQ':>10}  {'BAL':>10}")
    for r, lvl in zip(thresholds_mse, target_mse_levels):
        nu = _labels_to_drop_below(n_fm_uq, vals_fm_uq, lvl)
        nb = _labels_to_drop_below(n_fm_bal, vals_fm_bal, lvl)
        print(
            f"{int(round((r - 1) * 100)):>7}%  "
            f"{(f'{int(nu)}' if nu is not None else '—'):>10}  "
            f"{(f'{int(nb)}' if nb is not None else '—'):>10}"
        )
    if uq_peak_ratio is not None:
        print(
            f"\n(Best gap so far: UQ {(uq_peak_ratio - 1) * 100:+.1f}%, "
            f"BAL {(bal_peak_ratio - 1) * 100:+.1f}% above full-data MSE)"
        )

    plt.close(fig)


if __name__ == "__main__":
    main()
