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

"""
ViT strong-scaling plots — latency and memory vs GPU count (separate 2D / 3D).

Reads CSV benchmark results from the results/ directory and produces
publication-quality line plots showing how inference/training
latency and memory scale across GPU counts at various image sizes.
Each dimensionality (2D, 3D) gets its own set of side-by-side panels.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = _SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Publication-quality rcParams  (NVIDIA-themed, matching ring-attention plots)
# ---------------------------------------------------------------------------
PUBRC: dict = {
    # --- Font ---
    "font.family": "sans-serif",
    "font.sans-serif": [
        "NVIDIA",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ],
    "font.size": 14,
    "mathtext.fontset": "dejavusans",
    # --- Axes ---
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    # --- Grid ---
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    # --- Ticks ---
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    # --- Legend ---
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    # --- Lines & markers ---
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    # --- Figure ---
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    # --- Saving ---
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

mpl.rcParams.update(PUBRC)

# Bold, high-contrast palette — every colour easily distinguishable
GPU_COLORS = {
    1: "#76B900",  # NVIDIA green
    2: "#0288D1",  # blue
    4: "#E65100",  # orange
    8: "#6A1B9A",  # purple
    16: "#D32F2F",  # red
}

GPU_MARKERS = {
    1: "o",
    2: "s",
    4: "D",
    8: "^",
    16: "v",
}

DIM_LABEL = {"2": "2D", "3": "3D"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Matches new single-file-per-size format only (with size suffix)
# e.g. benchmark_results_1bs_3d_FP32_16dp_1ddp_768-768px.csv
_FNAME_RE = re.compile(
    r"benchmark_results_(?P<bs>\d+)bs_(?P<dim>\d+)d_(?P<dtype>\w+)"
    r"_(?P<dp>\d+)dp_(?P<ddp>\d+)ddp_\d+-\d+px\.csv$"
)

_NUMERIC_COLS = {
    "Size (px)",
    "Global BS",
    "Local BS",
    "Params",
    "Fwd (s)",
    "Train (s)",
    "Inf. Mem (GB)",
    "Inf. (samp/s)",
    "Inf. (samp/s/gpu)",
    "Train Mem (GB)",
    "Train (samp/s)",
    "Train (samp/s/gpu)",
}


def _parse_filename(fname: str) -> dict | None:
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    return {
        "batch_size": int(m.group("bs")),
        "dim": m.group("dim"),
        "dtype": m.group("dtype"),
        "dp": int(m.group("dp")),
        "ddp": int(m.group("ddp")),
        "num_gpus": int(m.group("dp")) * int(m.group("ddp")),
    }


def load_results(results_dir: str | Path | None = None) -> pd.DataFrame:
    """Read all CSV files in *results_dir* into a single tidy DataFrame."""
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    frames = []

    for fpath in sorted(results_dir.glob("*.csv")):
        meta = _parse_filename(fpath.name)
        if meta is None:
            continue
        df = pd.read_csv(fpath)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        # Coerce OOM / non-numeric values to NaN
        for col in df.columns:
            if col in _NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Drop rows where all key metrics are NaN (pure OOM or empty rows)
        key_metrics = ["Fwd (s)", "Train (s)", "Inf. Mem (GB)", "Train Mem (GB)"]
        present = [c for c in key_metrics if c in df.columns]
        if present:
            df.dropna(subset=present, how="all", inplace=True)
        if df.empty:
            continue
        for k, v in meta.items():
            df[k] = v
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No matching CSVs in {results_dir}")

    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate: keep first occurrence per (dim, num_gpus, Size (px))
    combined.drop_duplicates(
        subset=["dim", "num_gpus", "Size (px)"], keep="first", inplace=True
    )
    combined.sort_values(["dim", "num_gpus", "Size (px)"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_series(ax, dim_df, col, *, label_fmt="{g} GPU{s}"):
    """Plot one line per GPU count on *ax*.  Returns handles for a shared legend."""
    handles = []
    for g in sorted(dim_df["num_gpus"].unique()):
        sub = dim_df[dim_df["num_gpus"] == g].sort_values("Size (px)")
        x = sub["Size (px)"].values.astype(float)
        y = sub[col].values.astype(float)
        mask = np.isfinite(y) & (y > 0)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue
        label = label_fmt.format(g=g, s="s" if g > 1 else "")
        (h,) = ax.plot(
            x,
            y,
            color=GPU_COLORS.get(g, "#808080"),
            marker=GPU_MARKERS.get(g, "o"),
            label=label,
            zorder=3,
        )
        handles.append(h)
    return handles


_DIM_SUPERSCRIPT = {"2": "\u00b2", "3": "\u00b3"}
_DIM_MIN_SIZE = {"2": 256, "3": 128}


def _fmt_speedup(sp: float) -> str:
    """Format a speedup value for bar annotation."""
    if sp >= 100:
        return f"{sp:.0f}×"
    if sp >= 10:
        return f"{sp:.1f}×"
    return f"{sp:.1f}×"


def _bar_latency_plot(ax, dim_df, lat_col, *, title, dim="2"):
    """Grouped bar chart: x = image resolution, bars = GPU counts, speedup on top.

    Each cluster of bars corresponds to one image resolution.  Within a
    cluster, each bar is one GPU count.  Missing bars where a GPU count
    has data at lower resolutions are drawn as hatched OOM placeholders.
    The speedup relative to the minimum GPU count is annotated above
    every non-baseline bar.
    """
    min_size = _DIM_MIN_SIZE.get(dim, 0)
    sup = _DIM_SUPERSCRIPT.get(dim, "")

    pivot = (
        dim_df[["Size (px)", "num_gpus", lat_col]]
        .dropna(subset=[lat_col])
        .pivot_table(
            index="Size (px)", columns="num_gpus", values=lat_col, aggfunc="first"
        )
    )
    pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if pivot.empty:
        return

    # Filter out smallest resolution(s)
    pivot = pivot.loc[pivot.index >= min_size]
    if pivot.empty:
        return

    sizes = pivot.index.tolist()
    gpus = sorted(pivot.columns.tolist())
    n_gpus = len(gpus)
    min_g = min(gpus)

    # For each GPU count, find the largest resolution with real data.
    # Resolutions beyond that are OOM.
    max_valid_size = {}
    for g in gpus:
        valid = pivot.index[pivot[g].notna() & (pivot[g] > 0)]
        max_valid_size[g] = valid.max() if len(valid) > 0 else -1

    group_width = 0.75
    bar_w = group_width / n_gpus
    x_base = np.arange(len(sizes))

    # We'll need the y-axis top for OOM bar height — do a first pass to find it
    all_vals = pivot.values[np.isfinite(pivot.values) & (pivot.values > 0)]
    y_top = all_vals.max() * 1.8 if len(all_vals) > 0 else 1.0

    for gi, g in enumerate(gpus):
        offset = (gi - n_gpus / 2 + 0.5) * bar_w
        color = GPU_COLORS.get(g, "#808080")
        label = f"{g} GPU{'s' if g > 1 else ''}"
        first_real = True
        first_oom = True

        for si, s in enumerate(sizes):
            xp = x_base[si] + offset
            val = pivot.loc[s, g] if g in pivot.columns else np.nan
            has_data = pd.notna(val) and val > 0

            if has_data:
                ax.bar(
                    xp,
                    val,
                    bar_w * 0.92,
                    color=color,
                    label=label if first_real else "",
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.6,
                )
                first_real = False

                # Speedup annotation (skip for baseline GPU count)
                if g != min_g and min_g in pivot.columns:
                    base_val = pivot.loc[s, min_g]
                    if pd.notna(base_val) and base_val > 0:
                        sp = base_val / val
                        ax.text(
                            xp,
                            val,
                            f" {_fmt_speedup(sp)}",
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                            fontweight="bold",
                            color="#333333",
                            rotation=90,
                        )

            elif s > max_valid_size[g] >= 0:
                # OOM: this GPU ran at lower resolutions but not here
                oom_label = "OOM" if first_oom else ""
                ax.bar(
                    xp,
                    y_top,
                    bar_w * 0.92,
                    color=color,
                    alpha=0.10,
                    hatch="//",
                    edgecolor=color,
                    linewidth=0.4,
                    zorder=2,
                    label=oom_label if first_oom and gi == 0 else "",
                )
                ax.text(
                    xp,
                    y_top * 0.45,
                    "OOM",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color=color,
                    alpha=0.7,
                    rotation=90,
                )
                first_oom = False

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{int(s)}{sup}" for s in sizes])
    ax.set_xlabel(f"Image Resolution (px{sup})")
    ax.set_ylabel("Latency (s)")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(top=y_top)
    ax.legend(fontsize=11, loc="upper left")


def _bar_memory_plot(ax, dim_df, mem_col, *, title, dim="2"):
    """Grouped bar chart for memory: x = resolution, bars = GPU counts.

    Annotations show memory reduction relative to the baseline (min GPU count).
    Horizontal reference lines mark common GPU memory capacities.
    Missing bars at higher resolutions are shown as hatched OOM placeholders.
    """
    min_size = _DIM_MIN_SIZE.get(dim, 0)
    sup = _DIM_SUPERSCRIPT.get(dim, "")

    pivot = (
        dim_df[["Size (px)", "num_gpus", mem_col]]
        .dropna(subset=[mem_col])
        .pivot_table(
            index="Size (px)", columns="num_gpus", values=mem_col, aggfunc="first"
        )
    )
    pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if pivot.empty:
        return

    pivot = pivot.loc[pivot.index >= min_size]
    if pivot.empty:
        return

    sizes = pivot.index.tolist()
    gpus = sorted(pivot.columns.tolist())
    n_gpus = len(gpus)
    min_g = min(gpus)

    max_valid_size = {}
    for g in gpus:
        valid = pivot.index[pivot[g].notna() & (pivot[g] > 0)]
        max_valid_size[g] = valid.max() if len(valid) > 0 else -1

    group_width = 0.75
    bar_w = group_width / n_gpus
    x_base = np.arange(len(sizes))

    all_vals = pivot.values[np.isfinite(pivot.values) & (pivot.values > 0)]
    y_data_max = all_vals.max() if len(all_vals) > 0 else 1.0

    # GPU capacity reference lines (drawn first so bars overlay them)
    gpu_capacities = [
        ("B200", 180, "#D32F2F"),
        ("H200", 141, "#E65100"),
        ("H100", 80, "#6A1B9A"),
        ("RTX 5090", 32, "#00695C"),
    ]
    y_top = y_data_max * 1.25
    cap_handles = []
    for gpu_name, cap_gb, cap_color in gpu_capacities:
        if cap_gb <= y_top * 1.1:
            ln = ax.axhline(
                cap_gb, ls=":", color=cap_color, lw=1.4, alpha=0.8, zorder=1
            )
            cap_handles.append((ln, f"{gpu_name} ({cap_gb} GB)"))
            y_top = max(y_top, cap_gb * 1.12)

    for gi, g in enumerate(gpus):
        offset = (gi - n_gpus / 2 + 0.5) * bar_w
        color = GPU_COLORS.get(g, "#808080")
        label = f"{g} GPU{'s' if g > 1 else ''}"
        first_real = True
        first_oom = True

        for si, s in enumerate(sizes):
            xp = x_base[si] + offset
            val = pivot.loc[s, g] if g in pivot.columns else np.nan
            has_data = pd.notna(val) and val > 0

            if has_data:
                ax.bar(
                    xp,
                    val,
                    bar_w * 0.92,
                    color=color,
                    label=label if first_real else "",
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.6,
                )
                first_real = False

                # Memory reduction annotation (skip for baseline GPU count)
                if g != min_g and min_g in pivot.columns:
                    base_val = pivot.loc[s, min_g]
                    if pd.notna(base_val) and base_val > 0:
                        ratio = base_val / val
                        ax.text(
                            xp,
                            val + y_top * 0.01,
                            f" {_fmt_speedup(ratio)}",
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                            fontweight="bold",
                            color="#333333",
                            rotation=90,
                        )

            elif s > max_valid_size[g] >= 0:
                oom_label = "OOM" if first_oom else ""
                ax.bar(
                    xp,
                    y_top,
                    bar_w * 0.92,
                    color=color,
                    alpha=0.10,
                    hatch="//",
                    edgecolor=color,
                    linewidth=0.4,
                    zorder=2,
                    label=oom_label if first_oom and gi == 0 else "",
                )
                ax.text(
                    xp,
                    y_top * 0.45,
                    "OOM",
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color=color,
                    alpha=0.7,
                    rotation=90,
                )
                first_oom = False

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{int(s)}{sup}" for s in sizes])
    ax.set_xlabel(f"Image Resolution (px{sup})")
    ax.set_ylabel("Peak Memory per GPU (GB)")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_ylim(bottom=0, top=y_top)

    # Combined legend: GPU counts + capacity lines
    main_legend = ax.legend(fontsize=11, loc="upper left")
    if cap_handles:
        ax.add_artist(main_legend)
        ax.legend(
            [h for h, _ in cap_handles],
            [l for _, l in cap_handles],
            fontsize=9,
            loc="center right",
            title="GPU Memory",
            title_fontsize=10,
        )


def _plot_ratio(ax, dim_df, col, *, label_fmt="{g} GPU{s}"):
    """Plot baseline/value ratio per GPU count.  Baseline = min GPU count."""
    min_g = int(dim_df["num_gpus"].min())
    base = (
        dim_df[dim_df["num_gpus"] == min_g]
        .drop_duplicates("Size (px)")
        .set_index("Size (px)")[col]
    )

    handles = []
    for g in sorted(dim_df["num_gpus"].unique()):
        sub = (
            dim_df[dim_df["num_gpus"] == g]
            .drop_duplicates("Size (px)")
            .set_index("Size (px)")
        )
        common = sorted(base.index.intersection(sub.index))
        x_vals, y_vals = [], []
        for s in common:
            bv, sv = float(base.loc[s]), float(sub.loc[s, col])
            if np.isfinite(bv) and np.isfinite(sv) and sv > 0:
                x_vals.append(s)
                y_vals.append(bv / sv)
        if not x_vals:
            continue

        label = label_fmt.format(g=g, s="s" if g > 1 else "")
        (h,) = ax.plot(
            x_vals,
            y_vals,
            color=GPU_COLORS.get(g, "#808080"),
            marker=GPU_MARKERS.get(g, "o"),
            label=label,
            zorder=3,
        )
        handles.append(h)

    ax.axhline(1.0, ls="--", color="grey", lw=1, zorder=0)
    return handles


def _shared_legend(fig, handles, *, ncol=None):
    """Place a single shared legend at the bottom of the figure."""
    if not handles:
        return
    labels = [h.get_label() for h in handles]
    if ncol is None:
        ncol = len(handles)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=True,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_results()
    print(f"Loaded {len(df)} rows from {DEFAULT_RESULTS_DIR}\n")
    print(
        df[
            [
                "dim",
                "Size (px)",
                "num_gpus",
                "Fwd (s)",
                "Train (s)",
                "Inf. Mem (GB)",
                "Train Mem (GB)",
            ]
        ].to_string(index=False)
    )
    print()

    dims = sorted(df["dim"].unique())

    for dim in dims:
        dim_df = df[df["dim"] == dim]
        tag = DIM_LABEL.get(dim, dim + "D")

        # -------------------------------------------------------------------
        # 1. Latency bar charts  (separate inference & training figures)
        # -------------------------------------------------------------------
        for lat_col, mode_label, mode_tag in [
            ("Fwd (s)", "Inference", "inference"),
            ("Train (s)", "Training", "training"),
        ]:
            fig, ax = plt.subplots(figsize=(14, 7))
            _bar_latency_plot(
                ax,
                dim_df,
                lat_col,
                dim=dim,
                title=f"ViT {tag} {mode_label} Latency — Strong Scaling",
            )
            fname = f"vit_{mode_tag}_latency_{tag.lower()}.png"
            fig.savefig(_SCRIPT_DIR / fname)
            print(f"Saved {fname}")

        # -------------------------------------------------------------------
        # 2. Memory bar charts  (separate inference & training figures)
        # -------------------------------------------------------------------
        for mem_col, mode_label, mode_tag in [
            ("Inf. Mem (GB)", "Inference", "inference"),
            ("Train Mem (GB)", "Training", "training"),
        ]:
            fig, ax = plt.subplots(figsize=(14, 7))
            _bar_memory_plot(
                ax,
                dim_df,
                mem_col,
                dim=dim,
                title=f"ViT {tag} {mode_label} Memory — Strong Scaling",
            )
            fname = f"vit_{mode_tag}_memory_{tag.lower()}.png"
            fig.savefig(_SCRIPT_DIR / fname)
            print(f"Saved {fname}")

        # -------------------------------------------------------------------
        # 3. Speedup  (inference | training)
        # -------------------------------------------------------------------
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

        _plot_ratio(ax_l, dim_df, "Fwd (s)")
        ax_l.set_title(f"Speedup — Inference")
        ax_l.set_ylabel("Speedup  (T₁ / Tₙ)")
        ax_l.set_xlabel("Image Size (px)")

        handles = _plot_ratio(ax_r, dim_df, "Train (s)")
        ax_r.set_title(f"Speedup — Training")
        ax_r.set_xlabel("Image Size (px)")

        fig.suptitle(
            f"ViT {tag} Speedup — Strong Scaling", fontsize=20, fontweight="bold"
        )
        _shared_legend(fig, handles)

        fname = f"vit_speedup_{tag.lower()}.png"
        fig.savefig(_SCRIPT_DIR / fname)
        print(f"Saved {fname}")

        # -------------------------------------------------------------------
        # 4. Memory reduction  (inference | training)
        # -------------------------------------------------------------------
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

        _plot_ratio(ax_l, dim_df, "Inf. Mem (GB)")
        ax_l.set_title(f"Memory Reduction — Inference")
        ax_l.set_ylabel("Memory Reduction  (M₁ / Mₙ)")
        ax_l.set_xlabel("Image Size (px)")

        handles = _plot_ratio(ax_r, dim_df, "Train Mem (GB)")
        ax_r.set_title(f"Memory Reduction — Training")
        ax_r.set_xlabel("Image Size (px)")

        fig.suptitle(
            f"ViT {tag} Memory Reduction — Strong Scaling",
            fontsize=20,
            fontweight="bold",
        )
        _shared_legend(fig, handles)

        fname = f"vit_memory_reduction_{tag.lower()}.png"
        fig.savefig(_SCRIPT_DIR / fname)
        print(f"Saved {fname}")

        # -------------------------------------------------------------------
        # 5. Polynomial memory fit  (inference | training)
        #    2D → quadratic (A·x² + B·x + C)
        #    3D → cubic     (A·x³ + B·x² + C·x + D)
        # -------------------------------------------------------------------
        sup = _DIM_SUPERSCRIPT.get(dim, "")
        fit_degree = int(dim)  # 2 for 2D, 3 for 3D

        # All unique measured sizes (unfiltered — fit uses all data)
        all_sizes_fit = sorted(
            set(int(s) for s in dim_df["Size (px)"].dropna().unique())
        )
        data_max = max(all_sizes_fit) if all_sizes_fit else 4096
        # Extend fit modestly beyond data for trend visualisation
        fit_extrap_max = int(data_max * 1.35)

        # Representative sizes for the printed estimate table
        est_candidates = sorted(
            set(
                [data_max]
                + [
                    s
                    for s in [256, 512, 1024, 2048, 3072, 4096]
                    if data_max // 2 <= s <= fit_extrap_max
                ]
            )
        )
        est_sizes = est_candidates[-3:] if len(est_candidates) > 3 else est_candidates

        fit_name = {2: "Quadratic", 3: "Cubic"}[fit_degree]
        fit_formula = {
            2: "A·x² + B·x + C",
            3: "A·x³ + B·x² + C·x + D",
        }[fit_degree]
        print(f"\n{'=' * 100}")
        print(f"{fit_name} fits ({tag}):  Memory (GB) = {fit_formula}")
        print(f"{'=' * 100}")

        fit_panels = [
            ("Inf. Mem (GB)", "Inference", "inference"),
            ("Train Mem (GB)", "Training", "training"),
        ]

        for mem_col, mode_label, fname_tag in fit_panels:
            fig, ax = plt.subplots(figsize=(14, 7))

            est_hdr = "  ".join(f"{'Est @ ' + str(s):>12s}" for s in est_sizes)
            print(f"\n--- {tag} {mode_label} Memory ---")
            if fit_degree == 3:
                print(
                    f"  {'GPUs':>4s}  {'A (GB/px³)':>14s}  {'B (GB/px²)':>14s}  "
                    f"{'C (GB/px)':>12s}  {'D (GB)':>10s}  "
                    f"{'R²':>8s}  {est_hdr}"
                )
            else:
                print(
                    f"  {'GPUs':>4s}  {'A (GB/px²)':>14s}  {'B (GB/px)':>12s}  "
                    f"{'C (GB)':>10s}  {'R²':>8s}  {est_hdr}"
                )

            panel_handles = []
            for g in sorted(dim_df["num_gpus"].unique()):
                sub = dim_df[dim_df["num_gpus"] == g][["Size (px)", mem_col]].dropna()
                x = sub["Size (px)"].values.astype(float)
                y = sub[mem_col].values.astype(float)
                if len(x) < fit_degree + 1:
                    continue

                coeffs = np.polyfit(x, y, fit_degree)
                poly = np.poly1d(coeffs)
                y_pred = poly(x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                est = {res: poly(res) for res in est_sizes}
                est_vals = "  ".join(f"{est[s]:>12.3f}" for s in est_sizes)
                if fit_degree == 3:
                    print(
                        f"  {g:>4d}  {coeffs[0]:>14.4e}  {coeffs[1]:>14.4e}  "
                        f"{coeffs[2]:>12.6f}  {coeffs[3]:>10.4f}  "
                        f"{r2:>8.4f}  {est_vals}"
                    )
                else:
                    print(
                        f"  {g:>4d}  {coeffs[0]:>14.4e}  {coeffs[1]:>12.6f}  "
                        f"{coeffs[2]:>10.4f}  {r2:>8.4f}  {est_vals}"
                    )

                color = GPU_COLORS.get(g, "#808080")
                marker = GPU_MARKERS.get(g, "o")
                ax.scatter(x, y, color=color, marker=marker, s=50, zorder=4)

                x_fit = np.linspace(x.min(), fit_extrap_max, 200)
                lbl = f"{g} GPU{'s' if g > 1 else ''}  (R²={r2:.4f})"
                (h,) = ax.plot(
                    x_fit, poly(x_fit), color=color, lw=2.0, label=lbl, zorder=3
                )
                panel_handles.append(h)

            # GPU memory capacity reference lines
            gpu_capacities = [
                ("B200", 180, "#D32F2F"),
                ("H200", 141, "#E65100"),
                ("H100", 80, "#6A1B9A"),
                ("RTX 5090", 32, "#00695C"),
            ]
            y_max = 200
            cap_handles = []
            for gpu_name, cap_gb, cap_color in gpu_capacities:
                if cap_gb <= y_max:
                    ln = ax.axhline(
                        cap_gb, ls=":", color=cap_color, lw=1.4, alpha=0.8, zorder=1
                    )
                    cap_handles.append((ln, f"{gpu_name} ({cap_gb} GB)"))
            ax.set_ylim(bottom=0, top=y_max)

            # Sensible x-axis: ticks at measured sizes, range matched to data
            ax.set_xlim(left=min(all_sizes_fit) * 0.85, right=fit_extrap_max * 1.02)
            ax.set_xticks(all_sizes_fit)
            ax.set_xticklabels(
                [str(s) for s in all_sizes_fit], rotation=45, ha="right", fontsize=10
            )
            ax.set_xlabel("Side Length (px)")
            ax.set_ylabel("Peak Memory per GPU (GB)")

            # Combined legend: fit curves + GPU capacity lines
            if cap_handles:
                all_h = panel_handles + [h for h, _ in cap_handles]
                all_l = [h.get_label() for h in panel_handles] + [
                    l for _, l in cap_handles
                ]
                ax.legend(all_h, all_l, fontsize=10, loc="upper left")
            else:
                ax.legend(fontsize=10, loc="upper left")

            ax.set_title(
                f"ViT {tag} {mode_label} Memory — {fit_name} Fit & Extrapolation",
                fontsize=18,
                fontweight="bold",
            )

            fname = f"vit_memory_fit_{fname_tag}_{tag.lower()}.png"
            fig.savefig(_SCRIPT_DIR / fname)
            print(f"  → Saved {fname}")

    plt.show()
