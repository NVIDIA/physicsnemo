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
@Author: clement etienam
"""

from pathlib import Path
import math
import pandas as pd
from omegaconf import DictConfig
import numpy as np
import hydra
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from physicsnemo.distributed import DistributedManager


def plot_all_metrics(cfg, mlruns_root: Path | str | None = None):
    """
    Search for the single 'metrics' folder under the given mlruns root,
    read all metric files, and plot one subplot per file.

    Parameters
    ----------
    cfg : object
        Config object that contains cfg.custom.fno_type ("PINO" or "FNO").
    mlruns_root : Path or str, optional
        Root directory that contains the 'mlruns' folder.
        If None, assumes this script is located inside 'src' and mlruns is at src/mlruns.
    """

    # ── reader ────────────────────────────────────────────────────────────────
    def read_metric_file(fp: Path):
        """Read one MLflow metric file (timestamp, value, step)."""
        try:
            df = pd.read_csv(
                fp,
                sep=r"\s+",
                header=None,
                names=["timestamp", "value", "step"],
                engine="python",
            )
            df = df[
                pd.to_numeric(df["step"], errors="coerce").notna()
                & pd.to_numeric(df["value"], errors="coerce").notna()
            ]
            if df.empty:
                return None
            return df.sort_values("step")[["step", "value"]]
        except Exception:
            return None

    # ── elegant style ─────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "axes.edgecolor": "#2C3E50",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.facecolor": "white",
        }
    )

    # palette
    CLR_LINE = "#2471A3"  # steel blue
    CLR_FILL = "#AED6F1"  # light steel blue
    CLR_GRID = "#BDC3C7"  # soft grey
    CLR_ACCENT = "#C0392B"  # deep red — for min/max markers

    # ── choose model folder based on cfg ─────────────────────────────────────
    if cfg.custom.model_type == "FNO":
        model_dir = Path(
            "../MODELS/PINO" if cfg.custom.fno_type == "PINO" else "../MODELS/FNO"
        )
    else:
        model_dir = Path(
            "../MODELS/PI-TRANSOLVER"
            if cfg.custom.fno_type == "PINO"
            else "../MODELS/TRANSOLVER"
        )

    # ── locate mlruns metrics folder ─────────────────────────────────────────
    if mlruns_root is None:
        root = Path(__file__).resolve().parent
        mlruns_root = root / "mlruns"
    else:
        mlruns_root = Path(mlruns_root)

    metrics_dirs = list(mlruns_root.rglob("metrics"))
    if not metrics_dirs:
        raise RuntimeError(f"No 'metrics' folder found under: {mlruns_root}")
    metrics_dir = metrics_dirs[0]

    # ── read all metric files ────────────────────────────────────────────────
    files = sorted([p for p in metrics_dir.iterdir() if p.is_file()])
    if not files:
        raise RuntimeError(f"No metric files found in: {metrics_dir}")

    n = len(files)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    # ── plot each metric file ────────────────────────────────────────────────
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        if i >= n:
            ax.axis("off")
            continue

        fp = files[i]
        df = read_metric_file(fp)
        if df is None:
            ax.text(
                0.5,
                0.5,
                "unreadable / empty",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#7F8C8D",
                transform=ax.transAxes,
            )
            ax.set_title(fp.name, fontsize=10, fontweight="bold", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
            continue

        x = df["step"].to_numpy()
        y = df["value"].to_numpy()
        _y_min, y_max = float(np.min(y)), float(np.max(y))

        # auto log-scale heuristic
        positive_min = float(np.min(y[y > 0])) if np.any(y > 0) else None
        ratio = (y_max / positive_min) if (positive_min and y_max > 0) else None
        log_scale = bool(ratio and ratio > 1e3)

        # subtle area fill under the curve (linear scale only — log fills look noisy)
        if not log_scale:
            ax.fill_between(x, y, y.min(), color=CLR_FILL, alpha=0.35, zorder=1)

        # main curve
        ax.plot(
            x,
            y,
            color=CLR_LINE,
            lw=2.0,
            solid_capstyle="round",
            zorder=3,
        )

        # min / max markers
        idx_min = int(np.argmin(y))
        idx_max = int(np.argmax(y))
        ax.scatter(
            [x[idx_min], x[idx_max]],
            [y[idx_min], y[idx_max]],
            s=28,
            color=CLR_ACCENT,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

        # final-value annotation — top right
        ax.annotate(
            f"final = {y[-1]:.3g}",
            xy=(0.97, 0.93),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#1A5276",
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="white",
                edgecolor="#AED6F1",
                linewidth=1.0,
                alpha=0.92,
            ),
            zorder=6,
        )

        # title & axes
        ax.set_title(fp.name, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch (step)", fontsize=9.5, fontweight="bold", labelpad=4)
        ax.set_ylabel(
            "Value (log scale)" if log_scale else "Value",
            fontsize=9.5,
            fontweight="bold",
            labelpad=4,
        )

        if log_scale:
            ax.set_yscale("log")
            ax.grid(
                True,
                which="both",
                linestyle="--",
                linewidth=0.55,
                alpha=0.45,
                color=CLR_GRID,
            )
        else:
            ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45, color=CLR_GRID)
            ax.yaxis.set_major_locator(MaxNLocator(6))

        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(which="both", length=4)
        ax.spines[["top", "right"]].set_visible(False)

        # tighten x-limits to data
        ax.set_xlim(x.min(), x.max())

        # bold tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    # ── global title ─────────────────────────────────────────────────────────
    fno_tag = cfg.custom.fno_type
    fig.suptitle(
        f"Training Metrics — {fno_tag}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    model_dir.mkdir(parents=True, exist_ok=True)
    out_png = model_dir / f"metrics_grid_{fno_tag}.png"
    plt.savefig(out_png, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"  Saved → {out_png}")


@hydra.main(version_base="1.2", config_path="conf", config_name="DECK_CONFIG")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: initialise the distributed manager and plot training metrics.

    Calls ``plot_all_metrics`` on rank 0 only, reading MLflow metric files and
    generating a grid of training-curve PNG images.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra configuration object with experiment and path settings.

    Returns
    -------
    None
    """
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.rank == 0:
        plot_all_metrics(cfg)


if __name__ == "__main__":
    main()
