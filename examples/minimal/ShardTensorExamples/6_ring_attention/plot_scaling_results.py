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
Parse ring-attention benchmark results and prepare publication-quality
matplotlib settings.

Usage
-----
    import plot_scaling_results as psr

    df = psr.load_results()           # full DataFrame (one row per JSON file)
    train = psr.filter(df, mode="train")
    inf   = psr.filter(df, mode="inference", gpus=4)

    # rcParams are applied on import; tweak after if needed.
    import matplotlib.pyplot as plt
    # ... your plotting code ...
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

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
# Publication-quality rcParams
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
    "axes.prop_cycle": mpl.cycler(
        color=[
            "#76B900",  # NVIDIA green (primary)
            "#1A1A1A",  # NVIDIA black
            "#999999",  # silver
            "#1B5E7A",  # dark teal
            "#808080",  # neutral grey
            "#A3D54E",  # light green
            "#004831",  # dark green
            "#C8E66E",  # pale green
            "#B0CC38",  # lime accent
            "#5C5C5C",  # mid grey
        ]
    ),
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
    "figure.figsize": (7.0, 5.0),
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    # --- Saving ---
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

mpl.rcParams.update(PUBRC)

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"^(?P<topology>single_gpu|distributed_(?P<ngpu>\d+)gpu)"
    r"_(?P<mode>train|inference)"
    r"_(?P<dtype>[a-z0-9]+)"
    r"_seq(?P<seq_len>\d+)\.json$"
)


def _parse_filename(fname: str) -> dict | None:
    """Extract metadata encoded in the benchmark filename."""
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    ngpu = int(m.group("ngpu")) if m.group("ngpu") else 1
    return {
        "filename": fname,
        "topology": m.group("topology"),
        "mode": m.group("mode"),
        "dtype": m.group("dtype"),
        "num_gpus": ngpu,
        "seq_len": int(m.group("seq_len")),
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _flatten_record(data: dict, meta: dict) -> dict:
    """Merge filename metadata with the flattened JSON contents."""
    rec = dict(meta)

    # Top-level scalars
    rec["timestamp"] = data.get("timestamp")
    rec["parallelism_label"] = data.get("parallelism")
    rec["distributed"] = data.get("distributed")
    rec["world_size"] = data.get("world_size")

    # Config
    cfg = data.get("config", {})
    rec["cfg_seq_len"] = cfg.get("seq_len")
    rec["cfg_num_heads"] = cfg.get("num_heads")
    rec["cfg_head_dim"] = cfg.get("head_dim")
    rec["cfg_batch_size"] = cfg.get("batch_size")
    rec["cfg_dtype"] = cfg.get("dtype")

    # Benchmark timing
    bench = data.get("benchmark", {})
    rec["num_warmup"] = bench.get("num_warmup")
    rec["num_iterations"] = bench.get("num_iterations")
    rec["mean_time_s"] = bench.get("mean_time_s")
    rec["std_time_s"] = bench.get("std_time_s")
    rec["min_time_s"] = bench.get("min_time_s")
    rec["max_time_s"] = bench.get("max_time_s")
    rec["median_time_s"] = bench.get("median_time_s")
    rec["mean_time_ms"] = bench.get("mean_time_s", 0) * 1000.0
    rec["std_time_ms"] = bench.get("std_time_s", 0) * 1000.0
    rec["median_time_ms"] = bench.get("median_time_s", 0) * 1000.0
    rec["per_iteration_times_s"] = bench.get("per_iteration_times_s")

    # Memory
    mem = data.get("memory", {})
    rec["input_bytes"] = mem.get("input_tensors_bytes")
    rec["peak_allocated_bytes"] = mem.get("peak_allocated_bytes")
    rec["peak_reserved_bytes"] = mem.get("peak_reserved_bytes")
    rec["allocated_delta_bytes"] = mem.get("allocated_delta_bytes")
    rec["max_peak_allocated_bytes"] = mem.get("max_peak_allocated_across_ranks_bytes")
    # Convenience: memory in MiB
    _MiB = 1024**2
    rec["peak_allocated_MiB"] = mem.get("peak_allocated_bytes", 0) / _MiB
    rec["peak_reserved_MiB"] = mem.get("peak_reserved_bytes", 0) / _MiB
    rec["max_peak_allocated_MiB"] = (
        mem.get("max_peak_allocated_across_ranks_bytes", 0) / _MiB
    )

    return rec


def load_results(results_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Read every JSON result file in *results_dir* and return a tidy DataFrame.

    Parameters
    ----------
    results_dir : path-like, optional
        Directory containing the JSON benchmark results.
        Defaults to ``results_bf16/`` next to this script.

    Returns
    -------
    pd.DataFrame
        One row per benchmark run.  Key columns:

        =========== ========================================================
        Column      Description
        =========== ========================================================
        num_gpus    Number of GPUs (1, 2, 4, 8)
        mode        ``"train"`` or ``"inference"``
        seq_len     Sequence length
        mean_time_s Mean wall-clock time per iteration (seconds)
        std_time_s  Std-dev of wall-clock time (seconds)
        mean_time_ms  Same, in milliseconds
        peak_allocated_MiB  Peak GPU memory allocated (MiB)
        =========== ========================================================
    """
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    records: list[dict] = []

    for fname in sorted(os.listdir(results_dir)):
        meta = _parse_filename(fname)
        if meta is None:
            continue
        fpath = results_dir / fname
        with open(fpath) as f:
            data = json.load(f)
        records.append(_flatten_record(data, meta))

    df = pd.DataFrame(records)

    # Sort for convenient iteration
    df.sort_values(["mode", "dtype", "num_gpus", "seq_len"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Convenience filters
# ---------------------------------------------------------------------------


def filter(
    df: pd.DataFrame,
    *,
    mode: Optional[str] = None,
    gpus: Optional[int | list[int]] = None,
    seq_len: Optional[int | list[int]] = None,
    dtype: Optional[str | list[str]] = None,
) -> pd.DataFrame:
    """
    Return a filtered copy of the results DataFrame.

    Parameters
    ----------
    mode : {"train", "inference"}, optional
    gpus : int or list[int], optional
    seq_len : int or list[int], optional
    dtype : str or list[str], optional
    """
    mask = pd.Series(True, index=df.index)
    if mode is not None:
        mask &= df["mode"] == mode
    if gpus is not None:
        if not isinstance(gpus, list):
            gpus = [gpus]
        mask &= df["num_gpus"].isin(gpus)
    if seq_len is not None:
        if not isinstance(seq_len, list):
            seq_len = [seq_len]
        mask &= df["seq_len"].isin(seq_len)
    if dtype is not None:
        if not isinstance(dtype, list):
            dtype = [dtype]
        mask &= df["dtype"].isin(dtype)
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Quick summary table
# ---------------------------------------------------------------------------


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the DataFrame into a readable summary with sequence lengths as
    columns and (mode, num_gpus) as rows, showing mean_time_ms.
    """
    return df.pivot_table(
        index=["mode", "num_gpus"],
        columns="seq_len",
        values="mean_time_ms",
        aggfunc="first",
    )


# ---------------------------------------------------------------------------
# CLI: print a quick summary when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_results()
    print(f"Loaded {len(df)} benchmark records from {DEFAULT_RESULTS_DIR}\n")

    gpu_colors = {1: "C0", 2: "C1", 4: "C2", 8: "C3"}
    gpu_markers = {1: "o", 2: "s", 4: "D", 8: "^"}
    x_fmt = plt.FuncFormatter(lambda v, _: f"$2^{{{int(np.log2(v))}}}$")

    plot_dtype = "float32"

    # -------------------------------------------------------------------
    # Latency: one figure per mode (float32 only)
    # -------------------------------------------------------------------
    for mode in ["inference", "train"]:
        fig, ax = plt.subplots(figsize=(9, 7))
        for n_gpus in sorted(df["num_gpus"].unique()):
            res = filter(df, mode=mode, gpus=n_gpus, dtype=plot_dtype)
            if res.empty:
                continue
            ax.plot(
                res["seq_len"],
                res["mean_time_ms"],
                label=f"{n_gpus} GPU",
                marker=gpu_markers.get(n_gpus, "o"),
                color=gpu_colors.get(n_gpus, None),
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.xaxis.set_major_formatter(x_fmt)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Latency — {mode.capitalize()}")
        ax.legend()
        fig.savefig(f"ring_attention_shard_tensor_{mode}.png")
        plt.close(fig)
