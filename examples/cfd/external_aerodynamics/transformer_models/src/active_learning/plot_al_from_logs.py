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

"""Plot active learning results by parsing slurm log files directly.

Extracts metrics and selection counts from log lines matching:
  - "Step N | n_train=... | R²=... | field_MSE=... | per_class={...}"
  - "Selected 50 samples: {'E': 47, 'N': 2, 'F': 1}"

Usage::

    python plot_al_from_logs.py \\
        slurm_logs/drivaer-al-uq-fix_5125861.out \\
        slurm_logs/drivaer-al-uq-fix_5125862.out \\
        slurm_logs/drivaer-al-uq-fix_5125863.out \\
        slurm_logs/drivaer-al-uq-fix_5125864.out \\
        -o al_results_from_logs.png
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Old format: "Step N | n_train=X | R²=Y | field_MSE=Z | per_class={...}"
METRIC_OLD_RE = re.compile(
    r"Step (\d+) \| n_train=(\d+) \| R²=([\d.]+) \| field_MSE=([\d.]+) "
    r"\| per_class=(\{.*\})"
)

# New format: "Step N | n_train=X | R²_gp=Y | R²_trans=T | field_MSE=Z | per_class_gp={...} | per_class_trans={...} | per_class_fmse={...}"
METRIC_NEW_RE = re.compile(
    r"Step (\d+) \| n_train=(\d+) \| R²_gp=([\d.]+)"
    r"(?: \| R²_trans=([\d.]+))?"
    r" \| field_MSE=([\d.]+) \| "
    r"per_class_gp=(\{.*?\}) \| "
    r"per_class_trans=(\{.*?\}) \| "
    r"per_class_fmse=(\{.*?\})"
)

SELECTION_RE = re.compile(
    r"Selected (\d+) samples: (\{.*\})"
)

ROUND_RE = re.compile(
    r"=== Active Learning Round (\d+)/(\d+) ==="
)


def parse_logs(paths: list[str]) -> tuple[list[dict], list[dict]]:
    metrics: dict[int, dict] = {}
    selections: dict[int, dict] = {}
    current_round = None

    for path in paths:
        with open(path) as f:
            for line in f:
                m = ROUND_RE.search(line)
                if m:
                    current_round = int(m.group(1))

                m = METRIC_NEW_RE.search(line)
                if m:
                    step = int(m.group(1))
                    metrics[step] = {
                        "step": step,
                        "n_train": int(m.group(2)),
                        "drag_r2": float(m.group(3)),
                        "drag_r2_transolver": float(m.group(4)) if m.group(4) else None,
                        "field_mse": float(m.group(5)),
                        "per_class_r2": ast.literal_eval(m.group(6)),
                        "per_class_r2_transolver": ast.literal_eval(m.group(7)),
                        "per_class_field_mse": ast.literal_eval(m.group(8)),
                    }
                else:
                    m = METRIC_OLD_RE.search(line)
                    if m:
                        step = int(m.group(1))
                        metrics[step] = {
                            "step": step,
                            "n_train": int(m.group(2)),
                            "drag_r2": float(m.group(3)),
                            "drag_r2_transolver": None,
                            "field_mse": float(m.group(4)),
                            "per_class_r2": ast.literal_eval(m.group(5)),
                            "per_class_r2_transolver": None,
                            "per_class_field_mse": None,
                        }

                m = SELECTION_RE.search(line)
                if m and current_round is not None:
                    counts = ast.literal_eval(m.group(2))
                    selections[current_round] = {
                        "step": current_round,
                        "counts": counts,
                    }

    sorted_metrics = [metrics[k] for k in sorted(metrics)]
    sorted_selections = [selections[k] for k in sorted(selections)]
    return sorted_metrics, sorted_selections


def main():
    parser = argparse.ArgumentParser(
        description="Plot AL results directly from slurm log files"
    )
    parser.add_argument("logs", nargs="+", help="Slurm .out log files (in order)")
    parser.add_argument("-o", "--output", default="al_results_from_logs.png")
    args = parser.parse_args()

    metrics, selections = parse_logs(args.logs)

    if not metrics:
        print("No metrics found in the provided log files.")
        return

    has_per_class = "per_class_r2" in metrics[0]
    has_trans = any(r.get("drag_r2_transolver") is not None for r in metrics)
    has_fmse = any(r.get("per_class_field_mse") is not None for r in metrics)
    has_selections = len(selections) > 0

    class_colors = {"F": "#1f77b4", "N": "#ff7f0e", "E": "#2ca02c"}
    class_labels = {"F": "Fastback", "N": "Notchback", "E": "Estateback"}

    panels = ["overall_r2"]
    if has_per_class:
        panels.append("per_class_r2_gp")
    if has_trans:
        panels.append("per_class_r2_trans")
    if has_fmse:
        panels.append("per_class_fmse")
    if has_selections:
        panels.append("selections")

    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = list(axes)

    uq_n = [r["n_train"] for r in metrics]
    col = 0

    def _style(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)

    # ---- Overall R² (GP + Transolver) ----
    ax = axes[col]; col += 1
    ax.plot(uq_n, [r["drag_r2"] for r in metrics],
            "o-", color="#1f77b4", lw=2.5, ms=8, label="GP Head")
    if has_trans:
        ax.plot(uq_n, [r["drag_r2_transolver"] for r in metrics],
                "s--", color="#d62728", lw=2, ms=7, label="GeoTransolver (integrated)")
    _style(ax, "AL Samples Added", "Drag R²", "Overall Drag R²")

    # ---- Per-class R² (GP) ----
    if "per_class_r2_gp" in panels:
        ax = axes[col]; col += 1
        for cls in sorted(metrics[0]["per_class_r2"].keys()):
            ax.plot(uq_n, [r["per_class_r2"][cls] for r in metrics],
                    "o-", color=class_colors.get(cls, "gray"), lw=2, ms=7,
                    label=class_labels.get(cls, cls))
        _style(ax, "AL Samples Added", "Drag R²", "Per-Class R² (GP Head)")

    # ---- Per-class R² (GeoTransolver) ----
    if "per_class_r2_trans" in panels:
        ax = axes[col]; col += 1
        for cls in sorted(metrics[0]["per_class_r2"].keys()):
            vals = [r["per_class_r2_transolver"].get(cls, None)
                    if r.get("per_class_r2_transolver") else None
                    for r in metrics]
            if any(v is not None for v in vals):
                ax.plot(uq_n, vals, "s--", color=class_colors.get(cls, "gray"),
                        lw=2, ms=7, label=class_labels.get(cls, cls))
        _style(ax, "AL Samples Added", "Drag R²", "Per-Class R² (GeoTransolver)")

    # ---- Per-class field MSE ----
    if "per_class_fmse" in panels:
        ax = axes[col]; col += 1
        for cls in sorted(metrics[0]["per_class_r2"].keys()):
            vals = [r["per_class_field_mse"].get(cls, None)
                    if r.get("per_class_field_mse") else None
                    for r in metrics]
            if any(v is not None for v in vals):
                ax.plot(uq_n, vals, "o-", color=class_colors.get(cls, "gray"),
                        lw=2, ms=7, label=class_labels.get(cls, cls))
        _style(ax, "AL Samples Added", "Field MSE", "Per-Class Field MSE")

    # ---- Stacked bar: class composition ----
    if "selections" in panels:
        ax2 = axes[col]; col += 1
        rounds = [s["step"] for s in selections]
        all_classes = sorted({c for s in selections for c in s["counts"]})
        x = np.arange(len(rounds))
        bottom = np.zeros(len(rounds))
        for cls in all_classes:
            vals = np.array([s["counts"].get(cls, 0) for s in selections])
            ax2.bar(x, vals, bottom=bottom, width=0.6,
                    color=class_colors.get(cls, "gray"),
                    label=class_labels.get(cls, cls),
                    edgecolor="white", linewidth=0.5)
            for i, v in enumerate(vals):
                if v > 3:
                    ax2.text(x[i], bottom[i] + v / 2, str(v), ha="center",
                             va="center", fontsize=9, fontweight="bold", color="white")
            bottom += vals
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"R{r}" for r in rounds], fontsize=10)
        _style(ax2, "AL Round", "Samples Selected", "Class Composition per Round")
        ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout(w_pad=3)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")

    print("\n--- Parsed metrics ---")
    for r in metrics:
        pc_gp = r.get("per_class_r2", {})
        gp_str = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(pc_gp.items()))
        trans_r2 = r.get("drag_r2_transolver")
        trans_str = f" | R²_trans={trans_r2:.4f}" if trans_r2 is not None else ""
        pc_trans = r.get("per_class_r2_transolver") or {}
        trans_cls = (" | trans: " + " ".join(f"{k}:{v:.4f}" for k, v in sorted(pc_trans.items()))) if pc_trans else ""
        pc_fmse = r.get("per_class_field_mse") or {}
        fmse_cls = (" | fmse: " + " ".join(f"{k}:{v:.4f}" for k, v in sorted(pc_fmse.items()))) if pc_fmse else ""
        print(f"  Step {r['step']:>2} | n_train={r['n_train']:>3} | "
              f"R²_gp={r['drag_r2']:.4f}{trans_str} | gp: {gp_str}{trans_cls}{fmse_cls}")

    print("\n--- Parsed selections ---")
    for s in selections:
        print(f"  Round {s['step']}: {s['counts']}")

    plt.close(fig)


if __name__ == "__main__":
    main()
