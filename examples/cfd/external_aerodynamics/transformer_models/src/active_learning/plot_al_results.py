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

"""Plot active learning results: learning curves and class composition.

Usage::

    python plot_al_results.py \\
        --uq runs/geotransolver/surface/active_learning/joint_uq/validation_metrics.json \\
        --random runs/geotransolver/surface/active_learning/random/validation_metrics.json \\
              runs/geotransolver/surface/active_learning/random_seed2/validation_metrics.json \\
              runs/geotransolver/surface/active_learning/random_seed3/validation_metrics.json \\
        --uq-history runs/geotransolver/surface/active_learning/joint_uq/selection_history.json \\
        -o runs/geotransolver/surface/active_learning/al_results.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_history(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot AL learning curves")
    parser.add_argument("--uq", required=True, help="UQ metrics JSON")
    parser.add_argument(
        "--random", nargs="+", default=None, help="Random metrics JSON(s)"
    )
    parser.add_argument("--uq-history", default=None, help="UQ selection_history.json")
    parser.add_argument("-o", "--output", default="al_results.png")
    args = parser.parse_args()

    uq_metrics = load_metrics(args.uq)
    random_runs = [load_metrics(p) for p in args.random] if args.random else []

    has_history = args.uq_history is not None
    has_bar = has_history
    n_cols = 1 + int(has_bar) + int(len(uq_metrics) > 0 and "per_class_r2" in uq_metrics[0])
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = list(axes)

    col = 0

    # ---- Learning curve ----
    ax = axes[col]
    col += 1

    uq_n = [r["n_train"] for r in uq_metrics]
    uq_r2 = [r["drag_r2"] for r in uq_metrics]
    ax.plot(uq_n, uq_r2, "o-", color="#1f77b4", lw=2.5, ms=8, label="Joint UQ", zorder=3)

    if len(random_runs) == 1:
        rnd = random_runs[0]
        rnd_n = [r["n_train"] for r in rnd]
        rnd_r2 = [r["drag_r2"] for r in rnd]
        ax.plot(rnd_n, rnd_r2, "s--", color="#d62728", lw=2, ms=7, label="Random", zorder=2)
    elif len(random_runs) > 1:
        all_r2 = np.array([[r["drag_r2"] for r in run] for run in random_runs])
        all_n = np.array([r["n_train"] for r in random_runs[0]])
        mean_r2 = all_r2.mean(axis=0)
        std_r2 = all_r2.std(axis=0)
        ax.plot(all_n, mean_r2, "s--", color="#d62728", lw=2, ms=7, label="Random (mean)", zorder=2)
        ax.fill_between(
            all_n, mean_r2 - std_r2, mean_r2 + std_r2,
            alpha=0.2, color="#d62728", label="Random (±1σ)"
        )

    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("Drag R²", fontsize=12)
    ax.set_title("Active Learning: UQ-Guided vs Random", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    # ---- Per-class R² curves ----
    if len(uq_metrics) > 0 and "per_class_r2" in uq_metrics[0]:
        ax_cls = axes[col]
        col += 1
        class_colors = {"F": "#1f77b4", "N": "#ff7f0e", "E": "#2ca02c"}
        class_labels = {"F": "Fastback", "N": "Notchback", "E": "Estateback"}
        for cls in sorted(uq_metrics[0]["per_class_r2"].keys()):
            cls_r2 = [r["per_class_r2"][cls] for r in uq_metrics]
            ax_cls.plot(
                uq_n, cls_r2, "o-", color=class_colors.get(cls, "gray"),
                lw=2, ms=7, label=class_labels.get(cls, cls),
            )
        ax_cls.set_xlabel("Number of AL Samples Added", fontsize=12)
        ax_cls.set_ylabel("Drag R² (per class)", fontsize=12)
        ax_cls.set_title("Per-Class R² Progression", fontsize=13, fontweight="bold")
        ax_cls.legend(loc="lower right", fontsize=10)
        ax_cls.grid(True, alpha=0.2)
        ax_cls.spines["top"].set_visible(False)
        ax_cls.spines["right"].set_visible(False)
        ax_cls.tick_params(labelsize=11)

    # ---- Class composition per round (stacked bar) ----
    def _draw_bar(ax2, rounds, class_counts_per_round, all_classes):
        class_colors = {"F": "#1f77b4", "N": "#ff7f0e", "E": "#2ca02c"}
        class_labels = {"F": "Fastback", "N": "Notchback", "E": "Estateback"}
        x = np.arange(len(rounds))
        bottom = np.zeros(len(rounds))
        for cls in sorted(all_classes):
            vals = np.array(class_counts_per_round[cls])
            ax2.bar(
                x, vals, bottom=bottom, width=0.6,
                color=class_colors.get(cls, "gray"),
                label=class_labels.get(cls, cls),
                edgecolor="white", linewidth=0.5,
            )
            for i, v in enumerate(vals):
                if v > 3:
                    ax2.text(
                        x[i], bottom[i] + v / 2, str(v),
                        ha="center", va="center", fontsize=9, fontweight="bold", color="white",
                    )
            bottom += vals
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Round {r}" for r in rounds], fontsize=10)
        ax2.set_xlabel("AL Round", fontsize=12)
        ax2.set_ylabel("Samples Selected", fontsize=12)
        ax2.set_title("UQ-Guided Selection: Class Composition", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(True, alpha=0.2, axis="y")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(labelsize=11)

    if has_history:
        ax2 = axes[col]
        col += 1
        history = load_history(args.uq_history)
        rounds = []
        class_counts_per_round: dict[str, list[int]] = defaultdict(list)
        all_classes = set()
        for record in history:
            round_counts: dict[str, int] = defaultdict(int)
            for entry in record["selected"]:
                round_counts[entry["class"]] += 1
                all_classes.add(entry["class"])
            rounds.append(record.get("step", len(rounds) + 1))
            for cls in all_classes:
                class_counts_per_round[cls].append(round_counts.get(cls, 0))
        for cls in sorted(all_classes):
            if len(class_counts_per_round[cls]) < len(rounds):
                class_counts_per_round[cls].extend(
                    [0] * (len(rounds) - len(class_counts_per_round[cls]))
                )
        _draw_bar(ax2, rounds, class_counts_per_round, all_classes)

    fig.tight_layout(w_pad=3)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
