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

Either ``--uq`` or ``--random`` (or both) must be provided. When only one is
given, the figure becomes a single-method view (no comparison line). Class
labels/colors are recognised for F/N/E and SE/SF; unknown classes fall back
to a default palette.

Usage::

    # UQ + Random comparison (multi-seed random):
    python plot_al_results.py \\
        --uq runs/geotransolver/surface/active_learning/joint_uq/validation_metrics.json \\
        --random runs/geotransolver/surface/active_learning/random/validation_metrics.json \\
              runs/geotransolver/surface/active_learning/random_seed2/validation_metrics.json \\
              runs/geotransolver/surface/active_learning/random_seed3/validation_metrics.json \\
        --history runs/geotransolver/surface/active_learning/joint_uq/selection_history.json \\
        -o runs/geotransolver/surface/active_learning/al_results.png

    # UQ only:
    python plot_al_results.py \\
        --uq .../joint_uq/validation_metrics.json \\
        --history .../joint_uq/selection_history.json \\
        --title "Joint UQ — fix-3" -o uq_fix3.png

    # Random only (single seed) with selection bar:
    python plot_al_results.py \\
        --random .../al_random_seed_42/random/validation_metrics.json \\
        --history .../al_random_seed_42/random/selection_history.json \\
        --method-label "Random (seed 42)" --title "Random — fix-3" \\
        -o random_fix3_seed42.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_COLORS = {
    "F": "#1f77b4",  # Fastback (DrivAerStar)
    "N": "#ff7f0e",  # Notchback
    "E": "#2ca02c",  # Estateback
    "SE": "#9467bd", # SUV Estateback (cross-dataset)
    "SF": "#8c564b", # SUV Fastback
}
CLASS_LABELS = {
    "F": "Fastback",
    "N": "Notchback",
    "E": "Estateback",
    "SE": "SUV Estateback",
    "SF": "SUV Fastback",
}
_FALLBACK_PALETTE = [
    "#17becf", "#bcbd22", "#7f7f7f", "#e377c2", "#aec7e8", "#ffbb78",
]


def _color_for(cls: str, fallback_idx: list[int]) -> str:
    if cls in CLASS_COLORS:
        return CLASS_COLORS[cls]
    color = _FALLBACK_PALETTE[fallback_idx[0] % len(_FALLBACK_PALETTE)]
    fallback_idx[0] += 1
    return color


def _label_for(cls: str) -> str:
    return CLASS_LABELS.get(cls, cls)


def _dedup_by_step(records: list[dict]) -> list[dict]:
    """De-duplicate records by ``step`` keeping the latest occurrence.

    The chain-resume bug (lex-sort of ``checkpoint_round_*``) caused several
    rounds to be re-run after the chain rolled back to round 9.  Both
    ``validation_metrics.json`` and ``selection_history.json`` were appended
    on each re-run, so the same ``step`` appears multiple times.  Last-write
    wins gives us the most recent model state for that round; selections are
    deterministic in the seeded path so first/last are equivalent for
    selection_history.
    """
    by_step: dict = {}
    for r in records:
        if "step" in r:
            by_step[r["step"]] = r
    if not by_step:
        return records
    return [by_step[k] for k in sorted(by_step.keys())]


def load_metrics(path: str | Path) -> list[dict]:
    with open(path) as f:
        return _dedup_by_step(json.load(f))


def load_history(path: str | Path) -> list[dict]:
    with open(path) as f:
        return _dedup_by_step(json.load(f))


def main():
    parser = argparse.ArgumentParser(description="Plot AL learning curves")
    parser.add_argument("--uq", default=None, help="UQ metrics JSON")
    parser.add_argument(
        "--random", nargs="+", default=None, help="Random metrics JSON(s)"
    )
    parser.add_argument(
        "--history", default=None,
        help="selection_history.json for whichever method is being plotted "
             "(used for the stacked-bar panel)."
    )
    parser.add_argument(
        "--uq-history", default=None,
        help="[Deprecated alias for --history]"
    )
    parser.add_argument(
        "--method-label", default=None,
        help="Override the legend label for the primary method "
             "(default: 'Joint UQ' or 'Random')."
    )
    parser.add_argument(
        "--title", default=None,
        help="Override the headline-panel title."
    )
    parser.add_argument("-o", "--output", default="al_results.png")
    args = parser.parse_args()

    if args.uq is None and not args.random:
        parser.error("Provide at least one of --uq or --random.")

    uq_metrics = load_metrics(args.uq) if args.uq else []
    random_runs = [load_metrics(p) for p in args.random] if args.random else []

    history_path = args.history or args.uq_history
    has_history = history_path is not None

    # Pick the source of per-class data: UQ if it has any populated per_class_r2
    # entry (early ShiftSUV runs had empty dicts before the class-mapping fix),
    # else random.
    def _has_per_class(records: list[dict]) -> bool:
        return any(r.get("per_class_r2") for r in records)

    if uq_metrics and _has_per_class(uq_metrics):
        per_class_source = "uq"
    elif random_runs and any(_has_per_class(r) for r in random_runs):
        per_class_source = "random"
    else:
        per_class_source = None

    has_field_mse = (
        (uq_metrics and any("field_mse" in r for r in uq_metrics))
        or (random_runs and any("field_mse" in r for r in random_runs[0]))
    )

    # UQ-signal panel needs both UQ metrics and a history file whose selection
    # entries carry joint_uq / disagreement / gp_std fields.
    has_uq_signal = False
    history_records: list[dict] | None = None
    if has_history:
        history_records = load_history(history_path)
        if (
            uq_metrics
            and history_records
            and history_records[0].get("selected")
            and "joint_uq" in history_records[0]["selected"][0]
        ):
            has_uq_signal = True

    panels: list[str] = ["drag_r2"]
    if has_field_mse:
        panels.append("field_mse")
    if per_class_source is not None:
        panels.append("per_class_r2")
    if has_uq_signal:
        panels.append("uq_signal")
    if has_history:
        panels.append("selection_bar")

    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = list(axes)

    col = 0

    # ---- Learning curve ----
    ax = axes[col]
    col += 1

    uq_n: list[int] = []
    if uq_metrics:
        uq_n = [r["n_train"] for r in uq_metrics]
        uq_r2 = [r["drag_r2"] for r in uq_metrics]
        ax.plot(
            uq_n, uq_r2, "o-", color="#1f77b4", lw=2.5, ms=8,
            label=args.method_label if (args.method_label and not random_runs) else "Joint UQ",
            zorder=3,
        )

    rnd_label = args.method_label if (args.method_label and not uq_metrics) else "Random"
    if len(random_runs) == 1:
        rnd = random_runs[0]
        rnd_n = [r["n_train"] for r in rnd]
        rnd_r2 = [r["drag_r2"] for r in rnd]
        ax.plot(
            rnd_n, rnd_r2, "s--", color="#d62728", lw=2, ms=7,
            label=rnd_label, zorder=2,
        )
    elif len(random_runs) > 1:
        n_steps = min(len(r) for r in random_runs)
        all_r2 = np.array([[r["drag_r2"] for r in run[:n_steps]] for run in random_runs])
        all_n = np.array([r["n_train"] for r in random_runs[0][:n_steps]])
        mean_r2 = all_r2.mean(axis=0)
        std_r2 = all_r2.std(axis=0)
        ax.plot(
            all_n, mean_r2, "s--", color="#d62728", lw=2, ms=7,
            label=f"{rnd_label} (mean, {len(random_runs)} seeds)", zorder=2,
        )
        ax.fill_between(
            all_n, mean_r2 - std_r2, mean_r2 + std_r2,
            alpha=0.2, color="#d62728", label=f"{rnd_label} (±1σ)",
        )

    if args.title:
        title = args.title
    elif uq_metrics and random_runs:
        title = "Active Learning: UQ-Guided vs Random"
    elif uq_metrics:
        title = "Active Learning: UQ-Guided"
    else:
        title = "Active Learning: Random Baseline"

    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("Drag R²", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    # Auto-clip y-axis if R² ever dips below -1 — OOD datasets (ShiftSUV) start
    # with R² ≈ -40 which crushes the interesting 0-1 region.  Leaves enough
    # head/footroom to read the asymptote and approach curves.
    all_r2 = []
    if uq_metrics:
        all_r2.extend(r["drag_r2"] for r in uq_metrics if r.get("drag_r2") is not None)
    for run in random_runs:
        all_r2.extend(r["drag_r2"] for r in run if r.get("drag_r2") is not None)
    if all_r2 and min(all_r2) < -1.0:
        ax.set_ylim(bottom=-0.1, top=min(1.05, max(all_r2) * 1.05 + 0.05))

    # ---- Field MSE ----
    if "field_mse" in panels:
        ax_fm = axes[col]
        col += 1
        if uq_metrics:
            ax_fm.plot(
                uq_n, [r.get("field_mse") for r in uq_metrics],
                "o-", color="#1f77b4", lw=2.5, ms=8,
                label=args.method_label if (args.method_label and not random_runs) else "Joint UQ",
                zorder=3,
            )
        if len(random_runs) == 1:
            rnd = random_runs[0]
            ax_fm.plot(
                [r["n_train"] for r in rnd], [r.get("field_mse") for r in rnd],
                "s--", color="#d62728", lw=2, ms=7, label=rnd_label, zorder=2,
            )
        elif len(random_runs) > 1:
            n_steps = min(len(r) for r in random_runs)
            all_fm = np.array([
                [r.get("field_mse", np.nan) for r in run[:n_steps]]
                for run in random_runs
            ])
            all_n = np.array([r["n_train"] for r in random_runs[0][:n_steps]])
            mean_fm = np.nanmean(all_fm, axis=0)
            std_fm = np.nanstd(all_fm, axis=0)
            ax_fm.plot(
                all_n, mean_fm, "s--", color="#d62728", lw=2, ms=7,
                label=f"{rnd_label} (mean, {len(random_runs)} seeds)", zorder=2,
            )
            ax_fm.fill_between(
                all_n, mean_fm - std_fm, mean_fm + std_fm,
                alpha=0.2, color="#d62728",
            )
        ax_fm.set_xlabel("Number of Training Samples", fontsize=12)
        ax_fm.set_ylabel("Field MSE (validation)", fontsize=12)
        ax_fm.set_title("Surface-Field MSE", fontsize=13, fontweight="bold")
        ax_fm.set_yscale("log")
        ax_fm.legend(loc="upper right", fontsize=10)
        ax_fm.grid(True, alpha=0.2, which="both")
        ax_fm.spines["top"].set_visible(False)
        ax_fm.spines["right"].set_visible(False)
        ax_fm.tick_params(labelsize=11)

    # ---- Per-class R² curves ----
    if per_class_source is not None:
        ax_cls = axes[col]
        col += 1
        fb_idx = [0]
        if per_class_source == "uq":
            n_axis = uq_n
            classes = sorted({c for r in uq_metrics for c in r.get("per_class_r2", {})})
            for cls in classes:
                cls_r2 = [
                    r.get("per_class_r2", {}).get(cls, np.nan) for r in uq_metrics
                ]
                ax_cls.plot(
                    n_axis, cls_r2, "o-", color=_color_for(cls, fb_idx),
                    lw=2, ms=7, label=_label_for(cls),
                )
            cls_subtitle = "UQ-Guided"
        elif per_class_source == "random" and len(random_runs) == 1:
            rnd = random_runs[0]
            n_axis = [r["n_train"] for r in rnd]
            classes = sorted({c for r in rnd for c in r.get("per_class_r2", {})})
            for cls in classes:
                cls_r2 = [r.get("per_class_r2", {}).get(cls, np.nan) for r in rnd]
                ax_cls.plot(
                    n_axis, cls_r2, "s-", color=_color_for(cls, fb_idx),
                    lw=2, ms=7, label=_label_for(cls),
                )
            cls_subtitle = rnd_label
        else:
            # Multi-seed random: aggregate per-class across seeds.
            n_steps = min(len(r) for r in random_runs)
            n_axis = [random_runs[0][i]["n_train"] for i in range(n_steps)]
            classes = sorted({
                c for run in random_runs for r in run[:n_steps]
                for c in r.get("per_class_r2", {})
            })
            for cls in classes:
                cls_arr = np.array([
                    [run[i].get("per_class_r2", {}).get(cls, np.nan) for i in range(n_steps)]
                    for run in random_runs
                ])
                mean_v = np.nanmean(cls_arr, axis=0)
                std_v = np.nanstd(cls_arr, axis=0)
                color = _color_for(cls, fb_idx)
                ax_cls.plot(
                    n_axis, mean_v, "s-", color=color, lw=2, ms=7,
                    label=_label_for(cls),
                )
                ax_cls.fill_between(
                    n_axis, mean_v - std_v, mean_v + std_v,
                    alpha=0.15, color=color,
                )
            cls_subtitle = f"{rnd_label} (mean ± σ)"
        # Same OOD clip as the headline R² panel.
        if all_r2 and min(all_r2) < -1.0:
            ax_cls.set_ylim(bottom=-0.5, top=min(1.05, max(all_r2) * 1.05 + 0.05))
        ax_cls.set_xlabel("Number of AL Samples Added", fontsize=12)
        ax_cls.set_ylabel("Drag R² (per class)", fontsize=12)
        ax_cls.set_title(f"Per-Class R²: {cls_subtitle}", fontsize=13, fontweight="bold")
        ax_cls.legend(loc="lower right", fontsize=10)
        ax_cls.grid(True, alpha=0.2)
        ax_cls.spines["top"].set_visible(False)
        ax_cls.spines["right"].set_visible(False)
        ax_cls.tick_params(labelsize=11)

    # ---- Class composition per round (stacked bar) ----
    def _draw_bar(ax2, rounds, class_counts_per_round, all_classes, bar_title):
        fb_idx = [0]
        x = np.arange(len(rounds))
        bottom = np.zeros(len(rounds))
        for cls in sorted(all_classes):
            vals = np.array(class_counts_per_round[cls])
            ax2.bar(
                x, vals, bottom=bottom, width=0.6,
                color=_color_for(cls, fb_idx),
                label=_label_for(cls),
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
        ax2.set_title(bar_title, fontsize=13, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(True, alpha=0.2, axis="y")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(labelsize=11)

    # ---- UQ acquisition-signal decomposition (UQ runs only) ----
    if "uq_signal" in panels:
        ax_sig = axes[col]
        col += 1
        sig_rounds = []
        avg_dis = []
        avg_2std = []
        avg_juq = []
        max_dis = []
        max_2std = []
        for record in history_records:
            sig_rounds.append(record.get("step", len(sig_rounds) + 1))
            sel = record.get("selected", [])
            if not sel:
                avg_dis.append(np.nan); avg_2std.append(np.nan); avg_juq.append(np.nan)
                max_dis.append(np.nan); max_2std.append(np.nan)
                continue
            d = np.array([s.get("disagreement", 0.0) for s in sel])
            g = np.array([s.get("gp_std", 0.0) for s in sel])
            j = np.array([s.get("joint_uq", 0.0) for s in sel])
            avg_dis.append(d.mean())
            avg_2std.append((2.0 * g).mean())
            avg_juq.append(j.mean())
            max_dis.append(d.max())
            max_2std.append((2.0 * g).max())
        ax_sig.plot(
            sig_rounds, avg_juq, "o-", color="#1f77b4", lw=2.5, ms=8,
            label="avg joint UQ", zorder=3,
        )
        ax_sig.plot(
            sig_rounds, avg_dis, "s--", color="#d62728", lw=2, ms=7,
            label="avg |disagreement|", zorder=2,
        )
        ax_sig.plot(
            sig_rounds, avg_2std, "^--", color="#2ca02c", lw=2, ms=7,
            label="avg 2·GP_std", zorder=2,
        )
        ax_sig.set_xlabel("AL Round", fontsize=12)
        ax_sig.set_ylabel("Acquisition signal (Cd units)", fontsize=12)
        ax_sig.set_title("UQ Signal Decomposition", fontsize=13, fontweight="bold")
        ax_sig.legend(loc="upper right", fontsize=10)
        ax_sig.grid(True, alpha=0.2)
        ax_sig.spines["top"].set_visible(False)
        ax_sig.spines["right"].set_visible(False)
        ax_sig.tick_params(labelsize=11)
        if len(sig_rounds) > 0:
            ax_sig.set_xticks(sig_rounds)

    if has_history:
        ax2 = axes[col]
        col += 1
        history = history_records  # already loaded above

        # Two-pass: first collect the full class set so every per-round vector
        # has identical length and proper round alignment. The previous
        # implementation grew the class set as it walked rounds, which silently
        # shifted later-appearing classes left by one column.
        all_classes: set[str] = set()
        for record in history:
            for entry in record["selected"]:
                all_classes.add(entry["class"])

        rounds: list[int] = []
        class_counts_per_round: dict[str, list[int]] = {c: [] for c in all_classes}
        for record in history:
            round_counts: dict[str, int] = defaultdict(int)
            for entry in record["selected"]:
                round_counts[entry["class"]] += 1
            rounds.append(record.get("step", len(rounds) + 1))
            for cls in all_classes:
                class_counts_per_round[cls].append(round_counts.get(cls, 0))

        if uq_metrics and not random_runs:
            bar_title = "UQ-Guided Selection: Class Composition"
        elif random_runs and not uq_metrics:
            bar_title = f"{rnd_label} Selection: Class Composition"
        else:
            bar_title = "Selection: Class Composition"
        _draw_bar(ax2, rounds, class_counts_per_round, all_classes, bar_title)

    fig.tight_layout(w_pad=3)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
