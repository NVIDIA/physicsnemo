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

r"""Per-channel field-error metrics for the SuperWing test split.

Reads the ``predictions.npz`` produced by ``inference.py`` and computes,
per case and per surface channel (``Cp``, ``Cf_tau``, ``Cf_z``):

* Relative L2 = ``|| pred - gt ||_2 / || gt ||_2``
* RMSE = ``sqrt(mean((pred - gt)^2))``
* MAE = ``mean(|pred - gt|)``

Writes one CSV with the per-case + per-channel breakdown and a
human-readable summary text file with the test-split mean / std of each
metric. The summary is the artifact to compare against paper tables.

Run as a script:

.. code-block:: bash

    python -m src.postprocessing.superwing_metrics \
        --predictions outputs/<run>/inference/predictions.npz \
        --output outputs/<run>/inference/field_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


SUPERWING_CHANNEL_NAMES: tuple[str, ...] = ("Cp", "Cf_tau", "Cf_z")

# Field metrics reported in the paper table, in column order.
FIELD_METRICS: tuple[str, ...] = (
    "rel_l2",
    "rel_l1",
    "rmse_over_gtmax",
    "mae_over_gtmax",
    "rmse",
    "mae",
)


def per_case_field_metrics(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
    chunk_size: int = 512,
) -> dict[str, np.ndarray]:
    r"""Compute the paper's per-case, per-channel field metrics.

    Parameters
    ----------
    pred : np.ndarray
        Predicted field of shape ``(B, C, H, W)`` (physical units).
    target : np.ndarray
        Ground-truth field of shape ``(B, C, H, W)``.
    eps : float, optional
        Floor on denominators to avoid division by zero on degenerate
        targets. Default ``1e-12``.
    chunk_size : int, optional
        Number of cases processed per block. Each metric reduces over a
        single case independently, so chunking is numerically exact; it
        just bounds the peak memory of the float64 temporaries (the full
        ``(B, C, H, W)`` fields would otherwise need several GB at once).
        Default ``512``.

    Returns
    -------
    dict
        Keys (each a ``(B, C)`` array), matching the paper columns:
        ``rel_l2`` (``||d|| / ||t||``), ``rel_l1`` (``sum|d| / sum|t|``),
        ``rmse_over_gtmax`` and ``mae_over_gtmax`` (RMSE / MAE divided by the
        per-case per-channel ground-truth max magnitude), ``rmse``, ``mae``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must share shape; "
            f"got {tuple(pred.shape)} vs {tuple(target.shape)}."
        )
    ax = (-1, -2)
    n_cases, n_channels = int(pred.shape[0]), int(pred.shape[1])
    out = {k: np.empty((n_cases, n_channels), dtype=np.float64) for k in FIELD_METRICS}
    step = max(1, int(chunk_size))
    for start in range(0, n_cases, step):
        sl = slice(start, start + step)
        t = target[sl].astype(np.float64)
        diff = pred[sl].astype(np.float64) - t
        sq = diff * diff
        rmse = np.sqrt(sq.mean(axis=ax))
        mae = np.abs(diff).mean(axis=ax)
        out["rel_l2"][sl] = np.sqrt(sq.sum(axis=ax)) / np.maximum(
            np.sqrt((t * t).sum(axis=ax)), eps
        )
        out["rel_l1"][sl] = np.abs(diff).sum(axis=ax) / np.maximum(
            np.abs(t).sum(axis=ax), eps
        )
        gt_max = np.maximum(np.abs(t).max(axis=ax), eps)  # per case, per channel
        out["rmse_over_gtmax"][sl] = rmse / gt_max
        out["mae_over_gtmax"][sl] = mae / gt_max
        out["rmse"][sl] = rmse
        out["mae"][sl] = mae
    return out


def summarise(
    metrics: dict[str, np.ndarray],
    *,
    channel_names: tuple[str, ...] = SUPERWING_CHANNEL_NAMES,
) -> dict[str, dict[str, dict[str, float]]]:
    r"""Reduce per-case metrics to per-channel mean / median / std.

    Parameters
    ----------
    metrics : dict
        Output of :func:`per_case_field_metrics` — values of shape
        ``(B, C)``.
    channel_names : tuple of str, optional
        Display names matching the channel axis of ``metrics``.

    Returns
    -------
    dict
        Nested dict ``summary[channel][metric_name][stat]`` where
        ``stat`` is one of ``mean``, ``median``, ``std``.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ch_idx, ch_name in enumerate(channel_names):
        ch_dict: dict[str, dict[str, float]] = {}
        for metric_name, arr in metrics.items():
            col = arr[:, ch_idx]
            ch_dict[metric_name] = {
                "mean": float(col.mean()),
                "median": float(np.median(col)),
                "std": float(col.std(ddof=0)),
            }
        out[ch_name] = ch_dict
    return out


def _write_csv(
    *,
    out_path: Path,
    case_ids: np.ndarray,
    aoa_deg: np.ndarray,
    mach: np.ndarray,
    metrics: dict[str, np.ndarray],
    channel_names: tuple[str, ...],
) -> None:
    keys = [k for k in FIELD_METRICS if k in metrics]
    n_cases, n_channels = metrics["rel_l2"].shape
    header = ["case_id", "aoa_deg", "mach"]
    for ch in channel_names[:n_channels]:
        header += [f"{k}_{ch}" for k in keys]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n_cases):
            row: list[str] = [
                str(case_ids[i]),
                f"{float(aoa_deg[i]):.4f}",
                f"{float(mach[i]):.4f}",
            ]
            for ch_idx in range(n_channels):
                row += [f"{float(metrics[k][i, ch_idx]):.6f}" for k in keys]
            w.writerow(row)


def _write_summary_text(
    *,
    out_path: Path,
    summary: dict[str, dict[str, dict[str, float]]],
    n_cases: int,
) -> None:
    lines = [
        f"SuperWing test-split field metrics over {n_cases} cases",
        "",
        f"{'channel':<10s}  {'metric':<16s}  {'mean':>10s}  {'median':>10s}  {'std':>10s}",
        "-" * 62,
    ]
    for ch_name, metric_dict in summary.items():
        for metric_name in FIELD_METRICS:
            if metric_name not in metric_dict:
                continue
            stat = metric_dict[metric_name]
            lines.append(
                f"{ch_name:<10s}  {metric_name:<16s}  "
                f"{stat['mean']:>10.5f}  "
                f"{stat['median']:>10.5f}  "
                f"{stat['std']:>10.5f}"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.npz produced by inference.py.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV path. A `<stem>_summary.txt` is written next to it.",
    )
    return p.parse_args()


def main() -> None:
    """Command-line entry point — see module docstring."""
    args = _parse_args()
    preds = np.load(args.predictions, allow_pickle=False)

    pred_field = preds["pred_field"]
    target_field = preds["target_field"]
    metrics = per_case_field_metrics(pred=pred_field, target=target_field)
    summary = summarise(metrics)

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_path=out_csv,
        case_ids=preds["case_ids"],
        aoa_deg=preds["aoa_deg"],
        mach=preds["mach"],
        metrics=metrics,
        channel_names=SUPERWING_CHANNEL_NAMES,
    )
    print(f"Wrote {out_csv}")

    summary_path = out_csv.with_name(f"{out_csv.stem}_summary.txt")
    _write_summary_text(
        out_path=summary_path,
        summary=summary,
        n_cases=int(pred_field.shape[0]),
    )
    print(f"Wrote {summary_path}")
    print()
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
