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

from typing import Any, Dict, Optional

import numpy as np
import torch

from qoi import (
    evaluate_hohlraum_qoi_torch,
    evaluate_lattice_qoi_torch,
    extract_geometry_params,
)

__all__ = [
    "compute_metrics",
    "aggregate_metrics",
    "compute_sample_qoi",
    "aggregate_qoi",
]


def compute_metrics(
    pred: np.ndarray, target: np.ndarray, eps: float = 1e-10
) -> Dict[str, float]:
    """Compute the full metric panel for one ``(pred, target)`` pair."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    diff = pred_flat - target_flat
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(abs_diff)),
        "l2_relative_error": float(
            np.linalg.norm(diff) / (np.linalg.norm(target_flat) + eps)
        ),
        "relative_error": float(np.mean(abs_diff / (np.abs(target_flat) + eps))),
        "max_error": float(np.max(abs_diff)),
    }


def aggregate_metrics(per_sample: list[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-sample metrics into mean/min/max."""
    if not per_sample:
        return {}
    keys = per_sample[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [s[k] for s in per_sample]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
        out[f"{k}_min"] = float(np.min(vals))
        out[f"{k}_max"] = float(np.max(vals))
    return out


def compute_sample_qoi(
    pred: torch.Tensor,
    target: torch.Tensor,
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    sample: Any,
    case_type: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute QoI(pred) vs QoI(target) for one sample on the tensors' device.

    All tensor inputs may live on GPU; only the scalar QoI values are
    materialized to host (via ``.item()``). Returns ``{region: {predicted,
    ground_truth, absolute_error, relative_error_pct}}`` or ``None`` for the
    hohlraum case when geometry params are missing from ``sample``.

    Args:
        sample: For ``case_type="hohlraum"``, a per-sample mapping carrying
            the eight 0-D float32 geometry tensors (``ulr`` ... ``cy``),
            typically the batch sliced at index ``b``, or a fresh dict
            built from those entries by the caller. Ignored for lattice.
    """
    # The QoI evaluators expect ``(1, N)`` batched flux + flat (N,) cell fields.
    pred_batched = pred.float().reshape(1, -1)
    target_batched = target.float().reshape(1, -1)
    centers = cell_centers.float()
    areas = cell_areas.float().flatten()
    sigma_t_flat = sigma_t.float().flatten()
    sigma_s_flat = sigma_s.float().flatten()
    # Placeholder — the final-time QoI evaluators accept ``sim_times`` only
    # for callsite uniformity with the time-dependent variants.
    sim_times = torch.zeros(1, device=pred.device)

    if case_type == "lattice":
        qoi_pred = evaluate_lattice_qoi_torch(
            centers, areas, sigma_t_flat, sigma_s_flat, pred_batched, sim_times
        )
        qoi_target = evaluate_lattice_qoi_torch(
            centers, areas, sigma_t_flat, sigma_s_flat, target_batched, sim_times
        )
    elif case_type == "hohlraum":
        geometry_params = extract_geometry_params(sample)
        if not geometry_params:
            return None
        qoi_pred = evaluate_hohlraum_qoi_torch(
            centers,
            areas,
            sigma_t_flat,
            sigma_s_flat,
            pred_batched,
            sim_times,
            geometry_params,
        )
        qoi_target = evaluate_hohlraum_qoi_torch(
            centers,
            areas,
            sigma_t_flat,
            sigma_s_flat,
            target_batched,
            sim_times,
            geometry_params,
        )
    else:
        raise ValueError(f"Unknown case_type: {case_type}")

    out: Dict[str, Dict[str, float]] = {}
    for region in qoi_pred:
        pred_value = float(qoi_pred[region][0].item())
        target_value = float(qoi_target[region][0].item())
        abs_err = abs(pred_value - target_value)
        out[region] = {
            "predicted": pred_value,
            "ground_truth": target_value,
            "absolute_error": abs_err,
            "relative_error_pct": abs_err / (abs(target_value) + 1e-10) * 100.0,
        }
    return out


def aggregate_qoi(
    per_sample_qoi: list[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-sample QoI dicts into per-region summary statistics."""
    by_region: Dict[str, list] = {}
    for sample in per_sample_qoi:
        if not sample:
            continue
        for region, entry in sample.items():
            by_region.setdefault(region, []).append(entry)

    summary: Dict[str, Dict[str, float]] = {}
    for region, entries in by_region.items():
        abs_errs = np.array([e["absolute_error"] for e in entries])
        rel_errs = np.array([e["relative_error_pct"] for e in entries])
        summary[region] = {
            "num_samples": len(entries),
            "mae": float(np.mean(abs_errs)),
            "rmse": float(np.sqrt(np.mean(abs_errs**2))),
            "max_error": float(np.max(abs_errs)),
            "mean_relative_error_pct": float(np.mean(rel_errs)),
            "median_relative_error_pct": float(np.median(rel_errs)),
            "max_relative_error_pct": float(np.max(rel_errs)),
        }
    return summary
