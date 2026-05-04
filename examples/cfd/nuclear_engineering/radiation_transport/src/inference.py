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

"""Inference / evaluation: load checkpoint, run on test set, compute metrics + plots.

Standalone CLI invoked after training. Loads the Hydra config that was saved
alongside the checkpoint, builds the test dataloader, runs forward passes,
denormalizes predictions to physical-flux units, computes pointwise + QoI
metrics, and emits a few canonical plots.

Usage::

    python src/inference.py \\
        --checkpoint_dir outputs/.../checkpoints/best_qoi \\
        --data_path /path/to/data_root \\
        --case_type lattice
"""

import argparse
import os
import pathlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Flat sibling imports — keep this module self-contained relative to ``src/``.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dataset import load_flux_stats  # noqa: E402
from loader import build_dataloaders, collate_no_padding  # noqa: E402
from transforms import denormalize_flux  # noqa: E402

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils.checkpoint import load_checkpoint  # noqa: E402


# =========================================================================
# Checkpoint loading
# =========================================================================


def load_hydra_config(checkpoint_dir: Union[str, Path]) -> DictConfig:
    """Load the Hydra config saved next to a checkpoint.

    Walks up from ``checkpoint_dir`` looking for a ``hydra/config.yaml``;
    this lets users point at either the run directory or a specific
    ``checkpoints/best_*`` subdirectory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    search = checkpoint_dir
    for _ in range(4):
        config_path = search / "hydra" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = OmegaConf.create(yaml.safe_load(f))
            OmegaConf.resolve(cfg)
            return cfg
        if search == search.parent:
            break
        search = search.parent
    raise FileNotFoundError(
        f"No hydra/config.yaml found in {checkpoint_dir} or its ancestors"
    )


def find_best_checkpoint(run_dir: Union[str, Path]) -> Path:
    """Find the default checkpoint directory under a training run.

    Explicit checkpoint directories are consumed by the caller. When a run or
    ``checkpoints`` directory is supplied instead, default to ``top_model``.
    """
    run_dir = Path(run_dir)
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        # User may already have pointed at the checkpoints dir.
        checkpoint_root = run_dir

    top = checkpoint_root / "top_model"
    if top.exists() and list(top.glob("checkpoint.0.*.pt")):
        return top

    raise FileNotFoundError(
        f"No top_model checkpoint found under {checkpoint_root}. Pass a specific "
        "checkpoint directory to evaluate something other than top_model."
    )


def load_model_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, DictConfig, Dict[str, Any]]:
    """Build the Transolver model from the saved Hydra config and load weights.

    Returns (model in eval mode, resolved config, metadata dict).
    """
    import hydra

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    cfg = load_hydra_config(checkpoint_dir)

    _initialize_distributed_manager()

    # Build model from cfg.model. Strip RTE-specific keys consumed elsewhere.
    cfg_model = OmegaConf.to_container(cfg.model, resolve=True)
    for k in ("num_spatial_points", "include_q_in_embedding"):
        cfg_model.pop(k, None)
    model = hydra.utils.instantiate(cfg_model).to(device)

    metadata: Dict[str, Any] = {}
    epoch = load_checkpoint(
        path=str(checkpoint_dir),
        models=model,
        metadata_dict=metadata,
        device=device,
    )
    metadata.setdefault("epoch", epoch)

    model.eval()
    print(
        f"Loaded model from {checkpoint_dir} "
        f"(epoch={metadata.get('epoch', '?')}, "
        f"params={sum(p.numel() for p in model.parameters()):,})"
    )
    return model, cfg, metadata


def _initialize_distributed_manager() -> None:
    """Use distributed init only for torchrun/explicit distributed launches."""
    if DistributedManager.is_initialized():
        return

    explicit_method = os.getenv("PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD")
    torchrun_env = os.getenv("RANK") is not None and os.getenv("WORLD_SIZE") is not None
    openmpi_env = os.getenv("OMPI_COMM_WORLD_RANK") is not None

    if explicit_method or torchrun_env or openmpi_env:
        DistributedManager.initialize()
        return

    DistributedManager._shared_state["_is_initialized"] = True
    dist = DistributedManager()
    dist._initialization_method = "single"
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.device)


# =========================================================================
# Metrics
# =========================================================================


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def l2_relative_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> float:
    """Sample-wise L2 relative error: ||pred - target||_2 / ||target||_2."""
    num = np.linalg.norm(pred.flatten() - target.flatten())
    den = np.linalg.norm(target.flatten()) + eps
    return float(num / den)


def relative_error(pred: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> float:
    """Mean pointwise relative error |pred - target| / (|target| + eps)."""
    return float(np.mean(np.abs(pred - target) / (np.abs(target) + eps)))


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute the full metric panel for one (pred, target) pair."""
    p, t = pred.flatten(), target.flatten()
    return {
        "mse": mse(p, t),
        "rmse": rmse(p, t),
        "mae": mae(p, t),
        "l2_relative_error": l2_relative_error(p, t),
        "relative_error": relative_error(p, t),
        "max_error": float(np.max(np.abs(p - t))),
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


# =========================================================================
# QoI (numpy side)
# =========================================================================


def _extract_geometry_params(filename: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse hohlraum geometry parameters out of a simulation filename."""
    if not filename:
        return None
    patterns = {
        "cx": r"cx([-\d.]+)",
        "cy": r"cy([-\d.]+)",
        "ulr": r"ulr([-\d.]+)",
        "llr": r"llr([-\d.]+)",
        "urr": r"urr([-\d.]+)",
        "lrr": r"lrr([-\d.]+)",
        "hlr": r"hlr([-\d.]+)",
        "hrr": r"hrr([-\d.]+)",
    }
    params: Dict[str, float] = {}
    for key, pat in patterns.items():
        m = re.search(pat, filename)
        if m:
            params[key] = float(m.group(1).rstrip("."))
    return params if params else None


def evaluate_lattice_qoi(
    cell_centers: np.ndarray,
    cell_areas: np.ndarray,
    sigma_t: np.ndarray,
    sigma_s: np.ndarray,
    scalar_flux: np.ndarray,
) -> Dict[str, float]:
    """Lattice absorption QoI in the absorbing blocks. Matches KiT-RT.

    ``scalar_flux`` is shape ``(N,)`` for a single steady-state snapshot.
    Returns ``{"cur_absorption": ...}``.
    """
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    sigma_a = sigma_t - sigma_s

    xy_corrector = -3.5
    lbounds = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + xy_corrector
    ubounds = np.array([2.0, 3.0, 4.0, 5.0, 6.0]) + xy_corrector
    in_absorption = np.zeros_like(x, dtype=bool)
    for k in range(5):
        for l in range(5):  # noqa: E741
            if (l + k) % 2 == 1:
                continue
            if (k == 2 and l == 2) or (k == 2 and l == 4):
                continue
            in_absorption |= (
                (x >= lbounds[k])
                & (x <= ubounds[k])
                & (y >= lbounds[l])
                & (y <= ubounds[l])
            )

    flux = scalar_flux.flatten()
    absorption_density = flux * sigma_a * cell_areas
    return {"cur_absorption": float(np.sum(absorption_density * in_absorption))}


def evaluate_hohlraum_qoi(
    cell_centers: np.ndarray,
    cell_areas: np.ndarray,
    sigma_t: np.ndarray,
    sigma_s: np.ndarray,
    scalar_flux: np.ndarray,
    geometry_params: Dict[str, float],
) -> Dict[str, float]:
    """Hohlraum absorption QoI (center / vertical / horizontal). Matches KiT-RT.

    ``scalar_flux`` is shape ``(N,)``. ``geometry_params`` carries ``cx``,
    ``cy``, ``hlr``, ``hrr``, ``llr``, ``ulr``, ``urr``.
    """
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]

    cx = geometry_params["cx"]
    cy = geometry_params["cy"]
    hlr = geometry_params["hlr"]
    hrr = geometry_params["hrr"]
    llr = geometry_params["llr"]
    ulr = geometry_params["ulr"]
    urr = geometry_params["urr"]

    sigma_a = sigma_t - sigma_s

    in_center = (x > -0.2 + cx) & (x < 0.2 + cx) & (y > -0.4 + cy) & (y < 0.4 + cy)
    # Note: KiT-RT uses ``llr`` for both vertical-wall lower bounds.
    in_vertical = ((x < hlr) & (y > llr) & (y < ulr)) | (
        (x > hrr) & (y > llr) & (y < urr)
    )
    in_horizontal = (y > 0.6) | (y < -0.6)

    flux = scalar_flux.flatten()
    absorption_density = flux * sigma_a * cell_areas
    return {
        "cur_absorption_center": float(np.sum(absorption_density * in_center)),
        "cur_absorption_vertical": float(np.sum(absorption_density * in_vertical)),
        "cur_absorption_horizontal": float(np.sum(absorption_density * in_horizontal)),
    }


def compute_sample_qoi(
    pred: np.ndarray,
    target: np.ndarray,
    metadata: Dict[str, Any],
    case_type: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Compute QoI(pred) vs QoI(target) for one sample.

    Returns a dict ``{region: {predicted, ground_truth, absolute_error,
    relative_error_pct}}`` or ``None`` if metadata is incomplete.
    """
    coords = metadata.get("coordinates")
    cell_areas = metadata.get("cell_areas")
    sigma_t = metadata.get("sigma_t")
    sigma_s = metadata.get("sigma_s")
    if coords is None or cell_areas is None or sigma_t is None or sigma_s is None:
        return None

    if case_type == "lattice":
        qp = evaluate_lattice_qoi(coords, cell_areas, sigma_t, sigma_s, pred)
        qt = evaluate_lattice_qoi(coords, cell_areas, sigma_t, sigma_s, target)
    elif case_type == "hohlraum":
        gp = _extract_geometry_params(metadata.get("filename"))
        if gp is None:
            return None
        qp = evaluate_hohlraum_qoi(coords, cell_areas, sigma_t, sigma_s, pred, gp)
        qt = evaluate_hohlraum_qoi(coords, cell_areas, sigma_t, sigma_s, target, gp)
    else:
        raise ValueError(f"Unknown case_type: {case_type}")

    out: Dict[str, Dict[str, float]] = {}
    for region in qp:
        p, t = qp[region], qt[region]
        abs_err = abs(p - t)
        out[region] = {
            "predicted": p,
            "ground_truth": t,
            "absolute_error": abs_err,
            "relative_error_pct": abs_err / (abs(t) + 1e-10) * 100.0,
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


def collect_qoi_series(
    per_sample_qoi: list[Dict[str, Dict[str, float]]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Collect per-component QoI arrays and add a total for multi-component QoIs."""
    component_names: list[str] = []
    for sample in per_sample_qoi:
        for name in sample:
            if name not in component_names:
                component_names.append(name)

    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in component_names:
        target_vals = []
        pred_vals = []
        for sample in per_sample_qoi:
            entry = sample.get(name)
            if entry is None:
                continue
            target_vals.append(entry["ground_truth"])
            pred_vals.append(entry["predicted"])
        if target_vals:
            series[name] = (np.array(target_vals), np.array(pred_vals))

    if len(series) > 1:
        totals_target = []
        totals_pred = []
        for sample in per_sample_qoi:
            if not all(name in sample for name in series):
                continue
            totals_target.append(sum(sample[name]["ground_truth"] for name in series))
            totals_pred.append(sum(sample[name]["predicted"] for name in series))
        if totals_target:
            series["total"] = (np.array(totals_target), np.array(totals_pred))

    return series


def summarize_qoi_series(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    """Summarize absolute and relative QoI errors for one component."""
    abs_errs = np.abs(prediction - target)
    rel_errs = abs_errs / (np.abs(target) + 1e-10) * 100.0
    return {
        "num_samples": int(target.size),
        "mae": float(np.mean(abs_errs)),
        "rmse": float(np.sqrt(np.mean(abs_errs**2))),
        "max_error": float(np.max(abs_errs)),
        "mean_relative_error_pct": float(np.mean(rel_errs)),
        "median_relative_error_pct": float(np.median(rel_errs)),
        "max_relative_error_pct": float(np.max(rel_errs)),
    }


# =========================================================================
# Plots
# =========================================================================


def plot_flux_panels(
    coordinates: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Union[str, Path],
    *,
    log_flux: bool = False,
    figsize: Tuple[int, int] = (16, 5),
    dpi: int = 150,
) -> Path:
    """Render a 3-panel figure: target | prediction | absolute error."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = target.flatten()
    prediction = prediction.flatten()
    error = np.abs(prediction - target)

    x, y = coordinates[:, 0], coordinates[:, 1]
    x_pad = (x.max() - x.min()) * 0.01
    y_pad = (y.max() - y.min()) * 0.01
    xlim = (x.min() - x_pad, x.max() + x_pad)
    ylim = (y.min() - y_pad, y.max() + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    flux_vmin = min(target.min(), prediction.min())
    flux_vmax = max(target.max(), prediction.max())
    flux_norm = None
    if log_flux:
        positive_flux = np.concatenate(
            [target[target > 0.0], prediction[prediction > 0.0]]
        )
        if positive_flux.size:
            flux_vmin = float(positive_flux.min())
            flux_vmax = float(positive_flux.max())
            if flux_vmin == flux_vmax:
                flux_vmax = flux_vmin * 1.01
            flux_norm = LogNorm(vmin=flux_vmin, vmax=flux_vmax)
        else:
            log_flux = False
    cmap_flux = plt.get_cmap("viridis")
    cmap_err = plt.get_cmap("hot")

    for ax, label, vals, cmap, vmin, vmax, norm in (
        (axes[0], "Target", target, cmap_flux, flux_vmin, flux_vmax, flux_norm),
        (
            axes[1],
            "Prediction",
            prediction,
            cmap_flux,
            flux_vmin,
            flux_vmax,
            flux_norm,
        ),
        (axes[2], "Absolute Error", error, cmap_err, 0.0, float(error.max()), None),
    ):
        plot_vals = np.clip(vals, flux_vmin, None) if norm is not None else vals
        sc = ax.scatter(
            x,
            y,
            c=plot_vals,
            cmap=cmap,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
            s=1,
        )
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{label} (log)" if norm is not None else label)
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_true_vs_pred_scatter(
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Union[str, Path],
    *,
    max_points: int = 200_000,
    dpi: int = 150,
) -> Path:
    """Scatter of predicted vs ground truth with the y=x reference line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = target.flatten()
    p = prediction.flatten()
    if t.size > max_points:
        idx = np.random.default_rng(0).choice(t.size, max_points, replace=False)
        t, p = t[idx], p[idx]

    lo = float(min(t.min(), p.min()))
    hi = float(max(t.max(), p.max()))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.scatter(t, p, s=1, alpha=0.3)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y = x")
    ax.set_xlabel("Ground truth flux")
    ax.set_ylabel("Predicted flux")
    ax.set_title("Predicted vs. ground-truth flux")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_histogram(
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Union[str, Path],
    *,
    bins: int = 80,
    dpi: int = 150,
) -> Path:
    """Histogram of pointwise absolute errors (log y)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors = np.abs(prediction.flatten() - target.flatten())

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    ax.hist(errors, bins=bins, color="C0", edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("|prediction - target|")
    ax.set_ylabel("Count (log)")
    ax.set_title(
        f"Pointwise error histogram (mean={errors.mean():.3e}, "
        f"max={errors.max():.3e})"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _subplot_grid(num_panels: int) -> Tuple[int, int]:
    """Choose a compact subplot grid for QoI component plots."""
    ncols = min(num_panels, 3)
    nrows = int(np.ceil(num_panels / ncols))
    return nrows, ncols


def plot_qoi_true_vs_pred(
    qoi_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Union[str, Path],
    *,
    dpi: int = 150,
) -> Path:
    """Scatter predicted vs ground-truth QoI values for each component."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(qoi_series.items())
    nrows, ncols = _subplot_grid(len(items))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), dpi=dpi, squeeze=False
    )

    for ax, (name, (target, prediction)) in zip(axes.flat, items):
        lo = float(min(target.min(), prediction.min()))
        hi = float(max(target.max(), prediction.max()))
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1e-12)
            lo -= pad
            hi += pad

        ax.scatter(target, prediction, s=18, alpha=0.75)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y = x")
        ax.set_title(name)
        ax.set_xlabel("Ground truth QoI")
        ax.set_ylabel("Predicted QoI")
        ax.set_aspect("equal")
        ax.legend(loc="best")

    for ax in axes.flat[len(items) :]:
        ax.axis("off")

    fig.suptitle("QoI predicted vs. ground truth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_qoi_error_histograms(
    qoi_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Union[str, Path],
    *,
    bins: int = 40,
    dpi: int = 150,
) -> Path:
    """Plot absolute QoI error histograms for each component."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(qoi_series.items())
    nrows, ncols = _subplot_grid(len(items))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.0 * nrows), dpi=dpi, squeeze=False
    )

    for ax, (name, (target, prediction)) in zip(axes.flat, items):
        errors = np.abs(prediction - target)
        ax.hist(errors, bins=bins, color="C0", edgecolor="black", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{name} error")
        ax.set_xlabel("|prediction - target|")
        ax.set_ylabel("Count (log)")

    for ax in axes.flat[len(items) :]:
        ax.axis("off")

    fig.suptitle("QoI absolute error histograms")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# =========================================================================
# Main inference loop
# =========================================================================


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


def _denormalize(flux_norm: torch.Tensor, stats: Dict[str, float]) -> np.ndarray:
    """Apply ``denormalize_flux`` (RTEFluxLogClip + Normalize inverse)."""
    return denormalize_flux(flux_norm.detach().cpu(), stats).numpy()


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    flux_stats: Dict[str, float],
    *,
    use_amp: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Yield ``(prediction, target, metadata)`` for each test sample.

    Predictions and targets are returned as flattened numpy arrays in
    physical-flux units (denormalized). ``metadata`` carries the per-sample
    coordinates / cell areas / sigma fields / filename needed for QoI and
    plotting.
    """
    model.eval()
    n = 0

    for batch in tqdm(dataloader, desc="evaluating"):
        if max_samples is not None and n >= max_samples:
            break
        batch = _move_to_device(batch, device)

        amp_enabled = use_amp and device.type == "cuda"
        with autocast(device_type=device.type, enabled=amp_enabled):
            pred = model(fx=batch["fx"], embedding=batch["embedding"])
        pred = pred.float()
        target = batch["flux_target"].float()

        # Denormalize back to physical flux. ``denormalize_flux`` handles the
        # full RTEFluxLogClip + Normalize inverse using the stats dict that
        # the dataset transform recorded on the sample.
        stats = batch.get("flux_normalization_stats", flux_stats)
        if isinstance(stats, list):
            stats = stats[0] if stats else flux_stats

        # Batches always carry an outer batch dim of 1 (collate_no_padding).
        for b in range(pred.shape[0]):
            pred_phys = _denormalize(pred[b].squeeze(-1), stats).flatten()
            target_phys = _denormalize(target[b].squeeze(-1), stats).flatten()

            metadata: Dict[str, Any] = {}
            coords = batch.get("coordinates_unnormalized")
            if coords is None:
                coords = batch.get("fx")
            if coords is not None:
                metadata["coordinates"] = coords[b].detach().cpu().numpy()
            for key in ("cell_areas", "sigma_t", "sigma_s"):
                if key in batch:
                    metadata[key] = batch[key][b].detach().cpu().numpy().flatten()
            sim_time = batch.get("sim_time")
            if sim_time is not None:
                metadata["sim_time"] = float(sim_time[b].flatten()[0].item())

            raw_meta = batch.get("metadata") or {}
            if isinstance(raw_meta, list):
                raw_meta = raw_meta[0] if raw_meta else {}
            filename = raw_meta.get("filename") if isinstance(raw_meta, dict) else None
            if filename:
                metadata["filename"] = filename

            n += 1
            yield pred_phys, target_phys, metadata
            if max_samples is not None and n >= max_samples:
                return


# =========================================================================
# CLI entry
# =========================================================================


def _resolve_data_path(
    cfg: DictConfig,
    cli_data_path: str,
    split_file: Union[str, Path],
) -> None:
    """Override ``case.data_root`` and ``data.input_dir`` to the user-supplied path."""
    OmegaConf.update(cfg, "case.data_root", cli_data_path, force_add=True)
    case_type = cfg.case.type
    OmegaConf.update(
        cfg, "case.data_path", str(Path(cli_data_path) / case_type), force_add=True
    )
    OmegaConf.update(
        cfg,
        "data.input_dir",
        str(Path(cli_data_path) / case_type),
        force_add=True,
    )
    flux_stats_file = (
        Path(cli_data_path) / "stats" / f"{case_type}_flux_stats.yaml"
    )
    OmegaConf.update(
        cfg,
        "data.flux_normalization_stats_file",
        str(flux_stats_file),
        force_add=True,
    )
    split_file = Path(split_file)
    OmegaConf.update(cfg, "case.split_file", str(split_file), force_add=True)
    OmegaConf.update(cfg, "data.split_file", str(split_file), force_add=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RTE Transolver model on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Path to a checkpoint directory (e.g. .../checkpoints/best_qoi). "
        "May also point at the run directory; the script will search.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Dataset root containing <case>/, splits/, and stats/.",
    )
    parser.add_argument(
        "--case_type",
        type=str,
        required=True,
        choices=("lattice", "hohlraum"),
        help="Which benchmark to evaluate.",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        required=True,
        help="Explicit train/val/test split JSON to use for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write metrics + figures. "
        "Defaults to <run_dir>/evaluation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap on the number of test samples (default: all).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Test DataLoader workers."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override torch device."
    )
    parser.add_argument(
        "--num_plot_samples",
        type=int,
        default=3,
        help="Number of per-sample flux-panel figures to render.",
    )
    args = parser.parse_args()

    # Resolve checkpoint: accept either the run dir or a specific checkpoint.
    ckpt_dir = args.checkpoint_dir
    if not list(ckpt_dir.glob("checkpoint.0.*.pt")):
        ckpt_dir = find_best_checkpoint(ckpt_dir)
        print(f"Using best checkpoint: {ckpt_dir}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, cfg, metadata = load_model_from_checkpoint(ckpt_dir, device=device)

    # The saved cfg still holds the training-machine paths. Rewrite the
    # data-related fields to whatever the user supplied at the CLI.
    if cfg.case.type != args.case_type:
        print(
            f"Warning: checkpoint trained on '{cfg.case.type}', "
            f"but --case_type={args.case_type}. Using --case_type."
        )
        OmegaConf.update(cfg, "case.type", args.case_type, force_add=True)
    if not args.split_file.exists():
        raise FileNotFoundError(f"Split file not found: {args.split_file}")
    _resolve_data_path(cfg, str(args.data_path), args.split_file)

    num_spatial_points = cfg.model.get("num_spatial_points", -1)
    if num_spatial_points != -1:
        print(
            "Warning: evaluation will use the checkpoint's "
            f"num_spatial_points={num_spatial_points}; field metrics and QoI "
            "are computed on that subsampled point set."
        )

    # Output dir defaults to ``<run_dir>/evaluation``.
    if args.output_dir is None:
        run_dir = ckpt_dir
        # Walk up to a directory that has a hydra/ subdirectory.
        for _ in range(4):
            if (run_dir / "hydra").exists():
                break
            run_dir = run_dir.parent
        output_dir = run_dir / "evaluation"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Build the test loader. ``test_batch_size=1`` matches the point-cloud
    # adapter's invariant.
    loaders, _ = build_dataloaders(
        cfg,
        dist=None,
        adapter="transolver",
        collate_fn=collate_no_padding,
        phases=("test",),
        test_batch_size=1,
        test_num_workers=args.num_workers,
    )
    test_loader = loaders["test"]
    print(f"Test set size: {len(test_loader.dataset)}")

    flux_stats = load_flux_stats(cfg.data.flux_normalization_stats_file)

    # Run the inference loop and accumulate metrics + plots.
    per_sample_metrics: list[Dict[str, float]] = []
    per_sample_qoi: list[Dict[str, Dict[str, float]]] = []
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    plot_indices = set()
    if args.num_plot_samples > 0:
        # Evenly sample plot indices across the test set.
        n_total = (
            args.num_samples
            if args.num_samples is not None
            else len(test_loader.dataset)
        )
        step = max(n_total // max(args.num_plot_samples, 1), 1)
        plot_indices = set(range(0, n_total, step))

    for idx, (pred, target, meta) in enumerate(
        run_evaluation(
            model,
            test_loader,
            device,
            flux_stats,
            max_samples=args.num_samples,
        )
    ):
        per_sample_metrics.append(compute_metrics(pred, target))
        qoi = compute_sample_qoi(pred, target, meta, args.case_type)
        if qoi is not None:
            per_sample_qoi.append(qoi)
        all_targets.append(target)
        all_preds.append(pred)

        if idx in plot_indices and "coordinates" in meta:
            plot_flux_panels(
                meta["coordinates"],
                target,
                pred,
                figures_dir / f"flux_panels_{idx:04d}.png",
                log_flux=args.case_type == "lattice",
            )

    if not per_sample_metrics:
        raise RuntimeError("No samples evaluated; check the test split / data path.")

    # Aggregate metrics over every sample (concatenate first for global stats).
    all_target_arr = np.concatenate(all_targets)
    all_pred_arr = np.concatenate(all_preds)
    overall_metrics = compute_metrics(all_pred_arr, all_target_arr)
    aggregated = aggregate_metrics(per_sample_metrics)

    metrics_out = {
        "num_samples": len(per_sample_metrics),
        "overall": overall_metrics,
        "per_sample_aggregate": aggregated,
    }
    with open(output_dir / "metrics.yaml", "w") as f:
        yaml.safe_dump(metrics_out, f, sort_keys=False)
    print(f"\nMetrics:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.6e}")

    # QoI summary.
    if per_sample_qoi:
        qoi_summary = aggregate_qoi(per_sample_qoi)
        qoi_series = collect_qoi_series(per_sample_qoi)
        if "total" in qoi_series:
            total_target, total_prediction = qoi_series["total"]
            qoi_summary["total"] = summarize_qoi_series(
                total_target, total_prediction
            )
        with open(output_dir / "qoi_metrics.yaml", "w") as f:
            yaml.safe_dump(qoi_summary, f, sort_keys=False)
        plot_qoi_true_vs_pred(qoi_series, figures_dir / "qoi_true_vs_pred.png")
        plot_qoi_error_histograms(qoi_series, figures_dir / "qoi_error_histogram.png")
        print("\nQoI summary:")
        for region, stats in qoi_summary.items():
            print(
                f"  {region}: mae={stats['mae']:.4e}, "
                f"mean_rel_err={stats['mean_relative_error_pct']:.3f}%"
            )

    # Global plots over every concatenated point.
    plot_true_vs_pred_scatter(
        all_target_arr, all_pred_arr, figures_dir / "true_vs_pred.png"
    )
    plot_error_histogram(
        all_target_arr, all_pred_arr, figures_dir / "error_histogram.png"
    )

    print(f"\nResults written to: {output_dir}")
    print(f"  metrics.yaml")
    if per_sample_qoi:
        print(f"  qoi_metrics.yaml")
    print(f"  figures/ ({len(plot_indices)} flux panels + 2 global plots)")


if __name__ == "__main__":
    main()
