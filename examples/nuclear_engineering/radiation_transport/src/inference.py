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

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from tqdm import tqdm

from physicsnemo.datapipes import DataLoader

from checkpointing import load_model_from_checkpoint
from dataset import load_flux_stats
from evaluation_metrics import (
    aggregate_metrics,
    aggregate_qoi,
    compute_metrics,
    compute_sample_qoi,
)
from loader import build_dataloaders, collate_no_padding
from transforms import denormalize_flux
from viz import plot_flux_panels, plot_qoi_true_vs_pred

from physicsnemo.distributed import DistributedManager


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    flux_stats: Dict[str, float],
    case_type: str,
    use_amp: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[
    Tuple[
        np.ndarray,
        np.ndarray,
        Optional[Dict[str, Dict[str, float]]],
        Optional[np.ndarray],
        Optional[str],
    ]
]:
    """Yield ``(prediction, target, qoi, coordinates, filename)`` per sample.

    Predictions and targets are denormalized to physical-flux units and
    returned as flattened numpy arrays for downstream pointwise metrics and
    plotting. The QoI dict (or ``None``) is computed on-device before the
    GPU->CPU transfer to avoid round-tripping per-mesh tensors through numpy.
    ``coordinates`` is the per-sample point cloud (or ``None`` if absent);
    ``filename`` is the sidecar filename (or ``None``).
    """
    model.eval()
    n = 0

    for batch in tqdm(dataloader, desc="evaluating"):
        if max_samples is not None and n >= max_samples:
            break
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

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

        coords_t = batch.get("coordinates_unnormalized")
        cell_areas_t = batch.get("cell_areas")
        sigma_t_t = batch.get("sigma_t")
        sigma_s_t = batch.get("sigma_s")
        raw_meta = batch.get("metadata") or {}
        if isinstance(raw_meta, list):
            raw_meta = raw_meta[0] if raw_meta else {}

        # Batches always carry an outer batch dim of 1 (collate_no_padding).
        for b in range(pred.shape[0]):
            pred_phys_t = denormalize_flux(pred[b].squeeze(-1), stats).flatten()
            target_phys_t = denormalize_flux(target[b].squeeze(-1), stats).flatten()

            qoi: Optional[Dict[str, Dict[str, float]]] = None
            if (
                coords_t is not None
                and cell_areas_t is not None
                and sigma_t_t is not None
                and sigma_s_t is not None
            ):
                qoi = compute_sample_qoi(
                    pred_phys_t,
                    target_phys_t,
                    coords_t[b],
                    cell_areas_t[b],
                    sigma_t_t[b],
                    sigma_s_t[b],
                    batch,
                    case_type,
                )

            coords_np: Optional[np.ndarray] = None
            if coords_t is not None:
                coords_np = coords_t[b].detach().cpu().numpy()
            filename = raw_meta.get("filename") if isinstance(raw_meta, dict) else None

            n += 1
            yield (
                pred_phys_t.detach().cpu().numpy(),
                target_phys_t.detach().cpu().numpy(),
                qoi,
                coords_np,
                filename,
            )
            if max_samples is not None and n >= max_samples:
                return


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry: load checkpoint, run evaluation, write metrics + figures."""
    DistributedManager.initialize()
    # Full-mesh evaluation always — disable any training-time subsampling.
    OmegaConf.update(cfg, "model.num_spatial_points", -1)

    output_dir = Path(cfg.inference.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.inference.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Downstream calls (load_model_from_checkpoint, build_dataloaders ->
    # MeshDataReader / split-file loader, load_flux_stats) all raise
    # ``FileNotFoundError`` with the offending path if anything is missing.
    model, _ = load_model_from_checkpoint(
        Path(cfg.inference.checkpoint_path), cfg, device
    )

    # Build the test loader. ``test_batch_size=1`` matches the point-cloud
    # adapter's invariant.
    loaders, _ = build_dataloaders(
        cfg,
        dist=None,
        collate_fn=collate_no_padding,
        phases=("test",),
        test_batch_size=1,
    )
    test_loader = loaders["test"]
    print(f"Test set size: {len(test_loader.dataset)}")

    flux_stats = load_flux_stats(cfg.data.flux_normalization_stats_file)
    case_type = cfg.case.type

    # Evenly sample plot indices across the test set.
    n_total = cfg.inference.num_samples or len(test_loader.dataset)
    n_plots = cfg.inference.num_plot_samples
    plot_indices: set[int] = set()
    if n_plots > 0:
        plot_indices = set(np.linspace(0, n_total - 1, n_plots, dtype=int).tolist())

    per_sample_metrics: list[Dict[str, float]] = []
    per_sample_qoi: list[Dict[str, Dict[str, float]]] = []
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for idx, (pred, target, qoi, coords, _filename) in enumerate(
        run_evaluation(
            model,
            test_loader,
            device,
            flux_stats,
            case_type,
            use_amp=cfg.inference.use_amp,
            max_samples=cfg.inference.num_samples,
        )
    ):
        per_sample_metrics.append(compute_metrics(pred, target))
        if qoi is not None:
            per_sample_qoi.append(qoi)
        all_targets.append(target)
        all_preds.append(pred)

        if idx in plot_indices and coords is not None:
            plot_flux_panels(
                coords,
                target,
                pred,
                figures_dir / f"flux_panels_{idx:04d}.png",
                log_flux=case_type == "lattice",
            )

    if not per_sample_metrics:
        raise RuntimeError("No samples evaluated; check the test split / data path.")

    # Aggregate metrics over every sample (concatenate first for global stats).
    all_target_arr = np.concatenate(all_targets)
    all_pred_arr = np.concatenate(all_preds)
    overall_metrics = compute_metrics(all_pred_arr, all_target_arr)
    aggregated = aggregate_metrics(per_sample_metrics)

    metrics_out: Dict[str, Any] = {
        "num_samples": len(per_sample_metrics),
        "overall": overall_metrics,
        "per_sample_aggregate": aggregated,
    }
    with open(output_dir / "metrics.yaml", "w") as f:
        yaml.safe_dump(metrics_out, f, sort_keys=False)
    print("\nMetrics:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.6e}")

    # QoI summary.
    if per_sample_qoi:
        qoi_summary = aggregate_qoi(per_sample_qoi)
        with open(output_dir / "qoi_metrics.yaml", "w") as f:
            yaml.safe_dump(qoi_summary, f, sort_keys=False)
        plot_qoi_true_vs_pred(per_sample_qoi, figures_dir / "qoi_true_vs_pred.png")
        print("\nQoI summary:")
        for region, stats in qoi_summary.items():
            print(
                f"  {region}: mae={stats['mae']:.4e}, "
                f"mean_rel_err={stats['mean_relative_error_pct']:.3f}%"
            )

    print(f"\nResults written to: {output_dir}")
    print("  metrics.yaml")
    if per_sample_qoi:
        print("  qoi_metrics.yaml")


if __name__ == "__main__":
    main()
