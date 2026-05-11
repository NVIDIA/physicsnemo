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
Inference Script for Transolver VOF Prediction.

Loads trained model and runs inference on test data, saving predicted
VOF values to VTP files with preserved mesh structure for visualization.
"""

import os
import sys
import logging
import tempfile
from dataclasses import dataclass

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig
from tabulate import tabulate

import torch
from torch.utils.data import DataLoader

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint

from datapipe import simsample_collate


EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# Small tensor helpers
# (duplicated in train.py by design — keeping each script self-contained)
# ═══════════════════════════════════════════════════════════════════════════════


def _to_tensor(value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a scalar / list / numpy array / tensor to a torch tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _stats_to_device(
    stats: dict, device: torch.device, dtype: torch.dtype = torch.float32
) -> dict:
    """Convert a stats dict of tensors/lists to tensors on the given device."""
    return {k: _to_tensor(v, dtype=dtype).to(device) for k, v in stats.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Per-timestep statistics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TimestepStats:
    """Statistics for a single timestep."""

    mean: float
    std: float
    min_val: float
    max_val: float
    filled_pct: float  # % of points with VOF > threshold

    @classmethod
    def from_array(cls, arr: np.ndarray, threshold: float = 0.5) -> "TimestepStats":
        """Build statistics from a flattened VOF (or similar) array."""
        arr = arr.flatten()
        return cls(
            mean=float(arr.mean()),
            std=float(arr.std()),
            min_val=float(arr.min()),
            max_val=float(arr.max()),
            filled_pct=float((arr > threshold).sum() / len(arr) * 100),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tabulate-based logging helpers
# ═══════════════════════════════════════════════════════════════════════════════


def log_section(logger, title: str) -> None:
    """Log a section header."""
    width = max(60, len(title) + 4)
    logger.info("")
    logger.info("=" * width)
    logger.info(f"  {title}")
    logger.info("=" * width)


def log_config(logger, title: str, config: dict) -> None:
    """Log a key/value configuration block using tabulate."""
    logger.info("")
    logger.info(f"[{title}]")
    rows = [[k, str(v)] for k, v in config.items()]
    logger.info(tabulate(rows, tablefmt="presto"))


def log_per_step_stats(
    logger,
    per_step_rows: list[dict],
    has_gt: bool,
) -> None:
    """
    Log per-timestep prediction statistics as a table.

    Each row dict has:
        step, pred_mean, pred_std, pred_min, pred_max, pred_filled,
        gt_mean?, gt_std?, gt_min?, gt_max?, gt_filled?, mae?, rmse?
    """
    if has_gt:
        headers = [
            "t",
            "pred mean",
            "pred std",
            "pred range",
            "pred fill%",
            "gt mean",
            "gt std",
            "gt range",
            "gt fill%",
            "mae",
            "rmse",
        ]
        rows = [
            [
                r["step"],
                f"{r['pred_mean']:.4f}",
                f"{r['pred_std']:.4f}",
                f"[{r['pred_min']:.2f},{r['pred_max']:.2f}]",
                f"{r['pred_filled']:.1f}",
                f"{r['gt_mean']:.4f}" if r.get("gt_mean") is not None else "-",
                f"{r['gt_std']:.4f}" if r.get("gt_std") is not None else "-",
                f"[{r['gt_min']:.2f},{r['gt_max']:.2f}]"
                if r.get("gt_min") is not None
                else "-",
                f"{r['gt_filled']:.1f}" if r.get("gt_filled") is not None else "-",
                f"{r['mae']:.5f}" if r.get("mae") is not None else "-",
                f"{r['rmse']:.5f}" if r.get("rmse") is not None else "-",
            ]
            for r in per_step_rows
        ]
    else:
        headers = [
            "t",
            "pred mean",
            "pred std",
            "pred range",
            "pred fill%",
        ]
        rows = [
            [
                r["step"],
                f"{r['pred_mean']:.4f}",
                f"{r['pred_std']:.4f}",
                f"[{r['pred_min']:.2f},{r['pred_max']:.2f}]",
                f"{r['pred_filled']:.1f}",
            ]
            for r in per_step_rows
        ]

    logger.info("")
    logger.info(tabulate(rows, headers=headers, tablefmt="presto"))


# ═══════════════════════════════════════════════════════════════════════════════
# Denormalization helpers
# ═══════════════════════════════════════════════════════════════════════════════


def denormalize_vof(
    y: torch.Tensor, vof_mean: torch.Tensor, vof_std: torch.Tensor
) -> torch.Tensor:
    """Denormalize VOF predictions."""
    if y.ndim == 2:
        return y * vof_std.view(1, -1) + vof_mean.view(1, -1)
    elif y.ndim == 3:
        return y * vof_std.view(1, 1, -1) + vof_mean.view(1, 1, -1)
    else:
        raise AssertionError(f"Expected [N,1] or [T,N,1], got {y.shape}")


def denormalize_coords(
    coords: torch.Tensor, pos_mean: torch.Tensor, pos_std: torch.Tensor
) -> torch.Tensor:
    """Denormalize coordinates [N, 3]."""
    return coords * pos_std.view(1, -1) + pos_mean.view(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# VTP Saving with Statistics
# ═══════════════════════════════════════════════════════════════════════════════


def save_vtp_predictions(
    coords: torch.Tensor,
    preds: list[torch.Tensor],
    source_dir: str,
    output_dir: str,
    prefix: str = "frame",
    compute_error: bool = True,
    gt_seq: list[torch.Tensor] | None = None,
) -> dict:
    """
    Save predicted VOF values to VTP files, preserving mesh structure.

    Ground-truth resolution order:
      1. Dataset-provided ``gt_seq`` (denormalized tensor list).
      2. VOF arrays found inside the source VTP file on disk.

    Only the first source that succeeds is used so that statistics are
    never double-counted.

    Returns:
        Dictionary with statistics and the rows used for logging.
    """
    os.makedirs(output_dir, exist_ok=True)

    coords_np = coords.detach().cpu().numpy()
    N = coords_np.shape[0]
    T = len(preds)

    all_mae: list[float] = []
    all_rmse: list[float] = []
    gt_available_count = 0
    per_step_rows: list[dict] = []

    reference_mesh = None
    reference_file = os.path.join(source_dir, f"{prefix}_000.vtp")
    if os.path.exists(reference_file):
        try:
            reference_mesh = pv.read(reference_file)
        except Exception as e:
            logging.warning(f"Could not read reference mesh: {e}")

    for t in range(T):
        timestep = t + 1
        pred_np = preds[t].detach().cpu().numpy().squeeze(-1)

        if pred_np.shape[0] != N:
            logging.warning(f"Point mismatch at t={timestep}")
            continue

        pred_stats = TimestepStats.from_array(pred_np)

        gt_np = None
        gt_stats = None
        mae = None
        rmse = None
        mesh = None
        source_file = os.path.join(source_dir, f"{prefix}_{timestep:03d}.vtp")

        # ── Ground-truth source 1: dataset-provided gt_seq ──────────────
        if compute_error and gt_seq is not None and len(gt_seq) >= timestep:
            try:
                gt_np = gt_seq[t].detach().cpu().numpy().squeeze()
                gt_stats = TimestepStats.from_array(gt_np)
                error = pred_np - gt_np
                mae = float(np.abs(error).mean())
                rmse = float(np.sqrt((error**2).mean()))
                all_mae.append(mae)
                all_rmse.append(rmse)
                gt_available_count += 1
            except Exception:
                gt_np = None
                gt_stats = None

        # ── Load source VTP for mesh structure (and fallback GT) ────────
        if os.path.exists(source_file):
            try:
                mesh = pv.read(source_file)
                if mesh.n_points != N:
                    mesh = None
                elif gt_stats is None and compute_error:
                    for key in [
                        "epoxy_vof",
                        f"epoxy_vof_step{timestep:02d}",
                        "vof",
                    ]:
                        if key in mesh.point_data:
                            gt_np = np.array(mesh.point_data[key]).squeeze()
                            break

                    if gt_np is not None and gt_np.shape[0] == N:
                        gt_stats = TimestepStats.from_array(gt_np)
                        error = pred_np - gt_np
                        mae = float(np.abs(error).mean())
                        rmse = float(np.sqrt((error**2).mean()))
                        all_mae.append(mae)
                        all_rmse.append(rmse)
                        gt_available_count += 1
            except Exception as e:
                logging.warning(f"Could not read source mesh {source_file}: {e}")

        # Record statistics for later tabulate logging
        row = {
            "step": timestep,
            "pred_mean": pred_stats.mean,
            "pred_std": pred_stats.std,
            "pred_min": pred_stats.min_val,
            "pred_max": pred_stats.max_val,
            "pred_filled": pred_stats.filled_pct,
        }
        if gt_stats is not None:
            row.update(
                {
                    "gt_mean": gt_stats.mean,
                    "gt_std": gt_stats.std,
                    "gt_min": gt_stats.min_val,
                    "gt_max": gt_stats.max_val,
                    "gt_filled": gt_stats.filled_pct,
                    "mae": mae,
                    "rmse": rmse,
                }
            )
        per_step_rows.append(row)

        # Build mesh and save prediction
        if mesh is None:
            if reference_mesh is not None and reference_mesh.n_points == N:
                mesh = reference_mesh.copy()
            else:
                mesh = pv.PolyData(coords_np)

        mesh.point_data.clear()
        mesh.point_data["epoxy_vof_pred"] = pred_np
        if gt_np is not None and gt_np.shape[0] == N:
            mesh.point_data["epoxy_vof_exact"] = gt_np
            mesh.point_data["epoxy_vof_error"] = pred_np - gt_np
            mesh.point_data["epoxy_vof_abs_error"] = np.abs(pred_np - gt_np)

        out_file = os.path.join(output_dir, f"{prefix}_{timestep:03d}_pred.vtp")
        mesh.save(out_file)

    stats = {
        "num_timesteps": T,
        "gt_available_count": gt_available_count,
        "has_ground_truth": gt_available_count > 0,
        "per_step_rows": per_step_rows,
    }
    if gt_available_count > 0:
        stats["total_mae"] = float(np.mean(all_mae))
        stats["total_rmse"] = float(np.mean(all_rmse))
        stats["total_mse"] = float(np.mean([r**2 for r in all_rmse]))

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Worker
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceWorker:
    """Inference worker for Transolver VOF prediction."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: PythonLogger,
        dist: DistributedManager,
    ):
        self.cfg = cfg
        self.logger = logger
        self.dist = dist
        self.device = dist.device

        if dist.rank == 0:
            log_section(self.logger, "TRANSOLVER VOF INFERENCE")

        # Build and load model
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.device)
        self.model.eval()

        ckpt_path = cfg.training.ckpt_path
        load_checkpoint(ckpt_path, models=self.model, device=self.device)

        # Configuration
        self.vtp_prefix = cfg.inference.get("vtp_prefix", "frame")
        self.write_vtp = cfg.inference.get("write_vtp", True)
        self.compute_error = cfg.inference.get("compute_error", True)
        self.output_dir = cfg.inference.get("output_dir", "./predictions")
        self.verbose = cfg.inference.get("verbose", True)

        self.rollout_steps = cfg.training.num_time_steps - 1
        self.num_workers = cfg.training.num_dataloader_workers

        if dist.rank == 0:
            log_config(
                self.logger,
                "Inference configuration",
                {
                    "checkpoint": ckpt_path,
                    "output dir": self.output_dir,
                    "vtp prefix": self.vtp_prefix,
                    "rollout steps": self.rollout_steps,
                    "compute error": self.compute_error,
                    "device": str(self.device),
                },
            )

        self.logger.info(f"[Rank {dist.rank}] Loaded checkpoint {ckpt_path}")

    @torch.no_grad()
    def run_on_single_run(self, run_path: str):
        """Process a single run directory."""
        run_name = os.path.basename(run_path)

        if self.verbose and self.dist.rank == 0:
            log_section(self.logger, f"Processing run: {run_name}")

        self.logger.info(f"[Rank {self.dist.rank}] Processing run: {run_name}")

        with tempfile.TemporaryDirectory() as tmpdir:
            symlink_path = os.path.join(tmpdir, run_name)
            os.symlink(run_path, symlink_path)

            dataset = instantiate(
                self.cfg.datapipe,
                name="vof_inference",
                split="test",
                num_steps=self.cfg.training.num_time_steps,
                num_samples=1,
                logger=self.logger,
                data_dir=symlink_path,
            )

            data_stats = dict(
                node=_stats_to_device(dataset.node_stats, self.device),
                feature=_stats_to_device(dataset.feature_stats, self.device),
            )

            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=simsample_collate,
            )

            pos_mean = data_stats["node"]["pos_mean"]
            pos_std = data_stats["node"]["pos_std"]
            vof_mean = data_stats["feature"]["feature_mean"]
            vof_std = data_stats["feature"]["feature_std"]

            if self.verbose and self.dist.rank == 0:
                log_config(
                    self.logger,
                    "Normalization statistics",
                    {
                        "position mean": [
                            f"{pos_mean[i].item():.6f}" for i in range(3)
                        ],
                        "position std": [f"{pos_std[i].item():.6f}" for i in range(3)],
                        "vof mean": f"{vof_mean.item():.6f}",
                        "vof std": f"{vof_std.item():.6f}",
                    },
                )

            for local_idx, sample in enumerate(dataloader):
                if isinstance(sample, list):
                    sample = sample[0]
                sample = sample.to(self.device)

                input_vof = sample.node_features["features"].cpu().numpy().flatten()
                input_vof_denorm = (
                    input_vof * vof_std.cpu().item() + vof_mean.cpu().item()
                )

                if self.verbose and self.dist.rank == 0:
                    log_config(
                        self.logger,
                        "Input (t=0)",
                        {
                            "vof mean": f"{input_vof_denorm.mean():.6f}",
                            "vof std": f"{input_vof_denorm.std():.6f}",
                            "vof range": f"[{input_vof_denorm.min():.4f}, {input_vof_denorm.max():.4f}]",
                            "filled (%)": f"{(input_vof_denorm > 0.5).sum() / len(input_vof_denorm) * 100:.1f}",
                        },
                    )

                # ── Autoregressive rollout ──────────────────────────────
                pred_seq = self.model(sample=sample, data_stats=data_stats)

                # ── Denormalize predictions ─────────────────────────────
                coords_norm = sample.node_features["coords"]
                coords = denormalize_coords(coords_norm, pos_mean, pos_std)
                pred_seq_denorm = [
                    denormalize_vof(pred_seq[t], vof_mean, vof_std)
                    for t in range(pred_seq.size(0))
                ]

                # ── Denormalize ground truth for comparison ─────────────
                gt_seq_denorm = sample.node_target.transpose(0, 1).unsqueeze(-1)
                gt_seq_denorm = [
                    denormalize_vof(gt_seq_denorm[t], vof_mean, vof_std)
                    for t in range(gt_seq_denorm.size(0))
                ]

                N = coords.size(0)
                T_pred = len(pred_seq_denorm)

                if self.write_vtp:
                    out_dir = os.path.join(
                        self.output_dir,
                        f"rank{self.dist.rank}",
                        run_name,
                    )

                    stats = save_vtp_predictions(
                        coords=coords,
                        preds=pred_seq_denorm,
                        source_dir=run_path,
                        output_dir=out_dir,
                        prefix=self.vtp_prefix,
                        compute_error=self.compute_error,
                        gt_seq=gt_seq_denorm,
                    )

                    if self.verbose and self.dist.rank == 0:
                        log_per_step_stats(
                            self.logger,
                            stats["per_step_rows"],
                            has_gt=stats["has_ground_truth"],
                        )

                        overall = {
                            "timesteps predicted": stats["num_timesteps"],
                        }
                        if stats["has_ground_truth"]:
                            overall.update(
                                {
                                    "mae": f"{stats['total_mae']:.6f}",
                                    "rmse": f"{stats['total_rmse']:.6f}",
                                    "mse": f"{stats['total_mse']:.6f}",
                                }
                            )
                        else:
                            overall["ground truth"] = "not available"
                        log_config(self.logger, "Overall statistics", overall)

                        log_config(
                            self.logger,
                            "Run summary",
                            {
                                "run name": run_name,
                                "num points": f"{N:,}",
                                "timesteps": T_pred,
                                "output dir": out_dir,
                            },
                        )

                    self.logger.info(f"[Rank {self.dist.rank}] Saved to {out_dir}")

            self.logger.info(f"[Rank {self.dist.rank}] Finished run: {run_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Hydra entry point: discover test runs, shard across ranks, run inference."""
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("inference")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()
    logging.getLogger().setLevel(logging.INFO)

    parent_dir = to_absolute_path(cfg.inference.raw_data_dir_test)
    if not os.path.isdir(parent_dir):
        logger0.error(f"Parent directory not found: {parent_dir}")
        return

    run_dirs = sorted(d.path for d in os.scandir(parent_dir) if d.is_dir())
    if len(run_dirs) == 0:
        logger0.error(f"No run directories found under: {parent_dir}")
        return

    if dist.rank == 0:
        log_section(logger0, "Data discovery")
        preview = [[i + 1, os.path.basename(d)] for i, d in enumerate(run_dirs[:10])]
        logger0.info(tabulate(preview, headers=["#", "run"], tablefmt="presto"))
        if len(run_dirs) > 10:
            logger0.info(f"... and {len(run_dirs) - 10} more")
        logger0.info("")
        logger0.info(f"Data directory: {parent_dir}")
        logger0.info(
            f"Found {len(run_dirs)} run(s), "
            f"distributed across {dist.world_size} rank(s)."
        )

    logger0.info(f"Found {len(run_dirs)} runs under {parent_dir}")

    my_runs = run_dirs[dist.rank :: dist.world_size]
    logger.info(f"[Rank {dist.rank}] Assigned {len(my_runs)} runs.")

    worker = InferenceWorker(cfg, logger, dist)

    for i, run_path in enumerate(my_runs):
        if dist.rank == 0:
            logger0.info(f"\nProgress: {i + 1}/{len(my_runs)} runs")
        worker.run_on_single_run(run_path)

    if dist.rank == 0:
        log_section(logger0, "Inference complete")
        log_config(
            logger0,
            "Summary",
            {
                "processed runs": len(my_runs),
                "output dir": worker.output_dir,
            },
        )

    logger0.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
