# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
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

import torch
from torch.utils.data import DataLoader

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint

from datapipe import simsample_collate

EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Utilities
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimestepStats:
    """Statistics for a single timestep."""
    mean: float
    std: float
    min_val: float
    max_val: float
    filled_pct: float  # % of points with VOF > 0.5
    
    @classmethod
    def from_array(cls, arr: np.ndarray, threshold: float = 0.5) -> "TimestepStats":
        arr = arr.flatten()
        return cls(
            mean=float(arr.mean()),
            std=float(arr.std()),
            min_val=float(arr.min()),
            max_val=float(arr.max()),
            filled_pct=float((arr > threshold).sum() / len(arr) * 100),
        )


def print_header(title: str, width: int = 80):
    """Print a formatted header."""
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def print_subheader(title: str, width: int = 80):
    """Print a formatted subheader."""
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_config_table(config: dict, title: str = "Configuration"):
    """Print configuration as ASCII table."""
    print(f"\n  ┌{'─' * 60}┐")
    print(f"  │  {title:<56}  │")
    print(f"  ├{'─' * 60}┤")
    for key, value in config.items():
        print(f"  │  {key:<25}: {str(value):<30}  │")
    print(f"  └{'─' * 60}┘")


def print_prediction_stats_header():
    """Print header for prediction statistics table."""
    print("\n  ┌────────┬───────────────────────────────────┬───────────────────────────────────┬────────────────────┐")
    print("  │        │           PREDICTION              │           GROUND TRUTH            │       ERROR        │")
    print("  │ Step   ├─────────┬─────────┬───────────────┼─────────┬─────────┬───────────────┼──────────┬─────────┤")
    print("  │        │  Mean   │   Std   │ Range  │Fill% │  Mean   │   Std   │ Range  │Fill% │   MAE    │  RMSE   │")
    print("  ├────────┼─────────┼─────────┼────────┼──────┼─────────┼─────────┼────────┼──────┼──────────┼─────────┤")


def print_prediction_stats_row(
    timestep: int,
    pred_stats: TimestepStats,
    gt_stats: TimestepStats = None,
    mae: float = None,
    rmse: float = None,
):
    """Print a single row of prediction statistics."""
    pred_range = f"[{pred_stats.min_val:.2f},{pred_stats.max_val:.2f}]"
    
    if gt_stats is not None:
        gt_range = f"[{gt_stats.min_val:.2f},{gt_stats.max_val:.2f}]"
        print(f"  │ t={timestep:2d}   │ {pred_stats.mean:7.4f} │ {pred_stats.std:7.4f} │{pred_range:>8}│{pred_stats.filled_pct:5.1f}%│"
              f" {gt_stats.mean:7.4f} │ {gt_stats.std:7.4f} │{gt_range:>8}│{gt_stats.filled_pct:5.1f}%│"
              f" {mae:8.5f} │ {rmse:7.5f} │")
    else:
        print(f"  │ t={timestep:2d}   │ {pred_stats.mean:7.4f} │ {pred_stats.std:7.4f} │{pred_range:>8}│{pred_stats.filled_pct:5.1f}%│"
              f"    -    │    -    │   -    │  -   │"
              f"    -     │    -    │")


def print_prediction_stats_footer():
    """Print footer for prediction statistics table."""
    print("  └────────┴─────────┴─────────┴────────┴──────┴─────────┴─────────┴────────┴──────┴──────────┴─────────┘")


def print_summary_stats(
    total_mae: float,
    total_rmse: float,
    total_mse: float,
    num_timesteps: int,
    has_ground_truth: bool,
):
    """Print summary statistics."""
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │                    OVERALL STATISTICS                           │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  Total timesteps predicted:  {num_timesteps:<32} │")
    
    if has_ground_truth:
        print(f"  │  Mean Absolute Error (MAE):  {total_mae:<32.6f} │")
        print(f"  │  Root Mean Square Error:     {total_rmse:<32.6f} │")
        print(f"  │  Mean Square Error (MSE):    {total_mse:<32.6f} │")
    else:
        print(f"  │  Ground truth:               {'Not available':<32} │")
    
    print("  └─────────────────────────────────────────────────────────────────┘")


def print_run_summary(run_name: str, num_points: int, num_timesteps: int, output_dir: str):
    """Print summary for a single run."""
    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │                       RUN SUMMARY                               │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  Run name:          {run_name:<42} │")
    print(f"  │  Number of points:  {num_points:<42,} │")
    print(f"  │  Timesteps:         {num_timesteps:<42} │")
    print(f"  │  Output directory:  {output_dir:<42} │")
    print("  └─────────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _to_tensor(value, dtype=torch.float32) -> torch.Tensor:
    """Safely convert a value to a torch tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _stats_to_device(stats: dict, device: torch.device, dtype=torch.float32) -> dict:
    """Convert stats dict to tensors and move to device."""
    return {k: _to_tensor(v, dtype=dtype).to(device) for k, v in stats.items()}


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
    verbose: bool = True,
    gt_seq: list[torch.Tensor] | None = None,
) -> dict:
    """
    Save predicted VOF values to VTP files, preserving mesh structure.
    
    Returns:
        Dictionary with statistics for logging.
    """
    os.makedirs(output_dir, exist_ok=True)

    coords_np = coords.detach().cpu().numpy()
    N = coords_np.shape[0]
    T = len(preds)
    
    # Statistics storage
    all_pred_stats = []
    all_gt_stats = []
    all_mae = []
    all_rmse = []
    gt_available_count = 0
    
    # Try to find reference mesh
    reference_mesh = None
    reference_file = os.path.join(source_dir, f"{prefix}_000.vtp")
    
    if os.path.exists(reference_file):
        try:
            reference_mesh = pv.read(reference_file)
        except Exception as e:
            logging.warning(f"Could not read reference mesh: {e}")
    
    # Print statistics header
    if verbose:
        print_prediction_stats_header()
    
    for t in range(T):
        timestep = t + 1
        pred_np = preds[t].detach().cpu().numpy().squeeze(-1)
        
        if pred_np.shape[0] != N:
            logging.warning(f"Point mismatch at t={timestep}")
            continue

        # Compute prediction statistics
        pred_stats = TimestepStats.from_array(pred_np)
        all_pred_stats.append(pred_stats)
        
        # Try to read source file
        source_file = os.path.join(source_dir, f"{prefix}_{timestep:03d}.vtp")
        
        mesh = None
        gt_np = None
        gt_stats = None
        mae = None
        rmse = None
        # Prefer dataset-provided ground truth if available
        if compute_error and gt_seq is not None and len(gt_seq) >= timestep:
            try:
                gt_np = gt_seq[t].detach().cpu().numpy().squeeze()
                gt_stats = TimestepStats.from_array(gt_np)
                all_gt_stats.append(gt_stats)
                error = pred_np - gt_np
                mae = float(np.abs(error).mean())
                rmse = float(np.sqrt((error ** 2).mean()))
                all_mae.append(mae)
                all_rmse.append(rmse)
                gt_available_count += 1
            except Exception:
                pass
        
        if os.path.exists(source_file):
            try:
                mesh = pv.read(source_file)
                
                if mesh.n_points != N:
                    mesh = None
                else:
                    # Extract ground truth
                    #if compute_error:
                    if True:
                        for key in ["epoxy_vof", f"epoxy_vof_step{timestep:02d}", "vof"]:
                            if key in mesh.point_data:
                                gt_np = np.array(mesh.point_data[key]).squeeze()
                                break
                        
                        if gt_np is not None and gt_np.shape[0] == N:
                            gt_stats = TimestepStats.from_array(gt_np)
                            all_gt_stats.append(gt_stats)
                            
                            # Compute errors
                            error = pred_np - gt_np
                            mae = float(np.abs(error).mean())
                            rmse = float(np.sqrt((error ** 2).mean()))
                            all_mae.append(mae)
                            all_rmse.append(rmse)
                            gt_available_count += 1
                            
            except Exception as e:
                logging.warning(f"Could not read source mesh {source_file}: {e}")
        
        # Print row statistics
        if verbose:
            print_prediction_stats_row(timestep, pred_stats, gt_stats, mae, rmse)
        
        # Prepare mesh for saving
        if mesh is None:
            if reference_mesh is not None and reference_mesh.n_points == N:
                mesh = reference_mesh.copy()
            else:
                mesh = pv.PolyData(coords_np)

        # Clear and add point data
        mesh.point_data.clear()
        mesh.point_data["epoxy_vof_pred"] = pred_np
        
        if gt_np is not None and gt_np.shape[0] == N:
            mesh.point_data["epoxy_vof_exact"] = gt_np
            mesh.point_data["epoxy_vof_error"] = pred_np - gt_np
            mesh.point_data["epoxy_vof_abs_error"] = np.abs(pred_np - gt_np)

        # Save
        out_file = os.path.join(output_dir, f"{prefix}_{timestep:03d}_pred.vtp")
        mesh.save(out_file)
    
    # Print footer
    if verbose:
        print_prediction_stats_footer()
    
    # Compute overall statistics
    stats = {
        "num_timesteps": T,
        "gt_available_count": gt_available_count,
        "has_ground_truth": gt_available_count > 0,
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

    def __init__(self, cfg: DictConfig, logger: PythonLogger, dist: DistributedManager):
        self.cfg = cfg
        self.logger = logger
        self.dist = dist
        self.device = dist.device

        # Print initialization header
        if dist.rank == 0:
            print_header("TRANSOLVER VOF INFERENCE")
        
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

        # Print configuration
        if dist.rank == 0:
            print_config_table({
                "Checkpoint": ckpt_path,
                "Output directory": self.output_dir,
                "VTP prefix": self.vtp_prefix,
                "Rollout steps": self.rollout_steps,
                "Compute error": self.compute_error,
                "Device": str(self.device),
            }, title="Inference Configuration")

        self.logger.info(f"[Rank {dist.rank}] Loaded checkpoint {ckpt_path}")

    @torch.no_grad()
    def run_on_single_run(self, run_path: str):
        """Process a single run directory."""
        run_name = os.path.basename(run_path)
        
        if self.verbose:
            print_subheader(f"Processing Run: {run_name}")
        
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

            # Print normalization stats
            if self.verbose:
                print("\n  Normalization Statistics:")
                print(f"    Position mean: [{pos_mean[0].item():.6f}, {pos_mean[1].item():.6f}, {pos_mean[2].item():.6f}]")
                print(f"    Position std:  [{pos_std[0].item():.6f}, {pos_std[1].item():.6f}, {pos_std[2].item():.6f}]")
                print(f"    VOF mean:      {vof_mean.item():.6f}")
                print(f"    VOF std:       {vof_std.item():.6f}")

            for local_idx, sample in enumerate(dataloader):
                if isinstance(sample, list):
                    sample = sample[0]
                sample = sample.to(self.device)

                # Get input statistics
                input_vof = sample.node_features["features"].cpu().numpy().flatten()
                input_vof_denorm = (input_vof * vof_std.cpu().item() + vof_mean.cpu().item())
                
                if self.verbose:
                    print("\n  Input (t=0) Statistics:")
                    print(f"    VOF mean:  {input_vof_denorm.mean():.6f}")
                    print(f"    VOF std:   {input_vof_denorm.std():.6f}")
                    print(f"    VOF range: [{input_vof_denorm.min():.4f}, {input_vof_denorm.max():.4f}]")
                    print(f"    Filled:    {(input_vof_denorm > 0.5).sum() / len(input_vof_denorm) * 100:.1f}%")

                # Forward rollout
                if self.verbose:
                    print("\n  Running autoregressive rollout...")
                
                pred_seq = self.model(sample=sample, data_stats=data_stats)

                # Denormalize
                coords_norm = sample.node_features["coords"]
                coords = denormalize_coords(coords_norm, pos_mean, pos_std)

                pred_seq_denorm = [
                    denormalize_vof(pred_seq[t], vof_mean, vof_std)
                    for t in range(pred_seq.size(0))
                ]

                # Build ground-truth sequence from dataset target (denormalized)
                gt_seq_denorm = (
                    sample.node_target.transpose(0, 1).unsqueeze(-1)
                )
                gt_seq_denorm = [
                    denormalize_vof(gt_seq_denorm[t], vof_mean, vof_std)
                    for t in range(gt_seq_denorm.size(0))
                ]

                N = coords.size(0)
                T_pred = len(pred_seq_denorm)

                # Save and get statistics
                if self.write_vtp:
                    out_dir = os.path.join(
                        self.output_dir, f"rank{self.dist.rank}", run_name
                    )
                    
                    stats = save_vtp_predictions(
                        coords=coords,
                        preds=pred_seq_denorm,
                        source_dir=run_path,
                        output_dir=out_dir,
                        prefix=self.vtp_prefix,
                        compute_error=self.compute_error,
                        verbose=self.verbose,
                        gt_seq=gt_seq_denorm,
                    )

                    # Print summary
                    if self.verbose:
                        print_summary_stats(
                            total_mae=stats.get("total_mae", 0),
                            total_rmse=stats.get("total_rmse", 0),
                            total_mse=stats.get("total_mse", 0),
                            num_timesteps=stats["num_timesteps"],
                            has_ground_truth=stats["has_ground_truth"],
                        )
                        
                        print_run_summary(
                            run_name=run_name,
                            num_points=N,
                            num_timesteps=T_pred,
                            output_dir=out_dir,
                        )

                    self.logger.info(f"[Rank {self.dist.rank}] Saved to {out_dir}")

            self.logger.info(f"[Rank {self.dist.rank}] Finished run: {run_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main inference entry point."""
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("inference")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Discover run directories
    parent_dir = to_absolute_path(cfg.inference.raw_data_dir_test)
    
    if not os.path.isdir(parent_dir):
        logger0.error(f"Parent directory not found: {parent_dir}")
        return

    run_dirs = [d.path for d in os.scandir(parent_dir) if d.is_dir()]
    run_dirs.sort()

    if len(run_dirs) == 0:
        logger0.error(f"No run directories found under: {parent_dir}")
        return

    # Print discovery summary
    if dist.rank == 0:
        print_header("DATA DISCOVERY")
        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  Data directory: {parent_dir:<45} │")
        print(f"  │  Found {len(run_dirs)} run(s):{' ' * 51}│")
        for i, run_dir in enumerate(run_dirs[:10]):  # Show first 10
            run_name = os.path.basename(run_dir)
            print(f"  │    {i+1:2d}. {run_name:<55} │")
        if len(run_dirs) > 10:
            print(f"  │    ... and {len(run_dirs) - 10} more{' ' * 47}│")
        print(f"  └─────────────────────────────────────────────────────────────────┘")

    logger0.info(f"Found {len(run_dirs)} runs under {parent_dir}")

    # Distribute runs across ranks
    my_runs = run_dirs[dist.rank :: dist.world_size]
    
    if dist.rank == 0:
        print(f"\n  Distribution across {dist.world_size} rank(s):")
        print(f"    Rank {dist.rank}: {len(my_runs)} run(s) assigned")
    
    logger.info(f"[Rank {dist.rank}] Assigned {len(my_runs)} runs.")

    # Create worker and process
    worker = InferenceWorker(cfg, logger, dist)

    for i, run_path in enumerate(my_runs):
        if dist.rank == 0:
            print(f"\n  Progress: {i+1}/{len(my_runs)} runs")
        worker.run_on_single_run(run_path)

    # Final summary
    if dist.rank == 0:
        print_header("INFERENCE COMPLETE")
        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  ✓ Successfully processed {len(my_runs)} run(s){' ' * 35}│")
        print(f"  │  ✓ Output saved to: {worker.output_dir:<41} │")
        print(f"  └─────────────────────────────────────────────────────────────────┘\n")
        
    logger0.info("Inference completed successfully.")


if __name__ == "__main__":
    main()

