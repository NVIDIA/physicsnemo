#!/usr/bin/env python3
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
Evaluate a trained model on the Norne field test dataset.

Supports two evaluation modes:
  full_mapping:   Feed all timesteps at once (single forward pass).
  autoregressive: Roll out step-by-step (L context -> K predicted),
                  matching the XMGN autoregressive inference protocol.

Automatically detects TNO variant and feedback-channel usage from the
saved checkpoint config so you don't need to specify them manually.

Usage:
    # Pressure evaluation (auto-detect model variant from checkpoint)
    python scripts/evaluate_norne.py --variable pressure
or  sbatch eval_norne.sbatch pressure

    # Saturation (SWAT)
    python scripts/evaluate_norne.py --variable swat
or  sbatch eval_norne.sbatch swat

    # Custom checkpoint + mode
    python scripts/evaluate_norne.py --checkpoint path/to/model.pth --mode autoregressive --L 1 --K 3
    CHECKPOINT=checkpoints/best_model_swat_deeponet3d_tno_spatial.pth sbatch eval_norne.sbatch swat

    NORMALIZE=0 sbatch eval_norne.sbatch pressure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse

import numpy as np
import torch
from data.dataloader import ReservoirDataset
from training.ar_utils import ar_validate_full_rollout
from training.metrics import (
    compute_r2_score,
    compute_relative_l2_error,
)
from utils.checkpoint import build_model_from_config


def load_model(model_config, device):
    """Reconstruct model from saved config."""
    model, _ = build_model_from_config(model_config, device=device)
    return model


def print_metrics(
    all_predictions, all_targets, variable, num_timesteps, spatial_mask=None
):
    """Compute and print all metrics in XMGN-compatible format."""
    if spatial_mask is not None:
        m = spatial_mask
        for _ in range(all_predictions.ndim - m.ndim):
            m = (
                m[np.newaxis]
                if m.ndim < all_predictions.ndim - 1
                else m[..., np.newaxis]
            )
        m = np.broadcast_to(m, all_predictions.shape)
        pf, gf = all_predictions[m], all_targets[m]
    else:
        pf, gf = all_predictions.ravel(), all_targets.ravel()
    overall_mae = np.mean(np.abs(pf - gf))
    overall_mse = np.mean((pf - gf) ** 2)
    overall_rmse = np.sqrt(overall_mse)
    overall_rel_l2 = compute_relative_l2_error(pf, gf)
    overall_r2 = compute_r2_score(pf, gf)

    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test samples processed: {all_predictions.shape[0]}")
    print(f"Overall MAE:           {overall_mae:.6e}")
    print(f"Overall MSE:           {overall_mse:.6e}")
    print(f"Overall RMSE:          {overall_rmse:.6e}")
    print(f"Relative L2 Error:     {overall_rel_l2:.6e}")
    print(f"R2 Score:              {overall_r2:.6f}")
    print()

    print("Per-Variable Metrics:")
    print("-" * 70)
    print(
        f"  {variable:>12s}  |  MAE: {overall_mae:>12.6e}  |  RMSE: {overall_rmse:>12.6e}"
    )
    print()

    print("Per-Timestep Metrics (averaged over all samples):")
    print(f"{'t':>4s} | {'MAE':>12s} | {'MSE':>12s} | {'RMSE':>12s}")
    print("-" * 50)
    for t in range(num_timesteps):
        pred_t = all_predictions[..., t]
        gt_t = all_targets[..., t]
        if spatial_mask is not None:
            mt = np.broadcast_to(spatial_mask[np.newaxis], pred_t.shape)
            pt, gt = pred_t[mt], gt_t[mt]
        else:
            pt, gt = pred_t.ravel(), gt_t.ravel()
        t_mae = np.mean(np.abs(pt - gt))
        t_mse = np.mean((pt - gt) ** 2)
        t_rmse = np.sqrt(t_mse)
        print(f"{t:4d} | {t_mae:12.6e} | {t_mse:12.6e} | {t_rmse:12.6e}")

    print()
    print("Per-Sample Summary (first 10 and last 5):")
    print(
        f"{'sample':>6s} | {'MAE':>12s} | {'RMSE':>12s} | {'RelL2':>12s} | {'R2':>8s}"
    )
    print("-" * 60)
    n = all_predictions.shape[0]
    show_idx = list(range(min(10, n))) + list(range(max(10, n - 5), n))
    for i in show_idx:
        pi, gi = all_predictions[i], all_targets[i]
        if spatial_mask is not None:
            ms = np.broadcast_to(spatial_mask[..., np.newaxis], pi.shape)
            pi_f, gi_f = pi[ms], gi[ms]
        else:
            pi_f, gi_f = pi.ravel(), gi.ravel()
        s_mae = np.mean(np.abs(pi_f - gi_f))
        s_rmse = np.sqrt(np.mean((pi_f - gi_f) ** 2))
        s_rel_l2 = compute_relative_l2_error(pi_f, gi_f)
        s_r2 = compute_r2_score(pi_f, gi_f)
        print(
            f"{i:6d} | {s_mae:12.6e} | {s_rmse:12.6e} | {s_rel_l2:12.6e} | {s_r2:8.4f}"
        )

    print()
    print("=" * 70)


def main():
    """Run evaluation on the test set and print metrics."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on Norne test set (XMGN-compatible metrics)"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/norne",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="norne_{mode}_a.pt",
        help="Input file pattern ({mode} replaced with train/val/test)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file pattern. If not set, inferred from --variable: "
        "pressure->norne_{mode}_pressure.pt, swat->norne_{mode}_swat.pt, sgas->norne_{mode}_sgas.pt",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="pressure",
        choices=["pressure", "swat", "sgas"],
        help="Variable to evaluate (default: pressure)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--tno", action="store_true", help="TNO mode: feed predictions back as branch2"
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Auto-detect ACTNUM and evaluate on active cells only",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Enable feedback channel (append previous prediction as extra input). "
        "Auto-detected from checkpoint if feedback_channel is saved.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full_mapping",
        choices=["full_mapping", "autoregressive"],
        help="full_mapping: single forward pass. autoregressive: AR rollout.",
    )
    parser.add_argument(
        "--L", type=int, default=1, help="AR input window (context timesteps)"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="AR output window (predicted timesteps per step)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Load data normalized (for models trained with normalize=true). "
        "Metrics are reported on denormalized (physical) values.",
    )
    args = parser.parse_args()

    # Infer output file from variable if not explicitly set
    VARIABLE_FILE_MAP = {
        "pressure": "norne_{mode}_pressure.pt",
        "swat": "norne_{mode}_swat.pt",
        "sgas": "norne_{mode}_sgas.pt",
    }
    if args.output_file is None:
        args.output_file = VARIABLE_FILE_MAP.get(
            args.variable, f"norne_{{mode}}_{args.variable}.pt"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Checkpoint --
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_dir = Path("checkpoints")
        candidates = sorted(
            ckpt_dir.glob("best_model_*.pth"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            raise FileNotFoundError("No checkpoints found. Specify --checkpoint.")
        ckpt_path = candidates[-1]

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_config = checkpoint["model_config"]

    # Auto-detect TNO variant and feedback channel from checkpoint
    is_tno = model_config.get("variant", "") == "tno"
    feedback_channel = model_config.get("feedback_channel", None)
    if args.tno:
        is_tno = True
    if args.feedback:
        feedback_channel = 1

    print("=" * 70)
    print("NEURAL OPERATOR FACTORY - NORNE EVALUATION")
    print("=" * 70)
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Model:       {model_config['model_arch_name']}")
    print(f"Epoch:       {checkpoint['epoch']}")
    print(f"Val loss:    {checkpoint['val_loss']:.6e}")
    print(f"Variable:    {args.variable}")
    print(f"Mode:        {args.mode}")
    if args.mode == "autoregressive":
        print(f"  L={args.L}, K={args.K}")
        total_steps = "unknown (determined from data)"
        print(f"  AR steps to cover trajectory: ~{total_steps}")
    print()

    # -- Data --
    # When normalize=True, we need train stats to denormalize predictions
    # Resolve {mode} patterns for file names
    input_pattern = args.input_file
    output_pattern = args.output_file

    output_mean, output_std = 0.0, 1.0
    if args.normalize:
        train_dataset = ReservoirDataset(
            data_path=args.data_path,
            mode="train",
            input_file=input_pattern,
            output_file=output_pattern,
            normalize=True,
        )
        norm_stats = train_dataset.get_normalization_stats()
        output_mean = norm_stats[2].item()
        output_std = norm_stats[3].item()
        print(
            f"Normalization: output_mean={output_mean:.4f}, output_std={output_std:.4f}"
        )
        del train_dataset

    test_dataset = ReservoirDataset(
        data_path=args.data_path,
        mode="test",
        input_file=input_pattern,
        output_file=output_pattern,
        normalize=args.normalize,
    )
    if args.normalize:
        # Re-load train stats for the test dataset normalization
        tmp_train = ReservoirDataset(
            data_path=args.data_path,
            mode="train",
            input_file=input_pattern,
            output_file=output_pattern,
            normalize=True,
        )
        test_dataset.set_normalization(*tmp_train.get_normalization_stats())
        del tmp_train
    sample_x, sample_y = test_dataset[0]
    num_timesteps = sample_y.shape[-1]

    spatial_mask = None
    if args.mask:
        mask_ds = ReservoirDataset(
            data_path=args.data_path,
            mode="test",
            input_file=args.input_file,
            output_file=args.output_file,
            normalize=False,
            use_mask=True,
        )
        spatial_mask = mask_ds.get_static_mask()
        if spatial_mask is not None:
            sm = spatial_mask.numpy()
            print(
                f"Mask:          {sm.sum()}/{sm.size} active ({100 * sm.mean():.1f}%)"
            )
        else:
            print("Mask:          no ACTNUM detected")
        del mask_ds

    print(f"Test samples:  {len(test_dataset)}")
    print(f"Input shape:   {tuple(sample_x.shape)}")
    print(f"Output shape:  {tuple(sample_y.shape)}")
    print(f"Timesteps:     {num_timesteps}")
    if args.mode == "autoregressive":
        actual_ar_steps = (num_timesteps - args.L) // args.K
        print(
            f"AR steps:      {actual_ar_steps} (from {num_timesteps} timesteps, L={args.L}, K={args.K})"
        )
    print()

    # -- Model --
    model = load_model(model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:    {num_params:,}")
    print()

    # -- Inference --
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    all_predictions = []
    all_targets = []

    if args.mode == "autoregressive":
        print(f"Running AUTOREGRESSIVE inference (L={args.L}, K={args.K})...")
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                y_batch_dev = y_batch.to(device)

                pred_batch = ar_validate_full_rollout(
                    model,
                    x_batch,
                    y_batch_dev,
                    L=args.L,
                    K=args.K,
                    is_tno=is_tno,
                    feedback_channel=feedback_channel,
                )

                all_predictions.append(pred_batch.cpu().numpy())
                all_targets.append(y_batch.numpy())

                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"  {(batch_idx + 1) * args.batch_size}/{len(test_dataset)} samples"
                    )
    else:
        print("Running FULL-MAPPING inference...")
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                pred_batch = model(x_batch).cpu().numpy()

                all_predictions.append(pred_batch)
                all_targets.append(y_batch.numpy())

                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"  {(batch_idx + 1) * args.batch_size}/{len(test_dataset)} samples"
                    )

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    print(f"  {len(test_dataset)}/{len(test_dataset)} samples - done.\n")

    # Denormalize to physical units for fair metric comparison
    if args.normalize:
        all_predictions = all_predictions * output_std + output_mean
        all_targets = all_targets * output_std + output_mean
        print(
            f"Denormalized to physical units (mean={output_mean:.4f}, std={output_std:.4f})"
        )
        print(
            f"  Pred range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]"
        )
        print(f"  GT range:   [{all_targets.min():.4f}, {all_targets.max():.4f}]")
        print()

    # -- Metrics --
    sm_np = spatial_mask.numpy() if spatial_mask is not None else None
    print_metrics(all_predictions, all_targets, args.variable, num_timesteps, sm_np)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
