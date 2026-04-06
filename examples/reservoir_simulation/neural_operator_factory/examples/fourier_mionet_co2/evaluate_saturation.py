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
CO2 gas saturation evaluation — matches the validated U-FNO evaluation algorithm.

No denormalization. MPE computed within the plume region.
R² is computed globally across all samples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
from data.dataloader import ReservoirDataset
from data.validation import print_validation_summary, validate_sample_dimensions
from utils.checkpoint import build_model_from_config


def mean_plume_error(y_pred, y_true):
    """Compute mean plume error for saturation predictions."""
    mask = (y_pred != 0) & (y_true != 0)
    y_pred_masked = y_pred[mask]
    y_true_masked = y_true[mask]
    if len(y_pred_masked) == 0:
        return 0.0
    return np.mean(np.abs(y_pred_masked - y_true_masked))


def mean_absolute_error(y_pred, y_true):
    """Compute mean absolute error between predictions and targets."""
    return np.mean(np.abs(y_pred - y_true))


def main():
    """Run evaluation on the test set and print metrics."""
    parser = argparse.ArgumentParser(
        description="Evaluate CO2 saturation model (U-FNO example)"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/co2",
    )
    parser.add_argument("--batch_size", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    variable = "saturation"
    data_path = Path(args.data_path)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob(f"best_model_{variable}_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for {variable}. Specify --checkpoint"
            )
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected checkpoint: {checkpoint_path}")

    print(f"\nLoading test dataset for {variable}...")
    test_dataset = ReservoirDataset(
        data_path=data_path, mode="test", variable=variable, normalize=False
    )
    print(f"Test dataset size: {len(test_dataset)}")

    sample_input, sample_target = test_dataset[0]
    validate_sample_dimensions(sample_input, sample_target, variable)
    print_validation_summary(
        tuple(sample_input.shape),
        tuple(sample_target.shape),
        variable,
        is_batch=False,
    )

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    print(f"Model: {model_config.get('model_arch_name', 'unknown')}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint['val_loss']:.6f}")

    model, model_arch_name = build_model_from_config(model_config, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded: {model_arch_name}")

    print("\n" + "=" * 80)
    print(f"Evaluating on {len(test_dataset)} test samples...")
    print("=" * 80)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    mpe_list = []
    mae_list = []
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        _has_b2 = model_config.get("variant", "") in ("mionet", "fourier_mionet", "tno")

        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            fwd = {"x_branch2": x_batch[:, 0, 0, 0, :]} if _has_b2 else {}
            pred_batch = model(x_batch, **fwd)

            x_np = x_batch.cpu().numpy()
            y_np = y_batch.cpu().numpy()
            pred_np = pred_batch.cpu().numpy()

            for rr in range(x_np.shape[0]):
                mask = x_np[rr, :, :, 0, 0] != 0
                thickness = int(np.sum(mask[:, 0]))

                y_plot = y_np[rr][mask].reshape((thickness, 200, 24, -1))
                pred_plot = pred_np[rr][mask].reshape((thickness, 200, 24, -1))

                mpe = mean_plume_error(pred_plot, y_plot)
                mae = mean_absolute_error(pred_plot, y_plot)

                mpe_list.append(np.mean(mpe))
                mae_list.append(mae)
                y_true_all.append(y_plot)
                y_pred_all.append(pred_plot)

            if (batch_idx + 1) % 10 == 0:
                n_done = (batch_idx + 1) * args.batch_size
                print(f"  {n_done}/{len(test_dataset)} samples...")

    print(f"  {len(test_dataset)}/{len(test_dataset)} samples - done.")

    overall_mpe = np.mean(mpe_list)
    overall_mae = np.mean(mae_list)

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    ss_res = np.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_true_all - y_true_all.mean()) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print("\n" + "=" * 80)
    print(f"Gas Saturation — Test Set ({len(test_dataset)} samples)")
    print("=" * 80)
    print(f"MPE:  {overall_mpe:.4f}")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"R2:   {r2_score:.4f}")
    print("=" * 80)

    # Visualization (sample 0)
    print("\nCreating visualization for sample 0...")
    x, y = test_dataset[0]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        vfwd = {"x_branch2": x[:, 0, 0, 0, :]} if _has_b2 else {}
        pred = model(x, **vfwd)

    x_plot = x.cpu().numpy()
    y_plot = y.numpy()
    pred_plot = pred.squeeze(0).cpu().numpy()
    mask = x_plot[0, :, :, 0, 0] != 0
    thickness = int(np.sum(mask[:, 0]))

    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))
    times = np.cumsum(np.power(1.421245, range(24)))
    time_labels = [
        f"{int(t)} d" if t < 365 else f"{round(int(t) / 365, 1)} y" for t in times
    ]

    def pcolor(data):
        """Plot a 2D pseudocolor map on the reservoir grid."""
        plt.jet()
        return plt.pcolor(
            X[:thickness, :],
            Y[:thickness, :],
            np.flipud(data),
            shading="auto",
        )

    t_lst = [14, 20, 23]
    plt.figure(figsize=(15, 12))
    for j, t in enumerate(t_lst):
        plt.subplot(3, 3, j + 1)
        pcolor(y_plot[:, :, t][mask].reshape((thickness, -1)))
        plt.title(f"True Sg, t={time_labels[t]}")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        plt.subplot(3, 3, j + 4)
        pcolor(pred_plot[:, :, t][mask].reshape((thickness, -1)))
        plt.title(f"Pred Sg, t={time_labels[t]}")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        plt.subplot(3, 3, j + 7)
        error = pred_plot[:, :, t][mask].reshape((thickness, -1)) - y_plot[:, :, t][
            mask
        ].reshape((thickness, -1))
        pcolor(error)
        plt.colorbar(fraction=0.02)
        plt.title(f"Error, t={time_labels[t]}")
        plt.xlim([0, 3500])

    plt.tight_layout()
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"saturation_{model_arch_name}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
