#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Evaluation script for U-FNO pressure prediction model.
Visualizes predictions and computes error metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from ufno import UFNONet
from physicsnemo_unet import StandaloneUNet
from dataset import CO2SequestrationDataset
from metrics import (
    mean_relative_error,
    mean_absolute_error,
    compute_r2_score,
    compute_relative_l2_error,
)
from utils import (
    dnorm_dP,
    dnorm_inj,
    dnorm_temp,
    dnorm_P,
    dnorm_lam,
    dnorm_Swi,
    extract_reservoir_mask,
)
from data_validation import validate_sample_dimensions, print_validation_summary


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on CO2 sequestration dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (if not specified, will try to auto-detect)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="pressure",
        choices=["saturation", "pressure"],
        help="Variable to evaluate (default: pressure)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/wdyab/physicsnemo/data_lustre",
        help="Path to data directory",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    data_path = Path(args.data_path)
    variable = args.variable

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Try to auto-detect the most recent checkpoint for this variable
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob(f"best_model_{variable}_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for {variable}. Please specify --checkpoint"
            )
        # Use the most recently modified checkpoint
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load test dataset
    print(f"\nLoading test dataset for {variable}...")
    test_dataset = CO2SequestrationDataset(
        data_path=data_path,
        mode="test",
        variable=variable,
        normalize=False,  # Data is already normalized
        device="cpu",
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Input shape: {test_dataset[0][0].shape}")
    print(f"Output shape: {test_dataset[0][1].shape}")

    # Validate data dimensions (dynamic validation)
    print("\nValidating data dimensions...")
    sample_input, sample_target = test_dataset[0]

    # Validate using centralized validation function
    validate_sample_dimensions(sample_input, sample_target, variable)

    # Print validation summary
    print_validation_summary(
        input_shape=tuple(sample_input.shape),
        target_shape=tuple(sample_target.shape),
        variable=variable,
        is_batch=False,
        logger=None,  # Use print instead of logger
    )

    # Load checkpoint to get model configuration
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if model config is saved in checkpoint
    if "model_config" not in checkpoint:
        raise ValueError(
            "Model configuration not found in checkpoint. "
            "This checkpoint was created before model config saving was implemented. "
            "Please retrain the model or manually specify the model architecture."
        )

    model_config = checkpoint["model_config"]
    model_type = model_config["model_type"]
    model_arch_name = model_config.get("model_arch_name", "unknown")

    print(f"\nModel type: {model_type}")
    print(f"Model architecture: {model_arch_name}")
    print(f"Loaded from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")

    # Create model based on saved configuration
    print(f"\nCreating {model_arch_name} model from checkpoint config...")

    if model_type == "ufno":
        # Create U-FNO/Conv-FNO/FNO model
        model = UFNONet(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            width=model_config["width"],
            modes1=model_config["modes1"],
            modes2=model_config["modes2"],
            modes3=model_config["modes3"],
            num_fno_layers=model_config["num_fno_layers"],
            num_unet_layers=model_config["num_unet_layers"],
            num_conv_layers=model_config["num_conv_layers"],
            padding=model_config["padding"],
            conv_kernel_size=model_config["conv_kernel_size"],
            unet_kernel_size=model_config["unet_kernel_size"],
            unet_dropout=model_config["unet_dropout"],
            unet_type=model_config["unet_type"],
            activation_fn=model_config["activation_fn"],
            lifting_type=model_config["lifting_type"],
            lifting_layers=model_config["lifting_layers"],
            lifting_width=model_config["lifting_width"],
            decoder_type=model_config["decoder_type"],
            decoder_layers=model_config["decoder_layers"],
            decoder_width=model_config["decoder_width"],
        ).to(device)

    elif model_type == "unet":
        # Create standalone U-Net model
        model = StandaloneUNet(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            unet_type=model_config["unet_type"],
            **model_config["unet_kwargs"],
        ).to(device)

    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Model loaded successfully!")

    # Evaluate on entire test dataset
    print("\n" + "=" * 80)
    print(f"Evaluating on entire test dataset ({len(test_dataset)} samples)...")
    print("=" * 80)

    # Create dataloader for batch processing
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,  # Process 8 samples at a time
        shuffle=False,
    )

    # Lists to accumulate metrics across all samples
    all_mae = []
    all_mre = []
    all_r2 = []
    all_rel_l2 = []
    all_mae_per_timestep = [[] for _ in range(24)]
    all_mre_per_timestep = [[] for _ in range(24)]
    all_r2_per_timestep = [[] for _ in range(24)]
    all_rel_l2_per_timestep = [[] for _ in range(24)]

    print("Processing test samples...")

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            pred_batch = model(x_batch)

            # Move to CPU for metric computation
            x_plot_batch = x_batch.cpu().numpy()
            y_plot_batch = y_batch.cpu().numpy()
            pred_plot_batch = pred_batch.cpu().numpy()

            # Process each sample in the batch
            for i in range(x_plot_batch.shape[0]):
                x_plot = x_plot_batch[i : i + 1]
                y_plot = y_plot_batch[i]
                pred_plot = pred_plot_batch[i]

                # Denormalize pressure predictions
                pred_plot_denorm = dnorm_dP(pred_plot)

                # Extract mask
                mask, thickness = extract_reservoir_mask(x_plot)

                # Extract masked regions
                y_plot_masked = y_plot[mask].reshape((thickness, 200, 24))
                pred_plot_masked = pred_plot_denorm[mask].reshape((thickness, 200, 24))

                # Compute overall metrics for this sample
                mae = mean_absolute_error(pred_plot_masked, y_plot_masked)
                mre = mean_relative_error(pred_plot_masked, y_plot_masked)
                r2 = compute_r2_score(pred_plot_masked, y_plot_masked)
                rel_l2 = compute_relative_l2_error(pred_plot_masked, y_plot_masked)

                all_mae.append(mae)
                all_mre.append(mre)
                all_r2.append(r2)
                all_rel_l2.append(rel_l2)

                # Compute per-timestep metrics
                for t in range(24):
                    mae_t = mean_absolute_error(
                        pred_plot_masked[:, :, t], y_plot_masked[:, :, t]
                    )
                    mre_t = mean_relative_error(
                        pred_plot_masked[:, :, t], y_plot_masked[:, :, t]
                    )
                    r2_t = compute_r2_score(
                        pred_plot_masked[:, :, t], y_plot_masked[:, :, t]
                    )
                    rel_l2_t = compute_relative_l2_error(
                        pred_plot_masked[:, :, t], y_plot_masked[:, :, t]
                    )

                    all_mae_per_timestep[t].append(mae_t)
                    all_mre_per_timestep[t].append(mre_t)
                    all_r2_per_timestep[t].append(r2_t)
                    all_rel_l2_per_timestep[t].append(rel_l2_t)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Processed {(batch_idx + 1) * test_loader.batch_size}/{len(test_dataset)} samples..."
                )

    print(f"  Processed all {len(test_dataset)} samples.")

    # Compute average and std metrics across all samples
    avg_mae = np.mean(all_mae)
    std_mae = np.std(all_mae)
    avg_mre = np.mean(all_mre)
    std_mre = np.std(all_mre)
    avg_r2 = np.mean(all_r2)
    std_r2 = np.std(all_r2)
    avg_rel_l2 = np.mean(all_rel_l2)
    std_rel_l2 = np.std(all_rel_l2)

    # Compute error metrics
    print("\n" + "=" * 80)
    print(f"Error Metrics (averaged over {len(test_dataset)} test samples):")
    print("=" * 80)

    print(f"Mean Absolute Error (MAE):   {avg_mae:.4f} ± {std_mae:.4f} bar")
    print(f"Mean Relative Error (MRE):   {avg_mre:.4f} ± {std_mre:.4f}")
    print(f"R2 Score:                    {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"Relative L2 Error:           {avg_rel_l2:.6f} ± {std_rel_l2:.6f}")

    # Compute per-timestep averages
    print("\nPer-timestep Metrics (averaged over all samples):")
    print("  t  |   MAE (bar)  |    MRE     |  R2 Score  | Rel L2 Error")
    print("-" * 70)
    for t in range(24):
        avg_mae_t = np.mean(all_mae_per_timestep[t])
        avg_mre_t = np.mean(all_mre_per_timestep[t])
        avg_r2_t = np.mean(all_r2_per_timestep[t])
        avg_rel_l2_t = np.mean(all_rel_l2_per_timestep[t])
        print(
            f"  {t:2d} |   {avg_mae_t:8.4f}   | {avg_mre_t:10.6f} | {avg_r2_t:10.6f} | {avg_rel_l2_t:10.6f}"
        )

    # Now visualize one sample (sample 0) for illustration
    print("\n" + "=" * 80)
    print("Creating visualization for sample 0...")
    print("=" * 80)

    x, y = test_dataset[0]
    x = x.unsqueeze(0).to(device)
    y = y.to(device)

    with torch.no_grad():
        pred = model(x)

    # Move to numpy for visualization
    x_plot = x.cpu().numpy()
    y_plot = y.cpu().numpy()
    pred_plot = pred.squeeze(0).cpu().numpy()

    # Denormalize pressure predictions
    pred_plot_denorm = dnorm_dP(pred_plot)

    # Extract mask (reservoir have different thickness as marked in the permeability map)
    mask, thickness = extract_reservoir_mask(x_plot)

    print(f"Reservoir thickness: {thickness} cells")

    # Extract input parameters
    poro_map = x_plot[0, :, :, 0, 2][mask].reshape((thickness, -1))
    kr_map = np.exp(x_plot[0, :, :, 0, 0][mask].reshape((thickness, -1)) * 15)
    kz_map = np.exp(x_plot[0, :, :, 0, 1][mask].reshape((thickness, -1)) * 15)
    inj_rate = dnorm_inj(x_plot[0, 0, 0, 0, 4])
    temperature = dnorm_temp(x_plot[0, 0, 0, 0, 6])
    pressure = dnorm_P(x_plot[0, 0, 0, 0, 5])
    Swi = dnorm_Swi(x_plot[0, 0, 0, 0, 7])
    lam = dnorm_lam(x_plot[0, 0, 0, 0, 8])

    print(f"\nInput parameters:")
    print(f"  Injection rate: {inj_rate:.2f} MT/yr")
    print(f"  Temperature: {temperature:.1f} °C")
    print(f"  Initial pressure: {pressure:.1f} bar")
    print(f"  Swi: {Swi:.2f}")
    print(f"  Lambda: {lam:.2f}")

    # Visualization
    print("\n" + "=" * 80)
    print("Creating visualization...")
    print("=" * 80)

    # Setup grid for plotting
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))

    # Time labels
    times = np.cumsum(np.power(1.421245, range(24)))
    time_print = []
    for t in range(times.shape[0]):
        if times[t] < 365:
            title = str(int(times[t])) + " d"
        else:
            title = f"{round(int(times[t]) / 365, 1)} y"
        time_print.append(title)

    def pcolor(x):
        plt.jet()
        return plt.pcolor(
            X[:thickness, :], Y[:thickness, :], np.flipud(x), shading="auto"
        )

    # Select timesteps to plot
    t_lst = [14, 20, 23]

    plt.figure(figsize=(15, 16))

    for j, t in enumerate(t_lst):
        # Row 1: Input parameters
        plt.subplot(4, 3, j + 1)
        if j == 2:
            pcolor(poro_map)
            plt.title("$\phi$ (-)")
        elif j == 1:
            pcolor(kz_map)
            plt.title("$k_z$ (mD)")
        else:
            pcolor(kr_map)
            plt.title("$k_r$ (mD)")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        # Row 2: Ground truth
        plt.subplot(4, 3, j + 4)
        pcolor(y_plot[:, :, t][mask].reshape((thickness, -1)))
        plt.title("$dP$ (bar), " + f"t={time_print[t]}")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        # Row 3: Prediction
        plt.subplot(4, 3, j + 7)
        pcolor(pred_plot_denorm[:, :, t][mask].reshape((thickness, -1)))
        plt.title("$\hat{dP}$ (bar), " + f"t={time_print[t]}")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        # Row 4: Error
        plt.subplot(4, 3, j + 10)
        error = pred_plot_denorm[:, :, t][mask].reshape((thickness, -1)) - y_plot[
            :, :, t
        ][mask].reshape((thickness, -1))
        pcolor(error)
        plt.colorbar(fraction=0.02)
        plt.title("|$dP-\hat{dP}$|, " + f"t={time_print[t]}")
        plt.xlim([0, 3500])

    plt.tight_layout()

    # Save figure
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"pressure_prediction_sample0.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_file}")

    plt.show()

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
