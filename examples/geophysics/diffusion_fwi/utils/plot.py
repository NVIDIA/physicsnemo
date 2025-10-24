# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Dict, List

import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import wandb
from pathlib import Path


def plot_prediction(
    sample_idx: int,
    inputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    statistics: Dict[str, Dict[str, torch.Tensor]],
    metrics: Dict[str, Dict[str, float]],
    save_dir: Path,
    sources_to_plot: int = 3,
    samples_to_plot: List[int] = [0, 1, 2],
):
    """
    Plot predictions vs ground truth for any variables and inputs.
    Also plots ensemble samples and variances if available.

    Parameters
    ----------
    sample_idx : int, optional
        Sample index for wandb logging. Used for wandb logging.
    inputs : dict
        Dictionary containing input tensors (e.g., 'vx', 'vz')
    targets : dict
        Dictionary containing ground truth tensors
    predictions : dict
        Dictionary containing prediction tensors, and optionally ensemble data
    statistics : dict
        Dictionary containing mean and std of prediction tensors
    metrics : dict
        Dictionary containing RMSE and MAE values for variables
    save_dir : Path
        Directory to save the plots
    sources_to_plot : int
        Number of source channels to plot from each input
    samples_to_plot : list
        Indices of ensemble members to visualize
    """

    # Load input wavefields
    vx = inputs["vx"][0].cpu().numpy()  # shape: [nb_sources, H, W]
    vz = inputs["vz"][0].cpu().numpy()

    # Load targets
    vp_true = targets["vp"][0].cpu().numpy()
    vs_true = targets["vs"][0].cpu().numpy()
    rho_true = targets["rho"][0].cpu().numpy()

    # Load ensembles
    vp_ensemble = predictions["vp"].cpu().numpy()
    vs_ensemble = predictions["vs"].cpu().numpy()
    rho_ensemble = predictions["rho"].cpu().numpy()

    # Load ensembles mean
    vp_pred = statistics["mean"]["vp"][0].cpu().numpy()
    vs_pred = statistics["mean"]["vs"][0].cpu().numpy()
    rho_pred = statistics["mean"]["rho"][0].cpu().numpy()

    # Extract metrics
    vp_rmse, vp_mae = metrics["rmse"]["vp"], metrics["mae"]["vp"]
    vs_rmse, vs_mae = metrics["rmse"]["vs"], metrics["mae"]["vs"]
    rho_rmse, rho_mae = metrics["rmse"]["rho"], metrics["mae"]["rho"]

    ########################
    # 1. Plot vx, vz inputs
    ########################
    nb_sources = vx.shape[0]
    source_indices = (
        list(range(nb_sources))
        if sources_to_plot >= nb_sources
        else np.linspace(0, nb_sources - 1, sources_to_plot, dtype=int).tolist()
    )

    if len(source_indices) > 0:
        fig1 = plt.figure(figsize=(3 * len(source_indices), 6))
        gs = gridspec.GridSpec(
            2, len(source_indices), figure=fig1, wspace=0.05, hspace=0.1
        )
        H, W = vx[0].shape

        for j, src_idx in enumerate(source_indices):
            ax = fig1.add_subplot(gs[0, j])
            im = ax.imshow(
                vx[src_idx], cmap="viridis", extent=[0, W, 0, H], aspect="auto"
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"vx source {src_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig1.add_subplot(gs[1, j])
            im = ax.imshow(
                vz[src_idx], cmap="viridis", extent=[0, W, 0, H], aspect="auto"
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"vz source {src_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        inputs_path = Path(save_dir) / "inputs.png"
        fig1.savefig(inputs_path)

        wandb.log({f"sample_{sample_idx}/inputs": wandb.Image(str(inputs_path))})

        plt.close(fig1)

    ########################
    # 2. Plot Predictions
    ########################
    def plot_output_comparison(var_name, true, pred, ensemble, rmse, mae, idx_row):
        vmin = min(true.min(), pred.min())
        vmax = max(true.max(), pred.max())

        ax = fig2.add_subplot(gs[idx_row, 0])
        im = ax.imshow(true.squeeze(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Ground Truth {var_name}")
        ax.set_xticks([])
        ax.set_yticks([])

        for i, idx in enumerate(samples_to_plot):
            ax = fig2.add_subplot(gs[idx_row, 1 + i])
            ax.imshow(ensemble[idx].squeeze(), cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"Sample {i + 1} {var_name}")
            ax.set_xticks([])
            ax.set_yticks([])

        ax = fig2.add_subplot(gs[idx_row, 1 + len(samples_to_plot)])
        ax.imshow(pred.squeeze(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Mean {var_name}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar_ax = fig2.add_subplot(gs[idx_row, -1])
        plt.colorbar(im, cax=cbar_ax)

    num_samples = len(samples_to_plot)
    fig2 = plt.figure(figsize=(5 * (num_samples + 2), 12))
    width_ratios = [1] * (num_samples + 2) + [0.05]
    gs = fig2.add_gridspec(
        3, num_samples + 3, width_ratios=width_ratios, wspace=0.05, hspace=0.25
    )

    plot_output_comparison(
        "vp", vp_true, vp_pred, vp_ensemble, vp_rmse, vp_mae, idx_row=0
    )
    plot_output_comparison(
        "vs", vs_true, vs_pred, vs_ensemble, vs_rmse, vs_mae, idx_row=1
    )
    plot_output_comparison(
        "rho", rho_true, rho_pred, rho_ensemble, rho_rmse, rho_mae, idx_row=2
    )

    predictions_path = Path(save_dir) / "predictions.png"
    fig2.savefig(predictions_path)
    wandb.log({f"sample_{sample_idx}/predictions": wandb.Image(str(predictions_path))})

    plt.close(fig2)

    ########################
    # 3. Plot Ensemble Variance
    ########################
    vp_var = np.var(vp_ensemble, axis=0)
    vs_var = np.var(vs_ensemble, axis=0)
    rho_var = np.var(rho_ensemble, axis=0)

    fig3 = plt.figure(figsize=(15, 4))

    ax1 = fig3.add_subplot(1, 3, 1)
    im1 = ax1.imshow(vp_var.squeeze(), cmap="plasma")
    ax1.set_title("VP Ensemble Variance")
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1)

    ax2 = fig3.add_subplot(1, 3, 2)
    im2 = ax2.imshow(vs_var.squeeze(), cmap="plasma")
    ax2.set_title("VS Ensemble Variance")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2)

    ax3 = fig3.add_subplot(1, 3, 3)
    im3 = ax3.imshow(rho_var.squeeze(), cmap="plasma")
    ax3.set_title("RHO Ensemble Variance")
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3)

    fig3.tight_layout()
    ensemble_variance_path = Path(save_dir) / "ensemble_variance.png"
    fig3.savefig(ensemble_variance_path)
    wandb.log(
        {
            f"sample_{sample_idx}/ensemble_variance": wandb.Image(
                str(ensemble_variance_path)
            )
        }
    )

    plt.close(fig3)

    return
