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

r"""
Rollout prediction and evaluation module.

Compatible with neuralop 2.0.0 API.
"""

import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils.plotting import (
    create_rollout_animation,
    generate_publication_maps,
    generate_max_value_maps,
    generate_combined_analysis_maps,
    plot_volume_conservation,
    plot_conditional_error_analysis,
    plot_aggregated_scalar_metrics,
    plot_event_magnitude_analysis,
)


def compute_csi(threshold, pred, gt):
    r"""
    Compute Critical Success Index (CSI) for binary classification.

    Parameters
    ----------
    threshold : float
        Threshold value for binary classification.
    pred : np.ndarray
        Predicted values.
    gt : np.ndarray
        Ground truth values.

    Returns
    -------
    float
        CSI score.
    """
    event_pred, event_gt = pred >= threshold, gt >= threshold
    TP = np.sum(event_pred & event_gt)
    FP = np.sum(event_pred & (~event_gt))
    FN = np.sum((~event_pred) & event_gt)
    return TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1.0


def rollout_prediction(
    model,
    rollout_dataset,
    rollout_length,
    history_steps,
    dynamic_norm,
    target_norm,
    boundary_norm,
    device,
    skip_before_timestep,
    dt,
    out_dir="./rollout_gifs",
    logger=None,
):
    r"""
    Performs autoregressive rollout, computing and plotting metrics for both water depth and velocity.

    Compatible with neuralop 2.0.0 API.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    rollout_dataset : Dataset
        Dataset for rollout evaluation.
    rollout_length : int
        Length of rollout to perform.
    history_steps : int
        Number of history timesteps.
    dynamic_norm : UnitGaussianNormalizer
        Normalizer for dynamic features.
    target_norm : UnitGaussianNormalizer
        Normalizer for target.
    boundary_norm : UnitGaussianNormalizer
        Normalizer for boundary conditions.
    device : str or torch.device
        Device to run inference on.
    skip_before_timestep : int
        Number of timesteps to skip at beginning.
    dt : float
        Time step size in seconds.
    out_dir : str, optional, default="./rollout_gifs"
        Output directory for results.
    logger : Any, optional
        Optional logger instance.
    """
    if logger is None:
        # Fallback to print if no logger provided
        def log_info(msg):
            print(msg)

        logger = type("Logger", (), {"info": lambda self, msg: log_info(msg)})()

    logger.info(f"Starting rollout prediction on {len(rollout_dataset)} samples")
    logger.info(f"Rollout length: {rollout_length}, History steps: {history_steps}")
    logger.info(f"Output directory: {out_dir}")
    
    model = model.to(device)
    model.eval()
    dynamic_norm = dynamic_norm.to(device)
    target_norm = target_norm.to(device)
    boundary_norm = boundary_norm.to(device)

    # Initialize lists for aggregated metrics
    aggregated_metrics = {
        'rmse_wd': [], 'csi_005': [], 'csi_03': [], 'rmse_vx': [], 'rmse_vy': [],
        'h_V2_rmse': [], 'fhca': [],
        'arrival_mae': [], 'duration_mae': [],
    }

    # Initialize lists for event magnitude analysis
    event_q_peaks = []
    event_total_volumes = []
    event_avg_rmse_wd = []

    # Initialize list to store inference times
    rollout_inference_times = []

    for idx, sample in enumerate(tqdm(rollout_dataset, desc="Performing rollout evaluation")):
        run_id = sample.get("run_id", f"sample_{idx}")
        full_dynamic = sample["dynamic"].to(device)
        full_boundary = sample["boundary"].to(device)
        geometry = sample["geometry"]
        
        cell_area = sample.get("cell_area", None)
        if cell_area is not None:
            cell_area = cell_area.cpu().numpy()

        # Calculate hydrograph characteristics for the current event
        unnormalized_boundary = boundary_norm.inverse_transform(full_boundary).squeeze(0)
        inflow_hydrograph = unnormalized_boundary[:, 0, 0].cpu().numpy()
        q_peak = np.max(inflow_hydrograph)
        total_volume = np.sum(inflow_hydrograph) * dt
        event_q_peaks.append(q_peak)
        event_total_volumes.append(total_volume)

        start_pred_t = skip_before_timestep + history_steps
        end_pred_t = start_pred_t + rollout_length
        gt_rollout = full_dynamic[start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

        wd_pred_list, wd_gt_list = [], []
        vx_pred_list, vy_pred_list = [], []
        vx_gt_list, vy_gt_list = [], []
        run_ts_metrics = {'rmse_wd': [], 'csi_005': [], 'csi_03': [], 'rmse_vx': [], 'rmse_vy': []}

        # Record start time for the rollout
        start_time = time.time()

        current_dynamic = full_dynamic[skip_before_timestep:start_pred_t].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()

        for t in range(rollout_length):
            # Prepare input tensors
            dyn_flat = current_dynamic.permute(1, 0, 2).reshape(1, current_dynamic.shape[1], -1)
            bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
            x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)

            with torch.no_grad():
                # Call model with GINO signature
                pred = model(
                    input_geom=geometry.to(device).unsqueeze(0),
                    latent_queries=sample["query_points"].to(device).unsqueeze(0),
                    output_queries=geometry.to(device).unsqueeze(0),
                    x=x
                )

            # Inverse transform predictions and ground truth
            inv_pred = target_norm.inverse_transform(pred)
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))
            
            # Extract water depth and velocity components
            wd_pred, vx_pred, vy_pred = [ch.cpu().numpy() for ch in inv_pred[0].T]
            wd_gt, vx_gt, vy_gt = [ch.cpu().numpy() for ch in inv_gt[0].T]

            wd_pred_list.append(wd_pred)
            wd_gt_list.append(wd_gt)
            vx_pred_list.append(vx_pred)
            vx_gt_list.append(vx_gt)
            vy_pred_list.append(vy_pred)
            vy_gt_list.append(vy_gt)

            # Time-step metrics
            run_ts_metrics['rmse_wd'].append(np.sqrt(np.mean((wd_pred - wd_gt) ** 2)))
            run_ts_metrics['csi_005'].append(compute_csi(0.05, wd_pred, wd_gt))
            run_ts_metrics['csi_03'].append(compute_csi(0.3, wd_pred, wd_gt))
            run_ts_metrics['rmse_vx'].append(np.sqrt(np.mean((vx_pred - vx_gt) ** 2)))
            run_ts_metrics['rmse_vy'].append(np.sqrt(np.mean((vy_pred - vy_gt) ** 2)))

            # Update current dynamic state with prediction
            current_dynamic = torch.cat([current_dynamic[1:], pred.squeeze(0).unsqueeze(0)], dim=0)
            current_boundary = torch.cat([current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0)

        # Record end time and append the duration
        end_time = time.time()
        rollout_inference_times.append(end_time - start_time)

        # Convert lists to numpy arrays for this run
        wd_pred_arr, wd_gt_arr = np.stack(wd_pred_list), np.stack(wd_gt_list)
        vx_pred_arr, vy_pred_arr = np.stack(vx_pred_list), np.stack(vy_pred_list)
        vx_gt_arr, vy_gt_arr = np.stack(vx_gt_list), np.stack(vy_gt_list)

        # Store the overall error for this event
        avg_rmse_for_run = np.mean(run_ts_metrics['rmse_wd'])
        event_avg_rmse_wd.append(avg_rmse_for_run)

        # Append run-averaged metrics to aggregated lists
        for key in ['rmse_wd', 'csi_005', 'csi_03', 'rmse_vx', 'rmse_vy']:
            aggregated_metrics[key].append(np.array(run_ts_metrics[key]))

        figures_path = os.path.join(out_dir, "figures_final")
        os.makedirs(figures_path, exist_ok=True)

        # Generate Plots and Scalar Metrics for this Run
        logger.info(f"Generating plots for run {run_id}...")
        generate_publication_maps(
            geometry,
            wd_gt_arr,
            wd_pred_arr,
            vx_gt_arr,
            vy_gt_arr,
            vx_pred_arr,
            vy_pred_arr,
            [12, 24, 36, 48, 60, 72],
            figures_path,
            run_id,
        )
        logger.info(f"Saved publication maps for run {run_id}")
        generate_max_value_maps(
            geometry,
            wd_gt_arr,
            wd_pred_arr,
            vx_gt_arr,
            vy_gt_arr,
            vx_pred_arr,
            vy_pred_arr,
            figures_path,
            run_id,
        )
        logger.info(f"Saved max value maps for run {run_id}")

        mae_arrival, mae_duration, rmse_hv2, fhca = generate_combined_analysis_maps(
            geometry,
            wd_gt_arr,
            wd_pred_arr,
            vx_gt_arr,
            vy_gt_arr,
            vx_pred_arr,
            vy_pred_arr,
            dt,
            figures_path,
            run_id,
        )
        logger.info(f"Saved combined analysis plot for run {run_id}")
        aggregated_metrics["arrival_mae"].append(mae_arrival)
        aggregated_metrics["duration_mae"].append(mae_duration)
        aggregated_metrics["h_V2_rmse"].append(rmse_hv2)
        aggregated_metrics["fhca"].append(fhca)

        # Generate Volume Conservation Plot
        plot_volume_conservation(wd_gt_arr, wd_pred_arr, cell_area, dt, figures_path, run_id)
        logger.info(f"Saved volume conservation plot for run {run_id}")

        # Generate Conditional Error Plot
        plot_conditional_error_analysis(
            wd_gt_arr,
            wd_pred_arr,
            vx_gt_arr,
            vy_gt_arr,
            vx_pred_arr,
            vy_pred_arr,
            figures_path,
            run_id,
        )
        logger.info(f"Saved conditional error plot for run {run_id}")

        # Generate Rollout Animation (GIF)
        create_rollout_animation(
            geometry,
            wd_gt_arr,
            wd_pred_arr,
            vx_gt_arr,
            vy_gt_arr,
            vx_pred_arr,
            vy_pred_arr,
            run_id=run_id,
            out_dir=out_dir,
            filename_prefix="rollout",
            dt_seconds=dt,
        )
        logger.info(f"Saved rollout animation for run {run_id}")

    if aggregated_metrics['rmse_wd']:
        # Process and plot aggregated metrics across all runs
        ts_metrics = {
            k: np.stack(v) for k, v in aggregated_metrics.items()
            if k in ['rmse_wd', 'csi_005', 'csi_03', 'rmse_vx', 'rmse_vy']
        }
        ts_stats = {key: {'mean': arr.mean(axis=0), 'std': arr.std(axis=0)} for key, arr in ts_metrics.items()}
        time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0

        fig, axs = plt.subplots(3, 2, figsize=(16, 18), tight_layout=True)
        axs = axs.flatten()
        fig.suptitle("Time-Series Metrics During Rollout", fontsize=18)
        plot_info = {
            0: ('rmse_wd', 'RMSE (Depth)', 'RMSE (m)'),
            1: ('rmse_vx', r'RMSE ($V_{x}$)', 'RMSE (m/s)'),
            2: ('rmse_vy', r'RMSE ($V_{y}$)', 'RMSE (m/s)'),
            3: ('csi_005', 'CSI (0.05m)', 'CSI'),
            4: ('csi_03', 'CSI (0.3m)', 'CSI')
        }
        for i, ax in enumerate(axs):
            if i in plot_info:
                key, title, ylabel = plot_info[i]
                mean, std = ts_stats[key]['mean'], ts_stats[key]['std']
                ax.plot(time_hours, mean, label=f'{title} Mean', marker='o', markersize=4)
                ax.fill_between(time_hours, mean - std, mean + std, alpha=0.3, label='+/-1 Std Dev')
                ax.set_title(title)
                ax.set_xlabel("Time (hour)")
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True, linestyle='--')
            else:
                ax.set_visible(False)
        plt.savefig(os.path.join(out_dir, "rollout_metrics_summary.png"))
        plt.close(fig)
        logger.info(f"Saved aggregated rollout metrics plot to: {os.path.join(out_dir, 'rollout_metrics_summary.png')}")

        # Scalar metrics plotting
        scalar_metrics_for_plot = {
            'h_V2_rmse': aggregated_metrics['h_V2_rmse'],
            'fhca': aggregated_metrics['fhca'],
            'arrival_mae_hrs': np.array(aggregated_metrics['arrival_mae']) / 3600.0,
            'duration_mae_hrs': np.array(aggregated_metrics['duration_mae']) / 3600.0,
        }
        plot_aggregated_scalar_metrics(scalar_metrics_for_plot, out_dir)
        logger.info("Saved aggregated scalar metrics plots")

        # Call the event magnitude analysis plotting function
        plot_event_magnitude_analysis(
            q_peaks=event_q_peaks,
            total_volumes=event_total_volumes,
            avg_rmses_wd=event_avg_rmse_wd,
            out_dir=out_dir,
        )
        logger.info("Saved event magnitude analysis plots")

        # Calculate and log timing statistics
        if rollout_inference_times:
            mean_inference_time = np.mean(rollout_inference_times)
            std_inference_time = np.std(rollout_inference_times)
            logger.info("=" * 60)
            logger.info("Inference Time Summary")
            logger.info("=" * 60)
            logger.info(f"Time per full rollout (averaged over {len(rollout_inference_times)} hydrographs):")
            logger.info(f"  Mean: {mean_inference_time:.4f} seconds")
            logger.info(f"  Std Dev: {std_inference_time:.4f} seconds")

        # Final summary and data saving
        logger.info("")
        logger.info("=" * 60)
        logger.info("Aggregated Rollout Metrics Summary")
        logger.info("=" * 60)
        scalar_stats = {
            key: {'mean': np.nanmean(v), 'std': np.nanstd(v)}
            for key, v in scalar_metrics_for_plot.items()
        }
        logger.info("Scalar Hydrological Metrics (averaged over all runs):")
        for key, stat in scalar_stats.items():
            logger.info(f"  {key:<20}: Mean={stat['mean']:.4f}, Std={stat['std']:.4f}")

        npz_data = {'time_hours': time_hours}
        for key, stat_dict in ts_stats.items():
            npz_data[f'{key}_mean'] = stat_dict['mean']
            npz_data[f'{key}_std'] = stat_dict['std']
        for key, data in scalar_metrics_for_plot.items():
            npz_data[f'{key}_all_runs'] = np.array(data)

        # Add timings to saved data
        if rollout_inference_times:
            npz_data['rollout_inference_times'] = np.array(rollout_inference_times)

        metrics_file = os.path.join(out_dir, "rollout_metrics_data.npz")
        np.savez(metrics_file, **npz_data)
        logger.info(f"Saved all aggregated rollout metrics data to: {metrics_file}")
