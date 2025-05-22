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


"""
rollout_script.py

A standalone script that uses Hydra to load the shared configuration,
instantiates the test dataset and the trained MeshGraphKAN model, loads the checkpoint,
and performs an iterative rollout for each test hydrograph sample.
For each sample, a fancy four-panel animation is generated that shows:
  1. Prediction (node colors represent predicted actual water depth)
  2. Ground Truth (node colors represent actual water depth)
  3. Absolute Error (difference between prediction and ground truth)
  4. RMSE curve over time (updated with each rollout step)

The model checkpoint is loaded using the provided load_checkpoint utility.
"""

import os
import torch
import hydra
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Import the load_checkpoint utility from Modulus Launch.
from physicsnemo.launch.utils import load_checkpoint

# Import the dataset and model.
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN

# For converting DGLGraph to networkx.
from dgl import to_networkx


def create_animation(
    rollout_predictions,
    ground_truth,
    initial_graph,
    rmse_list,
    output_path,
    time_per_step=20 / 60,
):
    """
    Create a four-panel animation for one hydrograph rollout.

    Parameters:
      rollout_predictions: list of predicted actual water depth tensors (each shape: [num_nodes])
      ground_truth: list of ground truth water depth tensors (each shape: [num_nodes])
      initial_graph: the initial DGL graph sample (used for node positions and edges)
      rmse_list: list of RMSE values computed at each rollout step
      output_path: file path to save the animation (e.g. a GIF file)
      time_per_step: simulation time (in hours) corresponding to each rollout step.
    """
    # Set professional style.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    # Create figure and extra axes for colorbars.
    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    cax1 = fig.add_axes([0.05, 0.53, 0.02, 0.35])
    cax2 = fig.add_axes([0.95, 0.53, 0.02, 0.35])
    cax3 = fig.add_axes([0.05, 0.1, 0.02, 0.35])

    num_frames = len(rollout_predictions)
    # Use the first two columns of node features for positions.
    init_node_feats = initial_graph.ndata["x"]
    pos = {
        i: (init_node_feats[i, 0].item(), init_node_feats[i, 1].item())
        for i in range(init_node_feats.shape[0])
    }

    # Compute global color scaling based on both predictions and ground truth.
    all_vals = torch.cat(rollout_predictions + ground_truth)
    vmin_global = all_vals.min().item()
    vmax_global = all_vals.max().item()

    def update(frame):
        for ax in axes.flat:
            ax.clear()
        current_time = (frame + 1) * time_per_step

        # Panel 1: Prediction.
        pred_vals = rollout_predictions[frame].cpu().numpy()
        # Ensure the graph is on CPU before converting.
        g_pred = to_networkx(initial_graph.cpu())
        g_pred = g_pred.to_undirected()
        nodes_pred = nx.draw_networkx_nodes(
            g_pred,
            pos,
            node_color=pred_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 0],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_pred, pos, alpha=0.5, ax=axes[0, 0])
        axes[0, 0].set_title(f"Time {current_time:.2f} Hours - Prediction", fontsize=24)
        fig.colorbar(nodes_pred, cax=cax1)

        # Panel 2: Ground Truth.
        gt_vals = ground_truth[frame].cpu().numpy()
        g_gt = to_networkx(initial_graph.cpu())
        g_gt = g_gt.to_undirected()
        nodes_gt = nx.draw_networkx_nodes(
            g_gt,
            pos,
            node_color=gt_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 1],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_gt, pos, alpha=0.5, ax=axes[0, 1])
        axes[0, 1].set_title(
            f"Time {current_time:.2f} Hours - Ground Truth", fontsize=24
        )
        fig.colorbar(nodes_gt, cax=cax2)

        # Panel 3: Absolute Error.
        abs_error = torch.abs(rollout_predictions[frame] - ground_truth[frame])
        abs_vals = abs_error.cpu().numpy()
        g_error = to_networkx(initial_graph.cpu())
        g_error = g_error.to_undirected()
        nodes_error = nx.draw_networkx_nodes(
            g_error,
            pos,
            node_color=abs_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[1, 0],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(g_error, pos, alpha=0.5, ax=axes[1, 0])
        axes[1, 0].set_title(
            f"Time {current_time:.2f} Hours - Absolute Error", fontsize=24
        )
        fig.colorbar(nodes_error, cax=cax3)

        # Panel 4: RMSE Curve.
        times = [(i + 1) * time_per_step for i in range(frame + 1)]
        axes[1, 1].plot(
            times,
            rmse_list[: frame + 1],
            label="Water Depth RMSE",
            color="b",
            linewidth=3,
        )
        axes[1, 1].set_title("RMSE Over Time", fontsize=24)
        axes[1, 1].set_xlabel("Time (Hours)", fontsize=24)
        axes[1, 1].set_ylabel("RMSE", fontsize=24)
        axes[1, 1].legend(fontsize=20)
        axes[1, 1].grid(True)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(output_path, writer="pillow", fps=2)
    plt.close(fig)
    print(f"Animation saved to {output_path}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function that loads the configuration, instantiates the test dataset and model,
    loads the checkpoint using load_checkpoint, performs iterative rollout, and generates animations.
    """
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    rollout_length = cfg.get(
        "num_test_time_steps", 10
    )  # Rollout length (number of future steps)
    n_time_steps = cfg.get("n_time_steps", 2)
    prefix = cfg.get("prefix", "M80")
    data_dir = cfg.get("test_dir")
    test_ids_file = cfg.get("test_ids_file", "test.txt")
    ckpt_path = cfg.get("ckpt_path")
    anim_output_dir = cfg.get("animation_output_dir", "animations")
    os.makedirs(anim_output_dir, exist_ok=True)

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Instantiate the test dataset.
    test_dataset = HydroGraphDataset(
        data_dir=data_dir,
        prefix=prefix,
        n_time_steps=n_time_steps,
        hydrograph_ids_file=test_ids_file,
        split="test",
        rollout_length=rollout_length,
        force_reload=False,
        verbose=True,
        return_physics=False,
    )
    print(f"Loaded test dataset with {len(test_dataset)} hydrographs.")

    # Instantiate the model.
    num_input_features = cfg.get("num_input_features", 16)
    num_edge_features = cfg.get("num_edge_features", 3)
    num_output_features = cfg.get("num_output_features", 2)
    model = MeshGraphKAN(num_input_features, num_edge_features, num_output_features)
    model.to(device)

    # Load model checkpoint using the provided load_checkpoint utility.
    epoch_loaded = load_checkpoint(
        to_absolute_path(ckpt_path),
        models=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device=device,
    )
    print(f"Checkpoint loaded from epoch {epoch_loaded}")
    model.eval()

    all_rmse_all = []

    # Loop over each test hydrograph.
    for idx in range(len(test_dataset)):
        g, rollout_data = test_dataset[idx]
        g = g.to(device)
        edge_features = g.edata["x"].to(device)
        X_current = g.ndata["x"].to(device)  # Expected shape: [num_nodes, 16]
        num_nodes = X_current.size(0)

        rollout_preds = (
            []
        )  # To store predicted actual water depth values for each step.
        ground_truth_list = []  # To store ground truth water depth values.
        rmse_list = []  # RMSE at each rollout step.

        # Rollout data tensors.
        # Note: inflow_seq is a 1D tensor of length rollout_length.
        inflow_seq = rollout_data["inflow"].to(device)
        precip_seq = rollout_data["precipitation"].to(device)
        wd_gt_seq = rollout_data["water_depth_gt"].to(device)

        X_iter = X_current.clone()

        for t in range(rollout_length):
            # Split into static and dynamic parts.
            static_part = X_iter[
                :, :12
            ]  # columns 0-11: static features (including flow/precip)
            water_depth_window = X_iter[
                :, 12 : 12 + n_time_steps
            ]  # e.g., columns 12-13 for n_time_steps=2
            volume_window = X_iter[
                :, 12 + n_time_steps : 12 + 2 * n_time_steps
            ]  # e.g., columns 14-15

            # Use the full dynamic window as input.
            X_input = torch.cat(
                [static_part, water_depth_window, volume_window], dim=1
            )  # shape remains 16

            # Predict the differences (delta).
            pred = model(X_input, edge_features, g)  # shape: (num_nodes, 2)
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]

            # Update dynamic window: drop the oldest time step and append the new prediction.
            water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)

            # Update static part: since inflow_seq and precip_seq are 1D,
            # we unsqueeze and expand them to shape (num_nodes, 1).
            new_flow = inflow_seq[t].unsqueeze(0).expand(num_nodes, 1)
            new_precip = precip_seq[t].unsqueeze(0).expand(num_nodes, 1)
            static_part_updated = static_part.clone()
            static_part_updated[:, 10:12] = torch.cat([new_flow, new_precip], dim=1)

            # Form updated X_iter.
            X_iter = torch.cat(
                [static_part_updated, water_depth_updated, volume_updated], dim=1
            )

            # Save the predicted actual water depth.
            rollout_preds.append(new_wd.squeeze(1).detach().cpu())
            ground_truth_list.append(wd_gt_seq[t].detach().cpu())

            # Compute RMSE for this rollout step.
            rmse = torch.sqrt(
                torch.mean((new_wd.squeeze(1) - wd_gt_seq[t]) ** 2)
            ).item()
            rmse_list.append(rmse)

        all_rmse_all.append(rmse_list)
        mean_rmse_sample = sum(rmse_list) / len(rmse_list)
        sample_id = test_dataset.dynamic_data[idx].get("hydro_id", idx)
        print(f"Hydrograph {sample_id}: Mean RMSE = {mean_rmse_sample:.4f}")

        anim_filename = os.path.join(anim_output_dir, f"animation_{sample_id}.gif")
        create_animation(rollout_preds, ground_truth_list, g, rmse_list, anim_filename)

    all_rmse_tensor = torch.tensor(all_rmse_all)
    overall_mean_rmse = torch.mean(all_rmse_tensor, dim=0)
    overall_std_rmse = torch.std(all_rmse_tensor, dim=0)
    print("Overall Mean RMSE over rollout steps:", overall_mean_rmse)
    print("Overall Std RMSE over rollout steps:", overall_std_rmse)

    timesteps = [(i + 1) * (20 / 60) for i in range(rollout_length)]
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, overall_mean_rmse.numpy(), label="Mean RMSE", linewidth=3)
    plt.fill_between(
        timesteps,
        (overall_mean_rmse - overall_std_rmse).numpy(),
        (overall_mean_rmse + overall_std_rmse).numpy(),
        alpha=0.3,
        label="± Std",
    )
    plt.xlabel("Time (Hours)", fontsize=20)
    plt.ylabel("RMSE (Water Depth)", fontsize=20)
    plt.title("Overall RMSE Curve Over Rollout", fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
