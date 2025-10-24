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
Utility functions for physics-based loss computation and custom loss definitions.
"""

import torch
import torch.nn.functional as F


def get_batch_vector(graph):
    """
    Build a batch vector from node counts for a batched DGL graph.

    Args:
        graph (DGLGraph): A batched DGL graph.

    Returns:
        torch.Tensor: A tensor where each node is assigned the index of its graph in the batch.
    """
    node_counts = graph.batch_num_nodes()
    if not isinstance(node_counts, torch.Tensor):
        node_counts = torch.tensor(node_counts, device=graph.device)
    # Create a batch vector where each node receives the corresponding graph index.
    batch_vec = torch.cat(
        [
            torch.full((int(n),), i, device=graph.device)
            for i, n in enumerate(node_counts)
        ]
    )
    return batch_vec


def compute_physics_loss(pred, physics_data, graph, delta_t=1200.0):
    """
    Compute a physics-based continuity loss in the denormalized domain.

    For each graph sample, the predicted total volume is computed as:
        predicted_total_volume = past_volume_denorm + volume_std * (sum of predicted volume differences)
    where:
        past_volume_denorm = past_volume_norm * volume_std + (num_nodes * volume_mean)

    Future volume is denormalized similarly:
        future_volume_denorm = future_volume_norm * volume_std + (num_nodes * volume_mean)

    Two continuity terms are computed:
        - term1: Uses average inflow and precipitation (denorm_avg_inflow and denorm_avg_precip)
        - term2: Uses next step's inflow and precipitation (denorm_next_inflow and denorm_next_precip)

    An effective precipitation term is computed as:
        new_precip_term = base_precip * infiltration_area_sum

    Finally, the physics loss is the mean of the sum of term1 and term2 across all graph samples.

    Args:
        pred (torch.Tensor): Model predictions (expected volume difference).
        physics_data (dict): Dictionary containing various denormalized physics parameters.
        graph (DGLGraph): Batched DGL graph.
        delta_t (float): Time delta over which the continuity is enforced.

    Returns:
        torch.Tensor: Mean physics loss across all graph samples.
    """
    batch = get_batch_vector(graph)
    unique_ids = torch.unique(batch)
    predicted_diff = pred[:, 1]  # Predicted volume difference (normalized)
    physics_losses = []

    for uid in unique_ids:
        mask = batch == uid
        pred_diff_sum = predicted_diff[mask].sum()

        idx = (unique_ids == uid).nonzero(as_tuple=False).item()
        past_volume_norm = physics_data["past_volume"][idx]
        future_volume_norm = physics_data["future_volume"][idx]
        # For term1: use average inflow and precipitation
        denorm_avg_inflow = physics_data["avg_inflow"][idx]
        denorm_avg_precip = physics_data["avg_precipitation"][idx]
        # For term2: use next step inflow and precipitation
        denorm_next_inflow = physics_data["next_inflow"][idx]
        denorm_next_precip = physics_data["next_precip"][idx]

        volume_mean = physics_data["volume_mean"][idx]
        volume_std = physics_data["volume_std"][idx]
        num_nodes = physics_data["num_nodes"][idx]
        area_sum = physics_data["area_sum"][idx]
        infiltration_area_sum = physics_data["infiltration_area_sum"][idx]

        # Denormalize past and future volumes.
        past_volume_denorm = past_volume_norm * volume_std + num_nodes * volume_mean
        future_volume_denorm = future_volume_norm * volume_std + num_nodes * volume_mean

        # Compute the predicted total volume.
        pred_total_volume = past_volume_denorm + volume_std * pred_diff_sum

        # Compute effective precipitation terms.
        new_precip_term = denorm_avg_precip * infiltration_area_sum
        new_next_precip_term = denorm_next_precip * infiltration_area_sum

        temp1 = pred_total_volume - (
            past_volume_denorm + delta_t * (denorm_avg_inflow + new_precip_term)
        )

        temp2 = (
            future_volume_denorm
            - pred_total_volume
            - delta_t * (denorm_next_inflow + new_next_precip_term)
        )

        # Compute continuity terms using ReLU to enforce non-negativity.
        term1 = (
            F.relu(
                (
                    pred_total_volume
                    - (
                        past_volume_denorm
                        + delta_t * (denorm_avg_inflow + new_precip_term)
                    )
                )
                / area_sum
            )
            ** 2
        )
        term2 = (
            F.relu(
                (
                    future_volume_denorm
                    - pred_total_volume
                    - delta_t * (denorm_next_inflow + new_next_precip_term)
                )
                / area_sum
            )
            ** 2
        )

        physics_losses.append(term1 + term2)

    if physics_losses:
        return torch.stack(physics_losses).mean()
    else:
        return torch.tensor(0.0, device=pred.device)


def custom_loss(pred, targets):
    """
    Compute a custom loss as the sum of MSE losses on water depth and volume predictions.

    Args:
        pred (torch.Tensor): Model predictions with two columns (depth and volume difference).
        targets (torch.Tensor): Ground truth targets.

    Returns:
        dict: Dictionary containing the total loss and individual losses for depth and volume.
    """
    pred_depth = pred[:, 0]
    pred_volume = pred[:, 1]
    target_depth = targets[:, 0]
    target_volume = targets[:, 1]
    loss_depth = F.mse_loss(pred_depth, target_depth, reduction="mean")
    loss_volume = F.mse_loss(pred_volume, target_volume, reduction="mean")
    total_loss = loss_depth + loss_volume
    return {
        "total_loss": total_loss,
        "loss_depth": loss_depth,
        "loss_volume": loss_volume,
    }
