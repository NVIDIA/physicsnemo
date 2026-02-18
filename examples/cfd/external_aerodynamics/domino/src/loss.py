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

import torch
from typing import Literal, Any

from physicsnemo.models.domino.utils import unnormalize

from typing import Literal, Any

import torch.cuda.nvtx as nvtx

from physicsnemo.models.domino.utils import *


def compute_physics_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    dims: tuple[int, ...] | None,
    first_deriv: torch.nn.Module,
    eqn: Any,
    bounding_box: torch.Tensor,
    vol_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physics-based loss terms for Navier-Stokes equations.

    Args:
        output: Model output containing (output, coords_neighbors, output_neighbors, neighbors_list)
        target: Ground truth values
        mask: Mask for valid values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        dims: Dimensions for loss calculation
        first_deriv: First derivative calculator
        eqn: Equations
        bounding_box: Bounding box for normalization
        vol_factors: Volume factors for normalization

    Returns:
        Tuple of (data_loss, continuity_loss, momentum_x_loss, momentum_y_loss, momentum_z_loss)
    """
    # Physics loss enabled
    output, coords_neighbors, output_neighbors, neighbors_list = output
    batch_size = output.shape[1]
    fields, num_neighbors = output_neighbors.shape[3], output_neighbors.shape[2]
    coords_total = coords_neighbors[0, :]
    output_total = output_neighbors[0, :]
    output_total_unnormalized = unnormalize(
        output_total, vol_factors[0], vol_factors[1]
    )
    coords_total_unnormalized = unnormalize(
        coords_total, bounding_box[0], bounding_box[1]
    )

    # compute first order gradients on all the nodes from the neighbors_list
    grad_list = {}
    for parent_id, neighbor_ids in neighbors_list.items():
        neighbor_ids_tensor = torch.tensor(neighbor_ids).to(
            output_total_unnormalized.device
        )
        du = (
            output_total_unnormalized[:, [parent_id]]
            - output_total_unnormalized[:, neighbor_ids_tensor]
        )
        dv = (
            coords_total_unnormalized[:, [parent_id]]
            - coords_total_unnormalized[:, neighbor_ids_tensor]
        )
        grads = first_deriv.forward(
            coords=None, connectivity_tensor=None, y=None, du=du, dv=dv
        )
        grad = torch.cat(grads, dim=1)
        grad_list[parent_id] = grad

    # compute second order gradients on only the center node
    neighbor_ids_tensor = torch.tensor(neighbors_list[0]).to(
        output_total_unnormalized.device
    )
    grad_neighbors_center = torch.stack([v for v in grad_list.values()], dim=1)
    grad_neighbors_center = grad_neighbors_center.reshape(
        batch_size, len(neighbors_list[0]) + 1, -1
    )

    du = grad_neighbors_center[:, [0]] - grad_neighbors_center[:, neighbor_ids_tensor]
    dv = (
        coords_total_unnormalized[:, [0]]
        - coords_total_unnormalized[:, neighbor_ids_tensor]
    )

    # second order gradients
    ggrads_center = first_deriv.forward(
        coords=None, connectivity_tensor=None, y=None, du=du, dv=dv
    )
    ggrad_center = torch.cat(ggrads_center, dim=1)
    grad_neighbors_center = grad_neighbors_center.reshape(
        batch_size, len(neighbors_list[0]) + 1, 3, -1
    )

    # Get the outputs on the original nodes
    fields_center_unnormalized = output_total_unnormalized[:, 0, :]
    grad_center = grad_neighbors_center[:, 0, :, :]
    grad_grad_uvw_center = ggrad_center[:, :, :9]

    nu = 1.507 * 1e-5

    dict_mapping = {
        "u": fields_center_unnormalized[:, [0]],
        "v": fields_center_unnormalized[:, [1]],
        "w": fields_center_unnormalized[:, [2]],
        "p": fields_center_unnormalized[:, [3]],
        "nu": nu + fields_center_unnormalized[:, [4]],
        "u__x": grad_center[:, 0, [0]],
        "u__y": grad_center[:, 1, [0]],
        "u__z": grad_center[:, 2, [0]],
        "v__x": grad_center[:, 0, [1]],
        "v__y": grad_center[:, 1, [1]],
        "v__z": grad_center[:, 2, [1]],
        "w__x": grad_center[:, 0, [2]],
        "w__y": grad_center[:, 1, [2]],
        "w__z": grad_center[:, 2, [2]],
        "p__x": grad_center[:, 0, [3]],
        "p__y": grad_center[:, 1, [3]],
        "p__z": grad_center[:, 2, [3]],
        "nu__x": grad_center[:, 0, [4]],
        "nu__y": grad_center[:, 1, [4]],
        "nu__z": grad_center[:, 2, [4]],
        "u__x__x": grad_grad_uvw_center[:, 0, [0]],
        "u__x__y": grad_grad_uvw_center[:, 1, [0]],
        "u__x__z": grad_grad_uvw_center[:, 2, [0]],
        "u__y__x": grad_grad_uvw_center[:, 1, [0]],  # same as __x__y
        "u__y__y": grad_grad_uvw_center[:, 1, [1]],
        "u__y__z": grad_grad_uvw_center[:, 2, [1]],
        "u__z__x": grad_grad_uvw_center[:, 2, [0]],  # same as __x__z
        "u__z__y": grad_grad_uvw_center[:, 2, [1]],  # same as __y__z
        "u__z__z": grad_grad_uvw_center[:, 2, [2]],
        "v__x__x": grad_grad_uvw_center[:, 0, [3]],
        "v__x__y": grad_grad_uvw_center[:, 1, [3]],
        "v__x__z": grad_grad_uvw_center[:, 2, [3]],
        "v__y__x": grad_grad_uvw_center[:, 1, [3]],  # same as __x__y
        "v__y__y": grad_grad_uvw_center[:, 1, [4]],
        "v__y__z": grad_grad_uvw_center[:, 2, [4]],
        "v__z__x": grad_grad_uvw_center[:, 2, [3]],  # same as __x__z
        "v__z__y": grad_grad_uvw_center[:, 2, [4]],  # same as __y__z
        "v__z__z": grad_grad_uvw_center[:, 2, [5]],
        "w__x__x": grad_grad_uvw_center[:, 0, [6]],
        "w__x__y": grad_grad_uvw_center[:, 1, [6]],
        "w__x__z": grad_grad_uvw_center[:, 2, [6]],
        "w__y__x": grad_grad_uvw_center[:, 1, [6]],  # same as __x__y
        "w__y__y": grad_grad_uvw_center[:, 1, [7]],
        "w__y__z": grad_grad_uvw_center[:, 2, [7]],
        "w__z__x": grad_grad_uvw_center[:, 2, [6]],  # same as __x__z
        "w__z__y": grad_grad_uvw_center[:, 2, [7]],  # same as __y__z
        "w__z__z": grad_grad_uvw_center[:, 2, [8]],
    }
    continuity = eqn["continuity"].evaluate(dict_mapping)["continuity"]
    momentum_x = eqn["momentum_x"].evaluate(dict_mapping)["momentum_x"]
    momentum_y = eqn["momentum_y"].evaluate(dict_mapping)["momentum_y"]
    momentum_z = eqn["momentum_z"].evaluate(dict_mapping)["momentum_z"]

    # Compute the weights for the equation residuals
    weight_continuity = torch.sigmoid(0.5 * (torch.abs(continuity) - 10))
    weight_momentum_x = torch.sigmoid(0.5 * (torch.abs(momentum_x) - 10))
    weight_momentum_y = torch.sigmoid(0.5 * (torch.abs(momentum_y) - 10))
    weight_momentum_z = torch.sigmoid(0.5 * (torch.abs(momentum_z) - 10))

    weighted_continuity = weight_continuity * torch.abs(continuity)
    weighted_momentum_x = weight_momentum_x * torch.abs(momentum_x)
    weighted_momentum_y = weight_momentum_y * torch.abs(momentum_y)
    weighted_momentum_z = weight_momentum_z * torch.abs(momentum_z)

    # Compute data loss
    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * target**2.0, dims)
    else:
        denom = torch.sum(mask)

    del coords_total, output_total
    torch.cuda.empty_cache()

    return (
        torch.mean(num / denom),
        torch.mean(torch.abs(weighted_continuity)),
        torch.mean(torch.abs(weighted_momentum_x)),
        torch.mean(torch.abs(weighted_momentum_y)),
        torch.mean(torch.abs(weighted_momentum_z)),
    )


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
) -> torch.Tensor:
    """Calculate mean squared error or root mean squared error with masking for padded values.

    Args:
        output: Predicted values from the model
        target: Ground truth values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        padded_value: Value used for padding in the tensor

    Returns:
        Calculated loss as a scalar tensor
    """
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * (target - torch.mean(target, (0, 1))) ** 2.0, dims)
        loss = torch.mean(num / denom)
    elif loss_type == "mse":
        denom = torch.sum(mask)
        loss = torch.mean(num / denom)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss


def loss_fn_with_physics(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
    first_deriv: torch.nn.Module = None,
    eqn: Any = None,
    bounding_box: torch.Tensor = None,
    vol_factors: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate loss with physics-based terms for appropriate equations.

    Args:
        output: Predicted values from the model (with neighbor data when physics enabled)
        target: Ground truth values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        padded_value: Value used for padding in the tensor
        first_deriv: First derivative calculator
        eqn: Equations
        bounding_box: Bounding box for normalization
        vol_factors: Volume factors for normalization

    Returns:
        Tuple of (data_loss, continuity_loss, momentum_x_loss, momentum_y_loss, momentum_z_loss)
    """
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    # Call the physics loss computation function
    return compute_physics_loss(
        output=output,
        target=target,
        mask=mask,
        loss_type=loss_type,
        dims=dims,
        first_deriv=first_deriv,
        eqn=eqn,
        bounding_box=bounding_box,
        vol_factors=vol_factors,
    )


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Combined scalar and vector loss as a scalar tensor
    """
    # Separate the scalar and vector components:
    output_scalar, output_vector = torch.split(output, [1, 3], dim=2)
    target_scalar, target_vector = torch.split(target, [1, 3], dim=2)

    numerator = torch.mean((output_scalar - target_scalar) ** 2.0)
    vector_diff_sq = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))
    if loss_type == "mse":
        masked_loss_pres = numerator
        masked_loss_ws = torch.sum(vector_diff_sq)
    else:
        denom = torch.mean((target_scalar - torch.mean(target_scalar, (0, 1))) ** 2.0)
        masked_loss_pres = numerator / denom

        # Compute the mean diff**2 of the vector component, leave the last dimension:
        masked_loss_ws_num = vector_diff_sq
        masked_loss_ws_denom = torch.mean(
            (target_vector - torch.mean(target_vector, (0, 1))) ** 2.0, (0, 1)
        )
        masked_loss_ws = torch.sum(masked_loss_ws_num / masked_loss_ws_denom)

    loss = masked_loss_pres + masked_loss_ws

    return loss / 4.0


def loss_fn_area(
    output: torch.Tensor,
    target: torch.Tensor,
    normals: torch.Tensor,
    area: torch.Tensor,
    area_scaling_factor: float,
    loss_type: Literal["mse", "rmse"],
) -> torch.Tensor:
    """Calculate area-weighted loss for surface data considering normal vectors.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        normals: Normal vectors for the surface
        area: Area values for surface elements
        area_scaling_factor: Scaling factor for area weighting
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Area-weighted loss as a scalar tensor
    """
    area = area * area_scaling_factor
    area_scale_factor = area

    # Separate the scalar and vector components.
    target_scalar, target_vector = torch.split(
        target * area_scale_factor, [1, 3], dim=2
    )
    output_scalar, output_vector = torch.split(
        output * area_scale_factor, [1, 3], dim=2
    )

    # Apply the normals to the scalar components (only [:,:,0]):
    normals, _ = torch.split(normals, [1, normals.shape[-1] - 1], dim=2)
    target_scalar = target_scalar * normals
    output_scalar = output_scalar * normals

    # Compute the mean diff**2 of the scalar component:
    masked_loss_pres = torch.mean(((output_scalar - target_scalar) ** 2.0), dim=(0, 1))
    if loss_type == "rmse":
        masked_loss_pres /= torch.mean(
            (target_scalar - torch.mean(target_scalar, (0, 1))) ** 2.0, dim=(0, 1)
        )

    # Compute the mean diff**2 of the vector component, leave the last dimension:
    masked_loss_ws = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))
    if loss_type == "rmse":
        masked_loss_ws /= torch.mean(
            (target_vector - torch.mean(target_vector, (0, 1))) ** 2.0, (0, 1)
        )

    # Combine the scalar and vector components:
    loss = 0.25 * (masked_loss_pres + torch.sum(masked_loss_ws))

    return loss


def integral_loss_fn(
    output, target, area, normals, stream_velocity=None, padded_value=-10
):
    drag_loss = drag_loss_fn(
        output, target, area, normals, stream_velocity=stream_velocity, padded_value=-10
    )
    lift_loss = lift_loss_fn(
        output, target, area, normals, stream_velocity=stream_velocity, padded_value=-10
    )
    return lift_loss + drag_loss


def lift_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3

    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    normals = torch.select(normals, 2, 2)
    # output_true_0 = output_true[:, :, 0]
    output_true_0 = output_true.select(2, 0)
    output_pred_0 = output_pred.select(2, 0)

    pres_true = output_true_0 * normals
    pres_pred = output_pred_0 * normals

    wz_true = output_true[:, :, -1]
    wz_pred = output_pred[:, :, -1]

    masked_pred = torch.mean(pres_pred + wz_pred, (1))
    masked_truth = torch.mean(pres_true + wz_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def drag_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 0]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 0]

    wx_true = output_true[:, :, 1]
    wx_pred = output_pred[:, :, 1]

    masked_pred = torch.mean(pres_pred + wx_pred, (1))
    masked_truth = torch.mean(pres_true + wx_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def compute_loss_dict(
    prediction_vol: torch.Tensor,
    prediction_surf: torch.Tensor,
    batch_inputs: dict,
    loss_fn_type: dict,
    integral_scaling_factor: float,
    surf_loss_scaling: float,
    vol_loss_scaling: float,
    first_deriv: torch.nn.Module | None = None,
    eqn: Any = None,
    bounding_box: torch.Tensor | None = None,
    vol_factors: torch.Tensor | None = None,
    add_physics_loss: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the loss terms in a single function call.

    Computes:
    - Volume loss if prediction_vol is not None
    - Surface loss if prediction_surf is not None
    - Integral loss if prediction_surf is not None
    - Total loss as a weighted sum of the above

    Returns:
    - Total loss as a scalar tensor
    - Dictionary of loss terms (for logging, etc)
    """
    nvtx.range_push("Loss Calculation")
    total_loss_terms = []
    loss_dict = {}

    if prediction_vol is not None:
        target_vol = batch_inputs["volume_fields"]

        if add_physics_loss:
            loss_vol = loss_fn_with_physics(
                prediction_vol,
                target_vol,
                loss_fn_type.loss_type,
                padded_value=-10,
                first_deriv=first_deriv,
                eqn=eqn,
                bounding_box=bounding_box,
                vol_factors=vol_factors,
            )
            loss_dict["loss_vol"] = loss_vol[0]
            loss_dict["loss_continuity"] = loss_vol[1]
            loss_dict["loss_momentum_x"] = loss_vol[2]
            loss_dict["loss_momentum_y"] = loss_vol[3]
            loss_dict["loss_momentum_z"] = loss_vol[4]
            total_loss_terms.append(loss_vol[0])
            total_loss_terms.append(loss_vol[1])
            total_loss_terms.append(loss_vol[2])
            total_loss_terms.append(loss_vol[3])
            total_loss_terms.append(loss_vol[4])
        else:
            loss_vol = loss_fn(
                prediction_vol,
                target_vol,
                loss_fn_type.loss_type,
                padded_value=-10,
            )
            loss_dict["loss_vol"] = loss_vol
            total_loss_terms.append(loss_vol)

    if prediction_surf is not None:
        target_surf = batch_inputs["surface_fields"]
        surface_areas = batch_inputs["surface_areas"]
        surface_areas = torch.unsqueeze(surface_areas, -1)
        surface_normals = batch_inputs["surface_normals"]

        # Needs to be taken from the dataset
        stream_velocity = batch_inputs["global_params_values"][:, 0, :]

        loss_surf = loss_fn_surface(
            prediction_surf,
            target_surf,
            loss_fn_type.loss_type,
        )

        loss_surf_area = loss_fn_area(
            prediction_surf,
            target_surf,
            surface_normals,
            surface_areas,
            area_scaling_factor=loss_fn_type.area_weighing_factor,
            loss_type=loss_fn_type.loss_type,
        )

        if loss_fn_type.loss_type == "mse":
            loss_surf = loss_surf * surf_loss_scaling
            loss_surf_area = loss_surf_area * surf_loss_scaling

        total_loss_terms.append(loss_surf)
        loss_dict["loss_surf"] = loss_surf
        total_loss_terms.append(loss_surf_area)
        loss_dict["loss_surf_area"] = loss_surf_area
        loss_integral = (
            integral_loss_fn(
                prediction_surf,
                target_surf,
                surface_areas,
                surface_normals,
                stream_velocity,
                padded_value=-10,
            )
        ) * integral_scaling_factor
        loss_dict["loss_integral"] = loss_integral
        total_loss_terms.append(loss_integral)

    total_loss = sum(total_loss_terms)
    loss_dict["total_loss"] = total_loss
    nvtx.range_pop()

    return total_loss, loss_dict
