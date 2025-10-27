"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====================================================================
                            GRADIENTS EXTRACTION
=====================================================================

This module provides gradient extraction capabilities for reservoir
simulation forward modeling. It includes functions for computing
numerical derivatives, processing gradient data, and analyzing
spatial gradients in simulation results.

Key Features:
- Numerical derivative computation
- Gradient data processing and validation
- Spatial gradient analysis
- Integration with simulation workflows

Usage:
    from forward.gradients_extract import (
        dx,
        dy,
        dz,
        process_and_print,
        extract_gradients
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import re
import logging
from pathlib import Path
from typing import Tuple, Union
from shutil import rmtree
from collections import OrderedDict

# 🔧 Third-party Libraries
import numpy as np
import numpy.linalg
import numpy.matlib

# 🔥 Torch & PhyNeMo
import torch
import torch.nn.functional as F
import hashlib

# 📦 Local Modules
from forward.binaries_extract import (
    normalize_tensors_adjusted,
)

from forward.simulator import (
    calc_mu_g,
    calc_bg,
    calc_bo,
    StoneIIModel,
)

from compare.batch.misc_forward_utils import EclBinaryParser


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def replace_with_mean(tensor):
    tensor = tensor.to(torch.float32)
    valid_elements = tensor[torch.isfinite(tensor)]
    if valid_elements.numel() > 0:  # Check if there are any valid elements
        mean_value = valid_elements.mean()  # ✅ Retains gradients
        perturbation = torch.normal(mean=0.0, std=0.01, size=(1,), device=tensor.device)
        perturbed_mean_value = mean_value + perturbation
    else:
        perturbed_mean_value = torch.tensor(
            1e-4, device=tensor.device, dtype=torch.float32, requires_grad=True
        )
    ouut = torch.where(
        torch.isnan(tensor) | torch.isinf(tensor), perturbed_mean_value, tensor
    )
    ouut = torch.clamp(ouut, min=1e-6)  # ✅ Keeps gradients flowing
    return ouut


def dx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor"
    var = inpt[:, channel : channel + 1, :, :].to(torch.float32)
    if order == 1:
        ddx1D = torch.tensor([-0.5, 0.0, 0.5], device=inpt.device, dtype=torch.float32)
    elif order == 3:
        ddx1D = torch.tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ],
            device=inpt.device,
            dtype=torch.float32,
        )
    ddx3D = ddx1D.view([1, 1] + dim * [1] + [-1] + (1 - dim) * [1])
    padding_size = 4 * [(ddx1D.shape[0] - 1) // 2]
    if padding == "zeros":
        var = F.pad(var, padding_size, "constant", 0)
    elif padding == "replication":
        var = F.pad(var, padding_size, "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output.mul_(1.0 / dx)
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]
    return output


def ddx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute second order numerical derivatives of input tensor"
    var = inpt[:, channel : channel + 1, :, :].to(torch.float32)
    if order == 1:
        ddx1D = torch.tensor([1.0, -2.0, 1.0], device=inpt.device, dtype=torch.float32)
    elif order == 3:
        ddx1D = torch.tensor(
            [
                1.0 / 90.0,
                -3.0 / 20.0,
                3.0 / 2.0,
                -49.0 / 18.0,
                3.0 / 2.0,
                -3.0 / 20.0,
                1.0 / 90.0,
            ],
            device=inpt.device,
            dtype=torch.float32,
        )
    ddx3D = ddx1D.view([1, 1] + dim * [1] + [-1] + (1 - dim) * [1])
    padding_size = 4 * [(ddx1D.shape[0] - 1) // 2]
    if padding == "zeros":
        var = F.pad(var, padding_size, "constant", 0)
    elif padding == "replication":
        var = F.pad(var, padding_size, "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output.mul_(1.0 / dx**2)
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]
    return output


def compute_differential(u, dxf):
    batch_size, channels, nz, height, width = u.shape
    derivatives_x = []
    derivatives_y = []
    derivatives_z = []  # List to store derivatives in z direction
    for i in range(nz):
        slice_u = u[:, :, i, :, :]
        dudx_fdm = dx(slice_u, dx=dxf, channel=0, dim=0, order=1, padding="replication")
        dudy_fdm = dx(slice_u, dx=dxf, channel=0, dim=1, order=1, padding="replication")
        derivatives_x.append(dudx_fdm)
        derivatives_y.append(dudy_fdm)
        if i > 0 and i < nz - 1:
            dudz_fdm = (u[:, :, i + 1, :, :] - u[:, :, i - 1, :, :]) / (2 * dxf)
            derivatives_z.append(dudz_fdm)
        else:
            dudz_fdm = torch.zeros_like(slice_u)
            derivatives_z.append(dudz_fdm)
    dudx_fdm = torch.stack(derivatives_x, dim=2)
    dudy_fdm = torch.stack(derivatives_y, dim=2)
    dudz_fdm = torch.stack(derivatives_z, dim=2)  # Stack the z derivatives
    return dudx_fdm, dudy_fdm, dudz_fdm  # Return the z derivatives as well


def rmsee(predictions, targets):
    noww = predictions.reshape(-1, 1)
    measurment = targets.reshape(-1, 1)
    rmse_val = (np.sum(((noww - measurment) ** 2))) ** (0.5) / (measurment.shape[0])
    return rmse_val


def compute_second_differential(u, dxf):
    batch_size, channels, nz, height, width = u.shape
    second_derivatives_x = []
    second_derivatives_y = []
    second_derivatives_z = []  # List to store second derivatives in z direction
    for i in range(nz):
        slice_u = u[:, :, i, :, :]  # Extract the ith slice in the nz dimension
        dduddx_fdm = ddx(
            slice_u, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        dduddy_fdm = ddx(
            slice_u, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )
        second_derivatives_x.append(dduddx_fdm)
        second_derivatives_y.append(dduddy_fdm)
        if i > 1 and i < nz - 2:
            dduddz_fdm = (u[:, :, i + 2, :, :] - 2 * slice_u + u[:, :, i - 2, :, :]) / (
                dxf**2
            )
            second_derivatives_z.append(dduddz_fdm)
        else:
            dduddz_fdm = torch.zeros_like(slice_u)
            second_derivatives_z.append(dduddz_fdm)
    dduddx_fdm = torch.stack(second_derivatives_x, dim=2)
    dduddy_fdm = torch.stack(second_derivatives_y, dim=2)
    dduddz_fdm = torch.stack(second_derivatives_z, dim=2)
    return dduddx_fdm, dduddy_fdm, dduddz_fdm


def extra_loss(ytrue, y_hat):
    error = ytrue - y_hat
    numerator_1 = torch.linalg.vector_norm(
        error, ord=2, dim=(1, 2, 3, 4)
    )  # Batch-wise norm
    denominator_1 = (
        torch.linalg.vector_norm(ytrue, ord=2, dim=(1, 2, 3, 4)) + 1e-6
    )  # Avoid division by zero
    term1 = numerator_1 / denominator_1
    dy_dx = ytrue[:, :, :, 1:, :] - ytrue[:, :, :, :-1, :]
    dy_hat_dx = y_hat[:, :, :, 1:, :] - y_hat[:, :, :, :-1, :]
    dy_dy = ytrue[:, :, :, :, 1:] - ytrue[:, :, :, :, :-1]
    dy_hat_dy = y_hat[:, :, :, :, 1:] - y_hat[:, :, :, :, :-1]
    dy_dz = ytrue[:, :, 1:, :, :] - ytrue[:, :, :-1, :, :]
    dy_hat_dz = y_hat[:, :, 1:, :, :] - y_hat[:, :, :-1, :, :]
    grad_error_dx = dy_dx - dy_hat_dx
    grad_error_dy = dy_dy - dy_hat_dy
    grad_error_dz = dy_dz - dy_hat_dz
    numerator_dx = torch.linalg.vector_norm(grad_error_dx, ord=2, dim=(1, 2, 3, 4))
    denominator_dx = torch.linalg.vector_norm(dy_dx, ord=2, dim=(1, 2, 3, 4)) + 1e-6
    numerator_dy = torch.linalg.vector_norm(grad_error_dy, ord=2, dim=(1, 2, 3, 4))
    denominator_dy = torch.linalg.vector_norm(dy_dy, ord=2, dim=(1, 2, 3, 4)) + 1e-6
    numerator_dz = torch.linalg.vector_norm(grad_error_dz, ord=2, dim=(1, 2, 3, 4))
    denominator_dz = torch.linalg.vector_norm(dy_dz, ord=2, dim=(1, 2, 3, 4)) + 1e-6
    term2 = (
        numerator_dx / denominator_dx
        + numerator_dy / denominator_dy
        + numerator_dz / denominator_dz
    ) / 3
    total_loss = term1 + term2
    return total_loss.mean()


def kmeans(data, num_clusters=2, num_iters=10):
    data_flat = data.view(-1, 1)
    centroids = data_flat[torch.randperm(data_flat.size(0))[:num_clusters]]
    for _ in range(num_iters):
        distances = torch.cdist(data_flat, centroids)
        labels = distances.argmin(dim=1)
        new_centroids = torch.stack(
            [data_flat[labels == i].mean(dim=0) for i in range(num_clusters)]
        )
        if torch.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels.view_as(data), centroids


def compute_boundary_mask(cluster_labels, cluster_value_1=0, cluster_value_2=1):
    cluster1_mask = (cluster_labels == cluster_value_1).float()
    cluster2_mask = (cluster_labels == cluster_value_2).float()
    kernel = (
        torch.ones((3, 3, 3), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(cluster_labels.device)
    )
    cluster1_padded = F.pad(
        cluster1_mask.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1, 1, 1),
        mode="constant",
        value=0,
    )
    cluster2_padded = F.pad(
        cluster2_mask.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1, 1, 1),
        mode="constant",
        value=0,
    )
    boundary1 = F.conv3d(cluster1_padded, kernel).squeeze()
    boundary2 = F.conv3d(cluster2_padded, kernel).squeeze()
    boundary = (boundary1 > 0) & (boundary2 > 0)
    return boundary.float()


def compute_hamming_distance(mask1, mask2):
    mask1_flat = mask1.view(-1)
    mask2_flat = mask2.view(-1)
    hamming_distance = torch.sum(mask1_flat != mask2_flat).item()
    return hamming_distance


def process_tensor_sat(tensor, truee, num_clusters=2):
    B, T, nz, nx, ny = tensor.shape
    total_hamming_distance = 0
    num_elements = B * T  # Total number of B and T pairs
    for b in range(B):
        for t in range(T):
            volume = tensor[b, t]
            volumet = truee[b, t]
            cluster_labels, _ = kmeans(volume, num_clusters=num_clusters)
            cluster_labelst, _ = kmeans(volumet, num_clusters=num_clusters)
            predicted_boundary_mask = compute_boundary_mask(cluster_labels)
            true_boundary_mask = compute_boundary_mask(cluster_labelst)
            hamming_distance = compute_hamming_distance(
                predicted_boundary_mask, true_boundary_mask
            )
            total_hamming_distance += hamming_distance
    mean_hamming_distance = total_hamming_distance / num_elements
    return mean_hamming_distance


def process_tensor(tensor):
    tensor = torch.where(
        torch.isnan(tensor) | torch.isinf(tensor),
        torch.tensor(1e-6, dtype=torch.float32),
        tensor,
    )
    tensor = tensor.to(dtype=torch.float32)
    return tensor


class Labelledset:
    def __init__(self, data, device):
        self.device = device
        self.data = {
            key: torch.from_numpy(data[key]).to(self.device, torch.float32)
            for key in data.keys()
        }

    def __getitem__(self, index):
        return {key: self.data[key][index] for key in self.data.keys()}

    def __len__(self):
        first_key = next(iter(self.data))
        return len(self.data[first_key])


def _l2_relative_error(pred_var, true_var):
    epsilon = 1e-8  # Small constant to avoid division by zero
    new_var = torch.sqrt(
        torch.sum(torch.square(true_var - pred_var)) / (torch.var(true_var) + epsilon)
    )
    return new_var


def loss_func(x, y, types, lambda_weighting, p=2.0):
    batch_size = x.shape[0]
    if types == "eliptical":  # Loss for pressure system
        L2 = lambda_weighting * (x - y).pow_(2).sum()
        lyes = lambda_weighting * _l2_relative_error(x, y)
        loss = lyes + L2  # +  l1
    elif types == "hyperbolic":  # For hyperbolic saturation system
        L2 = lambda_weighting * (x - y).pow_(2).sum()
        # l1 = lambda_weighting * (extra_loss(y, x).sum())
        lyes = lambda_weighting * _l2_relative_error(x, y)
        loss = lyes + L2
    else:
        loss1 = lambda_weighting * (x - y).pow_(2).sum()
        lyes = lambda_weighting * _l2_relative_error(x, y)
        loss = lyes + loss1  # + gradient_loss
    loss = loss / batch_size
    return loss


def loss_func_physics(x, lambda_weighting):
    batch_size = x.shape[0]
    loss = (lambda_weighting * torch.abs(torch.nan_to_num(x, nan=0.0))).sum()
    loss = loss / batch_size
    return loss


def compute_gradient_3d(inpt, dx, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor for 3D data"
    inpt = inpt.to(torch.float32)
    if order == 1:
        ddx1D = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32, device=inpt.device)
    elif order == 3:
        ddx1D = torch.tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ],
            dtype=torch.float32,
            device=inpt.device,
        )
    padding_sizes = [(0, 0), (0, 0), (0, 0)]
    if dim == 0:
        ddx3D = ddx1D.view(1, 1, -1, 1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    elif dim == 1:
        ddx3D = ddx1D.view(1, 1, 1, -1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    else:  # dim == 2
        ddx3D = ddx1D.view(1, 1, 1, 1, -1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    outputs = []
    for ch in range(inpt.shape[1]):
        channel_data = inpt[:, ch : ch + 1]
        if padding == "zeros":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "constant",
                0,
            )
        elif padding == "replication":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "replicate",
            )
        out_ch = F.conv3d(channel_data, ddx3D, padding=0) * (1.0 / dx)
        outputs.append(out_ch)
    output = torch.cat(outputs, dim=1).to(torch.float32)
    return output


def compute_second_order_gradient_3d(inpt, dx, dim, padding="zeros"):
    "Compute second order numerical derivatives (Laplacian) of input tensor for 3D data"
    inpt = inpt.to(torch.float32)
    ddx1D = torch.tensor([-1.0, 2.0, -1.0], dtype=torch.float32, device=inpt.device)
    padding_sizes = [(0, 0), (0, 0), (0, 0)]
    if dim == 0:
        ddx3D = ddx1D.view(1, 1, -1, 1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    elif dim == 1:
        ddx3D = ddx1D.view(1, 1, 1, -1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    else:  # dim == 2
        ddx3D = ddx1D.view(1, 1, 1, 1, -1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    outputs = []
    for ch in range(inpt.shape[1]):
        channel_data = inpt[:, ch : ch + 1]
        if padding == "zeros":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "constant",
                0,
            )
        elif padding == "replication":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "replicate",
            )
        out_ch = F.conv3d(channel_data, ddx3D, padding=0) * (1.0 / (dx**2))
        outputs.append(out_ch)
    output = torch.cat(outputs, dim=1).to(torch.float32)
    return output


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    return rescaled_tensor * max_val


def replace_nans_and_infs(tensor, value=0.0):
    tensor[torch.isnan(tensor) | torch.isinf(tensor)] = value
    return tensor


def scale_tensor_abs(tensor, target_min, target_max):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    rescaled_tensor = tensor / max_val
    return min_val, max_val, rescaled_tensor


def scale_tensor_abs_pressure(tensor, max_val):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    rescaled_tensor = tensor / max_val
    return np.min(tensor), max_val, rescaled_tensor


def scale_tensor_absS(tensor, lenwels, N_pr):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    C = []
    Cmax = []
    Cmin = []
    for k in range(lenwels):
        Anow = tensor[:, :, k * N_pr : (k + 1) * N_pr]
        min_val = np.min(Anow)
        max_val = np.max(Anow)
        rescaled_tensor = Anow / max_val
        C.append(rescaled_tensor)
        Cmax.append(max_val)
        Cmin.append(min_val)
    get_it2 = np.concatenate(C, 2)
    return get_it2, Cmax, Cmin


def scale_tensor_absSin(tensor, N_pr):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    C = []
    Cmax = np.zeros((1, 6))
    Cmin = np.zeros((1, 6))
    Anow = tensor[:, :, :N_pr]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 0] = max_val
    Cmin[:, 0] = min_val
    Anow = tensor[:, :, N_pr : N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 1] = max_val
    Cmin[:, 1] = min_val
    Anow = tensor[:, :, N_pr + 1 : 2 * N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 2] = max_val
    Cmin[:, 2] = min_val
    Anow = tensor[:, :, 2 * N_pr + 1 : 3 * N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 3] = max_val
    Cmin[:, 3] = min_val
    Anow = tensor[:, :, 3 * N_pr + 1 : 4 * N_pr + 1]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 4] = max_val
    Cmin[:, 4] = min_val
    Anow = tensor[:, :, 4 * N_pr + 1 : 4 * N_pr + 2]
    min_val = np.min(Anow)
    max_val = np.max(Anow)
    rescaled_tensor = Anow / max_val
    C.append(rescaled_tensor)
    Cmax[:, 5] = max_val
    Cmin[:, 5] = min_val
    get_it2 = np.concatenate(C, 2)
    return get_it2, Cmax, Cmin


#
def replace_large_and_invalid_values(arr, placeholder=0.0):
    threshold = np.finfo(np.float32).max
    invalid_indices = (np.isnan(arr)) | (np.isinf(arr)) | (np.abs(arr) > threshold)
    arr[invalid_indices] = placeholder
    return arr


def clean_dict_arrays(data_dict):
    for key in data_dict:
        data_dict[key] = replace_large_and_invalid_values(data_dict[key])
    return data_dict


def clip_and_convert_to_float32(array):
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def clip_and_convert_to_float3(array):
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def Make_correct(array):
    new_array = np.zeros(
        (array.shape[0], array.shape[1], array.shape[3], array.shape[4], array.shape[2])
    )
    for kk in range(array.shape[0]):
        perm_big = np.zeros(
            (array.shape[1], array.shape[3], array.shape[4], array.shape[2])
        )
        for j in range(array.shape[1]):
            j1 = np.zeros((array.shape[3], array.shape[4], array.shape[2]))
            for i in range(array.shape[2]):
                j1[:, :, i] = array[kk, :, :, :, :][j, :, :, :][i, :, :]
            perm_big[j, :, :, :] = j1
        new_array[kk, :, :, :, :] = perm_big
    return new_array


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def extract_qs(steppi, steppi_indices, filenameui, injectors, gass, filename):
    well_namesg = [entry[-1] for entry in gass]  # gas injectors well names
    well_namesw = [entry[-1] for entry in injectors]  # water injectors well names
    unsmry_file = filenameui
    parser = EclBinaryParser(unsmry_file)
    vectorsdd = parser.read_vectors()
    namez = "WGIR"
    dfaa = vectorsdd[namez]
    filtered_columns = [
        coll
        for coll in dfaa.columns
        if any(well_namee in coll for well_namee in well_namesg)
    ]
    filtered_df = dfaa[filtered_columns]
    filtered_df = filtered_df[well_namesg]
    start_row = find_first_numeric_row(filtered_df)
    if start_row is not None:
        numeric_df = filtered_df.iloc[start_row:]
        all_arrays = numeric_df.to_numpy()
    else:
        all_arrays = None
    final_arrayg = all_arrays
    final_arrayg[final_arrayg <= 0] = 0
    outg = final_arrayg[steppi_indices - 1, :].astype(float)
    outg[outg <= 0] = 0
    namez = "WWIR"
    dfaa = vectorsdd[namez]
    filtered_columns = [
        coll
        for coll in dfaa.columns
        if any(well_namee in coll for well_namee in well_namesw)
    ]
    filtered_df = dfaa[filtered_columns]
    filtered_df = filtered_df[well_namesw]
    start_row = find_first_numeric_row(filtered_df)
    if start_row is not None:
        numeric_df = filtered_df.iloc[start_row:]
        all_arrays = numeric_df.to_numpy()
    else:
        all_arrays = None
    final_arrayg = all_arrays
    final_arrayg[final_arrayg <= 0] = 0
    outw = final_arrayg[steppi_indices - 1, :].astype(float)
    outw[outw <= 0] = 0
    return outg, outw


def get_dyna(steppi, well_indices, swatuse):
    mean_big_all = []
    for xx in range(steppi):
        mean_big = []  # Collects mean values for this particular timestep
        for idx, list1 in well_indices.items():  # Direct access to lists via .items()
            temp_perm_values = [
                swatuse[xx, i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else swatuse[xx, i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in list1
            ]
            mean_all = np.mean(temp_perm_values)
            mean_big.append(mean_all)
        mean_big_all.append(mean_big)
    outt2 = np.array(mean_big_all)
    return outt2


def get_dyna2(
    steppi, well_indices, well_indicesg, well_indiceso, swatuse, gasuse, oiluse, Q, Qg
):
    unique_well_names = OrderedDict()
    for idx, tuple_entry in enumerate(well_indices):
        well_name = tuple_entry[0]
        if well_name not in unique_well_names:
            unique_well_names[well_name] = len(unique_well_names)
    well_name_to_index = {name: index for index, name in enumerate(unique_well_names)}
    for xx in range(steppi):
        for well_name, q_idx in well_name_to_index.items():
            # Find all tuples corresponding to this well name to update swatuse accordingly
            entries_for_well = [t for t in well_indices if t[0] == well_name]
            total_value = Q[xx, q_idx]
            average_value = (
                total_value / len(entries_for_well) if entries_for_well else 0
            )
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_well:
                # logger.debug(f"Processing indices: {i_idx}, {j_idx}, {k_idx}")
                if int(k_idx) - 1 == int(l_idx) - 1:
                    swatuse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = (
                        average_value
                    )
                else:
                    swatuse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = average_value
    unique_well_namesg = OrderedDict()
    for idx, tuple_entry in enumerate(well_indicesg):
        well_nameg = tuple_entry[0]
        if well_nameg not in unique_well_namesg:
            unique_well_namesg[well_nameg] = len(unique_well_namesg)
    well_name_to_indexg = {name: index for index, name in enumerate(unique_well_namesg)}
    for xx in range(steppi):
        for well_nameg, q_idxg in well_name_to_indexg.items():
            entries_for_wellg = [t for t in well_indicesg if t[0] == well_nameg]
            total_valueg = Q[xx, q_idxg]
            average_valueg = (
                total_valueg / len(entries_for_wellg) if entries_for_wellg else 0
            )
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_wellg:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    gasuse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = (
                        average_valueg
                    )
                else:
                    gasuse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = average_valueg
    unique_well_nameso = OrderedDict()
    for idx, tuple_entry in enumerate(well_indiceso):
        well_nameo = tuple_entry[0]
        if well_nameo not in unique_well_nameso:
            unique_well_nameso[well_nameo] = len(unique_well_nameso)
    well_name_to_indexo = {name: index for index, name in enumerate(unique_well_nameso)}
    for xx in range(steppi):
        for well_nameo, q_idxo in well_name_to_indexo.items():
            entries_for_wello = [t for t in well_indiceso if t[0] == well_nameo]
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_wello:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    oiluse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = -1
                else:
                    oiluse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = -1
    return swatuse, gasuse, oiluse


def find_first_numeric_row(df):
    """Find the first row in the DataFrame where all data is numeric."""
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None


def process_data(data):
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )

    return well_indices


def read_compdats2(filename, file_path):
    with open(filename, "r") as file:
        data_gas = []  # List to collect gas entries
        data_water = []  # List to collect water entries
        data_oil = []  # List to collect oil entries
        injector_gas = set()  # Set to collect gas injector well names
        injector_water = set()  # Set to collect water injector well names
        producer_oil = set()
        start_collecting_welspecs = False
        start_collecting_wconinje = False
        start_collecting_wconhist = False
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "WELSPECS" in stripped_line:
                start_collecting_welspecs = True
                continue
            if start_collecting_welspecs and stripped_line.startswith("/"):
                start_collecting_welspecs = False
                continue
            if "WCONINJE" in stripped_line:
                start_collecting_wconinje = True
                continue
            if start_collecting_wconinje and stripped_line.startswith("/"):
                start_collecting_wconinje = False
                continue
            if "WCONHIST" in stripped_line:
                start_collecting_wconhist = True
                continue
            if start_collecting_wconhist and stripped_line.startswith("/"):
                start_collecting_wconhist = False
                continue
            if start_collecting_welspecs:
                parts = stripped_line.split()
                if (
                    len(parts) > 5
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    i = parts[2]
                    j = parts[3]
                    if parts[5].strip("'") == "GAS":
                        data_gas.append((well_name, i, j))
                    elif parts[5].strip("'") == "WATER":
                        data_water.append((well_name, i, j))
                    elif parts[5].strip("'") == "OIL":
                        data_oil.append((well_name, i, j))
            if start_collecting_wconinje:
                parts = stripped_line.split()
                if (
                    len(parts) > 3
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    fluid_type = parts[1].strip("'")
                    if fluid_type == "GAS":
                        injector_gas.add(well_name)

                    elif fluid_type == "WATER":
                        injector_water.add(well_name)
            if start_collecting_wconhist:
                parts = stripped_line.split()
                if (
                    len(parts) > 3
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    producer_oil.add(well_name)
    data = convert_to_list(process_data2(data_oil))
    data.sort(key=lambda x: x[2])
    with open(file_path, "r") as file:
        lines = file.readlines()
    well_namesoil = set()
    capture = False
    for line in lines:
        line = line.strip()
        if line == "WOPR":
            capture = True
            continue
        if capture:
            if line == "/":
                break
            well_name = line.strip(" '")
            well_namesoil.add(well_name)
    gass, water, oil = extract_tuples(injector_gas, injector_water, well_namesoil, data)
    return gass, oil, water


def process_data2(data):
    well_indices = {}
    for entry in data:
        well_name = entry[0]
        if well_name not in well_indices:
            well_indices[well_name] = []
        i_index = int(entry[1]) - 1  # Convert to zero-based index
        j_index = int(entry[2]) - 1  # Convert to zero-based index
        well_indices[well_name].append((i_index, j_index))
    return well_indices


def convert_to_list(well_data):
    output_list = []
    for well_name, indices in well_data.items():
        for i, j in indices:
            output_list.append((i, j, well_name))
    return output_list


def extract_tuples(set1, set2, set3, tuples_list):
    extracted_set1 = [tup for tup in tuples_list if tup[2] in set1]
    extracted_set1.sort(key=lambda x: x[2])
    extracted_set2 = [tup for tup in tuples_list if tup[2] in set2]
    extracted_set2.sort(key=lambda x: x[2])
    combined_set = list(set1) + list(set2)
    extracted_set3 = [tup for tup in tuples_list if tup[2] in set3]
    extracted_set3.sort(key=lambda x: x[2])
    final_remaining_list = [tup for tup in extracted_set3 if tup[2] not in combined_set]
    final_remaining_list.sort(key=lambda x: x[2])
    return extracted_set1, extracted_set2, final_remaining_list


def read_compdats(filename, well_names):
    with open(filename, "r") as file:
        start_collecting = False
        data = []  # List to collect all entries
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "COMPDAT" in stripped_line:
                start_collecting = True
                continue
            if start_collecting and stripped_line.startswith("/"):
                start_collecting = False
                continue
            if start_collecting and stripped_line:
                parts = stripped_line.split()
                well_name = parts[0].strip("'")
                if well_name in well_names:
                    data.append((well_name, parts[1], parts[2], parts[3], parts[4]))
    return data


def process_dataframe(name, producer_well_names, vectors):
    df = vectors[name]
    filtered_columns = [
        col
        for col in df.columns
        if any(well_name in col for well_name in producer_well_names)
    ]
    filtered_df = df[filtered_columns]
    start_row = find_first_numeric_row(filtered_df)
    if start_row is not None:
        numeric_df = filtered_df.iloc[start_row:]
        result_array = numeric_df.to_numpy()
    else:
        result_array = None
    Time = vectors["TIME"]
    start_row = find_first_numeric_row(Time)
    if start_row is not None:
        numeric_df = Time.iloc[start_row:]
        time_array = numeric_df.to_numpy()

    else:
        time_array = None
    return result_array, time_array


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def linear_interp(x, xp, fp):
    contiguous_xp = xp.contiguous()
    left_indices = torch.clamp(
        torch.searchsorted(contiguous_xp, x) - 1, 0, len(contiguous_xp) - 2
    )
    denominators = contiguous_xp[left_indices + 1] - contiguous_xp[left_indices]
    close_to_zero = denominators.abs() < 1e-10
    denominators[close_to_zero] = 1.0  # or any non-zero value to avoid NaN
    interpolated_value = (
        ((fp[left_indices + 1] - fp[left_indices]) / denominators)
        * (x - contiguous_xp[left_indices])
    ) + fp[left_indices]
    return interpolated_value


def replace_nan_with_zero(tensor):
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    invalid_mask = nan_mask | inf_mask
    valid_elements = tensor[~invalid_mask]  # Elements that are not NaN or Inf
    if valid_elements.numel() > 0:  # Ensure there are valid elements to calculate mean
        mean_value = valid_elements.mean()
    else:
        mean_value = torch.tensor(1e-6, device=tensor.device)
    return torch.where(invalid_mask, mean_value, tensor)


def interp_torch(cuda, reference_matrix1, reference_matrix2, tensor1):
    chunk_size = 1
    chunks = torch.chunk(tensor1, chunks=chunk_size, dim=0)
    processed_chunks = []
    for start_idx in range(chunk_size):
        interpolated_chunk = linear_interp(
            chunks[start_idx], reference_matrix1, reference_matrix2
        )
        processed_chunks.append(interpolated_chunk)
    torch.cuda.empty_cache()
    return processed_chunks


def get_model_hash(model):
    state_dict = model.state_dict()
    state_bytes = torch.save(state_dict, None, _use_new_zipfile_serialization=False)
    return hashlib.sha256(state_bytes).hexdigest()


def compute_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_true_mean) ** 2)
    RSS = np.sum((y_true - y_pred) ** 2)
    R2 = 1 - (RSS / TSS)
    L2_accuracy = 1 - np.sqrt(RSS) / np.sqrt(TSS)
    return R2, L2_accuracy


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


def dx1(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor"
    inpt = inpt.to(torch.float32)
    var = inpt[:, channel : channel + 1, :, :]
    if order == 1:
        ddx1D = torch.tensor(
            [
                -0.5,
                0.0,
                0.5,
            ],
            dtype=torch.float32,
            device=inpt.device,
        )
    elif order == 3:
        ddx1D = torch.tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ],
            dtype=torch.float32,
            device=inpt.device,
        )
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]
    return output.to(torch.float32)


def ddx1(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute second order numerical derivatives of input tensor"
    inpt = inpt.to(torch.float32)
    var = inpt[:, channel : channel + 1, :, :]
    if order == 1:
        ddx1D = torch.tensor(
            [
                1.0,
                -2.0,
                1.0,
            ],
            dtype=torch.float32,
            device=inpt.device,
        )
    elif order == 3:
        ddx1D = torch.tensor(
            [
                1.0 / 90.0,
                -3.0 / 20.0,
                3.0 / 2.0,
                -49.0 / 18.0,
                3.0 / 2.0,
                -3.0 / 20.0,
                1.0 / 90.0,
            ],
            dtype=torch.float32,
            device=inpt.device,
        )
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx**2) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]
    return output.to(torch.float32)


def to_absolute_path_and_create(
    *args: Union[str, Path],
) -> Union[Path, str, Tuple[Union[Path, str]]]:
    """Converts file path to absolute path based on the current working directory and creates the subfolders."""
    out = ()
    base = Path(os.getcwd())
    for path in args:
        p = Path(path)
        if p.is_absolute():
            ret = p
        else:
            ret = base / p
        ret.mkdir(parents=True, exist_ok=True)
        if isinstance(path, str):
            out = out + (str(ret),)
        else:
            out = out + (ret,)
    if len(args) == 1:
        out = out[0]
    return out


def predict_with_params(x, params):
    a, b, c, d = params
    x = x.clone()
    x = replace_with_mean(x)
    x = torch.clamp(x, 0, 1)
    interpolated_values = (a * x**3) + (b * x**2) + (c * x) + d
    # Ensure this function is properly defined or adjusted
    interpolated_values = replace_with_mean(interpolated_values)
    interpolated_values = torch.clamp(interpolated_values, 1e-6, 1)
    return interpolated_values


def Black_oil_peacemann(
    input_var,
    UO,
    BO,
    UW,
    BW,
    DZ,
    RE,
    device,
    max_inn_fcn,
    max_out_fcn,
    paramz,
    p_bub,
    p_atm,
    steppi,
    CFO,
    Relperm,
    SWI,
    SWR,
    SWOW,
    SWOG,
    params1_swow,
    params2_swow,
    params1_swog,
    params2_swog,
    N_pr,
    lenwels,
):
    in_var = input_var["X"].clone()
    out_var = input_var["Y"].clone()  # ✅ Clone to prevent in-place modification
    out_var = out_var.clamp(1e-6, 1)  # ✅ Now, this won't modify the original tensor
    in_var = in_var.clamp(1e-6, 1)
    skin = 0
    rwell = 200
    spit = torch.zeros(0, lenwels * N_pr, steppi).to(
        device
    )  # ✅ Initialize empty tensor
    N = in_var.shape[0]
    pwf_producer = 100
    for i in range(N):
        inn = in_var[i, :, :].T * max_inn_fcn
        outt = out_var[i, :, :].T * max_out_fcn
        oil_rate = outt[:, :N_pr]
        water_rate = outt[:, N_pr : 2 * N_pr]
        gas_rate = outt[:, 2 * N_pr : 3 * N_pr]
        permeability = inn[:, :N_pr]
        pressure = inn[:, N_pr : N_pr + 1]
        gas = inn[:, 2 * N_pr + 1 : 3 * N_pr + 1]
        water = inn[:, 3 * N_pr + 1 : 4 * N_pr + 1]
        # ✅ Avoid in-place operations
        gas = gas.clamp(1e-6, 1)
        water = water.clamp(1e-6, 1)
        # Compute relative permeability
        if Relperm == 1:
            one_minus_swi_swr = 1 - (SWI + SWR)
            soa = (1 - (water + gas) - SWR) / one_minus_swi_swr
            swa = (water - SWI) / one_minus_swi_swr
            sga = gas / one_minus_swi_swr
            soa = replace_with_mean(soa)
            swa = replace_with_mean(swa)
            sga = replace_with_mean(sga)
            KROW = predict_with_params(water, params1_swow)
            krw = predict_with_params(water, params2_swow)
            KROG = predict_with_params(gas, params1_swog)
            krg = predict_with_params(gas, params2_swog)
            kro = (KROW / (1 - swa)) * (KROG / (1 - sga)) * soa
        else:
            krw, kro, krg = StoneIIModel(paramz, device, gas, water)
        krw, kro, krg = StoneIIModel(paramz, device, gas, water)
        krw = replace_with_mean(krw)
        kro = replace_with_mean(kro)
        krg = replace_with_mean(krg)
        BO = calc_bo(p_bub, p_atm, CFO, pressure.mean())
        up = UO * BO
        down = 2 * torch.pi * permeability * kro * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        drawdown = pressure.mean() - pwf_producer
        qoil = torch.abs(-(drawdown * J))
        loss_oil = (qoil - oil_rate) / N
        up = UW * BW
        down = 2 * torch.pi * permeability * krw * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        drawdown = pressure.mean() - pwf_producer
        qwater = torch.abs(-(drawdown * J))
        loss_water = (qwater - water_rate) / N
        UG = calc_mu_g(pressure.mean())
        BG = calc_bg(p_bub, p_atm, pressure.mean())
        up = UG * BG
        down = 2 * torch.pi * permeability * krg * DZ
        right = torch.log(RE / rwell) + skin
        J = down / (up * right)
        drawdown = pressure.mean() - pwf_producer
        qgas = torch.abs(-(drawdown * J))
        loss_gas = (qgas - gas_rate) / N
        overall_loss = torch.cat((loss_oil, loss_water, loss_gas), dim=1).T
        spit = torch.cat((spit, overall_loss.unsqueeze(0)), dim=0)
    output_var = {"peacemanned": spit}
    return normalize_tensors_adjusted(output_var)


def pdeinp(input_var, neededM):
    in_var_water = input_var["water_sat"]
    in_var_oil = input_var["oil_sat"]
    in_var_gas = input_var["gas_sat"]
    actnum = (
        neededM["actnum"]
        .to(torch.float32)
        .repeat(in_var_water.shape[0], 1, 1, 1, 1)
        .clamp(min=1e-6)
    )
    waterd = in_var_water - torch.abs(actnum - (in_var_oil + in_var_gas))
    oild = in_var_oil - torch.abs(actnum - (in_var_water + in_var_gas))
    gasd = in_var_gas - torch.abs(actnum - (in_var_oil + in_var_water))
    output_var = {"oild": oild, "waterd": waterd, "gasd": gasd}

    return (
        torch.mean(normalize_tensors_adjusted(output_var)["waterd"])
        + torch.mean(normalize_tensors_adjusted(output_var)["oild"])
        + torch.mean(normalize_tensors_adjusted(output_var)["gasd"])
    )


def linear_interp5D(x, xp, fp):
    xp_contiguous = xp.contiguous()
    fp_contiguous = fp.contiguous()
    N, T, nz, nx, ny = x.shape
    interpolated_values = torch.zeros_like(x)
    for n in range(N):
        for t in range(T):
            x_flat = x[n, t].flatten()
            interpolated_flat = torch.zeros_like(x_flat)
            left_indices = torch.clamp(
                torch.searchsorted(xp_contiguous, x_flat) - 1, 0, len(xp_contiguous) - 2
            )
            denominators = xp_contiguous[left_indices + 1] - xp_contiguous[left_indices]
            close_to_zero = denominators.abs() < 1e-6
            denominators[close_to_zero] = 1  # Avoid division by zero
            interpolated_flat = (
                (
                    (fp_contiguous[left_indices + 1] - fp_contiguous[left_indices])
                    / denominators
                )
                * (x_flat - xp_contiguous[left_indices])
            ) + fp_contiguous[left_indices]
            interpolated_flat = torch.where(
                torch.isnan(interpolated_flat) | torch.isinf(interpolated_flat),
                torch.zeros_like(interpolated_flat),
                interpolated_flat,
            )
            interpolated_values[n, t] = interpolated_flat.view(nz, nx, ny)
    interpolated_values = replace_with_mean(interpolated_values)
    interpolated_values = torch.clamp(interpolated_values, 0, 1)
    return interpolated_values


def process_and_print(data_dict, dict_name):
    logger = setup_logging()
    for key in data_dict.keys():
        data_dict[key][np.isnan(data_dict[key])] = 1e-6
        # Convert infinity to a small value
        data_dict[key][np.isinf(data_dict[key])] = 1e-6
        data_dict[key] = clip_and_convert_to_float32(data_dict[key])
    for key, value in data_dict.items():
        logger.info(f"For key '{key}' in {dict_name}:")
        logger.info("\tContains inf: %s", np.isinf(value).any())
        logger.info("\tContains -inf: %s", np.isinf(-value).any())
        logger.info("\tContains NaN: %s", np.isnan(value).any())
        logger.info("\tSize = : %s", value.shape)


def normalize_tensors_adjusted2(tensor):
    tensor = tensor.to(torch.float32)
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if max_val - min_val > 0:
        tensor = (tensor - min_val) / (max_val - min_val)  # ✅ Out-of-place
        perturbation = torch.clamp(
            torch.normal(mean=0.1, std=0.01, size=tensor.size(), device=tensor.device),
            min=0.1,
        )
        tensor = tensor * 0.9 + perturbation  # ✅ Out-of-place
    else:
        perturbation = torch.clamp(
            torch.normal(mean=0.1, std=0.01, size=tensor.size(), device=tensor.device),
            min=0.1,
        )
        tensor = torch.zeros_like(tensor) + perturbation  # ✅ Out-of-place
    normalized_dict = tensor
    del min_val, max_val, perturbation  # Free memory of intermediate variables
    return normalized_dict
