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

import logging
import sys

import numpy as np
import numpy.linalg
import numpy.matlib

import torch
import torch.nn.functional as F

from forward.simulator import StoneIIModel, calc_bo, calc_bg, calc_mu_g
from data_extract.opm_extract_rates import normalize_tensors_adjusted


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output."""
    logger = logging.getLogger("forward problem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-entry

    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")

    # File handler — plain, no colors (keeps log file clean)
    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    # Console handler — colored
    class _ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[0;36m",  # cyan
            logging.INFO: "\033[0;32m",  # green
            logging.WARNING: "\033[1;33m",  # yellow
            logging.ERROR: "\033[0;31m",  # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"

        def format(self, record):
            msg = super().format(record)
            if sys.stdout.isatty():
                color = self.COLORS.get(record.levelno, "")
                return f"{color}{msg}{self.RESET}"
            return msg

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(
        _ColorFormatter(" %(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(c_handler)

    return logger


logger = setup_logging()


def replace_with_mean(tensor, name="tensor"):
    """Replace NaN and Inf values in a tensor with a perturbed mean.

    A warning is emitted whenever any non-finite values were actually replaced,
    so that silent data corruption is visible to the operator.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor that may contain NaN or Inf values.
    name : str, optional
        Human-readable label included in the warning message to identify
        which quantity was corrupted. Default ``"tensor"``.

    Returns
    -------
    torch.Tensor
        Tensor with NaN/Inf replaced by perturbed finite mean, clamped to 1e-6.
    """
    tensor = tensor.to(torch.float32)
    invalid_mask = torch.isnan(tensor) | torch.isinf(tensor)
    n_invalid = int(invalid_mask.sum().item()) if invalid_mask.any() else 0
    if n_invalid > 0:
        logger.warning(
            "replace_with_mean: replaced %d NaN/Inf values in %s "
            "(out of %d) with perturbed finite mean",
            n_invalid,
            name,
            tensor.numel(),
        )
    valid_elements = tensor[torch.isfinite(tensor)]
    if valid_elements.numel() > 0:  # Check if there are any valid elements
        mean_value = valid_elements.mean()  # Retains gradients
        perturbation = torch.normal(mean=0.0, std=0.01, size=(1,), device=tensor.device)
        perturbed_mean_value = mean_value + perturbation
    else:
        perturbed_mean_value = torch.tensor(
            1e-4, device=tensor.device, dtype=torch.float32, requires_grad=True
        )
    ouut = torch.where(invalid_mask, perturbed_mean_value, tensor)
    return torch.clamp(ouut, min=1e-6)  # Keeps gradients flowing


def dx(inpt, dx, channel, dim, order=1, padding="zeros"):
    """Compute first-order numerical derivatives of a 2-D input tensor.

    Parameters
    ----------
    inpt : torch.Tensor
        Input tensor of shape ``(N, C, H, W)``.
    dx : float
        Grid spacing used to scale the derivative.
    channel : int
        Channel index to differentiate.
    dim : int
        Spatial dimension along which to differentiate (0 = x, 1 = y).
    order : int, optional
        Finite-difference order: 1 (3-point) or 3 (7-point). Default is 1.
    padding : str, optional
        Padding strategy: ``"zeros"`` or ``"replication"``. Default is ``"zeros"``.

    Returns
    -------
    torch.Tensor
        First-order spatial derivative tensor of the selected channel.
    """
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
    """Compute second-order numerical derivatives of a 2-D input tensor.

    Parameters
    ----------
    inpt : torch.Tensor
        Input tensor of shape ``(N, C, H, W)``.
    dx : float
        Grid spacing used to scale the derivative.
    channel : int
        Channel index to differentiate.
    dim : int
        Spatial dimension along which to differentiate (0 = x, 1 = y).
    order : int, optional
        Finite-difference order: 1 (3-point) or 3 (7-point). Default is 1.
    padding : str, optional
        Padding strategy: ``"zeros"`` or ``"replication"``. Default is ``"zeros"``.

    Returns
    -------
    torch.Tensor
        Second-order spatial derivative tensor of the selected channel.
    """
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
    """Compute first-order spatial derivatives of a 5-D field along x, y, and z.

    Parameters
    ----------
    u : torch.Tensor
        Input field of shape ``(N, C, nz, H, W)``.
    dxf : float
        Uniform grid spacing in all three spatial directions.

    Returns
    -------
    tuple of torch.Tensor
        Three tensors ``(dudx, dudy, dudz)`` with the same shape as ``u``.
    """
    _batch_size, _channels, nz, _height, _width = u.shape
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
    """Compute normalised root-mean-square error between predictions and targets.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values array of any shape.
    targets : np.ndarray
        Ground-truth values array of any shape.

    Returns
    -------
    float
        RMSE divided by the number of target elements.
    """
    noww = predictions.reshape(-1, 1)
    measurment = targets.reshape(-1, 1)
    return (np.sum((noww - measurment) ** 2)) ** (0.5) / (measurment.shape[0])


def compute_second_differential(u, dxf):
    """Compute second-order spatial derivatives of a 5-D field along x, y, and z.

    Parameters
    ----------
    u : torch.Tensor
        Input field of shape ``(N, C, nz, H, W)``.
    dxf : float
        Uniform grid spacing in all three spatial directions.

    Returns
    -------
    tuple of torch.Tensor
        Three tensors ``(d2udx2, d2udy2, d2udz2)`` with the same shape as ``u``.
    """
    _batch_size, _channels, nz, _height, _width = u.shape
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


def kmeans(data, num_clusters=2, num_iters=10):
    """Run a simple k-means clustering on a tensor.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of any shape; flattened internally.
    num_clusters : int, optional
        Number of cluster centroids. Default is 2.
    num_iters : int, optional
        Maximum number of Lloyd iterations. Default is 10.

    Returns
    -------
    labels : torch.Tensor
        Cluster assignment tensor with the same shape as ``data``.
    centroids : torch.Tensor
        Final centroid values of shape ``(num_clusters, 1)``.
    """
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
    """Compute a binary boundary mask between two cluster regions in 3-D.

    Parameters
    ----------
    cluster_labels : torch.Tensor
        3-D integer tensor of cluster assignments.
    cluster_value_1 : int, optional
        Label value for the first cluster. Default is 0.
    cluster_value_2 : int, optional
        Label value for the second cluster. Default is 1.

    Returns
    -------
    torch.Tensor
        Float tensor of ones where the two clusters share a boundary, zeros elsewhere.
    """
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
    """Compute the Hamming distance between two binary masks.

    Parameters
    ----------
    mask1 : torch.Tensor
        First binary mask tensor of any shape.
    mask2 : torch.Tensor
        Second binary mask tensor of the same shape as ``mask1``.

    Returns
    -------
    int
        Number of positions where the two masks differ.
    """
    mask1_flat = mask1.view(-1)
    mask2_flat = mask2.view(-1)
    return torch.sum(mask1_flat != mask2_flat).item()


def process_tensor_sat(tensor, truee, num_clusters=2):
    """Compute mean Hamming distance of saturation front boundaries over batch and time.

    Parameters
    ----------
    tensor : torch.Tensor
        Predicted saturation field of shape ``(B, T, nz, nx, ny)``.
    truee : torch.Tensor
        Ground-truth saturation field of the same shape as ``tensor``.
    num_clusters : int, optional
        Number of k-means clusters used for boundary detection. Default is 2.

    Returns
    -------
    float
        Mean Hamming distance across all batch and time indices.
    """
    B, T, _nz, _nx, _ny = tensor.shape
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
    return total_hamming_distance / num_elements


def process_tensor(tensor):
    """Replace NaN/Inf values with 1e-6 and cast a tensor to float32.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of any shape and dtype.

    Returns
    -------
    torch.Tensor
        Cleaned float32 tensor with no NaN or Inf values.
    """
    tensor = torch.where(
        torch.isnan(tensor) | torch.isinf(tensor),
        torch.tensor(1e-6, dtype=torch.float32),
        tensor,
    )
    return tensor.to(dtype=torch.float32)


class Labelledset:
    """Labelledset."""

    def __init__(self, data, device):
        """Move a dict of NumPy arrays to a torch device as float32 tensors.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Mapping from variable name to NumPy array.
        device : torch.device
            Target device for all tensors.
        """
        self.device = device
        self.data = {
            key: torch.from_numpy(data[key]).to(self.device, torch.float32)
            for key in data
        }

    def __getitem__(self, index):
        """Return a slice of all tensors at *index*.

        Parameters
        ----------
        index : int or slice
            Batch index or slice applied to every tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Sliced tensors keyed by variable name.
        """
        return {key: self.data[key][index] for key in self.data}

    def __len__(self):
        """Return the number of samples (length of the first tensor axis).

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        first_key = next(iter(self.data))
        return len(self.data[first_key])


def _l2_relative_error(pred_var, true_var):
    """Compute relative L2 error normalised by the variance of the true field.

    Parameters
    ----------
    pred_var : torch.Tensor
        Predicted field tensor of any shape.
    true_var : torch.Tensor
        Ground-truth field tensor of the same shape as ``pred_var``.

    Returns
    -------
    torch.Tensor
        Scalar relative L2 error value.
    """
    epsilon = 1e-8  # Small constant to avoid division by zero
    return torch.sqrt(
        torch.sum(torch.square(true_var - pred_var)) / (torch.var(true_var) + epsilon)
    )


def loss_func(x, y, types, lambda_weighting, p=2.0):
    """Weighted L-p loss matching PhysicsNeMo Sym's PointwiseLossNorm.

    Sym computes per term: ``w * sum(|pred - target|^p)`` and aggregates
    across keys via a plain summation (Sum aggregator). No batch-averaging
    is performed at the loss level — the network sees the raw summed error,
    which is why Sym's loss weights look small for high-resolution grids.

    Parameters
    ----------
    x : torch.Tensor
        Predicted tensor.
    y : torch.Tensor
        Target tensor with the same shape as ``x``.
    types : str
        Retained for API compatibility. Ignored — Sym applies the same form
        to elliptical (pressure) and hyperbolic (saturation) systems; the
        differentiation lives in the per-key weights, not the loss form.
    lambda_weighting : float
        Per-term weight from cfg.loss.weights.
    p : float, optional
        Lp order. Default 2.0. Sym defaults to 2.

    Returns
    -------
    torch.Tensor
        Scalar weighted L-p sum for this term.
    """
    return lambda_weighting * (x - y).abs().pow(p).sum()


# ============================================================
# Sobolev / relative H^1 loss for sharp-feature fields
# ============================================================


def extra_loss(ytrue, y_hat, weight=1.0):
    """
    Relative H^1 loss: scale-invariant value error + scale-invariant
    spatial gradient error.

    Per-sample relative norms make the loss equally meaningful across fields
    of different magnitudes (pressure vs saturations) without needing per-field
    weight tuning. The gradient term focuses learning on cells where the
    target actually has structure (fronts, well regions).

    Args
    ----
    ytrue, y_hat : tensors of shape (B, T, nz, nx, ny)
    weight : scalar lambda (matches your existing cfg.loss.weights.* convention)

    Returns
    -------
    Scalar loss tensor.
    """
    error = ytrue - y_hat

    # Value term: ||error|| / ||ytrue|| per sample, then mean
    term1 = torch.linalg.vector_norm(error, ord=2, dim=(1, 2, 3, 4)) / (
        torch.linalg.vector_norm(ytrue, ord=2, dim=(1, 2, 3, 4)) + 1e-6
    )

    # Spatial finite differences along nx (dim=3), ny (dim=4), nz (dim=2)
    dy_dx = ytrue[:, :, :, 1:, :] - ytrue[:, :, :, :-1, :]
    dy_hat_dx = y_hat[:, :, :, 1:, :] - y_hat[:, :, :, :-1, :]
    dy_dy = ytrue[:, :, :, :, 1:] - ytrue[:, :, :, :, :-1]
    dy_hat_dy = y_hat[:, :, :, :, 1:] - y_hat[:, :, :, :, :-1]
    dy_dz = ytrue[:, :, 1:, :, :] - ytrue[:, :, :-1, :, :]
    dy_hat_dz = y_hat[:, :, 1:, :, :] - y_hat[:, :, :-1, :, :]

    def rel(num, den):
        return torch.linalg.vector_norm(num, ord=2, dim=(1, 2, 3, 4)) / (
            torch.linalg.vector_norm(den, ord=2, dim=(1, 2, 3, 4)) + 1e-6
        )

    term2 = (
        rel(dy_dx - dy_hat_dx, dy_dx)
        + rel(dy_dy - dy_hat_dy, dy_dy)
        + rel(dy_dz - dy_hat_dz, dy_dz)
    ) / 3

    return weight * (term1 + term2).mean()


def combined_loss(ytrue, y_hat, weight, n_cells, alpha=1.0, beta=1.0, p=2.0):
    """
    Combined loss = relative H^1 (extra_loss) + mean L^p (rescaled loss_func).

    Both terms are normalized to O(1) magnitude so they're directly comparable
    and `weight` retains its usual meaning across grid resolutions.

    Args
    ----
    ytrue, y_hat : tensors of shape (B, T, nz, nx, ny)
    weight       : outer per-head lambda (cfg.loss.weights.*)
    n_cells      : total number of spatial cells (nz * nx * ny)
    alpha        : weight on the relative H^1 term — drives structure /
                   fronts / well singularities
    beta         : weight on the mean L^p term — drives absolute accuracy
    p            : L^p exponent. Default 2.

    Returns
    -------
    Scalar loss tensor.
    """
    sobolev_term = extra_loss(ytrue, y_hat, weight=1.0)
    lp_term = (y_hat - ytrue).abs().pow(p).sum() / n_cells
    return weight * (alpha * sobolev_term + beta * lp_term)


def loss_func_physics(x, lambda_weighting):
    """Compute batch-averaged absolute physics residual loss.

    Parameters
    ----------
    x : torch.Tensor
        Physics residual tensor with batch dimension as the first axis.
    lambda_weighting : float
        Scalar weight applied to the residual.

    Returns
    -------
    torch.Tensor
        Scalar batch-averaged weighted absolute residual.
    """
    loss = (lambda_weighting * torch.abs(torch.nan_to_num(x, nan=0.0))).sum()
    return loss


def compute_gradient_3d(inpt, dx, dim, order=1, padding="zeros"):
    """Compute first-order numerical derivatives of a 5-D input tensor for 3-D data.

    Parameters
    ----------
    inpt : torch.Tensor
        Input tensor of shape ``(N, C, nz, H, W)``; cast to float32 internally.
    dx : float
        Grid spacing used to scale the derivative.
    dim : int
        Spatial dimension along which to differentiate (0 = z, 1 = y, 2 = x).
    order : int, optional
        Finite-difference order: 1 (3-point) or 3 (7-point). Default is 1.
    padding : str, optional
        Padding strategy: ``"zeros"`` or ``"replication"``. Default is ``"zeros"``.

    Returns
    -------
    torch.Tensor
        First-order spatial derivative tensor of shape ``(N, C, nz, H, W)`` in float32.
    """
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
    return torch.cat(outputs, dim=1).to(torch.float32)


def compute_second_order_gradient_3d(inpt, dx, dim, padding="zeros"):
    """Compute second-order numerical derivatives of a 5-D input tensor for 3-D data.

    Parameters
    ----------
    inpt : torch.Tensor
        Input tensor of shape ``(N, C, nz, H, W)``; cast to float32 internally.
    dx : float
        Grid spacing used to scale the second derivative.
    dim : int
        Spatial dimension along which to differentiate (0 = z, 1 = y, 2 = x).
    padding : str, optional
        Padding strategy: ``"zeros"`` or ``"replication"``. Default is ``"zeros"``.

    Returns
    -------
    torch.Tensor
        Second-order spatial derivative tensor of shape ``(N, C, nz, H, W)`` in float32.
    """
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
    return torch.cat(outputs, dim=1).to(torch.float32)


# convert_back is imported from utils.array_utils above.


def replace_nans_and_infs(tensor, value=0.0, name="tensor"):
    """Replace NaN and Inf entries of a tensor in-place with a constant value.

    A warning is emitted whenever any non-finite values were actually replaced,
    so that silent data corruption is visible to the operator.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor modified in-place.
    value : float, optional
        Replacement scalar value. Default is 0.0.
    name : str, optional
        Human-readable label included in the warning message to identify
        which quantity was corrupted. Default ``"tensor"``.

    Returns
    -------
    torch.Tensor
        The same tensor with NaN/Inf entries replaced.
    """
    invalid_mask = torch.isnan(tensor) | torch.isinf(tensor)
    n_invalid = int(invalid_mask.sum().item()) if invalid_mask.any() else 0
    if n_invalid > 0:
        logger.warning(
            "replace_nans_and_infs: replaced %d NaN/Inf values in %s "
            "(out of %d) with %s",
            n_invalid,
            name,
            tensor.numel(),
            value,
        )
    tensor[invalid_mask] = value
    return tensor


def scale_tensor_abs(tensor, target_min, target_max):
    """Normalise a NumPy array by its maximum absolute value.

    Parameters
    ----------
    tensor : np.ndarray
        Input array; NaN and Inf entries are zeroed in-place.
    target_min : float
        Intended target minimum (currently unused).
    target_max : float
        Intended target maximum (currently unused).

    Returns
    -------
    min_val : float
        Minimum value of the cleaned array.
    max_val : float
        Maximum value of the cleaned array.
    rescaled_tensor : np.ndarray
        Array divided by its maximum value.
    """
    if np.any(np.isnan(tensor) | np.isinf(tensor)):
        logger.warning("NaN/Inf values detected in tensor; replacing with 0.")
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    rescaled_tensor = tensor / max_val
    return min_val, max_val, rescaled_tensor


def scale_tensor_abs_pressure(tensor, max_val):
    """Normalise a pressure array by a supplied maximum value.

    Parameters
    ----------
    tensor : np.ndarray
        Pressure array; NaN and Inf entries are zeroed in-place.
    max_val : float
        Maximum value used for normalisation.

    Returns
    -------
    min_val : float
        Minimum value of the cleaned array.
    max_val : float
        The supplied maximum value echoed back.
    rescaled_tensor : np.ndarray
        Array divided by ``max_val``.
    """
    if np.any(np.isnan(tensor) | np.isinf(tensor)):
        logger.warning("NaN/Inf values detected in tensor; replacing with 0.")
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    rescaled_tensor = tensor / max_val
    return np.min(tensor), max_val, rescaled_tensor


def scale_tensor_absS(tensor, lenwels, N_pr):
    """Normalise per-well segments of a saturation array by their respective maxima.

    Parameters
    ----------
    tensor : np.ndarray
        Saturation array; NaN and Inf entries are zeroed in-place.
    lenwels : int
        Number of wells; controls how the array is sliced along axis 2.
    N_pr : int
        Number of production time steps per well segment.

    Returns
    -------
    get_it2 : np.ndarray
        Concatenated normalised segments along axis 2.
    Cmax : list of float
        Per-segment maximum values.
    Cmin : list of float
        Per-segment minimum values.
    """
    if np.any(np.isnan(tensor) | np.isinf(tensor)):
        logger.warning("NaN/Inf values detected in tensor; replacing with 0.")
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
    """Normalise injection-rate segments of an array by their respective maxima.

    Parameters
    ----------
    tensor : np.ndarray
        Injection array; NaN and Inf entries are zeroed in-place.
    N_pr : int
        Number of time steps per primary segment.

    Returns
    -------
    get_it2 : np.ndarray
        Concatenated normalised segments along axis 2.
    Cmax : np.ndarray
        Row vector of shape ``(1, 6)`` holding per-segment maxima.
    Cmin : np.ndarray
        Row vector of shape ``(1, 6)`` holding per-segment minima.
    """
    if np.any(np.isnan(tensor) | np.isinf(tensor)):
        logger.warning("NaN/Inf values detected in tensor; replacing with 0.")
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
    """Replace NaN, Inf, and float32-overflow values in an array with a placeholder.

    Parameters
    ----------
    arr : np.ndarray
        Input array modified in-place.
    placeholder : float, optional
        Scalar replacement for invalid entries. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array with all invalid entries replaced by ``placeholder``.
    """
    threshold = np.finfo(np.float32).max
    invalid_indices = (np.isnan(arr)) | (np.isinf(arr)) | (np.abs(arr) > threshold)
    arr[invalid_indices] = placeholder
    return arr


def clean_dict_arrays(data_dict):
    """Apply ``replace_large_and_invalid_values`` to every array in a dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping string keys to ``np.ndarray`` values.

    Returns
    -------
    dict
        The same dictionary with all arrays cleaned in-place.
    """
    for key in data_dict:
        data_dict[key] = replace_large_and_invalid_values(data_dict[key])
    return data_dict


def clip_and_convert_to_float32(array):
    """Clip an array to float32 representable range and cast it to float32.

    Parameters
    ----------
    array : np.ndarray
        Input array of any dtype.

    Returns
    -------
    np.ndarray
        Float32 array clipped to ``[np.finfo(np.float32).min, np.finfo(np.float32).max]``.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def clip_and_convert_to_float3(array):
    """Clip an array to float32 range and return it as float32 (alias variant).

    Parameters
    ----------
    array : np.ndarray
        Input array of any dtype.

    Returns
    -------
    np.ndarray
        Float32 array clipped to the float32 representable range.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def Make_correct(array):
    """Reorder a 5-D array from ``(N, C, nz, H, W)`` to ``(N, C, H, W, nz)`` layout.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape ``(N, C, nz, H, W)``.

    Returns
    -------
    np.ndarray
        Reordered array of shape ``(N, C, H, W, nz)``.
    """
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
    """Split a matrix into equal parts along axis 0.

    Parameters
    ----------
    matrix : np.ndarray
        Input array to split along the first axis.
    sizee : int
        Number of equal sections to split into.

    Returns
    -------
    list of np.ndarray
        List of sub-arrays resulting from the split.
    """
    return np.split(matrix, sizee, axis=0)


def normalize_tensors_adjusted2(tensor):
    """Min-max normalise a tensor to roughly ``[0.1, 1]`` with additive perturbation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of any shape.

    Returns
    -------
    torch.Tensor
        Normalised tensor with a small Gaussian perturbation added.
    """
    tensor = tensor.to(torch.float32)
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if max_val - min_val > 0:
        tensor = (tensor - min_val) / (max_val - min_val)  # ✅ Out-of-place
        perturbation = torch.clamp(
            torch.normal(mean=0.1, std=0.01, size=tensor.size(), device=tensor.device),
            min=0.1,
        )
        tensor = tensor * 0.9 + perturbation
    else:
        perturbation = torch.clamp(
            torch.normal(mean=0.1, std=0.01, size=tensor.size(), device=tensor.device),
            min=0.1,
        )
        tensor = torch.zeros_like(tensor) + perturbation  # ✅ Out-of-place
    return tensor


def predict_with_params(x, params):
    """Predict with params."""
    a, b, c, d = params
    x = x.clone()
    x = replace_with_mean(x)
    x = torch.clamp(x, 0, 1)
    interpolated_values = (a * x**3) + (b * x**2) + (c * x) + d
    # Ensure this function is properly defined or adjusted
    interpolated_values = replace_with_mean(interpolated_values)
    return torch.clamp(interpolated_values, 1e-6, 1)


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
    """Black oil peacemann."""
    in_var = input_var["X"].clone()
    out_var = input_var["Y"].clone()
    out_var = out_var.clamp(1e-6, 1)
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
    """Pdeinp."""
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
