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
Evaluation metrics for reservoir simulation models.

This module provides standard metrics for evaluating pressure and saturation predictions.
Imports available metrics from PhysicsNemo and provides additional domain-specific metrics.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance

# ============================================================================
# PhysicsNemo Imports (official implementations)
# ============================================================================
from physicsnemo.metrics.general.mse import mse, rmse
from physicsnemo.metrics.general.reduction import WeightedMean, WeightedVariance

# Re-export PhysicsNemo metrics for convenience
__all__ = [
    # PhysicsNemo imports
    "mse",
    "rmse",
    "Mean",
    "Variance",
    "WeightedMean",
    "WeightedVariance",
    # NumPy-based metrics
    "mean_relative_error",
    "mean_plume_error",
    "mean_absolute_error",
    "max_absolute_error",
    "compute_r2_score",
    "compute_relative_l2_error",
    "compute_relative_l1_error",
    "normalized_mse",
    "peak_signal_to_noise_ratio",
    # PyTorch-based metrics
    "mse_torch",
    "rmse_torch",
    "mae_torch",
    "relative_l2_torch",
    "relative_l1_torch",
    "r2_score_torch",
    "max_error_torch",
    "psnr_torch",
]


# ============================================================================
# NumPy-based Metrics (for evaluation/post-processing)
# ============================================================================


def mean_relative_error(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Mean Relative Error (MRE) - normalized by range.
    Appropriate for pressure predictions.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        eps: Small value to avoid division by zero

    Returns:
        float: MRE value between 0 and 1
    """
    data_range = y_true.max() - y_true.min()
    return float(np.mean(np.abs(y_pred - y_true)) / (data_range + eps))


def mean_plume_error(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Mean Plume Error (MPE) - error only where both are non-zero.
    Appropriate for saturation predictions where we care about
    accuracy within the CO2 plume region.

    Args:
        y_pred: Predicted saturation values
        y_true: Ground truth saturation values
        threshold: Threshold for considering values as "non-zero"

    Returns:
        float: MPE value (absolute error in plume region)
    """
    mask = (np.abs(y_pred) > threshold) & (np.abs(y_true) > threshold)
    y_pred_masked = y_pred[mask]
    y_true_masked = y_true[mask]

    if len(y_pred_masked) == 0:
        return 0.0

    return float(np.mean(np.abs(y_pred_masked - y_true_masked)))


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).

    Args:
        y_pred: Predicted values
        y_true: Ground truth values

    Returns:
        float: MAE value
    """
    return float(np.mean(np.abs(y_pred - y_true)))


def max_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Maximum Absolute Error.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values

    Returns:
        float: Maximum absolute error value
    """
    return float(np.max(np.abs(y_pred - y_true)))


def compute_r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) score.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values

    Returns:
        float: R² score (1.0 is perfect, negative means worse than mean)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - (ss_res / ss_tot))


def compute_relative_l2_error(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute relative L2 error.

    Args:
        pred: Predicted values
        target: Ground truth values
        eps: Small value to avoid division by zero

    Returns:
        float: Relative L2 error
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    diff_norm = np.linalg.norm(pred_flat - target_flat, ord=2)
    target_norm = np.linalg.norm(target_flat, ord=2)
    return float(diff_norm / (target_norm + eps))


def compute_relative_l1_error(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute relative L1 error.

    Args:
        pred: Predicted values
        target: Ground truth values
        eps: Small value to avoid division by zero

    Returns:
        float: Relative L1 error
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    diff_norm = np.linalg.norm(pred_flat - target_flat, ord=1)
    target_norm = np.linalg.norm(target_flat, ord=1)
    return float(diff_norm / (target_norm + eps))


def normalized_mse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    normalize_by: str = "variance",
    eps: float = 1e-8,
) -> float:
    """
    Normalized Mean Squared Error.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        normalize_by: 'variance' or 'range'
        eps: Small value to avoid division by zero

    Returns:
        float: Normalized MSE value
    """
    mse_val = np.mean((y_pred - y_true) ** 2)

    if normalize_by == "variance":
        normalizer = np.var(y_true) + eps
    elif normalize_by == "range":
        normalizer = (y_true.max() - y_true.min()) ** 2 + eps
    else:
        raise ValueError(
            f"normalize_by must be 'variance' or 'range', got {normalize_by}"
        )

    return float(mse_val / normalizer)


def peak_signal_to_noise_ratio(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    data_range: Optional[float] = None,
    eps: float = 1e-8,
) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR).

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        data_range: Dynamic range of the data (max - min). If None, computed from y_true.
        eps: Small value to avoid log(0)

    Returns:
        float: PSNR in dB
    """
    if data_range is None:
        data_range = y_true.max() - y_true.min()

    mse_val = np.mean((y_pred - y_true) ** 2)
    if mse_val < eps:
        return float("inf")

    return float(20 * np.log10(data_range / (np.sqrt(mse_val) + eps)))


# ============================================================================
# PyTorch-based Metrics (for use during training)
# ============================================================================


def mse_torch(pred: Tensor, target: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Mean Squared Error (PyTorch).
    Wrapper around PhysicsNemo's mse for consistent API.

    Args:
        pred: Predicted tensor
        target: Target tensor
        dim: Reduction dimension (None for full reduction)

    Returns:
        MSE value(s)
    """
    return mse(pred, target, dim=dim)


def rmse_torch(pred: Tensor, target: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Root Mean Squared Error (PyTorch).
    Wrapper around PhysicsNemo's rmse for consistent API.

    Args:
        pred: Predicted tensor
        target: Target tensor
        dim: Reduction dimension (None for full reduction)

    Returns:
        RMSE value(s)
    """
    return rmse(pred, target, dim=dim)


def mae_torch(pred: Tensor, target: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Mean Absolute Error (PyTorch).

    Args:
        pred: Predicted tensor
        target: Target tensor
        dim: Reduction dimension (None for full reduction)

    Returns:
        MAE value(s)
    """
    return torch.mean(torch.abs(pred - target), dim=dim)


def relative_l2_torch(
    pred: Tensor,
    target: Tensor,
    dim: Optional[int] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Relative L2 Error (PyTorch).

    Args:
        pred: Predicted tensor
        target: Target tensor
        dim: Dimension(s) for computing norms. If None, computes over flattened tensors.
        eps: Small value to avoid division by zero

    Returns:
        Relative L2 error
    """
    if dim is None:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        diff_norm = torch.norm(pred_flat - target_flat, p=2)
        target_norm = torch.norm(target_flat, p=2)
    else:
        diff_norm = torch.norm(pred - target, p=2, dim=dim)
        target_norm = torch.norm(target, p=2, dim=dim)

    return diff_norm / (target_norm + eps)


def relative_l1_torch(
    pred: Tensor,
    target: Tensor,
    dim: Optional[int] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Relative L1 Error (PyTorch).

    Args:
        pred: Predicted tensor
        target: Target tensor
        dim: Dimension(s) for computing norms. If None, computes over flattened tensors.
        eps: Small value to avoid division by zero

    Returns:
        Relative L1 error
    """
    if dim is None:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        diff_norm = torch.norm(pred_flat - target_flat, p=1)
        target_norm = torch.norm(target_flat, p=1)
    else:
        diff_norm = torch.norm(pred - target, p=1, dim=dim)
        target_norm = torch.norm(target, p=1, dim=dim)

    return diff_norm / (target_norm + eps)


def r2_score_torch(pred: Tensor, target: Tensor) -> Tensor:
    """
    R² Score (coefficient of determination) in PyTorch.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        R² score
    """
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)

    if ss_tot == 0:
        return torch.tensor(1.0 if ss_res == 0 else 0.0, device=pred.device)

    return 1 - (ss_res / ss_tot)


def max_error_torch(pred: Tensor, target: Tensor) -> Tensor:
    """
    Maximum Absolute Error (PyTorch).

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        Maximum absolute error
    """
    return torch.max(torch.abs(pred - target))


def psnr_torch(
    pred: Tensor,
    target: Tensor,
    data_range: Optional[float] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Peak Signal-to-Noise Ratio (PyTorch).

    Args:
        pred: Predicted tensor
        target: Target tensor
        data_range: Dynamic range of data. If None, computed from target.
        eps: Small value to avoid log(0)

    Returns:
        PSNR in dB
    """
    if data_range is None:
        data_range = target.max() - target.min()

    mse_val = torch.mean((pred - target) ** 2)

    if mse_val < eps:
        return torch.tensor(float("inf"), device=pred.device)

    return 20 * torch.log10(data_range / (torch.sqrt(mse_val) + eps))
