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
Evaluation metrics for CO2 sequestration models.

This module provides standard metrics for evaluating pressure and saturation predictions.
"""

import numpy as np


def mean_relative_error(y_pred, y_true):
    """
    Mean Relative Error (MRE) - normalized by range.

    Appropriate for pressure predictions.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values

    Returns:
        float: MRE value between 0 and 1
    """
    return np.mean(np.abs((y_pred - y_true)) / (y_true.max() - y_true.min()))


def mean_plume_error(y_pred, y_true):
    """
    Mean Plume Error (MPE) - only where both are non-zero.

    Appropriate for saturation predictions where we care about
    accuracy within the CO2 plume region.

    Args:
        y_pred: Predicted saturation values
        y_true: Ground truth saturation values

    Returns:
        float: MPE value (absolute error in plume region)
    """
    mask = (y_pred != 0) & (y_true != 0)
    y_pred_masked = y_pred[mask]
    y_true_masked = y_true[mask]

    if len(y_pred_masked) == 0:
        return 0.0

    return np.mean(np.abs(y_pred_masked - y_true_masked))


def mean_absolute_error(y_pred, y_true):
    """
    Mean Absolute Error (MAE).

    Args:
        y_pred: Predicted values
        y_true: Ground truth values

    Returns:
        float: MAE value
    """
    return np.mean(np.abs((y_pred - y_true)))


def compute_r2_score(y_pred, y_true):
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
    return 1 - (ss_res / ss_tot)


def compute_relative_l2_error(pred, target):
    """
    Compute relative L2 error.

    Args:
        pred: Predicted values
        target: Ground truth values

    Returns:
        float: Relative L2 error
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    diff_norm = np.linalg.norm(pred_flat - target_flat, ord=2)
    target_norm = np.linalg.norm(target_flat, ord=2)
    return diff_norm / target_norm
