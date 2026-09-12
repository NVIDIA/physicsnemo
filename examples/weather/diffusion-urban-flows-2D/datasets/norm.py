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

"""Normalization and denormalization utilities for flow field data."""

import numpy as np


def normalize(x, u_max, u_min, v_max, v_min):
    """
    Normalizes the input tensor `x` with shape (num,2, nx, ny) by using min
    and max values from train dataset to range [-1, 1].

    Parameters:
    x (np.ndarray): numpy array with shape (num, 2, h, w).
    u_max (float): Maximum value for the first channel.
    u_min (float): Minimum value for the first channel.
    v_max (float): Maximum value for the second channel.
    v_min (float): Minimum value for the second channel.

    Returns:
    np.ndarray: normalized tensor with the same shape as `x`, in the range [-1 to 1].
    """
    x = np.clip(x, a_min=-1, a_max=1)

    eps = 1e-9
    center = np.array([u_min, v_min]).reshape((2, 1, 1))
    scale = np.array([u_max - u_min, v_max - v_min]).reshape((2, 1, 1))
    x_scaled = (x - center) / (scale + eps)

    return (2 * x_scaled) - 1


def renormalize(x_norm, u_max, u_min, v_max, v_min):
    """
    Renormalizes the input tensor `x_norm` with shape (num,2, h, w) from the range [-1, 1]
    back to the original range defined by u_max, u_min, v_max, and v_min.

    Parameters:
    x_norm (np.ndarray): Normalized tensor with shape (num,2, h, w) in the range [-1, 1].
    u_max (float): Maximum value for the first channel.
    u_min (float): Minimum value for the first channel.
    v_max (float): Maximum value for the second channel.
    v_min (float): Minimum value for the second channel.

    Returns:
    np.ndarray: Renormalized tensor with the same shape as `x_norm`, in the original range.
    """

    eps = 1e-9  # Small epsilon to avoid division by zero
    center = np.array([u_min, v_min]).reshape((2, 1, 1))
    scale = np.array([u_max - u_min, v_max - v_min]).reshape((2, 1, 1))

    # Scale back to [0, 1] range
    x_rescaled = (x_norm + 1) / 2

    # Shift and scale back to original range
    x_renormalized = x_rescaled * (scale + eps) + center

    return x_renormalized
