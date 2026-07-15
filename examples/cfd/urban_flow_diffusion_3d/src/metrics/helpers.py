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

import numpy as np
from typing import Optional
from src.utils import *


def extract_data_2d(
    U: np.ndarray,
    x_pixel: int,
    y_pixel: Optional[int],
    z_pixel: Optional[int],
    data: str,
) -> np.ndarray:
    """
    Extract data from a 2D input array based on the specified data type.

    Args:
        U (np.ndarray): Input 2D array.
        x_pixel (int): X coordinate in pixels.
        y_pixel (Optional[int]): Y coordinate in pixels (can be None).
        z_pixel (Optional[int]): Z coordinate in pixels (can be None).
        data (str): Type of data to extract ('point', 'line-x', 'line-y', 'plane').

    Returns:
        np.ndarray: Extracted data.
    """
    if z_pixel is None:  # xy plane
        if data == "point":
            return U[:, x_pixel, y_pixel]
        elif data == "line-x":
            return U[:, :, y_pixel]
        elif data == "line-y":
            return U[:, x_pixel, :]
        elif data == "plane":
            return U[:, :, :]
    elif y_pixel is None:  # xz plane
        if data == "point":
            return U[:, x_pixel, z_pixel]
        elif data == "line-x":
            return U[:, :, z_pixel]
        elif data == "line-z":
            return U[:, x_pixel, :]
        elif data == "plane":
            return U[:, :, :]


def extract_data_3d(
    U: np.ndarray, x_pixel: int, y_pixel: int, z_pixel: int, data: str
) -> np.ndarray:
    """
    Extract data from a 3D input array based on the specified data type.

    Args:
        U (np.ndarray): Input 3D array.
        x_pixel (int): X coordinate in pixels.
        y_pixel (int): Y coordinate in pixels.
        z_pixel (int): Z coordinate in pixels.
        data (str): Type of data to extract ('point', 'line-x', 'line-y', 'xy_plane', 'xz_plane').

    Returns:
        np.ndarray: Extracted data.
    """
    if data == "point":
        return U[:, x_pixel, y_pixel, z_pixel]
    elif data == "line-x":
        return U[:, :, y_pixel, z_pixel]
    elif data == "line-y":
        return U[:, x_pixel, :, z_pixel]
    elif data == "xy_plane":
        return U[:, :, :, z_pixel]
    elif data == "xz_plane":
        return U[:, :, y_pixel, :]


def get_data_for_stats(
    U: np.ndarray,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
    input_data_type: str = "2D",
    data: Optional[str] = None,
    mean_over_time: bool = False,
    ds_ratio: Optional[float] = None,
) -> np.ndarray:
    """
    Extract specific data from a multidimensional array for statistical analysis.

    Args:
        U (np.ndarray): Input array.
        x (Optional[float]): X coordinate in units.
        y (Optional[float]): Y coordinate in units.
        z (Optional[float]): Z coordinate in units.
        input_data_type (str): Type of input data ('2D' or '3D').
        data (Optional[str]): Type of data to extract ('point', 'line-x', 'line-y', 'line-z', 'xy_plane', 'xz_plane', 'plane').
        mean_over_time (bool): Whether to average the extracted data over time.
        ds_ratio (Optional[float]): Downsampling ratio.

    Returns:
        np.ndarray: Extracted (and possibly averaged) data.
    """
    # Convert units to pixel values
    x_pixel, y_pixel, z_pixel = convert_unit2pixel(
        x=x, y=y, z=z, ds_ratio=ds_ratio, flipped=False
    )

    # Extract data based on the input data type
    if input_data_type == "2D":
        extracted_data = extract_data_2d(U, x_pixel, y_pixel, z_pixel, data)
    elif input_data_type == "3D":
        extracted_data = extract_data_3d(U, x_pixel, y_pixel, z_pixel, data)
    else:
        raise ValueError(f"Invalid input_data_type: {input_data_type}")

    # Optionally average the extracted data over time
    if mean_over_time:
        extracted_data = np.mean(extracted_data, axis=0)

    return extracted_data
