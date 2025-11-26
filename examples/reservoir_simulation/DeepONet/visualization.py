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
Visualization utilities for CO2 sequestration predictions.

This module provides common plotting functions and grid setup
for visualizing model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt


def setup_plotting_grid():
    """
    Setup the spatial grid for plotting reservoir data.

    Returns:
        tuple: (X, Y, dx) meshgrid arrays for plotting
    """
    dx = np.cumsum(3.5938 * np.power(1.035012, range(200))) + 0.1
    X, Y = np.meshgrid(dx, np.linspace(0, 200, num=96))
    return X, Y, dx


def get_time_labels():
    """
    Generate time labels for the 24 timesteps.

    Returns:
        list: Time labels as strings (e.g., "10 d", "2.5 y")
    """
    times = np.cumsum(np.power(1.421245, range(24)))
    time_print = []

    for t in range(times.shape[0]):
        if times[t] < 365:
            title = str(int(times[t])) + " d"
        else:
            title = f"{round(int(times[t]) / 365, 1)} y"
        time_print.append(title)

    return time_print


def create_pcolor_func(X, Y, thickness):
    """
    Create a pcolor plotting function for reservoir data.

    Args:
        X: X meshgrid
        Y: Y meshgrid
        thickness: Number of vertical cells in reservoir

    Returns:
        function: Plotting function that takes 2D array
    """

    def pcolor(x):
        plt.jet()
        return plt.pcolor(
            X[:thickness, :], Y[:thickness, :], np.flipud(x), shading="auto"
        )

    return pcolor


def plot_4x3_comparison(
    x_plot,
    y_plot,
    pred_plot,
    mask,
    thickness,
    variable="pressure",
    timesteps=[14, 20, 23],
):
    """
    Create a 4x3 comparison plot showing inputs, ground truth, predictions, and errors.

    Args:
        x_plot: Input array
        y_plot: Ground truth array
        pred_plot: Prediction array
        mask: Boolean mask for reservoir region
        thickness: Number of vertical cells
        variable: 'pressure' or 'saturation'
        timesteps: List of 3 timesteps to visualize

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Setup grid
    X, Y, dx = setup_plotting_grid()
    time_print = get_time_labels()
    pcolor = create_pcolor_func(X, Y, thickness)

    # Extract input maps
    poro_map = x_plot[0, :, :, 0, 2][mask].reshape((thickness, -1))
    kr_map = np.exp(x_plot[0, :, :, 0, 0][mask].reshape((thickness, -1)) * 15)
    kz_map = np.exp(x_plot[0, :, :, 0, 1][mask].reshape((thickness, -1)) * 15)

    # Create figure
    fig = plt.figure(figsize=(15, 16))

    # Set labels based on variable type
    if variable == "pressure":
        pred_label = "$\hat{dP}$ (bar)"
        true_label = "$dP$ (bar)"
        error_label = "|$dP-\hat{dP}$|"
    else:  # saturation
        pred_label = "$\hat{S}_g$ (-)"
        true_label = "$S_g$ (-)"
        error_label = "|$S_g-\hat{S}_g$|"

    for j, t in enumerate(timesteps):
        # Row 1: Input parameters
        plt.subplot(4, 3, j + 1)
        if j == 2:
            pcolor(poro_map)
            plt.title("$\phi$ (-)")
        elif j == 1:
            pcolor(kz_map)
            plt.title("$k_z$ (mD)")
        else:
            pcolor(kr_map)
            plt.title("$k_r$ (mD)")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        # Row 2: Ground truth
        plt.subplot(4, 3, j + 4)
        pcolor(y_plot[:, :, t][mask].reshape((thickness, -1)))
        plt.title(f"{true_label}, t={time_print[t]}")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        # Row 3: Prediction
        plt.subplot(4, 3, j + 7)
        pcolor(pred_plot[:, :, t][mask].reshape((thickness, -1)))
        plt.title(f"{pred_label}, t={time_print[t]}")
        plt.colorbar(fraction=0.02)
        plt.xlim([0, 3500])

        # Row 4: Error
        plt.subplot(4, 3, j + 10)
        error = pred_plot[:, :, t][mask].reshape((thickness, -1)) - y_plot[:, :, t][
            mask
        ].reshape((thickness, -1))
        pcolor(error)
        plt.colorbar(fraction=0.02)
        plt.title(f"{error_label}, t={time_print[t]}")
        plt.xlim([0, 3500])

    plt.tight_layout()

    return fig
