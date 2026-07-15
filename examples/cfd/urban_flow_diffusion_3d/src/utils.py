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

from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np
from typing import Optional
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import datetime


def deep_clean_omegaconf(obj):
    """
    Recursively convert OmegaConf types to plain Python.
    Hydra configs like DictConfig/ListConfig can't be saved directly as JSON.
    Convert model._args to plain dict using OmegaConf.to_container before checkpointing.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        return {k: deep_clean_omegaconf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clean_omegaconf(v) for v in obj]
    return obj


def convert_unit2pixel(
    x: float = None,
    y: float = None,
    z: float = None,
    ds_ratio: float = None,
    flipped: bool = True,
) -> tuple:
    """
    Convert unit coordinates to pixel coordinates.

    Parameters:
    x (float): X-coordinate in original units.
    y (float): Y-coordinate in original units.
    z (float): Z-coordinate in original units.
    ds_ratio (float): Downsampling ratio.
    flipped (bool): Whether the z-coordinate is flipped or not.

    Returns:
    tuple: (x_pixel, y_pixel, z_pixel) coordinates in pixels.
    """
    if ds_ratio is None:
        raise ValueError("ds_ratio must be provided")

    x_pixel, y_pixel, z_pixel = None, None, None

    if x is not None:
        # x goes from -1 to 5 in original dataset
        x_pixel = int(x * (301 // ds_ratio) / (5 - -1) + 50 // ds_ratio)

    if y is not None:
        y_pixel = int(y * (101 // ds_ratio) / (2 - 0))

    if z is not None:
        if flipped:
            z_pixel = int(z * 151 / (1.5 - -1.5))
        else:
            z_pixel = int(151 / 2 + z * 151 / (1.5 - -1.5))

    return x_pixel, y_pixel, z_pixel


def convert_pixel2unit(
    x_pixel: int = None,
    y_pixel: int = None,
    z_pixel: int = None,
    ds_ratio: float = None,
    flipped: bool = True,
) -> tuple:
    """
    Convert pixel coordinates to unit coordinates.

    Parameters:
    x_pixel (int): X-coordinate in pixels.
    y_pixel (int): Y-coordinate in pixels.
    z_pixel (int): Z-coordinate in pixels.
    ds_ratio (float): Downsampling ratio.
    flipped (bool): Whether the z-coordinate is flipped or not.

    Returns:
    tuple: (x, y, z) coordinates in original units.
    """
    if ds_ratio is None:
        raise ValueError("ds_ratio must be provided")

    x, y, z = None, None, None

    if x_pixel is not None:
        # Reverse the transformation for x
        x = (x_pixel - 50 // ds_ratio) * (5 - (-1)) / (301 // ds_ratio)

    if y_pixel is not None:
        # Reverse the transformation for y
        y = y_pixel * (2 - 0) / (101 // ds_ratio)

    if z_pixel is not None:
        # Reverse the transformation for z
        if flipped:
            z = z_pixel * (1.5 - (-1.5)) / 151
        else:
            z = (z_pixel - 151 / 2) * (1.5 - (-1.5)) / 151

    return x, y, z


def select_random(arr, num_elements=1, seed=None, only_indices=False):
    """
    Randomly select one or more elements from a NumPy array.

    Args:
    arr (numpy.ndarray): Input array.
    num_elements (int): Number of elements to select randomly. Default is 1.

    Returns:
    numpy.ndarray or list: Randomly selected element or list of elements.
    """
    if num_elements < 1:
        raise ValueError("Number of elements to select must be at least 1.")

    np.random.seed(seed)

    # Generate random indices within the range of the array
    random_indices = np.sort(
        np.random.choice(arr.shape[0], size=num_elements, replace=False)
    )

    if only_indices:
        return random_indices
    else:
        # Return the selected elements
        if num_elements == 1:
            return arr[random_indices[0]]
        else:
            return arr[random_indices]


def calculate_mse(test_data, pred_data, max_val, std=False):
    """
    Calculate the Mean Squared Error (MSE) between test and predicted data.

    Parameters:
        test_data (numpy.ndarray): The ground truth data.
        pred_data (numpy.ndarray): The predicted data.
        max_val (float): The normalization factor to scale the errors.
        std (bool, optional): If True, return the standard deviation of the errors as well.

    Returns:
        tuple:
            - Mean squared error (numpy.ndarray) along axis 0.
            - (Optional) Standard deviation of errors along axis 0.
            - Error wake (numpy.ndarray): Element-wise normalized squared errors.
    """
    if max_val == 0:
        raise ValueError("max_val must be non-zero to avoid division by zero.")

    error = ((test_data - pred_data) ** 2) / (max_val**2)

    if std:
        return np.mean(error, axis=0), np.std(error, axis=0), error

    else:
        return np.mean(error, axis=0), error


def get_current_time():
    """
    Print the current date and time.
    """
    now = datetime.datetime.now()
    # print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    return now


def get_time_difference(start_time, end_time):
    """
    Print the difference between two datetime objects.
    """
    time_difference = end_time - start_time

    return time_difference
    # print(f"Time passed: {time_difference}")


def get_velocity_magnitude(data=None):
    return np.sqrt(data[:, 0, :, :] ** 2 + data[:, 1, :, :] ** 2)


##Common plot Utils
def basic_plt_setup():
    import matplotlib.pyplot as plt

    size = 20  # 40
    plt.rc("font", family="serif")
    plt.rc("text", usetex="true")
    plt.rc("font", size=size)
    plt.rc("axes", labelsize=size, linewidth=2)
    plt.rc("legend", fontsize=size, handletextpad=0.1)
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)

    return


def add_obstacle_patch(ax, plot_config, color="k"):
    # Obstacle dimensions & location (For one obstacle dataset)
    pos_x, pos_y = (
        plot_config.common.obs_pos_x,
        plot_config.common.obs_pos_y,
    )  # x position, y position
    width, height = (
        plot_config.common.obs_width,
        plot_config.common.obs_height,
    )  # width, height of the obstacle
    obstacle = patches.Rectangle(
        (pos_x, pos_y), width, height, linewidth=2, edgecolor=color, facecolor=color
    )
    ax.add_patch(obstacle)


def plot_subplot(
    ax=None,
    data=None,
    title=None,
    extent=None,
    vmin=None,
    vmax=None,
    plot_config=None,
    fontsize_title=None,
    fontsize=None,
    cbar_label=None,
    cbar_orientation="vertical",
    add_patch=True,
    errors=False,
):
    """
    Plot a subplot with the given data on the provided axes.
    #TODO: Remove the fontsize args, as now we are using the global rc.params
    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        data (numpy.ndarray): The data to be plotted.
        title (str): The title of the subplot.
        errors (bool): Whether to use error colormap limits. Default is False.
        add_patch (bool): Whether to add an obstacle patch to the plot. Default is True.

    Returns:
        im (matplotlib.image.AxesImage): The image object created by imshow.
    """

    x_label = plot_config.common.x_label
    y_label = plot_config.common.y_label
    x_ticks = plot_config.common.x_ticks
    y_ticks = plot_config.common.y_ticks
    ticksize = plot_config.common.ticksize
    cmap = plot_config.common.cmap

    if errors:
        vmin, vmax = 0, 50
    else:
        vmin, vmax = vmin, vmax

    im = ax.imshow(
        data.T,
        cmap=cmap,
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)  # , fontsize=fontsize_title

    ax.set_xlabel(x_label)  # , fontsize=fontsize
    ax.set_ylabel(y_label)  # , fontsize=fontsize

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.tick_params(axis="both")  # , labelsize=ticksize

    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation)
    # cbar.ax.tick_params(labelsize=ticksize)

    if cbar_label is not None:
        cbar.set_label(cbar_label)  # , fontsize=fontsize

    if add_patch:
        add_obstacle_patch(ax, plot_config)

    if plot_config.common.tight_layout:
        plt.tight_layout()

    return im


def error_in_wake(gtruth_data=None, pred_data=None, mask=None, ds_ratio=None):
    """
    Calculate the error and mean squared error (MSE) in the wake region of an obstacle in a flow field.

    The test and predicted data are compared to compute the error and MSE in the specified region. The wake region is masked
    based on the percentage of the obstacle's height.

    Parameters:
    -----------
    gtruth_data : np.ndarray   Ground truth data representing the flow field.
    pred_data : np.ndarray   Predicted data from a model to be compared against the ground truth.
    mask : float             Percentage (0 to 100) of the obstacle's height that defines the wake region to be analyzed.

    Returns:
    --------
    gtruth_data_wake : np.ndarray  Ground truth data for the wake region.
    pred_data_wake : np.ndarray  Predicted data for the wake region.
    error_wake : np.ndarray      Element-wise squared error between test and predicted data in the wake region.
    mse_wake : np.ndarray        Mean squared error for the wake region.
    mean_mse : np.ndarray        Mean of the MSE over spatial dimensions (1 and 2).
    """

    # Obstacle dimensions (in unit coordinates)
    pos_x, pos_y = -0.125, 0
    width, height = 0.25, 1

    # Convert dimensions to pixel values
    pos_x_pixel, pos_y_pixel, _ = convert_unit2pixel(
        x=pos_x, y=pos_y, ds_ratio=ds_ratio
    )
    width_pixel, height_pixel, _ = convert_unit2pixel(
        x=width, y=height, ds_ratio=ds_ratio
    )

    # Masked wake region (based on the height mask percentage)
    masked_height_pixel = int((mask / 100) * height_pixel)

    # Slice the wake region from the data
    gtruth_data_wake = gtruth_data[:, :, width_pixel:, masked_height_pixel:height_pixel]
    pred_data_wake = pred_data[:, :, width_pixel:, masked_height_pixel:height_pixel]

    # Calculate MSE and error for wake region
    max_gtruth_data = np.max(gtruth_data)
    mse_wake, error_wake = calculate_mse(
        gtruth_data_wake, pred_data_wake, max_gtruth_data
    )

    # Compute the mean over spatial dimensions
    mean_mse = np.mean(mse_wake, axis=(1, 2))

    return gtruth_data_wake, pred_data_wake, error_wake, mse_wake, mean_mse


def plot_inst_comp_and_error(
    random_indices=None,
    test_data=None,
    pred_data=None,
    error=None,
    mask=None,
    comp="streamwise",
    x_axis=None,
    y_axis=None,
    colormap="viridis",
    wake_region=False,
    ds_ratio=None,
    vmin=None,
    vmax=None,
):
    """
    Plot instantaneous comparison of test data, predicted data, and the error for selected indices in the specified component.

    Parameters:
    -----------
    random_indices : list of int    List of indices for which the comparison plots are generated.
    test_data : np.ndarray          Ground truth data (test data) for comparison.
    pred_data : np.ndarray          Model predicted data to compare with the test data.
    error : np.ndarray              Error data (e.g., MSE) between test and predicted data.
    comp : str, optional            Component to plot. Must be either 'streamwise' (default) or 'wall-normal'.
    x_axis : np.ndarray             Array representing the x-axis coordinates.
    y_axis : np.ndarray             Array representing the y-axis coordinates.
    colormap : str, optional        Colormap to use for the plots. Default is 'viridis'.
    wake_region : bool, optional    Whether to you want to compute and plot wake region specifically for same indices

    Returns:
    --------
    pdf : PdfPages object or None   If a PDF object is provided, the function returns the modified PDF object. Otherwise, it shows the plots and returns None.
    """

    # Check if comp is valid
    if comp not in ["streamwise", "wall-normal"]:
        raise ValueError("comp must be either 'streamwise' or 'wall-normal'.")

    # Select the correct channel
    channel = 0 if comp == "streamwise" else 1

    # Number of test data points
    num_indices = len(random_indices)

    # Create subplots
    fig, axs = plt.subplots(
        num_indices,
        3,
        figsize=(
            3 * plot_config.figure.figsize[0],
            num_indices * plot_config.figure.figsize[1],
        ),
    )

    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]

    fontsize_title, fontsize = plot_config.axes.fontsize, plot_config.axes.fontsize
    labels = plot_config.axes.fluc_label

    if wake_region:
        test_data_wake, pred_data_wake, error_wake, _, _ = error_in_wake(
            gtruth_data=test_data, pred_data=pred_data, mask=mask, ds_ratio=ds_ratio
        )

        for i, num in enumerate(random_indices):
            im0 = plot_subplot(
                axs[i, 0],
                test_data_wake[num, channel, :, :],
                "",
                extent,
                vmin,
                vmax,
                colormap,
                fontsize_title=fontsize_title,
                fontsize=fontsize,
                add_patch=False,
            )
            im1 = plot_subplot(
                axs[i, 1],
                pred_data_wake[num, channel, :, :],
                "",
                extent,
                vmin,
                vmax,
                colormap,
                fontsize_title=fontsize_title,
                fontsize=fontsize,
                add_patch=False,
            )
            im2 = plot_subplot(
                axs[i, 2],
                error_wake[num, channel, :, :],
                "",
                extent,
                0,
                0.1,
                colormap,
                fontsize_title=fontsize_title,
                fontsize=fontsize,
                add_patch=False,
            )

    # Loop through the data
    for i, num in enumerate(random_indices):
        im0 = plot_subplot(
            axs[i, 0],
            test_data[num, channel, :, :],
            "",
            extent,
            vmin,
            vmax,
            colormap,
            fontsize_title=fontsize_title,
            fontsize=fontsize,
            add_patch=True,
        )
        im1 = plot_subplot(
            axs[i, 1],
            pred_data[num, channel, :, :],
            "",
            extent,
            vmin,
            vmax,
            colormap,
            fontsize_title=fontsize_title,
            fontsize=fontsize,
            add_patch=True,
        )
        im2 = plot_subplot(
            axs[i, 2],
            error[num, channel, :, :],
            "",
            extent,
            0,
            0.1,
            colormap,
            fontsize_title=fontsize_title,
            fontsize=fontsize,
            add_patch=True,
        )

    return fig


def error_evolution(
    errors=None, x_axis=None, x_label=None, y_label=None, legend_labels=None
):
    """
    Plot error evolution and histograms.

    Parameters:
    - errors: ndarray of shape (n, 2), containing error values for each series.
    - x_axis: ndarray of shape (n, 1), containing x-axis values.
    - x_label: str, label for the x-axis.
    - y_label: str, label for the y-axis.
    - legend_labels: list of str, labels for the legend corresponding to error series.

    Returns:
    - fig: Matplotlib figure object.
    """
    # Determine number of error series
    num_series = errors.shape[1]

    # Create subplots: 1 row for evolution plots and 1 row for histograms
    fig, axs = plt.subplots(2, num_series, figsize=(num_series * 6, 10))

    for i in range(num_series):
        if num_series > 1:
            # Evolution plot
            axs[0, i].scatter(
                x_axis, errors[:, i], label=legend_labels[i], color="blue"
            )
            axs[0, i].set_xlabel(x_label)
            axs[0, i].set_ylabel(f"{legend_labels[i]}")
            axs[0, i].grid(True)

            # Histogram plot
            axs[1, i].hist(
                errors[:, i], bins=20, color="orange", alpha=0.7, edgecolor="black"
            )
            axs[1, i].set_xlabel(legend_labels[i])
            axs[1, i].set_ylabel("Frequency")
            axs[1, i].grid(True)
        else:
            axs[0].scatter(x_axis, errors[:, i], label=legend_labels[i], color="blue")
            axs[0].set_xlabel(x_label)
            axs[0].set_ylabel(f"{legend_labels[i]}")
            axs[0].grid(True)

            # Histogram plot
            axs[1].hist(
                errors[:, i], bins=20, color="orange", alpha=0.7, edgecolor="black"
            )
            axs[1].set_xlabel(legend_labels[i])
            axs[1].set_ylabel("Frequency")
            axs[1].grid(True)

    plt.tight_layout()
    return fig


def plot_two_comps(
    data=None,
    x_axis=None,
    y_axis=None,
    x_label=None,
    y_label=None,
    vmin=None,
    vmax=None,
):
    fig, axs = plt.subplots(
        1, 2, figsize=(2 * plot_config.figure.figsize[0], plot_config.figure.figsize[1])
    )
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]

    if vmin is None or vmax is None:
        vmin, vmax = np.min(data), np.max(data)

    fontsize_title, fontsize = plot_config.axes.fontsize, plot_config.axes.fontsize

    im0 = plot_subplot(
        ax=axs[0],
        data=data[0],
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        fontsize_title=fontsize_title,
        fontsize=fontsize,
        add_patch=True,
    )
    im1 = plot_subplot(
        ax=axs[1],
        data=data[1],
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        fontsize_title=fontsize_title,
        fontsize=fontsize,
        add_patch=True,
    )

    return fig
