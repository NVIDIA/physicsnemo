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
General utility helper functions for coordinate conversion, data extraction, and visualization.

This module provides a collection of utility functions for:
- Converting between unit and pixel coordinates
- Extracting data from 2D and 3D arrays based on spatial coordinates
- Combining velocity fields into multi-dimensional arrays
- Statistical analysis and visualization of flow field data
- Plotting utilities for creating comparison plots and error visualizations
"""

import numpy as np
from typing import Optional
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import argparse

from conf.plot_configs import plot_dict


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


def combine_fields(
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine the fields u, v, and optionally w into a single numpy array.

    Parameters:
    u (np.ndarray): The u field array with shape [26000, 144, 48].
    v (np.ndarray): The v field array with shape [26000, 144, 48].
    w (np.ndarray, optional): The w field array with shape [26000, 144, 48]. Defaults to None.

    Returns:
    np.ndarray: Combined array with shape [26000, 2, 144, 48] or [26000, 3, 144, 48].
    """

    if u is None or v is None:
        raise ValueError("Atleast two fields must be provided")

    # Add a new axis to both u and v
    u_expanded = np.expand_dims(u, axis=1)
    v_expanded = np.expand_dims(v, axis=1)

    if w is not None:
        w_expanded = np.expand_dims(w, axis=1)
        # Concatenate u, v, and w along the new axis
        data = np.concatenate((u_expanded, v_expanded, w_expanded), axis=1)
    else:
        # Concatenate only u and v along the new axis
        data = np.concatenate((u_expanded, v_expanded), axis=1)

    # print(f"Shape of the combined data: {data.shape}")  # Should print either (26000, 2, 144, 48) or (26000, 3, 144, 48)

    return data


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


def select_random(arr, num_elements=1, seed=None):
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
    random_indices = np.random.choice(arr.shape[0], size=num_elements, replace=False)

    # Return the selected elements
    if num_elements == 1:
        return arr[random_indices[0]]
    else:
        return arr[random_indices]


def dict2namespace(config):
    """
    Convert a nested dictionary to an argparse.Namespace object.

    This function recursively converts dictionary configurations into a namespace
    object, which allows accessing dictionary keys as object attributes using
    dot notation (e.g., namespace.key instead of dict['key']).

    Parameters
    ----------
    config : dict
        A dictionary configuration to be converted. Can contain nested dictionaries
        which will be recursively converted to nested Namespace objects.

    Returns
    -------
    argparse.Namespace
        A namespace object with dictionary keys as attributes. Nested dictionaries
        are recursively converted to nested Namespace objects.

    Examples
    --------
    >>> config = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>> ns = dict2namespace(config)
    >>> ns.a
    1
    >>> ns.b.c
    2
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


##Common plot Utils
plot_config = dict2namespace(plot_dict)


def add_obstacle_patch(ax, color="k"):
    """
    Add a rectangular obstacle patch to a matplotlib axes object.

    This function draws a rectangular patch representing an obstacle in a flow field
    visualization. The obstacle position and dimensions are retrieved from the global
    plot configuration.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object on which to add the obstacle patch.
    color : str, optional
        The color of the obstacle patch. Can be a color name (e.g., 'k' for black,
        'r' for red) or a hex color code. Default is 'k' (black).

    Returns
    -------
    None

    Notes
    -----
    The obstacle dimensions and position are obtained from the global `plot_config`
    object, specifically from:
    - plot_config.figure.obs_pos_x: x-coordinate of the obstacle position
    - plot_config.figure.obs_pos_y: y-coordinate of the obstacle position
    - plot_config.figure.obs_width: width of the obstacle
    - plot_config.figure.obs_height: height of the obstacle

    The obstacle is drawn as a filled rectangle with a visible edge.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> add_obstacle_patch(ax, color='red')
    >>> ax.set_xlim(-1, 5)
    >>> ax.set_ylim(0, 2)
    >>> plt.show()
    """
    # Obstacle dimensions & location (For one obstacle dataset)
    pos_x, pos_y = (
        plot_config.figure.obs_pos_x,
        plot_config.figure.obs_pos_y,
    )  # x position, y position
    width, height = (
        plot_config.figure.obs_width,
        plot_config.figure.obs_height,
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
    colormap=plot_config.plot.snap_cmap,
    fontsize_title=plot_config.axes.fontsize,
    fontsize=plot_config.axes.fontsize,
    x_label=plot_config.axes.x_label,
    y_label=plot_config.axes.y_label,
    x_ticks=plot_config.axes.x_ticks,
    y_ticks=plot_config.axes.y_ticks,
    ticksize=plot_config.axes.ticksize,
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

    if errors:
        vmin, vmax = 0, 50
    else:
        vmin, vmax = vmin, vmax

    im = ax.imshow(
        data.T,
        cmap=colormap,
        extent=extent,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)  #  , fontsize=fontsize_title)

    ax.set_xlabel(x_label)  # , fontsize=fontsize)
    ax.set_ylabel(y_label)  # , fontsize=fontsize)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.tick_params(axis="both")  # , labelsize=ticksize)

    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation)
    # cbar.ax.tick_params(labelsize=ticksize)

    if cbar_label is not None:
        cbar.set_label(cbar_label)  # , fontsize=fontsize)

    if add_patch:
        add_obstacle_patch(ax)

    if plot_config.figure.tight_layout:
        plt.tight_layout()

    return im


def calculate_mse(test_data, pred_data, max_val):
    """Calculate the Mean Squared Error (MSE) between test and predicted data."""
    error_wake = ((test_data - pred_data) ** 2 / max_val**2) * 100
    return np.mean(error_wake, axis=0), error_wake


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
    pdf=None,
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
    pdf : PdfPages object, optional If provided, the plots will be saved to this PDF. Otherwise, plots will be shown.
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
    vmin, vmax = np.min(test_data), np.max(test_data)

    fontsize_title, fontsize = plot_config.axes.fontsize, plot_config.axes.fontsize

    if wake_region:
        test_data_wake, pred_data_wake, error_wake, _, _ = error_in_wake(
            gtruth_data=test_data, pred_data=pred_data, mask=mask, ds_ratio=ds_ratio
        )

        for i, num in enumerate(random_indices):
            plot_subplot(
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
            plot_subplot(
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
            plot_subplot(
                axs[i, 2],
                error_wake[num, channel, :, :],
                "",
                extent,
                0,
                50,
                colormap,
                fontsize_title=fontsize_title,
                fontsize=fontsize,
                add_patch=False,
            )

    # Loop through the data
    for i, num in enumerate(random_indices):
        plot_subplot(
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
        plot_subplot(
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
        plot_subplot(
            axs[i, 2],
            error[num, channel, :, :],
            "",
            extent,
            0,
            40,
            colormap,
            fontsize_title=fontsize_title,
            fontsize=fontsize,
            add_patch=True,
        )

    # Save to PDF if provided
    if pdf is not None:
        pdf.savefig(fig)
        return pdf
    else:
        # plt.show()
        pass
    # Close the figure
    plt.close(fig)

    return None
