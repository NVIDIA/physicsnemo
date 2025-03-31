# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io import shapereader
from matplotlib import gridspec


COUNTY_SHAPE_FILE = (
    Path(__file__).parent
    / "figure_data"
    / "plotting_shapefiles"
    / "tl_2017_40_place.shp"
)


projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


def plot_obs_locations(ax, lat, lon, z, norm=None, cmap=None):
    latsss = np.where(~z.isnan(), lat, np.nan)
    lonsss = np.where(~z.isnan(), lon, np.nan)
    ax.scatter(
        lonsss,
        latsss,
        c=z,
        cmap=cmap,
        marker="^",
        norm=norm,
        edgecolor="black",
        s=80,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )


def add_features(ax):

    reader = shapereader.Reader(COUNTY_SHAPE_FILE.as_posix())

    for k, i in enumerate(reader.records()):
        # print(i.attributes)
        if i.attributes["NAME"] == "Oklahoma City":
            # print(i.attributes)
            # print(i.geometry)
            ok = i.geometry
            # print(k)
        if i.attributes["NAME"] == "Tulsa":
            t = i.geometry
    counties = [ok, t]

    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
    """Add features such as county, lake and other borders to an axes"""
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.8)
    ax.add_feature(
        cfeature.RIVERS.with_scale("10m"), alpha=0.8, edgecolor="gray", zorder=2
    )
    # ax.add_feature(USCOUNTIES, edgecolor='gray', alpha=0.8)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="black", linewidth=2.5)
    # ax.scatter(-97.5164, 35.4676, transform=ccrs.PlateCarree(),marker='x')
    ax.annotate(
        "Oklahoma City", (-97.8164, 35.7076), transform=ccrs.PlateCarree(), size=10
    )
    ax.annotate("Tulsa", (-96.05, 36.35), transform=ccrs.PlateCarree(), size=10)
    ax.add_feature(COUNTIES, facecolor="gray", edgecolor="none", linewidth=1, alpha=0.5)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1,
        color="gray",
        alpha=0.3,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )


def draw(
    x_star,
    mask=np.ones((128, 128), dtype=float),
    save=False,
    figname="test.png",
    dpi=300,
    show=True,
):
    plt.close("all")

    # Ensure tensor is on CPU and convert to numpy array
    mask = mask.astype(float)
    vmin_values = [-6, -6, 0]
    vmax_values = [6, 6, 0.1]
    u10_mean, u10_std = -0.262, 2.372
    v10_mean, v10_std = 0.865, 4.115
    logtp_mean, logtp_std = -8.117, 2.489

    m = np.array([u10_mean, v10_mean, logtp_mean])
    std = np.array([u10_std, v10_std, logtp_std])
    x_star = (x_star * std[np.newaxis, :, np.newaxis, np.newaxis]) + m[
        np.newaxis, :, np.newaxis, np.newaxis
    ]
    x_star[:, 2, :, :] = np.exp(x_star[:, 2, :, :] - 0.0001)
    tensor_np = x_star.cpu().numpy()

    n = tensor_np.shape[0]

    # Calculate the number of columns and rows
    cols = n
    rows = 3

    # Set up the figure and axes using gridspec
    fig = plt.figure(figsize=(cols * 5 + rows, rows * 5))
    gs = gridspec.GridSpec(
        rows, cols + 1, width_ratios=[1] * cols + [0.1], height_ratios=[1] * rows
    )  # Add extra space for colorbars
    # Set the colormap
    cmaps = ["RdBu", "RdBu", "Blues"]  # Choose your desired colormap here
    labels = ["10u", "10v", "tp"]
    # Loop through each row and plot the data with the provided vmin and vmax
    for i in range(rows):
        row_vmin = vmin_values[i]  # Get vmin for the current row
        row_vmax = vmax_values[i]  # Get vmax for the current row
        row_cmap = cmaps[i]
        # Loop through each column and plot the data with the provided vmin and vmax
        for j in range(cols):
            ax = plt.subplot(gs[i, j])
            # Plot masked data in light gray
            ax.imshow(
                tensor_np[j, i],
                cmap=row_cmap,
                vmin=row_vmin,
                vmax=row_vmax,
                alpha=mask,
                origin="lower",
            )
            ax.axis("off")

        # Add colorbar at the left side for each row
        ax = plt.subplot(gs[i, cols])
        ax.axis("off")
        norm = plt.Normalize(vmin=row_vmin, vmax=row_vmax)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=row_cmap), ax=ax, aspect=100
        )
        cbar.set_label(labels[i], size=20)
        cbar.ax.tick_params(labelsize=20)

    plt.tight_layout(pad=1.0)

    if save:
        plt.savefig(figname, format="png", dpi=dpi)

    if show:
        plt.show()

    return 0


def create_gif_from_pngs(png_dir, gif_path):
    """
    Create a GIF from PNG images in a directory.

    Args:
    - png_dir (str): Directory containing PNG images.
    - gif_path (str): Path to save the output GIF file.
    """
    # List PNG files in the directory and sort them in descending order
    png_files = sorted(
        [f for f in os.listdir(png_dir) if f.endswith(".png")], reverse=True
    )

    # Create a list to store image paths
    images = []

    # Iterate over PNG files and add them to the list
    for png_file in png_files:
        images.append(imageio.imread(os.path.join(png_dir, png_file)))

    # Save GIF file
    imageio.mimsave(gif_path, images)

    print(f"GIF created: {gif_path}")


def find_takeout(bool_array):
    # Find indices of True values
    true_indices = np.argwhere(bool_array)

    # Compute distances between each pair of True values
    distances = np.sqrt(np.sum((true_indices[:, None] - true_indices) ** 2, axis=2))

    # Exclude diagonal elements (distance to itself) by filling them with np.inf
    np.fill_diagonal(distances, np.inf)

    # Find the index of the minimum distance
    closest_pair_indices = np.unravel_index(np.argmin(distances), distances.shape)

    # now of the pair use the one with the smallest distance to the next point and return
    distances[closest_pair_indices[0]][closest_pair_indices[1]] = np.inf
    distances[closest_pair_indices[1]][closest_pair_indices[0]] = np.inf

    if np.min(distances[closest_pair_indices[0]]) < np.min(
        distances[closest_pair_indices[1]]
    ):
        takeout = true_indices[closest_pair_indices[0]]
    else:
        takeout = true_indices[closest_pair_indices[1]]
    # print(takeout)
    return takeout


def find_takeout_random(bool_array):
    # Find indices of True values
    true_indices = np.argwhere(bool_array)
    i = np.random.randint(0, len(true_indices))
    takeout = true_indices[i]

    # # Compute distances between each pair of True values
    # distances = np.sqrt(np.sum((true_indices[:, None] - true_indices) ** 2, axis=2))

    # # Exclude diagonal elements (distance to itself) by filling them with np.inf
    # np.fill_diagonal(distances, np.inf)

    # # Find the index of the minimum distance
    # closest_pair_indices = np.unravel_index(np.argmin(distances), distances.shape)

    # # now of the pair use the one with the smallest distance to the next point and return
    # distances[closest_pair_indices[0]][closest_pair_indices[1]]=np.inf
    # distances[closest_pair_indices[1]][closest_pair_indices[0]]=np.inf

    # if (np.min(distances[closest_pair_indices[0]]) < np.min(distances[closest_pair_indices[1]])):
    #     takeout = true_indices[closest_pair_indices[0]]
    # else:
    #     takeout = true_indices[closest_pair_indices[1]]
    # # print(takeout)
    return takeout


def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"saving figure to {path}.pdf ")
    plt.savefig(path + ".pdf")


def subplot(*args):
    ax = plt.subplot(*args, projection=projection)
    add_features(ax)
