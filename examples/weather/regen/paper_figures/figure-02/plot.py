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
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# import xarray as xr
import cartopy.crs as ccrs
import matplotlib

matplotlib.rcParams.update({"font.size": 14})
from common_plotting_background import addf
import matplotlib.colors as colors

projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


# %%
plt.plot(lons[0])

# %%
lats = np.load("../figure_data/latitudes.npy")
lons = np.load("../figure_data/longitudes.npy")
dconc = torch.load("../figure_data/assim_hrrr/guidance.pt")
dx = torch.load("../figure_data/assim_hrrr/assimilated_state.pt")
mask = torch.load("../figure_data/assim_hrrr/mask.pt")
toplot = dconc

# Set the colormap
import matplotlib.colors as colors

cmaps = ["RdBu", "RdBu", "Blues"]  # Choose your desired colormap here
labels = ["10u [m/s]", "10v [m/s]", "tp [mm/h]"]
vmin_values = [-6, -6, 1e-1]
vmax_values = [6, 6, 10]
fig = plt.figure(figsize=(18, 26))
norms = [
    colors.Normalize(vmin=vmin_values[0], vmax=vmax_values[0]),
    colors.Normalize(vmin=vmin_values[1], vmax=vmax_values[1]),
    colors.LogNorm(vmin=vmin_values[2], vmax=vmax_values[2]),
]

channels = 3
display = 5
for i in range(channels * display):
    ax = fig.add_subplot(display, channels, i + 1, projection=projection)
    row = i % channels
    column = i // channels
    row_vmin = vmin_values[row]
    row_vmax = vmax_values[row]
    row_cmap = cmaps[row]
    row_norm = norms[row]
    title = [
        "HRRR and station data",
        "HRRR subsampled to 1 in 8, 1.6%",
        "HRRR subsampled to 1 in 18, 0.3%",
        "HRRR at station locations",
        "station observations",
    ][column]
    if column == 0:
        im = ax.pcolormesh(
            lons,
            lats,
            dconc[0, row],
            cmap=row_cmap,
            norm=row_norm,
            transform=ccrs.PlateCarree(),
            alpha=1,
            rasterized=True,
        )
        toplotn = toplot[3, row][mask[3, row]]
        latss = lats[mask[3, row]]
        lonss = lons[mask[3, row]]
        latsss = np.where(~toplotn.isnan(), latss, np.nan)
        lonsss = np.where(~toplotn.isnan(), lonss, np.nan)
        ax.scatter(
            lonsss,
            latsss,
            c=toplotn.numpy(),
            cmap=row_cmap,
            marker="^",
            norm=row_norm,
            edgecolor="black",
            s=80,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    else:
        im = ax.pcolormesh(
            lons,
            lats,
            dx[column - 1][row],
            cmap=row_cmap,
            norm=row_norm,
            transform=ccrs.PlateCarree(),
            alpha=1,
            rasterized=True,
        )
        toplotn = toplot[column - 1, row][mask[column - 1, row]]
        latss = lats[mask[column - 1, row]]
        lonss = lons[mask[column - 1, row]]
        latsss = np.where(~toplotn.isnan(), latss, np.nan)
        lonsss = np.where(~toplotn.isnan(), lonss, np.nan)
        if column < 4:
            ax.scatter(
                lonsss,
                latsss,
                c=toplotn.numpy(),
                cmap=row_cmap,
                marker="p",
                norm=row_norm,
                edgecolor="black",
                s=80,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
        else:
            ax.scatter(
                lonsss,
                latsss,
                c=toplotn.numpy(),
                cmap=row_cmap,
                marker="^",
                norm=row_norm,
                edgecolor="black",
                s=80,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.87)
    cbar.set_label(labels[row])
    addf(ax)
    if row == 0:
        ax.text(
            -0.07,
            0.55,
            title,
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        # ax.set_ylabel()


plt.tight_layout()
plt.savefig("hrrr_subsampled_to_obs_rotated.pdf", format="pdf", dpi=300)


# %%
