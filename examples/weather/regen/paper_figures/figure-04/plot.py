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
import numpy as np
import torch
import matplotlib.pyplot as plt
import xarray as xr
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


lats = np.load("../figure_data/latitudes.npy")
lons = np.load("../figure_data/longitudes.npy")
toplot = torch.load("../figure_data/assim_obs/guidance.pt")
dhrrr = torch.load("../figure_data/assim_obs/hrrr_comparison.pt")
dx = torch.load("../figure_data/assim_obs/assimilated_state.pt")
mask = torch.load("../figure_data/assim_obs/mask.pt")


cmaps = ["RdBu", "RdBu", "Blues"]  # Choose your desired colormap here
labels = ["10u [m/s]", "10v [m/s]", "tp [mm/h]"]
vmin_values = [-6, -6, 1e-1]
vmax_values = [6, 6, 10]
fig = plt.figure(figsize=(24, 18))
norms = [
    colors.Normalize(vmin=vmin_values[0], vmax=vmax_values[0]),
    colors.Normalize(vmin=vmin_values[1], vmax=vmax_values[1]),
    colors.LogNorm(vmin=vmin_values[2], vmax=vmax_values[2]),
]

channels = 3
display = 4
for i in range(channels * display):
    ax = fig.add_subplot(channels, display, i + 1, projection=projection)
    column = i % display
    row = i // display
    row_vmin = vmin_values[row]
    row_vmax = vmax_values[row]
    row_cmap = cmaps[row]
    row_norm = norms[row]
    title = [
        "HRRR",
        "One SDA ensemble member",
        "Ensemble mean",
        "Ensemble standard deviation",
        "station observations",
    ][column]
    if column == 0:
        im = ax.pcolormesh(
            lons,
            lats,
            dhrrr[row],
            cmap=row_cmap,
            norm=row_norm,
            transform=ccrs.PlateCarree(),
            alpha=1,
            rasterized=True,
        )
    if column == 1:
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
        toplotn = toplot[0, row][mask[0, row]]
        latss = lats[mask[column - 1, row]]
        lonss = lons[mask[column - 1, row]]
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
    if column == 2:
        im = ax.pcolormesh(
            lons,
            lats,
            dx.mean(axis=0)[row],
            cmap=row_cmap,
            norm=row_norm,
            transform=ccrs.PlateCarree(),
            alpha=1,
            rasterized=True,
        )
    if column == 3:
        norm = [None, None, colors.LogNorm(vmin=vmin_values[2], vmax=2e1)][row]
        im = ax.pcolormesh(
            lons,
            lats,
            dx.std(axis=0)[row],
            cmap="viridis",
            norm=norm,
            transform=ccrs.PlateCarree(),
            alpha=1,
            rasterized=True,
        )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(labels[row])
    addf(ax)
    if row == 0:
        ax.set_title(title)
        # ax.text(-0.07, 0.55, title, va='bottom', ha='center',
        # rotation='vertical', rotation_mode='anchor',
        # transform=ax.transAxes)
        # ax.set_ylabel()


plt.tight_layout()
plt.savefig("../figures/assim_obs.pdf", format="pdf", dpi=300)
