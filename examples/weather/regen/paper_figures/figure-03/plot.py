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
dhrrr = torch.load("../figure_data/assim_obs/hrrr_comparison.pt")
dx = torch.load("../figure_data/missing_channel/assimilated_state.pt")


cmaps = ["RdBu", "RdBu", "Blues"]  # Choose your desired colormap here
labels = ["10u [m/s]", "10v [m/s]", "tp [mm/h]"]
vmin_values = [-6, -6, 1e-1]
vmax_values = [6, 6, 10]
fig = plt.figure(figsize=(13, 13))
norms = [
    colors.Normalize(vmin=vmin_values[0], vmax=vmax_values[0]),
    colors.Normalize(vmin=vmin_values[1], vmax=vmax_values[1]),
    colors.LogNorm(vmin=vmin_values[2], vmax=vmax_values[2]),
]
shrink = 0.72

ax = fig.add_subplot(2, 2, 1, projection=projection)
im = ax.pcolormesh(
    lons,
    lats,
    dhrrr[1],
    cmap="RdBu",
    norm=norms[0],
    transform=ccrs.PlateCarree(),
    alpha=1,
    rasterized=True,
)
cbar = fig.colorbar(im, ax=ax, shrink=shrink)
cbar.set_label(labels[1])
ax.set_title("HRRR")
addf(ax)

ax = fig.add_subplot(2, 2, 2, projection=projection)
im = ax.pcolormesh(
    lons,
    lats,
    dx[0, 1],
    cmap="RdBu",
    norm=norms[0],
    transform=ccrs.PlateCarree(),
    alpha=1,
    rasterized=True,
)
cbar = fig.colorbar(im, ax=ax, shrink=shrink)
cbar.set_label(labels[1])
ax.set_title("10v generated from 10u and tp")
addf(ax)
q_subs = 2
ax = fig.add_subplot(2, 2, 3, projection=projection)
im = ax.pcolormesh(
    lons,
    lats,
    dhrrr[2],
    cmap="Blues",
    norm=norms[2],
    transform=ccrs.PlateCarree(),
    alpha=1,
    rasterized=True,
)
Q1 = ax.quiver(
    lons[::q_subs, ::q_subs],
    lats[::q_subs, ::q_subs],
    dhrrr[0].numpy()[::q_subs, ::q_subs],
    dhrrr[1].numpy()[::q_subs, ::q_subs],
    transform=ccrs.PlateCarree(),
    scale=400,
)
ax.quiverkey(Q1, X=0.85, Y=1.05, U=10, label="10 m/s", labelpos="E")
cbar = fig.colorbar(im, ax=ax, shrink=shrink)
cbar.set_label(labels[2])
addf(ax)

ax = fig.add_subplot(2, 2, 4, projection=projection)
im = ax.pcolormesh(
    lons,
    lats,
    dx[0, 2],
    cmap="Blues",
    norm=norms[2],
    transform=ccrs.PlateCarree(),
    alpha=1,
    rasterized=True,
)
Q2 = ax.quiver(
    lons[::q_subs, ::q_subs],
    lats[::q_subs, ::q_subs],
    dx[0, 0].numpy()[::q_subs, ::q_subs],
    dx[0, 1].numpy()[::q_subs, ::q_subs],
    transform=ccrs.PlateCarree(),
    scale=400,
)
ax.quiverkey(Q2, X=0.85, Y=1.05, U=10, label="10 m/s", labelpos="E")
cbar = fig.colorbar(im, ax=ax, shrink=shrink)
cbar.set_label(labels[2])
addf(ax)

plt.tight_layout()
plt.savefig("../figures/missing_channel.pdf", format="pdf", dpi=300)
