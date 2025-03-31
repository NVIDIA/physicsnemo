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
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 14})
import cartopy.crs as ccrs
import xarray as xr
from common_plotting_background import addf

# %%

projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

lats = np.load("../figure_data/latitudes.npy")
lons = np.load("../figure_data/longitudes.npy", allow_pickle=True)
stat_loc = xr.open_dataarray("../station_locations_on_grid.nc")
bool_array = stat_loc.values.astype(bool)
valid = np.load("../evenmore_random_val_stations.npy")
num_leave = 25
valid = valid[:num_leave]
bool_array = stat_loc.values.astype(bool)
for indices in valid:
    bool_array[indices[0], indices[1]] = False
tune = np.zeros_like(bool_array)
for indices in valid:
    tune[indices[0], indices[1]] = True


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection=projection)
addf(ax)
latss = lats[bool_array]
lonss = lons[bool_array]
ax.scatter(
    lonss,
    latss,
    c="C00",
    marker="^",
    edgecolor="black",
    s=100,
    transform=ccrs.PlateCarree(),
    zorder=2,
)

latss = lats[tune]
lonss = lons[tune]
ax.scatter(
    lonss,
    latss,
    c="C01",
    marker="s",
    edgecolor="black",
    s=100,
    transform=ccrs.PlateCarree(),
    zorder=2,
)
ax.legend(["Assimilation", "Evaluation"])

plt.savefig("figures/25_left_out_stations.pdf", format="pdf")

# %%
