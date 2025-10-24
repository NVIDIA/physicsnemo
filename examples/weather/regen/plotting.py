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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_sample(x, obs, lat, lon):
    """
    Args:
        x: shaped (channel, x, y)
        obs: shaped (channel, x, y)
        lat: shaped (x, y)
        lon: shaped (x, y)
    """

    try:
        reader = shpreader.Reader(
            "sda/experiments/corrdiff/figure_data/plotting_shapefiles/tl_2017_40_place.shp"
        )
    except Exception:
        counties = []
    else:
        for k, i in enumerate(reader.records()):
            if i.attributes["NAME"] == "Oklahoma City":
                ok = i.geometry
            if i.attributes["NAME"] == "Tulsa":
                t = i.geometry
        counties = [ok, t]

    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    # In[31]:

    projection = ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )

    fig, axs = plt.subplots(
        1, 3, subplot_kw=dict(projection=projection), constrained_layout=True
    )

    # In[32]:

    def addf(ax):
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.8)
        ax.add_feature(
            cfeature.RIVERS.with_scale("10m"), alpha=0.8, edgecolor="gray", zorder=2
        )
        # ax.add_feature(USCOUNTIES, edgecolor='gray', alpha=0.8)
        ax.add_feature(
            cfeature.STATES.with_scale("10m"), edgecolor="black", linewidth=2.5
        )
        # ax.scatter(-97.5164, 35.4676, transform=ccrs.PlateCarree(),marker='x')
        ax.annotate(
            "Oklahoma City", (-97.8164, 35.7076), transform=ccrs.PlateCarree(), size=10
        )
        ax.annotate("Tulsa", (-96.05, 36.35), transform=ccrs.PlateCarree(), size=10)
        ax.add_feature(
            COUNTIES, facecolor="gray", edgecolor="none", linewidth=1, alpha=0.5
        )
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

    # In[34]:

    toplot = x
    latss = lat
    lonss = lon

    # In[38]:

    cmaps = ["RdBu", "RdBu", "Blues"]
    labels = ["10u [m/s]", "10v [m/s]", "tp [mm/h]"]
    vmin_values = [-6, -6, 1e-1]
    vmax_values = [6, 6, 10]
    norms = [
        colors.Normalize(vmin=vmin_values[0], vmax=vmax_values[0]),
        colors.Normalize(vmin=vmin_values[1], vmax=vmax_values[1]),
        colors.LogNorm(vmin=vmin_values[2], vmax=vmax_values[2]),
    ]

    channels = 3
    for i in range(channels):
        ax = axs[i]
        row_cmap = cmaps[i]
        row_norm = norms[i]
        im = ax.pcolormesh(
            lon,
            lat,
            x[0, i],
            cmap=row_cmap,
            norm=row_norm,
            transform=ccrs.PlateCarree(),
            alpha=1,
            rasterized=True,
        )
        toplotn = obs
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
        cbar = fig.colorbar(im, ax=ax, shrink=0.87, orientation="horizontal")
        cbar.set_label(labels[i])
        addf(ax)
