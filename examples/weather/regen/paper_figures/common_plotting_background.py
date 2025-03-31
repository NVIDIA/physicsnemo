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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib

matplotlib.rcParams.update({"font.size": 14})

reader = shpreader.Reader("../figure_data/plotting_shapefiles/tl_2017_40_place.shp")

for k, i in enumerate(reader.records()):
    # print(i.attributes)
    if i.attributes["NAME"] == "Oklahoma City":
        # print(i.attributes)
        # print(i.geometry)
        ok = i.geometry
        # print(k)
    if i.attributes["NAME"] == "Tulsa":
        t = i.geometry
counties = list((ok, t))

COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())


def addf(ax):
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
