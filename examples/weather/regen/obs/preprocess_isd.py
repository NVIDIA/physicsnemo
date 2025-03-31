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
import pandas as pd
from itertools import product
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

from config import path_to_isp


def extract_wind_speed(x):
    if type(x) == str:
        if float(x.split(",")[3]) == 9999:
            return np.nan
        else:
            return float(x.split(",")[3]) / 10
    else:
        return np.nan


def extract_uv(row):
    x = row.WND
    if type(x) == str:
        if float(x.split(",")[3]) == 9999:
            return np.nan, np.nan
        elif (float(x.split(",")[0]) == 999) & (x.split(",")[2] == "C"):
            return 0.0, 0.0
        else:
            speed = float(x.split(",")[3]) / 10
            angle = float(x.split(",")[0])
            u = (
                -np.sin(np.radians(angle)) * speed
            )  # negative because we get "the angle, measured in a clockwise direction, between true north and the direction from which the wind is blowing."
            v = -np.cos(np.radians(angle)) * speed
            return u, v
    else:
        return np.nan, np.nan


def extract_mm(x):
    """
    Example:
        01,0000,9,5 -> time of measurement, 10^-5 m, quality flag, source of data ( different data catalog)

    quality flag - always valid when data present in csv
    source of data - different data catalog

    Returns:
        total precip in mm over the previous hour

    """
    if type(x) == str:
        if float(x.split(",")[1]) == 9999.0:
            return np.nan
        else:
            return float(x.split(",")[1]) / 10
    else:
        return np.nan


def lonlat_to_xy(projection, longitude, latitude):
    x, y = projection.transform_point(longitude, latitude, ccrs.PlateCarree())
    xmin = -2697520.142522
    x_dist = 3000.0
    ymin = -1587306.152557
    y_dist = 3000.0
    x_reg = 834
    y_reg = 353

    x = ((x - xmin) / x_dist) - x_reg
    y = ((y - ymin) / y_dist) - y_reg
    return x, y


def latlon_to_xy_wrapper(row):
    x, y = lonlat_to_xy(projection, row["LONGITUDE"], row["LATITUDE"])
    return x, y


projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)
year = 2017
yafter = year + 1

# import ISD data in csv format
stations = pd.read_csv(path_to_isp)
stations["DATE"] = pd.to_datetime(stations.DATE)
stations["tp"] = stations.AA1.map(extract_mm)
stations[["u10", "v10"]] = stations.apply(extract_uv, axis=1, result_type="expand")

stations_17 = stations.loc[
    stations.DATE <= pd.to_datetime(f"{yafter}-01-01T00:00:00.000000000")
]
stations_17 = stations_17.loc[
    stations.DATE > pd.to_datetime(f"{year}-01-01T00:00:00.000000000")
]
# stations_17.dropna(subset = ['u10','v10', 'tp'],how='all', inplace=True)

# Apply the function to each row of the DataFrame
stations_17[["x", "y"]] = stations_17.apply(
    latlon_to_xy_wrapper, axis=1, result_type="expand"
)

stations_17["x"] = stations_17.x.round()
stations_17["y"] = stations_17.y.round()


df = stations_17[["DATE", "u10", "v10", "tp", "x", "y"]]

unique_combinations = df[["x", "y"]].drop_duplicates()

# Get unique values of 'x' and 'y'
unique_x_values = np.arange(128)
unique_y_values = np.arange(128)

# Create an empty xarray DataArray with all zeros
data_array = xr.DataArray(
    np.zeros((len(unique_y_values), len(unique_x_values))),
    coords={"y": unique_y_values, "x": unique_x_values},
    dims=["y", "x"],
)

# Loop through each unique combination and update the entries of the xarray DataArray
for _, row in unique_combinations.iterrows():
    data_array.loc[row["y"], row["x"]] = 1

data_array.to_netcdf("station_locations_on_grid.nc")

# Interpolate in time so we align with the full hours of HRRR
# Add 30mins as we will round down to full hours later, but 8:45 should be rounded to 9:00
df["DATE"] = pd.to_datetime(df["DATE"]) + pd.Timedelta("30m")

# Group DataFrame by station and interpolate values
df_interpolated = (
    df.groupby(["x", "y", pd.Grouper(freq="H", key="DATE")])
    .mean()
    .apply(lambda group: group.interpolate(limit=1))
)

# Create a new DataFrame with timestamps at full hours
full_hours = pd.date_range(
    start=df_interpolated.index.get_level_values("DATE").min().floor("H"),
    end=df_interpolated.index.get_level_values("DATE").max().ceil("H"),
    freq="H",
)

# Create Cartesian product of 'DATE', 'x', and 'y' values
cartesian_product = list(
    product(
        full_hours, df_interpolated.index.levels[0], df_interpolated.index.levels[1]
    )
)

# Merge interpolated gust values with new DataFrame using full hour timestamps
df_final = pd.DataFrame(cartesian_product, columns=["DATE", "x", "y"])
df_final = df_final.merge(
    df_interpolated.reset_index(),
    on=[
        "x",
        "y",
        "DATE",
    ],
    how="left",
)

df_final.set_index(
    [
        "x",
        "y",
        "DATE",
    ],
    inplace=True,
)

stations_naive_grid = df_final.to_xarray()

stations_naive_grid = stations_naive_grid.sel(
    {"DATE": slice(f"{year}-01-01T01:00:00.00", f"{yafter}-01-01T00:00:00.00")}
)


# Grid the data to a regular grid
# Group by x and y coordinates and aggregate by mean
ds = stations_naive_grid

# Define the integer x and y positions
x_int = np.arange(128)
y_int = np.arange(128)

# Create a new dataset with integer x and y positions
ds_regrid = xr.Dataset(coords={"x": x_int, "y": y_int})

# Interpolate the original dataset to the new integer x-y grid
ds_interp = ds.interp(x=x_int, y=y_int, method="nearest")

# Use combine_first to ensure that each grid cell has only one value
ds_regrid["u10"] = ds_interp["u10"].combine_first(ds["u10"])
ds_regrid["v10"] = ds_interp["v10"].combine_first(ds["v10"])
ds_regrid["tp"] = ds_interp["tp"].combine_first(ds["tp"])

# The interpolation above duplicates entries in nearby pixels, so we need to ensure only one pixel has a value per station
ds_regrid = ds_regrid.where(data_array == 1)
