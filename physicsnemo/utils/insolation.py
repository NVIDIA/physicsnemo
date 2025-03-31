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
import pandas as pd


def insolation(
    dates, lat, lon, scale=1.0, daily=False, enforce_2d=False, clip_zero=True
):  # pylint: disable=invalid-name
    """
    Calculate the approximate solar insolation for given dates.

    For an example reference, see:
    https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html

    Parameters
    ----------
    dates:
    dates: np.ndarray
        1d array: datetime or Timestamp
    lat: np.ndarray
        1d or 2d array of latitudes
    lon: np.ndarray
        1d or 2d array of longitudes (0-360deg). If 2d, must match the shape of lat.
    scale: float, optional
        scaling factor (solar constant)
    daily: bool, optional
        if True, return the daily max solar radiation (lat and day of year dependent only)
    enforce_2d: bool, optional
        if True and lat/lon are 1-d arrays, turns them into 2d meshes.
    clip_zero: bool, optional
        if True, set values below 0 to 0
    Returns
    -------
    np.ndarray: insolation (date, lat, lon)
    """
    # pylint: disable=invalid-name
    if len(lat.shape) != len(lon.shape):
        raise ValueError("'lat' and 'lon' must have the same number of dimensions")
    if len(lat.shape) >= 2 and lat.shape != lon.shape:
        raise ValueError(
            f"shape mismatch between lat ({lat.shape} and lon ({lon.shape})"
        )
    if len(lat.shape) == 1 and enforce_2d:
        lon, lat = np.meshgrid(lon, lat)
    n_dim = len(lat.shape)

    # Constants for year 1995 (standard in climate modeling community)
    # Obliquity of Earth
    eps = 23.4441 * np.pi / 180.0
    # Eccentricity of Earth's orbit
    ecc = 0.016715
    # Longitude of the orbit's perihelion (when Earth is closest to the sun)
    om = 282.7 * np.pi / 180.0
    beta = np.sqrt(1 - ecc**2.0)

    # Get the day of year as a float.
    start_years = np.array(
        [pd.Timestamp(pd.Timestamp(d).year, 1, 1) for d in dates], dtype="datetime64"
    )
    days_arr = (np.array(dates, dtype="datetime64") - start_years) / np.timedelta64(
        1, "D"
    )
    for d in range(n_dim):
        days_arr = np.expand_dims(days_arr, -1)
    # For daily max values, set the day to 0.5 and the longitude everywhere to 0 (this is approx noon)
    if daily:
        days_arr = 0.5 + np.round(days_arr)
        new_lon = lon.copy().astype(np.float32)
        new_lon[:] = 0.0
    else:
        new_lon = lon.astype(np.float32)
    # Longitude of the earth relative to the orbit, 1st order approximation
    lambda_m0 = ecc * (1.0 + beta) * np.sin(om)
    lambda_m = lambda_m0 + 2.0 * np.pi * (days_arr - 80.5) / 365.0
    lambda_ = lambda_m + 2.0 * ecc * np.sin(lambda_m - om)
    # Solar declination
    dec = np.arcsin(np.sin(eps) * np.sin(lambda_))
    # Hour angle
    h = 2 * np.pi * (days_arr + new_lon / 360.0)
    # Distance
    rho = (1.0 - ecc**2.0) / (1.0 + ecc * np.cos(lambda_ - om))

    # Insolation
    sol = (
        scale
        * (
            np.sin(np.pi / 180.0 * lat[None, ...]) * np.sin(dec)
            - np.cos(np.pi / 180.0 * lat[None, ...]) * np.cos(dec) * np.cos(h)
        )
        * rho**-2.0
    )
    if clip_zero:
        sol[sol < 0.0] = 0.0

    return sol.astype(np.float32)
