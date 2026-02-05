# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

import dask
import numpy as np
import xarray as xr


# Downsample transform
def downsample_transform(dataset, downsample_factor=4):
    """
    Downsample the dataset by a factor of downsample_factor

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to downsample
    downsample_factor : int
        The factor to downsample by
    """

    dataset = dataset.coarsen(
        {"latitude": downsample_factor, "longitude": downsample_factor}, boundary="trim"
    ).mean()
    return dataset


# Trim lat from 721 to 720
def trim_lat720_transform(dataset):
    """
    Trim the latitude from 721 to 720

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to trim
    """

    dataset = dataset.isel(latitude=slice(0, -1))
    return dataset


# Helpix transform
# Reference:
# https://github.com/nathanielcresswellclay/zephyr/blob/main/data_processing/remap/healpix.py
def healpix_transform(dataset, nside=32, order="bilinear"):
    """
    Transform the dataset to healpix grid

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to transform
    nside : int
        The nside of the healpix grid
    order : str
        The order of the interpolation
    """

    # Get the lat/lon coordinates
    lon = len(dataset.longitude.values)
    lat = len(dataset.latitude.values)
    res = 360.0 / lon

    # Create the healpix grid
    wcs_input_dict = {
        "CTYPE1": "RA---CAR",  # can be further specified with, e.g., RA---MOL, GLON-MOL, ELON-MOL
        "CUNIT1": "deg",
        "CDELT1": -res,  # -r produces for some reason less NaNs
        "CRPIX1": lon / 2.0,
        "CRVAL1": 180.0,
        "NAXIS1": lon,  # does not seem to have an effect
        "CTYPE2": "DEC--CAR",  # can be further specified with, e.g., DEC--MOL, GLAT-MOL, ELAT-MOL
        "CUNIT2": "deg",
        "CDELT2": -res,
        "CRPIX2": (lat + 1) / 2.0,
        "CRVAL2": 0.0,
        "NAXIS2": lat,
    }
    wcs_ll2hpx = ap.wcs.WCS(wcs_input_dict)

    # Create healpix to 1d to 3d index mapping
    index_mapping_1d_to_3d = np.zeros((hp.nside2npix(nside), 3), dtype=np.int64)
    for i in range(hp.nside2npix(nside)):
        f = i // (nside * nside)
        hpxidx = format(i % (nside * nside), "b").zfill(nside)
        bits_even = hpxidx[::2]
        bits_odd = hpxidx[1::2]
        y = int(bits_even, 2)
        x = int(bits_odd, 2)
        index_mapping_1d_to_3d[i, :] = [f, x, y]

    # Create numpy remapping function
    def remap_func(x):
        # Use dask delayed to parallelize the remapping
        # while keeping the xarray structure
        @dask.delayed
        def _remap_func(x):
            # Convert to healpix grid
            x = x.values
            x = np.flip(x, axis=1)
            hpx1d, _ = rp.reproject_to_healpix(
                (x, wcs_ll2hpx),
                coord_system_out="icrs",
                nside=nside,
                order=order,
                nested=True,
            )
            hpx3d = np.zeros((12, nside, nside), dtype=x.dtype)
            hpx3d[
                index_mapping_1d_to_3d[:, 0],
                index_mapping_1d_to_3d[:, 1],
                index_mapping_1d_to_3d[:, 2],
            ] = hpx1d

            return hpx3d

        # Run the delayed function
        old_x = x
        x = _remap_func(x)

        # Convert back to dask array
        x = dask.array.from_delayed(x, shape=(12, nside, nside), dtype=old_x.dtype)

        # Convert back to xarray
        x = xr.DataArray(x, dims=["face", "nside_x", "nside_y"])

        return x

    # Apply the remapping function
    dataset["predicted"] = (
        dataset["predicted"]
        .groupby("time")
        .map(lambda x: x.groupby("predicted_channel").map(lambda y: remap_func(y)))
    )
    dataset["unpredicted"] = (
        dataset["unpredicted"]
        .groupby("time")
        .map(lambda x: x.groupby("unpredicted_channel").map(lambda y: remap_func(y)))
    )

    return dataset


transform_registry = {
    "downsample": downsample_transform,
    "trim_lat720": trim_lat720_transform,
}

try:
    import astropy as ap
    import healpy as hp
    import reproject as rp

    transform_registry["healpix"] = healpix_transform
except:
    import warnings

    warnings.warn("Unable to import healpix transform. Skipping...")
