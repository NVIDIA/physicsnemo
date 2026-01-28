#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess WeatherBench2 ERA5 climatology to HEALPix format.

Downloads from: gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr
Regrids to HPX level 6 (49,152 cells, ~1° resolution).

Usage:
    python preprocess_climatology.py --out /path/to/era5_climatology_hpx64.zarr
"""

import argparse
import os
import shutil
from multiprocessing import Pool

import earth2grid
import numpy as np
import torch
import xarray as xr
import zarr
from earth2grid import healpix
from tqdm import tqdm

# WeatherBench2 ERA5 climatology (public)
SRC_ZARR = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
)
HPX_LEVEL = 6
HPX_CELLS = 12 * (4**HPX_LEVEL)  # 49,152
TIME_CHUNK_SIZE = 12

# Field mappings: target name -> source name in WeatherBench2
# Only include fields needed for scoring
FIELD_MAP = {
    # Surface fields
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "tcwv": "total_column_water_vapour",
}

# 3D fields: base -> source name
FIELD_3D_BASES = {
    "T": "temperature",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "Z": "geopotential",
    "Q": "specific_humidity",
}

# Default pressure levels
DEFAULT_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def open_climatology():
    ds = xr.open_zarr(SRC_ZARR, chunks={}).rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    for v in ds.data_vars:
        if ds[v].dtype != np.float32:
            ds[v] = ds[v].astype(np.float32)
    # Stack to single time axis: (dayofyear, hour) -> time
    ds = ds.sortby(["dayofyear", "hour"]).stack(time=("dayofyear", "hour"))
    ds = ds.assign_coords(
        time=("time", np.arange(ds.sizes["time"], dtype=np.int32)),
        dayofyear=("time", ds["dayofyear"].values.astype(np.int32)),
        hour=("time", ds["hour"].values.astype(np.int32)),
    )
    return ds


def build_regridder(nlat, nlon):
    ll = earth2grid.latlon.equiangular_lat_lon_grid(nlat=nlat, nlon=nlon)
    hpx = healpix.Grid(level=HPX_LEVEL, pixel_order=healpix.PixelOrder.NEST)
    return earth2grid.get_regridder(ll, hpx).to(torch.float32)


def regrid_batch(batch, regridder):
    batch_tensor = torch.from_numpy(batch)
    return regridder(batch_tensor).float().cpu().numpy()


def process_field(args):
    out_path, field_name, src_name, level = args
    ds = open_climatology()
    nlat, nlon = ds.sizes["lat"], ds.sizes["lon"]
    regridder = build_regridder(nlat, nlon)

    g = zarr.open_group(out_path, mode="a")
    arr = g[field_name]

    da = ds[src_name] if level is None else ds[src_name].sel(level=int(level))
    da = da.transpose("time", "lat", "lon").astype("float32")

    ntime = da.sizes["time"]
    for t0 in tqdm(range(0, ntime, TIME_CHUNK_SIZE), desc=f"{field_name}", leave=False):
        t1 = min(t0 + TIME_CHUNK_SIZE, ntime)
        batch = np.asarray(da.isel(time=slice(t0, t1)).data)
        out = regrid_batch(batch, regridder)
        arr[t0:t1, :] = out

    return field_name


def main(out_path: str, levels: list[int]):
    print(f"Opening climatology from {SRC_ZARR}...")
    ds = open_climatology()
    ntime = ds.sizes["time"]

    # Get available levels
    avail_levels = None
    for src in FIELD_3D_BASES.values():
        if src in ds and "level" in ds[src].dims:
            avail_levels = ds[src]["level"].values.astype(np.int32).tolist()
            break
    keep_levels = [lv for lv in levels if avail_levels and lv in avail_levels]

    # Build channel list
    channels = []
    for tgt, src in FIELD_MAP.items():
        if src in ds:
            channels.append((tgt, src, None))
    for base, src in FIELD_3D_BASES.items():
        if src in ds and "level" in ds[src].dims:
            channels.extend((f"{base}{level}", src, level) for level in keep_levels)

    print(f"Processing {len(channels)} fields...")

    # Create output zarr
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    g = zarr.open_group(out_path, mode="w")

    for name, _, _ in channels:
        g.create_array(
            name,
            shape=(ntime, HPX_CELLS),
            chunks=(TIME_CHUNK_SIZE, HPX_CELLS),
            fill_value=np.nan,
            dtype="f4",
        )

    g.create_array("time", dtype="i4", shape=(ntime,), chunks=(ntime,))[:] = ds[
        "time"
    ].values
    g.create_array("dayofyear", dtype="i4", shape=(ntime,), chunks=(ntime,))[:] = ds[
        "dayofyear"
    ].values
    g.create_array("hour", dtype="i4", shape=(ntime,), chunks=(ntime,))[:] = ds[
        "hour"
    ].values
    g.create_array("cells", dtype="i4", shape=(HPX_CELLS,), chunks=(HPX_CELLS,))[:] = (
        np.arange(HPX_CELLS)
    )

    tasks = [(out_path, name, src, level) for name, src, level in channels]
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_field, tasks), total=len(tasks)):
            pass

    zarr.consolidate_metadata(g.store)
    print(f"Done! Wrote {out_path} with {len(channels)} fields.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ERA5 climatology to HEALPix"
    )
    parser.add_argument("--out", required=True, help="Output zarr path")
    parser.add_argument(
        "--levels", type=int, nargs="+", default=DEFAULT_LEVELS, help="Pressure levels"
    )
    args = parser.parse_args()
    main(args.out, args.levels)
