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

import asyncio
import csv
import os
import shutil
import sys
from datetime import timedelta

import numpy as np
import zarr
from earth2studio.data import HRRR, HRRR_FX
from loguru import logger

ENDPOINT_URL = "https://pdx.s8k.io"


def _append_missing_dates_to_csv(datetimes: np.ndarray, csv_path: str) -> None:
    """Append an array of numpy datetime64 values to a CSV file.

    Creates the file with a header if it does not exist.
    """
    file_exists = os.path.exists(csv_path)
    # Ensure 1-D array iteration
    flat_times = np.asarray(datetimes).ravel()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["datetime"])
        for t in flat_times:
            # Convert numpy datetime64 to ISO string (second precision)
            writer.writerow([np.datetime_as_string(t, unit="s")])


def init_zarr_store(zarr_root: str, time: np.ndarray, variable: str):
    """Initizes the HRRR Zarr store for training, will overwrite the coordinates every
    run but will not overwrite variable arrays if they exist. Each variable will be in
    a seperate array and chunked along the time dimension


    Parameters
    ----------
    zarr_path : str
        Location of zarr store
    time : np.ndarray
        Time array
    variable : str
        Variables to create arrays for
    """
    zarr_root = zarr.open_group(
        zarr_root, mode="a", storage_options={"endpoint_url": ENDPOINT_URL}
    )
    lat, lon = HRRR.grid()

    # Write the coordinates
    zarr_root.create_array(
        "time",
        shape=time.shape,
        chunks=time.shape,
        dtype=time.dtype,
        dimension_names=["time"],
        overwrite=True,
    )
    zarr_root["time"][:] = time
    zarr_root.create_array(
        "hrrr_y",
        shape=(lat.shape[0],),
        dtype=lat.dtype,
        dimension_names=["hrrr_y"],
        overwrite=True,
    )
    zarr_root["hrrr_y"][:] = np.arange(lat.shape[0])
    zarr_root.create_array(
        "hrrr_x",
        shape=(lat.shape[1],),
        dtype=lat.dtype,
        dimension_names=["hrrr_x"],
        overwrite=True,
    )
    zarr_root["hrrr_x"][:] = np.arange(lat.shape[1])

    zarr_root.create_array(
        "lat",
        shape=lat.shape,
        dtype=lat.dtype,
        dimension_names=["hrrr_y", "hrrr_x"],
        overwrite=True,
    )
    zarr_root["lat"][:] = lat
    zarr_root.create_array(
        "lon",
        shape=lon.shape,
        dtype=lon.dtype,
        dimension_names=["hrrr_y", "hrrr_x"],
        overwrite=True,
    )
    zarr_root["lon"][:] = lon

    # Set up chunking of variable arrays
    shape = (time.size, lat.shape[0], lat.shape[1])
    chunks = (128, lat.shape[0], lat.shape[1])  # adjust as needed

    # Create 3D variable arrays (time, hrrr_y, hrrr_x)
    for v in variable:
        try:
            zarr_root.create_array(
                v,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                fill_value=np.nan,
                dimension_names=["time", "hrrr_y", "hrrr_x"],
            )
        except zarr.errors.ContainsArrayError:
            logger.debug(f"Array {v} alread exists!")
            pass


async def pull_hrrr_data(
    zarr_root: str, time: np.ndarray, variable: str, batch_size: int = 1
):
    root = await zarr.api.asynchronous.open_group(
        zarr_root, mode="a", storage_options={"endpoint_url": ENDPOINT_URL}
    )
    zarr_time_coord = await (await root.get("time")).getitem(slice(None))
    ds = HRRR(verbose=False, cache=False)
    # Batch over times present
    for sidx in range(0, len(time), batch_size):
        eidx = min([time.shape[0] - 1, sidx + batch_size])
        logger.warning(f"Fetching times between {time[sidx]} to {time[eidx]}")
        batch_t = time[sidx : eidx + 1]

        retries = 4
        for attempt in range(retries + 1):
            try:
                da = await ds.fetch(batch_t, variable)
                shutil.rmtree(ds.cache)
            except Exception:
                logger.error("Failed to download data, re-attempting...")
                if attempt < retries:
                    await asyncio.sleep(4 * attempt + 1)
                else:
                    raise

        time_indices = np.where(np.isin(zarr_time_coord, da["time"].values))[0]

        # Save to zarr store using async open_array and concurrent writes
        logger.warning("Uploading data (async)")
        store = root.store

        async def _write_one(i: int, j: int, t: int, v: str):
            retries = 4
            for attempt in range(retries + 1):
                try:
                    arr = await root.get(v)
                    await arr.setitem((t, slice(None), slice(None)), da.values[i, j])
                except Exception as e:
                    logger.error(f"Write failed {e}, re-attempting...")
                    if attempt < retries:
                        await asyncio.sleep(4)
                    else:
                        raise

        jobs = []
        for i, t in enumerate(time_indices):
            for j, v in enumerate(variable):
                jobs.append(_write_one(i, j, t, v))

        await asyncio.gather(*jobs)


async def pull_hrrr_fx_data(
    zarr_root: str, time: np.ndarray, variable: str, batch_size: int = 1
):
    root = await zarr.api.asynchronous.open_group(
        zarr_root, mode="a", storage_options={"endpoint_url": ENDPOINT_URL}
    )
    zarr_time_coord = await (await root.get("time")).getitem(slice(None))
    ds = HRRR_FX(verbose=False, cache=False)
    # Batch over times present
    for sidx in range(0, len(time), batch_size):
        eidx = min([time.shape[0] - 1, sidx + batch_size])
        logger.warning(f"Fetching times between {time[sidx]} to {time[eidx]}")
        batch_t = time[sidx : eidx + 1]

        try:
            da = (await ds.fetch(batch_t, timedelta(hours=1), variable)).isel(
                lead_time=0
            )
        except FileNotFoundError as exc:
            logger.warning(
                f"File not found for times {batch_t[0]} to {batch_t[-1]} - recording and continuing. Details: {exc}"
            )
            _append_missing_dates_to_csv(batch_t, "missing_dates_hrrr.csv")
            continue
        time_indices = np.where(np.isin(zarr_time_coord, da["time"].values))[0]

        # Save to zarr store using async open_array and concurrent writes
        logger.warning("Uploading data (async)")
        store = root.store

        async def _write_one(i: int, j: int, t: int, v: str):
            retries = 4
            for attempt in range(retries + 1):
                try:
                    arr = await root.get(v)
                    await arr.setitem((t, slice(None), slice(None)), da.values[i, j])
                    return
                except Exception:
                    logger.error("Write failed, re-attempting...")
                    if attempt < retries:
                        await asyncio.sleep(4)
                    else:
                        raise

        jobs = []
        for i, t in enumerate(time_indices):
            for j, v in enumerate(variable):
                jobs.append(_write_one(i, j, t, v))

        await asyncio.gather(*jobs)


async def main():
    store = "s3://hrrr-surface-sda/zarr-v2"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("output.log", level="INFO")
    hrrr_variables = [
        "u10m",
        "v10m",
        "u80m",
        "v80m",
        "t2m",
        "d2m",
        "q2m",
        "sp",
        "fg10m",
        "tcc",
        "sde",
        "snowc",
        "refc",
        "rsds",
    ]
    hrrr_fx_variables = ["tp", "aerot"]
    variable = hrrr_variables + hrrr_fx_variables
    time = np.arange(
        "2017-01-01T00:00:00", "2027-01-01T00:00:00", dtype="datetime64[h]"
    )

    init_zarr_store(store, time, hrrr_variables)

    # Set start time
    start_date = np.datetime64("2021-01-01T00:00:00")
    end_date = np.datetime64("2021-06-01T00:00:00")
    sidx = np.where(time == start_date)[0][0]
    eidx = np.where(time == end_date)[0][0]
    step = 24 * 5  # 5-day batches (in hours)
    for bstart in range(sidx, eidx + 1, step):
        bend = min(eidx, bstart + step - 1)
        batch_time = time[bstart : bend + 1]
        (await pull_hrrr_data(store, batch_time, hrrr_variables, batch_size=12),)
        await pull_hrrr_fx_data(store, batch_time, hrrr_fx_variables, batch_size=12)


if __name__ == "__main__":
    asyncio.run(main())
