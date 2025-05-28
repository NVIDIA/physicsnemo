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

import os
import datetime
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import hydra
from omegaconf import DictConfig
import logging
import numpy as np
import matplotlib.pyplot as plt

from era5_mirror import ERA5Mirror


@hydra.main(version_base="1.2", config_path="conf", config_name="config_tas")
def main(cfg: DictConfig) -> None:
    # Make mirror data
    logging.getLogger().setLevel(logging.ERROR)  # Suppress logging from cdsapi
    mirror = ERA5Mirror(base_path=cfg.zarr_store_path)

    # split the years into train, validation, and test
    train_years = list(range(cfg.start_train_year, cfg.end_train_year + 1))
    test_years = cfg.test_years
    out_of_sample_years = cfg.out_of_sample_years
    all_years = train_years + test_years + out_of_sample_years

    # Set the variables to download for 34 var dataset
    date_range = (
        datetime.date(min(all_years), 1, 1),
        datetime.date(max(all_years), 12, 31),
    )
    hours = [cfg.dt * i for i in range(0, 24 // cfg.dt)]

    # Start the mirror
    zarr_paths = mirror.download(cfg.variables, date_range, hours)

    # Open the zarr files and construct the xarray from them
    zarr_arrays = [xr.open_zarr(path) for path in zarr_paths]
    era5_xarray = xr.concat(
        [z[list(z.data_vars.keys())[0]] for z in zarr_arrays], dim="channel"
    )
    era5_xarray = era5_xarray.transpose("time", "channel", "latitude", "longitude")
    era5_xarray.name = "fields"
    era5_xarray = era5_xarray.astype("float32")

    # Save mean and std
    if cfg.compute_mean_std:
        train_era5_xarray = era5_xarray.sel(
            time=era5_xarray.time.dt.year.isin(train_years)
        )
        stats_path = os.path.join(cfg.hdf5_store_path, "stats")
        print(f"Saving global mean and std at {stats_path}")
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
        era5_mean = np.array(
            train_era5_xarray.mean(dim=("time", "latitude", "longitude")).values
        )
        np.save(
            os.path.join(stats_path, "global_means.npy"), era5_mean.reshape(1, -1, 1, 1)
        )
        era5_std = np.array(
            train_era5_xarray.std(dim=("time", "latitude", "longitude")).values
        )
        np.save(
            os.path.join(stats_path, "global_stds.npy"), era5_std.reshape(1, -1, 1, 1)
        )
        print(f"Finished saving global mean and std at {stats_path}")

    # Make hdf5 files
    for year in all_years:
        # HDF5 filename
        split = (
            "train"
            if year in train_years
            else "test"
            if year in test_years
            else "out_of_sample"
        )
        hdf5_path = os.path.join(cfg.hdf5_store_path, split)
        os.makedirs(hdf5_path, exist_ok=True)
        hdf5_path = os.path.join(hdf5_path, f"{year}.h5")

        # Save year using dask
        print(f"Saving {year} at {hdf5_path}")
        with dask.config.set(
            scheduler="threads",
            num_workers=8,
            threads_per_worker=2,
            **{"array.slicing.split_large_chunks": False},
        ):
            with ProgressBar():
                # Get data for the current year
                year_data = era5_xarray.sel(time=era5_xarray.time.dt.year == year)

                # Save data to a temporary local file
                year_data.to_netcdf(hdf5_path, engine="h5netcdf", compute=True)
        print(f"Finished Saving {year} at {hdf5_path}")


if __name__ == "__main__":
    main()
