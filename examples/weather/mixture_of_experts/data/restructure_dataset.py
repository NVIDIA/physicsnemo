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

import xarray as xr
import numpy as np
import h5py
import os
import dask.array as da
from dask.diagnostics import ProgressBar
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    # Path to your Zarr dataset
    zarr_path = cfg.io.file_name

    # Path to output HDF5 file
    h5_dir_path = cfg.h5_dir_path
    # Open Zarr dataset
    ds = xr.open_zarr(zarr_path, chunks={"time": 4})
    ds = ds.drop_vars("timesteps_completed")
    # Drop lead time dimension
    ds = ds.isel(lead_time=1).drop_vars("lead_time")

    os.makedirs(h5_dir_path, exist_ok=True)
    os.makedirs(os.path.join(h5_dir_path, "train"), exist_ok=True)
    for year in tqdm(range(1980, 2016), desc="Processing years"):
        time_range = pd.date_range(
            f"{year}-01-01T06:00:00", f"{year}-12-31T18:00:00", freq="6h"
        )
        ds_year = ds.sel(time=time_range)

        # Get all data variables as a list of DataArrays
        data_vars = list(ds_year.data_vars)

        arrays = [
            ds_year[var].data for var in data_vars
        ]  # Each is a dask array [T, H, W]

        # Stack along a new axis (channels)
        fields = da.stack(arrays, axis=1)  # [T, C, H, W]

        # Write to HDF5
        h5_path = os.path.join(h5_dir_path, "train", f"{year}.h5")

        with ProgressBar():
            da.to_hdf5(h5_path, "/fields", fields)

        print(f"Saved HDF5 file to {h5_path}")


if __name__ == "__main__":
    main()
