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

import datetime
from pathlib import Path
from typing import List, Tuple, Union

import dask
import fsspec
import hydra
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from omegaconf import DictConfig, OmegaConf

# Add eval to OmegaConf TODO: Remove when OmegaConf is updated
OmegaConf.register_new_resolver("eval", eval)

from utils import get_filesystem
from transform.transform import transform_registry


class CurateERA5:
    """
    Curate a Zarr ERA5 dataset to a Zarr dataset used for training global weather models.
    """

    def __init__(
        self,
        unpredicted_variables: List[str],
        predicted_variables: List[str],
        dataset_filename: Union[str, Path] = "./data.zarr",
        fs: fsspec.filesystem = fsspec.filesystem("file"),
        curated_dataset_filename: Union[str, Path] = "./curated_data.zarr",
        curated_fs: fsspec.filesystem = fsspec.filesystem("file"),
        transform: None = None,
        date_range: Tuple[str, str] = ("2000-01-01", "2001-01-01"),
        dt: int = 1,  # 1 hour
        chunk_channels_together: bool = True,
        single_threaded: bool = False,
    ):
        super().__init__()

        # Store parameters
        self.unpredicted_variables = unpredicted_variables
        self.predicted_variables = predicted_variables
        self.dataset_filename = dataset_filename
        self.fs = fs
        self.curated_dataset_filename = curated_dataset_filename
        self.curated_fs = curated_fs
        self.transform = transform
        self.date_range = date_range
        assert dt in [1, 3, 6, 12], "dt must be 1, 3, 6, or 12"
        self.dt = dt
        self.chunk_channels_together = chunk_channels_together
        self.single_threaded = single_threaded

        # Open dataset to do curation from
        mapper = fs.get_mapper(self.dataset_filename)
        self.era5 = xr.open_zarr(mapper, consolidated=True)

        # Subset variables (this speeds up chunking)
        needed_variables = ["latitude", "longitude", "time", "level"]
        for variable in self.unpredicted_variables + self.predicted_variables:
            if not isinstance(variable, str):
                needed_variables.append(variable[0])
            else:
                needed_variables.append(variable)
        for variable in self.era5.variables:
            if variable not in needed_variables:
                self.era5 = self.era5.drop_vars(variable)

        # Chunk data
        self.era5 = self.era5.sel(
            time=slice(
                datetime.datetime.strptime(date_range[0], "%Y-%m-%d"),
                datetime.datetime.strptime(date_range[1], "%Y-%m-%d"),
            )
        )
        self.era5 = self.era5.sel(
            time=self.era5.time.dt.hour.isin(np.arange(0, 24, self.dt))
        )
        self.era5 = self.era5.chunk(
            {"time": 1, "level": 1, "latitude": 721, "longitude": 1440}
        )

        # Gather all predicted variables
        xarray_predicted_variables = []
        for variable in self.predicted_variables:
            if not isinstance(variable, str):  # TODO: better way to check if list
                pressure_variable = self.era5[variable[0]].sel(level=variable[1])
                pressure_variable = pressure_variable.drop("level")
                pressure_variable = pressure_variable.rename(
                    {"level": "predicted_channel"}
                )
                xarray_predicted_variables.append(pressure_variable)
            else:
                single_variable = self.era5[variable]
                single_variable = single_variable.expand_dims(
                    "predicted_channel", axis=1
                )
                xarray_predicted_variables.append(single_variable)

        # Gather all unpredicted variables
        xarray_unpredicted_variables = []
        for variable in self.unpredicted_variables:
            if not isinstance(variable, str):  # TODO: better way to check if list
                pressure_variable = self.era5[variable[0]].sel(level=variable[1])
                pressure_variable = pressure_variable.drop("level")
                pressure_variable = pressure_variable.rename(
                    {"level": "unpredicted_channel"}
                )
                xarray_unpredicted_variables.append(pressure_variable)
            else:
                single_variable = self.era5[variable]
                single_variable = single_variable.expand_dims(
                    "unpredicted_channel", axis=1
                )
                xarray_unpredicted_variables.append(single_variable)

        # Concatenate all variables
        self.era5_subset = xr.Dataset()
        self.era5_subset["predicted"] = xr.concat(
            xarray_predicted_variables, dim="predicted_channel"
        )
        self.era5_subset["unpredicted"] = xr.concat(
            xarray_unpredicted_variables, dim="unpredicted_channel"
        )
        self.era5_subset["time"] = self.era5["time"]

        # Chunk channels
        if self.chunk_channels_together:
            predicted_channel_chunk_size = self.era5_subset.predicted_channel.size
            unpredicted_channel_chunk_size = self.era5_subset.unpredicted_channel.size
        else:
            predicted_channel_chunk_size = 1
            unpredicted_channel_chunk_size = 1
        self.era5_subset = self.era5_subset.chunk(
            {
                "time": 1,
                "predicted_channel": predicted_channel_chunk_size,
                "unpredicted_channel": unpredicted_channel_chunk_size,
            }
        )

    def __call__(self):
        """
        Generate the zarr array
        """

        # Check if already exists
        if self.fs.exists(self.curated_dataset_filename):
            print(f"Zarr file {self.curated_dataset_filename} already exists")
            return

        # Run transform if specified
        if self.transform is not None:
            self.era5_subset = self.transform(self.era5_subset)

        # Save
        mapper = self.fs.get_mapper(self.curated_dataset_filename)
        delayed_obj = self.era5_subset.to_zarr(mapper, consolidated=True, compute=False)

        # Wait for save to finish (Single-threaded legacy issue)
        with ProgressBar():
            if self.single_threaded:
                with dask.config.set(scheduler="single-threaded"):
                    delayed_obj.compute()
            else:
                delayed_obj.compute()


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve config so that all values are concrete
    OmegaConf.resolve(cfg)

    # Get transform function
    try:
        transform = transform_registry[cfg.transform.name]
    except KeyError:
        raise NotImplementedError(f"Transform {cfg.transform.name} not implemented")
    if "kwargs" in cfg.transform:

        def wrapper_transform(transform, **kwargs):
            def _transform(x):
                return transform(x, **kwargs)

            return _transform

        transform = wrapper_transform(transform, **cfg.transform.kwargs)

    # Get filesystem
    fs = get_filesystem(
        cfg.filesystem.type,
        cfg.filesystem.key,
        cfg.filesystem.endpoint_url,
        cfg.filesystem.region_name,
    )

    # Make train data
    curate_train_era5 = CurateERA5(
        unpredicted_variables=cfg.curated_dataset.unpredicted_variables,
        predicted_variables=cfg.curated_dataset.predicted_variables,
        dataset_filename=cfg.dataset.dataset_filename,
        fs=fs,
        curated_dataset_filename=cfg.curated_dataset.train_dataset_filename,
        curated_fs=fs,
        transform=transform,
        date_range=cfg.curated_dataset.train_years,
        dt=cfg.curated_dataset.dt,
        chunk_channels_together=cfg.curated_dataset.chunk_channels_together,
    )
    curate_train_era5()

    # Make validation data
    curate_val_era5 = CurateERA5(
        unpredicted_variables=cfg.curated_dataset.unpredicted_variables,
        predicted_variables=cfg.curated_dataset.predicted_variables,
        dataset_filename=cfg.dataset.dataset_filename,
        fs=fs,
        curated_dataset_filename=cfg.curated_dataset.val_dataset_filename,
        curated_fs=fs,
        transform=transform,
        date_range=cfg.curated_dataset.val_years,
        dt=cfg.curated_dataset.dt,
        chunk_channels_together=cfg.curated_dataset.chunk_channels_together,
    )
    curate_val_era5()


if __name__ == "__main__":
    main()
