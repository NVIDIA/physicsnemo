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
import tempfile
import cdsapi
import xarray as xr
import datetime
import json
import dask
import calendar
from dask.diagnostics import ProgressBar
from typing import List, Tuple, Dict, Union
import urllib3
import logging
import numpy as np
import fsspec

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ERA5Mirror:
    """
    A class to manage downloading ERA5 datasets. The datasets are downloaded from the Copernicus Climate Data Store (CDS) and stored in Zarr format.

    Attributes
    ----------
    base_path : Path
        The path to the Zarr dataset.
    fs : fsspec.AbstractFileSystem
        The filesystem to use for the Zarr dataset. If None, the local filesystem will be used.
    """

    def __init__(self, base_path: str, fs: fsspec.AbstractFileSystem = None):
        # Get parameters
        self.base_path = base_path
        if fs is None:
            fs = fsspec.filesystem("file")
        self.fs = fs

        # Create the base path if it doesn't exist
        if not self.fs.exists(self.base_path):
            self.fs.makedirs(self.base_path)

        # Create metadata that will be used to track which chunks have been downloaded
        self.metadata_file = os.path.join(self.base_path, "metadata.json")
        self.metadata = self.get_metadata()

    def get_metadata(self):
        """Get metadata"""
        if self.fs.exists(self.metadata_file):
            with self.fs.open(self.metadata_file, "r") as f:
                try:
                    metadata = json.load(f)
                except json.decoder.JSONDecodeError:
                    metadata = {"chunks": []}
        else:
            metadata = {"chunks": []}
        return metadata

    def save_metadata(self):
        """Save metadata"""
        with self.fs.open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def chunk_exists(self, variable, year, month, hours, pressure_level):
        """Check if chunk exists"""
        for chunk in self.metadata["chunks"]:
            if (
                chunk["variable"] == variable
                and chunk["year"] == year
                and chunk["month"] == month
                and chunk["hours"] == hours
                and chunk["pressure_level"] == pressure_level
            ):
                return True
        return False

    def download_chunk(
        self,
        variable: str,
        year: int,
        month: int,
        hours: List[int],
        pressure_level: int = None,
    ):
        """
        Download ERA5 data for the specified variable, date range, hours, and pressure levels.

        Parameters
        ----------
        variable : str
            The ERA5 variable to download, e.g. 'tisr' for solar radiation or 'z' for geopotential.
        year : int
            The year to download.
        month : int
            The month to download.
        hours : List[int]
            A list of hours (0-23) for which data should be downloaded.
        pressure_level : int, optional
            A pressure level to include in the download, by default None. If None, the single-level data will be downloaded.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the downloaded data.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            # Get all days in the month
            days_in_month = calendar.monthrange(year, month)[1]

            # Make tmpfile to store the data
            output_file = os.path.join(
                tmpdir,
                f"{variable}_{year}_{month:02d}_{str(hours)}_{str(pressure_level)}.nc",
            )

            # start the CDS API client (maybe need to move this outside the loop?)
            c = cdsapi.Client(quiet=True)

            # Setup the request parameters
            request_params = {
                "product_type": "reanalysis",
                "variable": variable,
                "year": str(year),
                "month": str(month),
                "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
                "time": [f"{hour:02d}:00" for hour in hours],
                "format": "netcdf",
            }
            if pressure_level:
                request_params["pressure_level"] = [str(pressure_level)]
                dataset_name = "reanalysis-era5-pressure-levels"
            else:
                dataset_name = "reanalysis-era5-single-levels"

            # Download the data
            c.retrieve(
                dataset_name,
                request_params,
                output_file,
            )

            # Open the downloaded data
            ds = xr.open_dataset(output_file)
        return ds

    def variable_to_zarr_name(self, variable: str, pressure_level: int = None):
        """convert variable to zarr name"""
        # create zarr path for variable
        zarr_path = f"{self.base_path}/{variable}"
        if pressure_level:
            zarr_path += f"_pressure_level_{pressure_level}"
        zarr_path += ".zarr"
        return zarr_path

    def download_and_upload_chunk(
        self,
        variable: str,
        year: int,
        month: int,
        hours: List[int],
        pressure_level: int = None,
    ):
        """
        Downloads a chunk of ERA5 data for a specific variable and date range, and uploads it to a Zarr array.
        This downloads a 1-month chunk of data.

        Parameters
        ----------
        variable : str
            The variable to download.
        year : int
            The year to download.
        month : int
            The month to download.
        hours : List[int]
            A list of hours to download.
        pressure_level : int, optional
            Pressure levels to download, if applicable.
        """

        # Download the data
        ds = self.download_chunk(variable, year, month, hours, pressure_level)
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

        # Create the Zarr path
        zarr_path = self.variable_to_zarr_name(variable, pressure_level)

        # Specify the chunking options
        chunking = {"time": 1, "latitude": 721, "longitude": 1440}
        if "level" in ds.dims:
            chunking["level"] = 1

        # Re-chunk the dataset
        ds = ds.chunk(chunking)

        # Check if the Zarr dataset exists
        if self.fs.exists(zarr_path):
            mode = "a"
            append_dim = "time"
            create = False
        else:
            mode = "w"
            append_dim = None
            create = True

        # Upload the data to the Zarr dataset
        mapper = self.fs.get_mapper(zarr_path, create=create)
        ds.to_zarr(mapper, mode=mode, consolidated=True, append_dim=append_dim)

        # Update the metadata
        self.metadata["chunks"].append(
            {
                "variable": variable,
                "year": year,
                "month": month,
                "hours": hours,
                "pressure_level": pressure_level,
            }
        )
        self.save_metadata()

    def download(
        self,
        variables: List[Union[str, Tuple[str, int]]],
        date_range: Tuple[datetime.date, datetime.date],
        hours: List[int],
    ):
        """
        Start the process of mirroring the specified ERA5 variables for the given date range and hours.

        Parameters
        ----------
        variables : List[Union[str, Tuple[str, List[int]]]]
            A list of variables to mirror, where each element can either be a string (single-level variable)
            or a tuple (variable with pressure level).
        date_range : Tuple[datetime.date, datetime.date]
            A tuple containing the start and end dates for the data to be mirrored. This will download and store every month in the range.
        hours : List[int]
            A list of hours for which to download the data.

        Returns
        -------
        zarr_paths : List[str]
            A list of Zarr paths for each of the variables.
        """

        start_date, end_date = date_range

        # Reformat the variables list so all elements are tuples
        reformated_variables = []
        for variable in variables:
            if isinstance(variable, str):
                reformated_variables.append(tuple([variable, None]))
            else:
                reformated_variables.append(variable)

        # Start Downloading
        with ProgressBar():
            # Round dates to months
            current_date = start_date.replace(day=1)
            end_date = end_date.replace(day=1)

            while current_date <= end_date:
                # Create a list of tasks to download the data
                tasks = []
                for variable, pressure_level in reformated_variables:
                    if not self.chunk_exists(
                        variable,
                        current_date.year,
                        current_date.month,
                        hours,
                        pressure_level,
                    ):
                        task = dask.delayed(self.download_and_upload_chunk)(
                            variable,
                            current_date.year,
                            current_date.month,
                            hours,
                            pressure_level,
                        )
                        tasks.append(task)
                    else:
                        print(
                            f"Chunk for {variable} {pressure_level} {current_date.year}-{current_date.month} already exists. Skipping."
                        )

                # Execute the tasks with Dask
                print(f"Downloading data for {current_date.year}-{current_date.month}")
                if tasks:
                    dask.compute(*tasks)

                # Update the metadata
                self.save_metadata()

                # Update the current date
                days_in_month = calendar.monthrange(
                    year=current_date.year, month=current_date.month
                )[1]
                current_date += datetime.timedelta(days=days_in_month)

        # Return the Zarr paths
        zarr_paths = []
        for variable, pressure_level in reformated_variables:
            zarr_path = self.variable_to_zarr_name(variable, pressure_level)
            zarr_paths.append(zarr_path)

        # Check that Zarr arrays have correct dt for time dimension
        for zarr_path in zarr_paths:
            ds = xr.open_zarr(zarr_path)
            time_stamps = ds.time.values
            dt = time_stamps[1:] - time_stamps[:-1]
            assert np.all(
                dt == dt[0]
            ), f"Zarr array {zarr_path} has incorrect dt for time dimension. An error may have occurred during download. Please delete the Zarr array and try again."

        return zarr_paths
