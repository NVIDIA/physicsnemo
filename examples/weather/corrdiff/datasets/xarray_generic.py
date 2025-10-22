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
from typing import Any

import cftime
import json
import numpy as np
import xarray as xr

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.base import ChannelMetadata, DownscalingDataset


class XarrayDataset(DownscalingDataset):
    """
    Reader for generic Xarray dataset for CorrDiff training. It can also be used as a
    basis for custom dataset implementations that require custom logic, either by
    copy-pasting the code or by subclassing.

    You can use the create_sample_dataset function to generate sample and statistics
    files conforming to the expected format.

    Parameters
    ----------
    data_paths : str | list[str]
        Path(s) to the dataset(s). They can be in any format openable with
        xarray.open_dataset, e.g. Zarr or NetCDF4. The arrays in the datasets should be:
        - "input": the input samples
        - "output": the corresponding output samples
        - "invariant" (optional): the invariant data (e.g. elevation)
        - "lat" (optional): the latitude of each grid point in degrees
        - "lon" (optional): the longitude of each grid point in degrees
        with dimensions:
        - "input": ("time", "input_variable", "y", "x")
        - "output": ("time", "output_variable", "y", "x")
        - "invariant": ("invariant_variable", "y", "x")
        - "lat": ("y", "x")
        - "lon": ("y", "x")
        The coordinates should be as follows: - "time": the time of each sample, in a
        format decodable to numpy.datetime64.
        - "input_variable", "output_variable" and "invariant_variable": string arrays
          indicating the name of each variable.
        - "y": coordinates in the height dimension
        - "x": coordinates in the width dimension
        The files may have different numbers of samples(i.e. different "time"
        dimensions) and are not required to be chronologically ordered. The "x" and "y"
        dimensions must agree. The variables found in the files may differ, but if
        they do, an explicit variable list must be supplied using input_variables,
        output_variables and/or invariant_variables.
    stats_path : str
        Path to the normalization statistics JSON file. It should contain the mean and
        standard deviation of each variable in a dictionary as follows: {
            "input": {
                "variable1": {"mean": 2.5, "std": 0.3},
                "variable2": {"mean": -3.2, "std": 5.3},
                ...
            }, "output": {
                ...
            }, "invariant": {
                ...
            }
        }
    input_variables : list[str] | None, optional
        Input variable names to load. When None, loads all variables.
    output_variables : list[str] | None, optional
        Output variable names to load. When None, loads all variables.
    invariant_variables : list[str] | None, optional
        Invariant variable names to load. When None, loads all variables (pass an empty
        list to load *no* variables).
    load_to_memory : bool, optional, default False
        Preload all data into memory. This will enable very fast dataloading, but you
        may run out of memory with large datasets.
    open_dataset_kwargs : dict[str, Any], optional
        Additional keyword arguments for xarray.open_dataset. For example, {"engine":
        "zarr"} may be useful to get xarray to open zarr datasets.
    time_range : tuple[str, str] | None, optional
        When not None, return data only from the time period time_range[0] <= time <
        time_range[1]. Times must be strings in the format used by datetime.isoformat,
        e.g. "2025-10-26T03:00:00".
    exclude_times : list[str], optional
        Specific times to exclude, in the same format as time_range. Can be used e.g. to
        exclude bad data samples.
    shard : bool, optional, default True
        If enabled, each dataloader worker will load only a subset of data belonging
        exclusively to it. This improves caching performance and memory efficiency,
        especially when load_to_memory == True.
    """

    def __init__(
        self,
        data_paths: str | list[str],
        stats_path: str,
        input_variables: list[str] | None = None,
        output_variables: list[str] | None = None,
        invariant_variables: list[str] | None = None,
        load_to_memory: bool = False,
        open_dataset_kwargs: dict[str, Any] = {},
        time_range: tuple[str, str] | None = None,
        exclude_times: list[str] = [],
        shard: bool = True,
    ):
        # open files
        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        self.datasets = [
            xr.open_dataset(path, **open_dataset_kwargs) for path in data_paths
        ]

        # load inputs and outputs
        self.inputs = []
        self.outputs = []
        for i, ds in enumerate(self.datasets):
            (ip, invar) = _load_data(ds, "input", input_variables)
            (op, outvar) = _load_data(ds, "output", output_variables)
            self.inputs.append(ip)
            self.outputs.append(op)
            if i == 0:
                self.input_variables = invar
                self.output_variables = outvar
            else:
                if not (
                    np.array_equal(invar, self.input_variables)
                    and np.array_equal(outvar, self.output_variables)
                ):
                    raise ValueError(
                        "Data files contain different variables. Use explicit input_variables or output_variables to select a subset."
                    )

        # load invariants
        if invariant_variables or (invariant_variables is None):
            (self.invariants, self.invariant_variables) = _load_data(
                self.datasets[0], "invariant", invariant_variables
            )
            self.invariants = self.invariants.values
        else:
            self.invariants = []
            self.invariant_variables = []

        # load temporal and spatial coordinates
        self.times = np.concatenate([ds.coords["time"].values for ds in self.datasets])
        self.lat = self.datasets[0]["lat"].values if "lat" in self.datasets[0] else None
        self.lon = self.datasets[0]["lon"].values if "lon" in self.datasets[0] else None
        self.img_shape = self.inputs[0].shape[-2:]

        # load normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)
        (input_mean, input_std) = _load_stats(stats, self.input_variables, "input")
        (inv_mean, inv_std) = _load_stats(stats, self.invariant_variables, "invariant")
        self.input_mean = np.concatenate([input_mean, inv_mean], axis=0)
        self.input_std = np.concatenate([input_std, inv_std], axis=0)
        (self.output_mean, self.output_std) = _load_stats(
            stats, self.output_variables, "output"
        )

        # filter the samples to find the ones we actually use
        num_samples_per_file = [ip.shape[0] for ip in self.inputs]
        sample_indices = np.arange(sum(num_samples_per_file))
        if time_range is not None:  # filter to time_range[0] <= time < time_range[1]
            (t0, t1) = (np.datetime64(t, "s") for t in time_range)
            sample_indices = sample_indices[(t0 <= self.times) & (self.times < t1)]
        if exclude_times:  # exclude given times from dataset
            exclude_mask = ~np.isin(
                self.times[sample_indices], np.array(exclude_times, dtype="datetime64")
            )
            sample_indices = sample_indices[exclude_mask]
        if shard:  # use only a subset of dataset for this training process
            if not DistributedManager.is_initialized():
                DistributedManager.initialize()
            dist = DistributedManager()
            sample_indices = np.array_split(sample_indices, dist.world_size)[dist.rank]

        # select the samples we use from the datasets and map global index to dataset and local index
        self.times = self.times[sample_indices]
        dataset_boundaries = np.concatenate([[0], np.cumsum(num_samples_per_file)])
        self.global_idx_to_dataset = np.searchsorted(
            dataset_boundaries[1:], sample_indices, side="right"
        )
        indices_in_dataset = (
            sample_indices - dataset_boundaries[self.global_idx_to_dataset]
        )
        for i in range(len(self.inputs)):
            local_indices = indices_in_dataset[self.global_idx_to_dataset == i]
            self.inputs[i] = self.inputs[i][local_indices]
            self.outputs[i] = self.outputs[i][local_indices]
        self.global_idx_to_local_idx = np.concatenate(
            [np.arange(ip.shape[0]) for ip in self.inputs]
        )

        # preload all data from dataset to memory if selected
        self.load_to_memory = load_to_memory
        if load_to_memory:
            self.inputs = [ip.values for ip in self.inputs]
            self.outputs = [op.values for op in self.outputs]
            del self.datasets

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get data sample at index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Output and input arrays (y, x).
        """
        dataset_num = self.global_idx_to_dataset[idx]
        local_idx = self.global_idx_to_local_idx[idx]
        x = self.inputs[dataset_num][local_idx]
        y = self.outputs[dataset_num][local_idx]
        if not self.load_to_memory:
            x = x.values
            y = y.values
        if self.invariant_variables:
            x = np.concatenate([x, self.invariants], axis=0)

        x = self.normalize_input(x)
        y = self.normalize_output(y)

        return (y, x)

    def __len__(self):
        """
        Get number of samples.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.times)

    def longitude(self) -> np.ndarray:
        """
        Get longitude values.

        Returns
        -------
        np.ndarray
            Longitude array.
        """
        return np.full(self.img_shape, np.nan) if self.lon is None else self.lon

    def latitude(self) -> np.ndarray:
        """
        Get latitude values.

        Returns
        -------
        np.ndarray
            Latitude array.
        """
        return np.full(self.img_shape, np.nan) if self.lat is None else self.lat

    def input_channels(self) -> list[ChannelMetadata]:
        """
        Get input channel metadata.

        Returns
        -------
        list[ChannelMetadata]
            Metadata for each input channel.
        """
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        invariants = [
            ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables
        ]
        return inputs + invariants

    def output_channels(self) -> list[ChannelMetadata]:
        """
        Get output channel metadata.

        Returns
        -------
        list[ChannelMetadata]
            Metadata for each output channel.
        """
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> list[cftime.DatetimeGregorian]:
        """
        Get time values as cftime objects.

        Returns
        -------
        list[cftime.DatetimeGregorian]
            Time values.
        """
        datetimes = (
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) for t in self.times
        )
        return [convert_datetime_to_cftime(t) for t in datetimes]

    def image_shape(self) -> tuple[int, int]:
        """
        Get the (height, width) of the data (same for input and output).

        Returns
        -------
        tuple[int, int]
            Height and width of images.
        """
        return self.img_shape

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """
        Convert input from physical units to zero-mean, unit-variance normalized data.

        Parameters
        ----------
        x : np.ndarray
            Input in physical units.

        Returns
        -------
        np.ndarray
            Normalized input.
        """
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """
        Convert input from zero-mean, unit-variance normalized data to physical units.

        Parameters
        ----------
        x : np.ndarray
            Normalized input.

        Returns
        -------
        np.ndarray
            Input in physical units.
        """
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """
        Convert output from physical units to zero-mean, unit-variance normalized data.

        Parameters
        ----------
        x : np.ndarray
            Output in physical units.

        Returns
        -------
        np.ndarray
            Normalized output.
        """
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """
        Convert output from zero-mean, unit-variance normalized data to physical units.

        Parameters
        ----------
        x : np.ndarray
            Normalized output.

        Returns
        -------
        np.ndarray
            Output in physical units.
        """
        return x * self.output_std + self.output_mean


def _load_data(
    dataset: xr.Dataset, array: str, variables=None
) -> tuple[xr.DataArray, list[str]]:
    """
    Load subset of variables from an array in a Dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The xarray Dataset.
    array : str
        Name of the array to load.
    variables : list[str], optional
        Variable names to load. If None, loads all variables.

    Returns
    -------
    tuple[xarray.DataArray, list[str]]
        Data array and list of variable names.
    """
    var_coord = [v for v in list(dataset[array].coords) if v.endswith("_variable")][0]
    available_variables = list(dataset.coords[var_coord].values)
    if variables is None:
        var_indices = slice(None)  # load all variables
        variables = available_variables
    else:
        if variables == available_variables:
            var_indices = slice(None)
        else:
            try:
                var_indices = [available_variables.index(v) for v in variables]
            except ValueError:
                missing_vars = sorted(set(variables) - set(available_variables))
                raise ValueError(
                    f"Trying to load variable(s) not available in the dataset: {missing_vars}."
                )

    data = dataset[array][..., var_indices, :, :]
    return data, variables


def _load_stats(
    stats: dict, variables: list[str], array: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load mean and standard deviation stats from a dict and format them as NumPy arrays.

    Parameters
    ----------
    stats : dict
        Statistics dictionary.
    variables : list[str]
        Variable names to load stats for.
    array : str
        Array name (e.g., 'input', 'output').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mean and standard deviation arrays.
    """
    mean = np.array([stats[array][v]["mean"] for v in variables])[:, None, None].astype(
        np.float32
    )
    std = np.array([stats[array][v]["std"] for v in variables])[:, None, None].astype(
        np.float32
    )
    return (mean, std)


def create_sample_dataset(
    output_data_path: str,
    output_stats_path: str,
    num_times: int = 10,
    shape: tuple[int, int] = (64, 64),
    input_variables: list[str] | None = None,
    output_variables: list[str] | None = None,
    invariant_variables: list[str] | None = None,
    start_time: str = "2023-01-01T00:00:00",
    time_delta_hours: int = 1,
    seed: int = 42,
) -> None:
    """
    Create and save a sample dataset for XarrayDataset.

    This function generates synthetic data with the proper structure expected
    by the XarrayDataset class, including input, output, and
    invariant arrays, along with spatial coordinates and normalization statistics.

    Parameters
    ----------
    output_data_path : str
        Path where the dataset will be saved (e.g., "sample_data.nc" for NetCDF
        or "sample_data.zarr" for Zarr format).
    output_stats_path : str
        Path where the statistics JSON file will be saved.
    num_times : int, optional, default 10
        Number of time samples to generate.
    shape : tuple[int, int], optional, default (64, 64)
        Shape (height, width) of the spatial grid.
    input_variables : list[str] | None, optional
        Names of input variables. If None, defaults to ["u10m", "v10m", "t2m"].
    output_variables : list[str] | None, optional
        Names of output variables. If None, defaults to ["precipitation", "temperature"].
    invariant_variables : list[str] | None, optional
        Names of invariant variables. If None, defaults to ["elevation", "land_sea_mask"].
    start_time : str, optional, default "2023-01-01T00:00:00"
        Starting time in ISO format.
    time_delta_hours : int, optional, default 1
        Time interval between samples in hours.
    seed : int, optional, default 42
        Random seed for reproducible data generation.

    Returns
    -------
    None
        Saves the dataset and statistics files to disk.

    Examples
    --------
    create_sample_dataset(
        output_data_path="sample_data.nc",
        output_stats_path="sample_stats.json",
        num_times=100,
        shape=(128, 128)
    )
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Set default variable names if not provided
    if input_variables is None:
        input_variables = ["u10m", "v10m", "t2m"]
    if output_variables is None:
        output_variables = ["precipitation", "temperature"]
    if invariant_variables is None:
        invariant_variables = ["elevation", "land_sea_mask"]

    # Create time coordinates
    start_datetime = np.datetime64(start_time)
    times = np.array(
        [
            start_datetime + np.timedelta64(i * time_delta_hours, "h")
            for i in range(num_times)
        ]
    )

    # Create spatial coordinates
    (height, width) = shape
    y_coords = np.arange(height)
    x_coords = np.arange(width)

    # Generate latitude and longitude grids (example centered around 40°N, -100°W)
    lat_center = 40.0
    lon_center = -100.0
    lat_range = 10.0  # degrees
    lon_range = 10.0  # degrees

    lat = np.linspace(lat_center - lat_range / 2, lat_center + lat_range / 2, height)[
        :, None
    ] * np.ones((height, width))

    lon = (
        np.ones((height, 1))
        * np.linspace(lon_center - lon_range / 2, lon_center + lon_range / 2, width)[
            None, :
        ]
    )

    # Generate random input data
    input_data = np.random.randn(num_times, len(input_variables), height, width).astype(
        np.float32
    )

    # Scale input variables to reasonable physical ranges
    # u10m, v10m: wind in m/s (scaled to roughly -20 to 20 m/s)
    # t2m: temperature in K (scaled to roughly 250-310 K)
    for i, var in enumerate(input_variables):
        if var in ["u10m", "v10m"]:
            input_data[:, i] = input_data[:, i] * 5.0  # ~N(0, 5) m/s
        elif var in ["t2m", "temperature"]:
            input_data[:, i] = input_data[:, i] * 15.0 + 280.0  # ~N(280, 15) K
        else:
            input_data[:, i] = input_data[:, i] * 10.0  # generic scaling

    # Generate random output data
    output_data = np.random.randn(
        num_times, len(output_variables), height, width
    ).astype(np.float32)

    # Scale output variables to reasonable physical ranges
    for i, var in enumerate(output_variables):
        if var == "precipitation":
            output_data[:, i] = np.maximum(
                0, output_data[:, i] * 2.0 + 1.0
            )  # non-negative
        elif var == "temperature":
            output_data[:, i] = output_data[:, i] * 20.0 + 290.0  # ~N(290, 20) K
        else:
            output_data[:, i] = output_data[:, i] * 10.0  # generic scaling

    # Generate invariant data
    invariant_data = np.random.randn(len(invariant_variables), height, width).astype(
        np.float32
    )

    # Scale invariant variables to reasonable ranges
    for i, var in enumerate(invariant_variables):
        if var == "elevation":
            invariant_data[i] = np.maximum(
                0, invariant_data[i] * 500.0 + 500.0
            )  # 0-1500m
        elif var == "land_sea_mask":
            invariant_data[i] = (invariant_data[i] > 0).astype(
                np.float32
            )  # binary mask
        else:
            invariant_data[i] = invariant_data[i] * 10.0  # generic scaling

    # Create xarray Dataset
    dataset = xr.Dataset(
        data_vars={
            "input": (
                ["time", "input_variable", "y", "x"],
                input_data,
            ),
            "output": (
                ["time", "output_variable", "y", "x"],
                output_data,
            ),
            "invariant": (
                ["invariant_variable", "y", "x"],
                invariant_data,
            ),
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
        },
        coords={
            "time": times,
            "input_variable": input_variables,
            "output_variable": output_variables,
            "invariant_variable": invariant_variables,
            "y": y_coords,
            "x": x_coords,
        },
    )

    # Add metadata attributes
    dataset.attrs["description"] = (
        "Sample downscaling dataset for GenericDownscalingDataset"
    )
    dataset.attrs["created"] = datetime.datetime.now().isoformat()

    # Save dataset
    if output_data_path.endswith(".zarr"):
        dataset.to_zarr(output_data_path, mode="w")
    else:
        dataset.to_netcdf(output_data_path)

    print(f"Dataset saved to: {output_data_path}")

    # Compute statistics for normalization
    stats = {
        "input": {},
        "output": {},
        "invariant": {},
    }

    # Compute input statistics
    for i, var in enumerate(input_variables):
        stats["input"][var] = {
            "mean": float(np.mean(input_data[:, i])),
            "std": float(np.std(input_data[:, i])),
        }

    # Compute output statistics
    for i, var in enumerate(output_variables):
        stats["output"][var] = {
            "mean": float(np.mean(output_data[:, i])),
            "std": float(np.std(output_data[:, i])),
        }

    # Compute invariant statistics
    for i, var in enumerate(invariant_variables):
        stats["invariant"][var] = {
            "mean": float(np.mean(invariant_data[i])),
            "std": float(np.std(invariant_data[i])),
        }

    # Save statistics to JSON
    with open(output_stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to: {output_stats_path}")
    print(f"\nDataset summary:")
    print(f"  - Time samples: {num_times}")
    print(f"  - Spatial dimensions: {height} x {width}")
    print(f"  - Input variables: {input_variables}")
    print(f"  - Output variables: {output_variables}")
    print(f"  - Invariant variables: {invariant_variables}")
