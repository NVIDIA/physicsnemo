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


import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from itertools import chain

import h5py
import netCDF4 as nc
import numpy as np
import pytz
import torch

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Tuple, Union

from scipy.io import netcdf_file

from physicsnemo.datapipes.climate.utils.invariant import latlon_grid
from physicsnemo.datapipes.climate.utils.zenith_angle import cos_zenith_angle
from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData
from physicsnemo.launch.logging import PythonLogger

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "Climate"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class ClimateDataSourceSpec:
    """
    A data source specification for ClimateDatapipe.

    HDF5 files should contain the following variable with the corresponding
    name:
    `fields`: Tensor of shape (num_timesteps, num_channels, height, width),
    containing climate data. The order of the channels should match the order
    of the channels in the statistics files. The statistics files should be
    `.npy` files with the shape (1, num_channels, 1, 1).
    The names of the variables are found in the metadata file found in
    `metadata_path`.

    NetCDF4 files should contain a variable of shape
    (num_timesteps, height, width) for each variable they provide. Only the
    variables listed in `variables` will be loaded.

    Parameters
    ----------
    data_dir : str
        Directory where climate data is stored
    name: Union[str, None], optional
        The name that is used to label datapipe outputs from this source.
        If None, the datapipe uses the number of the source in sequential order.
    file_type: str
        Type of files to read, supported values are "hdf5" (default) and "netcdf4"
    stats_files: Union[Mapping[str, str], None], optional
        Numpy files to data statistics for normalization. Supports either a channels
        format, in which case the dict should contain the keys "mean" and "std", or a
        named-variable format, in which case the dict should contain the key "norm" .
        If None, no normalization will be used, by default None
    metadata_path: Union[Mapping[str, str], None], optional for NetCDF, required for HDF5
        Path to the metadata JSON file for the dataset (usually called data.json).
    channels : Union[List[int], None], optional
        Defines which climate variables to load, if None will use all in HDF5 file, by default None
    variables: Union[List[str], None], optional for HDF5 files, mandatory for NetCDF4 files
        List of named variables to load. Variables will be read in the order specified
        by this parameter. Must be used for NetCDF4 files. Supported for HDF5 files
        in which case it will override `channels`.
    use_cos_zenith: bool, optional
        If True, the cosine zenith angles corresponding to the coordinates of this
        data source will be produced, default False
    aux_variables : Union[Mapping[str, Callable], None], optional
        A dictionary mapping strings to callables that accept arguments
        (timestamps: numpy.ndarray, latlon: numpy.ndarray). These define any auxiliary
        variables returned from this source.
    num_steps : int, optional
        Number of timesteps to return, by default 1
    stride : int, optional
        Number of steps between input and output variables. For example, if the dataset
        contains data at every 6 hours, a stride 1 = 6 hour delta t and
        stride 2 = 12 hours delta t, by default 1
    """

    def __init__(
        self,
        data_dir: str,
        name: Union[str, None] = None,
        file_type: str = "hdf5",
        stats_files: Union[Mapping[str, str], None] = None,
        metadata_path: Union[str, None] = None,
        channels: Union[List[int], None] = None,
        variables: Union[List[str], None] = None,
        use_cos_zenith: bool = False,
        aux_variables: Union[Mapping[str, Callable], None] = None,
        num_steps: int = 1,
        stride: int = 1,
        backend_kwargs: Union[dict, None] = None,
    ):
        self.data_dir = Path(data_dir)
        self.name = name
        self.file_type = file_type
        self.stats_files = (
            {k: Path(fn) for (k, fn) in stats_files.items()}
            if stats_files is not None
            else None
        )
        self.metadata_path = Path(metadata_path) if metadata_path is not None else None
        self.channels = channels
        self.variables = variables
        self.use_cos_zenith = use_cos_zenith
        self.aux_variables = aux_variables if aux_variables is not None else {}
        self.num_steps = num_steps
        self.stride = stride
        self.backend_kwargs = {} if backend_kwargs is None else backend_kwargs
        self.logger = PythonLogger()

        if file_type == "netcdf4" and not variables:
            raise ValueError("Variables must be specified for a NetCDF4 source.")

        # check root directory exists
        if not self.data_dir.is_dir():
            raise IOError(f"Error, data directory {self.data_dir} does not exist")
        if self.stats_files is None:
            self.logger.warning(
                "Warning, no stats files specified, this will result in no normalisation"
            )

    def dimensions_compatible(self, other) -> bool:
        """
        Basic sanity check to test if two `ClimateDataSourceSpec` are
        compatible.
        """
        return (
            self.data_shape == other.data_shape
            and self.cropped_data_shape == other.cropped_data_shape
            and self.num_samples_per_year == other.num_samples_per_year
            and self.total_length == other.total_length
            and self.n_years == other.n_years
        )

    def parse_dataset_files(
        self,
        num_samples_per_year: Union[int, None] = None,
        patch_size: Union[int, None] = None,
    ) -> None:
        """Parses the data directory for valid files and determines training samples

        Parameters
        ----------
        num_samples_per_year : int, optional
            Number of samples taken from each year. If None, all will be used, by default None
        patch_size : Union[Tuple[int, int], int, None], optional
            If specified, crops input and output variables so image dimensions are
            divisible by patch_size, by default None

        Raises
        ------
        ValueError
            In channels specified or number of samples per year is not valid
        """
        # get all input data files
        suffix = {"hdf5": "h5", "netcdf4": "nc"}[self.file_type]
        self.data_paths = sorted(self.data_dir.glob(f"*.{suffix}"))
        for data_path in self.data_paths:
            self.logger.info(f"Climate data file found: {data_path}")
        self.n_years = len(self.data_paths)
        self.logger.info(f"Number of years: {self.n_years}")

        # get total number of examples and image shape from the first file,
        # assuming other files have exactly the same format.
        self.logger.info(f"Getting file stats from {self.data_paths[0]}")
        if self.file_type == "hdf5":
            with h5py.File(self.data_paths[0], "r") as f:
                dataset_shape = f["fields"].shape
        else:
            with nc.Dataset(self.data_paths[0], "r") as f:
                var_shape = f[self.variables[0]].shape
                dataset_shape = (var_shape[0], len(self.variables)) + var_shape[1:]

        # truncate the dataset to avoid out-of-range sampling
        data_samples_per_year = dataset_shape[0] - (self.num_steps - 1) * self.stride
        self.data_shape = dataset_shape[2:]

        # interpret list of variables into list of channels or vice versa
        if self.file_type == "hdf5":
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            data_vars = metadata["coords"]["channel"]
            if self.variables is not None:
                self.channels = [data_vars.index(v) for v in self.variables]
            else:
                if self.channels is None:
                    self.variables = data_vars
                else:
                    self.variables = [data_vars[i] for i in self.channels]

        # If channels not provided, use all of them
        if self.channels is None:
            self.channels = list(range(dataset_shape[1]))

        # If num_samples_per_year use all
        if num_samples_per_year is None:
            num_samples_per_year = data_samples_per_year
        self.num_samples_per_year = num_samples_per_year

        # Adjust image shape if patch_size defined
        if patch_size is not None:
            self.cropped_data_shape = tuple(
                s - s % patch_size[i] for i, s in enumerate(self.data_shape)
            )
        else:
            self.cropped_data_shape = self.data_shape
        self.logger.info(f"Input data shape: {self.cropped_data_shape}")

        # Get total length
        self.total_length = self.n_years * self.num_samples_per_year

        # Sanity checks
        if max(self.channels) >= dataset_shape[1]:
            raise ValueError(
                f"Provided channel has indexes greater than the number \
            of fields {dataset_shape[1]}"
            )

        if self.num_samples_per_year > data_samples_per_year:
            raise ValueError(
                f"num_samples_per_year ({self.num_samples_per_year}) > number of \
                samples available ({data_samples_per_year})!"
            )

        self._load_statistics()

        self.logger.info(f"Number of samples/year: {self.num_samples_per_year}")
        self.logger.info(f"Number of channels available: {dataset_shape[1]}")

    def _load_statistics(self) -> None:
        """Loads climate statistics from pre-computed numpy files

        The statistic files should be of name global_means.npy and global_std.npy with
        a shape of [1, C, 1, 1] located in the stat_dir.

        Raises
        ------
        IOError
            If statistics files are not found
        AssertionError
            If loaded numpy arrays are not of correct size
        """
        # If no stats files we just skip loading the stats
        if self.stats_files is None:
            self.mu = None
            self.sd = None
            return
        # load normalisation values
        if set(self.stats_files) == {"mean", "std"}:  # use mean and std files
            mean_stat_file = self.stats_files["mean"]
            std_stat_file = self.stats_files["std"]

            if not mean_stat_file.exists():
                raise IOError(f"Mean statistics file {mean_stat_file} not found")
            if not std_stat_file.exists():
                raise IOError(f"Std statistics file {std_stat_file} not found")

            # has shape [1, C, 1, 1]
            self.mu = np.load(str(mean_stat_file))[:, self.channels]
            # has shape [1, C, 1, 1]
            self.sd = np.load(str(std_stat_file))[:, self.channels]
        elif set(self.stats_files) == {
            "norm",
        }:  # use dict formatted file with named variables
            norm_stat_file = self.stats_files["norm"]
            if not norm_stat_file.exists():
                raise IOError(f"Statistics file {norm_stat_file} not found")

            norm = np.load(str(norm_stat_file), allow_pickle=True).item()
            mu = np.array([norm[var]["mean"] for var in self.variables])
            self.mu = mu.reshape((1, len(mu), 1, 1))
            sd = np.array([norm[var]["std"] for var in self.variables])
            self.sd = sd.reshape((1, len(sd), 1, 1))
        else:
            raise ValueError(("Invalid statistics file specification"))

        if not self.mu.shape == self.sd.shape == (1, len(self.channels), 1, 1):
            raise ValueError("Error, normalisation arrays have wrong shape")


class ClimateDatapipe(Datapipe):
    """
    A Climate DALI data pipeline. This pipeline loads data from
    HDF5/NetCDF4 files. It can also return additional data such as the
    solar zenith angle for each time step. Additionally, it normalizes
    the data if a statistics file is provided. The pipeline returns a dictionary
    with the following structure, where {name} indicates the name of the data
    source provided:

    - ``state_seq-{name}``: Tensors of shape
        (batch_size, num_steps, num_channels, height, width).
        This sequence is drawn from the data file and normalized if a
        statistics file is provided.
    - ``timestamps-{name}``: Tensors of shape (batch_size, num_steps), containing
        timestamps for each timestep in the sequence.
    - ``{aux_variable}-{name}``: Tensors of shape
        (batch_size, num_steps, aux_channels, height, width),
        containing the auxiliary variables returned by each data source
    - ``cos_zenith-{name}``: Tensors of shape (batch_size, num_steps, 1, height, width),
        containing the cosine of the solar zenith angle if specified.
    - ``{invariant_name}``: Tensors of shape (batch_size, invariant_channels, height, width),
        containing the time-invariant data (depending only on spatial coordinates)
        returned by the datapipe. These can include e.g.
        land-sea mask and geopotential/surface elevation.

    To use this data pipeline, your data directory must be structured as
    follows:
    ```
    data_dir
    ├── 1980.h5
    ├── 1981.h5
    ├── 1982.h5
    ├── ...
    └── 2020.h5
    ```

    The files are assumed have no metadata, such as timestamps.
    Because of this, it's important to specify the `dt` parameter and the
    `start_year` parameter so that the pipeline can compute the correct
    timestamps for each timestep. These timestamps are then used to compute the
    cosine of the solar zenith angle, if specified.

    Parameters
    ----------
    sources: Iterable[ClimateDataSourceSpec]
        A list of data specifications defining the sources for the climate variables
    batch_size : int, optional
        Batch size, by default 1
    dt : float, optional
        Time in hours between each timestep in the dataset, by default 6 hr
    start_year : int, optional
        Start year of dataset, by default 1980
    latlon_bounds : Tuple[Tuple[float, float], Tuple[float, float]], optional
        Bounds of latitude and longitude in the data, in the format
        ((lat_start, lat_end,), (lon_start, lon_end)).
        By default ((90, -90), (0, 360)).
    crop_window: Union[Tuple[Tuple[float, float], Tuple[float, float]], None], optional
        The window to crop the data to, in the format ((i0,i1), (j0,j1)) where the
        first spatial dimension will be cropped to i0:i1 and the second to j0:j1.
        If not given, all data will be used.
    invariants : Mapping[str,Callable], optional
        Specifies the time-invariant data (for example latitude and longitude)
        included in the data samples. Should be a dict where the keys are the
        names of the invariants and the values are the corresponding
        functions. The functions need to accept an argument of the shape
        (2, data_shape[0], data_shape[1]) where the first dimension contains
        latitude and longitude in degrees and the other dimensions corresponding
        to the shape of data in the data files. For example,
        invariants={"trig_latlon": invariants.LatLon()}
        will include the sin/cos of lat/lon in the output.
    num_samples_per_year : int, optional
        Number of samples taken from each year. If None, all will be used, by default None
    shuffle : bool, optional
        Shuffle dataset, by default True
    num_workers : int, optional
        Number of workers, by default 1
    device: Union[str, torch.device], optional
        Device for DALI pipeline to run on, by default cuda
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    """

    def __init__(
        self,
        sources: Iterable[ClimateDataSourceSpec],
        batch_size: int = 1,
        dt: float = 6.0,
        start_year: int = 1980,
        latlon_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (90, -90),
            (0, 360),
        ),
        crop_window: Union[
            Tuple[Tuple[float, float], Tuple[float, float]], None
        ] = None,
        invariants: Union[Mapping[str, Callable], None] = None,
        num_samples_per_year: Union[int, None] = None,
        shuffle: bool = True,
        num_workers: int = 1,  # TODO: is there a faster good default?
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(meta=MetaData())
        self.sources = list(sources)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dt = dt
        self.start_year = start_year
        self.data_latlon_bounds = latlon_bounds
        self.process_rank = process_rank
        self.world_size = world_size
        self.num_samples_per_year = num_samples_per_year
        self.logger = PythonLogger()

        if invariants is None:
            invariants = {}

        # Determine outputs of pipeline
        self.pipe_outputs = []
        for i, spec in enumerate(self.sources):
            name = spec.name if spec.name is not None else i
            self.pipe_outputs += [f"state_seq-{name}", f"timestamps-{name}"]
            self.pipe_outputs.extend(
                f"{aux_var}-{name}" for aux_var in spec.aux_variables
            )
            if spec.use_cos_zenith:
                self.pipe_outputs.append(f"cos_zenith-{name}")
        self.pipe_outputs.extend(invariants.keys())

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)

        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        # Load all data files and statistics
        for spec in sources:
            spec.parse_dataset_files(num_samples_per_year=num_samples_per_year)
        for i, spec_i in enumerate(sources):
            for spec_j in sources[i + 1 :]:
                if not spec_i.dimensions_compatible(spec_j):
                    raise ValueError("Incompatible data sources")

        self.data_latlon = np.stack(
            latlon_grid(bounds=self.data_latlon_bounds, shape=sources[0].data_shape),
            axis=0,
        )
        if crop_window is None:
            crop_window = (
                (0, sources[0].cropped_data_shape[0]),
                (0, sources[0].cropped_data_shape[1]),
            )
        self.crop_window = crop_window
        self.window_latlon = self._crop_to_window(self.data_latlon)
        self.window_latlon_dali = dali.types.Constant(self.window_latlon)

        # load invariants
        self.invariants = {
            var: callback(self.window_latlon) for (var, callback) in invariants.items()
        }

        # Create pipeline
        self.pipe = self._create_pipeline()

    def _source_cls_from_type(self, source_type: str) -> type:
        """Get the external source class based on a string descriptor."""
        return {
            "hdf5": ClimateHDF5DaliExternalSource,
            "netcdf4": ClimateNetCDF4DaliExternalSource,
        }[source_type]

    def _crop_to_window(self, x):
        cw = self.crop_window
        if isinstance(x, dali.pipeline.DataNode):
            # DALI doesn't support ellipsis notation
            return x[:, :, cw[0][0] : cw[0][1], cw[1][0] : cw[1][1]]
        else:
            return x[..., cw[0][0] : cw[0][1], cw[1][0] : cw[1][1]]

    def _source_outputs(self, spec: ClimateDataSourceSpec) -> List:
        """Create DALI outputs for a given data source specification.

        Parameters
        ----------
        spec: ClimateDataSourceSpec
            The data source specification.
        """
        # HDF5/NetCDF source
        source_cls = self._source_cls_from_type(spec.file_type)
        source = source_cls(
            data_paths=spec.data_paths,
            num_samples=spec.total_length,
            channels=spec.channels,
            latlon=self.data_latlon,
            variables=spec.variables,
            aux_variables=spec.aux_variables,
            stride=spec.stride,
            dt=self.dt,
            start_year=self.start_year,
            num_steps=spec.num_steps,
            num_samples_per_year=spec.num_samples_per_year,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            process_rank=self.process_rank,
            world_size=self.world_size,
        )

        # Update length of dataset
        self.total_length = len(source) // self.batch_size

        # Read current batch
        (state_seq, timestamps, *aux) = dali.fn.external_source(
            source,
            num_outputs=source.num_outputs(),
            parallel=True,
            batch=False,
        )

        # Crop
        state_seq = self._crop_to_window(state_seq)
        aux = (self._crop_to_window(x) for x in aux)

        # Normalize
        if spec.stats_files is not None:
            state_seq = dali.fn.normalize(state_seq, mean=spec.mu, stddev=spec.sd)

        # Make output list
        outputs = [state_seq, timestamps, *aux]

        # Get cosine zenith angle
        if spec.use_cos_zenith:
            cos_zenith = dali.fn.cast(
                cos_zenith_angle(timestamps, latlon=self.window_latlon_dali),
                dtype=dali.types.FLOAT,
            )
            outputs.append(cos_zenith)

        return outputs

    def _invariant_outputs(self):
        for inv in self.invariants.values():
            if self.crop_window is not None:
                inv = self._crop_to_window(inv)
            yield dali.types.Constant(inv)

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            Climate DALI pipeline
        """
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            # Concatenate outputs from all sources as well as invariants
            outputs = list(
                chain(
                    *(self._source_outputs(spec) for spec in self.sources),
                    self._invariant_outputs(),
                )
            )

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that
                outputs = [o.gpu() for o in outputs]

            # Set outputs
            pipe.set_outputs(*outputs)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], self.pipe_outputs)

    def __len__(self):
        return self.total_length


class ClimateDaliExternalSource(ABC):
    """DALI Source for lazy-loading the HDF5/NetCDF4 climate files

    Parameters
    ----------
    data_paths : Iterable[str]
        Directory where climate data is stored
    num_samples : int
        Total number of training samples
    channels : Iterable[int]
        List representing which climate variables to load
    num_steps : int
        Number of timesteps to load
    stride : int
        Number of steps between input and output variables
    dt : float, optional
        Time in hours between each timestep in the dataset, by default 6 hr
    start_year : int, optional
        Start year of dataset, by default 1980
    num_samples_per_year : int
        Number of samples randomly taken from each year
    variables: Union[List[str], None], optional for HDF5 files, mandatory for NetCDF4 files
        List of named variables to load. Variables will be read in the order specified
        by this parameter.
    aux_variables : Union[Mapping[str, Callable], None], optional
        A dictionary mapping strings to callables that accept arguments
        (timestamps: numpy.ndarray, latlon: numpy.ndarray). These define any auxiliary
        variables returned from this source.
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Shuffle dataset, by default True
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1

    Note
    ----
    For more information about DALI external source operator:
    https://docs.nvidia.com/deeplearning/dali/archives/dali_1_13_0/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    """

    def __init__(
        self,
        data_paths: Iterable[str],
        num_samples: int,
        channels: Iterable[int],
        num_steps: int,
        stride: int,
        dt: float,
        start_year: int,
        num_samples_per_year: int,
        latlon: np.ndarray,
        variables: Union[List[str], None] = None,
        aux_variables: List[Union[str, Callable]] = (),
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
        backend_kwargs: Union[dict, None] = None,
    ):
        self.data_paths = list(data_paths)
        # Will be populated later once each worker starts running in its own process.
        self.data_files = [None] * len(self.data_paths)
        self.num_samples = num_samples
        self.chans = list(channels)
        self.latlon = latlon
        self.variables = variables
        self.aux_variables = aux_variables
        self.num_steps = num_steps
        self.stride = stride
        self.dt = dt
        self.start_year = start_year
        self.num_samples_per_year = num_samples_per_year
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.backend_kwargs = {} if backend_kwargs is None else backend_kwargs

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

    @abstractmethod
    def _load_sequence(self, year_idx: int, idx: int) -> np.array:
        """Write data from year index `year_idx` and sample index `idx` to output"""
        pass

    def __call__(self, sample_info: dali.types.SampleInfo) -> Tuple[Tensor, np.ndarray]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index
        # TODO: This is very hacky, but it works for now
        idx = self.indices[sample_info.idx_in_epoch]
        year_idx = idx // self.num_samples_per_year
        in_idx = idx % self.num_samples_per_year

        state_seq = self._load_sequence(year_idx, in_idx)

        # Load sequence of timestamps
        year = self.start_year + year_idx
        start_time = datetime(year, 1, 1, tzinfo=pytz.utc) + timedelta(
            hours=int(in_idx) * self.dt
        )
        timestamps = np.array(
            [
                (start_time + timedelta(hours=i * self.stride * self.dt)).timestamp()
                for i in range(self.num_steps)
            ]
        )

        # outputs from auxiliary sources
        aux_outputs = (
            callback(timestamps, self.latlon)
            for callback in self.aux_variables.values()
        )

        return (state_seq, timestamps, *aux_outputs)

    def num_outputs(self):
        return 2 + len(self.aux_variables)

    def __len__(self):
        return len(self.indices)


class ClimateHDF5DaliExternalSource(ClimateDaliExternalSource):
    """DALI source for reading HDF5 formatted climate data files."""

    def _get_data_file(self, year_idx: int) -> h5py.File:
        """Return the opened file for year `year_idx`."""
        if self.data_files[year_idx] is None:
            # This will be called once per worker. Workers are persistent,
            # so there is no need to explicitly close the files - this will be done
            # when corresponding pipeline/dataset is destroyed.
            # Lazy opening avoids unnecessary file open ops when sharding.
            self.data_files[year_idx] = h5py.File(self.data_paths[year_idx], "r")
        return self.data_files[year_idx]

    def _load_sequence(self, year_idx: int, idx: int) -> np.array:
        # TODO: the data is returned in a weird (time, channels, width, height) shape
        data = self._get_data_file(year_idx)["fields"]
        return data[idx : idx + self.num_steps * self.stride : self.stride, self.chans]


class ClimateNetCDF4DaliExternalSource(ClimateDaliExternalSource):
    """DALI source for reading NetCDF4 formatted climate data files."""

    def _get_data_file(self, year_idx: int) -> netcdf_file:
        """Return the opened file for year `year_idx`."""
        if self.data_files[year_idx] is None:
            # This will be called once per worker. Workers are persistent,
            # so there is no need to explicitly close the files - this will be done
            # when corresponding pipeline/dataset is destroyed
            # Lazy opening avoids unnecessary file open ops when sharding.
            # NOTE: The SciPy NetCDF reader can be used if the netCDF4 library
            # causes crashes.
            reader = self.backend_kwargs.get("reader", "netcdf4")
            if reader == "scipy":
                self.data_files[year_idx] = netcdf_file(self.data_paths[year_idx])
            elif reader == "netcdf4":
                self.data_files[year_idx] = nc.Dataset(self.data_paths[year_idx], "r")
                self.data_files[year_idx].set_auto_maskandscale(False)

        return self.data_files[year_idx]

    def _load_sequence(self, year_idx: int, idx: int) -> np.array:
        data_file = self._get_data_file(year_idx)
        shape = data_file.variables[self.variables[0]].shape
        shape = (self.num_steps, len(self.variables)) + shape[1:]
        # TODO: this can be optimized to do the NetCDF scale/offset on GPU
        output = np.empty(shape, dtype=np.float32)
        for i, var in enumerate(self.variables):
            v = data_file.variables[var]
            output[:, i] = v[
                idx : idx + self.num_steps * self.stride : self.stride
            ].copy()  # .copy() avoids hanging references
            if hasattr(v, "scale_factor"):
                output[:, i] *= v.scale_factor
            if hasattr(v, "add_offset"):
                output[:, i] += v.add_offset
        return output
