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

import h5py
import numpy as np
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import pytz

from physicsnemo.datapipes.climate.utils.invariant import latlon_grid
from physicsnemo.datapipes.climate.utils.zenith_angle import cos_zenith_angle

from ..datapipe import Datapipe
from ..meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "ERA5HDF5"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class ERA5HDF5Datapipe(Datapipe):
    """ERA5 DALI data pipeline for HDF5 files

    Parameters
    ----------
    data_dir : str
        Directory where ERA5 data is stored
    stats_dir : Union[str, None], optional
        Directory to data statistic numpy files for normalization, if None, no normalization
        will be used, by default None
    channels : Union[List[int], None], optional
        Defines which ERA5 variables to load, if None will use all in HDF5 file, by default None
    batch_size : int, optional
        Batch size, by default 1
    stride : int, optional
        Number of steps between input and output variables. For example, if the dataset
        contains data at every 6 hours, a stride 1 = 6 hour delta t and
        stride 2 = 12 hours delta t, by default 1
    num_steps : int, optional
        Number of timesteps are included in the output variables, by default 1
    num_history : int, optional
        Number of previous timesteps included in the input variables, by default 0
    latlon_resolution: Tuple[int, int], optional
        The resolution for the latitude-longitude grid (H, W). Needs to be specified
        for cos zenith angle computation, or interpolation. By default None
    interpolation_type: str, optional
        Interpolation type for resizing. Supports ["INTERP_NN", "INTERP_LINEAR", "INTERP_CUBIC",
        "INTERP_LANCZOS3", "INTERP_TRIANGULAR", "INTERP_GAUSSIAN"]. By default None
        (no interpolation is done)
    patch_size : Union[Tuple[int, int], int, None], optional
        If specified, crops input and output variables so image dimensions are
        divisible by patch_size, by default None
    num_samples_per_year : int, optional
        Number of samples randomly taken from each year. If None, all will be use, by default None
    use_cos_zenith: bool, optional
        If True, the cosine zenith angles corresponding to the coordinates will be produced,
        by default False
    cos_zenith_args: Dict, optional
        Dictionary containing the following
        dt: float, optional
            Time in hours between each timestep in the dataset, by default 6 hr
        start_year: int, optional
            Start year of dataset, by default 1980
        latlon_bounds : Tuple[Tuple[float, float], Tuple[float, float]], optional
            Bounds of latitude and longitude in the data, in the format
            ((lat_start, lat_end,), (lon_start, lon_end)).
            By default ((90, -90), (0, 360)).
        Defaults are only applicable if use_cos_zenith is True. Otherwise, defaults to {}.
    use_time_of_year_index: bool
        If true, also returns the index that can be sued to determine the time of the year
        corresponding to each sample. By default False.
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
        data_dir: str,
        stats_dir: Union[str, None] = None,
        channels: Union[List[int], None] = None,
        batch_size: int = 1,
        num_steps: int = 1,
        num_history: int = 0,
        stride: int = 1,
        latlon_resolution: Union[Tuple[int, int], None] = None,
        interpolation_type: Union[str, None] = None,
        patch_size: Union[Tuple[int, int], int, None] = None,
        num_samples_per_year: Union[int, None] = None,
        use_cos_zenith: bool = False,
        cos_zenith_args: Dict = {},
        use_time_of_year_index: bool = False,
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(meta=MetaData())
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dir = Path(data_dir)
        self.stats_dir = Path(stats_dir) if stats_dir is not None else None
        self.channels = channels
        self.stride = stride
        self.latlon_resolution = latlon_resolution
        self.interpolation_type = interpolation_type
        self.num_steps = num_steps
        self.num_history = num_history
        self.num_samples_per_year = num_samples_per_year
        self.use_cos_zenith = use_cos_zenith
        self.cos_zenith_args = cos_zenith_args
        self.use_time_of_year_index = use_time_of_year_index
        self.process_rank = process_rank
        self.world_size = world_size

        # cos zenith defaults
        if use_cos_zenith:
            cos_zenith_args["dt"] = cos_zenith_args.get("dt", 6.0)
            cos_zenith_args["start_year"] = cos_zenith_args.get("start_year", 1980)
            cos_zenith_args["latlon_bounds"] = cos_zenith_args.get(
                "latlon_bounds",
                (
                    (90, -90),
                    (0, 360),
                ),
            )
        self.latlon_bounds = cos_zenith_args.get("latlon_bounds")

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        # check root directory exists
        if not self.data_dir.is_dir():
            raise IOError(f"Error, data directory {self.data_dir} does not exist")
        if self.stats_dir is not None and not self.stats_dir.is_dir():
            raise IOError(f"Error, stats directory {self.stats_dir} does not exist")

        # Check interpolation type
        if self.interpolation_type is not None:
            valid_interpolation = [
                "INTERP_NN",
                "INTERP_LINEAR",
                "INTERP_CUBIC",
                "INTERP_LANCZOS3",
                "INTERP_TRIANGULAR",
                "INTERP_GAUSSIAN",
            ]
            if self.interpolation_type not in valid_interpolation:
                raise ValueError(
                    f"Interpolation type {self.interpolation_type} not supported"
                )
            self.interpolation_type = getattr(dali.types, self.interpolation_type)

        # Layout
        # Avoiding API change for self.num_history == 0.
        # Need to use FCHW layout in the future regardless of the num_history.
        if self.num_history == 0:
            self.layout = ["CHW", "FCHW"]
        else:
            self.layout = ["FCHW", "FCHW"]

        self.output_keys = ["invar", "outvar"]

        # Get latlon for zenith angle
        if self.use_cos_zenith:
            if not self.latlon_resolution:
                raise ValueError("latlon_resolution must be set for cos zenith angle")
            self.data_latlon = np.stack(
                latlon_grid(bounds=self.latlon_bounds, shape=self.latlon_resolution),
                axis=0,
            )
            self.latlon_dali = dali.types.Constant(self.data_latlon)
            self.output_keys += ["cos_zenith"]

        if self.use_time_of_year_index:
            self.output_keys += ["time_of_year_idx"]

        self.parse_dataset_files()
        self.load_statistics()

        self.pipe = self._create_pipeline()

    def parse_dataset_files(self) -> None:
        """Parses the data directory for valid HDF5 files and determines training samples

        Raises
        ------
        ValueError
            In channels specified or number of samples per year is not valid
        """
        # get all input data files
        self.data_paths = sorted(self.data_dir.glob("????.h5"))
        for data_path in self.data_paths:
            self.logger.info(f"ERA5 file found: {data_path}")
        self.n_years = len(self.data_paths)
        self.logger.info(f"Number of years: {self.n_years}")

        # get total number of examples and image shape from the first file,
        # assuming other files have exactly the same format.
        self.logger.info(f"Getting file stats from {self.data_paths[0]}")
        with h5py.File(self.data_paths[0], "r") as f:
            # truncate the dataset to avoid out-of-range sampling and ensure each
            # rank has same number of samples (to avoid deadlocks)
            data_samples_per_year = (
                (
                    f["fields"].shape[0]
                    - (self.num_steps + self.num_history) * self.stride
                )
                // self.world_size
            ) * self.world_size
            if data_samples_per_year < 1:
                raise ValueError(
                    f"Not enough number of samples per year ({data_samples_per_year})"
                )
            self.img_shape = f["fields"].shape[2:]

            # If channels not provided, use all of them
            if self.channels is None:
                self.channels = [i for i in range(f["fields"].shape[1])]

            # If num_samples_per_year use all
            if self.num_samples_per_year is None:
                self.num_samples_per_year = data_samples_per_year

            # Adjust image shape if patch_size defined
            if self.patch_size is not None:
                if self.use_cos_zenith:
                    raise ValueError("Patching is not supported with cos zenith angle")
                self.img_shape = [
                    s - s % self.patch_size[i] for i, s in enumerate(self.img_shape)
                ]
            self.logger.info(f"Input image shape: {self.img_shape}")

            # Get total length
            self.total_length = self.n_years * self.num_samples_per_year
            self.length = self.total_length

            # Sanity checks
            if max(self.channels) >= f["fields"].shape[1]:
                raise ValueError(
                    f"Provided channel has indexes greater than the number \
                of fields {f['fields'].shape[1]}"
                )

            if self.num_samples_per_year > data_samples_per_year:
                raise ValueError(
                    f"num_samples_per_year ({self.num_samples_per_year}) > number of \
                    samples available ({data_samples_per_year})!"
                )

            self.logger.info(f"Number of samples/year: {self.num_samples_per_year}")
            self.logger.info(f"Number of channels available: {f['fields'].shape[1]}")

    def load_statistics(self) -> None:
        """Loads ERA5 statistics from pre-computed numpy files

        The statistic files should be of name global_means.npy and global_std.npy with
        a shape of [1, C, 1, 1] located in the stat_dir.

        Raises
        ------
        IOError
            If mean or std numpy files are not found
        AssertionError
            If loaded numpy arrays are not of correct size
        """
        # If no stats dir we just skip loading the stats
        if self.stats_dir is None:
            self.mu = None
            self.std = None
            return
        # load normalisation values
        mean_stat_file = self.stats_dir / Path("global_means.npy")
        std_stat_file = self.stats_dir / Path("global_stds.npy")

        if not mean_stat_file.exists():
            raise IOError(f"Mean statistics file {mean_stat_file} not found")
        if not std_stat_file.exists():
            raise IOError(f"Std statistics file {std_stat_file} not found")

        # has shape [1, C, 1, 1]
        self.mu = np.load(str(mean_stat_file))[:, self.channels]
        # has shape [1, C, 1, 1]
        self.sd = np.load(str(std_stat_file))[:, self.channels]

        if not self.mu.shape == self.sd.shape == (1, len(self.channels), 1, 1):
            raise AssertionError("Error, normalisation arrays have wrong shape")

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            HDF5 DALI pipeline
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
            source = ERA5DaliExternalSource(
                data_paths=self.data_paths,
                num_samples=self.total_length,
                channels=self.channels,
                stride=self.stride,
                num_steps=self.num_steps,
                num_history=self.num_history,
                num_samples_per_year=self.num_samples_per_year,
                use_cos_zenith=self.use_cos_zenith,
                cos_zenith_args=self.cos_zenith_args,
                use_time_of_year_index=self.use_time_of_year_index,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )
            # Update length of dataset
            self.length = len(source) // self.batch_size
            # Read current batch.
            invar, outvar, timestamps, time_of_year_idx = dali.fn.external_source(
                source,
                num_outputs=4,
                parallel=True,
                batch=False,
                layout=self.layout,
            )
            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that.
                invar = invar.gpu()
                outvar = outvar.gpu()

            # Crop.
            h, w = self.img_shape
            if self.num_history == 0:
                invar = invar[:, :h, :w]
            else:
                invar = invar[:, :, :h, :w]
            outvar = outvar[:, :, :h, :w]

            # Standardize.
            if self.stats_dir is not None:
                if self.num_history == 0:
                    invar = dali.fn.normalize(invar, mean=self.mu[0], stddev=self.sd[0])
                else:
                    invar = dali.fn.normalize(invar, mean=self.mu, stddev=self.sd)
                outvar = dali.fn.normalize(outvar, mean=self.mu, stddev=self.sd)

            # Resize.
            if self.interpolation_type is not None:
                invar = dali.fn.resize(
                    invar,
                    resize_x=self.latlon_resolution[1],
                    resize_y=self.latlon_resolution[0],
                    interp_type=self.interpolation_type,
                    antialias=False,
                )
                outvar = dali.fn.resize(
                    outvar,
                    resize_x=self.latlon_resolution[1],
                    resize_y=self.latlon_resolution[0],
                    interp_type=self.interpolation_type,
                    antialias=False,
                )

            # cos zenith angle
            if self.use_cos_zenith:
                cos_zenith = dali.fn.cast(
                    cos_zenith_angle(timestamps, latlon=self.latlon_dali),
                    dtype=dali.types.FLOAT,
                )
                if self.device.type == "cuda":
                    cos_zenith = cos_zenith.gpu()

            # # Time of the year
            # time_of_year_idx = dali.fn.cast(
            #         time_of_year_idx,
            #         dtype=dali.types.UINT32,
            #     )

            # Set outputs.
            outputs = (invar, outvar)
            if self.use_cos_zenith:
                outputs += (cos_zenith,)
            if self.use_time_of_year_index:
                outputs += (time_of_year_idx,)
            pipe.set_outputs(*outputs)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], self.output_keys)

    def __len__(self):
        return self.length


class ERA5DaliExternalSource:
    """DALI Source for lazy-loading the HDF5 ERA5 files

    Parameters
    ----------
    data_paths : Iterable[str]
        Directory where ERA5 data is stored
    num_samples : int
        Total number of training samples
    channels : Iterable[int]
        List representing which ERA5 variables to load
    start_year : int, optional
        Start year of dataset
    stride : int
        Number of steps between input and output variables
    num_steps : int
        Number of timesteps are included in the output variables
    num_history : int
        Number of previous timesteps included in the input variables
    num_samples_per_year : int
        Number of samples randomly taken from each year
    batch_size : int, optional
        Batch size, by default 1
    use_cos_zenith: bool
        If True, the cosine zenith angles corresponding to the coordinates will be produced,
    cos_zenith_args: Dict
        Dictionary containing the following
        dt: float
            Time in hours between each timestep in the dataset
        start_year
            Start year of dataset
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
        num_history: int,
        stride: int,
        num_samples_per_year: int,
        use_cos_zenith: bool,
        cos_zenith_args: Dict,
        use_time_of_year_index: bool,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):
        self.data_paths = list(data_paths)
        # Will be populated later once each worker starts running in its own process.
        self.data_files = None
        self.num_samples = num_samples
        self.chans = list(channels)
        self.num_steps = num_steps
        self.num_history = num_history
        self.stride = stride
        self.num_samples_per_year = num_samples_per_year
        self.use_cos_zenith = use_cos_zenith
        self.use_time_of_year_index = use_time_of_year_index
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

        # cos zenith args
        if self.use_cos_zenith:
            self.dt: float = cos_zenith_args.get("dt")
            self.start_year: int = cos_zenith_args.get("start_year")

    def __call__(
        self, sample_info: dali.types.SampleInfo
    ) -> Tuple[Tensor, Tensor, np.ndarray]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        if self.data_files is None:
            # This will be called once per worker. Workers are persistent,
            # so there is no need to explicitly close the files - this will be done
            # when corresponding pipeline/dataset is destroyed.
            self.data_files = [h5py.File(path, "r") for path in self.data_paths]

        # Shuffle before the next epoch starts.
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers.
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index.
        idx = self.indices[sample_info.idx_in_epoch]
        year_idx = idx // self.num_samples_per_year
        in_idx = idx % self.num_samples_per_year

        # Load sequence of timestamps
        if self.use_cos_zenith:
            year = self.start_year + year_idx
            start_time = datetime(year, 1, 1, tzinfo=pytz.utc) + timedelta(
                hours=int(in_idx) * self.dt
            )
            timestamps = np.array(
                [
                    (
                        start_time + timedelta(hours=i * self.stride * self.dt)
                    ).timestamp()
                    for i in range(self.num_history + self.num_steps + 1)
                ]
            )
        else:
            timestamps = np.array([])
        if self.use_time_of_year_index:
            time_of_year_idx = in_idx
        else:
            time_of_year_idx = -1

        data = self.data_files[year_idx]["fields"]
        if self.num_history == 0:
            # Has [C,H,W] shape.
            invar = data[in_idx, self.chans]
        else:
            # Has [T,C,H,W] shape.
            invar = data[
                in_idx : in_idx + (self.num_history + 1) * self.stride : self.stride,
                self.chans,
            ]

        # Has [T,C,H,W] shape.
        outvar = np.empty((self.num_steps,) + invar.shape[-3:], dtype=invar.dtype)

        for i in range(self.num_steps):
            out_idx = in_idx + (self.num_history + i + 1) * self.stride
            outvar[i] = data[out_idx, self.chans]

        return invar, outvar, timestamps, np.array([time_of_year_idx])

    def __len__(self):
        return len(self.indices)
