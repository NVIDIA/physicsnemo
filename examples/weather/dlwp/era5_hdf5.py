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
from typing import Iterable, List, Union, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

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
    num_input_steps : int, optional
        Number of timesteps are included in the input variables, by default 1
    num_output_steps : int, optional
        Number of timesteps are included in the output variables, by default 1
    grid_type : str, optional
        Type of grid in the input NetCDF file. Must be one of the following: "latlon",
        "cubesphere", by default "latlon"
    patch_size : Union[Tuple[int, int], int, None], optional
        If specified, crops input and output variables so image dimensions are
        divisible by patch_size, by default None
    num_samples_per_year : int, optional
        Number of samples randomly taken from each year. If None, all will be use, by default None
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
        num_input_steps: int = 1,
        num_output_steps: int = 1,
        grid_type: str = "latlon",
        stride: int = 1,
        patch_size: Union[Tuple[int, int], int, None] = None,
        num_samples_per_year: Union[int, None] = None,
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
        self.stats_dir = Path(stats_dir) if not stats_dir is None else None
        self.channels = channels
        self.stride = stride
        self.num_input_steps = num_input_steps
        self.num_output_steps = num_output_steps
        self.grid_type = grid_type
        self.num_samples_per_year = num_samples_per_year
        self.process_rank = process_rank
        self.world_size = world_size

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

        # check root directory exists
        if not self.data_dir.is_dir():
            raise IOError(f"Error, data directory {self.data_dir} does not exist")
        if not self.stats_dir is None and not self.stats_dir.is_dir():
            raise IOError(f"Error, stats directory {self.stats_dir} does not exist")

        # check valid grid type
        self.allowed_grid_types = ["latlon", "cubesphere"]
        if self.grid_type not in self.allowed_grid_types:
            raise ValueError(
                f"Invalid grid type. Must be one of: {', '.join(self.allowed_grid_types)}"
            )

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
            # truncate the dataset to avoid out-of-range sampling
            data_samples_per_year = (
                f["fields"].shape[0]
                - self.num_input_steps * self.stride
                - self.num_output_steps * self.stride
            )
            self.img_shape = f["fields"].shape[2:]

            # If channels not provided, use all of them
            if self.channels is None:
                self.channels = [i for i in range(f["fields"].shape[1])]

            # If num_samples_per_year use all
            if self.num_samples_per_year is None:
                self.num_samples_per_year = data_samples_per_year

            # Adjust image shape if patch_size defined
            if self.grid_type == "latlon":
                if self.patch_size is not None:
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

        if self.grid_type == "latlon":
            # has shape [1, C, 1, 1]
            self.mu = np.load(str(mean_stat_file))[:, self.channels]
            # has shape [1, C, 1, 1]
            self.sd = np.load(str(std_stat_file))[:, self.channels]
            if not self.mu.shape == self.sd.shape == (1, len(self.channels), 1, 1):
                raise AssertionError("Error, normalisation arrays have wrong shape")
        else:  # cubed sphere
            # has shape [1, C, 1, 1, 1]
            self.mu = np.load(str(mean_stat_file))[:, self.channels]
            self.mu = np.expand_dims(self.mu, -1)
            # has shape [1, C, 1, 1, 1]
            self.sd = np.load(str(std_stat_file))[:, self.channels]
            self.sd = np.expand_dims(self.sd, -1)
            if not self.mu.shape == self.sd.shape == (1, len(self.channels), 1, 1, 1):
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
                num_input_steps=self.num_input_steps,
                num_output_steps=self.num_output_steps,
                num_samples_per_year=self.num_samples_per_year,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )
            # Update length of dataset
            self.length = len(source) // self.batch_size
            # Read current batch.
            invar, outvar, invar_idx, outvar_idx, year_idx = dali.fn.external_source(
                source,
                num_outputs=5,
                parallel=False,
                batch=False,
            )
            # if self.device.type == "cuda":
            #    # Move tensors to GPU as external_source won't do that.
            #    invar = invar.gpu()
            #    outvar = outvar.gpu()

            # Crop.
            if self.grid_type == "latlon":
                h, w = self.img_shape
                invar = invar[:, :h, :w]
                outvar = outvar[:, :, :h, :w]
            # Standardize.
            if not self.stats_dir is None:
                invar = dali.fn.normalize(invar, mean=self.mu, stddev=self.sd)
                outvar = dali.fn.normalize(outvar, mean=self.mu, stddev=self.sd)

            # Set outputs.
            pipe.set_outputs(invar, outvar, invar_idx, outvar_idx, year_idx)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator(
            [self.pipe], ["invar", "outvar", "invar_idx", "outvar_idx", "year_idx"]
        )

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
    stride : int
        Number of steps between input and output variables
    num_input_steps : int
        Number of timesteps are included in the input variables
    num_output_steps : int
        Number of timesteps are included in the output variables
    num_samples_per_year : int
        Number of samples randomly taken from each year
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
        num_input_steps: int,
        num_output_steps: int,
        stride: int,
        num_samples_per_year: int,
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
        self.num_input_steps = num_input_steps
        self.num_output_steps = num_output_steps
        self.stride = stride
        self.num_samples_per_year = num_samples_per_year
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

    def __call__(
        self, sample_info: dali.types.SampleInfo
    ) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
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

        data = self.data_files[year_idx]["fields"]
        # Has [T,C,H,W] shape.
        invar_idx = []
        invar = np.empty((self.num_input_steps,) + data.shape[1:], dtype=data.dtype)
        for i in range(self.num_input_steps):
            in_idx = in_idx + i * self.stride
            invar_idx.append(in_idx)
            invar[i] = data[in_idx, self.chans]

        # Has [T,C,H,W] shape.
        outvar_idx = []
        outvar = np.empty((self.num_output_steps,) + data.shape[1:], dtype=data.dtype)
        for i in range(self.num_output_steps):
            out_idx = in_idx + (i + 1) * self.stride
            outvar_idx.append(out_idx)
            outvar[i] = data[out_idx, self.chans]

        invar_idx = np.array(invar_idx)
        outvar_idx = np.array(outvar_idx)
        year_idx = np.array(year_idx)

        return invar, outvar, invar_idx, outvar_idx, year_idx

    def __len__(self):
        return len(self.indices)
