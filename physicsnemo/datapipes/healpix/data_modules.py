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

# System modules
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

# numpy
import numpy as np

# distributed stuff
import xarray as xr

import warnings

# External modules
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager

from .coupledtimeseries_dataset import CoupledTimeSeriesDataset
from .timeseries_dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


def open_time_series_dataset_classic_prebuilt(
    directory: str, dataset_name: str, constants: bool = False, batch_size: int = 32
) -> xr.Dataset:
    """
    Opens an existing dataset

    Parameters
    ----------
    directory: str
        The directory that contains the dataset
    dataset_name: str
        The name of the dataset to open
    constants: typing.DefaultDict, optional
        Not used for open_time_series_dataset_classic_prebuilt, default False
    batch_size: str, optional
        The chunk size to use for the input datasets, default 32

    Returns
    -------
    xr.Dataset: The opened dataset

    Raises
    ------
    FileNotFoundError: If the dataset doesn't exist at the specified path
    """

    ds_path = Path(directory, dataset_name + ".zarr")

    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset doesn't exist at {ds_path}")

    result = xr.open_zarr(ds_path)
    return result


class TimeSeriesDataModule:
    """pytorch-lightning module for complete model train, validation, and test data loading. Uses
    dlwp.data.data_loading.TimeSeriesDataset under-the-hood. Loaded data files follow the naming scheme
    {directory}/{prefix}{variable/constant}{suffix}{[.nc, .zarr]}
    """

    def __init__(
        self,
        src_directory: str = ".",  # kept for backwards compatibility
        dst_directory: str = ".",
        dataset_name: str = "dataset",
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        data_format: str = "classic",  # kept for backwards compatibility
        batch_size: int = 32,
        drop_last: bool = False,
        input_variables: Optional[Sequence] = ["t2m"],
        output_variables: Optional[Sequence] = None,
        constants: Optional[DictConfig] = None,
        scaling: Optional[DictConfig] = None,
        splits: Optional[DictConfig] = None,
        presteps: int = 0,
        input_time_dim: int = 1,
        output_time_dim: int = 1,
        data_time_step: Union[int, str] = "3h",
        time_step: Union[int, str] = "6h",
        gap: Union[int, str, None] = None,
        shuffle: bool = True,
        add_insolation: bool = False,
        cube_dim: int = 64,  # kept for backwards compatibility
        num_workers: int = 4,
        pin_memory: bool = True,
        prebuilt_dataset: bool = True,  # kept for backwards compatibility
        forecast_init_times: Optional[Sequence] = None,
    ):
        """
        Parameters
        ----------
        dst_directory: str, optional
            The directory containing joint data files, default "."
        dataset_name: str, optional
            The name of the dataset, default "dataset"
        prefix: str, optional
            Prefix appended to all data files, default None
        suffix: str, optional
            Suffix appended to all data files, default None
        batch_size: int, optional
            Size of batches to draw from data, defualt 32
        drop_last: bool, optional
            Whether to drop the last batch if it is smaller than batch_size, it is
            recommended to set this to true to avoid issues with mismatched sizes, default True
        input_variables: Sequence, optional
            List of input variable names, to be found in data file name, default "t2m"
        output_variables: Sequence, optional
            List of output variables names. If None, defaults to `input_variables`. default None
        constants: DictConfig, optional
            Dictionary with {key: value} corresponding to {constant_name: variable name in file}.
            default None
        scaling: DictConfig, optional
            Dictionary containing scaling parameters for data variables, default None
        splits: DictConfig, optional
            Dictionary with train/validation/test set start/end dates. If not provided, loads the entire
            data time series as the test set. default None
        presteps: int, optional
            Number of time steps to initialize recurrent hidden states. default 0
        input_time_dim: int, optional
            Number of time steps in the input array, default 1
        output_time_dim: int, optional
            Number of time steps in the output array, default 1
        data_time_step: Union[int, str], optional
            Either integer hours or a str interpretable by pandas: time between steps in the
            original data time series, default "3h"
        time_step: Union[int, str], optional
            Either integer hours or a str interpretable by pandas: desired time between effective model
            time steps, default "6h"
        gap: Union[int, str], optional
            either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. Defaults to `time_step`.
        shuffle: bool, optional
            Option to shuffle the training data, default True
        add_insolation: bool, optional
            Option to add prescribed insolation as a decoder input feature, default True
        num_workers: int, optional
            Number of parallel data loading workers, default 4
        pin_memory: bool, optional
            Whether pinned (page locked) memory should be used to store the tensors, improves GPU I/O, default True
        forecast_init_times: Sequence, optional
            A Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for. default None
            Note:
                - this is only applied to the test dataloader
                - providing this parameter configures the data loader to only produce this number of samples, and
                    NOT produce any target array.
        """
        super().__init__()

        warnings.warn(
            "TimeSeriesDataModule will be removed in a future release, please switch to using TimeSeriesDataModuleZarr",
            DeprecationWarning,
            stacklevel=2
        )

        self.dst_directory = dst_directory
        self.dataset_name = dataset_name
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.input_variables = input_variables
        self.output_variables = output_variables or input_variables
        self.constants = constants
        self.scaling = scaling
        self.splits = splits
        self.input_time_dim = input_time_dim + (presteps * input_time_dim)
        self.output_time_dim = output_time_dim
        self.data_time_step = data_time_step
        self.time_step = time_step
        self.gap = gap
        self.shuffle = shuffle
        self.add_insolation = add_insolation
        self.cube_dim = cube_dim
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.forecast_init_times = forecast_init_times

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.dataset_batch_size = self.batch_size
        self.dataloader_batch_size = None
        self.collate_fn = None

        self.setup()

    def get_constants(self) -> Optional[np.ndarray]:
        """Returns the constants used in this dataset

        Returns
        -------
        np.ndarray: The list of constants, None if there are no constants
        """
        if self.constants is None:
            return None

        return (
            self.train_dataset.get_constants()
            if self.train_dataset is not None
            else self.test_dataset.get_constants()
        )

    def setup(self) -> None:
        """Setup the datasets used for this DataModule"""
        # make sure distributed manager is initalized
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        dataset = open_time_series_dataset_classic_prebuilt(
            directory=self.dst_directory,
            dataset_name=self.dataset_name,
            constants=self.constants is not None,
            batch_size=self.batch_size,
        )

        # Verify that all input variables are available in the dataset
        missing_input_vars = set(self.input_variables) - set(dataset.channel_in.values)
        if missing_input_vars:
            raise ValueError(
                f"Input variables not found in dataset: {missing_input_vars}"
            )

        # Verify that all output variables are available in the dataset
        missing_output_vars = set(self.output_variables) - set(
            dataset.channel_out.values
        )
        if missing_output_vars:
            raise ValueError(
                f"Output variables not found in dataset: {missing_output_vars}"
            )

        dataset = dataset.sel(
            channel_in=self.input_variables,
            channel_out=self.output_variables,
        )
        if self.constants is not None:
            # Verify that all constants are available in the dataset
            missing_constants = set(list(self.constants.values())) - set(
                dataset.channel_c.values
            )
            if missing_constants:
                raise ValueError(f"Constants not found in dataset: {missing_constants}")

            dataset = dataset.sel(channel_c=list(self.constants.values()))

        if self.splits is not None and self.forecast_init_times is None:
            self.train_dataset = TimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["train_date_start"], self.splits["train_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
            )
            self.val_dataset = TimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["val_date_start"], self.splits["val_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                # drop_last=False,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
            )
            self.test_dataset = TimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["test_date_start"], self.splits["test_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
            )
        else:
            self.test_dataset = TimeSeriesDataset(
                dataset,
                scaling=self.scaling,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
                forecast_init_times=self.forecast_init_times,
            )

    def train_dataloader(self, num_shards=1, shard_id=0) -> DataLoader:
        """Setup the training dataloader

        Parameters
        ----------
        num_shards: int, optional
            The total total number of distributed shards
            default is 1 meaning distributed training is not being used
        shard_id: int, optional
            The shard number of this instance of the dataloader, default 0

        Returns
        -------
        DataLoader: The training dataloader
        """
        sampler = None
        shuffle = self.shuffle
        drop_last = False
        if num_shards > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=num_shards,
                rank=shard_id,
                shuffle=shuffle,
                drop_last=True,
            )
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset=self.train_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            collate_fn=self.collate_fn,
            batch_size=self.dataloader_batch_size,
        )

        return loader, sampler

    def val_dataloader(self, num_shards=1, shard_id=0) -> DataLoader:
        """Setup the validation dataloader

        Parameters
        ----------
        num_shards: int, optional
            The total total number of distributed shards
            default is 1 meaning distributed validation is not being used
        shard_id: int, optional
            The shard number of this instance of the dataloader, default 0

        Returns
        -------
        DataLoader: The validation dataloader
        """
        sampler = None
        if num_shards > 1:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=num_shards,
                rank=shard_id,
                shuffle=False,
                drop_last=False,
            )

        loader = DataLoader(
            dataset=self.val_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
            collate_fn=self.collate_fn,
            batch_size=self.dataloader_batch_size,
        )

        return loader, sampler

    def test_dataloader(self, num_shards=1, shard_id=0) -> DataLoader:
        """Setup the test dataloader

        Parameters
        ----------
        num_shards: int, optional
            The total total number of distributed shards
            default is 1 meaning distributed test is not being used
        shard_id: int, optional
            The shard number of this instance of the dataloader, default 0

        Returns
        -------
        DataLoader: The test dataloader
        """
        sampler = None
        if num_shards > 1:
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=num_shards,
                rank=shard_id,
                shuffle=False,
                drop_last=False,
            )

        loader = DataLoader(
            dataset=self.test_dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
            collate_fn=self.collate_fn,
            batch_size=self.dataloader_batch_size,
        )

        return loader, sampler


class CoupledTimeSeriesDataModule(TimeSeriesDataModule):
    """
    Extension of TimeSeriesDataModule, designed for coupled models that take input from other
    earth system components.
    """

    def __init__(
        self,
        src_directory: str = ".",  # kept for backwards compatibility
        dst_directory: str = ".",
        dataset_name: str = "dataset",
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        data_format: str = "classic",  # kept for backwards compatibility
        batch_size: int = 32,
        drop_last: bool = False,
        input_variables: Optional[Sequence] = None,
        output_variables: Optional[Sequence] = None,
        constants: Optional[DictConfig] = None,
        scaling: Optional[DictConfig] = None,
        splits: Optional[DictConfig] = None,
        presteps: int = 0,
        input_time_dim: int = 1,
        output_time_dim: int = 1,
        data_time_step: Union[int, str] = "3h",
        time_step: Union[int, str] = "6h",
        gap: Union[int, str, None] = None,
        shuffle: bool = True,
        add_insolation: bool = False,
        cube_dim: int = 64,  # kept for backwards compatibility
        num_workers: int = 4,
        pin_memory: bool = True,
        prebuilt_dataset: bool = True,  # kept for backwards compatibility
        forecast_init_times: Optional[Sequence] = None,
        couplings: Sequence = None,
        add_train_noise: Optional[bool] = False,
        train_noise_params: Optional[DictConfig] = None,
        train_noise_seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        dst_directory: str, optional
            The directory containing joint data files, default "."
        dataset_name: str, optional
            The name of the dataset, default "dataset"
        prefix: str, optional
            Prefix appended to all data files, default None
        suffix: str, optional
            Suffix appended to all data files, default None
        batch_size: int, optional
            Size of batches to draw from data, defualt 32
        drop_last: bool, optional
            Whether to drop the last batch if it is smaller than batch_size, it is
            recommended to set this to true to avoid issues with mismatched sizes, default True
        input_variables: Sequence, optional
            List of input variable names, to be found in data file name, default None
        output_variables: Sequence, optional
            List of output variables names. If None, defaults to `input_variables`. default None
        constants: DictConfig, optional
            Dictionary with {key: value} corresponding to {constant_name: variable name in file}.
            default None
        scaling: DictConfig, optional
            Dictionary containing scaling parameters for data variables, default None
        splits: DictConfig, optional
            Dictionary with train/validation/test set start/end dates. If not provided, loads the entire
            data time series as the test set. default None
        presteps: int, optional
            Number of time steps to initialize recurrent hidden states. default 0
        input_time_dim: int, optional
            Number of time steps in the input array, default 1
        output_time_dim: int, optional
            Number of time steps in the output array, default 1
        data_time_step: Union[int, str], optional
            Either integer hours or a str interpretable by pandas: time between steps in the
            original data time series, default "3h"
        time_step: Union[int, str], optional
            Either integer hours or a str interpretable by pandas: desired time between effective model
            time steps, default "6h"
        gap: Union[int, str, None], optional
            either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. default None.
        shuffle: bool, optional
            Option to shuffle the training data, default True
        add_insolation: bool, optional
            Option to add prescribed insolation as a decoder input feature, default False
        cube_dim: int, optional
            Number of points on the side of a cube face. Not currently used.
        num_workers: int, optional
            Number of parallel data loading workers, default 4
        pin_memory: bool, optional
            Whether pinned (page locked) memory should be used to store the tensors, improves GPU I/O, default True
        forecast_init_times: Sequence, optional
            A Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for. default None
            Note:
                - this is only applied to the test dataloader
                - providing this parameter configures the data loader to only produce this number of samples, and
                    NOT produce any target array.
        couplings: Sequence, optional
            a Sequence of dictionaries that define the mechanics of couplings with other earth system
            components. default None
        add_train_noise: bool, optional
            Add noise to the training data to inputs and integrated couplings to improve generalization, default False
        train_noise_params: DictConfig, optional
            Dictionary containing parameters for adding noise to the training data
        train_noise_seed: int, optional
            Seed for the random number generator for adding noise to the training data, default 42
        """
        self.couplings = couplings
        self.add_train_noise = add_train_noise
        self.train_noise_params = train_noise_params
        self.train_noise_seed = train_noise_seed

        super().__init__(
            src_directory,
            dst_directory,
            dataset_name,
            prefix,
            suffix,
            data_format,
            batch_size,
            drop_last,
            input_variables,
            output_variables,
            constants,
            scaling,
            splits,
            presteps,
            input_time_dim,
            output_time_dim,
            data_time_step,
            time_step,
            gap,
            shuffle,
            add_insolation,
            cube_dim,
            num_workers,
            pin_memory,
            prebuilt_dataset,
            forecast_init_times,
        )

    def _get_coupled_vars(self):

        coupled_variables = []
        for d in self.couplings:
            coupled_variables = coupled_variables + d["params"]["variables"]
        return coupled_variables

    def setup(self) -> None:
        """Setup the datasets used for this DataModule"""
        coupled_variables = self._get_coupled_vars()
        # make sure distributed manager is initalized
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        dist = DistributedManager()

        dataset = open_time_series_dataset_classic_prebuilt(
            directory=self.dst_directory,
            dataset_name=self.dataset_name,
            constants=self.constants is not None,
            batch_size=self.batch_size,
        )

        # Verify that all input variables are available in the dataset
        missing_input_vars = set(self.input_variables) - set(dataset.channel_in.values)
        if missing_input_vars:
            raise ValueError(
                f"Input variables not found in dataset: {missing_input_vars}"
            )

        # Verify that all output variables are available in the dataset
        missing_output_vars = set(self.output_variables) - set(
            dataset.channel_out.values
        )
        if missing_output_vars:
            raise ValueError(
                f"Output variables not found in dataset: {missing_output_vars}"
            )

        dataset = dataset.sel(
            channel_in=self.input_variables + coupled_variables,
            channel_out=self.output_variables,
        )
        if self.constants is not None:
            # Verify that all constants are available in the dataset
            missing_constants = set(list(self.constants.values())) - set(
                dataset.channel_c.values
            )
            if missing_constants:
                raise ValueError(f"Constants not found in dataset: {missing_constants}")
            dataset = dataset.sel(channel_c=list(self.constants.values()))

        if self.splits is not None and self.forecast_init_times is None:
            self.train_dataset = CoupledTimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["train_date_start"], self.splits["train_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_variables=self.input_variables,
                output_variables=self.output_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
                couplings=self.couplings,
                add_train_noise=self.add_train_noise,
                train_noise_params=self.train_noise_params,
                train_noise_seed=self.train_noise_seed + int(dist.rank),
            )
            self.val_dataset = CoupledTimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["val_date_start"], self.splits["val_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_variables=self.input_variables,
                output_variables=self.output_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                # drop_last=False,
                drop_last=self.drop_last,
                add_insolation=self.add_insolation,
                couplings=self.couplings,
            )
            self.test_dataset = CoupledTimeSeriesDataset(
                dataset.sel(
                    time=slice(
                        self.splits["test_date_start"], self.splits["test_date_end"]
                    )
                ),
                scaling=self.scaling,
                input_variables=self.input_variables,
                output_variables=self.output_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
                couplings=self.couplings,
            )
        else:
            self.test_dataset = CoupledTimeSeriesDataset(
                dataset,
                scaling=self.scaling,
                input_variables=self.input_variables,
                output_variables=self.output_variables,
                input_time_dim=self.input_time_dim,
                output_time_dim=self.output_time_dim,
                data_time_step=self.data_time_step,
                time_step=self.time_step,
                gap=self.gap,
                batch_size=self.dataset_batch_size,
                drop_last=False,
                add_insolation=self.add_insolation,
                forecast_init_times=self.forecast_init_times,
                couplings=self.couplings,
            )
