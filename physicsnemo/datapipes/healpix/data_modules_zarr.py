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
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

# numpy
import numpy as np

# External modules
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager

from .coupledtimeseries_dataset_zarr import CoupledTimeSeriesDatasetZarr
from .timeseries_dataset_zarr import TimeSeriesDatasetZarr

logger = logging.getLogger(__name__)


class TimeSeriesDataModuleZarr:
    """Module for complete model train, validation, and test data loading. Uses
    dlwp.data.data_loading.TimeSeriesDataset under-the-hood.
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        dataloader_batch_size: Optional[int] = None,
        drop_last: bool = True,
        input_variables: Optional[Sequence] = ["t2m"],
        output_variables: Optional[Sequence] = None,
        constant_variables: Optional[Sequence] = None,
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
        num_workers: int = 4,
        pin_memory: bool = True,
        forecast_init_times: Optional[Sequence] = None,
        add_train_noise: Optional[bool] = False,
        train_noise_params: Optional[DictConfig] = None,
        train_noise_seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        dataset_path: str, optional
            The path to the dataset, default "."
        batch_size: int, optional
            The number of sequential samples to load from the dataset to load, default 32
        dataloader_batch_size: int, optional
            Passed to nn.DataLoader as batch_size. Used to assemble batches of samples from the dataloader
            The total number of samples will be batch_size*dataloader_batch_size, default None
        drop_last: bool, optional
            Whether to drop the last batch if it is smaller than batch_size, it is
            recommended to set this to true to avoid issues with mismatched sizes, default True
        input_variables: Sequence, optional
            List of input variable names, to be found in data file name, default "t2m"
        output_variables: Sequence, optional
            List of output variables names. If None, defaults to `input_variables`. default None
        constant_variables: Sequence, optional
            List of constant variables names. default None
        scaling: DictConfig, optional
            Dictionary containing scaling parameters for data variables, default None
        splits: DictConfig
            Dictionary with train/validation/test set start/end dates.
        presteps: int, optional
            Number of time steps to initialize recurrent hidden states. default 0
        input_time_dim: int, optional
            Number of time steps in the input array, default 1
        output_time_dim: int, optional
            Number of time steps in the target/ground truth array, default 1
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
            Whether to shuffle the training data, default True
        add_insolation: bool, optional
            Whether to add prescribed insolation as a decoder input feature, default False
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
        add_train_noise: bool, optional
            Wether to add noise to the training data to inputs and integrated couplings to improve generalization, default False
        train_noise_params: DictConfig, optional
            Dictionary containing parameters for adding noise to the training data
        train_noise_seed: int, optional
            Seed for the random number generator for adding noise to the training data, default 42
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.dataset_batch_size = batch_size
        self.dataloader_batch_size = dataloader_batch_size
        self.drop_last = drop_last
        self.input_variables = input_variables
        self.output_variables = output_variables or input_variables
        self.constant_variables = constant_variables
        self.constants = constant_variables
        self.scaling = scaling
        self.splits = splits
        self.input_time_dim = input_time_dim + (presteps * input_time_dim)
        self.output_time_dim = output_time_dim
        self.data_time_step = data_time_step
        self.time_step = time_step
        self.gap = gap
        self.shuffle = shuffle
        self.add_insolation = add_insolation
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.forecast_init_times = forecast_init_times
        self.add_train_noise = add_train_noise
        self.train_noise_params = train_noise_params
        self.train_noise_seed = train_noise_seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.collate_fn = None

        self.setup()

    def get_constants(self) -> Optional[np.ndarray]:
        """Returns the constants used in this dataset

        Returns
        -------
        np.ndarray: The list of constants, None if there are no constants
        """
        if self.constant_variables is None:
            return None

        return (
            self.train_dataset.get_constants()
            if self.train_dataset is not None
            else self.test_dataset.get_constants()
        )

    def _get_common_dataset_kwargs(self) -> dict:
        """Get common keyword arguments for dataset creation"""
        return {
            "dataset_path": self.dataset_path,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "constant_variables": self.constant_variables,
            "batch_size": self.dataset_batch_size,
            "add_insolation": self.add_insolation,
            "input_time_dim": self.input_time_dim,
            "output_time_dim": self.output_time_dim,
            "data_time_step": self.data_time_step,
            "time_step": self.time_step,
            "gap": self.gap,
            "scaling": self.scaling,
        }

    def _validate_setup_requirements(self) -> None:
        """Validate that required setup conditions are met"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        if self.splits is None and self.forecast_init_times is None:
            raise ValueError("Either splits or forecast_init_times must be provided")

        # sanity check if dates overlap
        if self.splits:
            train_test_overlap = (
                np.datetime64(self.splits["train_date_end"])
                >= np.datetime64(self.splits["test_date_start"])
            ) and (
                np.datetime64(self.splits["train_date_start"])
                <= np.datetime64(self.splits["test_date_end"])
            )
            if train_test_overlap:
                warnings.warn("Training and test date ranges overlap")

            train_val_overlap = (
                np.datetime64(self.splits["train_date_end"])
                >= np.datetime64(self.splits["val_date_start"])
            ) and (
                np.datetime64(self.splits["train_date_start"])
                <= np.datetime64(self.splits["val_date_end"])
            )
            if train_val_overlap:
                warnings.warn("Training and validation date ranges overlap")

            test_val_overlap = (
                np.datetime64(self.splits["test_date_end"])
                >= np.datetime64(self.splits["val_date_start"])
            ) and (
                np.datetime64(self.splits["test_date_start"])
                <= np.datetime64(self.splits["val_date_end"])
            )
            if test_val_overlap:
                warnings.warn("Test and validation date ranges overlap")

    def _initialize_distributed_manager(self):
        """Initialize distributed manager if not already initialized"""
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        return DistributedManager()

    def _get_dataset_class(self):
        """Get the dataset class to use for creating datasets"""
        return TimeSeriesDatasetZarr

    def _create_datasets(self, common_kwargs: dict, dataset_class, dist) -> None:
        """Create train, validation, and test datasets based on configuration"""
        # forecast_init_times are provided just create the test dataset
        if self.forecast_init_times is not None:
            self.test_dataset = dataset_class(
                drop_last=False,
                **common_kwargs,
                forecast_init_times=self.forecast_init_times,
            )
        # splits are provided, use them to split the dataset
        else:
            self.train_dataset = dataset_class(
                start_date=self.splits["train_date_start"],
                end_date=self.splits["train_date_end"],
                drop_last=self.drop_last,
                add_train_noise=self.add_train_noise,
                train_noise_params=self.train_noise_params,
                train_noise_seed=self.train_noise_seed + int(dist.rank),
                **common_kwargs,
            )
            self.val_dataset = dataset_class(
                start_date=self.splits["val_date_start"],
                end_date=self.splits["val_date_end"],
                drop_last=self.drop_last,
                **common_kwargs,
            )
            self.test_dataset = dataset_class(
                start_date=self.splits["test_date_start"],
                end_date=self.splits["test_date_end"],
                drop_last=False,
                **common_kwargs,
            )

    def setup(self) -> None:
        """Setup the datasets used for this DataModule"""
        self._validate_setup_requirements()
        dist = self._initialize_distributed_manager()

        common_kwargs = self._get_common_dataset_kwargs()
        dataset_class = self._get_dataset_class()

        self._create_datasets(common_kwargs, dataset_class, dist)

    def _base_dataloader(
        self, dataset, num_shards=1, shard_id=0, shuffle=False, drop_last=False
    ) -> DataLoader:
        """Setup a dataloader with common functionality

        Parameters
        ----------
        dataset: Dataset
            The dataset to create the dataloader for
        num_shards: int, optional
            The total total number of distributed shards
            default is 1 meaning distributed training is not being used
        shard_id: int, optional
            The shard number of this instance of the dataloader, default 0
        shuffle: bool, optional
            Whether to shuffle the data, default False

        Returns
        -------
        DataLoader: The configured dataloader
        """
        sampler = None
        drop_last = False
        if num_shards > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_shards,
                rank=shard_id,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset=dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=sampler,
            collate_fn=self.collate_fn,
            batch_size=self.dataloader_batch_size,
        )

        return loader, sampler

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
        return self._base_dataloader(
            dataset=self.train_dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            shuffle=self.shuffle,
            drop_last=True,
        )

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
        return self._base_dataloader(
            dataset=self.val_dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            shuffle=False,
            drop_last=True,
        )

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
        return self._base_dataloader(
            dataset=self.test_dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            shuffle=False,
            drop_last=True,
        )


class CoupledTimeSeriesDataModuleZarr(TimeSeriesDataModuleZarr):
    """
    Extension of TimeSeriesDataModule, designed for coupled models that take input from other
    earth system components.
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        dataloader_batch_size: Optional[int] = None,
        drop_last: bool = True,
        input_variables: Optional[Sequence] = None,
        output_variables: Optional[Sequence] = None,
        constant_variables: Optional[Sequence] = None,
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
        num_workers: int = 4,
        pin_memory: bool = True,
        forecast_init_times: Optional[Sequence] = None,
        couplings: Sequence = None,
        add_train_noise: Optional[bool] = False,
        train_noise_params: Optional[DictConfig] = None,
        train_noise_seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        dataset_path: str, optional
            The path to the dataset, default "."
        batch_size: int, optional
            The number of sequential samples to load from the dataset to load, default 32
        dataloader_batch_size: int, optional
            Passed to nn.DataLoader as batch_size. Used to assemble batches of samples from the dataloader
            The total number of samples will be dataloader_batch_size*dataloader_batch_size, default None
        drop_last: bool, optional
            Whether to drop the last batch if it is smaller than batch_size, it is
            recommended to set this to true to avoid issues with mismatched sizes, default True
        input_variables: Sequence, optional
            List of input variable names, to be found in data file name, default None
        output_variables: Sequence, optional
            List of output variables names. If None, defaults to `input_variables`. default None
        constant_variables: Sequence, optional
            List of constant variables names. default None
        scaling: DictConfig, optional
            Dictionary containing scaling parameters for data variables, default None
        splits: DictConfig
            Dictionary with train/validation/test set start/end dates.
        presteps: int, optional
            Number of time steps to initialize recurrent hidden states. default 0
        input_time_dim: int, optional
            Number of time steps in the input array, default 1
        output_time_dim: int, optional
            Number of time steps in the target/ground truth array, default 1
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
            Whether to shuffle the training data, default True
        add_insolation: bool, optional
            Whether to add prescribed insolation as a decoder input feature, default False
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
            Wether to add noise to the training data to inputs and integrated couplings to improve generalization, default False
        train_noise_params: DictConfig, optional
            Dictionary containing parameters for adding noise to the training data
        train_noise_seed: int, optional
            Seed for the random number generator for adding noise to the training data, default 42
        """
        self.couplings = couplings

        super().__init__(
            dataset_path,
            batch_size,
            dataloader_batch_size,
            drop_last,
            input_variables,
            output_variables,
            constant_variables,
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
            num_workers,
            pin_memory,
            forecast_init_times,
            add_train_noise,
            train_noise_params,
            train_noise_seed,
        )

    def _get_coupled_vars(self):
        """Get the coupled variables from the couplings"""
        coupled_variables = []
        for d in self.couplings:
            coupled_variables = coupled_variables + d["params"]["variables"]
        return coupled_variables

    def _get_common_dataset_kwargs(self) -> dict:
        """Get common keyword arguments for dataset creation (includes coupling-specific params)"""
        base_kwargs = super()._get_common_dataset_kwargs()
        base_kwargs.update(
            {
                "couplings": self.couplings,
            }
        )
        return base_kwargs

    def _get_dataset_class(self):
        """Get the dataset class to use for creating datasets"""
        return CoupledTimeSeriesDatasetZarr
