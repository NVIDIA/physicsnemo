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

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData
from modulus.utils.insolation import insolation

logger = logging.getLogger(__name__)


@dataclass
class MetaData(DatapipeMetaData):
    """Metadata for this datapipe"""

    name: str = "TimeSeries"
    # Optimization
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = False


class TimeSeriesDataset(Dataset, Datapipe):
    """
    Dataset for sampling from continuous time-series data, compatible with pytorch data loading.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        scaling: DictConfig = None,
        input_time_dim: int = 1,
        output_time_dim: int = 1,
        data_time_step: Union[int, str] = "3h",
        time_step: Union[int, str] = "6h",
        gap: Union[int, str, None] = None,
        batch_size: int = 32,
        drop_last: bool = False,
        add_insolation: bool = False,
        forecast_init_times: Optional[Sequence] = None,
        meta: DatapipeMetaData = MetaData(),
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset produced by one of the `open_*` methods herein
        scaling: DictConfig
            Dictionary containing scaling parameters for data variables
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
        batch_size: int, optional
            Size of batches to draw from data, default 32
        drop_last: bool, optional
            Whether to drop the last batch if it is smaller than batch_size, it is
            recommended to set this to true to avoid issues with mismatched sizes, default False
        add_insolation: bool, optional
            Option to add prescribed insolation as a decoder input feature, default True
        forecast_init_times: Sequence, optional
            A Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for.  default None
            Note that:
                - providing this parameter configures the data loader to only produce this number of samples, and
                    NOT produce any target array.
        meta: DatapipeMetaData, optional
            Data class for storing essential meta data
        """
        Datapipe.__init__(
            self,
            meta=meta,
        )
        self.ds = dataset
        self.scaling = OmegaConf.to_object(scaling) if scaling else None
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.data_time_step = self._convert_time_step(data_time_step)
        self.time_step = self._convert_time_step(time_step)
        self.gap = self._convert_time_step(gap if gap is not None else time_step)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.add_insolation = add_insolation
        self.forecast_init_times = forecast_init_times
        self.forecast_mode = self.forecast_init_times is not None

        # Time stepping
        if (self.time_step % self.data_time_step).total_seconds() != 0:
            raise ValueError(
                f"'time_step' must be a multiple of 'data_time_step' "
                f"(got {self.time_step} and {self.data_time_step}"
            )
        if (self.gap % self.data_time_step).total_seconds() != 0:
            raise ValueError(
                f"'gap' must be a multiple of 'data_time_step' "
                f"(got {self.gap} and {self.data_time_step}"
            )
        self.interval = self.time_step // self.data_time_step

        # Find indices of init times for forecast mode
        if self.forecast_mode:
            if self.batch_size != 1:
                self.batch_size = 1
                warnings.warn(
                    "providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`; "
                    "setting it now"
                )
            self._forecast_init_indices = np.array(
                [
                    int(np.where(self.ds["time"] == s)[0][0])
                    for s in self.forecast_init_times
                ],
                dtype="int",
            ) - ((self.input_time_dim - 1) * self.interval)
        else:
            self._forecast_init_indices = None

        # Length of the data window needed for one sample.
        if self.forecast_mode:
            self._window_length = self.interval * (self.input_time_dim - 1) + 1
        else:
            self._window_length = (
                self.interval * (self.input_time_dim - 1)
                + 1
                + (self.gap // self.data_time_step)
                + self.interval
                * (self.output_time_dim - 1)  # first point is counted by gap
            )
        self._batch_window_length = self.batch_size + self._window_length - 1
        self._output_delay = self.interval * (self.input_time_dim - 1) + (
            self.gap // self.data_time_step
        )
        # Indices within a batch
        self._input_indices = [
            list(range(n, n + self.interval * self.input_time_dim, self.interval))
            for n in range(self.batch_size)
        ]
        self._output_indices = [
            list(
                range(
                    n + self._output_delay,
                    n + self.interval * self.output_time_dim + self._output_delay,
                    self.interval,
                )
            )
            for n in range(self.batch_size)
        ]

        self.spatial_dims = (
            self.ds.sizes["face"],
            self.ds.sizes["height"],
            self.ds.sizes["width"],
        )

        self.input_scaling = None
        self.target_scaling = None
        self.constant_scaling = None
        self.constants = None
        if self.scaling:
            self._get_scaling_da()

        # setup constants
        if "constants" in self.ds.data_vars:
            # extract from ds:
            const = self.ds.constants.values

            if self.constant_scaling:
                const = (const - self.constant_scaling["mean"]) / self.constant_scaling[
                    "std"
                ]

            # transpose to match new format:
            # [C, F, H, W] -> [F, C, H, W]
            self.constants = np.transpose(const, axes=(1, 0, 2, 3))

    def get_constants(self):
        """Returns the constants used in this dataset

        Returns
        -------
        np.ndarray: The list of constants, None if there are no constants
        """
        return self.constants

    @staticmethod
    def _convert_time_step(dt):  # pylint: disable=invalid-name
        """convert time step to Timedelta

        Parameters
        ----------
        dt: int or sequence
            Timestep as tuple or int

        Returns
        -------
        pd.Timedelta
            The dt as a Timedelta
        """
        return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)

    def _get_scaling_da(self):
        """Setup the scaling values for this dataset"""
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
        scaling_da = scaling_df.to_xarray().astype("float32")

        # REMARK: we remove the xarray overhead from these
        try:
            # we use channel_out instead of channel_in because
            # the list of input channels may contain data fetched outside
            # the datasets such as coupled fields
            self.input_scaling = scaling_da.sel(
                index=self.ds.channel_out.values
            ).rename({"index": "channel_in"})
            self.input_scaling = {
                "mean": np.expand_dims(
                    self.input_scaling["mean"].to_numpy(), (0, 2, 3, 4)
                ),
                "std": np.expand_dims(
                    self.input_scaling["std"].to_numpy(), (0, 2, 3, 4)
                ),
            }
        except (ValueError, KeyError):
            missing = [
                m
                for m in self.ds.channel_in.values
                if m not in list(self.scaling.keys())
            ]
            raise KeyError(
                f"Input channels {missing} not found in the scaling config dict data.scaling ({list(self.scaling.keys())})"
            )
        try:
            self.target_scaling = scaling_da.sel(
                index=self.ds.channel_out.values
            ).rename({"index": "channel_out"})
            self.target_scaling = {
                "mean": np.expand_dims(
                    self.target_scaling["mean"].to_numpy(), (0, 2, 3, 4)
                ),
                "std": np.expand_dims(
                    self.target_scaling["std"].to_numpy(), (0, 2, 3, 4)
                ),
            }
        except (ValueError, KeyError):
            missing = [
                m
                for m in self.ds.channel_out.values
                if m not in list(self.scaling.keys())
            ]
            raise KeyError(
                f"Target channels {missing} not found in the scaling config dict data.scaling ({list(self.scaling.keys())})"
            )

        try:
            # not all datasets will have constants
            if "constants" in self.ds.data_vars:
                self.constant_scaling = scaling_da.sel(
                    index=self.ds.channel_c.values
                ).rename({"index": "channel_out"})
                self.constant_scaling = {
                    "mean": np.expand_dims(
                        self.constant_scaling["mean"].to_numpy(), (1, 2, 3)
                    ),
                    "std": np.expand_dims(
                        self.constant_scaling["std"].to_numpy(), (1, 2, 3)
                    ),
                }
        except (ValueError, KeyError):
            missing = [
                m
                for m in self.ds.channel_c.values
                if m not in list(self.scaling.keys())
            ]
            raise KeyError(
                f"Constant channels {missing} not found in the scaling config dict data.scaling ({list(self.scaling.keys())})"
            )

    def __len__(self):
        """Get number of samples in the dataset
        Returns
        -------
        int
            Number of samples available
        """
        if self.forecast_mode:
            return len(self._forecast_init_indices)
        length = (self.ds.sizes["time"] - self._window_length + 1) / self.batch_size
        if self.drop_last:
            return int(np.floor(length))
        return int(np.ceil(length))

    def _get_time_index(self, item):
        """Get the indices for the specified sample

        Parameters
        ----------
        item: int
            The smaple number for which indices are requested

        Returns
        -------
        tuple[tuple[int, int], int]
        """
        start_index = (
            self._forecast_init_indices[item]
            if self.forecast_mode
            else item * self.batch_size
        )
        # TODO: I think this should be -1 and still work (currently missing the last sample in last batch)
        max_index = (
            start_index + self._window_length
            if self.forecast_mode
            else (item + 1) * self.batch_size + self._window_length
        )
        if not self.drop_last and max_index > self.ds.sizes["time"]:
            batch_size = self.batch_size - (max_index - self.ds.sizes["time"])
        else:
            batch_size = self.batch_size
        return (start_index, max_index), batch_size

    def _get_forecast_sol_times(self, item):
        """Get the inoslation time for a specified sample

        Parameters
        ----------
        item: int
            The sample # for which to calculate insolation time

        Returns
        -------
        np.array
            The list of times for the specified sample
        """
        item
        time_index, _ = self._get_time_index(item)
        if self.forecast_mode:
            timedeltas = (
                np.array(self._input_indices[0] + self._output_indices[0])
                * self.data_time_step
            )
            return self.ds.time[time_index[0]].values + timedeltas
        return self.ds.time[slice(*time_index)].values

    def __getitem__(self, item):
        """Returns the requested sample

        Parameters
        ----------
        item: int
            The sample number

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            The requsted sample
            If in forecast mode only a target tensor is returned
            If in training mode an input and target tensor are returned

        """
        # start range
        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__")

        if item < 0:
            item = len(self) + item
        if item < 0 or item > len(self):
            raise IndexError(
                f"index {item} out of range for dataset with length {len(self)}"
            )

        # remark: load first then normalize
        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:load_batch")
        time_index, this_batch = self._get_time_index(item)
        batch = {"time": slice(*time_index)}
        load_time = time.time()

        input_array = self.ds["inputs"].isel(**batch).to_numpy()
        input_array = (input_array - self.input_scaling["mean"]) / self.input_scaling[
            "std"
        ]

        if not self.forecast_mode:
            target_array = self.ds["targets"].isel(**batch).to_numpy()
            target_array = (
                target_array - self.target_scaling["mean"]
            ) / self.target_scaling["std"]

        logger.log(5, "loaded batch data in %0.2f s", time.time() - load_time)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:process_batch")
        compute_time = time.time()
        # Insolation
        if self.add_insolation:
            sol = insolation(
                self._get_forecast_sol_times(item),
                self.ds.lat.values,
                self.ds.lon.values,
            )[:, None]
            decoder_inputs = np.empty(
                (this_batch, self.input_time_dim + self.output_time_dim, 1)
                + self.spatial_dims,
                dtype="float32",
            )

        # Get buffers for the batches, which we'll fill in iteratively.
        inputs = np.empty(
            (this_batch, self.input_time_dim, self.ds.sizes["channel_in"])
            + self.spatial_dims,
            dtype="float32",
        )
        if not self.forecast_mode:
            targets = np.empty(
                (this_batch, self.output_time_dim, self.ds.sizes["channel_out"])
                + self.spatial_dims,
                dtype="float32",
            )

        # Iterate over valid sample windows
        for sample in range(this_batch):
            inputs[sample] = input_array[self._input_indices[sample]]
            if not self.forecast_mode:
                targets[sample] = target_array[self._output_indices[sample]]
            if self.add_insolation:
                decoder_inputs[sample] = (
                    sol
                    if self.forecast_mode
                    else sol[self._input_indices[sample] + self._output_indices[sample]]
                )

        inputs_result = [inputs]
        if self.add_insolation:
            inputs_result.append(decoder_inputs)

        # we need to transpose channels and data:
        # [B, T, C, F, H, W] -> [B, F, T, C, H, W]
        inputs_result = [
            np.transpose(x, axes=(0, 3, 1, 2, 4, 5)) for x in inputs_result
        ]

        if self.constants is not None:
            # Add the constants as [F, C, H, W]
            inputs_result.append(self.constants)

        logger.log(5, "computed batch in %0.2f s", time.time() - compute_time)
        torch.cuda.nvtx.range_pop()

        # finish range
        torch.cuda.nvtx.range_pop()

        if self.forecast_mode:
            return inputs_result

        # we also need to transpose targets
        targets = np.transpose(targets, axes=(0, 3, 1, 2, 4, 5))

        return inputs_result, targets
