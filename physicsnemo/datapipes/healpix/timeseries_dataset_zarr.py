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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

from physicsnemo.datapipes.meta import DatapipeMetaData
from physicsnemo.utils.insolation import insolation

from .base_timeseries_dataset_zarr import BaseTimeSeriesDatasetZarr

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


class TimeSeriesDatasetZarr(BaseTimeSeriesDatasetZarr):
    """Dataset for sampling from continuous time-series data stored in Zarr format.

    This class implements the basic time series dataset functionality without coupling
    to external data sources. It provides data loading, scaling, and batching
    capabilities for training and inference.
    """

    def __init__(
        self,
        dataset_path: str,
        input_variables: Sequence,
        output_variables: Sequence = None,
        constant_variables: Sequence = None,
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
        start_date: Optional[Union[int, str]] = None,
        end_date: Optional[Union[int, str]] = None,
        add_train_noise: bool = False,
        train_noise_params: DictConfig = None,
        train_noise_seed: int = 42,
        meta: DatapipeMetaData = MetaData(),
    ):
        """Initialize time series dataset.

        Parameters are same as BaseTimeSeriesDatasetZarr.
        See base class for detailed parameter descriptions.
        """
        super().__init__(
            dataset_path=dataset_path,
            input_variables=input_variables,
            output_variables=output_variables,
            constant_variables=constant_variables,
            scaling=scaling,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            data_time_step=data_time_step,
            time_step=time_step,
            gap=gap,
            batch_size=batch_size,
            drop_last=drop_last,
            add_insolation=add_insolation,
            forecast_init_times=forecast_init_times,
            start_date=start_date,
            end_date=end_date,
            add_train_noise=add_train_noise,
            train_noise_params=train_noise_params,
            train_noise_seed=train_noise_seed,
            meta=meta,
        )

    def __getitem__(
        self, item: int
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
        """Get a batch of time series data.

        This implementation provides the basic time series data loading functionality:
        1. Load data window for batch
        2. Apply scaling
        3. Extract input and target sequences
        4. Add optional insolation and constant inputs
        5. Format dimensions appropriately

        Parameters
        ----------
        item : int
            Sample index

        Returns
        -------
        Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]
            In forecast mode: List of input arrays
            In training mode: Tuple of (input arrays, target array)

            Input arrays are in order:
            - Model inputs [B, F, T, C, H, W]
            - Insolation (if enabled) [B, F, T, 1, H, W]
            - Constants (if provided) [F, C, H, W]

            Target array has shape [B, F, T, C, H, W]
            where:
            B = batch size
            F = faces
            T = time steps
            C = channels
            H = height
            W = width

        Raises
        ------
        IndexError
            If item is out of range
        """
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

        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:load_staging_data")
        # data from the input and target arrays overlap, to avoid 2 seperate loads
        # we load both and then slice it later
        staging_ds = self.ds["inputs"][slice(*time_index)]
        staging_ds = staging_ds[:, self.all_variable_indices]
        torch.cuda.nvtx.range_pop()

        # we scale this dataset to avoid doing twice the work when we scale as both
        # input and output
        # we do the scaling as in place operations to avoid creating temp arrays
        # that result in a lot of data movement. This is around 4x faster than using
        # standard operations
        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:scale_batch")
        if self.all_scaling is not None:
            staging_ds -= self.all_scaling["mean"]
            staging_ds /= self.all_scaling["std"]

        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:load_input")
        input_array = staging_ds[:, self.input_variable_indices]
        torch.cuda.nvtx.range_pop()

        if not self.forecast_mode:
            torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:load_target")
            target_array = staging_ds[:, self.output_variable_indices]
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:process_batch")

        # Calculate insolation if needed
        if self.add_insolation:
            sol = insolation(
                self._get_forecast_sol_times(item),
                self.lat,
                self.lon,
            )[:, None]
            decoder_inputs = np.empty(
                (this_batch, self.input_time_dim + self.output_time_dim, 1)
                + self.spatial_dims,
                dtype="float32",
            )

        # Get buffers for the batches, which we'll fill in iteratively.
        inputs = np.empty(
            (this_batch, self.input_time_dim, len(self.input_variables))
            + self.spatial_dims,
            dtype="float32",
        )
        if not self.forecast_mode:
            targets = np.empty(
                (this_batch, self.output_time_dim, len(self.output_variables))
                + self.spatial_dims,
                dtype="float32",
            )

        # Iterate over valid sample windows
        torch.cuda.nvtx.range_push(
            "TimeSeriesDataset:__getitem__:copy_inputs_targets_insolation"
        )
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
        torch.cuda.nvtx.range_pop()

        if not self.forecast_mode and self.add_train_noise:
            torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:add_train_noise")
            # Iterate over C: inputs.shape = [B, T, C, F, H, W]
            for i in range(inputs.shape[2]):
                inputs[:, :, i] += self.rng.normal(
                    loc=0,
                    scale=self.train_noise_params["inputs"][self.input_variables[i]][
                        "std"
                    ],
                    size=inputs[:, :, i].shape,
                )
            torch.cuda.nvtx.range_pop()

        inputs_result = [inputs]
        torch.cuda.nvtx.range_push(
            "CoupledTimeSeriesDataset:__getitem__:add_insolation"
        )
        if self.add_insolation:
            inputs_result.append(decoder_inputs)
        torch.cuda.nvtx.range_pop()

        # Transpose dimensions to match expected format
        inputs_result = [
            np.transpose(x, axes=(0, 3, 1, 2, 4, 5)) for x in inputs_result
        ]

        if self.constant_variables:
            # Add constants
            inputs_result.append(self.constants)
        torch.cuda.nvtx.range_pop()

        if self.forecast_mode:
            torch.cuda.nvtx.range_pop()
            return inputs_result

        # Transpose targets to match input format
        targets = np.transpose(targets, axes=(0, 3, 1, 2, 4, 5))

        torch.cuda.nvtx.range_pop()
        return inputs_result, targets
