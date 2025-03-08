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
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from modulus.datapipes.meta import DatapipeMetaData
from modulus.utils.insolation import insolation

from . import couplers
from .timeseries_dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


@dataclass
class MetaData(DatapipeMetaData):
    """Metadata for this datapipe"""

    name: str = "CoupledTimeSeries"
    # Optimization
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = False


class CoupledTimeSeriesDataset(TimeSeriesDataset):
    """
    Dataset for coupling TimesSeriesDataset with external inputs from various earth system components
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        scaling: DictConfig,
        input_variables: Sequence,
        output_variables: Sequence = None,
        input_time_dim: int = 1,
        presteps: int = 0,
        output_time_dim: int = 1,
        data_time_step: Union[int, str] = "3h",
        time_step: Union[int, str] = "6h",
        gap: Union[int, str, None] = None,
        batch_size: int = 32,
        drop_last: bool = False,
        add_insolation: bool = False,
        forecast_init_times: Optional[Sequence] = None,
        couplings: Sequence = [],
        meta: DatapipeMetaData = MetaData(),
        add_train_noise: bool = False,
        train_noise_params: DictConfig = None,
        train_noise_seed: int = 42,
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset produced by one of the `open_*` methods herein
        scaling: DictConfig
            Dictionary containing scaling parameters for data variables
        input_variables: Sequence
            a sequence of variables that will be ingested in to model
        output_variables: Sequence, optional
            a sequence of variables that are outputs of the model, default None
        input_time_dim: int, optional
            Number of time steps in the input array, default 1
        presteps: int, optional
            number of steps to initialize GRU, default 0
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
        couplings: Sequence, optional
            a Sequence of dictionaries that define the mechanics of couplings with other earth system
            components
        add_train_noise: bool, optional
            Add noise to the training data to inputs and integrated couplings to improve generalization, default False
        train_noise_params: DictConfig, optional
            Dictionary containing parameters for adding noise to the training data
        train_noise_seed: int, optional
            Seed for the random number generator for adding noise to the training data, default 42
        """
        self.input_variables = input_variables
        self.output_variables = (
            input_variables if output_variables is None else output_variables
        )
        self.add_train_noise = add_train_noise
        self.train_noise_params = train_noise_params
        if self.add_train_noise:
            self.rng = np.random.default_rng(train_noise_seed)

        self.couplings = [
            getattr(couplers, c["coupler"])(
                dataset,
                **OmegaConf.to_object(DictConfig(c))["params"],
            )
            for c in couplings
        ]
        super().__init__(
            dataset=dataset,
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
            meta=meta,
        )
        # calculate static indices for coupling
        for c in self.couplings:
            c.compute_coupled_indices(self.interval, self.data_time_step)
        # keep track of integration steps
        self.integration_step = (
            1  # starts at 1 because first step is done by __getitem__
        )
        self.curr_item = None  # keeps track of current initialization

    def _get_scaling_da(self):
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
        scaling_da = scaling_df.to_xarray().astype("float32")

        # only thing we do different here is get the scaling for the coupled values
        for c in self.couplings:
            c.set_scaling(scaling_da)
        super()._get_scaling_da()

    def __getitem__(self, item):
        # start range
        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__")

        if item < 0:
            item = len(self) + item
        if item < 0 or item > len(self):
            raise IndexError(
                f"index {item} out of range for dataset with length {len(self)}"
            )

        # remark: load first then normalize
        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__:load_batch")
        time_index, this_batch = self._get_time_index(item)
        batch = {"time": slice(*time_index)}
        load_time = time.time()

        input_array = (
            self.ds["inputs"]
            .sel(channel_in=self.input_variables)
            .isel(**batch)
            .to_numpy()
        )
        # retrieve coupled inputs
        if len(self.couplings) > 0:
            integrated_couplings = np.concatenate(
                [
                    c.construct_integrated_couplings(batch, this_batch)
                    for c in self.couplings
                ],
                axis=2,
            )

        input_array = (input_array - self.input_scaling["mean"]) / self.input_scaling[
            "std"
        ]
        if not self.forecast_mode:
            # BAD NEWS: Indexing the array as commented out below causes unexpected behavior in target creation.
            #     leaving this in here as a warning
            # target_array = self.ds['targets'].isel(**batch).to_numpy()
            target_array = (
                self.ds["targets"]
                .sel(channel_out=self.output_variables)
                .isel(**batch)
                .to_numpy()
            )
            target_array = (
                target_array - self.target_scaling["mean"]
            ) / self.target_scaling["std"]
            # target_array = ((self.ds['targets'].isel(**batch) - self.target_scaling['mean']) /
            #                self.target_scaling['std']).compute()

        logger.log(5, "loaded batch data in %0.2f s", time.time() - load_time)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__:process_batch")
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
            # update current item and reset integration_step counter for further integrations which need
            # insolation but bypass this method see method "next_integration()" for details
            self.curr_item = item
            self.integration_step = 1

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

        if not self.forecast_mode and self.add_train_noise:
            logger.log(5, "Adding gaussian noise to inputs and integrated_couplings")
            # Iterate over C: inputs.shape = [B, T, C, F, H, W]
            for i in range(inputs.shape[2]):
                inputs[:, :, i] += self.rng.normal(
                    loc=0,
                    scale=self.train_noise_params["inputs"][self.input_variables[i]][
                        "std"
                    ],
                    size=inputs[:, :, i].shape,
                )
            for c in self.couplings:
                for i, v in enumerate(c.variables):
                    integrated_couplings[i, :, :] += self.rng.normal(
                        loc=0,
                        scale=self.train_noise_params["couplings"][v]["std"],
                        size=integrated_couplings[i, :, :].shape,
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

        # append integrated couplings
        if len(self.couplings) > 0:
            inputs_result.append(integrated_couplings)

        torch.cuda.nvtx.range_pop()

        # finish range
        torch.cuda.nvtx.range_pop()

        if self.forecast_mode:
            return inputs_result

        # we also need to transpose targets
        targets = np.transpose(targets, axes=(0, 3, 1, 2, 4, 5))

        return inputs_result, targets

    def next_integration(self, model_outputs, constants):
        inputs_result = []

        # grab last few model outputs for re-initialization
        init_time_dim = len(self._input_indices[0])
        prognostic_inputs = model_outputs[:, :, 0 - init_time_dim :]
        inputs_result.append(prognostic_inputs)

        # gather insolation inputs
        time_offset = self.time_step * (self.output_time_dim) * self.integration_step
        sol = torch.tensor(
            insolation(
                self._get_forecast_sol_times(self.curr_item) + time_offset,
                self.ds.lat.values,
                self.ds.lon.values,
            )[:, None]
        )
        decoder_inputs = np.empty(
            (1, self.input_time_dim + self.output_time_dim, 1) + self.spatial_dims,
            dtype="float32",
        )
        decoder_inputs[0] = sol
        inputs_result.append(torch.tensor(decoder_inputs.transpose(0, 3, 1, 2, 4, 5)))

        # append constant fields
        inputs_result.append(constants)
        # increment integration step
        self.integration_step += 1

        # append couplings inputs
        if len(self.couplings) > 0:
            integrated_couplings = np.concatenate(
                [c.construct_integrated_couplings() for c in self.couplings], axis=2
            )
            inputs_result.append(torch.tensor(integrated_couplings))

        # gather coupled_inputs
        return inputs_result
