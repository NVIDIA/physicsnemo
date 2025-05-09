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
from typing import Sequence

import numpy as np
import pandas as pd
import torch as th
import xarray as xr

logger = logging.getLogger(__name__)


class ConstantCoupler:
    """
    coupler used to interface two component of earth system

    constant coupler will take the the coupled field at integration time and
    force the model with this field consistently
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size: int,
        variables: Sequence,
        presteps: int = 0,
        input_time_dim: int = 2,
        output_time_dim: int = 2,
        input_times: Sequence = [pd.Timedelta("24h"), pd.Timedelta("48h")],
        prepared_coupled_data=True,
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset that holds coupled data
        batch_size: int
            number of batch size during training.
            forecasting batch size should be 1
        variables: Sequence
            sequence of strings that indicate the coupled variable
            names in the dataset
        presteps: int, optional
            the number of model steps used to initialize the hidden state.
            If not using a GRU, prestep is 0, default 0
        input_time_dim: int, optional
            number of input times into the model, default 2
        output_time_dim: int, optional
            number of output times for each model step, default 2
        input_times: Sequence, optional
            sequence of pandas Timedelta objects that indicate which times are to be coupled,
            default [pd.Timedelta("24h"), pd.Timedelta("48h")]
        prepared_coupled_data: boolean, optional
            If True assumes data in dataset has been prepared approiately for training:
            averages have already been calculated so that each time step denotes
            the right side of a averaging_window window.
            This is highly remcommended for training, default True
        """
        # extract important meta data from ds
        self.ds = dataset
        self.batch_size = batch_size
        self.spatial_dims = self.ds.inputs.shape[2:]
        self.variables = variables
        self.presteps = presteps
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.coupled_integration_dim = self._compute_coupled_integration_dim()
        self.input_times = [pd.Timedelta(t) for t in input_times]
        self.output_channels = len(self.variables) * len(self.input_times)
        self.timevar_dim = self._compute_timevar_dim()
        self.coupled_inputs_shape = None
        self.scaling_dict = None
        self._coupled_offsets = None
        self.coupled_mode = False
        self.integrated_couplings = None

        if not prepared_coupled_data:
            logger.log(
                logging.DEBUG,
                "Assuming coupled data is not preprocessed preparing data.",
            )
            self._prepare_coupled_data()
        else:
            logger.log(
                logging.DEBUG,
                "**Assuming coupled data has been prepared properly, using coupled field[s] from "
                'dataset "as-is"**',
            )

    def _prepare_coupled_data(self):
        # TODO: write function to lazily compute average as spcified in time scheme
        raise NotImplementedError("Data preparation not yet implemented")

    def _compute_coupled_integration_dim(self):

        return self.presteps + max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_timevar_dim(self):

        return len(self.input_times) * len(self.variables)

    def compute_coupled_indices(self, interval, data_time_step):

        """
        Called by CoupledDataset to compute static indices for training
        samples

        Parameters
        ----------
        interval: int
            ratio of dataset timestep to model dt
        data_time_step:
            dataset timestep
        """
        # create array of static coupled offstes that accompany each batch
        self._coupled_offsets = np.empty(
            [self.batch_size, self.coupled_integration_dim, len(self.input_times)]
        )
        for b in range(self.batch_size):
            for i in range(self.coupled_integration_dim):

                self._coupled_offsets[b, i, :] = b + np.array(
                    [ts / data_time_step for ts in self.input_times]
                )

        self._coupled_offsets = self._coupled_offsets.astype(int)

    def set_scaling(self, scaling_da):

        """
        Called by CoupledDataset to compute static indices for training
        samples

        Parameters
        ----------
        scaling_da: xarray.DataArray
            values used to scale input data, uses mean and std
        """
        coupled_scaling = scaling_da.sel(index=self.variables).rename(
            {"index": "channel_in"}
        )
        self.coupled_scaling = {
            "mean": np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
            "std": np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4)),
        }

    def setup_coupling(self, coupled_module):
        # To expediate the coupling process the coupled_forecast
        # get proper channels from coupled component output
        output_channels = coupled_module.output_variables
        # A bit convoluted. Prepared coupled variables
        # are given a suffix for training associated with their
        # trailing average increment e.g. 'z1000-48H'. To extract
        # thr proper field from the coupled model output, we see if
        # we check if the coupled model output var is in self.variables.
        #
        # for example 'z1000' is in 'z1000-48H'
        channel_indices = [
            i
            for i, oc in enumerate(output_channels)
            for v in self.variables
            if oc == v.split("-")[0]
        ]
        self.coupled_channel_indices = channel_indices

    def reset_coupler(self):

        self.coupled_mode = False
        self.integrated_couplings = None
        self.preset_coupled_fields = None

    def set_coupled_fields(self, coupled_fields: th.tensor):
        """
        Set the data for the coupled field for the next iteration of the dataloader.
        Instead of loading data from the dataset the data from coupled_fields will
        be returned instead.

        Parameters
        ----------
        coupled_fields: th.tensor
            The data to use when the dataloader requests coupled fields. Expected
            format is [B, F, T, C, H, W]
        """
        if coupled_fields.shape[0] != self.batch_size:
            raise ValueError(
                f"Batch size of coupled field {coupled_fields.shape[0]} doesn't "
                f" match configured batch size {self.batch_size}"
            )
        # create buffer for coupling
        coupled_fields = coupled_fields[
            :, :, :, self.coupled_channel_indices, :, :
        ].permute(2, 0, 3, 1, 4, 5)
        self.preset_coupled_fields = th.empty(
            [self.coupled_integration_dim, self.batch_size, self.timevar_dim]
            + list(self.spatial_dims)
        )
        # we use a constant set of values so we just copy time 0
        for i in range(len(self.preset_coupled_fields)):
            self.preset_coupled_fields[i, :, :, :, :, :] = coupled_fields[
                0, :, -1, :, :, :
            ]
        # flag for construct integrated coupling method to use this array
        self.coupled_mode = True

    def construct_integrated_couplings(
        self,
        batch=None,
        bsize=None,
    ):
        """
        Construct array of coupled inputs that includes values required for
        model integration steps.

        Parameters
        ----------
        batch: Sequence
            indices of dataset sample dimension associated with current batch
        bsize: int
            batch size

        Returns
        -------
        numpy.ndarray: The coupled data
        """
        if self.coupled_mode:
            return self.preset_coupled_fields
        else:
            # reset integrated couplings
            self.integrated_couplings = np.empty(
                (bsize, self.coupled_integration_dim, self.timevar_dim)
                + self.spatial_dims
            )

            index_range = slice(
                batch["time"].start,
                batch["time"].start + self._coupled_offsets[-1, -1, -1] + 1,
            )

            # extract coupled variables and scale lazily
            input_array = self.ds.inputs.sel(channel_in=self.variables)
            ds = (input_array - self.coupled_scaling["mean"]) / self.coupled_scaling[
                "std"
            ]
            # load before entering loop for efficiency
            ds_index_range = ds.isel(time=index_range).load()

            # use static offsets to create integrated coupling array
            for b in range(bsize):
                for i in range(self.coupled_integration_dim):
                    coupling_temp = ds_index_range.isel(
                        time=self._coupled_offsets[b, i, :]
                    )
                    self.integrated_couplings[
                        b, i, :, :, :
                    ] = coupling_temp.to_numpy().reshape(
                        (self.timevar_dim,) + coupling_temp.shape[2:]
                    )

            return self.integrated_couplings.transpose((1, 0, 2, 3, 4, 5)).astype(
                "float32"
            )  # cast to float for compatability


class TrailingAverageCoupler:
    """
    coupler used to inferface two components of the earth system

    Trailing average coupler uses coupled input times as the right side of
    an averag that is taken over an "averaging_window" window size.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size: int,
        variables: Sequence,
        presteps: int = 0,
        input_time_dim: int = 2,
        output_time_dim: int = 2,
        averaging_window: str = "24h",
        input_times: Sequence = [pd.Timedelta("24h"), pd.Timedelta("48h")],
        prepared_coupled_data=True,
    ):
        """
        Parameters
        ----------
        dataset: xr.Dataset
            xarray Dataset that holds coupled data
        batch_size: int
            number of batch size during training.
            forecasting batch size should be 1
        variables: Sequence
            sequence of strings that indicate the coupled variable
            names in the dataset
        presteps: int, optional
            the number of model steps used to initialize the hidden state.
            If not using a GRU, prestep is 0, default 0
        input_time_dim: int, optional
            number of input times into the model, default 2
        output_time_dim: int, optional
            number of output times for each model step, default 2
        averaging_window: str, optional
            period over which coupled data is averaged before sent back to model, default "24h"
        input_times: Sequence, optional
            sequence of pandas Timedelta objects that indicate which times are to be coupled,
            default [pd.Timedelta("24h"), pd.Timedelta("48h")]
        prepared_coupled_data: boolean, optional
            If True assumes data in dataset has been prepared approiately for training:
            averages have already been calculated so that each time step denotes
            the right side of a averaging_window window.
            This is highly remcommended for training, default True
        """
        # extract important meta data from ds
        self.ds = dataset
        self.batch_size = batch_size
        self.spatial_dims = self.ds.inputs.shape[2:]
        self.variables = variables
        self.presteps = presteps
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.input_times = [pd.Timedelta(t) for t in input_times]
        self.averaging_window = pd.Timedelta(averaging_window)
        self.output_channels = len(self.variables) * len(self.input_times)
        self._set_time_increments()
        self.coupled_integration_dim = self._compute_coupled_integration_dim()
        self.timevar_dim = self._compute_timevar_dim()
        self.coupled_inputs_shape = None
        self.scaling_dict = None
        self._coupled_offsets = None
        self.integrated_couplings = None
        self.coupled_mode = False  # if forecasting with another coupled model

        if not prepared_coupled_data:
            logger.log(
                logging.DEBUG,
                "Assuming coupled data is not preprocessed, averaging fields in as designed in"
                "TrailingAverageCoupler. See docs for specifics.",
            )
            self._prepare_coupled_data()
        else:
            logger.log(
                logging.DEBUG,
                "**Assuming coupled data has been prepared properly, using coupled field[s] from"
                'dataset "as-is"**',
            )

    def compute_coupled_indices(self, interval, data_time_step):

        """
        Called by CoupledDataset to compute static indices for training
        samples

        :param interval: int ratio of dataset timestep to model dt
        :param data_time_step: dataset timestep
        """

        # create array of static coupled offstes that accompany each batch
        self._coupled_offsets = np.empty(
            [self.batch_size, self.coupled_integration_dim, len(self.input_times)]
        )
        for b in range(self.batch_size):
            for i in range(self.coupled_integration_dim):

                self._coupled_offsets[b, i, :] = (
                    b
                    + (self.input_time_dim * i + 1) * interval
                    + np.array([ts / data_time_step for ts in self.input_times])
                )

        self._coupled_offsets = self._coupled_offsets.astype(int)

    def _prepare_coupled_data(self):

        # TODO: write function to lazily compute average as spcified in time scheme
        raise NotImplementedError("Data preparation not yet implemented")

    def set_scaling(self, scaling_da):

        coupled_scaling = scaling_da.sel(index=self.variables).rename(
            {"index": "channel_in"}
        )
        self.coupled_scaling = {
            "mean": np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
            "std": np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4)),
        }

    def _set_time_increments(self):

        # get the dt of the dataset
        dt = pd.Timedelta(
            self.ds.time[1].values - self.ds.time[0].values
        ).total_seconds()
        # assert that the time increments are divisible by the dt of the dataset
        if np.any([t.total_seconds() % dt != 0 for t in self.input_times]):
            raise ValueError(
                f"Coupled input times {self.input_times} "
                f"({[t.total_seconds() for t in self.input_times]} in secs) are not divisible by dataset dt: {dt}"
            )
        self.time_increments = [t.total_seconds() / dt for t in self.input_times]

    def _compute_timevar_dim(self):

        return len(self.input_times) * len(self.variables)

    def _compute_coupled_integration_dim(self):

        return self.presteps + max(self.output_time_dim // self.input_time_dim, 1)

    def setup_coupling(self, coupled_module):

        # To expediate the coupling process the coupled_forecast
        # get proper channels from coupled component output
        output_channels = coupled_module.output_variables
        # A bit convoluted. Prepared coupled variables
        # are given a suffix for training associated with their
        # trailing average increment e.g. 'z1000-48H'. To extract
        # thr proper field from the coupled model output, we see if
        # we check if the coupled model output var is in self.variables.
        #
        # for example 'z1000' is in 'z1000-48H'
        channel_indices = [
            i
            for i, oc in enumerate(output_channels)
            for v in self.variables
            if oc == v.split("-")[0]
        ]
        self.coupled_channel_indices = channel_indices

        # find averaging periods from componenet output
        averaging_window_max_indices = [
            i // pd.Timedelta(coupled_module.time_step) for i in self.input_times
        ]
        di = averaging_window_max_indices[0]
        # TODO: Now support output_time_dim =/= input_time_dim, but presteps need to be 0, will add support for presteps>0
        averaging_slices = []
        for j in range(self.coupled_integration_dim):
            averaging_slices.append([])
            for i, r in enumerate(averaging_window_max_indices):
                averaging_slices[j].append(
                    slice(
                        self.input_time_dim * j * di + i * di,
                        self.input_time_dim * j * di + r,
                    )
                )
        self.averaging_slices = averaging_slices

    def reset_coupler(self):

        self.coupled_mode = False
        self.integrated_couplings = None
        self.preset_coupled_fields = None

    def set_coupled_fields(self, coupled_fields):
        """
        Set the data for the coupled field for the next iteration of the dataloader.
        Instead of loading data from the dataset the data from coupled_fields will
        be returned instead.

        Parameters
        ----------
        coupled_fields: th.tensor
            The data to use when the dataloader requests coupled fields. Expected
            format is [B, F, T, C, H, W]
        """
        if coupled_fields.shape[0] != self.batch_size:
            raise ValueError(
                f"Batch size of coupled field {coupled_fields.shape[0]} doesn't "
                f" match configured batch size {self.batch_size}"
            )

        coupled_fields = coupled_fields[:, :, :, self.coupled_channel_indices, :, :]
        # TODO: Now support output_time_dim =/= input_time_dim, but presteps need to be 0, will add support for presteps>0
        coupled_averaging_periods = []
        for j in range(self.coupled_integration_dim):
            averaging_periods = [
                coupled_fields[:, :, s, :, :, :].mean(dim=2, keepdim=True)
                for s in self.averaging_slices[j]
            ]
            coupled_averaging_periods.append(th.concat(averaging_periods, dim=3))
        self.preset_coupled_fields = th.concat(
            coupled_averaging_periods, dim=2
        ).permute(2, 0, 3, 1, 4, 5)
        # flag for construct integrated coupling method to use this array
        self.coupled_mode = True

    def construct_integrated_couplings(
        self,
        batch=None,
        bsize=None,
    ):

        """
        Construct array of coupled inputs that includes values required for
        model integration steps.

        :param batch: indices of dataset sample dimension associated with
               current batch.
        :param bsize: int batch size
        """

        if self.coupled_mode:
            return self.preset_coupled_fields
        else:
            if (batch is None) or (bsize is None):
                raise ValueError(
                    "Coupled fields must be set if no batch or batch size is provided"
                )
            # reset integrated couplings
            self.integrated_couplings = np.empty(
                (bsize, self.coupled_integration_dim, self.timevar_dim)
                + self.spatial_dims
            )

            index_range = slice(
                batch["time"].start,
                batch["time"].start + self._coupled_offsets[-1, -1, -1] + 1,
            )

            # extract coupled variables and scale lazily
            ds = (
                self.ds.inputs.sel(channel_in=self.variables)
                - self.coupled_scaling["mean"]
            ) / self.coupled_scaling["std"]
            # load before entering loop for efficiency
            ds_index_range = ds.isel(time=index_range).load()

            # use static offsets to create integrated coupling array
            for b in range(bsize):
                for i in range(self.coupled_integration_dim):
                    # coupling_temp = \
                    #   ds.isel(time=batch["time"].start+self._coupled_offsets[b,i,:]) # changed i to 0 for debugging
                    # added to test speed, original is commented above
                    coupling_temp = ds_index_range.isel(
                        time=self._coupled_offsets[b, i, :]
                    )
                    self.integrated_couplings[
                        b, i, :, :, :
                    ] = coupling_temp.to_numpy().reshape(
                        (self.timevar_dim,) + coupling_temp.shape[2:]
                    )

            return self.integrated_couplings.transpose((1, 0, 2, 3, 4, 5)).astype(
                "float32"
            )  # cast to float32 for pytroch compatibility.
