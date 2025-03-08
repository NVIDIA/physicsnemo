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

from typing import Sequence

import numpy as np
import torch as th
import xarray as xr

"""
Custom dlwp compatible loss classes that allow for more sophisticated training optimization.

Each custom loss should inherit all methods of th.nn._Loss base class or subclasses thereof. 
Additionally, custom loss classes should define a setup function which receives the trainer object. 
The setup function should be used to move tensors to appropriate gpus and finalize configuration
of the loss calculation using information about the model (trainer.model) and trainer. Custom
losses should also redefine the forward function to contain a flag indicating whether or not to 
average output channels. This is used in the varible wise logging of validation loss by the trainer. 

"""


class BaseMSE(th.nn.MSELoss):
    """
    Base MSE class offers impementaion for basic MSE loss compatable with dlwp custom loss training
    """

    def __init__(
        self,
    ):
        """Constructer for BaseMSE"""
        super().__init__()
        self.device = None

    def setup(self, trainer):
        """
        Nothing to implement here
        """
        pass

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the base MSE class
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError("Expected predictions to have 6 dimensions")

        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.mean(d)
        else:
            return d


class WeightedMSE(th.nn.MSELoss):

    """
    Loss object that allows for user defined weighting of variables when calculating MSE
    """

    def __init__(
        self,
        weights: Sequence = [],
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.device = None

    def setup(self, trainer):
        """
        pushes weights to cuda device
        """

        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedMSE pass
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError("Expected predictions to have 6 dimensions")

        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5)) * self.loss_weights
        if average_channels:
            return th.mean(d)
        else:
            return d


class OceanMSE(th.nn.MSELoss):
    """
    Ocean MSE class offers impementaion for MSE loss weighted by a land-sea-mask field.
    """

    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "lsm"},
    ):
        """
        Parameters
        ----------
        lsm_file: str
            land-sea-mask file
        open_dict: dict, optional
            dictionary that store land-sea-mask file information
        selection_dict: dict, optional
            dictionary that store channel selection information
        """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None

    def setup(self, trainer):
        """
        reshape lsm and put on device
        """
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(
            self.selection_dict
        )
        # 1-lsm gives the percentage of pixel that has ocean
        self.lsm_tensor = 1 - th.tensor(
            np.expand_dims(self.lsm_ds.values, (0, 2, 3))
        ).to(trainer.device)

    def forward(self, prediction, target, average_channels=True):

        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(
                dim=(0, 1, 2, 4, 5)
            )
            self.lsm_sum_calculated = True
        # average weighted
        ocean_err = ((target - prediction) ** 2) * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.sum(ocean_mean_err) / self.lsm_sum
        else:
            return ocean_mean_err / self.lsm_var_sum


class WeightedOceanMSE(th.nn.MSELoss):
    """
    Ocean MSE class offers impementaion for MSE loss with:
    1) weighted by a land-sea-mask field.
    2) weighted by channel (e.g. sic more than sst)
    """

    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "lsm"},
        weights: Sequence = [],
    ):
        """ """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None
        self.loss_weights = th.tensor(weights)

    def setup(self, trainer):
        """
        reshape lsm and put on device
        pushes weights to cuda device
        """
        ### 1. OCEAN PREP ###
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(
            self.selection_dict
        )
        # 1-lsm gives the percentage of pixel that has ocean
        self.lsm_tensor = 1 - th.tensor(
            np.expand_dims(self.lsm_ds.values, (0, 2, 3))
        ).to(trainer.device)

        ### 2. WEIGHTS PREP ###

        if not len(trainer.output_variables) == len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True):

        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(
                dim=(0, 1, 2, 4, 5)
            )
            self.lsm_sum_calculated = True
        # average weighted
        ocean_err = ((target - prediction) ** 2) * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        ocean_mean_err = ocean_mean_err * self.loss_weights

        if average_channels:
            return th.sum(ocean_mean_err) / self.lsm_sum
        else:
            return ocean_mean_err / self.lsm_var_sum
