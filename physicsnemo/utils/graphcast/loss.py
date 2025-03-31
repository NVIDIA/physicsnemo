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
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable


class CellAreaWeightedLossFunction(nn.Module):
    """Loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """

        loss = (invar - outvar) ** 2
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss


class CustomCellAreaWeightedLossAutogradFunction(torch.autograd.Function):
    """Autograd fuunction for custom loss with cell area weighting."""

    @staticmethod
    def forward(ctx, invar: torch.Tensor, outvar: torch.Tensor, area: torch.Tensor):
        """Forward of custom loss function with cell area weighting."""

        diff = invar - outvar  # T x C x H x W
        loss = diff**2
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, area)
        loss = loss.mean()
        loss_grad = diff * (2.0 / (math.prod(invar.shape)))
        loss_grad *= area.unsqueeze(0).unsqueeze(0)
        ctx.save_for_backward(loss_grad)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss: torch.Tensor):
        """Backward method of custom loss function with cell area weighting."""

        # grad_loss should be 1, multiply nevertheless
        # to avoid issues with cases where this isn't the case
        (grad_invar,) = ctx.saved_tensors
        return grad_invar * grad_loss, None, None


class CustomCellAreaWeightedLossFunction(CellAreaWeightedLossFunction):
    """Custom loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area: torch.Tensor):
        super().__init__(area)

    def forward(self, invar: torch.Tensor, outvar: torch.Tensor) -> torch.Tensor:
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """

        return CustomCellAreaWeightedLossAutogradFunction.apply(
            invar, outvar, self.area
        )


class GraphCastLossFunction(nn.Module):
    """Loss function as specified in GraphCast.
    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area, channels_list, dataset_metadata_path, time_diff_std_path):
        super().__init__()
        self.area = area
        self.channel_dict = self.get_channel_dict(dataset_metadata_path, channels_list)
        self.variable_weights = self.assign_variable_weights()
        self.time_diff_std = self.get_time_diff_std(time_diff_std_path, channels_list)

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.
        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [T, C, H, W].
        outvar : torch.Tensor
            target values of shape [T, C, H, W].
        """

        # outvar normalization
        loss = (invar - outvar) ** 2  # [T,C,H,W]
        # weighted by inverse variance
        loss = (
            loss
            * 1.0
            / torch.square(self.time_diff_std.view(1, -1, 1, 1).to(loss.device))
        )
        # weighted by variables
        variable_weights = self.variable_weights.view(1, -1, 1, 1).to(loss.device)
        loss = loss * variable_weights  # [T,C,H,W]
        # weighted by area
        loss = loss.mean(dim=(0, 1))
        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss

    def get_time_diff_std(self, time_diff_std_path, channels_list):
        """Gets the time difference standard deviation"""
        if time_diff_std_path is not None:
            time_diff_np = np.load(time_diff_std_path)
            time_diff_np = time_diff_np[:, channels_list, ...]
            return torch.FloatTensor(time_diff_np)
        else:
            return torch.tensor([1.0], dtype=torch.float)

    def get_channel_dict(self, dataset_metadata_path, channels_list):
        """Gets lists of surface and atmospheric channels"""
        with open(dataset_metadata_path, "r") as f:
            data_json = json.load(f)
            channel_list = [data_json["coords"]["channel"][c] for c in channels_list]

            # separate atmosphere and surface variables
            channel_dict = {"surface": [], "atmosphere": []}
            for each_channel in channel_list:
                if each_channel[-1].isdigit():
                    channel_dict["atmosphere"].append(each_channel)
                else:
                    channel_dict["surface"].append(each_channel)
            return channel_dict

    def parse_variable(self, variable_list):
        """Parse variable into its letter and numeric parts."""
        for i, char in enumerate(variable_list):
            if char.isdigit():
                return variable_list[:i], int(variable_list[i:])

    def calculate_linear_weights(self, variables):
        """Calculate weights for each variable group."""
        groups = defaultdict(list)
        # Group variables by their first letter
        for variable in variables:
            letter, number = self.parse_variable(variable)
            groups[letter].append((variable, number))
        # Calculate weights for each group
        weights = {}
        for values in groups.values():
            total = sum(number for _, number in values)
            for variable, number in values:
                weights[variable] = number / total

        return weights

    def assign_surface_weights(self):
        """Assigns weights to surface variables"""
        surface_weights = {i: 0.1 for i in self.channel_dict["surface"]}
        if "t2m" in surface_weights:
            surface_weights["t2m"] = 1
        return surface_weights

    def assign_atmosphere_weights(self):
        """Assigns weights to atmospheric variables"""
        return self.calculate_linear_weights(self.channel_dict["atmosphere"])

    def assign_variable_weights(self):
        """assigns per-variable per-pressure level weights"""
        surface_weights_dict = self.assign_surface_weights()
        atmosphere_weights_dict = self.assign_atmosphere_weights()
        surface_weights = list(surface_weights_dict.values())
        atmosphere_weights = list(atmosphere_weights_dict.values())
        variable_weights = torch.cat(
            (torch.FloatTensor(surface_weights), torch.FloatTensor(atmosphere_weights))
        )  # [num_channel]
        return variable_weights
