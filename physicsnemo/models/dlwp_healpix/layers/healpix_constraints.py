# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""
Implementation of the Deep Learning Weather Prediction (DLWP) constraints on the HEALPix mesh.

This module contains the implementation of the Deep Learning Weather Prediction (DLWP) constraints on the HEALPix mesh.
The main classes are:
- NonnegativeConstraint: A class for the DLWP nonnegative constraint on the HEALPix mesh.
"""

import torch


class NonnegativeConstraint(torch.nn.Module):
    """Clamp selected output channels to be nonnegative in physical units.

    Applies a lower bound of zero (in unnormalized/physical space) to the
    specified ``variables`` by clamping the corresponding normalized channels
    to their per-variable, per-scaling threshold.
    """

    def __init__(
        self,
        variables: list[str],
        channels: list[str],
        scaling: dict[str, dict[str, float]],
    ):
        """
        Parameters
        ----------
        variables: list[str]
            List of variable names to apply the constraint to.
        channels: list[str]
            List of all input channel names in the model.
        scaling: dict[str, dict[str, float]]
            Dictionary containing the mean and std for each variable.
        """
        super().__init__()
        self.variables = variables
        self.channels = channels
        self.scaling = scaling

        # Only apply constraint to variables that are used by model
        self.variables = [var for var in self.variables if var in channels]

        var_indices = torch.tensor(
            [channels.index(var) for var in self.variables], dtype=torch.long
        )
        self.register_buffer("var_indices", var_indices, persistent=False)

        self.var_means = torch.tensor([scaling[var]["mean"] for var in self.variables])
        self.var_stds = torch.tensor([scaling[var]["std"] for var in self.variables])

        thresholds = (0.0 - self.var_means) / self.var_stds
        thresholds = thresholds.view(1, 1, 1, -1, 1, 1)
        self.register_buffer("thresholds", thresholds, persistent=False)

    def forward(self, x):
        """
        Tensors are expected to be in the shape [B, F, T, C, H, W]
        """
        selected_vars = torch.index_select(x, dim=3, index=self.var_indices)
        clamped = torch.maximum(selected_vars, self.thresholds).to(x.dtype)
        x.index_copy_(3, self.var_indices, clamped)

        return x
