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
This file contains specific MLPs for the DoMINO model.

The main feature here is we've locked in the number of layers.
"""

import torch.nn as nn

from physicsnemo.nn import Mlp


class AggregationModel(Mlp):
    """
    Neural network module to aggregate local geometry encoding with basis functions.

    This module combines basis function representations with geometry encodings
    to predict the final output quantities. It serves as the final prediction layer
    that integrates all available information sources.

    It is implemented as a straightforward MLP with 5 total layers.

    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        base_layer: int,
        activation: nn.Module,
    ):
        hidden_features = [base_layer, base_layer, base_layer, base_layer]

        super().__init__(
            in_features=input_features,
            hidden_features=hidden_features,
            out_features=output_features,
            act_layer=activation,
            drop=0.0,
        )


class LocalPointConv(Mlp):
    """Layer for local geometry point kernel

    This is a straight forward MLP, with exactly two layers.
    """

    def __init__(
        self,
        input_features: int,
        base_layer: int,
        output_features: int,
        activation: nn.Module,
    ):
        super().__init__(
            in_features=input_features,
            hidden_features=base_layer,
            out_features=output_features,
            act_layer=activation,
            drop=0.0,
        )
