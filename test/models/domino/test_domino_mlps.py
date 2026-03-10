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

import pytest
import torch

from .utils import validate_output_shape_and_values


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_aggregation_model(device, activation):
    """Test AggregationModel"""
    from physicsnemo.models.domino.mlps import AggregationModel
    from physicsnemo.models.domino.model import get_activation

    model = AggregationModel(
        input_features=100,
        output_features=1,
        base_layer=64,
        activation=get_activation(activation),
    ).to(device)

    x = torch.randn(2, 30, 100).to(device)
    output = model(x)

    validate_output_shape_and_values(output, (2, 30, 1))


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_local_point_conv(device, activation):
    """Test LocalPointConv"""
    from physicsnemo.models.domino.mlps import LocalPointConv
    from physicsnemo.models.domino.model import get_activation

    model = LocalPointConv(
        input_features=50,
        base_layer=128,
        output_features=32,
        activation=get_activation(activation),
    ).to(device)

    x = torch.randn(2, 100, 50).to(device)
    output = model(x)

    validate_output_shape_and_values(output, (2, 100, 32))
