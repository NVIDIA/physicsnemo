# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch

from physicsnemo.nn import Mlp
from test.common import (
    validate_forward_accuracy,
)


def test_mlp_forward_accuracy(device):
    torch.manual_seed(7)
    target_device = torch.device(device)

    model = Mlp(in_features=10, hidden_features=20, out_features=5).to(target_device)
    input_tensor = torch.randn(1, 10).to(
        target_device
    )  # Assuming a batch size of 1 for simplicity
    model(input_tensor)

    # Relative to test/
    file_name = "nn/module/data/mlp_output.pth"

    # Tack this on for the test, since model is not a physicsnemo Module:
    model.device = target_device

    assert validate_forward_accuracy(
        model,
        (input_tensor,),
        file_name=file_name,
        atol=1e-3,
    )


def test_mlp_activation_and_dropout():
    model = Mlp(in_features=10, hidden_features=20, out_features=5, drop=0.5)
    input_tensor = torch.randn(2, 10)  # Batch size of 2

    output_tensor = model(input_tensor)

    assert output_tensor.shape == torch.Size([2, 5])


def test_mlp_different_activation():
    model = Mlp(
        in_features=10, hidden_features=20, out_features=7, act_layer=torch.nn.ReLU
    )
    input_tensor = torch.randn(3, 10)  # Batch size of 3

    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([3, 7])


def test_multiple_hidden_layers():
    model = Mlp(in_features=10, hidden_features=[20, 30], out_features=5)
    input_tensor = torch.randn(4, 10)  # Batch size of 4

    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([4, 5])
