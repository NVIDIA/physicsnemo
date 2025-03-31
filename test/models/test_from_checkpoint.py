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

from pathlib import Path

import pytest
import torch

import physicsnemo


class MockModel(physicsnemo.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


class NewMockModel(physicsnemo.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("LoadModel", [MockModel, NewMockModel])
def test_from_checkpoint_custom(device, LoadModel):
    """Test checkpointing custom physicsnemo module"""
    torch.manual_seed(0)

    # Construct Mock Model and save it
    mock_model = MockModel().to(device)
    mock_model.save("checkpoint.mdlus")

    # Load from checkpoint using class
    LoadModel.from_checkpoint("checkpoint.mdlus")
    # Delete checkpoint file (it should exist!)
    Path("checkpoint.mdlus").unlink(missing_ok=False)
