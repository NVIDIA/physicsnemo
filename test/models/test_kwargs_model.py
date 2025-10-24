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

    def __init__(self, input_size=16, output_size=16, **other_kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.other_kwargs = other_kwargs
        self.layer = torch.nn.Linear(input_size, output_size)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("LoadModel", [MockModel])
def test_kwargs(device, LoadModel):
    """Test checkpointing custom physicsnemo module"""
    torch.manual_seed(0)

    # Construct Mock Model and save it
    input_size = 4
    output_size = 8
    other_kwargs = {"a": 1, "b": 2}
    mock_model = LoadModel(input_size, output_size=output_size, **other_kwargs).to(
        device
    )
    mock_model.save("checkpoint.mdlus")

    # Load from checkpoint using class
    LoadModel.from_checkpoint("checkpoint.mdlus")

    # Check that the model was loaded correctly
    assert mock_model.input_size == input_size
    assert mock_model.output_size == output_size
    assert mock_model.other_kwargs == other_kwargs

    # Delete checkpoint file (it should exist!)
    Path("checkpoint.mdlus").unlink(missing_ok=False)
