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

"""Tests for moving Python values to a device without a synchronization."""

import pytest
import torch

from physicsnemo.mesh.utilities._device import to_device
from test.mesh.mesh.test_slicing_sync import _cuda_sync_budget


def test_matching_tensors_are_returned_unchanged(device):
    tensor = torch.ones(3, device=device)
    assert to_device(tensor, tensor.device, tensor.dtype) is tensor


@pytest.mark.parametrize("value", [2.0, [1.0, 2.0, 3.0], torch.arange(3)])
def test_values_convert_like_as_tensor(value, device):
    expected = torch.as_tensor(value, device=device, dtype=torch.float64)
    torch.testing.assert_close(
        to_device(value, expected.device, torch.float64), expected, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_python_values_move_without_a_synchronization():
    device = torch.device("cuda")
    to_device([1.0, 2.0, 3.0], device)  # warm up lazy CUDA initialization
    torch.cuda.synchronize()

    with _cuda_sync_budget(0), torch.device("cuda"):
        to_device([1.0, 2.0, 3.0], device)
        to_device(2.0, device, torch.float64)
    torch.cuda.synchronize()
