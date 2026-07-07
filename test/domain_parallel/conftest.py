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

from physicsnemo.core.version_check import check_version_spec

if not check_version_spec("torch", "2.6.0", hard_fail=False):
    pytest.skip(
        "These tests require torch >= 2.6.0",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def skip_on_cpu(device):
    if device == "cpu":
        pytest.skip("Skip tests on cpu")


@pytest.fixture(autouse=True)
def disable_tf32():
    # The harness compares sharded vs. single-GPU fp32 paths at ~1e-4; TF32
    # (~1e-3 relative error) makes them diverge. Force full fp32.
    prev_cudnn = torch.backends.cudnn.allow_tf32
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
