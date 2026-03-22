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


@pytest.fixture(autouse=True)
def disable_tf32():
    """Disable TF32 for deterministic float32 precision across GPU architectures.

    TensorFloat-32 (TF32) is enabled by default on Ampere and newer GPUs (A100, etc.),
    which reduces float32 matrix multiplication precision from 23-bit to 10-bit mantissa.
    This can cause tests to pass on older GPUs but fail on newer ones due to ~1e-3 to 1e-4
    precision differences. Disabling TF32 ensures consistent behavior across all GPUs.
    """
    if not torch.cuda.is_available():
        yield
        return

    orig_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_matmul
    torch.backends.cudnn.allow_tf32 = orig_cudnn
