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

"""Shared fixtures and constants for diffusion tests."""

import random

import pytest
import torch
import torch._dynamo

# =============================================================================
# Shared Constants
# =============================================================================

GLOBAL_SEED = 42

CPU_TOLERANCES = {"atol": 1e-3, "rtol": 1e-3}
GPU_TOLERANCES = {"atol": 1e-2, "rtol": 5e-2}


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch._dynamo state between tests to avoid cross-test recompile errors."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def deterministic_settings():
    """Set deterministic settings for reproducibility, then restore old state."""
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_random_state = random.getstate()

    try:
        random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GLOBAL_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        random.setstate(old_random_state)


@pytest.fixture
def tolerances(device):
    """Return tolerances based on the device (CPU vs GPU)."""
    if device == "cpu":
        return CPU_TOLERANCES
    return GPU_TOLERANCES
