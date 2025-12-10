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

"""Shared pytest fixtures for DeepONet tests."""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def device():
    """Return available device for the test session."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def sample_input_tensor(device):
    """Create a sample input tensor for CO2 sequestration models."""
    # Shape: (batch, H, W, T, channels)
    # Using smaller dimensions for faster tests
    return torch.randn(2, 32, 64, 16, 12).to(device)


@pytest.fixture
def sample_target_tensor(device):
    """Create a sample target tensor."""
    # Shape: (batch, H, W, T)
    return torch.randn(2, 32, 64, 16).to(device)


@pytest.fixture
def sample_inputs_with_grid(device):
    """Create sample inputs with grid coordinates for loss functions."""
    B, H, W, T, C = 2, 32, 64, 16, 12
    inputs = torch.randn(B, H, W, T, C).to(device)

    # Set grid_x channel (channel -3) with increasing values
    grid_x = torch.linspace(0, 100, W).to(device)
    inputs[..., -3] = grid_x.view(1, 1, W, 1).expand(B, H, W, T)

    # Set grid_y channel (channel -2) with increasing values
    grid_y = torch.linspace(0, 50, H).to(device)
    inputs[..., -2] = grid_y.view(1, H, 1, 1).expand(B, H, W, T)

    # Set time channel (channel -1) with increasing values
    grid_t = torch.linspace(0, 30, T).to(device)
    inputs[..., -1] = grid_t.view(1, 1, 1, T).expand(B, H, W, T)

    return inputs


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU is available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
