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

"""Shared pytest configuration for the steel bridge example."""

import sys
from pathlib import Path

import pytest
import torch

# Make the example modules importable when pytest is started from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True, scope="session")
def single_thread():
    """Keep the CPU tests small and deterministic."""
    torch.set_num_threads(1)
    yield
