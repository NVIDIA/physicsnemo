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
"""Tests for prefetch_map background processing."""

import pytest
import torch

from physicsnemo.experimental.datapipes.healda.prefetch import prefetch_map


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_map_basic():
    """Prefetch with a simple doubling transform."""
    data = list(range(10))
    loader = prefetch_map(data, lambda x: 2 * x)
    assert list(loader) == list(range(0, 20, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_map_error_propagation():
    """Exceptions in the background thread propagate to the consumer."""
    data = list(range(4))

    def failing_transform(x):
        raise ValueError("Test error")

    loader = prefetch_map(data, failing_transform)
    with pytest.raises(ValueError, match="Test error"):
        list(loader)
