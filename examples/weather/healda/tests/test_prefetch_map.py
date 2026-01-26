# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from datasets.prefetch_map import prefetch_map


def test_prefetch_map_basic_functionality():
    """Test basic async dataloader functionality with simple data."""
    # Create simple test data using range
    data = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Simple transform that doubles the data
    def transform(x):
        return 2 * x

    # Create async loader
    async_loader = prefetch_map(data, transform)
    assert list(async_loader) == list(range(0, 20, 2))


def test_prefetch_map_error_handling():
    """Test error handling when transform raises an exception."""
    data = list(range(4))  # [0, 1, 2, 3]

    def failing_transform(x):
        raise ValueError("Test error")

    async_loader = prefetch_map(data, failing_transform)

    # Should raise the exception from the background thread
    with pytest.raises(ValueError, match="Test error"):
        list(async_loader)
