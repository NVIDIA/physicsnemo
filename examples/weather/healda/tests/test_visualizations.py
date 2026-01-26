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
import numpy as np
import pytest
from utils.visualizations import visualize


def test_visualize():
    """Test visualization with reasonable default settings."""
    # Create a simple 1D input array
    x = np.random.rand(12)  # HEALPix level 1 has 12 pixels

    # Test with reasonable defaults
    im = visualize(
        x,
        region="Robinson",
        title="Test Visualization",
        cmap="viridis",
        add_colorbar=True,
    )
    assert im is not None

    # Test that invalid input raises error
    with pytest.raises(ValueError):
        visualize(np.random.rand(4, 3))  # 2D input should raise ValueError
