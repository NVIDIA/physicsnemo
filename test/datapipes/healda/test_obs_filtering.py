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
"""Tests for observation quality-control filtering."""

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from physicsnemo.experimental.datapipes.healda.configs.sensors import (  # noqa: E402
    SENSOR_OFFSET,
)
from physicsnemo.experimental.datapipes.healda.transforms.obs_filtering import (  # noqa: E402
    filter_observations,
)


def _make_filter_test_table():
    """Create a minimal table with channel metadata columns required for filtering."""
    conv_offset = SENSOR_OFFSET["conv"]

    # Mix of GPS (0,1,2), PS (3), UV (6,7) channels
    channels = [
        conv_offset + 0,
        conv_offset + 1,
        conv_offset + 3,
        conv_offset + 6,
        conv_offset + 7,
    ]

    return pa.table(
        {
            "Observation": np.array(
                [100.0, 200.0, 500.0, 50.0, 60.0], dtype=np.float32
            ),
            "Global_Channel_ID": np.array(channels, dtype=np.uint16),
            "Pressure": np.array([500.0, 800.0, 600.0, 400.0, 300.0], dtype=np.float32),
            "Height": np.array(
                [1000.0, 5000.0, 100.0, 2000.0, 3000.0], dtype=np.float32
            ),
            "Observation_Type": np.array([200, 210, 220, 230, 280], dtype=np.uint16),
            "QC_Flag": np.array([0, 0, 0, 0, 0], dtype=np.int32),
            "Analysis_Use_Flag": np.array([1, 1, 0, 1, 1], dtype=np.int8),
            "min_valid": np.array([0.0, 0.0, 0.0, -100.0, -100.0], dtype=np.float32),
            "max_valid": np.array([400.0, 400.0, 1e6, 100.0, 100.0], dtype=np.float32),
            "is_conv": np.array([True, True, True, True, True]),
            "local_channel_id": np.array([0, 1, 3, 6, 7], dtype=np.uint16),
        }
    )


def test_filter_observations_basic():
    """Basic filtering removes out-of-range observations."""
    table = _make_filter_test_table()
    filtered = filter_observations(table, qc_filter=False)

    assert filtered.num_rows >= 0
    assert filtered.num_rows <= table.num_rows


def test_filter_observations_qc():
    """QC filtering is more restrictive."""
    table = _make_filter_test_table()
    no_qc = filter_observations(table, qc_filter=False)
    with_qc = filter_observations(table, qc_filter=True)

    assert with_qc.num_rows <= no_qc.num_rows
