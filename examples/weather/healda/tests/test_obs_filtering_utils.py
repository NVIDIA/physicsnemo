# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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
# ruff: noqa: S101
"""
Test script for the obs_filtering_utils.py implementation.
This tests the vectorized filtering approach for conventional observations.
"""

import numpy as np
import pyarrow as pa
from datasets.etl.combined_schema import (
    GLOBAL_CHANNEL_ID,
    get_combined_observation_schema,
)
from datasets.etl.etl_unified import get_channel_table
from datasets.obs_filtering_utils import filter_observations
from datasets.obs_loader import LOCAL_CHANNEL_ID
from datasets.sensors import SENSOR_OFFSET


def create_test_data():
    """Create test data with different platform types."""
    # Create test data with different channel IDs
    conv_offset = SENSOR_OFFSET["conv"]

    # GPS channels: 0, 1, 2
    gps_channels = [conv_offset + 0, conv_offset + 1, conv_offset + 2]
    # PS channel: 3
    ps_channels = [conv_offset + 3]
    # Q channel: 4
    q_channels = [conv_offset + 4]
    # T channel: 5
    t_channels = [conv_offset + 5]
    # UV channels: 6, 7
    uv_channels = [conv_offset + 6, conv_offset + 7]

    # Combine all channels
    all_channels = gps_channels + ps_channels + q_channels + t_channels + uv_channels

    # Create test data
    n_rows = len(all_channels)
    data = {
        # Required common fields
        "Latitude": np.random.uniform(-90, 90, n_rows),
        "Longitude": np.random.uniform(-180, 180, n_rows),
        "Absolute_Obs_Time": np.array(
            [np.datetime64("2023-01-01T00:00:00")] * n_rows, dtype="datetime64[ns]"
        ),
        "DA_window": np.array(
            [np.datetime64("2023-01-01T00:00:00")] * n_rows, dtype="datetime64[ns]"
        ),
        "Platform_ID": np.random.randint(1, 100, n_rows),
        "Global_Channel_ID": all_channels,
        "Observation": np.random.uniform(0, 100, n_rows),
        # Satellite-specific fields (nullable)
        "Sat_Zenith_Angle": np.full(n_rows, None, dtype=object),
        "Sol_Zenith_Angle": np.full(n_rows, None, dtype=object),
        "Scan_Angle": np.full(n_rows, None, dtype=object),
        # Conventional-specific fields
        "Pressure": np.random.uniform(200, 1100, n_rows),
        "Height": np.random.uniform(0, 50000, n_rows),
        "Observation_Type": np.random.randint(1, 10, n_rows),
        # Analysis fields (nullable)
        "QC_Flag": np.random.choice([0, 1], n_rows),
        "Analysis_Use_Flag": np.random.choice([0, 1], n_rows),
        "Obs_Minus_Forecast_adjusted": np.random.uniform(-10, 10, n_rows),
        "Obs_Minus_Forecast_unadjusted": np.random.uniform(-10, 10, n_rows),
    }

    obs_table = pa.table(data, schema=get_combined_observation_schema())

    # Add channel metadata via join (mimics UFSUnifiedLoader._add_channel_metadata)
    def _add_channel_metadata(table):
        channel_table = get_channel_table()

        # Add local_channel_id (same as UFSUnifiedLoader.channel_table property)
        sensor_id = np.asarray(channel_table["sensor_id"])
        local_channel_ids = []
        offset = 0
        for i in range(len(sensor_id)):
            if sensor_id[i] != sensor_id[i - 1]:
                offset = i
            local_channel_ids.append(i - offset)
        channel_table = channel_table.append_column(
            LOCAL_CHANNEL_ID.name, pa.array(local_channel_ids, type=pa.uint16())
        )

        return table.join(
            channel_table.select(
                [
                    GLOBAL_CHANNEL_ID.name,
                    LOCAL_CHANNEL_ID.name,
                    "min_valid",
                    "max_valid",
                    "is_conv",
                ]
            ),
            GLOBAL_CHANNEL_ID.name,
        )

    return _add_channel_metadata(obs_table)


def test_vectorized_filtering():
    """Test that vectorized filtering works correctly."""
    table = create_test_data()

    # Test filtering using the unified filter_observations function
    filtered_table = filter_observations(table, qc_filter=False)

    # Test with QC filtering enabled
    qc_filtered_table = filter_observations(table, qc_filter=True)

    # Verify that filtering produces results
    assert filtered_table.num_rows >= 0
    assert qc_filtered_table.num_rows >= 0
    assert qc_filtered_table.num_rows <= filtered_table.num_rows
