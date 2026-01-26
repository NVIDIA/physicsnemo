#!/usr/bin/env python3
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
"""
Combined PyArrow schema for data produced by etl_unified.py

This schema handles both satellite observations (atms, mhs, amsua, etc.)
and conventional observations (gps, ps, q, t, uv), as well as both 'ges'
(guess) and 'anl' (analysis) data types with optional fields that are only present
in certain contexts.

Key design decision: Conventional observations (u, v, T, q) are flattened
into a single 'Observation' column with multiple rows per location for
multi-component observations.
"""

import pyarrow as pa

GLOBAL_CHANNEL_ID = pa.field("Global_Channel_ID", pa.uint16(), nullable=False)
SENSOR_ID = pa.field("sensor_id", pa.uint16())


def get_combined_observation_schema() -> pa.Schema:
    """
    Create a combined PyArrow schema for both satellite and conventional observations.

    This schema accommodates:
    1. Common fields present in both data types
    2. Satellite-specific fields (angles, channel info)
    3. Conventional-specific fields (pressure, height, observation types)
    4. Flattened observation structure (all obs types in single Observation column)
    5. Optional analysis fields (present only in 'anl' data)

    Note: Conventional observations (u, v, T, q) are flattened into a single
    'Observation' column with multiple rows per location for multi-component obs.

    Returns:
        pa.Schema: Combined schema for all observation data
    """

    # Common fields present in both satellite and conventional data
    common_fields = [
        # Spatial and temporal information
        pa.field("Latitude", pa.float32()),
        pa.field("Longitude", pa.float32()),
        pa.field(
            "Absolute_Obs_Time", pa.timestamp("ns")
        ),  # nanosecond is excessively precise, but is valid from 1678 --2262, so good enough for our pruposes
        pa.field("DA_window", pa.timestamp("ns")),
        # Platform identification
        pa.field("Platform_ID", pa.uint16()),  # Maps to PLATFORM_NAME_TO_ID
        # Observation data - flattened structure
        pa.field("Observation", pa.float32()),  # Main observation value (required)
        GLOBAL_CHANNEL_ID,
    ]

    # Satellite-specific fields (from etl.py)
    satellite_fields = [
        # Angular information (satellite observations only)
        pa.field("Sat_Zenith_Angle", pa.float32(), nullable=True),
        pa.field("Sol_Zenith_Angle", pa.float32(), nullable=True),
        pa.field("Scan_Angle", pa.float32(), nullable=True),
    ]

    # Conventional observation specific fields (from etl_conv.py)
    conventional_fields = [
        # Metadata fields
        pa.field("Pressure", pa.float32(), nullable=True),
        pa.field("Height", pa.float32(), nullable=True),
        pa.field("Observation_Type", pa.uint16(), nullable=True),
    ]

    # Analysis fields (present only in 'anl' data)
    analysis_fields = [
        # Quality control and forecast differences
        pa.field("QC_Flag", pa.int32(), nullable=True),  # Satellite QC flag
        pa.field(
            "Analysis_Use_Flag", pa.int8(), nullable=True
        ),  # Conventional analysis flag
        # Forecast differences (flattened to single column)
        pa.field("Obs_Minus_Forecast_adjusted", pa.float32(), nullable=True),
        pa.field("Obs_Minus_Forecast_unadjusted", pa.float32(), nullable=True),
    ]

    # Combine all fields
    all_fields = (
        common_fields + satellite_fields + conventional_fields + analysis_fields
    )

    return pa.schema(all_fields)


def get_channel_table_schema():
    return pa.schema(
        [
            GLOBAL_CHANNEL_ID,
            pa.field("min_valid", pa.float32()),
            pa.field("max_valid", pa.float32()),
            SENSOR_ID,
            pa.field("is_conv", pa.bool_()),
            pa.field("name", pa.string()),
            pa.field("mean", pa.float32()),
            pa.field("stddev", pa.float32()),
        ]
    )
