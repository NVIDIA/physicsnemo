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
"""Combined PyArrow schema for unified satellite and conventional observation data.

This schema handles both satellite observations (atms, mhs, amsua, etc.)
and conventional observations (gps, ps, q, t, uv).  Conventional observations
are flattened into a single ``Observation`` column with multiple rows per
location for multi-component observations.
"""

from physicsnemo.core.version_check import OptionalImport

pa = OptionalImport("pyarrow")

GLOBAL_CHANNEL_ID = pa.field("Global_Channel_ID", pa.uint16(), nullable=False)
SENSOR_ID = pa.field("sensor_id", pa.uint16())


def get_combined_observation_schema() -> pa.Schema:
    """Create a combined PyArrow schema for satellite and conventional obs."""
    common_fields = [
        pa.field("Latitude", pa.float32()),
        pa.field("Longitude", pa.float32()),
        pa.field("Absolute_Obs_Time", pa.timestamp("ns")),
        pa.field("DA_window", pa.timestamp("ns")),
        pa.field("Platform_ID", pa.uint16()),
        pa.field("Observation", pa.float32()),
        GLOBAL_CHANNEL_ID,
    ]

    satellite_fields = [
        pa.field("Sat_Zenith_Angle", pa.float32(), nullable=True),
        pa.field("Sol_Zenith_Angle", pa.float32(), nullable=True),
        pa.field("Scan_Angle", pa.float32(), nullable=True),
    ]

    conventional_fields = [
        pa.field("Pressure", pa.float32(), nullable=True),
        pa.field("Height", pa.float32(), nullable=True),
        pa.field("Observation_Type", pa.uint16(), nullable=True),
    ]

    analysis_fields = [
        pa.field("QC_Flag", pa.int32(), nullable=True),
        pa.field("Analysis_Use_Flag", pa.int8(), nullable=True),
    ]

    all_fields = (
        common_fields + satellite_fields + conventional_fields + analysis_fields
    )
    return pa.schema(all_fields)


def get_channel_table_schema():
    """Schema for the channel metadata table."""
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
