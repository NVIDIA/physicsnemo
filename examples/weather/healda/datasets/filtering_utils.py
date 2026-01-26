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
Shared quality control filtering utilities for observation data.

This module provides reusable filtering functions that can be used by both
the original UFSDataset and the new UFSUnifiedLoader to ensure consistent
quality control across different data loading approaches.
"""

import pyarrow as pa
import pyarrow.compute as pc

from healda.datasets.da.sensors import (
    SENSOR_CONFIGS,
    SENSOR_OFFSET,
    CONV_GPS_LEVEL2_CHANNELS,
    CONV_UV_CHANNELS,
    CONV_UV_IN_SITU_TYPES,
)


def _get_index_range(sensor):
    start = SENSOR_OFFSET[sensor]
    end = start + SENSOR_CONFIGS[sensor].channels
    return start, end


# columns to use for filtering
height = pc.field("Height")
pressure = pc.field("Pressure")
obs = pc.field("Observation")
analysis_use = pc.field("Analysis_Use_Flag")
qc_flag = pc.field("QC_Flag")
min_valid = pc.field("min_valid")
max_valid = pc.field("max_valid")
local_id = pc.field("local_channel_id")
is_conv = pc.field("is_conv")
obs_type = pc.field("Observation_Type")


def _get_conv_filter_expr(
    table: pa.Table,
    qc_filter: bool = False,
    uv_in_situ_only: bool = False,
    gps_level1_only: bool = False,
):
    """Get filter expression for conventional observations."""
    is_gps = local_id <= 2

    height_ok = pc.is_finite(height) & ((height >= 0) & (height <= 60000))

    min_pressure = pc.if_else(is_gps, pa.scalar(0.5), pa.scalar(200))
    pressure_ok = pc.is_finite(pressure)
    pressure_ok &= (pressure >= min_pressure) & (pressure <= 1100)

    ok = pressure_ok & height_ok

    if qc_filter:
        ok &= analysis_use == pa.scalar(1)

    if uv_in_situ_only:
        is_uv_channel = pc.is_in(local_id, pa.array(CONV_UV_CHANNELS))
        is_in_situ = pc.is_in(
            obs_type,
            pa.array(CONV_UV_IN_SITU_TYPES, type=table["Observation_Type"].type),
        )
        ok &= ~is_uv_channel | is_in_situ

    if gps_level1_only:
        ok &= ~pc.is_in(local_id, pa.array(CONV_GPS_LEVEL2_CHANNELS))

    return ok


def filter_observations(
    table: pa.Table,
    qc_filter: bool = False,
    conv_uv_in_situ_only: bool = False,
    conv_gps_level1_only: bool = False,
) -> pa.Table:
    """
    Unified filtering function for observation data.

    Args:
        table: PyArrow table containing observation data
        qc_filter: Whether to apply QC flag filtering
        conv_uv_in_situ_only: Exclude satellite UV (keep in-situ only)
        conv_gps_level1_only: Exclude GPS T/Q retrievals (keep bending angle)

    Returns:
        Filtered PyArrow table
    """
    ok = pc.is_finite(obs)
    ok &= obs >= min_valid
    ok &= obs <= max_valid

    sat_ok = ok
    if qc_filter:
        sat_ok &= qc_flag == 0

    conv_filter = _get_conv_filter_expr(
        table, qc_filter, conv_uv_in_situ_only, conv_gps_level1_only
    )
    ok &= pc.if_else(is_conv, conv_filter, sat_ok)

    return table.filter(ok)
