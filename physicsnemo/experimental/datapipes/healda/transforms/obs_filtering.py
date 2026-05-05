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
"""Quality-control filtering for observation data.

Applies range checks, height/pressure limits, and optional QC flag filtering
to PyArrow observation tables. Used by ``UFSUnifiedLoader`` after joining
channel metadata.
"""

from physicsnemo.core.version_check import OptionalImport

pa = OptionalImport("pyarrow")
pc = OptionalImport("pyarrow.compute")

from physicsnemo.experimental.datapipes.healda.configs.sensors import (
    CONV_GPS_LEVEL2_CHANNELS,
    CONV_UV_CHANNELS,
    CONV_UV_IN_SITU_TYPES,
    QCLimits,
)

# Column references for filtering expressions
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
    """Build filter expression for conventional observations."""
    is_gps = local_id <= 2

    height_ok = pc.is_finite(height) & (
        (height >= QCLimits.HEIGHT_MIN) & (height <= QCLimits.HEIGHT_MAX)
    )

    min_pressure = pc.if_else(
        is_gps,
        pa.scalar(QCLimits.PRESSURE_MIN_GPS),
        pa.scalar(QCLimits.PRESSURE_MIN_DEFAULT),
    )
    pressure_ok = pc.is_finite(pressure)
    pressure_ok &= (pressure >= min_pressure) & (pressure <= QCLimits.PRESSURE_MAX)

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
    """Filter observations by range, QC flags, and conventional-specific criteria.

    Args:
        table: PyArrow table with observation data (must include channel metadata
            columns ``min_valid``, ``max_valid``, ``is_conv``, ``local_channel_id``).
        qc_filter: Whether to apply QC flag / analysis-use filtering.
        conv_uv_in_situ_only: Exclude satellite UV winds (keep in-situ only).
        conv_gps_level1_only: Exclude GPS T/Q retrievals (keep bending angle).

    Returns:
        Filtered PyArrow table.
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
