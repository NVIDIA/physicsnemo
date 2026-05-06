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
"""Tests for the combined observation schema and sensor config consistency."""

import pytest

pa = pytest.importorskip("pyarrow")

from physicsnemo.experimental.datapipes.healda.configs.combined_schema import (  # noqa: E402
    get_channel_table_schema,
    get_combined_observation_schema,
)
from physicsnemo.experimental.datapipes.healda.configs.sensors import (  # noqa: E402
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
)


def test_combined_schema_has_required_fields():
    schema = get_combined_observation_schema()
    required = [
        "Latitude",
        "Longitude",
        "Absolute_Obs_Time",
        "DA_window",
        "Platform_ID",
        "Observation",
        "Global_Channel_ID",
    ]
    for name in required:
        assert name in schema.names, f"Missing required field: {name}"


def test_combined_schema_satellite_fields():
    schema = get_combined_observation_schema()
    for name in ["Sat_Zenith_Angle", "Sol_Zenith_Angle", "Scan_Angle"]:
        assert name in schema.names


def test_combined_schema_conventional_fields():
    schema = get_combined_observation_schema()
    for name in ["Pressure", "Height", "Observation_Type"]:
        assert name in schema.names


def test_channel_table_schema():
    schema = get_channel_table_schema()
    assert "Global_Channel_ID" in schema.names
    assert "sensor_id" in schema.names
    assert "mean" in schema.names
    assert "stddev" in schema.names


def test_sensor_configs_consistent():
    """All sensors in SENSOR_CONFIGS have a matching SENSOR_NAME_TO_ID entry."""
    for name in SENSOR_CONFIGS:
        assert name in SENSOR_NAME_TO_ID


def test_sensor_channels_positive():
    for name, cfg in SENSOR_CONFIGS.items():
        assert cfg.channels > 0, f"Sensor {name} has non-positive channel count"
