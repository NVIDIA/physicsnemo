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
Integration test for UFS Unified Loader with combined schema.
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from datasets.etl.combined_schema import (
    get_channel_table_schema,
    get_combined_observation_schema,
)
from datasets.etl.etl_unified import get_channel_table
from datasets.obs_loader import UFSUnifiedLoader
from datasets.sensors import SENSOR_CONFIGS


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with sample data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sensor directory
        sensor_dir = os.path.join(temp_dir, "atms")
        os.makedirs(sensor_dir, exist_ok=True)

        # Create date directory
        date_dir = os.path.join(sensor_dir, "20200101")
        os.makedirs(date_dir, exist_ok=True)

        # Create sample data with all required schema fields
        n_obs = 50
        data = {
            # Common fields
            "Latitude": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "Longitude": np.random.uniform(-180, 180, n_obs).astype(np.float32),
            "Absolute_Obs_Time": pd.date_range(
                "2020-01-01", periods=n_obs, freq="1h"
            ).astype("datetime64[ns]"),
            "DA_window": pd.date_range("2020-01-01", periods=n_obs, freq="3h").astype(
                "datetime64[ns]"
            ),
            "Platform_ID": np.random.randint(0, 32, n_obs).astype(np.uint16),
            "Observation": np.random.uniform(0, 400, n_obs).astype(np.float32),
            "Global_Channel_ID": np.random.randint(0, 100, n_obs).astype(np.uint16),
            # Satellite-specific fields
            "Sat_Zenith_Angle": np.random.uniform(0, 90, n_obs).astype(np.float32),
            "Sol_Zenith_Angle": np.random.uniform(0, 90, n_obs).astype(np.float32),
            "Scan_Angle": np.random.uniform(-45, 45, n_obs).astype(np.float32),
            # Conventional fields (nullable)
            "Pressure": np.full(n_obs, np.nan, dtype=np.float32),
            "Height": np.full(n_obs, np.nan, dtype=np.float32),
            "Observation_Type": np.full(n_obs, np.nan, dtype=np.uint16),
            # Analysis fields (nullable)
            "QC_Flag": np.random.randint(0, 2, n_obs).astype(np.int32),
            "Analysis_Use_Flag": np.full(n_obs, np.nan, dtype=np.int8),
            "Obs_Minus_Forecast_adjusted": np.random.uniform(-10, 10, n_obs).astype(
                np.float32
            ),
            "Obs_Minus_Forecast_unadjusted": np.random.uniform(-10, 10, n_obs).astype(
                np.float32
            ),
        }

        # Write parquet file
        schema = get_combined_observation_schema()
        table = pa.table(data, schema=schema)
        parquet_path = os.path.join(date_dir, "0.parquet")
        pa.parquet.write_table(table, parquet_path)

        # Create channel table for normalization
        channel_table = get_channel_table()
        channel_table_path = os.path.join(temp_dir, "channel_table.parquet")
        pa.parquet.write_table(channel_table, channel_table_path)

        # No need for availability_df.pkl - using try/catch approach

        yield temp_dir


@pytest.mark.parametrize("normalization", ["zscore", "minmax"])
@pytest.mark.asyncio
async def test_ufs_unified_loader(temp_data_dir, normalization):
    """Test UFSUnifiedLoader basic functionality."""
    # Initialize loader
    loader = UFSUnifiedLoader(
        data_path=temp_data_dir,
        sensors=["atms"],
        filesystem_type="local",
        normalization=normalization,
    )

    # Test basic properties
    assert loader.sensors == ["atms"]

    # Test data loading
    times = pd.DatetimeIndex([datetime(2020, 1, 1, 12)])
    result = await loader.sel_time(times)

    for result in result["obs_v2"]:
        # Validate schema matches expected output schema
        expected_schema = loader.output_schema
        assert result.schema.equals(expected_schema)

        # Check normalization (minmax should give [0,1] range) if data exists
        if normalization == "minmax" and result.num_rows > 0:
            df = result.to_pandas()
            assert (df["Observation"] >= 0).all()
            assert (df["Observation"] <= 1).all()


@pytest.mark.parametrize("normalization", ["zscore", "minmax"])
@pytest.mark.asyncio
async def test_ufs_unified_loader_empty_dataset(normalization):
    """Test UFSUnifiedLoader with empty dataset (no data files)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create empty directory structure
        sensor_dir = os.path.join(temp_dir, "atms")
        os.makedirs(sensor_dir, exist_ok=True)

        # Initialize loader with empty directory
        loader = UFSUnifiedLoader(
            data_path=temp_dir,
            sensors=["atms"],
            filesystem_type="local",
            normalization=normalization,
        )

        # Test basic properties
        assert loader.sensors == ["atms"]

        # Test data loading with empty dataset
        times = pd.DatetimeIndex([datetime(2020, 1, 1, 12)])
        result = await loader.sel_time(times)

        result = result["obs_v2"][0]

        # Check result structure - should return empty table with proper schema
        assert isinstance(result, pa.Table)
        assert result.num_rows == 0

        # Validate schema matches expected output schema
        expected_schema = loader.output_schema
        assert result.schema.equals(expected_schema)


def test_get_channel_table_structure():
    """Test that get_channel_table returns correct table structure and schema."""
    table = get_channel_table()

    # Check that it's a PyArrow table
    assert isinstance(table, pa.Table)

    # Check schema matches expected channel table schema
    expected_schema = get_channel_table_schema()
    assert table.schema.equals(expected_schema)


def test_get_channel_table_sensor_mapping():
    """Test that sensor IDs and channel IDs are correctly mapped."""
    table = get_channel_table()

    # Convert to pandas for easier analysis
    df = table.to_pandas()

    # Calculate expected total channels
    expected_total_channels = sum(cfg.channels for cfg in SENSOR_CONFIGS.values())
    assert len(df) == expected_total_channels

    # Check that Global_Channel_ID is sequential starting from 0
    assert df["Global_Channel_ID"].min() == 0
    assert df["Global_Channel_ID"].max() == expected_total_channels - 1
    assert df["Global_Channel_ID"].is_monotonic_increasing

    # Check sensor_id mapping
    sensor_names = list(SENSOR_CONFIGS.keys())
    for i, sensor_name in enumerate(sensor_names):
        sensor_mask = df["sensor_id"] == i
        expected_channels = SENSOR_CONFIGS[sensor_name].channels
        assert sensor_mask.sum() == expected_channels


def test_get_channel_table_conventional_handling():
    """Test that conventional sensors are handled correctly."""
    table = get_channel_table()
    df = table.to_pandas()

    # Find conventional sensor index
    conv_sensor_id = list(SENSOR_CONFIGS.keys()).index("conv")
    conv_mask = df["sensor_id"] == conv_sensor_id

    # Check is_conv flag
    assert df[conv_mask]["is_conv"].all()
    assert not df[~conv_mask]["is_conv"].any()

    # Check conventional sensor naming
    conv_names = df[conv_mask]["name"].tolist()
    expected_conv_names = ["gps_angle", "gps_t", "gps_q", "ps", "q", "t", "u", "v"]
    assert conv_names == expected_conv_names


def test_get_channel_table_consistency():
    """Test that channel table is internally consistent."""
    table = get_channel_table()
    df = table.to_pandas()

    # Check that all Global_Channel_IDs are unique
    assert df["Global_Channel_ID"].nunique() == len(df)

    # Check that sensor_id values are valid
    max_sensor_id = len(SENSOR_CONFIGS) - 1
    assert df["sensor_id"].min() >= 0
    assert df["sensor_id"].max() <= max_sensor_id

    # Check that min_valid <= max_valid for all channels
    assert (df["min_valid"] <= df["max_valid"]).all()

    # Check that all names are non-empty strings
    assert df["name"].str.len().min() > 0
    assert df["name"].notna().all()
