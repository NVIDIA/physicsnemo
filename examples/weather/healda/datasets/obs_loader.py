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
UFS Unified Loader for the new combined schema data format.

This loader handles both satellite and conventional observations using the
unified schema produced by etl_unified.py. It provides an async interface
compatible with TimeMergedDataset and includes quality control filtering,
normalization, and innovation filtering.
"""

import functools
import io
import os
from datetime import datetime
from typing import List, Literal

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from utils import storage

from datasets.etl.combined_schema import (
    GLOBAL_CHANNEL_ID,
    SENSOR_ID,
    get_combined_observation_schema,
)
from datasets.obs_filtering_utils import filter_observations
from datasets.sensors import (
    SENSOR_CONFIGS,
)
from physicsnemo.models.healda import profiling

LOCAL_CHANNEL_ID = pa.field("local_channel_id", pa.uint16())


def get_channel_table():
    import config.environment as config

    return UFSUnifiedLoader(
        config.UFS_OBS_PATH,
        sensors=[],
        obs_context_hours=(-3, 3),
        normalization="zscore",
        filesystem_type="s3" if config.UFS_OBS_PATH.startswith("s3://") else "local",
        remote_name=config.UFS_OBS_PROFILE,
    ).channel_table


class UFSUnifiedLoader:
    """
    Unified loader for UFS observation data using the new combined schema.

    This loader handles both satellite and conventional observations in a
    unified format, providing async interface compatibility with TimeMergedDataset.
    """

    def __init__(
        self,
        data_path: str,
        sensors: List[str],
        filesystem_type: Literal["s3", "local"] = "local",
        remote_name: str = "pdx",
        normalization: Literal["minmax", "zscore"] = "minmax",
        innovation_type: Literal["none", "adjusted", "unadjusted"] = "none",
        qc_filter: bool = False,
        filter_innovation: bool = False,
        check_corrected: bool = True,
        obs_context_hours: tuple[int, int] = (-24, 0),
        data_spacing: int = 3,  # hours
        drop_obs_channel_ids: list[int] | None = None,
        conv_uv_in_situ_only: bool = False,
        conv_gps_level1_only: bool = False,
    ):
        """
        Initialize the UFS Unified Loader.

        Args:
            data_path: Path to the processed observation data
            sensors: List of sensors to load (e.g., ['atms', 'mhs', 'conv'])
            filesystem_type: Type of filesystem ('s3' or 'local')
            remote_name: Remote storage name for S3
            normalization: Normalization method ('minmax' or 'zscore')
            innovation_type: Innovation type to use ('none', 'adjusted', 'unadjusted')
            qc_filter: Whether to apply quality control filtering
            filter_innovation: Whether to filter based on innovation values
            check_corrected: Whether to validate corrected observation values
            obs_context_hours: Hours relative to target time for observation context
            data_spacing: Hours between data points
            drop_obs_channel_ids: Global channel IDs to drop
            conv_uv_in_situ_only: Exclude satellite UV (keep in-situ only)
            conv_gps_level1_only: Exclude GPS T/Q (keep bending angle)
        """
        self.data_path = data_path
        self.sensors = sensors
        self.filesystem_type = filesystem_type
        self.remote_name = remote_name
        self.normalization = normalization
        self.innovation_type = innovation_type
        self.qc_filter = qc_filter
        self.filter_innovation = filter_innovation
        self.check_corrected = check_corrected
        self.obs_context_hours = obs_context_hours
        self.data_spacing = data_spacing
        # Optional list of global observation channel IDs (GLOBAL_CHANNEL_ID)
        # to drop before normalization and further processing.
        self.drop_obs_channel_ids = (
            list(drop_obs_channel_ids) if drop_obs_channel_ids is not None else []
        )
        self.conv_uv_in_situ_only = conv_uv_in_situ_only
        self.conv_gps_level1_only = conv_gps_level1_only

        # Validate sensors
        for sensor in self.sensors:
            if sensor not in SENSOR_CONFIGS:
                raise ValueError(
                    f"Unconfigured sensor: {sensor}. Available: {list(SENSOR_CONFIGS.keys())}"
                )

        # Setup filesystem
        if self.filesystem_type == "s3":
            self.fs = fsspec.filesystem(
                "s3", **storage.get_storage_options(remote_name)
            )
        elif self.filesystem_type == "local":
            self.fs = None
        else:
            raise ValueError(
                f"Unsupported filesystem_type: {filesystem_type}. Use 's3' or 'local'"
            )

        # Load channel table for normalization
        self._channel_table = None

    @property
    def output_schema(self) -> pa.Schema:
        """Get the output schema including the sensor and platform columns."""
        base_schema = get_combined_observation_schema()
        return base_schema.append(LOCAL_CHANNEL_ID).append(SENSOR_ID)

    @functools.cached_property
    def channel_table(self) -> pa.Table:
        """Load the channel table for normalization."""
        channel_table_path = os.path.join(self.data_path, "channel_table.parquet")
        if self.fs is not None:
            file = io.BytesIO(self.fs.cat_file(channel_table_path))
        else:
            file = channel_table_path

        table = pq.read_table(file)
        sensor_id = np.asarray(table["sensor_id"])
        local_channel_ids = []
        offset = 0
        for i in range(len(sensor_id)):
            if sensor_id[i] != sensor_id[i - 1]:
                offset = i
            local_channel_ids.append(i - offset)
        array = pa.array(local_channel_ids).cast(LOCAL_CHANNEL_ID.type)
        return table.append_column(LOCAL_CHANNEL_ID, array)

    def _get_interval_times(self, dt: datetime) -> pd.DatetimeIndex:
        """Get times in the observation context interval."""
        start, end = self.obs_context_hours
        start += self.data_spacing  # Window times are end-aligned

        return pd.date_range(
            dt + pd.Timedelta(hours=start),
            dt + pd.Timedelta(hours=end),
            freq=f"{self.data_spacing}h",
        )

    def _get_parquet_files_to_read(self, interval_times: pd.DatetimeIndex):
        """Get parquet files to read for given time interval."""
        required_dates = {t.strftime("%Y%m%d") for t in interval_times}

        for sensor in self.sensors:
            for date in required_dates:
                file_path = os.path.join(self.data_path, sensor, f"{date}", "0.parquet")
                yield (sensor, file_path)

    def _iterate_parquet_da_windows(
        self,
        parquet_path: str,
        target_windows: pd.DatetimeIndex,
    ):
        """
        Stream Arrow Tables, one per DA_window row-group.

        Args:
            parquet_path: Path to parquet file
            target_windows: Only yield these DA_window

        Yields:
            PyArrow tables, one per DA window

        Note:
            Silently skips files that don't exist or can't be read
        """
        try:
            if self.fs is not None:
                file = io.BytesIO(self.fs.cat_file(parquet_path))
            else:
                file = parquet_path

            parquet = pq.ParquetFile(file)
            schema = parquet.schema_arrow

            # With uniform schema, just read all columns
            da_idx = schema.get_field_index("DA_window")

            for row_group_idx in range(parquet.num_row_groups):
                stats = (
                    parquet.metadata.row_group(row_group_idx).column(da_idx).statistics
                )
                row_group_lo, row_group_hi = stats.min, stats.max

                this_window = None
                for w in target_windows:
                    if row_group_lo <= w <= row_group_hi:
                        this_window = w

                if this_window is None:
                    continue

                # Read all columns - no need for platform-specific selection
                table = parquet.read_row_group(row_group_idx)

                # Filter if row-group spans multiple windows
                if row_group_lo != row_group_hi:
                    mask = pc.is_in(table["DA_window"], pa.array(list(target_windows)))
                    table = table.filter(mask)

                if table.num_rows == 0:
                    continue

                yield this_window, table
        except (FileNotFoundError, OSError):
            # File doesn't exist or can't be read - silently skip
            return

    def _filter_observations(self, table: pa.Table) -> pa.Table:
        return filter_observations(
            table,
            self.qc_filter,
            conv_uv_in_situ_only=self.conv_uv_in_situ_only,
            conv_gps_level1_only=self.conv_gps_level1_only,
        )

    def _normalize_observations(
        self,
        table: pa.Table,
    ) -> pa.Table:
        """Normalize observation data using PyArrow compute functions."""
        if self.normalization == "minmax":
            # Simple minmax normalization (0-400 range)
            normalized = pc.divide(pc.subtract(table["Observation"], 0), 400 - 0)
        elif self.normalization == "zscore":
            # Normalize using the joined mean and stddev columns
            normalized = pc.divide(
                pc.subtract(table["Observation"], table["mean"]), table["stddev"]
            )
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")
        return table.set_column(
            table.schema.get_field_index("Observation"),
            "Observation",
            normalized,
        )

    _extra_channel_fields = ["min_valid", "max_valid", "is_conv", "mean", "stddev"]

    def _add_channel_metadata(self, table):
        return table.join(
            self.channel_table.select(
                [
                    GLOBAL_CHANNEL_ID.name,
                    LOCAL_CHANNEL_ID.name,
                    SENSOR_ID.name,
                    *self._extra_channel_fields,
                ]
            ),
            GLOBAL_CHANNEL_ID.name,
        )

    async def sel_time(self, times: pd.DatetimeIndex) -> pa.Table:
        """
        Load observation data for specified times.

        Args:
            times: Target times to load data for

        Returns:
            PyArrow table containing observation data (sorted by sensor and the obs_window)
        """
        # Get all times needed for the context window
        all_times = set()
        for t in times:
            interval_times = self._get_interval_times(t)
            all_times.update(interval_times)

        interval_times = pd.DatetimeIndex(sorted(all_times))

        # Get files to read
        files_to_read = self._get_parquet_files_to_read(interval_times)

        # Load data from all files

        tables = {}
        for sensor, file_path in files_to_read:
            for interval_time, table in self._iterate_parquet_da_windows(
                file_path, interval_times
            ):
                table = self._add_channel_metadata(table)
                table = self._filter_observations(table)
                # Drop specified global channels, if any
                if self.drop_obs_channel_ids:
                    mask = pc.is_in(
                        table[GLOBAL_CHANNEL_ID.name],
                        pa.array(self.drop_obs_channel_ids).cast(
                            table[GLOBAL_CHANNEL_ID.name].type
                        ),
                    )
                    # Keep rows whose GLOBAL_CHANNEL_ID is NOT in drop list
                    table = table.filter(pc.invert(mask))
                table = self._normalize_observations(table)
                table = table.drop(self._extra_channel_fields)
                # Apply normalization to observations using PyArrow
                tables.setdefault(interval_time, []).append(table)

        # Combine all observations
        def process(t):
            all_tables = []
            for interval_time in self._get_interval_times(t):
                for table in tables.get(interval_time, []):
                    all_tables.append(table)

            if not all_tables:
                return empty

            table = pa.concat_tables(all_tables)
            # table = table.combine_chunks()
            # Cast to ensure proper nullability and types
            # it's 3x faster to filter the combined table
            return table.cast(self.output_schema)

        empty = self._get_empty_table()
        return {"obs_v2": [process(t) for t in times]}

    def _get_empty_table(self):
        # Return empty table with proper schema
        # Create empty arrays for each field in the schema
        empty_arrays = []
        for field in self.output_schema:
            empty_arrays.append(pa.array([], type=field.type))
        template = pa.table(empty_arrays, schema=self.output_schema)
        return template
