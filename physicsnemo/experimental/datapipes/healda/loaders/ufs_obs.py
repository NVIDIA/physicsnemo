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
"""UFS Unified observation loader for combined satellite and conventional data.

``UFSUnifiedLoader`` implements the ``ObsLoader`` protocol, loading
parquet-based observations produced by the ETL pipeline. It provides
quality-control filtering, normalization, and DA-window alignment.

Example::

    loader = UFSUnifiedLoader(
        data_path="/path/to/processed_obs",
        sensors=["atms", "mhs", "conv"],
        obs_context_hours=(-3, 3),
    )
    result = await loader.sel_time(pd.DatetimeIndex([...]))
    tables = result["obs"]  # list[pa.Table], one per timestamp
"""

import functools
import io
import os
from datetime import datetime
from typing import List, Literal

import fsspec
import numpy as np
import pandas as pd
from physicsnemo.core.version_check import OptionalImport

pa = OptionalImport("pyarrow")
pc = OptionalImport("pyarrow.compute")
pq = OptionalImport("pyarrow.parquet")

from physicsnemo.experimental.datapipes.healda.configs.combined_schema import (
    GLOBAL_CHANNEL_ID,
    SENSOR_ID,
    get_combined_observation_schema,
)
from physicsnemo.experimental.datapipes.healda.configs.sensors import SENSOR_CONFIGS
from physicsnemo.experimental.datapipes.healda.transforms.obs_filtering import filter_observations

LOCAL_CHANNEL_ID = pa.field("local_channel_id", pa.uint16())


def get_channel_table(data_path: str, filesystem=None):
    """Load the channel metadata table used for normalization.

    Args:
        data_path: Root path to the processed observation data.
        filesystem: Optional fsspec filesystem for remote access.
    """
    return UFSUnifiedLoader(
        data_path,
        sensors=[],
        obs_context_hours=(-3, 3),
        filesystem=filesystem,
    ).channel_table


class UFSUnifiedLoader:
    """Unified loader for UFS observation data in combined parquet format.

    Handles both satellite and conventional observations, providing an async
    interface compatible with ``ObsERA5Dataset``.

    Args:
        data_path: Path to the processed observation data directory.
        sensors: List of sensor names to load (e.g. ``["atms", "mhs", "conv"]``).
        filesystem: Optional fsspec filesystem for remote access (e.g. S3).
        innovation_type: Innovation type (``"none"``, ``"adjusted"``, ``"unadjusted"``).
        qc_filter: Whether to apply quality-control filtering.
        filter_innovation: Whether to filter based on innovation values.
        check_corrected: Whether to validate corrected observation values.
        obs_context_hours: ``(start, end)`` hours relative to target time.
        data_spacing: Hours between data points (default 3).
        drop_obs_channel_ids: Global channel IDs to drop.
        conv_uv_in_situ_only: Exclude satellite UV (keep in-situ only).
        conv_gps_level1_only: Exclude GPS T/Q (keep bending angle).
    """

    def __init__(
        self,
        data_path: str,
        sensors: List[str],
        filesystem: fsspec.AbstractFileSystem | None = None,
        innovation_type: Literal["none", "adjusted", "unadjusted"] = "none",
        qc_filter: bool = False,
        filter_innovation: bool = False,
        check_corrected: bool = True,
        obs_context_hours: tuple[int, int] = (-24, 0),
        data_spacing: int = 3,
        drop_obs_channel_ids: list[int] | None = None,
        conv_uv_in_situ_only: bool = False,
        conv_gps_level1_only: bool = False,
    ):
        self.data_path = data_path
        self.sensors = sensors
        self.fs = filesystem
        self.innovation_type = innovation_type
        self.qc_filter = qc_filter
        self.filter_innovation = filter_innovation
        self.check_corrected = check_corrected
        self.obs_context_hours = obs_context_hours
        self.data_spacing = data_spacing
        self.drop_obs_channel_ids = (
            list(drop_obs_channel_ids) if drop_obs_channel_ids is not None else []
        )
        self.conv_uv_in_situ_only = conv_uv_in_situ_only
        self.conv_gps_level1_only = conv_gps_level1_only

        for sensor in self.sensors:
            if sensor not in SENSOR_CONFIGS:
                raise ValueError(
                    f"Unconfigured sensor: {sensor}. "
                    f"Available: {list(SENSOR_CONFIGS.keys())}"
                )

        self._channel_table = None

    @functools.cached_property
    def _base_schema(self) -> pa.Schema:
        return get_combined_observation_schema()

    @functools.cached_property
    def _read_columns(self) -> list[str]:
        return self._base_schema.names

    @property
    def output_schema(self) -> pa.Schema:
        return self._base_schema.append(LOCAL_CHANNEL_ID).append(SENSOR_ID)

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
            if i == 0 or sensor_id[i] != sensor_id[i - 1]:
                offset = i
            local_channel_ids.append(i - offset)
        array = pa.array(local_channel_ids).cast(LOCAL_CHANNEL_ID.type)
        return table.append_column(LOCAL_CHANNEL_ID, array)

    def _get_interval_times(self, dt: datetime) -> pd.DatetimeIndex:
        start, end = self.obs_context_hours
        start += self.data_spacing
        return pd.date_range(
            dt + pd.Timedelta(hours=start),
            dt + pd.Timedelta(hours=end),
            freq=f"{self.data_spacing}h",
        )

    def _get_parquet_files_to_read(self, interval_times: pd.DatetimeIndex):
        required_dates = {t.strftime("%Y%m%d") for t in interval_times}
        for sensor in self.sensors:
            for date in required_dates:
                file_path = os.path.join(
                    self.data_path, sensor, f"{date}", "0.parquet"
                )
                yield (sensor, file_path)

    def _iterate_parquet_da_windows(self, parquet_path, target_windows):
        try:
            if self.fs is not None:
                file = io.BytesIO(self.fs.cat_file(parquet_path))
            else:
                file = parquet_path

            parquet = pq.ParquetFile(file)
            schema = parquet.schema_arrow
            da_idx = schema.get_field_index("DA_window")

            for row_group_idx in range(parquet.num_row_groups):
                stats = (
                    parquet.metadata.row_group(row_group_idx)
                    .column(da_idx)
                    .statistics
                )
                row_group_window = stats.min
                if row_group_window != stats.max:
                    raise ValueError(
                        f"Expected one DA_window per row group, got "
                        f"[{stats.min}, {stats.max}] for {parquet_path} row_group={row_group_idx}"
                    )
                if row_group_window not in target_windows:
                    continue
                table = parquet.read_row_group(row_group_idx, columns=self._read_columns)
                if table.num_rows == 0:
                    continue
                yield row_group_window, table
        except (FileNotFoundError, OSError):
            return

    def _filter_observations(self, table: pa.Table) -> pa.Table:
        return filter_observations(
            table,
            self.qc_filter,
            conv_uv_in_situ_only=self.conv_uv_in_situ_only,
            conv_gps_level1_only=self.conv_gps_level1_only,
        )

    def _normalize_observations(self, table: pa.Table) -> pa.Table:
        normalized = pc.divide(
            pc.subtract(table["Observation"], table["mean"]),
            table["stddev"],
        )
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

    async def sel_time(self, times: pd.DatetimeIndex) -> dict:
        """Load observation data for specified times.

        Args:
            times: Target times to load data for.

        Returns:
            ``{"obs": [pa.Table, ...]}``, one table per timestamp.
        """
        all_times = set()
        for t in times:
            interval_times = self._get_interval_times(t)
            all_times.update(interval_times)

        interval_times = pd.DatetimeIndex(sorted(all_times))
        files_to_read = self._get_parquet_files_to_read(interval_times)

        tables = {}
        for sensor, file_path in files_to_read:
            for interval_time, table in self._iterate_parquet_da_windows(
                file_path, interval_times
            ):
                table = self._add_channel_metadata(table)
                table = self._filter_observations(table)
                if self.drop_obs_channel_ids:
                    mask = pc.is_in(
                        table[GLOBAL_CHANNEL_ID.name],
                        pa.array(self.drop_obs_channel_ids).cast(
                            table[GLOBAL_CHANNEL_ID.name].type
                        ),
                    )
                    table = table.filter(pc.invert(mask))
                table = self._normalize_observations(table)
                table = table.drop(self._extra_channel_fields)
                tables.setdefault(interval_time, []).append(table)

        def process(t):
            all_tables = []
            for interval_time in self._get_interval_times(t):
                for table in tables.get(interval_time, []):
                    all_tables.append(table)

            if not all_tables:
                return empty

            table = pa.concat_tables(all_tables)
            return table.cast(self.output_schema)

        empty = self._get_empty_table()
        return {"obs": [process(t) for t in times]}

    def _get_empty_table(self):
        empty_arrays = []
        for field in self.output_schema:
            empty_arrays.append(pa.array([], type=field.type))
        return pa.table(empty_arrays, schema=self.output_schema)
