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
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import logging

import healda.storage
from healda.config.environment import UFS_OBS_PATH, UFS_OBS_PROFILE
from healda.datasets.da.v2.etl.combined_schema import get_combined_observation_schema


logger = logging.getLogger(__name__)

TIME_COLUMN = "Absolute_Obs_Time"


def _da_window_from_time(t: pd.Timestamp) -> pd.Timestamp:
    # da window at 21 UTC is (21-3, 21]
    t0 = np.datetime64("2000-01-01T00")
    dt = np.timedelta64(3, "h")

    delta = (t - t0) % dt
    return t if delta == np.timedelta64(0, "h") else t + dt - delta


def _get_file_names(
    dir,
    sensors,
    time_min: pd.Timestamp,
    time_max: pd.Timestamp,
):
    da_min = _da_window_from_time(time_min)
    da_max = _da_window_from_time(time_max)

    # get date range
    dates = []
    this_date = da_min.date()
    day = pd.Timedelta(1, "d")
    while this_date <= da_max.date():
        dates.append(this_date)
        this_date += day

    file_names = [
        os.path.join(dir, sensor, date.strftime("%Y%m%d"), "0.parquet")
        for date in dates
        for sensor in sensors
    ]
    return file_names


def _scan_parquet_dir(dir, sensors, time_min, time_max, columns=(), filesystem=None):
    if dir.startswith("s3://"):
        dir = dir[len("s3://") :]
    for file_name in _get_file_names(dir, sensors, time_min, time_max):
        try:
            with pq.ParquetFile(file_name, filesystem=filesystem) as f:
                yield from _scan_parquet_file(
                    f,
                    column=TIME_COLUMN,
                    time_min=time_min,
                    time_max=time_max,
                    columns=columns,
                )
        except FileNotFoundError:
            logging.getLogger(__name__).debug(f"{file_name} not found.")
            pass


def _open_channel_table(dir, filesystem=None) -> pa.Table:
    if dir.startswith("s3://"):
        dir = dir[len("s3://") :]
    channel_table_path = os.path.join(dir, "channel_table.parquet")
    with pq.ParquetFile(channel_table_path, filesystem=filesystem) as f:
        return f.read()


def _scan_parquet_file(
    parquet,
    column,
    time_min,
    time_max,
    columns=(),
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
    schema = parquet.schema_arrow

    # With uniform schema, just read all columns
    da_idx = schema.get_field_index(column)

    for row_group_idx in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(row_group_idx).column(da_idx).statistics

        # print(stats.min < time_min, stats.max, time_min)
        if stats.max < time_min:
            continue

        if stats.min > time_max:
            continue

        # Read all columns - no need for platform-specific selection
        table = parquet.read_row_group(row_group_idx, list(columns) + [column])

        time_col = pc.field(column)
        filter = (time_col >= time_min) & (time_col <= time_max)
        table = table.filter(filter)
        table = table.select(columns)

        if table.num_rows == 0:
            continue

        yield table


class Loader:
    """Parquet obs loader

    Allows selecting data based on a time range. only supports scalar time
    bounds at the moment.
    """

    def __init__(
        self,
        sensors=("atms", "conv"),
        columns=(
            "Latitude",
            "Longitude",
            "Global_Channel_ID",
            "Height",
            "Pressure",
            TIME_COLUMN,
        ),
        join_channel_table: bool = True,
    ):
        self.sensors = sensors
        self._filesystem = healda.storage.get_pyarrow_filesystem(
            UFS_OBS_PROFILE, connect_timeout=1_000, request_timeout=1_000
        )
        self.channel_table = _open_channel_table(UFS_OBS_PATH, self._filesystem)
        self.columns = columns
        self.join_channel_table = join_channel_table

    @property
    def schema(self):
        schema = get_combined_observation_schema()
        schema = pa.schema([f for f in schema if f.name in self.columns])
        return schema

    def _get_empty(self):
        obs = pa.table([[]] * len(self.schema), schema=self.schema)
        return obs

    def sel_time_range(
        self, time_min: pd.Timestamp, time_max: pd.Timestamp
    ) -> pa.Table:
        scanner = _scan_parquet_dir(
            UFS_OBS_PATH,
            sensors=self.sensors,
            columns=self.columns,
            time_min=time_min,
            time_max=time_max,
            filesystem=self._filesystem,
        )
        obs = list(scanner)
        if len(obs) > 0:
            obs = pa.concat_tables(obs)
        else:
            logger.warning(f"No observations loaded for {time_min} -- {time_max}")
            obs = self._get_empty()

        if self.join_channel_table:
            obs = obs.join(self.channel_table, "Global_Channel_ID")

        return obs
