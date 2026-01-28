#!/usr/bin/env python3
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
# ruff: noqa: S608  # SQL built from internal config, not user input
"""
Compute normalization statistics for observation data.

Uses channel_table.parquet for min/max valid ranges per channel,
ensuring consistent filtering with training code.
"""

import argparse
import os
import time

import duckdb
from dotenv import load_dotenv

from datasets.sensors import CONV_GPS_GLOBAL_IDS, SENSOR_OFFSET, QCLimits

load_dotenv()

DEFAULT_SENSORS = ["conv", "mhs", "amsua", "atms", "amsub"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalizations for UFS observation data (satellite and conventional)"
    )
    parser.add_argument(
        "--sensors",
        type=str,
        nargs="+",
        default=DEFAULT_SENSORS,
        help=f"Sensors to process (default: {DEFAULT_SENSORS})",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.getenv("UFS_OBS_PATH"),
        help="Root directory with processed obs (default: $UFS_OBS_PATH)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: etl/normalizations/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_root:
        raise ValueError("--data-root required (or set UFS_OBS_PATH in .env)")

    sensors = args.sensors
    data_root = args.data_root
    # Default: store normalizations in code directory (etl/normalizations/)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "normalizations"
    )
    channel_table = os.path.join(data_root, "channel_table.parquet")

    print(f"Processing sensors: {sensors}")
    print(f"Data root: {data_root}")
    print(f"Channel table: {channel_table}")
    print(f"Output directory: {output_dir}")

    if not os.path.exists(channel_table):
        raise FileNotFoundError(
            f"Channel table not found: {channel_table}. Please run etl_unified.py first."
        )

    os.makedirs(output_dir, exist_ok=True)

    conn = duckdb.connect()
    conn.execute("PRAGMA threads=32;")
    conn.execute("SET preserve_insertion_order=false;")

    # Pre-load small channel_table into memory (only ~330 rows)
    conn.execute(
        f"CREATE TABLE channels AS SELECT Global_Channel_ID, min_valid, max_valid FROM read_parquet('{channel_table}')"
    )

    for sensor in sensors:
        # Use glob pattern directly - DuckDB handles this efficiently
        parquet_glob = os.path.join(data_root, sensor, "*", "*.parquet")
        csv_path = os.path.join(output_dir, f"{sensor}_normalizations.csv")

        # Check if sensor directory exists
        sensor_dir = os.path.join(data_root, sensor)
        if not os.path.exists(sensor_dir):
            print(f"\nSkipping {sensor}: directory not found")
            continue

        print(f"\nProcessing {sensor}...")

        start = time.time()

        # Conv needs additional height/pressure filtering (from QCLimits)
        # GPS channels use 0.5 hPa min pressure, others use 200 hPa
        if sensor == "conv":
            gps_ids = ", ".join(str(x) for x in CONV_GPS_GLOBAL_IDS)
            extra_where = f"""
            AND o.Height BETWEEN {QCLimits.HEIGHT_MIN} AND {QCLimits.HEIGHT_MAX}
            AND o.Pressure <= {QCLimits.PRESSURE_MAX}
            AND o.Pressure >= CASE 
              WHEN o.Global_Channel_ID IN ({gps_ids}) THEN {QCLimits.PRESSURE_MIN_GPS}
              ELSE {QCLimits.PRESSURE_MIN_DEFAULT}
            END
            """
        else:
            extra_where = ""

        # Raw_Channel_ID is 1-indexed
        sensor_offset = SENSOR_OFFSET[sensor]

        # Stream directly using DuckDB glob - efficient for many files
        sql = f"""
        COPY (
          -- per-platform stats
          SELECT
            o.Global_Channel_ID - {sensor_offset} + 1 AS Raw_Channel_ID,
            o.Platform_ID,
            STDDEV(o.Observation) AS obs_std,
            AVG(o.Observation) AS obs_mean,
            MIN(o.Observation) AS obs_min,
            MAX(o.Observation) AS obs_max
          FROM read_parquet('{parquet_glob}') o
          JOIN channels c ON o.Global_Channel_ID = c.Global_Channel_ID
          WHERE o.Observation BETWEEN c.min_valid AND c.max_valid
            AND o.Observation IS NOT NULL
            {extra_where}
          GROUP BY o.Global_Channel_ID, o.Platform_ID

          UNION ALL

          -- overall stats (Platform_ID = -1)
          SELECT
            o.Global_Channel_ID - {sensor_offset} + 1 AS Raw_Channel_ID,
            -1 AS Platform_ID,
            STDDEV(o.Observation) AS obs_std,
            AVG(o.Observation) AS obs_mean,
            MIN(o.Observation) AS obs_min,
            MAX(o.Observation) AS obs_max
          FROM read_parquet('{parquet_glob}') o
          JOIN channels c ON o.Global_Channel_ID = c.Global_Channel_ID
          WHERE o.Observation BETWEEN c.min_valid AND c.max_valid
            AND o.Observation IS NOT NULL
            {extra_where}
          GROUP BY o.Global_Channel_ID

          ORDER BY Raw_Channel_ID, Platform_ID
        )
        TO '{csv_path}'
        (HEADER TRUE, DELIMITER ',');
        """

        try:
            conn.execute(sql)
            print(f"  Wrote to {csv_path} ({time.time() - start:.1f}s)")
        except Exception as e:
            print(f"  Error: {e}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
