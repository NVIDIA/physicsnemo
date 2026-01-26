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
import os
import duckdb
import pathlib
import time
import argparse

DEFAULT_SENSORS = [
    "mhs",
    "amsua",
    "atms",
    "amsub",
]  #  'iasi', 'cris-fsr'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalizations for UFS observation data"
    )
    parser.add_argument(
        "--sensors",
        type=str,
        nargs="+",
        default=DEFAULT_SENSORS,
        help=f"List of sensor names to process (default: {DEFAULT_SENSORS})",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/lustre/fs1/portfolios/coreai/projects/coreai_climate_earth2/datasets/ufs-replay/processed_obs_v4/",
        help="Root directory containing processed observation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for normalization files (default: src/earth2obs/datasets/normalizations)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sensors = args.sensors
    data_root = args.data_root
    output_dir = args.output_dir or (pathlib.Path(__file__).parent / "normalizations")

    print(f"Processing sensors: {sensors}")
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    conn = duckdb.connect()
    conn.execute("PRAGMA threads=32;")
    channel_col, platform_col = "raw_channel_id", "platform_id"

    for sensor in sensors:
        parquet_glob = os.path.join(data_root, sensor, "*.parquet")
        csv_path = os.path.join(output_dir, f"{sensor}_normalizations.csv")

        print(f"Processing {sensor}...")
        start = time.time()
        sql = f"""
        COPY (
          -- per-platform_id stats
          SELECT
            {channel_col},
            {platform_col},
            STDDEV(Observation) AS obs_std,
            AVG(Observation)   AS obs_mean
          FROM read_parquet('{parquet_glob}')
          WHERE Observation BETWEEN 0 AND 400
          GROUP BY {channel_col}, {platform_col}

          UNION ALL

          -- overall stats, tagged with platform_id = '-1'
          SELECT
            {channel_col},
            '-1'                AS {platform_col},
            STDDEV(Observation) AS obs_std,
            AVG(Observation)    AS obs_mean
          FROM read_parquet('{parquet_glob}')
          WHERE Observation BETWEEN 0 AND 400
          GROUP BY {channel_col}

          ORDER BY {channel_col}, {platform_col}
        )
        TO '{csv_path}'
        (HEADER TRUE, DELIMITER ',');
        """

        conn.execute(sql)
        print(
            f"Completed {sensor}. Wrote to {csv_path}. Took {time.time() - start:.2f} seconds"
        )

    conn.close()


if __name__ == "__main__":
    main()
