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
# ruff: noqa: S608  # SQL constructed from trusted internal config, not user input
import argparse
import glob
import os
import pathlib

import duckdb

from datasets.sensors import CONV_CHANNELS, CONV_PLATFORMS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalizations for conventional observation data"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/lustre/fs1/portfolios/coreai/projects/coreai_climate_earth2/datasets/ufs-replay/processed_obs_v6_ges/",
        help="Root directory containing processed observation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="normalizations",
        help="Output directory for normalization files ",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = args.data_root
    output_dir = args.output_dir or (pathlib.Path(__file__).parent / "normalizations")

    print("Processing conventional sensor data...")
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    conn = duckdb.connect()
    conn.execute("PRAGMA threads=32;")

    # Build channel mapping from CONV_CHANNELS
    channel_mapping = {
        platform: {
            ch.nc_column: i
            for i, ch in enumerate(CONV_CHANNELS, start=1)
            if ch.platform == platform
        }
        for platform in CONV_PLATFORMS
    }

    all_results = []

    # Process each platform separately
    for platform, columns in channel_mapping.items():
        print(f"\nProcessing {platform} platform...")

        # Find parquet files for this platform
        platform_dir = os.path.join(data_root, f"conv_{platform}")
        if not os.path.exists(platform_dir):
            print(f"  Platform directory not found: {platform_dir}")
            continue

        parquet_glob = os.path.join(platform_dir, "*.parquet")
        parquet_files = glob.glob(parquet_glob)
        print(f"  Found {len(parquet_files)} parquet files")

        if not parquet_files:
            print(f"  No parquet files found for {platform}")
            continue

        # Process each column for this platform
        for column_name, channel_id in columns.items():
            print(f"    Processing {column_name} (channel {channel_id})...")

            # Create list of files for SQL query
            files_str = "', '".join(parquet_files)

            # Build filtering conditions based on platform and column
            min_pressure = 200 if platform != "gps" else 0.5
            base_filters = [
                f"{column_name} IS NOT NULL",
                "Height BETWEEN 0 AND 60000",  # 0-60k meters
                f"Pressure BETWEEN {min_pressure} AND 1100",  # 200-1100 hPa
            ]

            # Add platform/column-specific filters
            if column_name == "Specific_Humidity_at_Obs_Location":
                base_filters.append(f"{column_name} BETWEEN 0 AND 1")
            elif column_name == "u_Observation" or column_name == "v_Observation":
                base_filters.append(f"{column_name} BETWEEN -100 AND 100")
            elif column_name == "Temperature_at_Obs_Location":
                base_filters.append(f"{column_name} BETWEEN 150 AND 350")
            elif column_name == "Observation":
                # GPS RO observations need special handling
                if platform == "gps":
                    # For GPS, we'll apply the same filters as q and temp
                    # Note: GPS Observation might need different ranges, adjust as needed
                    pass
                else:
                    # For other platforms (ps, q, t), apply standard ranges
                    if platform == "q":
                        base_filters.append(f"{column_name} BETWEEN 0 AND 1")
                    elif platform == "t":
                        base_filters.append(f"{column_name} BETWEEN 150 AND 350")
                    # ps doesn't need additional filtering beyond height/pressure

            # Combine all filters
            where_clause = " AND ".join(base_filters)

            # Compute statistics for this column (including min/max)
            sql = f"""  # noqa: S608
            SELECT
                {channel_id} AS Raw_Channel_ID,
                -1 AS Platform_ID,
                CAST(STDDEV({column_name}) AS DOUBLE) AS obs_std,
                CAST(AVG({column_name}) AS DOUBLE) AS obs_mean,
                CAST(MIN({column_name}) AS DOUBLE) AS obs_min,
                CAST(MAX({column_name}) AS DOUBLE) AS obs_max
            FROM read_parquet(['{files_str}'])
            WHERE {where_clause}
            """

            try:
                result = conn.execute(sql).fetchone()
                if result and result[2] is not None:  # Check if std is not null
                    all_results.append(
                        {
                            "Raw_Channel_ID": result[0],
                            "Platform_ID": result[1],
                            "obs_std": result[2],
                            "obs_mean": result[3],
                            "obs_min": result[4],
                            "obs_max": result[5],
                        }
                    )
                    print(
                        f"      Mean: {result[3]:.6f}, Std: {result[2]:.6f}, Min: {result[4]:.6f}, Max: {result[5]:.6f}"
                    )
                else:
                    print(f"      No valid data found for {column_name}")
            except Exception as e:
                print(f"      Error processing {column_name}: {e}")

    # Write results to CSV
    csv_path = os.path.join(output_dir, "conv_normalizations_v6.csv")

    if all_results:
        # Convert to DataFrame and sort
        import pandas as pd

        df = pd.DataFrame(all_results)
        df = df.sort_values(["Raw_Channel_ID", "Platform_ID"])

        # Write to CSV
        df.to_csv(csv_path, index=False)
        print(f"\nWrote {len(df)} normalization records to {csv_path}")

        # Print summary
        print("\nSummary:")
        for channel_id in sorted(df["Raw_Channel_ID"].unique()):
            channel_data = df[df["Raw_Channel_ID"] == channel_id]
            print(f"  Channel {channel_id}: {len(channel_data)} records")
    else:
        print("\nNo normalization data computed!")

    conn.close()


if __name__ == "__main__":
    main()
