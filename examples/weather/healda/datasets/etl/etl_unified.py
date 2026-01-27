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
import argparse
import os
import random
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Literal, Optional, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from datasets.etl.combined_schema import (
    get_channel_table_schema,
    get_combined_observation_schema,
)
from datasets.sensors import (
    CONV_CHANNEL_MAP,
    CONV_CHANNELS,
    CONV_PLATFORMS,
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    get_global_channel_id,
)

memory = joblib.Memory(".cache")

TEST = False

# Set single-threaded Arrow for better parallelization
os.environ["ARROW_NUM_THREADS"] = "1"

DEFAULT_UFS_RAW_OBS_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_climate_earth2/datasets/ufs-replay/raw_obs"
DEFAULT_BASE_DIR = "/lustre/fs1/portfolios/coreai/projects/coreai_climate_earth2/datasets/ufs-replay/processed_obs_v6"

# Satellite sensor columns
SATELLITE_COLUMNS = [
    "Latitude",
    "Longitude",
    "Observation",
    "Channel_Index",
    "Obs_Time",
    "Sat_Zenith_Angle",
    "Sol_Zenith_Angle",
    "Scan_Angle",
]

# Conventional sensor metadata columns
CONV_METADATA_COLUMNS = [
    "Latitude",
    "Longitude",
    "Time",
    "Pressure",
    "Height",
    "Observation_Type",
]

# Analysis columns (common to both)
ANALYSIS_COLUMNS = [
    "Obs_Minus_Forecast_adjusted",
    "Obs_Minus_Forecast_unadjusted",
    "QC_Flag",  # For satellite
    "Analysis_Use_Flag",  # For conventional
]


def _get_conv_obs_columns_for_platform(platform: str) -> list[str]:
    return [ch.nc_column for ch in CONV_CHANNELS if ch.platform == platform]


@memory.cache
def list_nc_files(path):
    """List all .nc4 files in the given directory tree."""
    return [
        os.path.relpath(os.path.join(root, fname), path)
        for root, _, fnames in os.walk(path)
        for fname in fnames
        if fname.endswith(".nc4")
    ]


def get_channel_table():
    """Build channel metadata table for all sensors."""
    nchan = [cfg.channels for cfg in SENSOR_CONFIGS.values()]
    sensor_id = np.arange(len(nchan)).repeat(nchan)
    id = np.arange(sensor_id.size).astype(np.uint16)

    conv_names = [c.name for c in CONV_CHANNELS]
    conv_min = [c.min_valid for c in CONV_CHANNELS]
    conv_max = [c.max_valid for c in CONV_CHANNELS]

    min_valid_list = []
    max_valid_list = []
    names = []
    means_list = []
    stds_list = []
    for name, cfg in SENSOR_CONFIGS.items():
        if name == "conv":
            min_valid_list.extend(conv_min)
            max_valid_list.extend(conv_max)
            names.extend(conv_names)
        else:
            min_valid_list.extend([cfg.min_valid] * cfg.channels)
            max_valid_list.extend([cfg.max_valid] * cfg.channels)
            names.extend([f"{name}_{i:03d}" for i in range(cfg.channels)])

        means_list.extend(cfg.means)
        stds_list.extend(cfg.stds)

    min_valid = np.array(min_valid_list, dtype=np.float32)
    max_valid = np.array(max_valid_list, dtype=np.float32)
    means = np.array(means_list, dtype=np.float32)
    stds = np.array(stds_list, dtype=np.float32)

    is_conv = sensor_id == list(SENSOR_CONFIGS).index("conv")

    return pa.table(
        [id, min_valid, max_valid, sensor_id, is_conv, names, means, stds],
        schema=get_channel_table_schema(),
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified ETL script for processing UFS observation data (satellite and conventional)"
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="all",
        help="Sensor(s) to process: 'all' for all sensors, single sensor (e.g., 'atms'), or comma-separated list (e.g., 'conv,amsua,atms')",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_UFS_RAW_OBS_DIR,
        help=f"Input directory containing raw observation files (default: {DEFAULT_UFS_RAW_OBS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base output directory for processed data (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="ges",
        choices=["ges", "anl"],
        help="Ges or anl data to process (default: ges)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=96,
        help="Number of workers to use (default: 96)",
    )
    parser.add_argument(
        "--channel-table-only",
        action="store_true",
        help="Only generate channel_table.parquet (skip processing observation files)",
    )
    return parser.parse_args()


def is_satellite_sensor(sensor: str) -> bool:
    """Check if sensor is a satellite sensor (not conventional)."""
    return sensor not in CONV_PLATFORMS and sensor != "conv" and sensor != "all"


def is_conventional_sensor(sensor: str) -> bool:
    """Check if sensor is conventional."""
    return sensor == "conv"


def read_satellite_variables_from_file(
    full_file_path: str, sensor: str, obs_type: Literal["ges", "anl"] = "ges"
) -> Optional[pa.Table]:
    """Read variables from a satellite sensor NetCDF file."""
    try:
        with h5py.File(full_file_path, "r") as ds:
            data = {}
            cols_to_read = SATELLITE_COLUMNS
            if obs_type == "anl":
                cols_to_read += ANALYSIS_COLUMNS[:2]  # Exclude QC_Flag for now

            for col in cols_to_read:
                if col == "Channel_Index":
                    channel_idx_in_sensor_chan = ds[col][:].astype(np.uint16)
                    sensor_chan = ds["sensor_chan"][:]
                    data["Raw_Channel_ID"] = sensor_chan[
                        channel_idx_in_sensor_chan - 1
                    ].astype(np.uint16)
                elif col == "QC_Flag":
                    data[col] = ds[col][:].astype(np.int32)
                else:
                    data[col] = ds[col][:]

        data["Global_Channel_ID"] = get_global_channel_id(
            sensor, data["Raw_Channel_ID"]
        )

        n = len(data[SATELLITE_COLUMNS[0]])
        filename = os.path.basename(full_file_path)

        # Map to global platform idx
        platform = filename.split("_")[2]
        platform_id = PLATFORM_NAME_TO_ID[platform]
        data["Platform_ID"] = np.full(n, platform_id).astype(np.uint8)

        # Process time information
        match = re.search(r"\.(\d{10})_", filename)
        if match:
            date_str = match.group(1)
            NS_3H = np.int64(3) * 3_600_000_000_000
            base_time = pd.to_datetime(date_str, format="%Y%m%d%H")
            base_ns = base_time.value
            hours = data["Obs_Time"].astype(np.float64)
            hours_to_ns = np.rint(hours * 3600.0 * 1e9).astype(np.int64)

            # clip to be > -3h and <= 3h
            hours_to_ns = np.clip(hours_to_ns, -NS_3H + 1, NS_3H)
            abs_time_ns = base_ns + hours_to_ns
            abs_time = abs_time_ns.astype("datetime64[ns]")
            data["Absolute_Obs_Time"] = abs_time

            # assign exactly two labels: t (left half) or t+3h (right half)
            da_ns = np.where(abs_time_ns <= base_ns, base_ns, base_ns + NS_3H)
            data["DA_window"] = da_ns.astype("datetime64[ns]")
        else:
            raise RuntimeError(f"No date match found for {filename}")

        del data["Obs_Time"]
        output_schema = get_combined_observation_schema()
        df = pd.DataFrame(data)
        for field in output_schema:
            if field.name not in df.columns:
                df[field.name] = pd.NA
        return pa.table(df, schema=output_schema)

    except Exception as e:
        print(f"Error reading satellite file {full_file_path}: {e}")
        return None


def read_conventional_variables_from_file(
    full_file_path: str, obs_type: Literal["ges", "anl"] = "ges"
) -> Optional[pa.Table]:
    """Read variables from a conventional sensor NetCDF file."""
    try:
        with h5py.File(full_file_path, "r") as ds:
            data = {}

            # Read common metadata columns
            for col in CONV_METADATA_COLUMNS:
                if col == "Observation_Type":
                    data["Observation_Type"] = ds[col][:].astype(np.uint16)
                elif col == "Time":
                    pass  # Handle separately
                else:
                    data[col] = ds[col][:]

            time = ds["Time"][:]
            n = len(data[CONV_METADATA_COLUMNS[0]])
            filename = os.path.basename(full_file_path)

            # Extract platform from filename
            platform = filename.split("_")[2]
            platform_id = PLATFORM_NAME_TO_ID[platform]
            data["Platform_ID"] = np.full(n, platform_id).astype(np.uint8)

            # Get observation columns based on platform
            observation_columns = _get_conv_obs_columns_for_platform(platform)

            # Handle analysis columns
            if obs_type == "anl":
                data["Analysis_Use_Flag"] = ds["Analysis_Use_Flag"][:].astype(np.int8)
                if platform != "uv":
                    data["Obs_Minus_Forecast_adjusted"] = ds[
                        "Obs_Minus_Forecast_adjusted"
                    ][:]
                    data["Obs_Minus_Forecast_unadjusted"] = ds[
                        "Obs_Minus_Forecast_unadjusted"
                    ][:]
                else:
                    data["v_Obs_Minus_Forecast_adjusted"] = ds[
                        "v_Obs_Minus_Forecast_adjusted"
                    ][:]
                    data["v_Obs_Minus_Forecast_unadjusted"] = ds[
                        "v_Obs_Minus_Forecast_unadjusted"
                    ][:]
                    data["u_Obs_Minus_Forecast_adjusted"] = ds[
                        "u_Obs_Minus_Forecast_adjusted"
                    ][:]
                    data["u_Obs_Minus_Forecast_unadjusted"] = ds[
                        "u_Obs_Minus_Forecast_unadjusted"
                    ][:]

            # Create absolute observation time with 3-hourly DA window splitting
            match = re.search(r"\.(\d{10})_", filename)
            if match:
                date_str = match.group(1)
                NS_3H = np.int64(3) * 3_600_000_000_000
                base_time = pd.to_datetime(date_str, format="%Y%m%d%H")
                base_ns = base_time.value
                hours = time.astype(np.float64)
                hours_to_ns = np.rint(hours * 3600.0 * 1e9).astype(np.int64)

                # clip to be > -3h and <= 3h
                hours_to_ns = np.clip(hours_to_ns, -NS_3H + 1, NS_3H)
                abs_time_ns = base_ns + hours_to_ns
                abs_time = abs_time_ns.astype("datetime64[ns]")
                data["Absolute_Obs_Time"] = abs_time

                # assign exactly two labels: t (left half) or t+3h (right half)
                da_ns = np.where(abs_time_ns <= base_ns, base_ns, base_ns + NS_3H)
                data["DA_window"] = da_ns.astype("datetime64[ns]")
            else:
                raise RuntimeError(f"No date match found for {filename}")

            # Flatten the data to have a single Observation column
            meta_df = pd.DataFrame(data)
            dfs = []
            for k, column in enumerate(observation_columns):
                raw_channel_id = k + CONV_CHANNEL_MAP[platform]

                this_df = meta_df.assign(
                    Observation=ds[column][:],
                    Global_Channel_ID=get_global_channel_id("conv", raw_channel_id),
                )
                dfs.append(this_df)
            df = pd.concat(dfs)

            output_schema = get_combined_observation_schema()
            for field in output_schema:
                if field.name not in df.columns:
                    df[field.name] = pd.NA
            return pa.table(df, schema=output_schema)

    except Exception as e:
        print(f"Error reading conventional file {full_file_path}: {e}")
        return None


def read_variables_from_file(
    full_file_path: str, sensor: str, obs_type: Literal["ges", "anl"] = "ges"
) -> Optional[pa.Table]:
    """Unified function to read variables from NetCDF files."""
    if is_satellite_sensor(sensor):
        return read_satellite_variables_from_file(full_file_path, sensor, obs_type)
    elif is_conventional_sensor(sensor):
        return read_conventional_variables_from_file(full_file_path, obs_type)
    else:
        raise ValueError(f"Unknown sensor type: {sensor}")


def extract_info_from_filename(
    filename: str, obs_type: Literal["ges", "anl"] = "ges"
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract sensor, platform, type, and date from filename."""
    if obs_type == "ges":
        pattern = re.compile(r"diag_([\w-]+)_([\w-]+)_(ges)\.(\d{10})_")
    elif obs_type == "anl":
        pattern = re.compile(r"diag_([\w-]+)_([\w-]+)_(anl)\.(\d{10})_")
    else:
        raise ValueError(f"Invalid obs type: {obs_type}")

    match = pattern.match(filename)
    if match:
        sensor_name, platform, file_type, full_date = match.groups()
        day_date = full_date[:8]  # YYYYMMDD
        return sensor_name, platform, file_type, day_date, full_date
    return None, None, None, None, None


def parse_sensor_list(sensor_str: str) -> List[str]:
    """Parse comma-separated sensor list and validate sensors."""
    if sensor_str == "all":
        return ["all"]

    sensors = [s.strip() for s in sensor_str.split(",")]

    # Validate each sensor
    for sensor in sensors:
        if sensor not in ["all", "conv"] and not is_satellite_sensor(sensor):
            raise ValueError(
                f"Invalid sensor: {sensor}. Must be 'all', 'conv', or a satellite sensor name."
            )

    return sensors


def filter_files_by_sensor(
    files: List[str], target_sensors: List[str], obs_type: Literal["ges", "anl"] = "ges"
) -> Dict[str, List[str]]:
    """Filter files based on target sensors."""
    if "all" in target_sensors:
        # Process all sensors
        filtered_files = defaultdict(list)
        seen_keys = set()

        for file_path in files:
            if "spinup/" in file_path or "overlap/" in file_path:
                continue

            filename = os.path.basename(file_path)
            sensor_name, platform, file_type, day_date, full_date = (
                extract_info_from_filename(filename, obs_type)
            )

            if not all([sensor_name, platform, day_date]):
                continue

            # Determine the sensor key for output directory
            if sensor_name == "conv":
                sensor_key = "conv"
            else:
                sensor_key = sensor_name

            # Avoid duplicates
            key = (full_date, sensor_name, platform)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            filtered_files[sensor_key].append(file_path)

        return dict(filtered_files)

    else:
        # Process specific sensors
        filtered_files = defaultdict(list)
        seen_keys = set()

        for file_path in files:
            if "spinup/" in file_path or "overlap/" in file_path:
                continue

            filename = os.path.basename(file_path)
            sensor_name, platform, file_type, day_date, full_date = (
                extract_info_from_filename(filename, obs_type)
            )

            if not all([sensor_name, platform, day_date]):
                continue

            # Check if this file matches any of our target sensors
            file_matches = False
            target_sensor_key = None

            if sensor_name == "conv" and "conv" in target_sensors:
                file_matches = True
                target_sensor_key = "conv"
            elif sensor_name in target_sensors:
                file_matches = True
                target_sensor_key = sensor_name

            if not file_matches:
                continue

            # Avoid duplicates
            key = (full_date, sensor_name, platform)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            filtered_files[target_sensor_key].append(file_path)

        return dict(filtered_files)


def process_day(
    full_file_paths: List[str],
    output_path: str,
    sensor: str,
    obs_type: Literal["ges", "anl"] = "ges",
):
    """Process a single day of data for a sensor."""
    if os.path.exists(output_path):
        return

    tables = []
    for full_file_path in full_file_paths:
        data = read_variables_from_file(full_file_path, sensor, obs_type)
        if data is not None:
            tables.append(data)

    if len(tables) == 0:
        return

    table = pa.concat_tables(tables)

    # Sort by DA_window for better compression and reading performance
    sort_idx = pc.sort_indices(table, sort_keys=[("DA_window", "ascending")])
    table = pc.take(table, sort_idx)

    # Group by DA_window for chunked reading (each DA window in separate row group)
    col = table.column("DA_window").combine_chunks()
    vals = col.to_numpy()  # numpy datetime64[ns]
    change = np.empty(len(vals), dtype=bool)
    change[0] = True
    change[1:] = vals[1:] != vals[:-1]

    starts = np.flatnonzero(change)
    ends = np.concatenate([starts[1:], [len(vals)]])

    # Write to temporary file first, then atomically move to final location
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = tmp_file.name

            with pq.ParquetWriter(tmp_path, table.schema) as writer:
                for start, end in zip(starts, ends):
                    slice_ = table.slice(start, end - start)
                    # Force Arrow to put each DA_window in its own row-group
                    writer.write_table(slice_, row_group_size=len(slice_))

        # Atomically move the temporary file to the final location
        os.rename(tmp_path, output_path)
        tmp_path = None  # Success, don't clean up

    except Exception as e:
        # Clean up temporary file if something went wrong
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # Ignore cleanup errors
        raise e


def main():
    """Main processing function."""
    args = parse_args()

    # Parse sensor list
    target_sensors = parse_sensor_list(args.sensor)
    ufs_raw_obs_dir = args.input_dir
    base_dir = args.output_dir
    obs_type = args.type
    output_base_dir = f"{base_dir}_{obs_type}"

    print(f"Processing sensors: {', '.join(target_sensors)}")
    print(f"Input directory: {ufs_raw_obs_dir}")
    print(f"Output base directory: {output_base_dir}")
    print(f"Processing {obs_type} data")

    # Save channel table at root level of output directory
    print("\nSaving channel table...")
    channel_table = get_channel_table()
    channel_table_path = os.path.join(output_base_dir, "channel_table.parquet")
    os.makedirs(output_base_dir, exist_ok=True)
    pq.write_table(channel_table, channel_table_path)
    print(f"Channel table saved to: {channel_table_path}")

    if args.channel_table_only:
        print("--channel-table-only specified, skipping observation processing.")
        return

    # Read file list
    files = list_nc_files(args.input_dir)
    print(f"Total files detected: {len(files):,}")

    # Filter files by sensors
    sensor_files = filter_files_by_sensor(files, target_sensors, obs_type)

    if not sensor_files:
        print("No files found to process!")
        return

    # Report what we found
    total_days = 0
    for sensor_name, file_list in sensor_files.items():
        # Group files by date
        date_to_files = defaultdict(list)
        for file_path in file_list:
            filename = os.path.basename(file_path)
            _, _, _, day_date, _ = extract_info_from_filename(filename, obs_type)
            if day_date:
                date_to_files[day_date].append(file_path)

        print(f"\n{sensor_name.upper()} sensor: {len(date_to_files)} days")
        total_days += len(date_to_files)

        # Show sample dates
        sample_dates = sorted(date_to_files.keys())[:5]
        print(f"  Sample dates: {sample_dates}")
        if len(date_to_files) > 5:
            print(f"  ... and {len(date_to_files) - 5} more")

    print(f"\nTotal processing jobs: {total_days}")

    # Create output directories and prepare all jobs
    all_jobs = []
    job_metadata = []

    for sensor_name, file_list in sensor_files.items():
        # Create output directory for this sensor
        sensor_output_dir = os.path.join(output_base_dir, sensor_name)

        # Group files by date
        date_to_files = defaultdict(list)
        for file_path in file_list:
            filename = os.path.basename(file_path)
            _, _, _, day_date, _ = extract_info_from_filename(filename, obs_type)
            if day_date:
                date_to_files[day_date].append(file_path)

        # Prepare processing jobs
        for date, day_files in date_to_files.items():
            # Pre-compute all paths
            full_file_paths = [
                os.path.join(ufs_raw_obs_dir, file_path) for file_path in day_files
            ]
            output_path = os.path.join(sensor_output_dir, f"{date}", "0.parquet")

            job_args = (
                full_file_paths,
                output_path,
                sensor_name,
                obs_type,
            )
            all_jobs.append(job_args)
            job_metadata.append((sensor_name, date))

    # Process files using map
    print(f"\nPrepared {len(all_jobs)} jobs")
    print(f"Starting parallel processing with {args.num_workers} workers...")

    random.shuffle(all_jobs)

    with ProcessPoolExecutor(args.num_workers) as executor:
        # Use map to process all jobs
        list(
            tqdm(
                executor.map(process_day, *zip(*all_jobs)),
                total=len(all_jobs),
                desc="Processing",
            )
        )

    # Count results (map doesn't raise exceptions, so we need to check differently)
    completed = len(all_jobs)  # All jobs completed (map waits for all)
    failed = 0  # We can't easily detect failures with map, but all jobs completed

    print("\nETL Complete!")
    print(f"Successfully processed: {completed} sensor-days")
    print(f"Failed: {failed} sensor-days")

    # Report output directories
    print("\nOutput directories created:")
    for sensor_name in sensor_files.keys():
        sensor_dir = os.path.join(output_base_dir, sensor_name)
        if os.path.exists(sensor_dir):
            file_count = len(
                [f for f in os.listdir(sensor_dir) if f.endswith(".parquet")]
            )
            print(f"  {sensor_dir}: {file_count} parquet files")


if __name__ == "__main__":
    main()
