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
import pathlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SensorConfig:
    """
    Sensor metadata that sets up data loading.
    Defines the sensor name, platforms, channels, and normalization stats.
    """

    name: str
    platforms: list[str]
    channels: int
    nc_file_template: str
    means: np.ndarray = field(init=False)
    stds: np.ndarray = field(init=False)
    min_valid: float = 0.0
    max_valid: float = 400.0
    sensor_type: str = "microwave"
    raw_to_local: np.ndarray = field(
        init=False
    )  # lookup-table: raw_id →  local channel

    def __post_init__(self):
        base = pathlib.Path(__file__).parent / "etl/normalizations"
        norm_file = base / f"{self.name}_normalizations.csv"

        if norm_file.exists():
            df = pd.read_csv(norm_file)
            # Col -1 is the avg across all platforms
            channel_col = "Raw_Channel_ID"
            df = df[df["Platform_ID"] == -1].sort_values(channel_col)

            self.means = df["obs_mean"].to_numpy()
            self.stds = df["obs_std"].to_numpy()

            # build a raw‑to‑local LUT
            raw_ids = df[channel_col].to_numpy()
            max_raw = raw_ids.max()
            lookup_table = np.full(max_raw + 1, 0, dtype=int)
            for local_idx, raw in enumerate(raw_ids, start=1):
                lookup_table[raw] = local_idx
            self.raw_to_local = lookup_table

        else:
            warnings.warn(
                f"No normalization file for {self.name!r}. "
                "Defaulting to means=0, stds=1, identity mapping."
            )
            self.means = np.zeros(self.channels, dtype=float)
            self.stds = np.ones(self.channels, dtype=float)
            self.raw_to_local = np.arange(self.channels + 1, dtype=int)


def get_global_channel_id(sensor, raw_channel_ids):
    """Map per-sensor raw channel IDs to unified global IDs (no overlap across sensors)."""
    raw_to_local = SENSOR_CONFIGS[sensor].raw_to_local
    channel_offset = SENSOR_OFFSET[sensor]
    local_channels = raw_to_local[raw_channel_ids] - 1  # Convert to 0-based indexing
    return (local_channels + channel_offset).astype(np.uint16)


SENSOR_CONFIGS = {
    "atms": SensorConfig(
        name="atms",
        platforms=["npp", "n20"],
        channels=22,
        nc_file_template="diag_atms_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    "mhs": SensorConfig(
        name="mhs",
        platforms=["metop-a", "metop-b", "metop-c", "n18", "n19"],
        channels=5,
        nc_file_template="diag_mhs_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    "amsua": SensorConfig(
        name="amsua",
        platforms=["metop-a", "metop-b", "metop-c", "n15", "n16", "n17", "n18", "n19"],
        channels=15,
        nc_file_template="diag_amsua_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    "amsub": SensorConfig(
        name="amsub",
        platforms=["n15", "n16", "n17"],
        channels=5,
        nc_file_template="diag_amsub_{platform}_ges.{date}_control.nc4",
        min_valid=0.0,
        max_valid=400.0,
        sensor_type="microwave",
    ),
    "iasi": SensorConfig(
        name="iasi",
        platforms=["metop-a", "metop-b", "metop-c"],
        channels=175,
        nc_file_template="diag_iasi_{platform}_ges.{date}_control.nc4",
        min_valid=150.0,
        max_valid=350.0,
        sensor_type="infrared",
    ),
    "cris-fsr": SensorConfig(
        name="cris-fsr",
        platforms=["npp", "n20"],
        channels=100,
        nc_file_template="diag_cris_fsr_{platform}_ges.{date}_control.nc4",
        min_valid=150.0,
        max_valid=350.0,
        sensor_type="infrared",
    ),
    "conv": SensorConfig(
        name="conv",
        platforms=[],  # platform idea doesn't apply to conv
        channels=8,  # all conv sensors stacked (gps angle, gps temp, gps spfh, ps, q, t, u, v)
        nc_file_template="conv_{platform}_ges.{date}_control.nc4",
        sensor_type="conv",
    ),
}


class QCLimits:
    """Conventional Observation QC filtering limits."""

    # Height limits (meters)
    HEIGHT_MIN = 0
    HEIGHT_MAX = 60000
    # Pressure limits (hPa)
    PRESSURE_MIN_GPS = 0.5
    PRESSURE_MIN_DEFAULT = 200
    PRESSURE_MAX = 1100


# Concept of platform for conv is only used in etl, does not apply outside of etl. All conv obs have platform 0
@dataclass(frozen=True)
class ConvChannel:
    """Conv sensor channel definition, used for ETL and creating channel table"""

    name: str
    platform: str
    nc_column: str
    min_valid: float
    max_valid: float


CONV_CHANNELS = [
    ConvChannel("gps_angle", "gps", "Observation", float("-inf"), float("inf")),
    ConvChannel("gps_t", "gps", "Temperature_at_Obs_Location", 150, 350),
    ConvChannel("gps_q", "gps", "Specific_Humidity_at_Obs_Location", 0.0, 1.0),
    ConvChannel("ps", "ps", "Observation", float("-inf"), float("inf")),
    ConvChannel("q", "q", "Observation", 0, 1),
    ConvChannel("t", "t", "Observation", 150, 350),
    ConvChannel("u", "uv", "u_Observation", -100, 100),
    ConvChannel("v", "uv", "v_Observation", -100, 100),
]

CONV_CHANNEL_NAMES = [c.name for c in CONV_CHANNELS]
CONV_PLATFORMS = list(dict.fromkeys(c.platform for c in CONV_CHANNELS))
CONV_GPS_CHANNELS = [i for i, c in enumerate(CONV_CHANNELS) if c.platform == "gps"]
CONV_GPS_LEVEL2_CHANNELS = [
    i for i, c in enumerate(CONV_CHANNELS) if c.name in ("gps_t", "gps_q")
]
CONV_UV_CHANNELS = [i for i, c in enumerate(CONV_CHANNELS) if c.platform == "uv"]
CONV_UV_IN_SITU_TYPES = [220, 221, 229, 230, 231, 232, 233, 234, 235, 280, 282]


def _build_conv_channel_map() -> dict[str, int]:
    """Build map from platform name to first channel ID (1-indexed)."""
    channel_map = {}
    for i, channel in enumerate(CONV_CHANNELS, start=1):
        if channel.platform not in channel_map:
            channel_map[channel.platform] = i
    return channel_map


CONV_CHANNEL_MAP = _build_conv_channel_map()


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


PLATFORM_NAME_TO_ID = {
    "aqua": 0,
    "aura": 1,
    "f10": 2,
    "f11": 3,
    "f13": 4,
    "f14": 5,
    "f15": 6,
    "g08": 7,
    "g10": 8,
    "g11": 9,
    "g12": 10,
    "m08": 11,
    "m09": 12,
    "m10": 13,
    "metop-a": 14,
    "metop-b": 15,
    "metop-c": 16,
    "n11": 17,
    "n12": 18,
    "n14": 19,
    "n15": 20,
    "n16": 21,
    "n17": 22,
    "n18": 23,
    "n19": 24,
    "n20": 25,
    "npp": 26,
    "gps": 27,
    "ps": 28,
    "q": 29,
    "t": 30,
    "uv": 31,
}

PLATFORM_ID_TO_NAME = {v: k for k, v in PLATFORM_NAME_TO_ID.items()}

NPLATFORMS = _next_power_of_two(max(len(PLATFORM_NAME_TO_ID), 64))  # 64

SENSOR_OFFSET = {}
offset = 0
for name, cfg in SENSOR_CONFIGS.items():
    SENSOR_OFFSET[name] = offset
    offset += cfg.channels
NCHANNEL = _next_power_of_two(max(offset, 1024))  # 1024

# GPS channel Global_Channel_IDs (for use in SQL queries against parquet)
CONV_GPS_GLOBAL_IDS = [SENSOR_OFFSET["conv"] + i for i in CONV_GPS_CHANNELS]


SENSOR_NAME_TO_ID = {name: idx for idx, name in enumerate(SENSOR_CONFIGS.keys())}
SENSOR_ID_TO_NAME = {idx: name for name, idx in SENSOR_NAME_TO_ID.items()}
