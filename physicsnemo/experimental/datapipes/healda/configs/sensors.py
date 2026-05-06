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
"""Sensor metadata and configuration for observation loading.

Defines sensor configurations, platform mappings, and channel offsets used
by ``UFSUnifiedLoader`` and the observation transform pipeline.
"""

import os
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Recipe-side directory containing per-sensor `*_normalizations.csv` files
# (and the ERA5 stats CSV consumed by `loaders.era5`). When unset, sensor
# stats fall back to zero-mean / unit-std so the package remains usable
# without recipe-specific data.
STATS_DIR_ENV = "HEALDA_STATS_DIR"


@dataclass
class SensorConfig:
    """Sensor metadata for data loading and normalization."""

    name: str
    platforms: list[str]
    channels: int
    nc_file_template: str
    means: np.ndarray = field(init=False)
    stds: np.ndarray = field(init=False)
    min_valid: float = 0.0
    max_valid: float = 400.0
    sensor_type: str = "microwave"
    raw_to_local: np.ndarray = field(init=False)

    def __post_init__(self):
        stats_dir = os.environ.get(STATS_DIR_ENV)
        norm_file = (
            pathlib.Path(stats_dir) / "normalizations" / f"{self.name}_normalizations.csv"
            if stats_dir
            else None
        )

        if norm_file is not None and norm_file.exists():
            df = pd.read_csv(norm_file)
            channel_col = "Raw_Channel_ID"
            df = df[df["Platform_ID"] == -1].sort_values(channel_col)

            self.means = df["obs_mean"].to_numpy()
            self.stds = df["obs_std"].to_numpy()

            raw_ids = df[channel_col].to_numpy()
            max_raw = raw_ids.max()
            lookup_table = np.full(max_raw + 1, 0, dtype=int)
            for local_idx, raw in enumerate(raw_ids, start=1):
                lookup_table[raw] = local_idx
            self.raw_to_local = lookup_table
        else:
            self.means = np.zeros(self.channels, dtype=float)
            self.stds = np.ones(self.channels, dtype=float)
            self.raw_to_local = None


def _build_identity_lut(raw_ids: np.ndarray) -> np.ndarray:
    """Build a 1-indexed identity LUT from observed raw channel IDs."""
    raw_ids = np.asarray(raw_ids).ravel()
    unique = np.unique(raw_ids)
    lut = np.zeros(int(unique.max()) + 1, dtype=int)
    for local_idx, raw in enumerate(unique, start=1):
        lut[raw] = local_idx
    return lut


def get_global_channel_id(sensor, raw_channel_ids):
    """Map per-sensor raw channel IDs to unified global IDs."""
    cfg = SENSOR_CONFIGS[sensor]
    if cfg.raw_to_local is None:
        cfg.raw_to_local = _build_identity_lut(raw_channel_ids)
    raw_to_local = cfg.raw_to_local
    channel_offset = SENSOR_OFFSET[sensor]
    raw_channel_ids = np.asarray(raw_channel_ids)
    safe_ids = np.minimum(raw_channel_ids, len(raw_to_local) - 1)
    local_channels = raw_to_local[safe_ids] - 1
    return (local_channels + channel_offset).astype(np.uint16)


# ---------------------------------------------------------------------------
# Sensor registry
# ---------------------------------------------------------------------------

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
        platforms=[
            "metop-a", "metop-b", "metop-c", "n15", "n16", "n17", "n18", "n19",
        ],
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
        platforms=[],
        channels=8,
        nc_file_template="conv_{platform}_ges.{date}_control.nc4",
        sensor_type="conv",
    ),
    "iasi-pca": SensorConfig(
        name="iasi-pca",
        platforms=["metop-a", "metop-b", "metop-c"],
        channels=32,
        nc_file_template="",
        min_valid=float("-inf"),
        max_valid=float("inf"),
        sensor_type="infrared",
    ),
    "cris-fsr-pca": SensorConfig(
        name="cris-fsr-pca",
        platforms=["npp", "n20"],
        channels=32,
        nc_file_template="",
        min_valid=float("-inf"),
        max_valid=float("inf"),
        sensor_type="infrared",
    ),
    "airs": SensorConfig(
        name="airs",
        platforms=["aqua"],
        channels=117,
        nc_file_template="diag_airs_{platform}_ges.{date}_control.nc4",
        min_valid=150.0,
        max_valid=350.0,
        sensor_type="infrared",
    ),
    "airs-pca": SensorConfig(
        name="airs-pca",
        platforms=["aqua"],
        channels=32,
        nc_file_template="",
        min_valid=float("-inf"),
        max_valid=float("inf"),
        sensor_type="infrared",
    ),
}


# ---------------------------------------------------------------------------
# QC filtering limits for conventional observations
# ---------------------------------------------------------------------------


class QCLimits:
    HEIGHT_MIN = 0
    HEIGHT_MAX = 60000
    PRESSURE_MIN_GPS = 0.5
    PRESSURE_MIN_DEFAULT = 200
    PRESSURE_MAX = 1100


# ---------------------------------------------------------------------------
# Conventional channel definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvChannel:
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
    channel_map = {}
    for i, channel in enumerate(CONV_CHANNELS, start=1):
        if channel.platform not in channel_map:
            channel_map[channel.platform] = i
    return channel_map


CONV_CHANNEL_MAP = _build_conv_channel_map()


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Platform and sensor ID mappings
# ---------------------------------------------------------------------------

PLATFORM_NAME_TO_ID = {
    "aqua": 0, "aura": 1, "f10": 2, "f11": 3, "f13": 4, "f14": 5,
    "f15": 6, "g08": 7, "g10": 8, "g11": 9, "g12": 10, "m08": 11,
    "m09": 12, "m10": 13, "metop-a": 14, "metop-b": 15, "metop-c": 16,
    "n11": 17, "n12": 18, "n14": 19, "n15": 20, "n16": 21, "n17": 22,
    "n18": 23, "n19": 24, "n20": 25, "npp": 26, "gps": 27, "ps": 28,
    "q": 29, "t": 30, "uv": 31,
}

PLATFORM_ID_TO_NAME = {v: k for k, v in PLATFORM_NAME_TO_ID.items()}
NPLATFORMS = _next_power_of_two(max(len(PLATFORM_NAME_TO_ID), 64))  # 64

# Global channel offsets (contiguous across sensors)
SENSOR_OFFSET = {}
offset = 0
for name, cfg in SENSOR_CONFIGS.items():
    SENSOR_OFFSET[name] = offset
    offset += cfg.channels
NCHANNEL = _next_power_of_two(max(offset, 1024))  # 1024

CONV_GPS_GLOBAL_IDS = [SENSOR_OFFSET["conv"] + i for i in CONV_GPS_CHANNELS]

SENSOR_NAME_TO_ID = {name: idx for idx, name in enumerate(SENSOR_CONFIGS.keys())}
SENSOR_ID_TO_NAME = {idx: name for name, idx in SENSOR_NAME_TO_ID.items()}
