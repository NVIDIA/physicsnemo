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
"""Datetime conversion utilities for HealDA data loading."""

import datetime

import cftime
import numpy as np
import pandas as pd


def as_pydatetime(time) -> datetime.datetime:
    """Convert a cftime or stdlib datetime to a timezone-aware Python datetime."""
    if isinstance(time, cftime.datetime):
        return datetime.datetime(*cftime.to_tuple(time), tzinfo=datetime.timezone.utc)
    elif isinstance(time, datetime.datetime):
        return time
    else:
        raise NotImplementedError(type(time))


def as_numpy(time) -> np.ndarray:
    """Standardize time input to ``np.ndarray`` of ``np.datetime64``."""
    if hasattr(time, "values"):  # Handle pandas Index
        time = time.values
    elif isinstance(time, (pd.Timestamp, datetime.datetime)):
        time = np.array([np.datetime64(time)])
    elif isinstance(time, cftime.datetime):
        return as_numpy(as_pydatetime(time))
    elif isinstance(time, np.datetime64):
        time = np.array([time])
    else:
        time = np.array([np.datetime64(t) for t in time])
    return time


def as_timestamp(time) -> np.ndarray:
    """Return *time* as an integer Unix timestamp (seconds since epoch)."""
    return as_numpy(time).astype("datetime64[s]").astype(int)


def second_of_day(time):
    """Return seconds elapsed since the start of the day for *time*."""
    begin_of_day = time.replace(hour=0, second=0, minute=0)
    return (time - begin_of_day).total_seconds()


def as_cftime(timestamp) -> cftime.DatetimeGregorian:
    """Convert a pandas Timestamp (or similar) to ``cftime.DatetimeGregorian``."""
    return cftime.DatetimeGregorian(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )


# ---------------------------------------------------------------------------
# cftime-based time encodings (used by transforms)
# ---------------------------------------------------------------------------


def cftime_to_timestamp(time: cftime.datetime) -> float:
    """Convert a cftime datetime to a Unix timestamp (seconds since epoch)."""
    return datetime.datetime(
        *cftime.to_tuple(time), tzinfo=datetime.timezone.utc
    ).timestamp()


def compute_second_of_day(time: cftime.datetime) -> float:
    """Return seconds elapsed since midnight for *time*."""
    day_start = time.replace(hour=0, minute=0, second=0)
    return (time - day_start) / datetime.timedelta(seconds=1)


def compute_day_of_year(time: cftime.datetime) -> float:
    """Return fractional day-of-year for *time*."""
    day_start = time.replace(hour=0, minute=0, second=0)
    year_start = day_start.replace(month=1, day=1)
    return (time - year_start) / datetime.timedelta(seconds=86400)


def compute_timestamp(time: cftime.datetime) -> int:
    """Return integer Unix timestamp for *time*."""
    return int(cftime_to_timestamp(time))
