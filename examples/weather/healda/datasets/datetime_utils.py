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
import datetime

import cftime
import numpy as np
import pandas as pd


def as_pydatetime(time) -> datetime.datetime:
    if isinstance(time, cftime.datetime):
        # very important to set the timezone to UTC for example when using timestamps
        return datetime.datetime(*cftime.to_tuple(time), tzinfo=datetime.timezone.utc)
    elif isinstance(time, datetime.datetime):
        return time
    else:
        raise NotImplementedError(type(time))


def as_numpy(time) -> np.ndarray:
    # Standardize time to np.ndarray of np.datetime64
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
    """return time as an int unix timestamp"""
    return as_numpy(time).astype("datetime64[s]").astype(int)


def second_of_day(time):
    begin_of_day = time.replace(hour=0, second=0, minute=0)
    return (time - begin_of_day).total_seconds()


def as_cftime(timestamp):
    return cftime.DatetimeGregorian(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )
