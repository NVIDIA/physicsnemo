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
"""Tests for datetime conversion utilities."""

import datetime

import numpy as np
import pandas as pd
import pytest

cftime = pytest.importorskip("cftime")

from physicsnemo.experimental.datapipes.healda.time_utils import (  # noqa: E402
    as_cftime,
    as_numpy,
    as_pydatetime,
    as_timestamp,
)


def test_as_numpy_from_pandas_index():
    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    result = as_numpy(idx)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.datetime64)
    assert len(result) == 3


def test_as_numpy_from_timestamp():
    ts = pd.Timestamp("2020-06-15T12:00:00")
    result = as_numpy(ts)
    assert result.shape == (1,)


def test_as_numpy_from_cftime():
    t = cftime.DatetimeGregorian(2022, 3, 1, 6, 0, 0)
    result = as_numpy(t)
    assert result.shape == (1,)


def test_as_cftime_roundtrip():
    ts = pd.Timestamp("2023-07-04T18:30:00")
    cf = as_cftime(ts)
    assert isinstance(cf, cftime.DatetimeGregorian)
    assert cf.year == 2023
    assert cf.month == 7
    assert cf.day == 4
    assert cf.hour == 18
    assert cf.minute == 30


def test_as_pydatetime_from_cftime():
    cf = cftime.DatetimeGregorian(2021, 12, 25, 0, 0, 0)
    result = as_pydatetime(cf)
    assert isinstance(result, datetime.datetime)
    assert result.tzinfo is not None  # UTC


def test_as_timestamp():
    idx = pd.date_range("2020-01-01", periods=1, freq="h")
    result = as_timestamp(idx)
    assert result.dtype == int
    assert result[0] == 1577836800  # 2020-01-01T00:00:00 UTC
