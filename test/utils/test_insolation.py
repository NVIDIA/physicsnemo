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

"""Regression tests pinning the day-of-year semantics of ``insolation``.

The values below are the ones the pandas-based implementation produced; the numpy
implementation must reproduce them exactly, including the local-calendar-year
behavior for timezone-aware input.
"""

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from physicsnemo.utils.insolation import _days_since_year_start, insolation

EST = timezone(timedelta(hours=-5))


def test_naive_inputs_agree_across_types():
    """datetime objects, datetime64 arrays and ISO strings give identical results."""
    as_datetime = [datetime(2020, 2, 29, 13, 30), datetime(1999, 12, 31, 23, 59, 59)]
    as_np = np.array(
        ["2020-02-29T13:30", "1999-12-31T23:59:59"], dtype="datetime64[ns]"
    )
    as_str = ["2020-02-29T13:30", "1999-12-31T23:59:59"]
    expected = np.array([59 + 13.5 / 24, 364 + (23 * 3600 + 59 * 60 + 59) / 86400])
    for dates in (as_datetime, as_np, as_str):
        np.testing.assert_array_equal(_days_since_year_start(dates), expected)


def test_coarse_units_and_wide_year_range():
    """Month-resolution arrays and years outside the nanosecond range are accepted."""
    months = np.array(["2020-02", "2021-12"], dtype="datetime64[M]")
    np.testing.assert_array_equal(_days_since_year_start(months), [31.0, 334.0])
    wide = [datetime(1600, 3, 1, 12), datetime(3000, 7, 4)]
    np.testing.assert_array_equal(_days_since_year_start(wide), [60.5, 184.0])


def test_timezone_aware_uses_local_calendar_year():
    """Dec 31 21:00 EST is 02:00 UTC on Jan 1, but the year is the local one (day 366)."""
    days = _days_since_year_start([datetime(2020, 12, 31, 21, 0, tzinfo=EST)])
    np.testing.assert_array_equal(days, [366 + 2 / 24])
    # Same instant expressed in UTC is day 0 of the new year.
    days_utc = _days_since_year_start([datetime(2021, 1, 1, 2, 0, tzinfo=UTC)])
    np.testing.assert_array_equal(days_utc, [2 / 24])
    # ISO strings with an offset follow the same rule.
    np.testing.assert_array_equal(
        _days_since_year_start(["2020-12-31T21:00-05:00"]), [366 + 2 / 24]
    )


def test_pandas_timestamps_match_datetime():
    """pandas Timestamps and a tz-aware DatetimeIndex behave like datetime objects."""
    pd = pytest.importorskip("pandas")
    ts = [pd.Timestamp("2020-12-31 21:00", tz="US/Eastern")]
    np.testing.assert_array_equal(_days_since_year_start(ts), [366 + 2 / 24])
    idx = pd.date_range("2020-12-30", periods=3, freq="12h")
    np.testing.assert_array_equal(_days_since_year_start(idx), [364.0, 364.5, 365.0])


def test_nat_is_rejected():
    """NaT entries raise instead of propagating silently."""
    with pytest.raises(ValueError, match="NaT"):
        _days_since_year_start(np.array(["NaT", "2020-06-01"], dtype="datetime64[D]"))


def test_insolation_shapes_and_daily_max():
    """End-to-end call on 1-D and 2-D grids, including the daily-max path."""
    dates = np.array(["2020-06-21T12:00", "2020-12-21T00:00"], dtype="datetime64[s]")
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(0, 360, 5, endpoint=False)
    out = insolation(dates, lat, lon)
    assert out.shape == (2, 5)
    out2d = insolation(dates, lat, lon, enforce_2d=True, daily=True)
    assert out2d.shape == (2, 5, 5)
    assert np.all(out2d >= 0)
