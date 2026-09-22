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

``_days_since_year_start`` replaced a pandas expression,
``(np.array(dates, "datetime64") - Timestamp(Timestamp(d).year, 1, 1)) / 1 day``,
and must reproduce it bit-for-bit. Two oracles enforce that:

* a pure-Python reference: the instant in UTC (numpy's conversion) minus
  January 1st of the *local* calendar year (pandas' ``Timestamp.year``);
* when pandas is installed, the original expression itself.

Both run over a sweep of dates (every month, leap days, year boundaries,
1600-3000), timezones (whole, half and 45-minute offsets, both extremes,
named zones), every ``datetime64`` unit, and every ISO offset spelling.
"""

import datetime as dt
import itertools
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from physicsnemo.utils.insolation import _days_since_year_start, insolation

# ---------------------------------------------------------------- fixtures --

UTC = dt.timezone.utc
OFFSETS = {
    "UTC": UTC,
    "-12:00": dt.timezone(dt.timedelta(hours=-12)),
    "-05:00": dt.timezone(dt.timedelta(hours=-5)),
    "-03:30": dt.timezone(dt.timedelta(hours=-3, minutes=-30)),
    "+01:00": dt.timezone(dt.timedelta(hours=1)),
    "+05:30": dt.timezone(dt.timedelta(hours=5, minutes=30)),
    "+05:45": dt.timezone(dt.timedelta(hours=5, minutes=45)),
    "+14:00": dt.timezone(dt.timedelta(hours=14)),
    "US/Eastern": ZoneInfo("America/New_York"),
    "Europe/Berlin": ZoneInfo("Europe/Berlin"),
    "Asia/Kolkata": ZoneInfo("Asia/Kolkata"),
    "Pacific/Kiritimati": ZoneInfo("Pacific/Kiritimati"),
    "Pacific/Apia": ZoneInfo("Pacific/Apia"),
}

# Naive wall-clock times spanning the year: one per month, leap days, and the
# instants on either side of every year boundary where a UTC offset can move
# the calendar year.
NAIVE_DATES = (
    [dt.datetime(2020, m, 15, 6 * (m % 4), 17, 3) for m in range(1, 13)]
    + [
        dt.datetime(2020, 2, 29, 23, 59, 59, 999999),  # leap day, last microsecond
        dt.datetime(2024, 2, 29, 0, 0, 0, 1),
        dt.datetime(1900, 3, 1),  # 1900 is not a leap year
        dt.datetime(2000, 2, 29, 12),  # 2000 is
        dt.datetime(1600, 3, 1, 12),  # outside pandas' ns range
        dt.datetime(1950, 6, 15, 12),  # pre-1970 (negative epoch)
        dt.datetime(1969, 12, 31, 23, 59, 59),
        dt.datetime(1970, 1, 1, 0, 0, 0),
        dt.datetime(2262, 4, 11, 12),  # edge of the ns range
        dt.datetime(3000, 7, 4),
    ]
    + [
        dt.datetime(y, 12, 31, h, mi)
        for y in (1999, 2019, 2020, 2023)
        for h, mi in ((9, 0), (12, 0), (20, 30), (23, 59))
    ]
    + [
        dt.datetime(y, 1, 1, h, mi)
        for y in (2000, 2020, 2021, 2024)
        for h, mi in ((0, 0), (0, 1), (3, 30), (14, 59))
    ]
)


def reference_days(d):
    """Pure-Python oracle for one date: UTC instant minus local-year start."""
    if d.tzinfo is not None:
        instant = d.astimezone(UTC).replace(tzinfo=None)  # numpy stores UTC
    else:
        instant = d
    start = dt.datetime(d.year, 1, 1)  # pandas: Timestamp(d).year is local
    # Exact integer microseconds divided once, which is what numpy does with
    # datetime64[us] arithmetic; timedelta.total_seconds() would round differently.
    delta = instant - start
    micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return micros / 86_400_000_000


# ------------------------------------------------------------- naive dates --


@pytest.mark.parametrize("date", NAIVE_DATES, ids=str)
def test_naive_datetime_matches_reference(date):
    """Every naive date agrees with the reference to the last bit."""
    np.testing.assert_array_equal(
        _days_since_year_start([date]), [reference_days(date)]
    )


def test_naive_input_types_are_interchangeable():
    """datetime objects, datetime64 arrays, datetime64 scalars and ISO strings agree."""
    dates = [d for d in NAIVE_DATES if 1678 < d.year < 2262]  # ns-representable
    expected = _days_since_year_start(dates)
    as_ns = np.array(dates, dtype="datetime64[ns]")
    as_scalars = [np.datetime64(d) for d in dates]
    as_str = [d.isoformat() for d in dates]
    for variant in (as_ns, as_scalars, as_str):
        np.testing.assert_array_equal(_days_since_year_start(variant), expected)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "m", "h", "D", "M", "Y"])
def test_every_datetime64_unit(unit):
    """Each numpy unit, including the non-linear month and year units, is accepted."""
    dates = np.array(
        ["2020-02-29T13:30:15.123456789", "2021-12-31T23:59:59.999999999"],
        dtype="datetime64[ns]",
    ).astype(f"datetime64[{unit}]")
    # The truth is the same values re-expressed as datetime objects at that unit's
    # resolution: convert back through microseconds and let the reference do the rest.
    truth = [reference_days(d) for d in dates.astype("datetime64[us]").astype(object)]
    got = _days_since_year_start(dates)
    if unit == "ns":
        # microsecond truth cannot see nanoseconds; assert against numpy directly
        ns = (
            dates - dates.astype("datetime64[Y]").astype("datetime64[us]")
        ) / np.timedelta64(1, "D")
        np.testing.assert_array_equal(got, ns)
    else:
        np.testing.assert_array_equal(got, truth)


# ---------------------------------------------------------- timezone-aware --


@pytest.mark.parametrize(
    ("date", "zone"),
    list(itertools.product(NAIVE_DATES, OFFSETS)),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_timezone_aware_uses_local_year_and_utc_instant(date, zone):
    """For every date x timezone, the year is local and the instant is UTC."""
    aware = date.replace(tzinfo=OFFSETS[zone])
    if aware.utcoffset().total_seconds() % 60:
        # Historical local-mean-time offsets carry seconds; numpy drops them when
        # converting tz-aware datetimes, and the original pandas code shared that
        # numpy conversion, so the pure-Python oracle cannot be exact here. The
        # pandas oracle below still covers these inputs bit-for-bit.
        pytest.skip("sub-minute UTC offset")
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning)  # numpy deprecates tz-aware datetime conversion
        got = _days_since_year_start([aware])
    np.testing.assert_array_equal(got, [reference_days(aware)])


def test_year_boundary_crossing_examples():
    """Hand-checked cases where the local and UTC calendar years differ."""
    est = OFFSETS["-05:00"]
    # Dec 31 21:00 EST is Jan 1 02:00 UTC: local year 2020 -> day 366 + 2h (leap year)
    np.testing.assert_array_equal(
        _days_since_year_start([dt.datetime(2020, 12, 31, 21, 0, tzinfo=est)]),
        [366 + 2 / 24],
    )
    # The same instant written in UTC is day 0 + 2h of 2021
    np.testing.assert_array_equal(
        _days_since_year_start([dt.datetime(2021, 1, 1, 2, 0, tzinfo=UTC)]), [2 / 24]
    )
    # Jan 1 03:00 IST is still Dec 31 21:30 UTC: local year 2021 -> negative day
    ist = OFFSETS["+05:30"]
    np.testing.assert_array_equal(
        _days_since_year_start([dt.datetime(2021, 1, 1, 3, 0, tzinfo=ist)]),
        [-(2.5 / 24)],
    )
    # Mid-year, the offset only shifts the fraction of the day
    np.testing.assert_array_equal(
        _days_since_year_start([dt.datetime(2020, 6, 1, 12, tzinfo=est)]),
        [reference_days(dt.datetime(2020, 6, 1, 17))],
    )


@pytest.mark.parametrize(
    "spelling",
    ["-05:00", "-0500", "-05", "+05:30", "+0530", "+14:00", "Z", "+00:00"],
)
def test_iso_strings_with_offsets(spelling):
    """Every ISO offset spelling takes the year from the wall-clock digits."""
    wall = "2020-12-31T21:00:00"
    aware = dt.datetime.fromisoformat(
        wall
        + (
            "+00:00"
            if spelling == "Z"
            else spelling
            if ":" in spelling or len(spelling) == 3
            else spelling[:3] + ":" + spelling[3:]
        )
    )
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        got = _days_since_year_start([wall + spelling])
    np.testing.assert_array_equal(got, [reference_days(aware)])


def test_mixed_naive_and_aware_lists():
    """Lists may mix naive datetimes, aware datetimes, dates and strings."""
    items = [
        dt.datetime(2020, 12, 31, 21, 0, tzinfo=OFFSETS["-05:00"]),
        dt.datetime(2020, 6, 1),
        dt.date(2021, 3, 1),
        "2021-12-31T23:30Z",
    ]
    expected = [
        reference_days(items[0]),
        reference_days(items[1]),
        reference_days(dt.datetime(2021, 3, 1)),
        reference_days(dt.datetime(2021, 12, 31, 23, 30, tzinfo=UTC)),
    ]
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        got = _days_since_year_start(items)
    np.testing.assert_array_equal(got, expected)


# ------------------------------------------------------------ pandas oracle --


@pytest.fixture
def pandas_oracle():
    """The exact pre-refactor expression, for environments that have pandas."""
    pd = pytest.importorskip("pandas")

    def _oracle(dates):
        start_years = np.array(
            [pd.Timestamp(pd.Timestamp(d).year, 1, 1) for d in dates],
            dtype="datetime64",
        )
        return (np.array(dates, dtype="datetime64") - start_years) / np.timedelta64(
            1, "D"
        )

    return pd, _oracle


def test_matches_pandas_expression_exactly(pandas_oracle):
    """Bit-identical to the original pandas code over the full date x zone sweep."""
    pd, oracle = pandas_oracle
    dates = [d for d in NAIVE_DATES if 1678 < d.year < 2262]
    inputs = list(dates)
    for zone in OFFSETS.values():
        inputs += [d.replace(tzinfo=zone) for d in dates if d.year >= 1900]
    inputs += [pd.Timestamp(d) for d in dates]
    inputs += [pd.Timestamp(d, tz="US/Eastern") for d in dates if d.year >= 1900]
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        np.testing.assert_array_equal(_days_since_year_start(inputs), oracle(inputs))


def test_pandas_containers(pandas_oracle):
    """DatetimeIndex (naive and tz-aware) and Series behave like the oracle."""
    pd, oracle = pandas_oracle
    idx = pd.date_range("2020-12-30 18:00", periods=12, freq="3h")
    tz_idx = pd.date_range("2020-12-30 18:00", periods=12, freq="3h", tz="US/Eastern")
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        for container in (idx, tz_idx, pd.Series(idx), idx.to_numpy()):
            np.testing.assert_array_equal(
                _days_since_year_start(container), oracle(container)
            )


# ------------------------------------------------------------- end-to-end --


def test_nat_is_rejected():
    """NaT entries raise instead of propagating silently."""
    with pytest.raises(ValueError, match="NaT"):
        _days_since_year_start(np.array(["NaT", "2020-06-01"], dtype="datetime64[D]"))
    with pytest.raises(ValueError, match="NaT"):
        _days_since_year_start([None, dt.datetime(2020, 6, 1)])


def test_insolation_shapes_daily_and_clipping():
    """End-to-end call on 1-D and 2-D grids, including daily max and clipping."""
    dates = np.array(["2020-06-21T12:00", "2020-12-21T00:00"], dtype="datetime64[s]")
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(0, 360, 5, endpoint=False)
    out = insolation(dates, lat, lon)
    assert out.shape == (2, 5) and np.all(out >= 0)
    unclipped = insolation(dates, lat, lon, clip_zero=False)
    assert np.any(unclipped < 0)
    out2d = insolation(dates, lat, lon, enforce_2d=True, daily=True)
    assert out2d.shape == (2, 5, 5)
    # daily max is longitude-independent
    np.testing.assert_array_equal(out2d, np.repeat(out2d[..., :1], 5, axis=-1))


def test_insolation_timezone_aware_end_to_end():
    """The full model sees identical fields for the same instant in any zone."""
    lon, lat = np.meshgrid(
        np.linspace(0, 360, 9, endpoint=False), np.linspace(-60, 60, 7)
    )
    est = dt.datetime(2020, 6, 30, 21, 0, tzinfo=OFFSETS["-05:00"])
    utc = dt.datetime(2020, 7, 1, 2, 0, tzinfo=UTC)
    with np.testing.suppress_warnings() as sup:
        sup.filter(DeprecationWarning)
        a, b = insolation([est], lat, lon), insolation([utc], lat, lon)
    assert a.shape == (1, 7, 9)
    np.testing.assert_array_equal(a, b)
