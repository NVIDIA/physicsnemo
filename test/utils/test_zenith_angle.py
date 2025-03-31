# ignore_header_test

# climt/LICENSE
# @mcgibbon
# BSD License
# Copyright (c) 2016, Rodrigo Caballero
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED

from datetime import datetime

import numpy as np
import pytest
from pytz import utc

from physicsnemo.utils.zenith_angle import (
    _datetime_to_julian_century,
    _timestamp_to_julian_century,
    cos_zenith_angle,
    cos_zenith_angle_from_timestamp,
    toa_incident_solar_radiation_accumulated,
)


@pytest.mark.parametrize(
    "time, lon, lat, expected",
    (
        [datetime(2020, 3, 21, 12, 0, 0), 0.0, 0.0, 0.9994836252135212],
        [datetime(2020, 3, 21, 18, 0, 0), -90.0, 0.0, 0.9994760971063111],
        [datetime(2020, 3, 21, 18, 0, 0), 270.0, 0.0, 0.99947609879941],
        [datetime(2020, 7, 6, 12, 0, 0), -90.0, 0.0, -0.019703903874316815],
        [datetime(2020, 7, 6, 9, 0, 0), 40.0, 40.0, 0.9501802266240413],
        [datetime(2020, 7, 6, 12, 0, 0), 0.0, 90.0, 0.3843918031907148],
    ),
)
def test_zenith_angle(time, lon, lat, expected):
    time = time.replace(tzinfo=utc)
    assert cos_zenith_angle(time, lon, lat) == pytest.approx(expected, abs=1e-10)
    timestamp = time.timestamp()
    assert cos_zenith_angle_from_timestamp(timestamp, lon, lat) == pytest.approx(
        expected, abs=1e-10
    )


def test_zenith_angle_array():
    timestamp = np.array([0, 1, 2])[:, None, None]
    lat = np.array([0.0, 0.0])[None, :, None]
    lon = np.array([0.0])[None, None, :]
    out = cos_zenith_angle_from_timestamp(timestamp, lon, lat)
    assert out.shape == (3, 2, 1)


@pytest.mark.parametrize(
    "t",
    [
        datetime(2020, 7, 6, 9, 0, 0),
        datetime(2000, 1, 1, 12, 0, 0),
        datetime(2000, 7, 1, 12, 0, 0),
        datetime(2000, 7, 1, 12, 0, 0, tzinfo=utc),
    ],
)
def test_timestamp_to_julian_centuries(t):
    a = _datetime_to_julian_century(t)
    b = _timestamp_to_julian_century(t.replace(tzinfo=utc).timestamp())
    assert a == b


def test_toa():
    t = datetime(2000, 7, 1, 12, 0, 0, tzinfo=utc).timestamp()
    lat, lon = 0.0, 0.0
    ans = toa_incident_solar_radiation_accumulated(t, lat, lon)
    assert ans >= 0
