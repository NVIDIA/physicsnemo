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

from datetime import UTC, datetime

import numpy as np
import pytest
import torch

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
    time = time.replace(tzinfo=UTC)
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
        datetime(2000, 7, 1, 12, 0, 0, tzinfo=UTC),
    ],
)
def test_timestamp_to_julian_centuries(t):
    a = _datetime_to_julian_century(t)
    b = _timestamp_to_julian_century(t.replace(tzinfo=UTC).timestamp())
    assert a == b


def test_toa():
    t = datetime(2000, 7, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    lat, lon = 0.0, 0.0
    ans = toa_incident_solar_radiation_accumulated(t, lat, lon)
    assert ans >= 0


# Timestamps used for numpy/torch parity tests.  The cases deliberately include
# daytime (sun above horizon), nighttime (sun below horizon, result == 0) and a
# high-latitude summer scenario so that both branches of _integrate_abs_cosz are
# exercised.
_PARITY_CASES = [
    datetime(2020, 3, 21, 12, 0, 0, tzinfo=UTC).timestamp(),  # equinox noon
    datetime(2020, 7, 6, 9, 0, 0, tzinfo=UTC).timestamp(),  # summer morning
    datetime(2020, 1, 15, 3, 0, 0, tzinfo=UTC).timestamp(),  # nighttime
]

_PARITY_LONS = np.array([-90.0, 0.0, 40.0, 120.0], dtype=np.float32)
_PARITY_LATS = np.array([-60.0, 0.0, 40.0, 80.0], dtype=np.float32)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("timestamp", _PARITY_CASES)
def test_cos_zenith_angle_numpy_torch_parity(timestamp, device):
    """numpy and torch paths must return equal float32 results."""
    lon_np = _PARITY_LONS[:, None]  # (4, 1)
    lat_np = _PARITY_LATS[None, :]  # (1, 4)

    out_np = cos_zenith_angle_from_timestamp(timestamp, lon_np, lat_np)

    lon_torch = torch.as_tensor(lon_np, dtype=torch.float32, device=device)
    lat_torch = torch.as_tensor(lat_np, dtype=torch.float32, device=device)
    out_torch = cos_zenith_angle_from_timestamp(timestamp, lon_torch, lat_torch)

    # The torch path returns float32; the numpy path returns float64 (numpy
    # auto-promotes float32 arrays when multiplied with float64 scalars).
    # Differences are bounded by float32 rounding of the float64 result.
    np.testing.assert_allclose(out_torch.cpu().numpy(), out_np, rtol=1e-5)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("timestamp", _PARITY_CASES)
def test_toa_numpy_torch_parity(timestamp, device):
    """numpy and torch paths must return equal results for toa_incident_solar_radiation_accumulated."""
    lat_np = _PARITY_LATS[:, None]  # (4, 1)
    lon_np = _PARITY_LONS[None, :]  # (1, 4)

    out_np = toa_incident_solar_radiation_accumulated(timestamp, lat_np, lon_np)

    lat_torch = torch.as_tensor(lat_np, dtype=torch.float32, device=device)
    lon_torch = torch.as_tensor(lon_np, dtype=torch.float32, device=device)
    out_torch = toa_incident_solar_radiation_accumulated(
        timestamp, lat_torch, lon_torch
    )

    # The toa integration runs in float32 for the torch path (lat/lon are float32
    # and numpy scalars are converted to Python floats by PyTorch, preserving
    # float32).  The numpy path promotes to float64 via numpy scalar arithmetic.
    # atol covers near-zero terminator cells where rtol alone is too strict.
    np.testing.assert_allclose(out_torch.cpu().numpy(), out_np, rtol=1e-5, atol=1.0)
