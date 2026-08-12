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
    zenith_azimuth_angles,
    zenith_azimuth_angles_from_timestamp,
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
    "time, lon, lat, expected_cos_zenith",
    (
        [datetime(2020, 3, 21, 12, 0, 0), 0.0, 0.0, 0.9994836252135212],
        [datetime(2020, 3, 21, 18, 0, 0), -90.0, 0.0, 0.9994760971063111],
        [datetime(2020, 3, 21, 18, 0, 0), 270.0, 0.0, 0.99947609879941],
        [datetime(2020, 7, 6, 12, 0, 0), -90.0, 0.0, -0.019703903874316815],
        [datetime(2020, 7, 6, 9, 0, 0), 40.0, 40.0, 0.9501802266240413],
    ),
)
def test_zenith_azimuth_angles(time, lon, lat, expected_cos_zenith):
    time = time.replace(tzinfo=UTC)

    sin_zenith, cos_zenith, sin_azimuth, cos_azimuth = zenith_azimuth_angles(
        time, lon, lat
    )

    # cos(zenith) must agree with the dedicated cos_zenith_angle implementation,
    # since both are derived from the same underlying formula.
    assert cos_zenith == pytest.approx(expected_cos_zenith, abs=1e-10)
    assert cos_zenith == pytest.approx(cos_zenith_angle(time, lon, lat), abs=1e-10)

    # sin(zenith) is non-negative by definition. sin(zenith)/cos(zenith) satisfy
    # the Pythagorean identity to float64 round-trip precision since sin_zenith
    # is derived directly from cos_zenith via sqrt(1 - cos_zenith**2). The
    # azimuth identity is looser: sin_azimuth/cos_azimuth are independently
    # computed formulas, and lon/lat are rounded to float32 internally
    # (see _deg2rad(..., dtype=dtype) with the module-level float32 dtype), so
    # that ~1e-7 relative input error propagates independently into each term.
    assert sin_zenith >= 0
    assert sin_zenith**2 + cos_zenith**2 == pytest.approx(1.0, abs=1e-10)
    assert sin_azimuth**2 + cos_azimuth**2 == pytest.approx(1.0, abs=1e-5)

    # the datetime and timestamp entry points must agree.
    timestamp = time.timestamp()
    out_from_timestamp = zenith_azimuth_angles_from_timestamp(timestamp, lon, lat)
    for value, value_from_timestamp in zip(
        (sin_zenith, cos_zenith, sin_azimuth, cos_azimuth), out_from_timestamp
    ):
        assert value == pytest.approx(value_from_timestamp, abs=1e-10)


@pytest.mark.parametrize(
    "time, lon, lat",
    (
        [datetime(2020, 3, 21, 12, 0, 0), 0.0, 0.0],
        [datetime(2020, 3, 21, 18, 0, 0), -90.0, 0.0],
        [datetime(2020, 3, 21, 18, 0, 0), 270.0, 0.0],
        [datetime(2020, 7, 6, 12, 0, 0), -90.0, 0.0],
        [datetime(2020, 7, 6, 9, 0, 0), 40.0, 40.0],
        [datetime(2020, 7, 6, 12, 0, 0), 0.0, 90.0],
    ),
)
def test_zenith_azimuth_angles_cos_zenith(time, lon, lat):
    """cos_zenith_angle does not clamp its output to [-1, 1], unlike the
    cos_zenith component returned by zenith_azimuth_angles (via
    _star_zenith_azimuth's explicit clamp), so cos_zenith_angle's output is
    clamped here for a fair, apples-to-apples comparison. cos(zenith) has no
    cos(lat) division, so the lat=90 pole case is safe to include here even
    though it is excluded elsewhere for the azimuth-specific checks.
    """
    time = time.replace(tzinfo=UTC)

    _, cos_zenith, _, _ = zenith_azimuth_angles(time, lon, lat)
    expected_cos_zenith = np.clip(cos_zenith_angle(time, lon, lat), -1.0, 1.0)
    assert cos_zenith == pytest.approx(expected_cos_zenith, abs=1e-10)


def test_zenith_azimuth_angles_reference_values():
    """Regression test against reference values verified by direct computation."""
    model_time = datetime(2002, 1, 1, 12, 0, 0, tzinfo=UTC)
    expected = (
        0.894125037441149,
        -0.4478173929414352,
        -0.8995401835123976,
        -0.43683802289462387,
    )

    out = zenith_azimuth_angles(model_time, lat=360, lon=120)
    for value, expected_value in zip(out, expected):
        assert value == pytest.approx(expected_value, abs=1e-9)

    out_from_timestamp = zenith_azimuth_angles_from_timestamp(
        model_time.timestamp(), lat=360, lon=120
    )
    for value, expected_value in zip(out_from_timestamp, expected):
        assert value == pytest.approx(expected_value, abs=1e-9)


def test_zenith_azimuth_angles_array():
    timestamp = np.array([0, 1, 2])[:, None, None]
    lat = np.array([0.0, 0.0])[None, :, None]
    lon = np.array([0.0])[None, None, :]
    out = zenith_azimuth_angles_from_timestamp(timestamp, lon, lat)
    assert len(out) == 4
    for component in out:
        assert component.shape == (3, 2, 1)


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

_PARITY_LONS = np.linspace(-180.0, 180.0, 360, dtype=np.float32, endpoint=False)
_PARITY_LATS = np.linspace(-90.0, 90.0, 181, dtype=np.float32)


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
@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
def test_cos_zenith_angle_numpy_torch_parity(timestamp, device, torch_dtype):
    """numpy and torch paths must return equal float32 and float64 results."""
    (lat_np, lon_np) = np.meshgrid(
        _PARITY_LATS, _PARITY_LONS, indexing="ij"
    )  # (181, 360)

    out_np = cos_zenith_angle_from_timestamp(timestamp, lon_np, lat_np)

    lon_torch = torch.as_tensor(lon_np, dtype=torch_dtype, device=device)
    lat_torch = torch.as_tensor(lat_np, dtype=torch_dtype, device=device)
    out_torch = cos_zenith_angle_from_timestamp(timestamp, lon_torch, lat_torch)

    # The torch path returns torch_dtype; the numpy path returns float64 (numpy
    # auto-promotes float32 arrays when multiplied with float64 scalars).
    # With torch.float32, differences are bounded by float32 rounding of the float64 result.
    atol = 1e-5 if torch_dtype == torch.float32 else 1e-7
    np.testing.assert_allclose(out_torch.cpu().numpy(), out_np, atol=atol)


# Exact poles (lat == +/-90) are excluded here to avoid the
# ill-defined azimuth near the poles
_PARITY_LATS_NO_POLES = _PARITY_LATS[1:-1]


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
@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
def test_zenith_azimuth_angles_numpy_torch_parity(timestamp, device, torch_dtype):
    """numpy and torch paths must return equal float32 and float64 results."""
    (lat_np, lon_np) = np.meshgrid(
        _PARITY_LATS_NO_POLES, _PARITY_LONS, indexing="ij"
    )  # (179, 360)

    sin_zenith_np, cos_zenith_np, sin_azimuth_np, cos_azimuth_np = (
        zenith_azimuth_angles_from_timestamp(timestamp, lon_np, lat_np)
    )

    lon_torch = torch.as_tensor(lon_np, dtype=torch_dtype, device=device)
    lat_torch = torch.as_tensor(lat_np, dtype=torch_dtype, device=device)
    sin_zenith_t, cos_zenith_t, sin_azimuth_t, cos_azimuth_t = (
        zenith_azimuth_angles_from_timestamp(timestamp, lon_torch, lat_torch)
    )

    # The torch path returns torch_dtype; the numpy path returns float64 (numpy
    # auto-promotes float32 arrays when multiplied with float64 scalars).
    # With torch.float32, differences are bounded by float32 rounding of the float64 result.
    atol = 1e-5 if torch_dtype == torch.float32 else 3e-7
    for np_component, torch_component in (
        (sin_zenith_np, sin_zenith_t),
        (cos_zenith_np, cos_zenith_t),
        (sin_azimuth_np, sin_azimuth_t),
        (cos_azimuth_np, cos_azimuth_t),
    ):
        np.testing.assert_allclose(
            torch_component.cpu().numpy(),
            np_component,
            atol=atol,
            rtol=1e-3,  # relatively permissive rtol needed near the sub/anti-solar points
        )


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
@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float64])
def test_toa_numpy_torch_parity(timestamp, device, torch_dtype):
    """numpy and torch paths must return equal results for toa_incident_solar_radiation_accumulated."""
    (lat_np, lon_np) = np.meshgrid(
        _PARITY_LATS, _PARITY_LONS, indexing="ij"
    )  # (181, 360)

    out_np = toa_incident_solar_radiation_accumulated(timestamp, lat_np, lon_np)

    lat_torch = torch.as_tensor(lat_np, dtype=torch_dtype, device=device)
    lon_torch = torch.as_tensor(lon_np, dtype=torch_dtype, device=device)
    out_torch = toa_incident_solar_radiation_accumulated(
        timestamp, lat_torch, lon_torch
    )

    # The toa integration runs in float32 for the torch path (lat/lon are float32
    # and numpy scalars are converted to Python floats by PyTorch, preserving
    # float32).  The numpy path promotes to float64 via numpy scalar arithmetic.
    # atol covers near-zero terminator cells where rtol alone is too strict.
    atol = 20.0 if torch_dtype == torch.float32 else 1.0
    np.testing.assert_allclose(out_torch.cpu().numpy(), out_np, rtol=1e-5, atol=atol)
