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
# OF THE POSSIBILITY OF SUCH DAMAGE.


import datetime
from collections.abc import Callable
from typing import TypeVar, Union

import numpy as np
import torch

# helper type
dtype = np.float32


T = TypeVar("T", torch.Tensor, np.ndarray, float)

TIMESTAMP_2000 = datetime.datetime(2000, 1, 1, 12, 0, tzinfo=datetime.UTC).timestamp()


def cos_zenith_angle(
    time: Union[T, datetime.datetime],
    lon: T,
    lat: T,
) -> T:  # pragma: no cover
    """
    Cosine of sun-zenith angle for lon, lat at time (UTC).
    If DataArrays are provided for the lat and lon arguments, their units will
    be assumed to be in degrees, unless they have a units attribute that
    contains "rad"; in that case they will automatically be converted to having
    units of degrees.

    Parameters
    ----------
    time: datetime.datetime, time in UTC
    lon: float, np.ndarray or torch.Tensor, longitude in degrees (E/W)
    lat: float, np.ndarray or torch.Tensor, latitude in degrees (N/S)

    Returns
    --------
    float, np.ndarray or torch.Tensor, cosine of the solar zenith angle

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> angle = cos_zenith_angle(model_time, lat=360, lon=120)
    >>> bool(abs(angle - -0.447817277) < 1e-6)
    True
    """
    lon_rad = _deg2rad(lon, dtype=dtype)
    lat_rad = _deg2rad(lat, dtype=dtype)
    julian_centuries = _datetime_to_julian_century(time)
    return _star_cos_zenith(julian_centuries, lon_rad, lat_rad)


def zenith_azimuth_angles(
    time: datetime.datetime,
    lon: T,
    lat: T,
    sin_zenith_epsilon: float = 1e-6,
) -> tuple[T, T, T, T]:
    r"""
    Sine and cosine of the solar zenith and azimuth angles for ``lon``,
    ``lat`` at ``time`` (UTC).

    The azimuth angle follows the north-clockwise convention (0 degrees is
    north, 90 degrees is east, 180 degrees is south, 270 degrees is west),
    per the
    `solar azimuth angle formulas <https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas>`_.
    At exact poles (latitude +/- 90 degrees), azimuth angle is ill defined and
    sin_azimuth/cos_azimuth return arbitrary values rather than a meaningful result.
    See :func:`~physicsnemo.utils.zenith_angle.cos_zenith_angle` if only the
    cosine of the zenith angle is required, and
    :func:`~physicsnemo.utils.zenith_angle.zenith_azimuth_angles_from_timestamp`
    for the UNIX-timestamp variant of this function.

    Parameters
    ----------
    time : datetime.datetime
        Time in UTC.
    lon : float, np.ndarray or torch.Tensor
        Longitude in degrees (E/W).
    lat : float, np.ndarray or torch.Tensor
        Latitude in degrees (N/S).
    sin_zenith_epsilon : float, optional, default=1e-6
        Small minimum value applied to sin_zenith to prevent division by zero at
        sub/anti-solar points.

    Returns
    -------
    tuple of (float, np.ndarray or torch.Tensor)
        Tuple ``(sin_zenith, cos_zenith, sin_azimuth, cos_azimuth)``: the
        sine and cosine of the solar zenith angle, followed by the sine and
        cosine of the solar azimuth angle. Each element has the same type
        as the inputs.

    Examples
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> sin_zenith, cos_zenith, sin_azimuth, cos_azimuth = zenith_azimuth_angles(
    ...     model_time, lat=360, lon=120
    ... )
    >>> [float(x) for x in (sin_zenith, cos_zenith, sin_azimuth, cos_azimuth)]
    [0.894125037441149, -0.4478173929414352, -0.8995401835123976, -0.43683802289462387]
    """
    lon_rad = _deg2rad(lon, dtype=dtype)
    lat_rad = _deg2rad(lat, dtype=dtype)
    julian_centuries = _datetime_to_julian_century(time)
    return _star_zenith_azimuth(
        julian_centuries, lon_rad, lat_rad, sin_zenith_epsilon=sin_zenith_epsilon
    )


def cos_zenith_angle_from_timestamp(
    timestamp: T,
    lon: T,
    lat: T,
) -> T:
    """
    Cosine of sun-zenith angle for lon, lat at a given UNIX timestamp (UTC).

    Since the UNIX timestamp is a floating point or integer this routine can be
    compiled with jax.

    Parameters
    ----------
    timestamp: float, np.ndarray or torch.Tensor, UNIX timestamp in seconds
    lon: float, np.ndarray or torch.Tensor, longitude in degrees (E/W)
    lat: float, np.ndarray or torch.Tensor, latitude in degrees (N/S)

    Returns
    -------
    float, np.ndarray or torch.Tensor, cosine of the solar zenith angle,
    same type as the inputs

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> angle = cos_zenith_angle_from_timestamp(model_time.timestamp(), lat=360, lon=120)
    >>> bool(abs(angle - -0.447817277) < 1e-6)
    True
    """
    lon_rad = _deg2rad(lon, dtype=dtype)
    lat_rad = _deg2rad(lat, dtype=dtype)
    julian_centuries = _timestamp_to_julian_century(timestamp)
    return _star_cos_zenith(julian_centuries, lon_rad, lat_rad)


def zenith_azimuth_angles_from_timestamp(
    timestamp: T,
    lon: T,
    lat: T,
    sin_zenith_epsilon: float = 1e-6,
) -> tuple[T, T, T, T]:
    r"""
    Sine and cosine of the solar zenith and azimuth angles for ``lon``,
    ``lat`` at a given UNIX timestamp (UTC).

    Since the UNIX timestamp is a floating point or integer this routine can
    be compiled with jax. The azimuth angle follows the north-clockwise
    convention (0 degrees is north, 90 degrees is east, 180 degrees is
    south, 270 degrees is west), per the
    `solar azimuth angle formulas <https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas>`_.
    At exact poles (latitude +/- 90 degrees), azimuth angle is ill defined and
    sin_azimuth/cos_azimuth return arbitrary values rather than a meaningful result.
    See :func:`~physicsnemo.utils.zenith_angle.cos_zenith_angle_from_timestamp`
    if only the cosine of the zenith angle is required, and
    :func:`~physicsnemo.utils.zenith_angle.zenith_azimuth_angles` for the
    ``datetime``-based variant of this function.

    Parameters
    ----------
    timestamp : float, np.ndarray or torch.Tensor
        UNIX timestamp in seconds.
    lon : float, np.ndarray or torch.Tensor
        Longitude in degrees (E/W).
    lat : float, np.ndarray or torch.Tensor
        Latitude in degrees (N/S).
    sin_zenith_epsilon : float, optional, default=1e-6
        Small minimum value applied to sin_zenith to prevent division by zero at
        sub/anti-solar points.

    Returns
    -------
    tuple of (float, np.ndarray or torch.Tensor)
        Tuple ``(sin_zenith, cos_zenith, sin_azimuth, cos_azimuth)``: the
        sine and cosine of the solar zenith angle, followed by the sine and
        cosine of the solar azimuth angle. Each element has the same type
        as the inputs.

    Examples
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> sin_zenith, cos_zenith, sin_azimuth, cos_azimuth = (
    ...     zenith_azimuth_angles_from_timestamp(
    ...         model_time.timestamp(), lat=360, lon=120
    ...     )
    ... )
    >>> [float(x) for x in (sin_zenith, cos_zenith, sin_azimuth, cos_azimuth)]
    [0.894125037441149, -0.4478173929414352, -0.8995401835123976, -0.43683802289462387]
    """
    lon_rad = _deg2rad(lon, dtype=dtype)
    lat_rad = _deg2rad(lat, dtype=dtype)
    julian_centuries = _timestamp_to_julian_century(timestamp)
    return _star_zenith_azimuth(
        julian_centuries, lon_rad, lat_rad, sin_zenith_epsilon=sin_zenith_epsilon
    )


def _deg2rad(x: T, dtype: np.typing.DTypeLike | torch.dtype | None = None) -> T:
    """Convert degrees to radians.

    Parameters
    ----------
    x: input float, ndarray or torch.Tensor, in degrees
    dtype: dtype to cast to; NumPy-compatible or torch-compatible dtype or None.
        NumPy dtypes can be used with torch Tensors.
        If None, will use the dtype of x.

    Returns
    -------
    float, np.ndarray or torch.Tensor, x in radians, with the specified dtype
    """

    if isinstance(x, torch.Tensor):
        if dtype is not None and not isinstance(dtype, torch.dtype):
            # dtype is a numpy-compatible type; convert to the equivalent torch.dtype
            dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        return torch.deg2rad(x.to(dtype=dtype))
    return np.deg2rad(x, dtype=dtype)


def _dispatch_torch_numpy(torch_fn: Callable, numpy_fn: Callable):
    """Create generic math function that routes to either a numpy function or the torch equivalent."""

    def fn(*args: T) -> T:
        return torch_fn(*args) if isinstance(args[0], torch.Tensor) else numpy_fn(*args)

    return fn


_arccos = _dispatch_torch_numpy(torch.arccos, np.arccos)
_arctan2 = _dispatch_torch_numpy(torch.arctan2, np.arctan2)
_cos = _dispatch_torch_numpy(torch.cos, np.cos)
_isnan = _dispatch_torch_numpy(torch.isnan, np.isnan)
_sqrt = _dispatch_torch_numpy(torch.sqrt, np.sqrt)
_sin = _dispatch_torch_numpy(torch.sin, np.sin)
_where = _dispatch_torch_numpy(torch.where, np.where)


def _maximum(x: T, y: T) -> T:
    if isinstance(x, torch.Tensor):
        # torch.maximum requires both arguments to be tensors
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)
    return np.maximum(x, y)


def _clamp(x: T, min: float | None = None, max: float | None = None) -> T:
    return (x.clamp if isinstance(x, torch.Tensor) else x.clip)(min=min, max=max)


def irradiance(
    t,
    S0=1361,
    e=0.0167,
    perihelion_longitude=282.895,
    mean_tropical_year=365.2422,
    newton_iterations: int = 3,
):
    """The flux of solar energy in W/m2 towards Earth

    The default orbital parameters are set to 2000 values.
    Over the period of 1900-2100 this will result in an error of at most 0.02%,
    so can be neglected for many applications.

    Parameters
    ----------
    t: linux timestamp
    S0: the solar constant in W/m2. This is the mean irradiance received by
        earth over a year.
    e: the eccentricity of earths elliptical orbit
    perihelion_longitude: spatial angle from moving vernal equinox to perihelion with Sun as angle vertex.
        Perihelion is moment when earth is closest to sun. vernal equinox is
        the longitude when the Earth crosses the equator from South to North.
    newton_iterations: number of iterations for newton solver for elliptic anomaly

    Returns
    -------
    float or torch.Tensor, solar irradiance at the top of atmosphere in W/m²

    Notes
    -----

    TISR can be computed from Berger's formulas:

        Berger, A. (1978). Long-Term Variations of Daily Insolation and
        Quaternary Climatic Changes. Journal of the Atmospheric Sciences,
        35(12), 2362–2367.
        https://doi.org/10.1175/1520-0469(1978)035<2362:LTVODI>2.0.CO;2

    NASA Example computing the orbital parameters: https://data.giss.nasa.gov/modelE/ar5plots/srorbpar.html. From 1900-2100 these are the ranges::
        Orbital Parmameters

                                            Long. of
        Year     Eccentri    Obliquity    Perihel.
        (A.D.)      city      (degrees)    (degrees)
        ------    --------    ---------    --------
            1900   0.016744      23.4528      281.183
            2000   0.016704      23.4398      282.895
            2100   0.016663      23.4268      284.609

    """
    seconds_per_solar_day = 86400
    mean_tropical_year = mean_tropical_year * seconds_per_solar_day

    year_2000_equinox = datetime.datetime(2000, 3, 20, 7, 35, tzinfo=datetime.UTC)

    # from appendix of Berger 1978
    M = (t - year_2000_equinox.timestamp()) % mean_tropical_year
    M = M / mean_tropical_year * 2 * np.pi
    M -= _deg2rad(perihelion_longitude)

    # to get the elliptic anomaly E from the "mean anomaly" M
    # use eq. 6.37
    # https://link.springer.com/book/10.1007/978-3-662-53045-0)
    # r / a = (1 - e cos E )
    # E - e sin(E) = M
    def f(E):
        return E - e * _sin(E) - M

    def fp(E):
        return 1 - e * _cos(E)

    # newton iterations
    # initial guess
    E = M
    for _ in range(newton_iterations):
        E = E - f(E) / fp(E)

    rho = 1 - e * _cos(E)
    return S0 / rho**2


def toa_incident_solar_radiation_accumulated(
    t,
    lat,
    lon,
    interval=3600,
    S0=1361,
    e=0.0167,
    perihelion_longitude=282.895,
    mean_tropical_year=365.2422,
):
    """Approximate ECMWF TISR with analytical formulas

    According to the ECWMF docs, the TISR variable is integrated over the
    preceeding hour.  Error is about 0.1% different from the ECMWF TISR
    variable.

    Parameters
    ----------
    t: linux timestamp
    lat, lon: latitude and longitude in degrees
    interval: the integral length in seconds over which the irradiance is integrated
    S0: the solar constant in W/m2. This is the mean irradiance received by
        earth over a year.
    e: the eccentricity of earths elliptical orbit
    perihelion_longitude: spatial angle from moving vernal equinox to perihelion with Sun as angle vertex.
        Perihelion is moment when earth is closest to sun. vernal equinox is
        the longitude when the Earth crosses the equator from South to North.


    Returns
    -------
    TOA incident solar radiation accumulated from [t-inteval, t] in J/m2

    Notes
    -----

    We make some approximations:

    The default orbital parameters are set to 2000 values.
    Over the period of 1900-2100 this will result in an error of at most 0.02%,
    so can be neglected for many applications.

    The irradiance is constant over the ``interval``.

    From ECWMF [docs](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations)

        Such parameters, which are only available from forecasts, have undergone particular types of statistical processing (temporal mean or accumulation, respectively) over a period of time called the processing period. In addition, these parameters may, or may not, have been averaged in time, to produce monthly means.

        The accumulations (over the accumulation/processing period) in the short forecasts (from 06 and 18 UTC) of ERA5 are treated differently compared with those in ERA-Interim and operational data (where the accumulations are from the beginning of the forecast to the validity date/time). In the short forecasts of ERA5, the accumulations are since the previous post processing (archiving), so for:

        reanalysis: accumulations are over the hour (the accumulation/processing period) ending at the validity date/time
        ensemble: accumulations are over the 3 hours (the accumulation/processing period) ending at the validity date/time
        Monthly means (of daily means, stream=moda/edmo): accumulations have been scaled to have an "effective" processing period of one day, see section Monthly means
        Mean rate/flux parameters in ERA5 (e.g. Table 4 for surface and single levels) provide similar information to accumulations (e.g. Table 3 for surface and single levels), except they are expressed as temporal means, over the same processing periods, and so have units of "per second".

        Mean rate/flux parameters are easier to deal with than accumulations because the units do not vary with the processing period.
        The mean rate hydrological parameters (e.g. the "Mean total precipitation rate") have units of "kg m-2 s-1", which are equivalent to "mm s-1". They can be multiplied by 86400 seconds (24 hours) to convert to kg m-2 day-1 or mm day-1.
        Note that:

        For the CDS time, or validity time, of 00 UTC, the mean rates/fluxes and accumulations are over the hour (3 hours for the EDA) ending at 00 UTC i.e. the mean or accumulation is during part of the previous day.
        Mean rates/fluxes and accumulations are not available from the analyses.
        Mean rates/fluxes and accumulations at step=0 have values of zero because the length of the processing period is zero.

    """  # noqa
    lat = _deg2rad(lat)
    lon = _deg2rad(lon)

    century = _timestamp_to_julian_century(t)
    ra, dec = _right_ascension_declination(century)
    interval_radians = interval / 86400 * 2 * np.pi
    # 0 <= h1 < 2 pi
    h1 = _local_hour_angle(century, lon, ra)
    h0 = h1 - interval_radians
    A = _sin(lat) * _sin(dec)
    B = _cos(lat) * _cos(dec)

    # assume irradiance is constant over the interval
    S = irradiance(t, S0, e, perihelion_longitude, mean_tropical_year)
    sec_per_rad = 86400 / (2 * np.pi)
    return S * _integrate_abs_cosz(A, B, h0, h1) * sec_per_rad


def _integrate_abs_cosz(A, B, h0, h1):
    """Analytically integrate max(A + B cos(h), 0) from h=h0 to h1"""

    hc = _arccos(-A / B)

    def integrate_cosz(left, right):
        return A * (right - left) + B * (_sin(right) - _sin(left))

    def integrate_abs_cosz_from_zero_to(a):
        root1 = -hc + 2 * np.pi
        T = np.pi * 2

        # how many periods
        n = a // T

        # if there is a root
        a = a % T
        C = integrate_cosz(0, _where(a < hc, a, hc))
        D = _where(root1 < a, integrate_cosz(root1, a), 0)
        total = integrate_cosz(0, hc) + integrate_cosz(root1, 2 * np.pi)
        return C + D + total * n

    return _where(
        _isnan(hc),
        _maximum(integrate_cosz(h0, h1), 0),
        integrate_abs_cosz_from_zero_to(h1) - integrate_abs_cosz_from_zero_to(h0),
    )


def _datetime_to_julian_century(time: datetime.datetime) -> float:
    return _days_from_2000(time) / 36525.0


def _days_from_2000(model_time):
    """Get the days since year 2000.

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> float(_days_from_2000(model_time))
    731.0
    """
    if isinstance(model_time, datetime.datetime):
        model_time = model_time.replace(tzinfo=datetime.UTC)

    date_type = type(np.asarray(model_time).ravel()[0])
    if date_type not in [datetime.datetime]:
        raise ValueError(
            f"model_time has an invalid date type. It must be "
            f"datetime.datetime. Got {date_type}."
        )
    return _total_days(
        model_time - date_type(2000, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    )


def _total_days(time_diff):
    """
    Total time in units of days
    """
    return np.asarray(time_diff).astype("timedelta64[us]") / np.timedelta64(1, "D")


def _timestamp_to_julian_century(timestamp):
    seconds_in_day = 86400
    days_in_julian_century = 36525.0
    return (timestamp - TIMESTAMP_2000) / days_in_julian_century / seconds_in_day


def _greenwich_mean_sidereal_time(jul_centuries):
    """
    Greenwich mean sidereal time, in radians.
    Reference:
        The AIAA 2006 implementation:
            http://www.celestrak.com/publications/AIAA/2006-6753/

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> g_time = _greenwich_mean_sidereal_time(c)
    >>> bool(abs(g_time - 4.903831411) < 1e-8)
    True
    """
    theta = 67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2e-6)
    )

    theta_radians = _deg2rad(theta / 240.0) % (2 * np.pi)

    return theta_radians


def _local_mean_sidereal_time(julian_centuries, longitude):
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm


    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> l_time = _local_mean_sidereal_time(c, np.deg2rad(90))
    >>> bool(abs(l_time - 6.474627737) < 1e-8)
    True
    """
    return _greenwich_mean_sidereal_time(julian_centuries) + longitude


def _sun_ecliptic_longitude(julian_centuries):
    """
    Ecliptic longitude of the sun.
    Reference:
        http://www.geoastro.de/elevaz/basics/meeus.htm

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> lon = _sun_ecliptic_longitude(c)
    >>> bool(abs(lon - 17.469114444) < 1e-8)
    True
    """

    # mean anomaly calculation
    mean_anomaly = _deg2rad(
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries
    )

    # mean longitude
    mean_longitude = _deg2rad(
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries**2)
    )

    d_l = _deg2rad(
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries**2))
        * _sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * _sin(2 * mean_anomaly)
        + 0.000290 * _sin(3 * mean_anomaly)
    )

    # true longitude
    return mean_longitude + d_l


def _obliquity_star(julian_centuries):
    """
    return obliquity of the sun
    Use 5th order equation from
    https://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> julian_centuries = _days_from_2000(model_time) / 36525.0
    >>> obl = _obliquity_star(julian_centuries)
    >>> bool(abs(obl - 0.409088056) < 1e-8)
    True
    """
    return _deg2rad(
        23.0
        + 26.0 / 60
        + 21.406 / 3600.0
        - (
            46.836769 * julian_centuries
            - 0.0001831 * (julian_centuries**2)
            + 0.00200340 * (julian_centuries**3)
            - 0.576e-6 * (julian_centuries**4)
            - 4.34e-8 * (julian_centuries**5)
        )
        / 3600.0
    )


def _right_ascension_declination(julian_centuries):
    """
    Right ascension and declination of the sun.
    Ref:
        http://www.geoastro.de/elevaz/basics/meeus.htm

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> out1, out2 = _right_ascension_declination(c)
    >>> bool(abs(out1 - -1.363787213) < 1e-8)
    True
    >>> bool(abs(out2 - -0.401270126) < 1e-8)
    True
    """
    eps = _obliquity_star(julian_centuries)
    eclon = _sun_ecliptic_longitude(julian_centuries)
    x = _cos(eclon)
    y = _cos(eps) * _sin(eclon)
    z = _sin(eps) * _sin(eclon)
    r = _sqrt(1.0 - z * z)
    # sun declination
    declination = _arctan2(z, r)
    # right ascension
    right_ascension = 2 * _arctan2(y, (x + r))
    return right_ascension, declination


def _local_hour_angle(julian_centuries, longitude, right_ascension):
    """
    Hour angle at model_time for the given longitude and right_ascension
    longitude in radians
    Ref:
        https://en.wikipedia.org/wiki/Hour_angle#Relation_with_the_right_ascension
    """
    return _local_mean_sidereal_time(julian_centuries, longitude) - right_ascension


def _star_cos_zenith(julian_centuries, lon, lat):
    """
    Return cosine of star zenith angle
    lon,lat in radians
    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """

    if isinstance(lon, torch.Tensor):
        output_dtype = lon.dtype
        # Astronomical polynomial evaluations involve values up to ~6e8 degrees,
        # which exceed float32 precision. Compute in float64 and cast back.
        lon = lon.to(torch.float64)
        lat = lat.to(torch.float64)
        julian_centuries = torch.as_tensor(
            julian_centuries, dtype=torch.float64, device=lon.device
        )
    else:
        output_dtype = None

    ra, dec = _right_ascension_declination(julian_centuries)
    h_angle = _local_hour_angle(julian_centuries, lon, ra)

    cosine_zenith = _sin(lat) * _sin(dec) + _cos(lat) * _cos(dec) * _cos(h_angle)

    if output_dtype is not None:
        cosine_zenith = cosine_zenith.to(output_dtype)

    return cosine_zenith


def _star_zenith_azimuth(julian_centuries, lon, lat, sin_zenith_epsilon=1e-6):
    """
    Return sine/cosine of solar zenith and azimuth angles.
    lon, lat in radians. Azimuth measured clockwise from north.
    sin_zenith is clamped to a small positive minimum value for numerical stability.

    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """
    if isinstance(lon, torch.Tensor):
        output_dtype = lon.dtype
        # Astronomical polynomial evaluations involve values up to ~6e8 degrees,
        # which exceed float32 precision. Compute in float64 and cast back.
        lon = lon.to(torch.float64)
        lat = lat.to(torch.float64)
        julian_centuries = torch.as_tensor(
            julian_centuries, dtype=torch.float64, device=lon.device
        )
    else:
        output_dtype = None

    ra, dec = _right_ascension_declination(julian_centuries)
    h_angle = _local_hour_angle(julian_centuries, lon, ra)

    sin_lat = _sin(lat)
    cos_lat = _cos(lat)
    sin_dec = _sin(dec)
    cos_dec = _cos(dec)
    sin_h = _sin(h_angle)
    cos_h = _cos(h_angle)

    cos_zenith = _clamp(sin_lat * sin_dec + cos_lat * cos_dec * cos_h, -1, 1)
    sin_zenith = _clamp(_sqrt(1.0 - cos_zenith * cos_zenith), min=sin_zenith_epsilon)
    sin_azimuth = _clamp(-sin_h * cos_dec / sin_zenith, -1, 1)
    cos_azimuth = _clamp(
        (sin_dec - cos_zenith * sin_lat) / (sin_zenith * cos_lat), -1, 1
    )

    if output_dtype is not None:
        sin_zenith = sin_zenith.to(output_dtype)
        cos_zenith = cos_zenith.to(output_dtype)
        sin_azimuth = sin_azimuth.to(output_dtype)
        cos_azimuth = cos_azimuth.to(output_dtype)

    return (sin_zenith, cos_zenith, sin_azimuth, cos_azimuth)
