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
from typing import TypeVar, Union

import numpy as np

# can replace this import with zoneinfo from the standard library in python3.9+.
import pytz

# helper type
dtype = np.float32


T = TypeVar("T", np.ndarray, float)

TIMESTAMP_2000 = datetime.datetime(2000, 1, 1, 12, 0, tzinfo=pytz.utc).timestamp()


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
    time: time in UTC
    lon: float or np.ndarray in degrees (E/W)
    lat: float or np.ndarray in degrees (N/S)

    Returns
    --------
    float, np.ndarray

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> angle = cos_zenith_angle(model_time, lat=360, lon=120)
    >>> abs(angle - -0.447817277) < 1e-6
    True
    """
    lon_rad = np.deg2rad(lon, dtype=dtype)
    lat_rad = np.deg2rad(lat, dtype=dtype)
    julian_centuries = _datetime_to_julian_century(time)
    return _star_cos_zenith(julian_centuries, lon_rad, lat_rad)


def cos_zenith_angle_from_timestamp(
    timestamp: T,
    lon: T,
    lat: T,
) -> T:
    """
    Compute cosine of zenith angle using UNIX timestamp

    Since the UNIX timestamp is a floating point or integer this routine can be
    compiled with jax.

    Parameters
    ----------
    timestamp: timestamp in seconds from UNIX epoch
    lon: longitude in degrees E
    lat: latitude in degrees N

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> angle = cos_zenith_angle_from_timestamp(model_time.timestamp(), lat=360, lon=120)
    >>> abs(angle - -0.447817277) < 1e-6
    True
    """
    lon_rad = np.deg2rad(lon, dtype=dtype)
    lat_rad = np.deg2rad(lat, dtype=dtype)
    julian_centuries = _timestamp_to_julian_century(timestamp)
    return _star_cos_zenith(julian_centuries, lon_rad, lat_rad)


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

    year_2000_equinox = datetime.datetime(2000, 3, 20, 7, 35, tzinfo=pytz.utc)

    # from appendix of Berger 1978
    M = (t - year_2000_equinox.timestamp()) % mean_tropical_year
    M = M / mean_tropical_year * 2 * np.pi
    M -= np.deg2rad(perihelion_longitude)

    # to get the elliptic anomaly E from the "mean anomaly" M
    # use eq. 6.37
    # https://link.springer.com/book/10.1007/978-3-662-53045-0)
    # r / a = (1 - e cos E )
    # E - e sin(E) = M
    def f(E):
        return E - e * np.sin(E) - M

    def fp(E):
        return 1 - e * np.cos(E)

    # newton iterations
    # initial guess
    E = M
    for _ in range(newton_iterations):
        E = E - f(E) / fp(E)

    rho = 1 - e * np.cos(E)
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
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    century = _timestamp_to_julian_century(t)
    ra, dec = _right_ascension_declination(century)
    interval_radians = interval / 86400 * 2 * np.pi
    # 0 <= h1 < 2 pi
    h1 = _local_hour_angle(century, lon, ra)
    h0 = h1 - interval_radians
    A = np.sin(lat) * np.sin(dec)
    B = np.cos(lat) * np.cos(dec)

    # assume irradiance is constant over the interval
    S = irradiance(t, S0, e, perihelion_longitude, mean_tropical_year)
    sec_per_rad = 86400 / (2 * np.pi)
    return S * _integrate_abs_cosz(A, B, h0, h1) * sec_per_rad


def _integrate_abs_cosz(A, B, h0, h1):
    """Analytically integrate max(A + B cos(h), 0) from h=h0 to h1"""

    hc = np.arccos(-A / B)

    def integrate_cosz(left, right):
        return A * (right - left) + B * (np.sin(right) - np.sin(left))

    def integrate_abs_cosz_from_zero_to(a):
        root1 = -hc + 2 * np.pi
        T = np.pi * 2

        # how many periods
        n = a // T

        # if there is a root
        a = a % T
        C = integrate_cosz(0, np.where(a < hc, a, hc))
        D = np.where(root1 < a, integrate_cosz(root1, a), 0)
        total = integrate_cosz(0, hc) + integrate_cosz(root1, 2 * np.pi)
        return C + D + total * n

    return np.where(
        np.isnan(hc),
        np.maximum(integrate_cosz(h0, h1), 0),
        integrate_abs_cosz_from_zero_to(h1) - integrate_abs_cosz_from_zero_to(h0),
    )


def _datetime_to_julian_century(time: datetime.datetime) -> float:
    return _days_from_2000(time) / 36525.0


def _days_from_2000(model_time):
    """Get the days since year 2000.

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> _days_from_2000(model_time)
    731.0
    """
    if isinstance(model_time, datetime.datetime):
        model_time = model_time.replace(tzinfo=pytz.utc)

    date_type = type(np.asarray(model_time).ravel()[0])
    if date_type not in [datetime.datetime]:
        raise ValueError(
            f"model_time has an invalid date type. It must be "
            f"datetime.datetime. Got {date_type}."
        )
    return _total_days(model_time - date_type(2000, 1, 1, 12, 0, 0, tzinfo=pytz.utc))


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
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> g_time = _greenwich_mean_sidereal_time(c)
    >>> abs(g_time - 4.903831411) < 1e-8
    True
    """
    theta = 67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2 * 10e-6)
    )

    theta_radians = np.deg2rad(theta / 240.0) % (2 * np.pi)

    return theta_radians


def _local_mean_sidereal_time(julian_centuries, longitude):
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm


    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> l_time = _local_mean_sidereal_time(c, np.deg2rad(90))
    >>> abs(l_time - 6.474627737) < 1e-8
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
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> lon = _sun_ecliptic_longitude(c)
    >>> abs(lon - 17.469114444) < 1e-8
    True
    """

    # mean anomaly calculation
    mean_anomaly = np.deg2rad(
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries
    )

    # mean longitude
    mean_longitude = np.deg2rad(
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries**2)
    )

    d_l = np.deg2rad(
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries**2))
        * np.sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * np.sin(2 * mean_anomaly)
        + 0.000290 * np.sin(3 * mean_anomaly)
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
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> julian_centuries = _days_from_2000(model_time) / 36525.0
    >>> obl = _obliquity_star(julian_centuries)
    >>> abs(obl - 0.409088056) < 1e-8
    True
    """
    return np.deg2rad(
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
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> c = _timestamp_to_julian_century(model_time.timestamp())
    >>> out1, out2 = _right_ascension_declination(c)
    >>> abs(out1 - -1.363787213) < 1e-8
    True
    >>> abs(out2 - -0.401270126) < 1e-8
    True
    """
    eps = _obliquity_star(julian_centuries)
    eclon = _sun_ecliptic_longitude(julian_centuries)
    x = np.cos(eclon)
    y = np.cos(eps) * np.sin(eclon)
    z = np.sin(eps) * np.sin(eclon)
    r = np.sqrt(1.0 - z * z)
    # sun declination
    declination = np.arctan2(z, r)
    # right ascension
    right_ascension = 2 * np.arctan2(y, (x + r))
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

    ra, dec = _right_ascension_declination(julian_centuries)
    h_angle = _local_hour_angle(julian_centuries, lon, ra)

    cosine_zenith = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(
        h_angle
    )

    return cosine_zenith
