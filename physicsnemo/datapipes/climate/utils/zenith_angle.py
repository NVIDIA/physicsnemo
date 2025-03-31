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

import numpy as np
import pytz

try:
    import nvidia.dali as dali
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

RAD_PER_DEG = np.pi / 180.0
DATETIME_2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=pytz.utc).timestamp()


def _dali_mod(a, b):
    return a - b * dali.math.floor(a / b)


def cos_zenith_angle(
    time: dali.types.DALIDataType,
    latlon: dali.types.DALIDataType,
):
    """
    Dali datapipe for computing Cosine of sun-zenith angle for lon, lat at time (UTC).

    Parameters
    ----------
    time : dali.types.DALIDataType
        Time in seconds since 2000-01-01 12:00:00 UTC. Shape `(seq_length,)`.
    latlon : dali.types.DALIDataType
        Latitude and longitude in degrees. Shape `(2, nr_lat, nr_lon)`.

    Returns
    -------
    dali.types.DALIDataType
        Cosine of sun-zenith angle. Shape `(seq_length, 1, nr_lat, nr_lon)`.
    """
    lat = latlon[dali.newaxis, 0:1, :, :] * RAD_PER_DEG
    lon = latlon[dali.newaxis, 1:2, :, :] * RAD_PER_DEG
    time = time[:, dali.newaxis, dali.newaxis, dali.newaxis]
    return _star_cos_zenith(time, lat, lon)


def _days_from_2000(model_time):  # pragma: no cover
    """Get the days since year 2000."""
    return (model_time - DATETIME_2000) / (24.0 * 3600.0)


def _greenwich_mean_sidereal_time(model_time):
    """
    Greenwich mean sidereal time, in radians.
    Reference:
        The AIAA 2006 implementation:
            http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    jul_centuries = _days_from_2000(model_time) / 36525.0
    theta = 67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2 * 10e-6)
    )

    theta_radians = _dali_mod((theta / 240.0) * RAD_PER_DEG, 2 * np.pi)
    return theta_radians


def _local_mean_sidereal_time(model_time, longitude):
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm
    """
    return _greenwich_mean_sidereal_time(model_time) + longitude


def _sun_ecliptic_longitude(model_time):
    """
    Ecliptic longitude of the sun.
    Reference:
        http://www.geoastro.de/elevaz/basics/meeus.htm
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0

    # mean anomaly calculation
    mean_anomaly = (
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries
    ) * RAD_PER_DEG

    # mean longitude
    mean_longitude = (
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries**2)
    ) * RAD_PER_DEG

    d_l = (
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries**2))
        * dali.math.sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * dali.math.sin(2 * mean_anomaly)
        + 0.000290 * dali.math.sin(3 * mean_anomaly)
    ) * RAD_PER_DEG

    # true longitude
    return mean_longitude + d_l


def _obliquity_star(julian_centuries):
    """
    return obliquity of the sun
    Use 5th order equation from
    https://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic
    """
    return (
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
    ) * RAD_PER_DEG


def _right_ascension_declination(model_time):
    """
    Right ascension and declination of the sun.
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries)

    eclon = _sun_ecliptic_longitude(model_time)
    x = dali.math.cos(eclon)
    y = dali.math.cos(eps) * dali.math.sin(eclon)
    z = dali.math.sin(eps) * dali.math.sin(eclon)
    r = dali.math.sqrt(1.0 - z * z)
    # sun declination
    declination = dali.math.atan2(z, r)
    # right ascension
    right_ascension = 2 * dali.math.atan2(y, (x + r))
    return right_ascension, declination


def _local_hour_angle(model_time, longitude, right_ascension):
    """
    Hour angle at model_time for the given longitude and right_ascension
    longitude in radians
    Ref:
        https://en.wikipedia.org/wiki/Hour_angle#Relation_with_the_right_ascension
    """
    return _local_mean_sidereal_time(model_time, longitude) - right_ascension


def _star_cos_zenith(model_time, lat, lon):
    """
    Return cosine of star zenith angle
    lon,lat in radians
    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """

    ra, dec = _right_ascension_declination(model_time)
    h_angle = _local_hour_angle(model_time, lon, ra)

    cosine_zenith = dali.math.sin(lat) * dali.math.sin(dec) + dali.math.cos(
        lat
    ) * dali.math.cos(dec) * dali.math.cos(h_angle)
    return cosine_zenith
