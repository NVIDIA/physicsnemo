# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os

import matplotlib.pyplot as plt
import numpy as np

import xarray
from scipy.fft import irfft
from scipy.signal import periodogram


def open_data(file, group=False):
    """
    Opens a dataset from a NetCDF file.

    Parameters:
        file (str): Path to the NetCDF file.
        group (bool, optional): Whether to open the file as a group. Default is False.

    Returns:
        xarray.Dataset: An xarray dataset containing the data from the NetCDF file.
    """
    root = xarray.open_dataset(file)
    root = root.set_coords(["lat", "lon"])
    ds = xarray.open_dataset(file, group=group)
    ds.coords.update(root.coords)
    ds.attrs.update(root.attrs)

    return ds


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two sets of latitude and longitude coordinates.

    The Haversine formula calculates the shortest distance between two points on the
    surface of a sphere (in this case, the Earth) given their latitude and longitude
    coordinates.

    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: The Haversine distance between the two points in meters.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Earth radius in meters
    earth_radius = 6371000  # Approximate value for the average Earth radius

    # Calculate differences in latitude and longitude
    dlat_rad = lat2_rad - lat1_rad
    dlon_rad = lon2_rad - lon1_rad

    # Haversine formula
    a = (
        np.sin(dlat_rad / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_meters = earth_radius * c

    return distance_meters


def compute_power_spectrum(data, d):
    """
    Compute the power spectrum of a 2D data array using the Fast Fourier Transform (FFT).

    The power spectrum represents the distribution of signal power as a function of frequency.

    Parameters:
        data (numpy.ndarray): 2D input data array.
        d (float): Sampling interval (time between data points).

    Returns:
        tuple: A tuple containing the frequency values and the corresponding power spectrum.
        - freqs (numpy.ndarray): Frequency values corresponding to the power spectrum.
        - power_spectrum (numpy.ndarray): Power spectrum of the input data.
    """

    # Compute the 2D FFT along the second dimension
    fft_data = np.fft.fft(data, axis=-2)

    # Compute the power spectrum by taking the absolute value and squaring
    power_spectrum = np.abs(fft_data) ** 2

    # Scale the power spectrum based on the sampling interval 'd'
    power_spectrum /= data.shape[-1] * d
    freqs = np.fft.fftfreq(data.shape[-1], d)

    return freqs, power_spectrum


def power_spectra_to_acf(f, pw):
    """
    Convert a one-sided power spectrum to an autocorrelation function.

    Args:
        f (numpy.ndarray): Frequencies.
        pw (numpy.ndarray): Power spectral density in units of V^2/Hz.

    Returns:
        numpy.ndarray: Autocorrelation function (ACF).
    """
    pw = pw.copy()
    pw[0] = 0
    # magic factor 4 comes from periodogram/irfft stuff
    # 1) a factor 2 comes from the  periodogram being one-sided.
    # I don't fully understasnd, but this ensures the acf is 1 at r=0
    sig2 = np.sum(pw * f[1]) * 4
    acf = irfft(pw) / sig2
    return acf


def average_power_spectrum(data, d):
    """
    Compute the average power spectrum of a 2D data array.

    This function calculates the power spectrum for each row of the input data and
    then averages them to obtain the overall power spectrum.
    The power spectrum represents the distribution of signal power as a function of frequency.

    Parameters:
        data (numpy.ndarray): 2D input data array.
        d (float): Sampling interval (time between data points).

    Returns:
        tuple: A tuple containing the frequency values and the average power spectrum.
        - freqs (numpy.ndarray): Frequency values corresponding to the power spectrum.
        - power_spectra (numpy.ndarray): Average power spectrum of the input data.
    """
    # Compute the power spectrum along the second dimension for each row
    freqs, power_spectra = periodogram(data, fs=1 / d, axis=-1)

    # Average along the first dimension
    while power_spectra.ndim > 1:
        power_spectra = power_spectra.mean(axis=0)

    return freqs, power_spectra


def main(file, output):
    """
    Generate and save multiple power spectrum plots from input data.

    Parameters:
        file (str): Path to the input data file.
        output (str): Directory where the generated plots will be saved.

    This function loads and processes various datasets from the input file,
    calculates their power spectra, and generates and saves multiple power spectrum plots.
    The plots include kinetic energy, temperature, and reflectivity power spectra.
    """

    def savefig(name):
        path = os.path.join(output, name + ".png")
        plt.savefig(path)

    samples = {}
    samples["prediction"] = open_data(file, group="prediction")
    samples["prediction_mean"] = samples["prediction"].mean("ensemble")
    samples["truth"] = open_data(file, group="truth")
    samples["ERA5"] = open_data(file, group="input")

    prediction = samples["prediction"]
    lat = prediction.lat
    lon = prediction.lon

    dx = haversine(lat[0, 0], lon[0, 0], lat[1, 0], lon[1, 0])
    dy = haversine(lat[0, 0], lon[0, 0], lat[0, 1], lon[0, 1])
    print(dx, dy)
    # the approximate resolution is dx=dy=2000m

    # in km
    d = 2

    # Plot the power spectrum
    for name, data in samples.items():
        freqs, spec_x = average_power_spectrum(data.eastward_wind_10m, d=d)
        _, spec_y = average_power_spectrum(data.northward_wind_10m, d=d)
        spec = spec_x + spec_y
        plt.loglog(freqs, spec, label=name)
        plt.xlabel("Frequency (1/km)")
        plt.ylabel("Power Spectrum")
        plt.ylim(bottom=1e-1)
    plt.title("Kinetic Energy power spectra")
    plt.grid()
    plt.legend()
    savefig("ke-spectra")

    plt.figure()
    for name, data in samples.items():
        freqs, spec = average_power_spectrum(data.temperature_2m, d=d)
        plt.loglog(freqs, spec, label=name)
        plt.xlabel("Frequency (1/km)")
        plt.ylabel("Power Spectrum")
        plt.ylim(bottom=1e-1)
    plt.title("T2M Power spectra")
    plt.grid()
    plt.legend()
    savefig("t2m-spectra")

    plt.figure()
    for name, data in samples.items():
        try:
            freqs, spec = average_power_spectrum(data.maximum_radar_reflectivity, d=d)
        except AttributeError:
            continue
        plt.loglog(freqs, spec, label=name)
        plt.xlabel("Frequency (1/km)")
        plt.ylabel("Power Spectrum")
        plt.ylim(bottom=1e-1)
    plt.title("Reflectivity Power spectra")
    plt.grid()
    plt.legend()
    savefig("reflectivity-spectra")


if __name__ == "__main__":
    typer.run(main)
